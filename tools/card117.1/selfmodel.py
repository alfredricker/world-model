"""Card 117.1: the self per patch, from proprioception.

A small network predicts every patch's features (blocks 2 and 12 of the frozen encoder, 256-px frames, 32 x 32
patches) from the agent's own pose alone: joint angles, effector position and yaw, gripper opening (12 numbers;
no velocities, no contact). A patch belongs to the self at time t where that prediction depends on the pose:
influence(p, t) = || f_p(pose_t) - mean over 256 other poses of f_p(pose) ||^2. No labels, no masks.

  train:  bin/prun python tools/card117.1/selfmodel.py --minutes 5      → runs/117.1/self.pt
  score:  bin/prun python tools/card117.1/selfmodel.py --eval 1         → runs/117.1/gate.json
          (card 122's 4 stored frames per episode: threshold by Otsu on episodes 900-999, tested on validation)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/117.1'
EP, SKIP, N, FEAT = 1001, 5, 1024, 768
POSE = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
OBJECTS = ['arm', 'cube', 'button0', 'button1', 'drawer', 'window']
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


class SelfModel(nn.Module):
    def __init__(self, d=64, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(len(POSE), 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d))
        self.pos = nn.Parameter(torch.randn(1, N, d) * 0.02)
        self.dec = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, FEAT))
        self.register_buffer('mu', torch.zeros(len(POSE)))
        self.register_buffer('sd', torch.ones(len(POSE)))

    def forward(self, pose, idx=None):                       # pose [B, 12] → [B, n, FEAT]
        h = self.enc((pose - self.mu) / self.sd)
        pos = self.pos if idx is None else self.pos[:, idx]
        return self.dec(torch.cat([h[:, None].expand(-1, pos.shape[1], -1), pos.expand(len(h), -1, -1)], -1))


@torch.no_grad()
def influence(model, pose, ref):
    """pose [B, 12], ref [R, 12] → [B, N]: how far each patch's prediction is from its mean over the reference poses."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        base = torch.stack([model(ref[i:i + 64]).float().sum(0) for i in range(0, len(ref), 64)]).sum(0) / len(ref)
        out = torch.cat([((model(pose[i:i + 64]).float() - base) ** 2).sum(-1) for i in range(0, len(pose), 64)])
    return out


def otsu(x):
    best, cut = -1, None
    for c in np.quantile(x, np.linspace(0.02, 0.98, 97)):
        a, b = x[x <= c], x[x > c]
        if len(a) and len(b):
            v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
            if v > best:
                best, cut = v, c
    return cut


def train():
    minutes, B = float(arg('--minutes', 5)), 64
    dev = 'cuda'
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    src = np.load(ROOT / 'runs/124/feat_train.npy', mmap_mode='r')                 # [450, 201, 1024, 768] fp16
    neps = src.shape[0]
    obs = np.load(DATA / 'scene-play-v0.npz')['observations'][:neps * EP].reshape(neps, EP, -1)[:, ::SKIP][:, :, POSE]
    model = SelfModel().to(dev)
    model.mu.copy_(torch.from_numpy(obs.reshape(-1, len(POSE)).mean(0)))
    model.sd.copy_(torch.from_numpy(obs.reshape(-1, len(POSE)).std(0) + 1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=4e-4)
    BUF, SWAP, REFRESH = 16, 4, 100
    held = list(rng.choice(neps, BUF, replace=False))
    buf = [np.ascontiguousarray(src[e]) for e in held]
    t0, step, log = time.time(), 0, []
    while time.time() - t0 < minutes * 60:
        if step and step % REFRESH == 0:
            for _ in range(SWAP):
                held.pop(0); buf.pop(0)
                e = int(rng.choice([x for x in range(neps) if x not in held]))
                held.append(e); buf.append(np.ascontiguousarray(src[e]))
        b, k = rng.integers(0, BUF, B), rng.integers(0, 201, B)
        idx = torch.randperm(N, device=dev)[:256]
        tgt = torch.from_numpy(np.stack([buf[i][j] for i, j in zip(b, k)])).to(dev)[:, idx].float()
        pose = torch.from_numpy(np.stack([obs[held[i], j] for i, j in zip(b, k)])).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.mse_loss(model(pose, idx).float(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0:
            log.append(dict(step=step, seconds=round(time.time() - t0, 1), loss=round(float(loss), 4)))
            print(json.dumps(log[-1]), flush=True)
    torch.save(dict(state=model.state_dict()), OUT / 'self.pt')
    json.dump(log, open(OUT / 'self.log.json', 'w'))


def load_model():
    model = SelfModel().to('cuda')
    model.load_state_dict(torch.load(OUT / 'self.pt', map_location='cuda')['state'])
    return model.eval()


def evaluate():
    model = load_model()
    rng = np.random.default_rng(0)
    tr = np.load(DATA / 'scene-play-v0.npz')['observations']
    ref = torch.from_numpy(tr[rng.integers(0, 450 * EP, 256)][:, POSE]).cuda()
    sets = {}
    for name in ['tail', 'val']:
        c = np.load(ROOT / f'runs/122/computed_{name}.npz')
        obs = tr[900 * EP:] if name == 'tail' else np.load(DATA / 'scene-play-v0-val.npz')['observations']
        sidx = c['sidx']
        inf = influence(model, torch.from_numpy(obs[sidx][:, POSE]).cuda(), ref).cpu().numpy()
        feats = np.concatenate([c['s2'], c['s12']], -1).astype(np.float32)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = torch.cat([model(torch.from_numpy(obs[sidx[i:i + 64]][:, POSE]).cuda()).float().cpu()
                              for i in range(0, len(sidx), 64)]).numpy()
        sets[name] = dict(inf=inf, masks=c['masks'][sidx].astype(bool), err=((pred - feats) ** 2).mean(-1))
    cut = otsu(np.log(sets['tail']['inf'].reshape(-1)))
    v = sets['val']
    sel, arm = np.log(v['inf']) > cut, v['masks'][:, 0]
    iou = lambda a, b: float((a & b).sum() / max((a | b).sum(), 1))
    per = [iou(sel[i], arm[i]) for i in range(len(sel))]
    cuts = np.quantile(np.log(v['inf']), np.linspace(0.5, 0.99, 50))
    res = dict(frames=len(sel), iou_with_arm=round(float(np.mean(per)), 3), pooled_iou=round(iou(sel, arm), 3),
               oracle_threshold_pooled_iou=round(max(iou(np.log(v['inf']) > c, arm) for c in cuts), 3),
               arm_covered=round(float(sel[arm].mean()), 3), self_share_of_patches=round(float(sel.mean()), 3),
               arm_share_of_patches=round(float(arm.mean()), 3),
               other_objects_in_self={o: round(float(sel[v['masks'][:, oi] & ~arm].mean()), 3)
                                      for oi, o in enumerate(OBJECTS) if oi},
               prediction_error=dict(on_arm=round(float(v['err'][arm].mean()), 3),
                                     off_arm=round(float(v['err'][~arm].mean()), 3)),
               log_threshold=float(cut))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'gate.json', 'w'), indent=1)


if __name__ == '__main__':
    evaluate() if int(arg('--eval', 0)) else train()
