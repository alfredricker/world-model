"""Card 124: tokens trained by consequences, on frozen DINOv2 patches of 256-px frames (32 x 32 patches).

Card 116.1's slot model (slot attention, K slots of 64 numbers, a transition between model steps, an MLP decoder
that gives each slot a feature and a mask per patch), with three changes to what trains it:
1. the transition reads the action chunk (25 numbers) and the contact, and its output is decoded and scored against
   the NEXT frame's features before that frame is seen (loss 'next'): a slot must carry what its part will do;
2. a cost on how many slots change between steps: the mean over slots of the L2 norm of each slot's change
   (loss 'sparse', weight --sparse): an event should be explained by few slots;
3. half of each batch are segments that contain a touch (--events 1), so rare events are not averaged away.
The present frame's reconstruction (loss 'now') is card 116.1's and stays.

Input and target: blocks 2 and 12 of the frozen encoder, concatenated (768 numbers per patch), encoded in the loop
from the stored frames. Segments of 4 model steps (5 environment steps apart). Adam 4e-4, warm-up, clipping.

  bin/prun python tools/card124/train.py --minutes 5 [--next 1] [--sparse 0.1] [--events 1] [--K 12] [--out smoke.pt]
  → runs/124/<out>, runs/124/<out>.log.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card114'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card116'))
from encode import MEAN, STD, load_encoder  # noqa: E402
from slots import SlotAttention  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/124'
EP, SKIP, T, RES, N, FEAT = 1001, 5, 4, 448, 1024, 768
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


@torch.no_grad()
def features(enc, rgb):
    """rgb uint8 [M, 256, 256, 3] on the device → [M, 1024, 768]: blocks 2 and 12, each through the final norm."""
    x = rgb.permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        a, b = enc.get_intermediate_layers(x, n=[1, 11], norm=True)
    return torch.cat([a.float(), b.float()], -1)


class TokenModel(nn.Module):
    def __init__(self, K=12, d=64, hidden=512):
        super().__init__()
        self.adapter = nn.Sequential(nn.LayerNorm(FEAT), nn.Linear(FEAT, FEAT), nn.GELU(), nn.Linear(FEAT, 128))
        self.sa = SlotAttention(128, d)
        self.init = nn.Parameter(torch.randn(1, K, d) * 0.1)
        self.act = nn.Sequential(nn.Linear(26, 128), nn.GELU(), nn.Linear(128, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.pred = nn.TransformerEncoder(layer, 2)
        self.pos = nn.Parameter(torch.randn(1, 1, N, d) * 0.02)
        self.dec = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, FEAT + 1))
        self.K, self.d, self.hidden = K, d, hidden

    def decode(self, slots, idx=None):                      # [B, K, d] → recon [B, N, F], masks [B, K, N]
        pos = self.pos if idx is None else self.pos[:, :, idx]   # idx: a subset of patches (training speed)
        out = self.dec(slots[:, :, None] + pos)
        masks = out[..., -1].softmax(1)
        return torch.einsum('bkn,bknf->bnf', masks, out[..., :-1]), masks

    def step(self, slots, act):
        """The transition: slots [B, K, d] and the action with contact [B, 26] → the slots expected next."""
        x = torch.cat([slots, self.act(act)[:, None]], 1)
        return self.pred(x)[:, :-1]

    def forward(self, feats, acts, patches=0):
        """feats [B, T, N, F], acts [B, T, 26] (the chunk after each frame, and the contact at it) →
        posterior slots [B, T, K, d], masks [B, T, K, N], and the losses' parts."""
        B, Tn = feats.shape[:2]
        x = self.adapter(feats)
        slots = self.init.expand(B, -1, -1)
        post, masks, now, nxt, sparse = [], [], 0.0, 0.0, 0.0
        idx = torch.randperm(N, device=feats.device)[:patches] if patches else None
        tgt = feats if idx is None else feats[:, :, idx]
        for t in range(Tn):
            if t > 0:
                prior = self.step(slots, acts[:, t - 1])
                nxt = nxt + F.mse_loss(self.decode(prior, idx)[0], tgt[:, t])
            else:
                prior = slots
            new, _ = self.sa(prior, x[:, t], 3 if t == 0 else 2)
            rec, m = self.decode(new, idx)
            now = now + F.mse_loss(rec, tgt[:, t])
            if t > 0:
                sparse = sparse + (new - slots).norm(dim=-1).mean()
            slots = new
            post.append(new); masks.append(m)
        return torch.stack(post, 1), torch.stack(masks, 1), now / Tn, nxt / (Tn - 1), sparse / (Tn - 1)


def load_data(neps):
    z = np.load(DATA / 'scene-play-v0.npz')
    n = neps * EP
    act = z['actions'][:n].reshape(neps, EP, 5)
    con = z['observations'][:n, 18].reshape(neps, EP)
    steps = np.arange(0, EP, SKIP)                                        # 201 model steps
    chunk = np.zeros((neps, len(steps), 26), np.float32)
    for k, t in enumerate(steps[:-1]):
        chunk[:, k, :25] = act[:, t:t + SKIP].reshape(neps, 25)
        chunk[:, k, 25] = con[:, t]
    touch = np.stack([con[:, t:t + SKIP * T].max(1) > 0.05 for t in steps[:len(steps) - T]], 1)   # [neps, starts]
    return chunk, touch


def main():
    minutes, B = float(arg('--minutes', 5)), int(arg('--batch', 12))
    K, lam = int(arg('--K', 12)), float(arg('--sparse', 0.1))
    use_next, events = int(arg('--next', 1)), int(arg('--events', 0))
    out = arg('--out', 'smoke.pt')
    patches = int(arg('--patches', 256))            # patches scored per step (0 = all 1,024)
    dev = 'cuda'
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    stored = (OUT / 'feat_train.npy').exists() and int(arg('--stored', 1))
    src = np.load(OUT / ('feat_train.npy' if stored else 'rgb_train.npy'), mmap_mode='r')
    neps = src.shape[0]
    chunk, touch = load_data(neps)
    ev = np.argwhere(touch)
    print(f'{neps} episodes; segments with a touch: {touch.mean():.3f}; stored features: {bool(stored)}', flush=True)
    # A rolling buffer of whole episodes in memory: every REFRESH steps the oldest SWAP episodes are replaced by
    # sequential reads. Random reads of 75 MB per step from the 142 GB file would load the disk for the whole run.
    BUF, SWAP, REFRESH = 16, 4, 100
    held = list(rng.choice(neps, BUF, replace=False))
    buf = [np.ascontiguousarray(src[e]) for e in held] if stored else None
    enc = None if stored else load_encoder(dev)
    model = TokenModel(K=K).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=4e-4)
    warm = 2500
    log, t0, step, t_enc = [], time.time(), 0, 0.0
    while time.time() - t0 < minutes * 60:
        if stored:
            if step and step % REFRESH == 0:
                for _ in range(SWAP):
                    held.pop(0); buf.pop(0)
                    e = int(rng.choice([x for x in range(neps) if x not in held]))
                    held.append(e); buf.append(np.ascontiguousarray(src[e]))
            slot = rng.integers(0, BUF, B)
            st = rng.integers(0, touch.shape[1], B)
            feats = torch.from_numpy(np.stack([buf[b][s:s + T] for b, s in zip(slot, st)])).to(dev).float()
            acts = torch.from_numpy(np.stack([chunk[held[b], s:s + T] for b, s in zip(slot, st)])).to(dev)
        else:
            idx = np.stack([rng.integers(0, neps, B), rng.integers(0, touch.shape[1], B)], 1)
            if events:
                idx[:B // 2] = ev[rng.integers(0, len(ev), B // 2)]
            idx = idx[np.lexsort((idx[:, 1], idx[:, 0]))]
            frames = np.stack([src[e, s:s + T] for e, s in idx])               # [B, T, 256, 256, 3]
            acts = torch.from_numpy(np.stack([chunk[e, s:s + T] for e, s in idx])).to(dev)
            te = time.time()
            feats = features(enc, torch.from_numpy(frames).to(dev).flatten(0, 1)).reshape(B, T, N, FEAT)
            t_enc += time.time() - te
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, now, nxt, sparse = model(feats, acts, patches)
        loss = now + (nxt if use_next else 0.0) + lam * sparse
        for g in opt.param_groups:
            g['lr'] = 4e-4 * min(1.0, (step + 1) / warm) * math.exp(-step / 100000)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        opt.step()
        step += 1
        if step % 100 == 0:
            el = time.time() - t0
            row = dict(step=step, seconds=round(el, 1), now=round(float(now), 4), next=round(float(nxt), 4),
                       sparse=round(float(sparse), 4), steps_per_s=round(step / el, 2),
                       encoder_share=round(t_enc / el, 2))
            log.append(row)
            print(json.dumps(row), flush=True)
    torch.save(dict(state=model.state_dict(), K=K, step=step,
                    args=dict(next=use_next, sparse=lam, events=events, batch=B, patches=patches)), OUT / out)
    json.dump(log, open(OUT / f'{out}.log.json', 'w'))
    print(f'final: {step} steps in {time.time() - t0:.0f}s; GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')


if __name__ == '__main__':
    main()
