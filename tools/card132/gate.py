"""Card 132, gate: a state carried from touch to touch, trained to predict the scene before the next touch.

Events from build.py. One step: tokens = the 1,024 patches of the frame before touch t (64 numbers each, embedded,
with a learned position) + one token for touch t (proprioception at contact and release) + 8 state tokens. A
2-layer transformer; the patch outputs give the change of each patch by the frame before touch t+1; the state
tokens' outputs, through a GRU cell, are the state handed to step t+1. --mem 0: the state is zero at every step
(a model of the current frame and touch only).
--wchange 1: the 50 patches that truly change most at a step are counted a second time in the loss (diagnostic). Trained on m400, m600, m800 and tail (400 episodes), teacher-forced
along each episode; scored on t500, t700 and val (300 episodes never used for anything trained).
Score (truth only scores):
- error on the patches that change most at each step, against copying the current frame;
- the reopened drawer: steps where the drawer goes from closed (under 3 cm) to open (over 10 cm). Is the cube in
  the predicted frame? Decided by the nearest of two centroids, real open-drawer frames with and without the cube
  (training sets), over the 20 patches where those differ most. The same rule on the real next frame is the upper
  bound, on the current frame (copy) the trivial baseline.

  bin/prun python tools/card132/gate.py --mem 1 [--steps 2000] → runs/132/gate_mem1.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'runs/132'
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
TRAIN, TEST, W, K = ('m400', 'm600', 'm800', 'tail'), ('t500', 't700', 'val'), 128, 8


class Model(nn.Module):
    def __init__(self, mem):
        super().__init__()
        self.mem = mem
        self.inp, self.pos = nn.Linear(64, W), nn.Parameter(torch.randn(1024, W) * 0.02)
        self.touch = nn.Sequential(nn.Linear(9, W), nn.GELU(), nn.Linear(W, W))
        self.s0 = nn.Parameter(torch.randn(K, W) * 0.02)
        self.sin = nn.Linear(W, W)
        layer = nn.TransformerEncoderLayer(W, 4, 4 * W, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, 2)
        self.cell = nn.GRUCell(W, W)
        self.out = nn.Linear(W, 64)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def step(self, x, a, h):                                               # x [B,1024,64], a [B,9], h [B,K,W]
        B = len(x)
        s = self.s0[None] + (self.sin(h) if self.mem else 0)
        y = self.body(torch.cat([self.inp(x) + self.pos, self.touch(a)[:, None], s.expand(B, K, W)], 1))
        h = self.cell(y[:, 1025:].reshape(B * K, W), h.reshape(B * K, W)).reshape(B, K, W)
        return x + self.out(y[:, :1024]), h


def load(names):
    eps = []
    for n in names:
        Z, e = torch.from_numpy(np.load(OUT / f'z_{n}.npy')).cuda(), dict(np.load(OUT / f'ev_{n}.npz'))
        for k in np.unique(e['episode']):
            i = np.where(e['episode'] == k)[0]
            if len(i) >= 2:
                eps.append(dict(z=Z[i[0]:i[-1] + 1], a=torch.from_numpy(e['touch'][i]).cuda(), cube=e['cube'][i], drawer=e['drawer'][i]))
    return eps


def run(model, batch, train):
    """Teacher-forced along the episodes of a batch → predictions [B,T-1,1024,64] (float32), mask."""
    T = max(len(b['z']) for b in batch)
    B = len(batch)
    z = torch.zeros(B, T, 1024, 64, device='cuda', dtype=torch.float16)
    a = torch.zeros(B, T, 9, device='cuda')
    m = torch.zeros(B, T - 1, device='cuda', dtype=torch.bool)
    for j, b in enumerate(batch):
        n = len(b['z'])
        z[j, :n], a[j, :n], m[j, :n - 1] = b['z'], b['a'], True
    h = torch.zeros(B, K, W, device='cuda')
    preds = []
    for t in range(T - 1):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p, h = model.step(z[:, t].float(), a[:, t], h)
        h = h.float()
        preds.append(p.float())
    return torch.stack(preds, 1), z[:, 1:].float(), z[:, :-1].float(), m


def inside(cube):
    return (np.abs(cube[:, 0] - 0.33) < 0.12) & (cube[:, 2] > 0.04) & (cube[:, 1] < -0.1)


def main():
    mem, steps, wch = int(arg('--mem', 1)), int(arg('--steps', 2000)), int(arg('--wchange', 0))
    tag = arg('--tag', f'mem{mem}')
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tr, te = load(TRAIN), load(TEST)
    a_all = torch.cat([b['a'] for b in tr])
    mu, sd = a_all.mean(0), a_all.std(0) + 1e-6
    for b in tr + te:
        b['a'] = (b['a'] - mu) / sd
    model = Model(mem).cuda()
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for s in range(steps):
        batch = [tr[i] for i in rng.choice(len(tr), 6, replace=False)]
        p, y, x, m = run(model, batch, True)
        e = ((p - y) ** 2).mean(3)                                         # [B,T-1,1024]
        loss = e.mean(2)[m].mean()
        if wch:                                                            # --wchange 1: the 50 patches that truly change most count again
            loss = loss + e.gather(2, ((y - x) ** 2).mean(3).topk(50, dim=2).indices).mean(2)[m].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if s % 200 == 0 or s == steps - 1:
            print(s, 'loss', round(float(loss), 4), 'copy', round(float(((x - y) ** 2).mean((2, 3))[m].mean()), 4), round(time.time() - t0), 's', flush=True)
    model.eval()
    torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, mem=mem), OUT / f'model_{tag}.pt')
    # the rule that says whether the cube shows in the open drawer: real training frames only
    zs = torch.cat([b['z'] for b in tr]).float()
    cube, drawer = np.concatenate([b['cube'] for b in tr]), np.concatenate([b['drawer'] for b in tr])
    op = drawer < -0.10
    w, wo = torch.from_numpy(op & inside(cube)).cuda(), torch.from_numpy(op & ~inside(cube)).cuda()
    cw, co = zs[w].mean(0), zs[wo].mean(0)
    top = (cw - co).norm(dim=1).topk(20).indices
    says = lambda f: ((f[:, top] - co[top]) ** 2).sum((1, 2)) - ((f[:, top] - cw[top]) ** 2).sum((1, 2))   # > 0: cube shows
    del zs
    res = dict(mem=mem, steps=steps, train_open_with=int(w.sum()), train_open_without=int(wo.sum()))
    err = dict(model=[], copy=[])
    ev = dict(label=[], model=[], copy=[], real=[], closed=[], open_model=[], open_copy=[], cube_model=[], cube_copy=[])
    with torch.no_grad():
        for i in range(0, len(te), 6):
            batch = te[i:i + 6]
            p, y, x, m = run(model, batch, False)
            ch = ((y - x) ** 2).mean(3)                                    # [B,T-1,1024] how much each patch changes
            idx = ch.topk(20, dim=2).indices[..., None].expand(-1, -1, -1, 64)
            err['model'] += ((p.gather(2, idx) - y.gather(2, idx)) ** 2).mean((2, 3))[m].tolist()
            err['copy'] += ((x.gather(2, idx) - y.gather(2, idx)) ** 2).mean((2, 3))[m].tolist()
            for j, b in enumerate(batch):
                d, ins = b['drawer'], inside(b['cube'])
                for t in np.where((d[:-1] > -0.03) & (d[1:] < -0.10))[0]:
                    ev['label'].append(bool(ins[t + 1])); ev['closed'].append(bool(ins[t]))
                    ev['model'].append(float(says(p[j, t][None]))); ev['copy'].append(float(says(x[j, t][None])))
                    ev['real'].append(float(says(y[j, t][None])))
                    k = idx[j, t]                                           # this step's 20 most-changed patches
                    ev['open_model'].append(float(((p[j, t].gather(0, k) - y[j, t].gather(0, k)) ** 2).mean()))
                    ev['open_copy'].append(float(((x[j, t].gather(0, k) - y[j, t].gather(0, k)) ** 2).mean()))
                    ev['cube_model'].append(float(((p[j, t][top] - y[j, t][top]) ** 2).mean()))
                    ev['cube_copy'].append(float(((x[j, t][top] - y[j, t][top]) ** 2).mean()))
    res['changed_patches_mse'] = {k: round(float(np.mean(v)), 4) for k, v in err.items()}
    lab = np.array(ev['label'])
    res['reopen_events'] = dict(with_cube=int(lab.sum()), without=int((~lab).sum()))
    for k in ('real', 'model', 'copy'):
        v = np.array(ev[k])
        auc = float((v[lab][:, None] > v[~lab][None]).mean())
        res[f'reopen_{k}'] = dict(auc=round(auc, 3), says_cube_when_there=round(float((v[lab] > 0).mean()), 3),
                                  says_cube_when_not=round(float((v[~lab] > 0).mean()), 3))
    for k in ('open_model', 'open_copy', 'cube_model', 'cube_copy'):
        v = np.array(ev[k])
        res[f'reopen_mse_{k}'] = dict(with_cube=round(float(v[lab].mean()), 3), without=round(float(v[~lab].mean()), 3))
    res['reopen_model_score_mean'] = dict(with_cube=round(float(np.array(ev['model'])[lab].mean()), 2), without=round(float(np.array(ev['model'])[~lab].mean()), 2),
                                          real_with=round(float(np.array(ev['real'])[lab].mean()), 2), real_without=round(float(np.array(ev['real'])[~lab].mean()), 2))
    json.dump(dict(res, events=ev), open(OUT / f'gate_{tag}.json', 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
