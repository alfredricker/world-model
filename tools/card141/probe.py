"""Card 141, arm 2 (upper bound): a labelled probe on the same frozen features.

og4's read-out network with six questions instead of ten places. Per patch a score per question; softmax over patches.
  cube position   weighted mean of (a learned position per patch + offset), squared error to the true cube (units of
                  10 cm), on frames where the cube is not hidden
  drawer, window  the weighted mean of the patches' 128 numbers → one number, squared error to the opening (10 cm)
  lock0, lock1, cube locatable (not hidden)   the same pooling → a logit
Trained on N random frames of the learning sets; 500 more are kept to set the slide "differs" thresholds.
--withhold: no cube loss where the cube is on the table with y over 0.102 m (a third of the cube touches og4 learned
from), and no slide loss where that slide is 4 to 10 cm open.

  bin/prun python tools/card141/probe.py [--withhold] → runs/141/probe[_withheld].npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

N, DEV, STEPS, YCUT = 5000, 500, 4000, 0.102


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.score, self.offset = nn.Linear(128, 6), nn.Linear(128, 3)
        nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
        self.where = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        self.out = nn.Parameter(torch.zeros(5, 128)); self.bias = nn.Parameter(torch.zeros(5))
        nn.init.normal_(self.out, std=0.05)
        g = torch.linspace(-1, 1, 32)
        self.register_buffer('grid', torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(1024, 2))

    def forward(self, f):                                                  # → cube [B,3], five numbers [B,5]
        h = self.trunk(f)
        w = torch.softmax(self.score(h), 1)
        cube = torch.einsum('bp,bpd->bd', w[:, :, 0], self.where(self.grid) + self.offset(h))
        pooled = torch.einsum('bpk,bpd->bkd', w[:, :, 1:], h)
        return cube, (pooled * self.out).sum(-1) + self.bias


def targets(t):
    cube = (t['cube'] - C.CEN) * 10
    five = np.c_[t['drawer'] * 10, t['window'] * 10, t['locks'], t['where'] != 'hidden'].astype(np.float32)
    return cube.astype(np.float32), five


def main():
    withhold = '--withhold' in sys.argv
    seed = int(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == '--seed'), 0))   # card 142's repair: several seeds
    t0 = time.time()
    tl, tt = C.truth(C.LEARN), C.truth(C.TEST)
    Xl, Xt = C.frames(C.LEARN), C.frames(C.TEST)
    pick = np.sort(np.random.default_rng(0).choice(len(tl['ep']), N + DEV, replace=False))
    dev = np.zeros(len(pick), bool); dev[np.random.default_rng(1).choice(len(pick), DEV, replace=False)] = True
    F = torch.empty((len(pick), 1024, 768), dtype=torch.float16, device='cuda')
    for j, i in enumerate(pick):
        F[j] = torch.from_numpy(np.asarray(Xl[tl['set'][i]][tl['row'][i]]))
    print('loaded', F.shape, f'{time.time() - t0:.0f}s', flush=True)
    cube, five = targets(tl)
    cube, five = torch.tensor(cube[pick]).cuda(), torch.tensor(five[pick]).cuda()
    where, cy = tl['where'][pick], tl['cube'][pick, 1]
    mc = where != 'hidden'
    ms = np.ones((len(pick), 2), bool)
    if withhold:
        mc &= ~((where == 'table') & (cy > YCUT))
        for j, s in enumerate(('drawer', 'window')):
            ms[:, j] = ~((tl[s][pick] >= 0.04) & (tl[s][pick] < 0.10))
    mc, ms = torch.tensor(mc).cuda().float(), torch.tensor(ms).cuda().float()
    tr = np.where(~dev)[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Probe().cuda()
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
    bce = nn.functional.binary_cross_entropy_with_logits
    for s in range(STEPS):
        b = torch.tensor(rng.choice(tr, 64, replace=False)).cuda()
        pc, p5 = model(F[b].float())
        loss = (((pc - cube[b]) ** 2).sum(-1) * mc[b]).sum() / mc[b].sum().clamp(min=1) \
            + (((p5[:, :2] - five[b, :2]) ** 2) * ms[b]).sum() / ms[b].sum().clamp(min=1) + 0.1 * bce(p5[:, 2:], five[b, 2:])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 500 == 0:
            print(s, round(loss.item(), 4), f'{time.time() - t0:.0f}s', flush=True)
    model.eval()
    out = {}
    with torch.no_grad():
        d = np.where(dev)[0]
        pc, p5 = zip(*[tuple(x.cpu().numpy() for x in model(F[torch.tensor(d[i:i + 64]).cuda()].float())) for i in range(0, len(d), 64)])
        out['dev_cube'], out['dev_five'], out['dev_rows'] = np.concatenate(pc) / 10 + C.CEN, np.concatenate(p5), pick[d]
        del F
        pc, p5 = [], []
        n = len(tt['ep'])
        for i0 in range(0, n, 64):
            f = torch.from_numpy(np.stack([np.asarray(Xt[tt['set'][i]][tt['row'][i]], np.float32) for i in range(i0, min(i0 + 64, n))])).cuda()
            a, b = model(f)
            pc.append(a.cpu().numpy()); p5.append(b.cpu().numpy())
        out['cube'], out['five'] = np.concatenate(pc) / 10 + C.CEN, np.concatenate(p5)
    np.savez(C.OUT / f'probe{"_withheld" if withhold else ""}{f"_s{seed}" if seed else ""}.npz', **out)
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
