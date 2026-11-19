"""Card 145, data check 2: do the frozen patch features keep what the scene's variables need when each patch's 768
numbers are cut to k principal components?

Card 141's labelled probe, unchanged, is trained and tested on features projected onto the first k components (fitted
on 300 random learning frames) and mapped back to 768. Compared with the probe on the full features (card 141).

  bin/prun python tools/card145/check_pca.py 64 → runs/145/check_pca_64.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C
from probe import DEV, N, STEPS, Probe, targets

K, t0 = int(sys.argv[1]), time.time()
tl, tt = C.truth(C.LEARN), C.truth(C.TEST)
Xl, Xt = C.frames(C.LEARN), C.frames(C.TEST)
rng = np.random.default_rng(5)
S = torch.cat([torch.from_numpy(np.asarray(Xl[tl['set'][i]][tl['row'][i]], np.float32)) for i in rng.choice(len(tl['ep']), 300, replace=False)]).cuda()
mean = S.mean(0)
ev, V = torch.linalg.eigh(torch.cov((S - mean).T))
ev, V = ev.flip(0), V.flip(1)
expl = {k: round(float(ev[:k].sum() / ev.sum()), 4) for k in (16, 32, 64, 128, 256)}
print('variance kept', expl, f'{time.time() - t0:.0f}s', flush=True)
V = V[:, :K]
enc = lambda f: ((f.cuda().float() - mean) @ V).half()
dec = lambda c: c.float() @ V.T + mean
pick = np.sort(np.random.default_rng(0).choice(len(tl['ep']), N + DEV, replace=False))
dev = np.zeros(len(pick), bool); dev[np.random.default_rng(1).choice(len(pick), DEV, replace=False)] = True
F = torch.empty((len(pick), 1024, K), dtype=torch.float16, device='cuda')
for j, i in enumerate(pick):
    F[j] = enc(torch.from_numpy(np.asarray(Xl[tl['set'][i]][tl['row'][i]])))
print('loaded', F.shape, f'{time.time() - t0:.0f}s', flush=True)
cube, five = targets(tl)
cube, five = torch.tensor(cube[pick]).cuda(), torch.tensor(five[pick]).cuda()
mc = torch.tensor(tl['where'][pick] != 'hidden').cuda().float()
tr = np.where(~dev)[0]
torch.manual_seed(0)
model = Probe().cuda()
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
bce = nn.functional.binary_cross_entropy_with_logits
r2 = np.random.default_rng(0)
for s in range(STEPS):
    b = torch.tensor(r2.choice(tr, 64, replace=False)).cuda()
    pc, p5 = model(dec(F[b]))
    loss = (((pc - cube[b]) ** 2).sum(-1) * mc[b]).sum() / mc[b].sum().clamp(min=1) + ((p5[:, :2] - five[b, :2]) ** 2).mean() + 0.1 * bce(p5[:, 2:], five[b, 2:])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if s % 1000 == 0:
        print(s, round(loss.item(), 4), f'{time.time() - t0:.0f}s', flush=True)
model.eval()
del F
pc, p5, n = [], [], len(tt['ep'])
with torch.no_grad():
    for i0 in range(0, n, 64):
        f = torch.from_numpy(np.stack([np.asarray(Xt[tt['set'][i]][tt['row'][i]], np.float32) for i in range(i0, min(i0 + 64, n))]))
        a, b = model(dec(enc(f)))
        pc.append(a.cpu().numpy()); p5.append(b.cpu().numpy())
pc, p5 = np.concatenate(pc) / 10 + C.CEN, np.concatenate(p5)
e = np.linalg.norm((pc - tt['cube'])[:, :2], axis=1) * 100
med = lambda x: round(float(np.median(x)), 2)
out = {'components': K, 'variance kept': expl, 'cube on the table, cm': med(e[tt['where'] == 'table']), 'cube in the open drawer, cm': med(e[tt['where'] == 'open']),
       'drawer opening, cm': med(np.abs(p5[:, 0] / 10 - tt['drawer']) * 100), 'window opening, cm': med(np.abs(p5[:, 1] / 10 - tt['window']) * 100),
       'lock0 right, %': round(100 * float(((p5[:, 2] > 0) == tt['locks'][:, 0]).mean()), 1), 'lock1 right, %': round(100 * float(((p5[:, 3] > 0) == tt['locks'][:, 1]).mean()), 1)}
json.dump(out, open(ROOT / f'runs/145/check_pca_{K}.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
