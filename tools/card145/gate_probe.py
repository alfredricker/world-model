"""Card 145, gate step 2: can the questions be answered at any step, arm in view, with labels?

Card 141's labelled probe (unchanged; features mapped back from the 64 components to 768), trained on 5,000 random
steps of the training episodes and tested on every fourth step of the validation episodes. For pairs of test steps
from different episodes, the area under the curve of "does this variable differ" from the probe's two answers, in the
cells of criterion 2; and the probe's value errors.

  bin/prun python tools/card145/gate_probe.py → runs/145/gate_probe.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'card141'))
import shared as C  # noqa: E402
from probe import Probe  # noqa: E402

CEN, STEPS, t0 = np.array([0.425, 0, 0]), 4000, time.time()
p = np.load(C.OUT / 'pca.npz')
pm, V = torch.from_numpy(p['mean']).cuda(), torch.from_numpy(p['V']).cuda()
dec = lambda c: c.float() @ V.T + pm


def load(name, e, t):
    X, M = np.load(C.OUT / f'feat_{name}.npy', mmap_mode='r'), np.load(C.OUT / f'meta_{name}.npz')
    F = torch.from_numpy(np.stack([np.asarray(X[a, b]) for a, b in zip(e, t)])).cuda()
    w = C.where(M)[e, t]
    cube = ((M['cube'][e, t, :3] - CEN) * 10).astype(np.float32)
    five = np.c_[M['drawer'][e, t] * 10, M['window'][e, t] * 10, M['locks'][e, t], w != 'hidden'].astype(np.float32)
    return F, torch.tensor(cube).cuda(), torch.tensor(five).cuda(), M, w


rng = np.random.default_rng(0)
e, t = rng.integers(0, 450, 5000), rng.integers(0, 201, 5000)
o = np.lexsort((t, e))
F, cube, five, _, w = load('train', e[o], t[o])
mc = torch.tensor(w != 'hidden').cuda().float()
print('loaded', F.shape, f'{time.time() - t0:.0f}s', flush=True)
torch.manual_seed(0)
model = Probe().cuda()
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS)
bce = nn.functional.binary_cross_entropy_with_logits
for s in range(STEPS):
    b = torch.from_numpy(rng.choice(5000, 64, replace=False)).cuda()
    pc, p5 = model(dec(F[b]))
    loss = (((pc - cube[b]) ** 2).sum(-1) * mc[b]).sum() / mc[b].sum().clamp(min=1) + ((p5[:, :2] - five[b, :2]) ** 2).mean() + 0.1 * bce(p5[:, 2:], five[b, 2:])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
model.eval()
del F
ee, tt = np.meshgrid(np.arange(100), np.arange(0, 201, 4), indexing='ij')
ee, tt = ee.reshape(-1), tt.reshape(-1)
F, _, _, M, w = load('val', ee, tt)
pc, p5 = [], []
with torch.no_grad():
    for i in range(0, len(ee), 64):
        a, b = model(dec(F[i:i + 64]))
        pc.append(a.cpu().numpy()); p5.append(b.cpu().numpy())
pc, p5 = np.concatenate(pc) / 10 + CEN, np.concatenate(p5)
cells = C.pair_cells(M, (ee, tt))
sig = lambda x: 1 / (1 + np.exp(-x))
score = {'cube': lambda a, b: np.linalg.norm(pc[a] - pc[b], axis=1), 'drawer': lambda a, b: np.abs(p5[a, 0] - p5[b, 0]),
         'window': lambda a, b: np.abs(p5[a, 1] - p5[b, 1]), 'lock0': lambda a, b: np.abs(sig(p5[a, 2]) - sig(p5[b, 2])),
         'lock1': lambda a, b: np.abs(sig(p5[a, 3]) - sig(p5[b, 3]))}
out = {'frames': len(ee)}
for name, (a, b, y) in cells.items():
    out[name] = dict(auc=C.auc(score[name.split()[0]](a, b), y), same=int((~y).sum()), differ=int(y.sum()))
err = np.linalg.norm((pc - M['cube'][ee, tt, :3])[:, :2], axis=1) * 100
med = lambda x: round(float(np.median(x)), 2)
out['values'] = {'cube not in the drawer, cm': med(err[w == 'table']), 'cube in the open drawer, cm': med(err[w == 'open']),
                 'drawer, cm': med(np.abs(p5[:, 0] / 10 - M['drawer'][ee, tt]) * 100), 'window, cm': med(np.abs(p5[:, 1] / 10 - M['window'][ee, tt]) * 100),
                 'lock0 right, %': round(100 * float(((p5[:, 2] > 0) == M['locks'][ee, tt, 0]).mean()), 1), 'lock1 right, %': round(100 * float(((p5[:, 3] > 0) == M['locks'][ee, tt, 1]).mean()), 1),
                 'cube error over 2 cm, %': round(100 * float((err[w == 'table'] > 2).mean()), 1)}
json.dump(out, open(C.OUT / 'gate_probe.json', 'w'), indent=1)
print(json.dumps(out, indent=1), f'{time.time() - t0:.0f}s')
