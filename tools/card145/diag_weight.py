"""Card 145, diagnostic after the first pass: how much of the reconstruction loss is each scene variable worth?

With the gate's two decoders (true state + pose; pose only) on 2,000 validation steps. A variable's worth is the
error added when its true value is swapped for that of another random step, as a share of the error the pose-only
decoder leaves (what the scene state as a whole could explain is that error minus the true-state decoder's).
Under three ways of weighting patches:
  uniform        every patch alike (the first pass)
  per place      each patch position divided by its own variance over time, floored at the 10th percentile
  where it moves each patch weighted, per step, by its change between the step before and the step after
The 9-minute decoder draws the cube only roughly (gate step 1), so the cube's worth is a lower bound.

  bin/prun python tools/card145/diag_weight.py → runs/145/diag_weight.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared as C  # noqa: E402
from model import Decoder  # noqa: E402
from train import norms  # noqa: E402

nz = norms()
X, M = np.load(C.OUT / 'feat_val.npy', mmap_mode='r'), np.load(C.OUT / 'meta_val.npz')
Mt = np.load(C.OUT / 'meta_train.npz')
st = lambda m: np.concatenate([m['cube'], m['drawer'][..., None], m['window'][..., None], m['locks']], -1).astype(np.float32)
s_tr = st(Mt).reshape(-1, 11)
mean, scale = torch.from_numpy(nz['mean']).cuda(), float(nz['scale'])
dec = {}
for k in ('truth', 'pose'):
    dec[k] = Decoder(11).cuda(); dec[k].load_state_dict(torch.load(C.OUT / f'gate_{k}.pt')['state']); dec[k].eval()
rng = np.random.default_rng(0)
n = 2000
e, t = rng.integers(0, 100, n), rng.integers(1, 200, n)
o = np.lexsort((t, e)); e, t = e[o], t[o]
truth = torch.from_numpy((st(M)[e, t] - s_tr.mean(0)) / (s_tr.std(0) + 1e-6)).cuda()
pose = torch.from_numpy(((M['pose'][e, t] - nz['pmean']) / nz['pstd']).astype(np.float32)).cuda()
cols = {'cube': list(range(7)), 'drawer': [7], 'window': [8], 'lock0': [9], 'lock1': [10]}
perm = torch.from_numpy(rng.permutation(n)).cuda()
E = {k: [] for k in ['pose', 'truth', *cols]}
MOVE, VAR = [], 0
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for i in range(0, n, 100):
        sl = slice(i, i + 100)
        get = lambda dt: (torch.from_numpy(np.stack([np.asarray(X[a, b + dt]) for a, b in zip(e[sl], t[sl])])).cuda().float() - mean) / scale
        x = get(0)
        MOVE.append(((get(1) - get(-1)) ** 2).sum(-1).cpu())
        VAR = VAR + (x ** 2).sum(-1).sum(0).cpu()
        E['pose'].append(((x - dec['pose'](truth[sl] * 0, pose[sl]).float()) ** 2).sum(-1).cpu())
        E['truth'].append(((x - dec['truth'](truth[sl], pose[sl]).float()) ** 2).sum(-1).cpu())
        for v, c in cols.items():
            s2 = truth[sl].clone(); s2[:, c] = truth[perm[sl]][:, c]
            E[v].append(((x - dec['truth'](s2, pose[sl]).float()) ** 2).sum(-1).cpu())
E = {k: torch.cat(v).numpy() for k, v in E.items()}                        # [n, 1024]
move, var = torch.cat(MOVE).numpy(), (VAR / n).numpy()
W = {'uniform': np.ones((n, 1024)), 'per place': np.broadcast_to(1 / np.maximum(var, np.quantile(var, 0.1)), (n, 1024)),
     'where it moves': move / move.sum(1, keepdims=True) * 1024}
out = {}
for name, w in W.items():
    base = (w * E['pose']).sum()
    out[name] = {'left by the true-state decoder': round(float((w * E['truth']).sum() / base), 3)}
    out[name].update({v: round(float((w * (E[v] - E['truth'])).sum() / base), 4) for v in cols})
json.dump(out, open(C.OUT / 'diag_weight.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
