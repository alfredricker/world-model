"""Card 145, diagnostic: what does a trained state hold? Read-outs trained on the state at every step of 100 training
episodes (20,100 labelled states; a diagnostic, far more labels than the agent may use) and tested on the validation
episodes: ridge regression, and a small network (2 layers of 256). Cube position where it is not hidden, slide
openings, locks. Beside them the same read-outs from the arm's pose alone (what needs no state) and from the state's
own dimensions that never open after an episode's first step.

  bin/prun python tools/card145/diag_state.py <name> → runs/145/diag_state_<name>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared as C  # noqa: E402

name = sys.argv[1]
d = np.load(C.OUT / f'states_{name}.npz')
M = {k: np.load(C.OUT / f'meta_{k}.npz') for k in ('train', 'val')}
nz = np.load(C.OUT / 'norm.npz')
pose = {k: ((M[k]['pose'][:100] - nz['pmean']) / nz['pstd']).reshape(-1, 12).astype(np.float32) for k in M}
Z = {'train': d['z_train'].reshape(-1, d['z_train'].shape[-1]), 'val': d['z_val'].reshape(-1, d['z_val'].shape[-1])}
mu, sd = Z['train'].mean(0), Z['train'].std(0) + 1e-6
Z = {k: (v - mu) / sd for k, v in Z.items()}
tg = {k: dict(cube=M[k]['cube'][:100, :, :2].reshape(-1, 2) * 100, drawer=M[k]['drawer'][:100].reshape(-1, 1) * 100, window=M[k]['window'][:100].reshape(-1, 1) * 100,
              lock0=M[k]['locks'][:100, :, 0].reshape(-1, 1).astype(np.float32), lock1=M[k]['locks'][:100, :, 1].reshape(-1, 1).astype(np.float32),
              seen=(C.where(M[k])[:100] != 'hidden').reshape(-1)) for k in M}


def fit(A, B, kind):
    out = {}
    for v in ('cube', 'drawer', 'window', 'lock0', 'lock1'):
        m_tr = tg['train']['seen'] if v == 'cube' else np.ones(len(A), bool)
        m_te = tg['val']['seen'] if v == 'cube' else np.ones(len(B), bool)
        y, yt = tg['train'][v][m_tr], tg['val'][v][m_te]
        ym = y.mean(0)
        if kind == 'ridge':
            X1 = np.c_[A[m_tr], np.ones(m_tr.sum())]
            W = np.linalg.solve(X1.T @ X1 + 1.0 * np.eye(X1.shape[1]), X1.T @ (y - ym))
            p = np.c_[B[m_te], np.ones(m_te.sum())] @ W + ym
        else:
            torch.manual_seed(0)
            net = nn.Sequential(nn.Linear(A.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, y.shape[1])).cuda()
            opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
            a, b = torch.from_numpy(A[m_tr]).cuda().float(), torch.from_numpy((y - ym).astype(np.float32)).cuda()
            for s in range(3000):
                i = torch.randint(len(a), (256,), device='cuda')
                loss = ((net(a[i]) - b[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                p = net(torch.from_numpy(B[m_te]).cuda().float()).cpu().numpy() + ym
        if v.startswith('lock'):
            out[v + ' right, %'] = round(100 * float(((p > 0.5) == (yt > 0.5)).mean()), 1)
        else:
            out[v + ', cm'] = round(float(np.median(np.linalg.norm(p - yt, axis=1))), 2)
    return out


still = d['g_train'].reshape(-1, d['g_train'].shape[-1]).mean(0) < 0.002
res = {'dimensions that never open after the first step': int(still.sum())}
for label, A, B in (('state', Z['train'], Z['val']), ('pose only', pose['train'], pose['val']),
                    ('state, dimensions that open', Z['train'][:, ~still], Z['val'][:, ~still])):
    if A.shape[1] == 0:
        continue
    res[label] = {'ridge': fit(A, B, 'ridge'), 'network': fit(A, B, 'net')}
    print(label, json.dumps(res[label]), flush=True)
base = {v + ', cm': round(float(np.median(np.linalg.norm(tg['val'][v] - tg['train'][v].mean(0), axis=1))), 2) for v in ('cube', 'drawer', 'window')}
res['always the mean'] = base
print('mean', base)
json.dump(res, open(C.OUT / f'diag_state_{name}.json', 'w'), indent=1)
