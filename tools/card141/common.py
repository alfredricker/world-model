"""Card 141, shared: stored frames, truth, pairs and scores of the representation test bed.

Frames: the frame before each touch of recorded play, as frozen patch features (runs/125/before_<set>.npy, fp16
[touches, 1024, 768]). Truth (runs/132/ev_<set>.npz), for scoring only: cube position (m), drawer and window opening
(m, made positive), two locks. Where the cube is: 'table'; 'open' (inside the drawer, drawer at least 3 cm open);
'hidden' (inside the drawer, drawer under 3 cm open).
Pairs: each test frame with one frame of its own episode and one of another episode of its set.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'runs/141'
LEARN, TEST = ['m400', 'm600', 'm800', 'tail'], ['t500', 't700', 'val']
CEN = np.array([0.425, 0, 0])
SLIDE_CELLS = (('1-4', 0.01, 0.04), ('4-10', 0.04, 0.10), ('over10', 0.10, 9.0))


def truth(sets):
    d = [np.load(ROOT / f'runs/132/ev_{n}.npz') for n in sets]
    t = dict(cube=np.concatenate([x['cube'] for x in d]).astype(np.float64),
             drawer=np.abs(np.concatenate([x['drawer'] for x in d])), window=np.abs(np.concatenate([x['window'] for x in d])),
             locks=np.concatenate([x['locks'] for x in d]).astype(int),
             set=np.concatenate([np.full(len(x['episode']), i) for i, x in enumerate(d)]),
             row=np.concatenate([np.arange(len(x['episode'])) for x in d]))
    t['ep'] = t['set'] * 1000 + np.concatenate([x['episode'] for x in d])
    c = t['cube']
    ins = (np.abs(c[:, 0] - 0.33) < 0.12) & (c[:, 2] > 0.04) & (c[:, 1] < -0.1)
    t['where'] = np.where(~ins, 'table', np.where(t['drawer'] >= 0.03, 'open', 'hidden'))
    return t


def frames(sets):
    return [np.load(ROOT / f'runs/125/before_{n}.npy', mmap_mode='r') for n in sets]


def pairs(t, seed=0):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for i in range(len(t['ep'])):
        same = np.where((t['ep'] == t['ep'][i]) & (np.arange(len(t['ep'])) != i))[0]
        other = np.where((t['set'] == t['set'][i]) & (t['ep'] != t['ep'][i]))[0]
        for pool in (same, other):
            if len(pool):
                a.append(i); b.append(int(rng.choice(pool)))
    return np.array(a), np.array(b)


def bal(pred, y):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    return round(100 * float((pred[y].mean() + (~pred[~y]).mean()) / 2), 1)


def score_pairs(t, a, b, differs):
    """differs: dict cube/drawer/window/lock0/lock1 → bool per pair (or None). → balanced accuracy per cell, with counts."""
    out = {}
    wa, wb = t['where'][a], t['where'][b]
    d = np.linalg.norm(t['cube'][a] - t['cube'][b], axis=1)
    same, diff = d < 0.01, d > 0.03
    cells = {'cube table': (wa == 'table') & (wb == 'table'),
             'cube open drawer': ((wa == 'open') | (wb == 'open')) & (wa != 'hidden') & (wb != 'hidden')}
    if differs.get('cube') is not None:
        for name, m in cells.items():
            k = m & (same | diff)
            out[name] = dict(bal=bal(differs['cube'][k], diff[k]), n=[int((k & same).sum()), int((k & diff).sum())])
        k = (wa == 'hidden') | (wb == 'hidden')
        y = (wa != wb)[k]
        out['cube hidden'] = dict(bal=bal(differs['cube'][k], y), n=[int((~y).sum()), int(y.sum())])
    for s in ('drawer', 'window'):
        if differs.get(s) is None:
            continue
        d = np.abs(t[s][a] - t[s][b])
        same = d < 0.01
        for name, lo, hi in SLIDE_CELLS:
            k = same | ((d >= lo) & (d < hi))
            out[f'{s} {name}'] = dict(bal=bal(differs[s][k], ~same[k]), n=[int(same.sum()), int((k & ~same).sum())])
    for j in (0, 1):
        if differs.get(f'lock{j}') is None:
            continue
        y = t['locks'][a, j] != t['locks'][b, j]
        out[f'lock{j}'] = dict(bal=bal(differs[f'lock{j}'], y), n=[int((~y).sum()), int(y.sum())])
    return out


def otsu(x):                                                               # as tools/card125/gate.py
    xs = np.sort(x)
    best, cut = -1, xs[0]
    for c in np.quantile(xs, np.linspace(0.02, 0.98, 97)):
        a, b = x[x <= c], x[x > c]
        if len(a) and len(b):
            v = len(a) * len(b) * (a.mean() - b.mean()) ** 2
            if v > best:
                best, cut = v, c
    return cut


def read_out(K=10):
    """og4's read-out, as tools/card129/readout_q.py (copied: that file's imports need packages this bed does not)."""
    import torch
    import torch.nn as nn

    class ReadOutQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
            self.score, self.offset = nn.Linear(128, K), nn.Linear(128, 3)
            nn.init.zeros_(self.offset.weight); nn.init.zeros_(self.offset.bias)
            self.where = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
            g = torch.linspace(-1, 1, 32)
            self.register_buffer('grid', torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(1024, 2))

        def forward(self, f):
            h = self.trunk(f)
            w = torch.softmax(self.score(h), 1)
            return torch.einsum('bpk,bpd->bkd', w, self.where(self.grid) + self.offset(h))
    return ReadOutQ()
