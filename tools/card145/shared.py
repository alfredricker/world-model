"""Card 145, shared: the truth of what changes per step, and the scores of an event signal."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/145'
NAMES = ['cube', 'drawer', 'window', 'lock0', 'lock1']


def changes(M):
    """[E, 200, 5]: does each variable change between step t and t + 1 (cube or a slide by over 0.3 cm; a lock flips)."""
    c = M['cube'][..., :3]
    return np.stack([np.linalg.norm(np.diff(c, axis=1), axis=-1) > 0.003, np.abs(np.diff(M['drawer'], axis=1)) > 0.003,
                     np.abs(np.diff(M['window'], axis=1)) > 0.003, np.diff(M['locks'][..., 0], axis=1) != 0, np.diff(M['locks'][..., 1], axis=1) != 0], -1)


def wide(x):
    y = x.copy(); y[:, 1:] |= x[:, :-1]; y[:, :-1] |= x[:, 1:]
    return y


def events(pred, ch, slack=True):
    """pred [E, 200] bool → precision against 'something changes', recall per variable; one step of slack by default."""
    any_ch = ch.any(-1)
    w = wide if slack else (lambda x: x)
    out = dict(share_of_steps=round(float(pred.mean()), 3), precision=round(float((pred & w(any_ch)).sum() / max(pred.sum(), 1)), 3),
               recall=round(float((any_ch & w(pred)).sum() / max(any_ch.sum(), 1)), 3))
    out.update({f'recall_{n}': round(float((ch[..., i] & w(pred)).sum() / max(ch[..., i].sum(), 1)), 3) for i, n in enumerate(NAMES)})
    return out


def where(M):
    """'table' (anywhere but inside the drawer, held included), 'open' (inside, drawer at least 3 cm open), 'hidden'; card 141's rule."""
    c = M['cube'][..., :3]
    ins = (np.abs(c[..., 0] - 0.33) < 0.12) & (c[..., 2] > 0.04) & (c[..., 1] < -0.1)
    return np.where(~ins, 'table', np.where(M['drawer'] >= 0.03, 'open', 'hidden'))


def auc(score, y):
    """Area under the curve of score for y (True = differs)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    r = rankdata(score)
    return round(float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())), 4)


def pair_cells(M, idx, seed=0):
    """Pairs of steps from different episodes for each cell of criterion 2. idx: (episode, step) arrays of the frames
    used. Random pairs, and for the cube also, for each frame, one frame of another episode with the cube within 1 cm
    where there is one (random pairs almost never have the cube in the same spot). → {cell: (a, b, differs)}"""
    e, t = idx
    rng = np.random.default_rng(seed)
    a = np.arange(len(e))
    b = rng.permutation(len(e))
    ok = e[a] != e[b]
    a, b = a[ok], b[ok]
    w, c = where(M)[e, t], M['cube'][e, t, :3]
    D = np.linalg.norm(c[:, None] - c[None], axis=-1)
    near = (D < 0.01) & (e[:, None] != e[None]) & (w[:, None] != 'hidden') & (w[None] != 'hidden')
    has = np.where(near.any(1))[0]
    a2 = np.r_[a, has]
    b2 = np.r_[b, [rng.choice(np.where(near[i])[0]) for i in has]].astype(int)
    d = D[a2, b2]
    gap = (d < 0.01) | (d > 0.03)
    wa, wb = w[a2], w[b2]
    cells = {'cube table': ((wa == 'table') & (wb == 'table') & gap, d > 0.03),
             'cube open drawer': (((wa == 'open') | (wb == 'open')) & (wa != 'hidden') & (wb != 'hidden') & gap, d > 0.03)}
    out = {k: (a2[m], b2[m], y[m]) for k, (m, y) in cells.items()}
    for s in ('drawer', 'window'):
        v = M[s][e, t]
        ds = np.abs(v[a] - v[b])
        m = (ds < 0.01) | ((ds >= 0.04) & (ds < 0.10))
        out[f'{s} 4-10'] = (a[m], b[m], ds[m] >= 0.04)
        m = (ds < 0.005) | ((ds >= 0.02) & (ds < 0.04))
        out[f'{s} 1-4'] = (a[m], b[m], ds[m] >= 0.02)
    for j in (0, 1):
        l = M['locks'][e, t, j]
        out[f'lock{j}'] = (a, b, l[a] != l[b])
    return out
