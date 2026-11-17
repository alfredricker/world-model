"""Card 144, data check: where does the smooth map lose its precision?

From the score maps of dump.py. The centre of the cube group's weights is taken four ways and turned into a position
by a second-order least-squares map on the agent's own table labels (training episodes, the same rows the fit saw):
  whole    softmax over all 1,024 patches (card 142's fitted-afterwards arm)
  sharp T  the same with the scores divided by T
  window w softmax only over the (2w+1)×(2w+1) patches around the highest-scoring patch
Also: how much of the weight lies outside the 5×5 window, and an upper bound for any map (the same centres fitted on
the true cube position of the kept test frames).

  bin/prun python tools/card144/check.py a 0
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

YCUT = 0.102
wh, seed = sys.argv[1], sys.argv[2]
d = np.load(ROOT / f'runs/144/scores_{wh}_{seed}.npz')
g = np.linspace(-1, 1, 32)
GR = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(1024, 2)
RC = np.stack(np.meshgrid(np.arange(32), np.arange(32), indexing='ij'), -1).reshape(1024, 2)


def sm(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def win(S, w):
    top = RC[S.argmax(1)]                                                   # [n, 2]
    return (np.abs(RC[None] - top[:, None]).max(-1) <= w)                   # [n, 1024]


def centre(S, kind, p):
    if kind == 'whole':
        return sm(S) @ GR
    if kind == 'sharp':
        return sm(S / p) @ GR
    m = win(S, p)
    return sm(np.where(m, S, -1e9)) @ GR


def q2(c):
    return np.stack([np.ones(len(c)), c[:, 0], c[:, 1], c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1]], 1)


t = C.truth(C.TEST)
tab, y = t['where'] == 'table', t['cube'][:, 1]
Y, ep = d['Y'], d['ep']
usual = np.median(Y[ep < 80, 2])
low = Y[:, 2] < usual + 0.3
keep = (Y[:, 1] <= YCUT * 10) if wh == 'w' else np.ones(len(Y), bool)
tr = (ep < 80) & low & keep
med = lambda x: round(float(np.median(x)), 2)
out = {'labels fitted': int(tr.sum()),
       'weight outside the 5x5 window, median': med(1 - (sm(d['St'][tab]) * win(d['St'][tab], 2)).sum(1)),
       'weight on the top patch, median': med(sm(d['St'][tab]).max(1))}
for kind, p in (('whole', 0), ('sharp', 0.5), ('sharp', 0.25), ('window', 1), ('window', 2), ('window', 3)):
    B = np.linalg.lstsq(q2(centre(d['S'][tr], kind, p)), Y[tr], rcond=None)[0]
    ct = centre(d['St'], kind, p)
    e = np.linalg.norm((q2(ct) @ B / 10 + C.CEN - t['cube'])[:, :2], axis=1) * 100
    k = tab & (y <= YCUT)
    Bt = np.linalg.lstsq(q2(ct[k][::2]), t['cube'][k][::2], rcond=None)[0]
    et = np.linalg.norm((q2(ct) @ Bt - t['cube'])[:, :2], axis=1) * 100
    k2 = np.where(k)[0][1::2]
    out[f'{kind} {p}'] = dict(table=med(e[tab]), kept=med(e[k]), withheld_part=med(e[tab & (y > YCUT)]), past_over_5=med(e[tab & (y > YCUT + 0.05)]),
                              p90_kept=round(float(np.quantile(e[k], 0.9)), 2),
                              true_map_kept=med(et[k2]), true_map_withheld_part=med(et[tab & (y > YCUT)]))
    print(kind, p, out[f'{kind} {p}'], flush=True)
print({k: v for k, v in out.items() if not isinstance(v, dict)})
json.dump(out, open(ROOT / f'runs/144/check_{wh}_{seed}.json', 'w'), indent=1)
