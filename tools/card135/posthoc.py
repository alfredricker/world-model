"""Card 135, after the run (not a declared arm): the staleness estimate from 100 nearest re-sightings instead of 25
(ages are whole numbers, so 25 nearest is an arbitrary 25 of the hundreds stored at one age). P = 0.25, test.

  bin/prun python tools/card135/posthoc.py → runs/135/posthoc.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import look as L
from look import B, C, D, F

mem, test = C.load(C.MEM), C.load(C.TEST)
tgm, chm, _ = C.targets(mem); tgt, cht, _ = C.targets(test)
lk, _ = B.lock_of(mem, tgm, chm)
p = 0.25
Rm, _, Ym, tm, mm = D.events(mem, tgm, chm, lk, p, 0)
rs = mm['resight']
tab = {o: rs[rs[:, 0] == o][:, 1:] for o in (3, 4)}


def stale(o, age):
    t = tab[o]; a = t[:, 0].max() if age is None else np.log1p(age)
    return float(t[np.argsort(np.abs(t[:, 0] - a), kind='stable')[:100], 1].mean())


out = dict(by_age={str(a): round(stale(3, a), 3) for a in (1, 2, 3, 4, 6, 8, 12, 16, None)})
print(out)
cond = {1: [3 + lk[1]], 2: [3 + lk[2]]}
for th in (0.08, 0.1, 0.12, 0.15, 0.2):
    out[f'age100_{th}'] = L.evaluate((mem, tgm, chm, lk, p, 0), (test, tgt, cht, lk, p, 1), cond, lambda e, i, o, age, w, th=th: stale(o, age) > th)
    print(th, out[f'age100_{th}'], flush=True)
json.dump(out, open(C.ROOT / 'runs/135/posthoc.json', 'w'), indent=1)
