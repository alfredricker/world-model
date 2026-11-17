"""Card 144: every arm of fit.py, and og4's form (runs/142/fit_og_*.npz) beside it, on the test bed's cube cells.

Median error (cm, in the plane) with the cube on the table (whole; the kept part; the withheld third; over 5 cm past
its edge) and in the open drawer; how often the height cluster is chosen rightly; the pair cells "both on the table"
and "one in the open drawer", with og4's rule for the cut (twice the 90th percentile of the arm's own held-out error).

  bin/prun python tools/card144/score.py → runs/144/score.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C

YCUT = 0.102
med = lambda x: round(float(np.median(x)), 2)
t = C.truth(C.TEST)
a, b = C.pairs(t)
tab, opn, y = t['where'] == 'table', t['where'] == 'open', t['cube'][:, 1]


def arm(d):
    e = np.linalg.norm(d['aim'][:, :2] - t['cube'][:, :2], axis=1) * 100
    own = d['own_err_cm'][d['own_group'] == 3] if 'own_group' in d else d['own_err_cm']
    cut = 2 * float(np.quantile(own, 0.9)) / 100
    pr = C.score_pairs(t, a, b, dict(cube=np.linalg.norm(d['aim'][a, :2] - d['aim'][b, :2], axis=1) > cut))
    r = dict(table=med(e[tab]), kept=med(e[tab & (y <= YCUT)]), withheld_part=med(e[tab & (y > YCUT)]), past_over_5=med(e[tab & (y > YCUT + 0.05)]),
             table_p90=round(float(np.quantile(e[tab], 0.9)), 2), open_drawer=med(e[opn]), open_within_1cm=round(100 * float((e[opn] < 1).mean()), 1),
             cut_cm=round(cut * 100, 2), pair_table=pr['cube table']['bal'], pair_open=pr['cube open drawer']['bal'])
    if 'chose_table' in d:
        eh = np.linalg.norm(d['aim_high'][:, :2] - t['cube'][:, :2], axis=1) * 100
        r.update(open_drawer_told_the_height=med(eh[opn]), chose_table_on_table=round(100 * float(d['chose_table'][tab].mean()), 1),
                 chose_above_in_open_drawer=round(100 * float((~d['chose_table'][opn]).mean()), 1),
                 shift_patches_table=med(d['shift_patches'][tab]), largest_score_table=med(d['top'][tab]), largest_score_open=med(d['top'][opn]))
    return r


out, by = dict(fits={}), {}
for f in sorted(glob.glob(str(ROOT / 'runs/142/fit_og_*.npz')) + glob.glob(str(ROOT / 'runs/144/fit_*.npz'))):
    name = Path(f).stem[4:]
    out['fits'][name] = arm(dict(np.load(f)))
    by.setdefault(name.rsplit('_', 1)[0], []).append(out['fits'][name])
out['mean'] = {k: dict(seeds=len(v), **{m: round(float(np.mean([x[m] for x in v])), 2) for m in v[0]},
                       per_seed={m: [x[m] for x in v] for m in ('table', 'withheld_part', 'open_drawer')}) for k, v in by.items()}
json.dump(out, open(ROOT / 'runs/144/score.json', 'w'), indent=1)
for k, v in out['mean'].items():
    print(k, json.dumps(v))
