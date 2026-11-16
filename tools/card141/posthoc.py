"""Card 141, after the run (not declared): is the probe's 1-4 cm slide cell low because "same" (under 1 cm) and
"different" (1 to 4 cm) touch with no gap? Rescored with a gap: same under 0.5 cm, different 2 to 4 cm, threshold 1.25 cm.
Also the probe's slide error tail, and the probe's "locatable" answer by how far the drawer is open.

  bin/prun python tools/card141/posthoc.py → runs/141/posthoc.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

t = C.truth(C.TEST)
a, b = C.pairs(t)
out = {}
for name in ('probe', 'probe_withheld'):
    p = np.load(C.OUT / f'{name}.npz')
    r = {}
    for j, s in enumerate(('drawer', 'window')):
        d, dp = np.abs(t[s][a] - t[s][b]), np.abs(p['five'][a, j] - p['five'][b, j]) / 10
        k = (d < 0.005) | ((d >= 0.02) & (d < 0.04))
        e = np.abs(p['five'][:, j] / 10 - t[s]) * 100
        r[s] = dict(gap_cell=C.bal(dp[k] > 0.0125, d[k] >= 0.02), n=[int((d < 0.005).sum()), int(((d >= 0.02) & (d < 0.04)).sum())],
                    error_cm_p50_p90_p99=[round(float(x), 2) for x in np.percentile(e, [50, 90, 99])], share_over_1cm=round(float((e > 1).mean()), 3))
    ins = t['where'] != 'table'
    view = p['five'][:, 4] > 0
    r['said_locatable_by_drawer_opening_cm'] = {f'{lo}-{hi}': [int((ins & (t['drawer'] * 100 >= lo) & (t['drawer'] * 100 < hi)).sum()),
                                                               round(float(view[ins & (t['drawer'] * 100 >= lo) & (t['drawer'] * 100 < hi)].mean()), 2)]
                                            for lo, hi in ((0, 1), (1, 3), (3, 5), (5, 8), (8, 20))}
    out[name] = r
cur = np.load(C.OUT / 'current.npz')
json.dump(out, open(C.OUT / 'posthoc.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
