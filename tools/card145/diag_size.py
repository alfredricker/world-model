"""Card 145, diagnostic: when a variable's dimensions open off its true steps of change, is the state's change small?

For each named variable (evaluate.py): the size of the state's change within its dimensions at openings that fall on
its true steps of change (one step of slack) and at openings that do not; and how well the size alone tells the two
apart (area under the curve).

  bin/prun python tools/card145/diag_size.py <name>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared as C  # noqa: E402

name = sys.argv[1]
d = json.load(open(C.OUT / f'eval_{name}.json'))
for k, v in d['3 variables'].items():
    print(' var', k, v)
for k, v in d['2 values'].items():
    print(' val', k, v['auc'], v['auc_whole_state'])
print(d['when a slide stops'])
s, M = np.load(C.OUT / f'states_{name}.npz'), np.load(C.OUT / 'meta_val.npz')
ch, z, g = C.changes(M)[:100], s['z_val'], s['g_val']
dz = np.abs(np.diff(z, axis=1))
for i, n in enumerate(C.NAMES):
    D = d['3 variables'][n]['dims']
    if not D:
        continue
    o, m, on = g[..., D].any(-1), np.linalg.norm(dz[..., D], axis=-1), C.wide(ch[..., i])
    print(n, 'dims', len(D), '| change at openings on its steps (median)', round(float(np.median(m[o & on])), 3), '| off its steps', round(float(np.median(m[o & ~on])), 3),
          '| size tells them apart, AUC', C.auc(m[o], on[o]), '| size against any step of change, all steps, AUC', C.auc(m.reshape(-1), ch[..., i].reshape(-1)))
