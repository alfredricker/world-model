"""Card 134, diagnostic after the run (not a declared arm): card 133.1's learned correction of the key (network
arm, beta chosen on held-out stored episodes) over the contents with and without the last effects, 400 episodes.

  bin/prun python tools/card134/learned.py → runs/134/learned.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card133.1'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card133'))
import effects as E
import prior as P
from effects import B, C

mem, test = C.load(C.MEM), C.load(C.TEST)
tgm, chm, _ = C.targets(mem)
tgt, cht, _ = C.targets(test)
lk, _ = B.lock_of(mem, tgm, chm)
Xm, Ym, mm = E.events(mem, tgm, chm, lk)
Xt, Yt, mt = E.events(test, tgt, cht, lk)
keep = ~mm['mixed']
out = {}
for a in ('seen', 'effects'):
    X, Y, ep = Xm[a][keep], Ym[keep], mm['ep'][keep]
    errs = {b: P.held_out_error(X, Y, ep, 'network', b) for b in P.BETAS}
    beta = min(errs, key=errs.get)
    rows = []
    for s in range(3):
        g = P.fit(X, Y, ep, 'network', beta, s)
        rows.append(E.score(*E.answer(g(X), Y, g(Xt[a]), mt['slide']), mt, None))
    out[a] = dict(beta=beta, **{k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}, pull_only_seeds=[r['pull_only'] for r in rows])
    print(a, json.dumps(out[a]), flush=True)
json.dump(out, open(C.ROOT / 'runs/134/learned.json', 'w'), indent=1)
