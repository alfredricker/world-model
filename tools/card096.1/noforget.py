"""Card 096.1 (check for 096.3): version 24's store built without 095.2's forgetting, for the few-occurrence curve.

  bin/prun python tools/card096.1/noforget.py 3      → runs/096.1/store_noforget_tier3.pkl
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for c in ("card095.1", "card095.2", "card095.3"):
    sys.path.insert(0, str(ROOT / "tools" / c))
import tok                                             # noqa: E402
import arm_b as AB                                     # noqa: E402
import scale as SC                                     # noqa: E402
import gate as GT                                      # noqa: E402

tier = int(sys.argv[1])
world = f"tier{tier}"
z = tok.load(world)
H, P, C, A = tok.build(z)
near_n = int((np.abs(z["wh"][:169]).sum(1) <= tok.DMAX).sum()) - 1
F, r, c = AB.fields(H, P, near_n)
ch, af, w = C[r, c], A[r, c], z["w"][r].astype(np.float64)
act = z["act"][r]
arr = z["arr"]
b_rep = json.loads((tok.RUNS / f"b_{world}.json").read_text())
stores = {}
for a in (3, 4, 5):
    s = np.flatnonzero(act == a)
    arm = GT.marginal(SC.rebuild_arm(arr, F[s], w[s], ch[s], b_rep[str(a)]["admitted"]))
    sc = s[ch[s]]
    u, inv = np.unique(np.concatenate([GT.mask_fields(F[sc]), af[sc][:, None]], 1), axis=0, return_inverse=True)
    wm = np.bincount(inv.ravel(), weights=w[sc], minlength=len(u))
    arm.Vt, arm._gc = None, None
    stores[a] = SC.Store(arm, u[:, :-1], u[:, -1], wm)
    print(a, "groups", len(arm.G), flush=True)
out = {"stores": stores, "alpha": json.loads((tok.RUNS / "c_alpha.json").read_text())[world], "arr": arr}
(ROOT / "runs" / "096.1" / f"store_noforget_{world}.pkl").write_bytes(pickle.dumps(out))
