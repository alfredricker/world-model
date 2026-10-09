"""Card 092's data check: in each tier's memory (066), the stored tries of pick up, drop and toggle after which a kind
of tile came into the believed view away from the front tile (in the view after the try, not in the believed view
before it, card 062's `pres`), by action and kind; and, for toggle, by the front tile's kind.

  bin/prun python tools/card092/datacheck.py        → runs/092/datacheck.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card066"))
sys.argv += ["--tier", "1"]
import tiers as T                                      # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "card061"))
import moves as M61                                    # noqa: E402

name = lambda c: T.VP.CO.name_of_code(int(c)).split("/")[0]
WORLDS = {"tier1": "runs/066/memory_tier1.npz", "tier2": "runs/066/memory_tier2.npz", "tier3": "runs/066/memory_tier3.npz"}
rep = {}
for w, p in WORLDS.items():
    z = np.load(ROOT / p)
    act, ego0, ego1, pres = z["act"], z["ego0"], z["ego1"], z["pres"]
    inter = np.flatnonzero(np.isin(act, T.INTER))
    vis1 = M61.visible(ego1[inter].astype(np.int64))
    by = {}
    for a in T.INTER:
        rows = np.flatnonzero(act[inter] == a)
        cnt, front, n_any = Counter(), Counter(), 0
        for r in rows:
            before = {c // 5 * 5 for c in np.flatnonzero(pres[r])}
            m = vis1[r, :T.NV].copy()
            m[T.FRONT] = False                         # the front tile's own change is the front effect
            after = {int(c) // 5 * 5 for c in ego1[inter[r], :T.NV][m]}
            new = {name(c) for c in after - before}
            new -= {"floor", "unseen", "wall"}
            if new:
                n_any += 1
                cnt.update(new)
                if a == T.VP.TOG:
                    front[name(ego0[inter[r], T.FRONT] // 5 * 5)] += 1
        by[T.VP.KNAME.get(a, str(a))] = {"tries": len(rows), "a kind came into view": n_any,
                                          "kinds": dict(cnt.most_common(12)), "toggle's front": dict(front.most_common(8))}
    rep[w] = by
    print(w, json.dumps(by)[:1500], flush=True)
(ROOT / "runs" / "092").mkdir(parents=True, exist_ok=True)
(ROOT / "runs" / "092" / "datacheck.json").write_text(json.dumps(rep, indent=1) + "\n")
