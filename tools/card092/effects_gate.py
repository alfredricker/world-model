"""Card 092's gate, "effect recall": on held-out stored toggles (a random fifth), whether each kind ever revealed came
into view, predicted by reveal.py's P(u comes into view | toggle j) from the other four fifths, against each kind's
overall frequency; mean Bernoulli log-likelihood per toggle and kind, and on the toggles that revealed something.

  <version 19's flags> bin/prun python tools/card092/effects_gate.py 2        → runs/092/effects_gate_tier2.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
tier = int(sys.argv[1])
sys.argv = ["run.py", "tiers", "--tier", str(tier), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "card092"))
import reveal as RV                                    # noqa: E402

T = RUN.T
W, info = T.setup(tier, lambda m: None)
RV.STATE.update(VP=T.VP, P={})
z = np.load(T.OUT / f"memory_tier{tier}.npz")
E_all = RV.effects(T, W, tier)
# per try (not per front), rebuilt here for the split
sys.path.insert(0, str(ROOT / "tools" / "card061"))
import moves as M61                                    # noqa: E402
act, ego0, ego1, pres = z["act"], z["ego0"], z["ego1"], z["pres"]
inter = np.flatnonzero(np.isin(act, T.INTER))
rows = np.flatnonzero(act[inter] == T.VP.TOG)
APP = T.Kd.APP
vis = M61.visible(ego1[inter[rows]].astype(np.int64))
skip = {int(APP[T.KR.code(None)])}
fr, rv = [], []
for i, r in enumerate(rows):
    before = {int(APP[c]) for c in np.flatnonzero(pres[r])}
    m = vis[i, :T.NV].copy()
    m[T.FRONT] = m[T.CENTRE] = False           # the front's own change; the agent's tile
    fr.append(int(APP[ego0[inter[r], T.FRONT]]))
    rv.append(frozenset({int(APP[c]) for c in ego1[inter[r], :T.NV][m]} - before - skip))
fr = np.array(fr)
rng = np.random.default_rng(92)
test = rng.random(len(fr)) < 0.2
kinds = sorted({u for s in rv for u in s})


def table(mask):
    uf = np.unique(fr[mask])
    idx = np.flatnonzero(mask)
    n = np.array([(fr[mask] == f).sum() for f in uf], np.float64)
    rev = [[rv[i] for i in idx if fr[i] == g and rv[i]] for g in uf]
    return {"fronts": uf, "n": n, "revealed": rev, "tries": int(mask.sum()), "with_reveal": 0}


RV.STATE["E"] = table(~test)
train_freq = {u: (sum(1 for i in np.flatnonzero(~test) if u in rv[i]) + 1e-3) / ((~test).sum() + 1e-3) for u in kinds}
ll_r, ll_f, ll_r_hit, ll_f_hit = [], [], [], []
ti = np.flatnonzero(test)
fronts_t = np.unique(fr[ti])
P = {(u, f): RV.p_reveal(W, u, int(f)) for u in kinds for f in fronts_t}
for i in ti:
    for u in kinds:
        y = u in rv[i]
        p = min(max(P[(u, fr[i])], 1e-6), 1 - 1e-6)
        q = min(max(train_freq[u], 1e-6), 1 - 1e-6)
        a, b = np.log(p if y else 1 - p), np.log(q if y else 1 - q)
        ll_r.append(a)
        ll_f.append(b)
        if y:
            ll_r_hit.append(a)
            ll_f_hit.append(b)
rep = {"tier": tier, "toggles": len(fr), "held_out": int(test.sum()), "kinds": len(kinds),
       "recall per toggle and kind": round(float(np.mean(ll_r)), 5), "frequency": round(float(np.mean(ll_f)), 5),
       "recall where the kind came into view": round(float(np.mean(ll_r_hit)), 3) if ll_r_hit else None,
       "frequency where the kind came into view": round(float(np.mean(ll_f_hit)), 3) if ll_f_hit else None,
       "reveals held out": len(ll_r_hit)}
print(json.dumps(rep), flush=True)
(ROOT / "runs" / "092" / f"effects_gate_tier{tier}.json").write_text(json.dumps(rep, indent=1) + "\n")
