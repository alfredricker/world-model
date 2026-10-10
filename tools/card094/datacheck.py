"""Card 094's data check, on version 20 (flags in the environment).

  memory:   the tokens in stored views, attended or not (attend.py's rule without the goal and kept choices);
            how many stored view sets hold unattended tokens; recall's admitted view conditions, attended or not
  episodes: per step, the planner's candidate tokens (things) against the attended ones, believed places against
            object-file places, recall's view set against its attended part; tokens acted on (picked up, toggled,
            dropped) that attention would leave out

  <version 20's flags> bin/prun python tools/card094/datacheck.py TIER OUT.json [SEED ...]   (no seeds: memory only)
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
tier, out, seeds = int(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
sys.argv = ["run.py", "tiers", "--tier", str(tier), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "card094"))
import attend as AT                                    # noqa: E402

T = RUN.T
VP, TK = T.VP, T.TK
T.CAP_SECONDS = float(os.environ.get("WM_CAP", T.CAP_SECONDS))
W, _ = T.setup(tier, lambda m: print(m, flush=True))
SP = sys.modules["slot_planner"]
inv = {int(hh): t for t, hh in reversed(list(enumerate(W.S.hof.tolist())))}
name = lambda h: VP.CO.name_of_code(inv[int(h)]) if int(h) in inv else f"h{int(h)}"
static = lambda h: AT.changeable(W, VP, h) or AT.ends(W, VP, h)

# ---------------------------------------------------------------- memory
rep = {"tier": tier, "memory": {}}
for a, an in ((VP.PICK, "pickup"), (VP.TOG, "toggle"), (VP.DROP, "drop")):
    kd = W.kinds[a]
    n = len(kd.keys)
    sets = Counter(int(k[2]) for k in kd.keys[:n])
    toks = Counter()
    with_un, size, size_att = 0, [], []
    for sid, cnt in sets.items():
        s = [int(h) for h in SP.SETS[sid]]
        toks.update({h: cnt for h in s})
        att = [h for h in s if static(h)]
        with_un += cnt * (len(att) < len(s))
        size += [len(s)] * cnt
        size_att += [len(att)] * cnt
    adm = (kd.report.get("conditions", {}) or {}).get("admitted", []) if hasattr(kd, "report") else []
    rep["memory"][an] = {
        "stored keys": n, "distinct view sets": len(sets),
        "tokens in stored views": {name(h): {"keys": c, "attended": bool(static(h))} for h, c in toks.most_common()},
        "keys whose view holds an unattended token": with_un,
        "mean view set": round(float(np.mean(size)), 2) if size else None,
        "mean attended part": round(float(np.mean(size_att)), 2) if size_att else None,
        "admitted conditions": adm,
    }
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tokens in stored views"}
                  for k, v in rep["memory"].items()}), flush=True)

# ---------------------------------------------------------------- episodes
CUR = {}
STEP = []
ACTED = []
PL = T.S7.Plan047
_choose = PL.choose


def choose(self, st):
    CUR["pl"], CUR["st"] = self, st
    f = self.facts[st[0]]
    th = self.things(f)
    att = [u for u in th if AT.attended(W, VP, self, st, u)]
    lat = f[TK.LAT]
    seen = lat[lat != TK.ABSENT]
    files = int(sum(AT.attended(W, VP, self, st, int(u)) for u in seen))
    ctx = [int(h) for h in SP.SETS[int(self.ctx_id(st[0], st[1]))]]
    STEP.append((len(th), len(att), len(seen), files, len(ctx), sum(static(h) for h in ctx)))
    return _choose(self, st)


PL.choose = choose
_learn = type(W).learn_try


def learn_try(self, Vb, a, Va, te, *k, **kw):
    pl, st = CUR.get("pl"), CUR.get("st")
    if a in (VP.PICK, VP.TOG, VP.DROP) and pl is not None:
        h = int(Vb[VP.HELD]) if a == VP.DROP else int(Vb[VP.FRONT])
        if a != VP.DROP or h != int(T.Kd.APP[0]):
            ACTED.append((int(a), h, bool(AT.attended(W, VP, pl, st, h))))
    return _learn(self, Vb, a, Va, te, *k, **kw)


type(W).learn_try = learn_try
eps = []
for s in seeds:
    STEP.clear(), ACTED.clear()
    r = RUN.episode((tier, s))
    S = np.array(STEP) if STEP else np.zeros((0, 6))
    missed = Counter(f"{['', '', '', 'pickup', 'drop', 'toggle'][a] if a < 6 else a} {name(h)}"
                     for a, h, att in ACTED if not att)
    e = {"seed": s, "success": bool(r["success"]), "steps": int(r["steps"]),
         "things": round(float(S[:, 0].mean()), 1) if len(S) else None,
         "attended things": round(float(S[:, 1].mean()), 1) if len(S) else None,
         "believed places": round(float(S[:, 2].mean()), 1) if len(S) else None,
         "object-file places": round(float(S[:, 3].mean()), 1) if len(S) else None,
         "recall view set": round(float(S[:, 4].mean()), 2) if len(S) else None,
         "attended part": round(float(S[:, 5].mean()), 2) if len(S) else None,
         "acted on": len(ACTED), "acted on, unattended": dict(missed)}
    eps.append(e)
    print(json.dumps(e), flush=True)
    rep["episodes"] = eps
    out.write_text(json.dumps(rep, indent=1) + "\n")
out.write_text(json.dumps(rep, indent=1) + "\n")
