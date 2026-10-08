"""Card 086's gate: version 19's recall (general.py) on card 084's tables, through the agent's own recall (W.outcome).

  WM_REL_ENCODER=runs/070/encoder.pt <version 18's flags> bin/prun python tools/card086/gate.py none   # decoy memory,
      nothing removed: card 079's toggle table (56), pick up (28), drop (42)
  ... gate.py tier2                                                    # tier 2's hand tables (pick up 9, toggle 3)
  ... gate.py red                                                      # a fold: the hue's openings removed (report)
Writes runs/086/gate_<mode>.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
TIER = 2 if MODE == "tier2" else 1
if TIER == 2:
    sys.argv = ["run.py", "tiers", "--tier", "2", "--online", "0", "--memory", "066"]
else:
    sys.argv = ["run.py", "decoy", "--fold", MODE if MODE != "none" else "red", "--hold", "opening", "--online", "0",
                "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "card086"))
import general as GEN                                  # noqa: E402

T, R = RUN.T, RUN.R
VP = T.VP
t00 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)

if TIER == 1:                                          # as card 084: decoy memory; a fold drops its hue's openings
    T.ENVS[1] = R.register()
    T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
    R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
    _memory = T.memory

    def memory(tier, pool=None):
        D, stats = _memory(tier, pool)
        if MODE == "none":
            return D, stats
        f = D["ego0"][:, T.FRONT].astype(np.int64) // 5 * 5
        h = D["ego0"][:, T.HELD].astype(np.int64) // 5 * 5
        drop = (D["act"] == VP.TOG) & (f == T.KR.code(("door", MODE, 0))) & (h == T.KR.code(("key", MODE)))
        inter = np.isin(D["act"], T.INTER)
        keep_t = ~drop[inter]
        D = {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}
        return D, {**stats, "opening_tries_removed": int(drop.sum()), "rows": int(len(D["act"]))}

    T.memory = memory

GEN.hook(T)
t_setup = time.monotonic()
W, info = T.setup(TIER, log)
t_setup = time.monotonic() - t_setup
ctx, empty = GEN.STATE["ctx"], GEN.STATE["empty"]
app = lambda o: int(T.Kd.APP[T.KR.code(o)])
NAME = {}
for h in range(len(W.S.arr)):                          # the evaluator's names, to build the tables only
    try:
        NAME.setdefault(W.judge.name(h), h)
    except Exception:                                  # noqa: BLE001
        pass
if TIER == 1:                                          # cards 079 and 082's tables (as card 084)
    names = {**{f"door {c}": app(("door", c, 0)) for c in T.HUES}, **{f"key {c}": app(("key", c)) for c in T.HUES},
             **{nm: NAME[nm] for nm in ("wall", "floor") if nm in NAME}, "nothing": empty}
    fronts = [n for n in names if n != "nothing"]
    TABLES = {
        "toggle": [(fn, hn, fn.startswith("door") and hn.startswith("key") and fn[5:] == hn[4:])
                   for fn in [f"door {c}" for c in T.HUES] + ["wall", "floor"]
                   for hn in [f"key {c}" for c in T.HUES] + ["nothing"]],
        "pick up": [(fn, hn, fn.startswith("key") and hn == "nothing") for fn in fronts for hn in ("nothing", "key red")],
        "drop": [(fn, hn, fn == "floor" and hn != "nothing") for fn in fronts for hn in ("nothing", "key red", "key blue")],
    }
else:                                                  # tier 2: the layout's own box, ball, key and locked door
    env = T.make(2)
    env.reset(seed=T.SEED_TEST + 2000)
    objs = {}
    g = env.unwrapped.grid
    for i in range(g.width):
        for j in range(g.height):
            o = g.get(i, j)
            if o is not None and o.type in ("box", "ball", "key", "door"):
                objs[o.type] = (o.type, o.color, 0) if o.type == "door" else (o.type, o.color)
    names = {**{f"{o[0]} {o[1]}": app(o) for o in objs.values()}, "nothing": empty}
    door, box, ball, key = (f"{objs[t][0]} {objs[t][1]}" for t in ("door", "box", "ball", "key"))
    TABLES = {
        "pick up": [(fn, hn, hn == "nothing") for fn in (box, ball, key) for hn in ("nothing", ball, key)],
        "toggle": [(door, hn, hn == key) for hn in ("nothing", ball, key)],
    }
ACT = {"pick up": VP.PICK, "drop": VP.DROP, "toggle": VP.TOG}
res = {"note": "Card 086 gate, tools/card086/gate.py", "mode": MODE, "setup_seconds": round(t_setup, 1),
       "general": info.get("general"), "tables": {}}
for an, rows in TABLES.items():
    a = ACT[an]
    kd = W.kinds[a]
    out = []
    for fn, hn, truth in rows:
        q = (names[fn], names[hn], ctx)
        c = int(W.outcome(a, q[0], q[1], ctx)[0])
        pv = kd.predict([q])[0]
        base = GEN.STATE["base"][a](kd, [q])[0]
        out.append({"front": fn, "held": hn, "truth": bool(truth), "v19": c, "p_change": round(float(pv[1:].sum()), 4),
                    "v18_p_change": round(float(base[1:].sum()), 4), "own": float(kd.own_counts(q).sum())})
    right = sum((r["v19"] != 0) == r["truth"] for r in out)
    res["tables"][an] = {"right": right, "of": len(out),
                         "wrong": [(r["front"], r["held"], r["p_change"]) for r in out if (r["v19"] != 0) != r["truth"]],
                         "cells": out}
    log(f"{an}: {right}/{len(out)} right; wrong {res['tables'][an]['wrong']}")
if TIER == 1 and MODE != "none":
    pair = [r for r in res["tables"]["toggle"]["cells"] if r["front"] == f"door {MODE}" and r["held"] == f"key {MODE}"][0]
    res["fold_pair"] = pair
    log(f"fold pair: {pair}")
out_p = ROOT / "runs" / "086" / f"gate_{MODE}.json"
out_p.parent.mkdir(parents=True, exist_ok=True)
out_p.write_text(json.dumps(res, indent=1, default=str) + "\n")
log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cells"} for k, v in res["tables"].items()}))
