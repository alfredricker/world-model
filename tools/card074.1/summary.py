"""Card 074.1's numbers from runs/074.1, against version 17 (runs/073) and card 074 (runs/074).

  python tools/card074.1/summary.py > runs/074.1/summary.json
"""
import json
import os
from math import comb

HUES = ("red", "green", "blue", "purple", "yellow", "grey")
O = "runs/074.1"


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def load(f):
    return json.load(open(f)) if os.path.exists(f) else None


def paired(new, old):
    a = {r["seed"]: r["success"] for r in new["per_episode"]}
    b = {r["seed"]: r["success"] for r in old["per_episode"]}
    only_new = sum(a[s] and not b[s] for s in a)
    only_old = sum(b[s] and not a[s] for s in a)
    return {"only_new": only_new, "only_old": only_old, "p": round(mcnemar(only_new, only_old), 4)}


def cause(e):
    """Card 074's first causes for a failed tier 2 episode."""
    if e["success"]:
        return "success"
    if e["timed_out"]:
        return "out of time"
    if e["explore"] / e["steps"] > 0.9:
        return "loop (explores throughout)"
    if e["random"] / e["steps"] > 0.5:
        return "mostly random"
    return "other"


KEYS = ("found", "followed", "single", "tie_kept", "tie_cost", "tie_order", "cycles", "flips", "steps", "steps_spliced",
        "split_hand_met", "split_view_held")
v17 = {"tier1": load("runs/073/tier1.json"), "tier2": load("runs/073/tier2.json")}
loops17 = [e["seed"] for e in v17["tier2"]["per_episode"] if cause(e) == "loop (explores throughout)"]
out = {"version17_loops": len(loops17)}
for arm in "AB":
    r = {}
    known = {h: load(f"{O}/{arm}_known_{h}.json") for h in HUES}
    r["known"] = {h: {"success": k["success"], "failed": k["failed_seeds"], "steps": k["mean_steps_when_successful"]}
                  for h, k in known.items() if k}
    t2 = load(f"{O}/{arm}_tier2.json")
    if t2:
        by = {e["seed"]: e for e in t2["per_episode"]}
        causes = {}
        for e in t2["per_episode"]:
            causes[cause(e)] = causes.get(cause(e), 0) + 1
        r["tier2"] = {"success": t2["success"], "steps": t2["mean_steps_when_successful"],
                      "out_of_time": t2["episodes_out_of_time"], "seconds_per_step": t2["seconds_per_step"],
                      "vs_v17": paired(t2, v17["tier2"]), "vs_074": paired(t2, load("runs/074/tier2.json")),
                      "v17_loops_no_longer_looping": sum(cause(by[s]) != "loop (explores throughout)" for s in loops17),
                      "v17_loops_solved": sum(by[s]["success"] for s in loops17), "causes": causes,
                      "counts": {k: t2["counts"].get(k) for k in KEYS}}
    t1 = load(f"{O}/{arm}_tier1.json")
    if t1:
        r["tier1"] = {"success": t1["success"], "steps": t1["mean_steps_when_successful"],
                      "seconds_per_step": t1["seconds_per_step"], "vs_v17": paired(t1, v17["tier1"])}
    folds = {h: load(f"{O}/{arm}_decoy_{h}.json") for h in HUES}
    r["folds"] = {h: {"success": d["success"], "v17": load(f"runs/073/decoy_{h}.json")["success"],
                      **paired(d, load(f"runs/073/decoy_{h}.json"))} for h, d in folds.items() if d}
    out[arm] = r
print(json.dumps(out, indent=1))
