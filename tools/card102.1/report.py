"""Card 102.1: the criteria from the runs' outputs → experiments/102.1-conditions-between-any-tokens/results.json.

  bin/prun python tools/card102.1/report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card102.1"))
import arms as AR                                      # noqa: E402

OUT = ROOT / "experiments" / "102.1-conditions-between-any-tokens" / "results.json"
J = lambda name: json.loads((AR.RUNS / f"{name}.json").read_text())


def nothing_changes(te, groups):
    y, sc = te["lab1"] != te["lab0"], te["scored"]
    L = AR.ll(np.zeros(y.shape), y.astype(np.float32), sc)
    whole = (~(y & sc)).all(1)
    return {g: {"log-likelihood per step": round(float(L[np.isin(te["act"], a)].mean()), 4),
                "whole scored next state right": round(float(whole[np.isin(te["act"], a)].mean()), 4)}
            for g, a in groups.items()}


def main():
    up, tok, two = J("crafter_truth"), J("crafter_tokens"), J("crafter_twoplace")
    cmp_ = J("compare")
    groups = {"make": list(AR.MAKE_SLOT), "do": [5], "place": [7, 8, 9, 10]}
    te = AR.crafter_steps((8, 9), "tokens")
    rep = {"crafter": {f: {g: r["results"][g] for g in ("make", "do", "place", "noop", "makes that made a tool")}
                       for f, r in (("upper bound", up), ("tokens (arm C, every token)", tok),
                                    ("two-place query", two))},
           "crafter nothing changes": nothing_changes(te, groups),
           "crafter tokens, arm A alone (change log-likelihood; results not saved)":
               {g: tok["arm A alone (results not saved: change log-likelihood only)"][g] for g in ("make", "do", "place")},
           "crafter paired bootstrap (tokens - two-place, per step, over episodes)": cmp_}
    c1 = {}
    c1["make: log-likelihood above the two-place query (p < 0.05)"] = cmp_["make"]["p (difference <= 0)"] < 0.05 \
        and cmp_["make"]["difference per step (tokens - two-place)"] > 0
    ratio = tok["results"]["make"]["whole scored next state right"] / up["results"]["make"]["whole scored next state right"]
    c1["make: whole scored next state right / upper bound's (>= 0.95)"] = round(ratio, 4)
    for g in ("do", "place"):
        c1[f"{g}: no worse than two-place within 0.01 nats"] = cmp_[g]["difference per step (tokens - two-place)"] >= -0.01
    c1["holds"] = bool(c1["make: log-likelihood above the two-place query (p < 0.05)"] and ratio >= 0.95
                       and all(c1[f"{g}: no worse than two-place within 0.01 nats"] for g in ("do", "place")))
    w1 = J("world1_tokens")
    c2 = {"crafter role test (tokens)": tok["role test (criterion 2)"],
          "crafter role test (two-place)": two["role test (criterion 2)"],
          "crafter role test (upper bound)": up["role test (criterion 2)"],
          "world 1 (tokens)": {k: w1["results"][k] for k in w1["results"] if k.startswith("lamp")},
          "world 1 (upper bound)": {k: v for k, v in J("world1_truth")["results"].items() if k.startswith("lamp")},
          "world 1 (two-place)": {k: v for k, v in J("world1_twoplace")["results"].items() if k.startswith("lamp")}}
    rt = tok["role test (criterion 2)"]["share passing (all 8 near >= 0.5, all 16 far < 0.5)"]
    ch = w1["results"]["lamps at pushes, changed"]["P >= 0.5 iff changed"]
    un = w1["results"]["lamps at pushes, unchanged"]["P >= 0.5 iff changed"]
    c2["holds"] = bool(rt >= 0.95 and ch >= 0.95 and un >= 0.95)
    mg = J("minigrid")
    c3 = {w: {k: mg[w][k] for k in mg[w]} for w in ("tier1", "tier2", "tier3")}
    box = mg["tier2"]["counterfactual boxes: hand holds that box"]
    c3["holds"] = bool(all(abs(mg[w]["difference from version 27 (points)"]) <= 0.01 or
                           mg[w]["difference from version 27 (points)"] > 0 for w in ("tier1", "tier2", "tier3"))
                       and all(v == 1.0 for v in box.values()))
    rep["criterion 1"], rep["criterion 2"], rep["criterion 3"] = c1, c2, c3
    rep["attention of the made tool's slot (tokens)"] = {k: {a: b for a, b in v.items() if "map" not in a}
                                                         for k, v in tok["attention of the made tool's slot"].items()}
    rep["cost"] = {f: {"query seconds per step": r["query seconds per step"],
                       "store instances (centroids)": r["store instances (centroids)"]}
                   for f, r in (("upper bound", up), ("tokens", tok), ("two-place", two))}
    rep["fit"] = {f: {k: r[k] for k in r if k.startswith(("kernel", "alpha", "median", "validation"))}
                  for f, r in (("upper bound", up), ("tokens", tok), ("two-place", two), ("world 1 tokens", w1))}
    OUT.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("criterion 1", "criterion 2", "criterion 3")}, indent=1))


if __name__ == "__main__":
    main()
