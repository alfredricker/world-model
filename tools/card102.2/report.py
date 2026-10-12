"""Card 102.2: the criteria from the runs' outputs → experiments/102.2-move-together-act-together/results.json.

  bin/prun python tools/card102.2/report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "102.2"
OUT = ROOT / "experiments" / "102.2-move-together-act-together" / "results.json"
J = lambda name: json.loads((RUNS / f"{name}.json").read_text())


def pooled(c2, when, kinds):
    n = sum(c2[f"{when}: {k}"]["steps"] for k in kinds)
    return round(sum(c2[f"{when}: {k}"]["every tile right"] * c2[f"{when}: {k}"]["steps"] for k in kinds) / n, 4)


def main():
    w2, cr = J("layers_world2"), J("layers_crafter")
    P = {f: J(f"predict_{f}") for f in ("tiles", "fate", "truth")}
    rep = {"gate (world 2 test)": {k: w2[k] for k in w2 if k.startswith("gate") or k == "moving or changing object tiles"},
           "learned": {"world 2": {k: w2[k] for k in w2 if k.startswith(("ground", "p (", "evidence", "prior", "share"))},
                       "crafter": {k: cr[k] for k in cr if k.startswith(("ground", "p (", "evidence", "prior", "share",
                                                                          "view shift"))}}}
    c1 = {"fate groups": w2["criterion 1: fate groups"], "colour and touching": w2["criterion 1: colour and touching"]}
    c1["holds"] = w2["criterion 1: fate groups"]["after"]["right"] >= 0.95
    when = "object moved or changed before"
    c2 = {f: P[f]["criterion 2"] for f in P}
    for f in P:
        c2[f"{f}, pushes that moved the shape"] = pooled(P[f]["criterion 2"], when, ["push, one colour", "push, several colours"])
        c2[f"{f}, blocked pushes"] = pooled(P[f]["criterion 2"], when, ["blocked push, one colour", "blocked push, several colours"])
    ratio = P["fate"]["criterion 2"][when]["every tile right"] / P["truth"]["criterion 2"][when]["every tile right"]
    c2["fate / upper bound (objects moved before; bar 0.95)"] = round(ratio, 4)
    c2["fate / upper bound, pushes that moved the shape"] = round(
        c2["fate, pushes that moved the shape"] / c2["truth, pushes that moved the shape"], 4)
    c2["holds"] = ratio >= 0.95
    c3 = {k: cr[k] for k in cr if k.startswith(("criterion 3", "states", "tracked", "fates per state", "false links"))}
    c3["holds"] = cr["criterion 3: states with a group of two or more tiles (all false in Crafter)"] <= 0.01
    rep["criterion 1"], rep["criterion 2"], rep["criterion 3"] = c1, c2, c3
    rep["first sight (the pair head, objects not yet moved)"] = {f: P[f]["first sight"] for f in P}
    rep["every tile's fate right, all test steps (own labels)"] = {
        f: P[f]["every tile's fate right (own labels), all test steps"] for f in P}
    diag = {}
    for tag, what in (("_diag", "half of each batch from event steps, 4,000 steps"),
                      ("_diagper", "prioritised replay with importance weights, 4,000 steps"),
                      ("_diagper0", "prioritised replay without importance weights, 4,000 steps")):
        if (RUNS / f"predict_tiles{tag}.json").exists():
            r = J(f"predict_tiles{tag}")["criterion 2"]
            diag[what] = {"pushes that moved the shape": pooled(r, when, ["push, one colour", "push, several colours"]),
                          "blocked pushes": pooled(r, when, ["blocked push, one colour", "blocked push, several colours"]),
                          "doors": r[f"{when}: door"]["every tile right"]}
    diag["uniform draws, 8,000 steps (fate form, first chain)"] = {"pushes that moved the shape": 0.0,
                                                                     "doors": 0.0}
    rep["training format diagnostics (per-tile form, objects moved before)"] = diag
    rep["training"] = {f: {"steps": P[f]["steps"], "replay": P[f]["replay"], "seconds": P[f]["seconds"]} for f in P}
    if all((RUNS / f"predict_{f}_r1.json").exists() for f in P):     # revision 1: a pair layer within each group
        Q = {f: J(f"predict_{f}_r1") for f in P}
        r1 = {f: Q[f]["criterion 2"] for f in Q}
        for f in Q:
            r1[f"{f}, pushes that moved the shape"] = pooled(Q[f]["criterion 2"], when, ["push, one colour", "push, several colours"])
            r1[f"{f}, blocked pushes"] = pooled(Q[f]["criterion 2"], when, ["blocked push, one colour", "blocked push, several colours"])
        r1["fate / upper bound (objects moved before; bar 0.95)"] = round(
            Q["fate"]["criterion 2"][when]["every tile right"] / Q["truth"]["criterion 2"][when]["every tile right"], 4)
        r1["fate / upper bound, pushes that moved the shape (bar 0.95)"] = round(
            r1["fate, pushes that moved the shape"] / r1["truth, pushes that moved the shape"], 4)
        r1["criterion 2 holds"] = r1["fate / upper bound (objects moved before; bar 0.95)"] >= 0.95
        r1["pushes reach the bar"] = r1["fate / upper bound, pushes that moved the shape (bar 0.95)"] >= 0.95
        r1["first sight"] = {f: Q[f]["first sight"] for f in Q}
        r1["seconds"] = {f: Q[f]["seconds"] for f in Q}
        rep["revision 1 (a pair layer within each group, attention pooling)"] = r1
    OUT.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k]["holds"] for k in ("criterion 1", "criterion 2", "criterion 3")}),
          c2["fate / upper bound (objects moved before; bar 0.95)"], c2["fate / upper bound, pushes that moved the shape"])


if __name__ == "__main__":
    main()
