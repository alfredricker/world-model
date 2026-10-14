"""Card 104.1, the main arm without a chosen weight: RAFT's evidence plus 103.4's habit, swept over every tempering tau,
on a source's evidence steps; and the share of points agreeing with the one-tile shift as a detector of motion (its
AUC and, for the evaluator only, its best threshold). Because no tau passes criteria 1 and 2 together, the
self-consistency choice of tau_M is not needed to decide.

  bin/prun python tools/card104.1/sweep.py dev       → runs/104.1/sweep_raft_w10.json (worker 10, 64 px x4 and 16 px x2)
  bin/prun python tools/card104.1/sweep.py test      → runs/104.1/sweep_raft_test.json (105.9's held-out set, 64 px x4)
"""
import json
import sys
from pathlib import Path

import numpy as np

MODE = sys.argv[1]
sys.argv = ["flow.py"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow as FL                                                    # noqa: E402


def sweep(d, F, Mk):
    t = d["steps"]
    true = 7 * (d["pos"][t + 1] - d["pos"][t]).astype(int)
    blocked = (true == 0).all(1)
    rows = []
    for tau in (1, 10, 100, 1e3, 1e4, 3e4, 1e5, 1e6):
        sc = FL.score(d, FL.decide(d, F, Mk, FL.PRIOR, tau, 0.01, 0.05))
        rows.append((tau, sc["blocked moves"][0], sc["moves that succeeded"][0]))
    a = d["act"].astype(int)[t]
    S = np.array([[s for s in FL.SHIFTS[np.flatnonzero(FL.CAND[x])] if np.abs(s).sum() > 0][0] for x in a])
    agree = ((np.linalg.norm(F.astype(np.float32) + S[:, None], axis=-1) < 1.5) & Mk).sum(1) / Mk.sum(1)
    pos, neg = agree[~blocked], agree[blocked]
    auc = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
    best = max(((((agree > th) == ~blocked)[blocked].mean(), ((agree > th) == ~blocked)[~blocked].mean(), th)
                for th in np.unique(np.round(agree, 3))), key=lambda r: min(r[0] / 0.95, r[1] / 0.994))
    return {"M + habit, by tau: (tau, blocked right, succeeded right)": rows,
            "share of points agreeing with the one-tile shift: mean on blocked, on succeeded":
                [round(float(neg.mean()), 4), round(float(pos.mean()), 4)],
            "its separation of blocked from succeeded (AUC)": round(auc, 4),
            "its best single threshold (evaluator's choice): blocked right, succeeded right, threshold":
                [round(float(best[0]), 4), round(float(best[1]), 4), float(best[2])],
            "n (blocked, succeeded)": [int(blocked.sum()), int((~blocked).sum())]}


def main():
    m = FL.Raft()
    out = {}
    if MODE == "dev":
        d = FL.source("play", 10, hi=True)
        for res in ("x4", "hi"):
            out[res] = sweep(d, *m(d, d["steps"], res))
            print(res, json.dumps(out[res]), flush=True)
    else:
        parts = []
        for kind, w in (("curr", 8), ("curr", 9), ("play", 8), ("play", 9)):
            d = FL.source(kind, w)
            F, Mk = m(d, d["steps"], "x4")
            out[f"{kind} {w}"] = sweep(d, F, Mk)
            print(kind, w, json.dumps(out[f"{kind} {w}"]), flush=True)
            parts.append((d, F, Mk))
        d = {k: np.concatenate([p[0][k] for p in parts]) for k in ("act", "pos", "light")}
        off = np.cumsum([0] + [len(p[0]["act"]) for p in parts[:-1]])
        d["steps"] = np.concatenate([p[0]["steps"] + o for p, o in zip(parts, off)])
        out["all"] = sweep(d, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))
        print("all", json.dumps(out["all"]), flush=True)
    Path(FL.OUT / f"sweep_raft_{'w10' if MODE == 'dev' else 'test'}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
