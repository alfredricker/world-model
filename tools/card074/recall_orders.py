"""Card 074's report on remembered conflicts: would recall over stored weighings have predicted the derivation's
orders on the test layouts?

Memory: every weighing of the training layouts (runs/074/train_*.json; seeds from 500,000), each with the two
conditions' kinds and actions, the codes and vectors of the tiles they name and of the tile held, its outcome
("n first" or "none") and how often it occurred. Recall, in its two levels (cards 042, 050): own tries first (the same
kinds and actions, the same three codes); otherwise similar tries of the same kinds and actions, weighted by
exp(-lambda * L1 distance) between the three tiles' vectors, lambda fitted by leave-one-out likelihood on the
training memory. Within an episode, each weighing is added after it is predicted. The derivation decided every
order; this only scores recall beside it. A simplification of recall's full admission (cards 049–051), declared in the
card's result.

  bin/prun python tools/card074/recall_orders.py --out runs/074/recall_orders.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == "--out"),
                "runs/074/recall_orders.json"))
PRIOR = 1.0


def episodes(f):
    d = json.load(open(f))
    return [r.get("weighings", []) for r in d["per_episode"]]


def group(w):
    return (tuple(w["A"]), tuple(w["n"]))


def own(w):
    c = lambda x: None if x is None else x[0]
    return group(w) + (c(w["uA"]), c(w["un"]), c(w["held"]))


def vec(w):
    v = lambda x: np.zeros(32) if x is None else np.asarray(x[1], float)
    return np.concatenate([v(w["uA"]), v(w["un"]), v(w["held"])])


class Memory:
    def __init__(self, ws, lam=1.0):
        self.lam = lam
        self.own = defaultdict(lambda: np.zeros(2))
        self.grp = defaultdict(list)
        for w in ws:
            self.add(w)

    def add(self, w):
        y = 1 if w["outcome"] == "n first" else 0
        self.own[own(w)][y] += w.get("count", 1)
        self.grp[group(w)].append((vec(w), y, w.get("count", 1), own(w)))

    def predict(self, w, exclude_own=False, local=None):
        """(P(n first), level) or (None, "none"); local: the episode's own earlier weighings, added to memory."""
        o = self.own.get(own(w), np.zeros(2)) + (local.own.get(own(w), np.zeros(2)) if local else 0)
        if o.sum() > 0 and not exclude_own:
            return (o[1] + PRIOR / 2) / (o.sum() + PRIOR), "own"
        rows = self.grp.get(group(w), []) + (local.grp.get(group(w), []) if local else [])
        rows = [r for r in rows if not (exclude_own and r[3] == own(w))]
        if not rows:
            return None, "none"
        X = np.stack([r[0] for r in rows])
        y = np.array([r[1] for r in rows], float)
        c = np.array([r[2] for r in rows], float)
        k = c * np.exp(-self.lam * np.abs(X - vec(w)).sum(1))
        return (k @ y + PRIOR / 2) / (k.sum() + PRIOR), "similar"


def fit_lambda(ws):
    best = None
    for lam in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        m = Memory(ws, lam)
        ll = 0.0
        for w in ws:
            p, lev = m.predict(w, exclude_own=True)
            if p is None:
                continue
            y = 1 if w["outcome"] == "n first" else 0
            ll += w.get("count", 1) * np.log(p if y else 1 - p)
        if best is None or ll > best[1]:
            best = (lam, ll)
    return best[0]


def score(mem, eps, name):
    rows = []
    for ep in eps:
        local = Memory([], mem.lam)
        for w in ep:
            p, lev = mem.predict(w, local=local)
            y = w["outcome"] == "n first"
            rows.append({"src": name, "group": "/".join(f"{a[0]}:{a[1]}" for a in group(w)), "level": lev,
                         "seen": own(w) in mem.own, "truth": y,
                         "pred": None if p is None else bool(p >= 0.5), "count": w.get("count", 1)})
            local.add(w)
    return rows


def summary(rows):
    def agg(rs):
        n = sum(r["count"] for r in rs)
        if not n:
            return None
        ans = [r for r in rs if r["pred"] is not None]
        agree = sum(r["count"] for r in ans if r["pred"] == r["truth"])
        conf = [r for r in rs if r["truth"]]
        hit = sum(r["count"] for r in conf if r["pred"])
        base = sum(r["count"] for r in rs if not r["truth"])
        return {"weighings": n, "answered": round(sum(r["count"] for r in ans) / n, 4),
                "agree": round(agree / n, 4), "always_none_baseline": round(base / n, 4),
                "orders_in_truth": sum(r["count"] for r in conf),
                "orders_predicted": round(hit / sum(r["count"] for r in conf), 4) if conf else None}
    out = {"all": agg(rows)}
    for key in ("src", "group", "level", "seen"):
        for v in sorted({str(r[key]) for r in rows}):
            out[f"{key}={v}"] = agg([r for r in rows if str(r[key]) == v])
    return out


def main():
    train = [w for f in ("runs/074/train_tier1.json", "runs/074/train_tier2.json", "runs/074/train_decoy.json")
             for ep in episodes(f) for w in ep]
    lam = fit_lambda(train)
    mem = Memory(train, lam)
    rows = []
    for f, name in ([(f"runs/074/known_{h}.json", "decoy key known") for h in
                     ("red", "green", "blue", "purple", "yellow", "grey")]
                    + [(f"runs/074/decoy_{h}.json", "decoy trying") for h in
                       ("red", "green", "blue", "purple", "yellow", "grey")]
                    + [("runs/074/tier1.json", "tier 1"), ("runs/074/tier2.json", "tier 2")]):
        if Path(f).exists():
            rows += score(mem, episodes(f), name)
    res = {"note": "Card 074: recall over stored weighings scored beside the derivation (report only)",
           "memory_weighings": int(sum(w.get("count", 1) for w in train)), "memory_distinct": len(train),
           "lambda": lam, "summary": summary(rows)}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("memory_weighings", "memory_distinct", "lambda")}))
    for k, v in res["summary"].items():
        print(k, v)


if __name__ == "__main__":
    main()
