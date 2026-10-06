"""Card 073's numbers from runs/073 (and version 16's from runs/072/norefit)."""
import json
import os
from math import comb

HUES = ("red", "green", "blue", "purple", "yellow", "grey")


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
    return only_new, only_old, mcnemar(only_new, only_old)


for hue in HUES:
    k, k16 = load(f"runs/073/known_{hue}.json"), load(f"runs/073/known_v16_{hue}.json")
    d, d16 = load(f"runs/073/decoy_{hue}.json"), load(f"runs/072/norefit/decoy_{hue}.json")
    line = [hue]
    if k:
        line += [f"known: order {k['success']} (failed {k['failed_seeds']}, steps {k['mean_steps_when_successful']}, "
                 f"{k['seconds_per_step']} s/step, {k['order_counts']})"]
    if k16:
        line += [f"v16 {k16['success']} steps {k16['mean_steps_when_successful']}"]
    if d and d16:
        line += [f"| folds: order {d['success']} decoy {d['decoy_tries_per_episode']} steps {d['mean_steps_when_successful']}"
                 f" vs v16 {d16['success']}; only new/only v16/p {paired(d, d16)}"]
    print(" ".join(map(str, line)))
for t in (1, 2):
    n, o = load(f"runs/073/tier{t}.json"), load(f"runs/072/norefit/tier{t}.json")
    if n:
        print("tier", t, {x: n.get(x) for x in ("success", "mean_steps_when_successful", "random_share", "seconds_per_step")},
              "v16", o and o.get("success"), "paired", o and paired(n, o))
