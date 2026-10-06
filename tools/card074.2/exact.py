"""Card 074.2's gate: the same action at every step with and without the caches (up to the shorter run).

  python tools/card074.2/exact.py runs/074.2/gate_tier2_cache0.json runs/074.2/gate_tier2_cache1.json
"""
import json
import sys

a = {e["seed"]: e for e in json.load(open(sys.argv[1]))["per_episode"]}
b = {e["seed"]: e for e in json.load(open(sys.argv[2]))["per_episode"]}
same, cut, differ = 0, 0, []
for s in sorted(a):
    x, y = a[s]["actions"], b[s]["actions"]
    n = min(len(x), len(y))
    if x[:n] != y[:n]:
        differ.append((s, next(i for i in range(n) if x[i] != y[i])))
    elif len(x) != len(y):
        cut += 1                                       # one run stopped early (out of time); the prefix agrees
    else:
        same += 1
t = lambda d: (round(sum(e["seconds"] for e in d.values()) / sum(e["steps"] for e in d.values()), 4),
               sum(e["timed_out"] for e in d.values()), sum(e["success"] for e in d.values()))
print(json.dumps({"episodes": len(a), "identical": same, "prefix_identical_one_cut": cut, "differ": differ,
                  "without_cache_s_per_step_timeouts_successes": t(a),
                  "with_cache_s_per_step_timeouts_successes": t(b)}))
