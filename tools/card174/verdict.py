"""Card 174: verdicts from runs/174/score_*.json.   verdict.py ladder | main <K*>"""
import json, sys
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parents[2] / "runs/174"
V = ["cube", "drawer", "window", "lock0", "lock1"]
L = lambda st, n: json.load(open(O / f"score_{st}_{n}.json"))
nm = lambda K, s: f"k{K}_s{s}h"


def line(tag, r):
    c2, c = r["criterion2"], r["criterion2c"]
    print(f"{tag}: own {list(r['reading_own'].values())} | factors {c2['n_factors']} {c2['factors']} distinct {c2['distinct']} share {list(c2['share_in_best'].values())} "
          f"at80 {c2['n_at_80']} | stray {list(c['per_variable_all_windows_pct'].values())} | held-out {r.get('held_out', {}).get('loss')} | std {r['std_per_coordinate']}")


if sys.argv[1] == "ladder":
    Ks = [4, 6, 8, 12, 16]; loss = {K: L("val", nm(K, 0))["held_out"]["loss"] for K in Ks}
    for K in Ks:
        line(f"val K={K}", L("val", nm(K, 0)))
    best = min(loss.values())
    print("held-out loss", loss, "K* =", min(K for K in Ks if loss[K] <= best + 0.02))
else:
    K = int(sys.argv[2])
    for st in ("fresh", "val"):
        R = [L(st, nm(K, s)) for s in range(3)]
        for s, r in enumerate(R):
            line(f"{st} seed {s}", r)
        own = {v: round(float(np.mean([r["reading_own"][v] for r in R])), 4) for v in V}
        sh = {v: round(float(np.mean([r["criterion2"]["share_in_best"][v] for r in R])), 1) for v in V}
        stray = {v: round(float(np.mean([r["criterion2c"]["per_variable_all_windows_pct"][v] for r in R])), 1) for v in V}
        nf = [r["criterion2"]["n_factors"] for r in R]
        print(f"{st}: mean own {own} c1 {min(own.values()) >= 0.95} | distinct seeds {sum(r['criterion2']['distinct'] for r in R)} c2a {sum(r['criterion2']['distinct'] for r in R) >= 2} "
              f"| mean share {sh} c2b {sum(x >= 80 for x in sh.values()) >= 4} | mean stray {stray} c2c {max(stray.values()) < 10} | factors {nf} c3 {len(set(nf)) == 1}")
