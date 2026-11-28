"""Card 173: read runs/173/score_*.json and print the gate's and the criteria's verdicts.   verdict.py gate | ladder | main <K*>"""
import json, sys
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parents[2] / "runs/173"
B = json.load(open(O.parent / "172/score_baselines.json"))
V = ["cube", "drawer", "window", "lock0", "lock1"]
L = lambda n: json.load(open(O / f"score_{n}.json"))
triv = {v: max(B[k]["criterion1"]["validation"][v]["whole"] for k in B if "168" not in k) for v in V}
c1 = lambda r, s="validation": {v: r["criterion1"][s][v]["whole"] for v in V}


def line(n, r):
    c2 = r["criterion2"]
    print(f"{n}: K {r['K']} val {c1(r)} | test {c1(r, 'test frames (card 141)')} | factors {c2['n_factors']} {c2['factors']} distinct {c2['distinct']} "
          f"share {c2['share_in_best']} at80 {c2['n_at_80']} | nothing/one {c2['nothing_differs_over_one_variable_pct']}% single {r['single_step_arm_only_over_one_variable_pct']}% "
          f"| held-out {r['held_out']} | edge {r['at_box_edge_pct']}% std {r['std_per_coordinate']}")


if sys.argv[1] == "gate":
    ok_b = False
    for n in ("bound6", "bound8"):
        r = L(n); line(n, r); c2 = r["criterion2"]
        p = min(c1(r).values()) >= 0.95 and c2["distinct"] and c2["n_at_80"] >= 4
        print("  bound passes:", p); ok_b |= p
    r = L("k8_s0"); line("k8_s0", r)
    read = all(c1(r)[v] >= triv[v] + 0.10 for v in V); arm = r["criterion2"]["nothing_differs_over_one_variable_pct"] < 20
    print("trivial", triv, "| reading passes:", read, "| arm passes:", arm, "| GATE:", ok_b and read and arm)
elif sys.argv[1] == "ladder":
    Ks = [4, 6, 8, 12, 16]; loss = {K: L(f"k{K}_s0")["held_out"]["loss"] for K in Ks}
    best = min(loss.values()); ks = min(K for K in Ks if loss[K] <= best + 0.02)
    for K in Ks:
        line(f"k{K}_s0", L(f"k{K}_s0"))
    print("held-out loss", loss, "K* =", ks)
else:
    K = int(sys.argv[2]); R = [L(f"k{K}_s{s}") for s in range(3)]
    for s, r in enumerate(R):
        line(f"k{K}_s{s}", r)
    m1 = {v: round(float(np.mean([c1(r)[v] for r in R])), 4) for v in V}
    sh = {v: round(float(np.mean([r["criterion2"]["share_in_best"][v] for r in R])), 1) for v in V}
    arm = round(float(np.mean([r["criterion2"]["nothing_differs_over_one_variable_pct"] for r in R])), 1)
    print("mean AUROC", m1, "| c1:", min(m1.values()) >= 0.95, "| distinct seeds", sum(r["criterion2"]["distinct"] for r in R),
          "| mean share", sh, "c2b:", sum(x >= 80 for x in sh.values()) >= 4, "| nothing/one", arm, "c2c:", arm < 10)
