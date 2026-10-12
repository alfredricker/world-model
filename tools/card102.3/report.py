"""Card 102.3: the criteria from the runs' outputs → experiments/102.3-short-term-memory/results.json.

Paired bootstrap of the return log-likelihood over segments (an episode's awake run), learned against each baseline.

  bin/prun python tools/card102.3/report.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card102.3"))
import belief as B                                     # noqa: E402

RUNS = B.RUNS
OUT = ROOT / "experiments" / "102.3-short-term-memory" / "results.json"


def returns_order(d):
    """Every return event in the order run() appends them: (step, true id), with the step's segment."""
    seg, o = d["seg"], d["obj"]
    at = {}
    for t, i, k, dx, dy in o:
        at.setdefault(int(t), []).append(int(i))
    out, last_in = [], {}
    for t in range(len(seg)):
        if seg[t] < 0 or t == 0 or seg[t - 1] != seg[t]:
            last_in = {}
            if seg[t] < 0:
                continue
        for tid in at.get(t, []):
            if tid in last_in and last_in[tid] < t - 1 and t - 1 - last_in[tid] <= B.HORIZON:
                out.append((t, tid, int(seg[t])))
        for tid in at.get(t, []):
            last_in[tid] = t
    return out


def bootstrap(world, d, a, b, n=2000):
    order = returns_order(d)
    E = {f: [e for e in np.load(RUNS / f"events_{world}_{f}.npy", allow_pickle=True) if e["kind"] != -1] for f in (a, b)}
    assert len(E[a]) == len(order) == len(E[b]), (len(E[a]), len(order), len(E[b]))
    day_only = world == "crafter"
    rows = [(s, ea["log-likelihood"] - eb["log-likelihood"]) for (t, tid, s), ea, eb in zip(order, E[a], E[b])
            if ea["kind"] in d["movers"] and (ea["day"] or not day_only)]
    segs = np.array([r[0] for r in rows])
    diff = np.array([r[1] for r in rows])
    u, inv = np.unique(segs, return_inverse=True)
    S, N = np.bincount(inv, diff), np.bincount(inv)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(u), len(u))
        bs.append(S[k].sum() / N[k].sum())
    bs = np.array(bs)
    return {"difference per return": round(float(diff.mean()), 4), "95% interval": [round(float(np.quantile(bs, q)), 4)
                                                                                     for q in (0.025, 0.975)],
            "p (difference <= 0)": round(float((bs <= 0).mean()), 4), "segments": int(len(u)), "returns": int(len(diff))}


def main():
    rep = {}
    for world in ("world3", "crafter"):
        r = json.loads((RUNS / f"belief_{world}.json").read_text())
        k = json.loads((RUNS / f"belief_{world}_K.json").read_text())
        d = B.load(world, "test")
        forms = {f: {a: v for a, v in r[f].items()} for f in ("learned", "still", "rule098", "truth")}
        boot = {f"learned - {b}": bootstrap(world, d, "learned", b) for b in ("still", "rule098")}
        Kc = {K: {"log-likelihood per return": k[f"learned, K = {K}"]["log-likelihood per return (all)"],
                  "identity kept": k[f"learned, K = {K}"]["identity kept (all returns)"],
                  "belief seconds per step": k[f"learned, K = {K}"]["belief seconds per step"]} for K in (1, 2, 4, 8, 16)}
        full = r["learned"]["log-likelihood per return (all)"]
        chosen = min(K for K in Kc if full - Kc[K]["log-likelihood per return"] <= 0.01)
        L = forms["learned"]
        c1 = {"inside the 90% region (bar 0.85-0.95)": L["inside the 90% region (returns with a file)"],
              "bootstrap": boot}
        c1["holds"] = bool(0.85 <= c1["inside the 90% region (bar 0.85-0.95)"] <= 0.95 and
                           all(v["p (difference <= 0)"] < 0.05 for v in boot.values()))
        bar2 = 0.80 if world == "crafter" else 0.90
        idk = L["identity kept (all returns)"]
        n = L["returns of movers within 50 steps"]
        c2 = {"identity kept": idk, "bar": bar2, "binomial 95% interval": [round(idk - 1.96 * (idk * (1 - idk) / n) ** 0.5, 4),
                                                                          round(idk + 1.96 * (idk * (1 - idk) / n) ** 0.5, 4)],
              "098.6's rule": forms["rule098"]["identity kept (all returns)"], "upper bound": forms["truth"]["identity kept (all returns)"]}
        c2["holds"] = bool(idk >= bar2)
        c3 = {"map places right when back in view": L["map places right when back in view"],
              "map places back in view": L["map places back in view"], "chosen K": chosen,
              "belief seconds per step at the chosen K": Kc[chosen]["belief seconds per step"]}
        c3["holds"] = bool((L["map places right when back in view"] or 0) >= 0.99 and Kc[chosen]["belief seconds per step"] <= 0.0012)
        rep[world] = {"forms": forms, "K curve (learned)": Kc, "criterion 1": c1, "criterion 2": c2, "criterion 3": c3,
                      "learned per code (own tracks)": r["learned per code (own tracks)"],
                      "learned per code (truth tracks)": r["learned per code (truth tracks)"], "ground codes": r["ground codes"]}
        print(world, json.dumps({c: rep[world][c]["holds"] for c in ("criterion 1", "criterion 2", "criterion 3")}),
              json.dumps(boot), "chosen K", chosen)
    OUT.write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
