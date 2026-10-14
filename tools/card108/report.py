"""Card 108's results: criteria 1-3 per form, returns paired against counted motion (bootstrap over segments and
McNemar's exact test), the gate's check, and criterion 3 against card 105.10's models.

  bin/prun python tools/card108/report.py      → experiments/108-learned-belief/results.json
"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "108"
FORMS = ["model_s1021", "model_s7", "modelq_s1021", "counted", "stays", "oracle"]
C3 = {"moves": lambda r: r["2 move steps: the agent's next place right, all"],
      "blocked moves": lambda r: r["2 move steps: the agent's next place right, blocked"],
      "makes that made a tool": lambda r: r["1 makes that made a tool: whole scored next state right"],
      "do's changes, saplings left out": lambda r: r["do"]["changed scored tokens right, saplings left out (105.8)"],
      "place, whole state": lambda r: r["place"]["whole scored next state right"]}


def mcnemar(b, c):
    """Exact two-sided p for b discordant pairs one way and c the other."""
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(ev, base):
    key = lambda e: (e["seg"], e["t"], e["id"])
    b = {key(e): e["kept"] for e in base}
    rows = [(e["seg"], float(e["kept"]) - float(b[key(e)])) for e in ev if key(e) in b]
    segs = sorted({s for s, _ in rows})
    by = {s: [] for s in segs}
    for s, d in rows:
        by[s].append(d)
    sums = np.array([sum(by[s]) for s in segs])
    cnt = np.array([len(by[s]) for s in segs])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(segs), (4000, len(segs)))
    boot = sums[idx].sum(1) / cnt[idx].sum(1)
    d = np.array([x for _, x in rows])
    return {"returns paired": len(rows), "difference in points": round(100 * float(d.mean()), 2),
            "95% interval (bootstrap over segments)": [round(100 * float(np.quantile(boot, q)), 2) for q in (0.025, 0.975)],
            "kept only here / only by counted": [int((d > 0).sum()), int((d < 0).sum())],
            "McNemar p": round(mcnemar(int((d > 0).sum()), int((d < 0).sum())), 5)}


def main():
    out = {"criterion 3 (k-step model against 105.10, same seed)": {}}
    for s in (1021, 7):
        new = json.loads((RUNS / f"model_s{s}.json").read_text())
        old = json.loads((ROOT / "runs" / "105.10" / f"tokens_Rb_s{s}.json").read_text())
        out["criterion 3 (k-step model against 105.10, same seed)"][f"seed {s}"] = {
            k: {"105.10": f(old), "card 108": f(new), "change in points": round(100 * (f(new) - f(old)), 2),
                "within 2 points": abs(f(new) - f(old)) <= 0.02} for k, f in C3.items()}
    for s in (1021, 7):
        p = RUNS / f"gate_s{s}.json"
        if p.exists():
            out[f"gate check, seed {s}"] = json.loads(p.read_text())
    base = np.load(RUNS / "events_counted.npy", allow_pickle=True)
    for f in FORMS:
        p = RUNS / f"belief_{f}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        r.pop("codes: P(leaves its place)", None)
        if f != "counted":
            r["1 paired against counted motion"] = paired(np.load(RUNS / f"events_{f}.npy", allow_pickle=True), base)
        out[f"belief: {f}"] = r
    (ROOT / "experiments" / "108-learned-belief" / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    for f in FORMS:
        r = out.get(f"belief: {f}")
        if r:
            pr = r.get("1 paired against counted motion", {})
            print(f"{f:14s} returns {r['1 identity kept on returns']}  IDF1 {r['2 IDF1']}  "
                  f"(098.6 form {r['IDF1, arrows as objects (098.6)']})  vs counted {pr.get('difference in points')} "
                  f"{pr.get('95% interval (bootstrap over segments)')} p {pr.get('McNemar p')}")


if __name__ == "__main__":
    main()
