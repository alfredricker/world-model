"""Card 103.4: evidence counted per independent observation. Card 103.1's placing sums a log-likelihood over pixels as
if each were independent; neighbouring pixels of one moved thing are not, so a stepping cow outvotes the action's
prior. The sum is divided by tau, the number of pixels per independent observation. tau is chosen without labels: the
value whose placements leave the agent's own world most self-consistent, measured on training play as the share of
confirmed world content that, seen again from a new view position, is found as it was written.

  bin/prun python tools/card103.4/tau.py select     → runs/103.4/select.json (training: consistency per tau, and,
                                                         for the evaluator only, the placements' accuracy)
  bin/prun python tools/card103.4/tau.py test TAU   → runs/103.4/test_tau<TAU>.json (worker 8, card 103.1's measures)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
TAU = sys.argv[2] if len(sys.argv) > 2 else "1"
sys.argv = ["place.py", "crafter", "--pi", "learned", "--tau", TAU]
sys.path.insert(0, str(ROOT / "tools" / "card103.1"))
import place as PL                                     # noqa: E402

OUT = ROOT / "runs" / "103.4"
NS = PL.NS


def passes(tr, n=2):
    P = {"s2": 1e-3, "eps": 0.05, "pi": np.load(ROOT / "runs" / "103.1" / "pi_crafter_world_learned.npy")}
    cand = None
    for it in range(n):
        pr = np.ones((17, NS * NS))
        org, sh, st, _, _ = PL.place(tr, pr, P, learn=True, cand=cand)
        P["s2"] = float(np.clip(st["in_r2"] / (3 * st["in_r"]), 1e-5, 1e-2))
        P["eps"] = float(np.clip(1 - st["in_r"] / st["n"], 1e-3, 0.5))
        q = pr / pr.sum(1, keepdims=True)
        cand = torch.as_tensor((q >= 1e-3).reshape(17, NS, NS), device=PL.dev)
    return P, pr, cand, sh, st


def accuracy(d, sh):
    seg = d["seg"]
    s = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:]))
    true = 7 * (d["pos"][s + 1] - d["pos"][s]).astype(np.int64)
    right = (sh[s] == true).all(1)
    moved = (true != 0).any(1)
    night = d["light"][s] < 0.3
    return {"right": round(float(right.mean()), 4),
            "moves, still": [round(float(right[moved].mean()), 4), round(float(right[~moved].mean()), 4)],
            "night": round(float(right[night].mean()), 4) if night.any() else None, "steps": int(len(s))}


def main():
    t0 = time.monotonic()
    OUT.mkdir(exist_ok=True, parents=True)
    tr = PL.load("train")
    if MODE == "select":
        P, pr, cand, sh, st = passes(tr)
        rep = {"tau": float(TAU), "consistency (re-seen confirmed content found as written)":
               round(st["re_same"] / max(st["re_n"], 1), 5), "re-seen pixels": int(st["re_n"]),
               "evaluator only: training placements": accuracy(tr, sh), "seconds": round(time.monotonic() - t0)}
        (OUT / f"select_tau{TAU}.json").write_text(json.dumps(rep, indent=1) + "\n")
    else:
        P, pr, cand, _, _ = passes(tr)
        te = PL.load("test")
        org, sh, _, _, _ = PL.place(te, pr.copy(), P, learn=False, cand=cand)
        rep = {"tau": float(TAU), "test, worker 8": accuracy(te, sh), "seconds": round(time.monotonic() - t0)}
        np.save(OUT / f"org_tau{TAU}_test.npy", org)
        (OUT / f"test_tau{TAU}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
