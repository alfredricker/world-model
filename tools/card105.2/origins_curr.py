"""Card 105.2: card 103.4's placing (tau = 300, as fitted by card 103.5 on workers 0-1 of random play) on the crafting
curriculum's worlds (card 102.1), so that their tiles have world places. The placing's parameters are not refitted.

  bin/prun python tools/card105.2/origins_curr.py W      → runs/105.2/org_curr_w<W>.npy
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
W = int(sys.argv[1])
sys.argv = ["origins.py", "place", str(W)]
spec = importlib.util.spec_from_file_location("origins1035", ROOT / "tools" / "card103.5" / "origins.py")
OR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OR)
PL, TA = OR.PL, OR.TA


def load_world(w):
    z = np.load(ROOT / "runs" / "102.1" / f"curr_w{w}.npz")
    d = {k: z[k] for k in ("act", "done", "pos", "sleep", "light", "ep")}
    d["obs"] = z["obs"][:, :PL.HW, :PL.WW]
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    d["seg"] = np.cumsum(brk)
    d["seg"][d["sleep"]] = -1
    return d


def main():
    t0 = time.monotonic()
    z = np.load(ROOT / "runs" / "103.5" / "placing_tau300.npz")
    P = {"s2": float(z["s2"]), "eps": float(z["eps"]), "pi": z["pi"]}
    d = load_world(W)
    org, sh, _, _, _ = PL.place(d, z["pr"].copy(), P, learn=False, cand=torch.as_tensor(z["cand"], device=PL.dev))
    np.save(ROOT / "runs" / "105.2" / f"org_curr_w{W}.npy", org)
    print(f"world {W}: evaluator only, placements {TA.accuracy(d, sh)}; {time.monotonic() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
