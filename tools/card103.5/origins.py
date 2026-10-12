"""Card 103.5: card 103.4's placing (tau = 300, the camera layer learned) for the workers the files level uses: training
play (workers 0-1, where the files' long-term memory is learned) and both test workers (8, 9). The placing's parameters
are fitted on workers 0-1 exactly as in card 103.4 and saved, so each worker is then placed in its own short run.

  bin/prun python tools/card103.5/origins.py fit          → runs/103.5/placing_tau300.npz
  bin/prun python tools/card103.5/origins.py place W      → runs/103.5/org_tau300_w<W>.npy
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else None
sys.argv = ["tau.py", "test", "300"]
sys.path.insert(0, str(ROOT / "tools" / "card103.4"))
import tau as TA                                       # noqa: E402
PL = TA.PL
OUT = ROOT / "runs" / "103.5"


def load_worker(w):
    z = np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")
    d = {k: z[k] for k in ("act", "done", "pos", "sleep", "light", "ep")}
    d["obs"] = z["obs"][:, :PL.HW, :PL.WW]
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    d["seg"] = np.cumsum(brk)
    d["seg"][d["sleep"]] = -1
    return d


def main():
    t0 = time.monotonic()
    OUT.mkdir(exist_ok=True, parents=True)
    if MODE == "fit":
        P, pr, cand, _, _ = TA.passes(PL.load("train"))
        np.savez(OUT / "placing_tau300.npz", s2=P["s2"], eps=P["eps"], pi=P["pi"], pr=pr, cand=cand.cpu().numpy())
    else:
        z = np.load(OUT / "placing_tau300.npz")
        P = {"s2": float(z["s2"]), "eps": float(z["eps"]), "pi": z["pi"]}
        d = load_worker(W)
        org, sh, _, _, _ = PL.place(d, z["pr"].copy(), P, learn=False, cand=torch.as_tensor(z["cand"], device=PL.dev))
        np.save(OUT / f"org_tau300_w{W}.npy", org)
        rep = TA.accuracy(d, sh)
        print(f"worker {W}: evaluator only, placements {rep}")
    print(f"{time.monotonic() - t0:.0f} s")


if __name__ == "__main__":
    main()
