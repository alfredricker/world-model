"""Card 104.1, a check of the pipeline: 64 frame pairs whose second frame is the first with its content moved exactly
one tile (7 px) to the right, or not moved; Crafter frames (worker 10) against random noise. Prints each model's median
flow and the share of points within 1.5 px of the truth.

  bin/prun python tools/card104.1/synthetic.py
"""
import sys, numpy as np, torch
sys.argv = ["flow.py"]; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); import flow as FL
d = FL.source("play", 10)
rng = np.random.default_rng(0)
t = d["steps"][:64]
for name, base in (("Crafter frames", d["obs"][t]), ("random noise", rng.integers(0, 256, (64, 49, 63, 3), dtype=np.uint8))):
    for dx in (7, 0):
        a = base.copy(); b = np.roll(base, dx, axis=2)          # content moves right by dx: flow (+dx, 0)
        inter = np.empty((128,) + a.shape[1:], np.uint8); inter[0::2] = a; inter[1::2] = b
        idx = np.arange(0, 128, 2)
        for mname, M in (("raft", FL.Raft), ("cotracker", FL.CoTracker)):
            m = M()
            if mname == "raft":
                F, Mk = m({"obs": inter}, idx, "x4")
            else:
                # two-frame clips: start = t, so the window is (t, t+1)
                st = np.arange(128) // 2 * 2
                F, Mk = m({"obs": inter, "start": st}, idx[:16], "x4")
            F = F.astype(np.float32)
            med = np.array([np.median(F[i][Mk[i]], 0) for i in range(len(F))])
            near = (np.linalg.norm(F - np.array([dx, 0]), axis=-1) < 1.5)
            print(f"{name}, content moved {dx} px: {mname} median flow {med.mean(0).round(2)}, share of points within 1.5 px of the truth {near[Mk].mean():.3f}", flush=True)
            del m; torch.cuda.empty_cache()
