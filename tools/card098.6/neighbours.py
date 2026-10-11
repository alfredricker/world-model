"""Card 098.6, section 7: how many of a mover token's neighbours (within radius r) are movers or static, by day and night,
on 3,000 random frames of each of workers 0-1 with the evaluator's classes.  bin/prun python tools/card098.6/neighbours.py"""
import sys
import numpy as np
sys.path.insert(0, "tools/card098.6")
NAMES = ["none", "water", "grass", "stone", "path", "sand", "tree", "lava", "coal", "iron", "diamond", "table",
         "furnace", "player", "cow", "zombie", "skeleton", "arrow", "plant"]
rng = np.random.default_rng(0)
Vs, Cs, Ls = [], [], []
for w in (0, 1):
    z = np.load(f"runs/098.5/play_w{w}.npz")
    V = np.load(f"runs/098.5/vec_w{w}.npy").astype(np.float32)
    sem = z["sem"].reshape(len(z["act"]), 63)
    light = np.repeat(z["light"][:, None], 63, 1)
    t = rng.choice(len(V), 3000, replace=False)
    Vs.append(V[t].reshape(-1, V.shape[-1])); Cs.append(sem[t].reshape(-1)); Ls.append(light[t].reshape(-1))
V, C, L = np.concatenate(Vs), np.concatenate(Cs), np.concatenate(Ls)
ok = C >= 0
V, C, L = V[ok], C[ok], L[ok]
static = (C <= 12) | (C == 18)
mover = np.isin(C, [14, 15, 16, 17])
print("tokens", len(V), "movers", mover.sum())
for night in (False, True):
    sel_m = np.flatnonzero(mover & ((L < 0.3) == night))
    q = sel_m[rng.choice(len(sel_m), min(400, len(sel_m)), replace=False)]
    d = np.sqrt(np.maximum((V[q] ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None] - 2 * V[q] @ V.T, 0))
    for r in (0.771, 0.5, 0.3, 0.2):
        nb = d <= r
        ms = (nb & mover[None]).sum(1)
        ss = (nb & static[None]).sum(1)
        share = ms / np.maximum(ms + ss, 1)
        print("night" if night else "day  ", f"r {r}: movers' neighbours that are movers: median {np.median(share):.3f}, "
              f"q10 {np.percentile(share, 10):.3f}; share of movers with static neighbours outnumbering mover ones "
              f"{np.mean(ss > ms):.3f}; static neighbours median {np.median(ss)}")
