"""Card 098.7, section 8: The records' share for mover and static tokens, and how many records it rests on, at the identity radius and at the
chosen width (memory after workers 0-7 of card 098.7; tokens sampled from worker 8 with the evaluator's classes)."""
import sys, pickle
import numpy as np
sys.argv = ["crafter.py", "track"]
sys.path.insert(0, "tools/card098.5"); sys.path.insert(0, "tools/card098.7")
import crafter as CR, __main__
__main__.euclid = CR.euclid
FS = pickle.loads(open("runs/098.6/state_tokens_0to7.pkl", "rb").read())
z = np.load("runs/098.5/play_w8.npz")
V = np.load("runs/098.5/vec_w8.npy").astype(np.float32)
sem = z["sem"].reshape(len(z["act"]), 63); light = z["light"]
rng = np.random.default_rng(0)
t = rng.choice(len(V), 4000, replace=False)
v, c, l = V[t].reshape(-1, V.shape[-1]), sem[t].reshape(-1), np.repeat(light[t], 63)
mover = np.isin(c, [14, 15, 16, 17]); static = (c >= 0) & ((c <= 12) | (c == 18))
R = FS.Rt
FS.widths, FS.rng_w, FS.next_fit = [], np.random.default_rng(11), 0
for h in [FS.radius * f for f in (1.0, 0.67, 0.44, 0.3, 0.2)]:
    FS.h = h
    for name, sel in (("movers day", mover & (l >= 0.3)), ("movers night", mover & (l < 0.3)), ("static day", static & (l >= 0.3))):
        idx = np.flatnonzero(sel)
        idx = idx[rng.choice(len(idx), min(300, len(idx)), replace=False)]
        pi = FS.individual(v[idx])
        n = FS.kern(v[idx], R.V[:R.n]).sum(1)
        print(f"width {h:.3f} {name:13s} share median {np.median(pi):.3f}, at or above 0.95: {np.mean(pi >= 0.95):.2f}, "
              f"at or above 0.05: {np.mean(pi >= 0.05):.2f}; records under the kernel (weighted) median {np.median(n):.1f}")
