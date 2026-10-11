"""Card 098.7: run the files level on workers 8-9 from a saved memory and keep its files (then where_lost.py).
  bin/prun python tools/card098.7/track_keep.py tokens state_tokens_0to7.pkl"""
import sys
from collections import defaultdict, Counter
import numpy as np
sys.argv = ["crafter.py", "track", "--mode", sys.argv[1], "--load", sys.argv[2], "--workers", "8,9"]
sys.path.insert(0, "tools/card098.5")
sys.path.insert(0, "tools/card098.7")
import crafter as CR
from assign import linear_sum_assignment
import __main__
__main__.euclid = CR.euclid
CR.MODE = sys.argv[3] if False else CR.MODE
out, FS = CR.track([8, 9])
import pickle
keep = {w: {"files": out[w]["files"], "agent": out[w]["agent"]} for w in out}
pickle.dump(keep, open("runs/098.7/kept_" + sys.argv[3] + ".pkl", "wb"))
print("saved", {w: len(out[w]["files"]) for w in out}, flush=True)
NAMES = {14: "cow", 15: "zombie", 16: "skeleton", 17: "arrow"}
C = Counter()
for w in (8, 9):
    z = np.load(f"runs/098.5/play_w{w}.npz")
    rec = out[w]
    ep, light = z["ep"], z["light"]
    gt = defaultdict(list)
    for t, i, k, dx, dy, fx, fy in z["obj"]:
        if k in NAMES:
            gt[int(t)].append(((int(ep[t]), int(i)), int(k), np.array([dx, dy], np.float64)))
    co = defaultdict(lambda: defaultdict(int))
    on = {}
    for t, seen in enumerate(rec["files"]):
        for pid, p in seen:
            rel = p - rec["agent"][t]
            for g, k, r in gt.get(t, []):
                if np.abs(rel - r).max() < 0.5:
                    co[int(ep[t])][(g, pid)] += 1
                    on[(t, g)] = pid
    best = {}
    for e, pairs in co.items():
        G = sorted({g for g, p in pairs}); Pd = sorted({p for g, p in pairs})
        gi, pi = {g: i for i, g in enumerate(G)}, {p: i for i, p in enumerate(Pd)}
        M = np.zeros((len(G), len(Pd)))
        for (g, p), c in pairs.items():
            M[gi[g], pi[p]] = c
        r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] > 0:
                best[G[i]] = Pd[j]
    for t in gt:
        tag = "night" if light[t] < 0.3 else "day"
        for g, k, r in gt[t]:
            f = on.get((t, g))
            cat = "no file" if f is None else ("its file" if best.get(g) == f else "another file")
            C[(tag, NAMES[k], cat)] += 1
            C[(tag, "all", cat)] += 1
for tag in ("day", "night"):
    for k in ("all", "cow", "zombie", "skeleton", "arrow"):
        n = sum(C[(tag, k, c)] for c in ("its file", "another file", "no file"))
        if n:
            print(tag, k, n, {c: round(C[(tag, k, c)] / n, 3) for c in ("its file", "another file", "no file")})
