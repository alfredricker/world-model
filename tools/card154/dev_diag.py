import sys, time, json, numpy as np
sys.path.insert(0, "tools/card154")
import feedback as Fb
from scipy import ndimage
W = Fb.W
seed, by, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
world = W.make_world(seed, by); eps = Fb.episodes(world, n, seed * 100 + 10)
mem = Fb.Memory(); movedk = {}; miss = []; big = []
orig = Fb.run_episode
for e, ep in enumerate(eps):
    L, rec, log = Fb.run_episode(ep, e, mem, seed=e)
    rows, sp = Fb.score_frame(L[0], rec, mem, ep["masks"][0], Fb.things_of(ep["objs"], e))
    keys = [q["key"] for q in mem.info[:len(mem) - log["stored"]]]
    for r in rows:
        if r["key"] in movedk and r["key"] != "hand" and not r["found"]:
            miss.append((e, r["key"], "pattern" if r["key"] in keys else "no pattern", [q["iou"] for q in mem.info if q["key"] == r["key"]], r["claimed"], r["iou"]))
    # what the last frame's labels look like for things that moved
    bgl = np.bincount(L[-1].ravel()).argmax()
    for i in [1] + [3 + j for j in range(len(ep["objs"]))]:
        pos = ep["hand"] if i == 1 else ep["pos"][:, i - 3]
        st = int((np.abs(np.diff(pos.astype(int), axis=0)).sum(1) > 0).sum())
        if st >= Fb.N_MIN:
            key = Fb.key_of(ep, e, i); movedk[key] = 1
            if i != 1 and key not in [q["key"] for q in mem.info]:
                m = ep["masks"][-1] == i
                if m.any():
                    k = np.bincount(L[-1][m]).argmax(); g = L[-1] == k
                    comp = {int(j): int((ep["masks"][-1][g] == j).sum()) for j in np.unique(ep["masks"][-1][g])}
                    ys, xs = np.nonzero(g); nl = ndimage.label(g)[1]
                    big.append((e, key, st, "label", int(k), "bg" if k == bgl else "", "pixels by true id", comp, "box", (int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)), "pieces", nl, "true size", int(m.sum())))
print("missed at frame 0:"); [print("  ", m) for m in miss]
print("moved, no pattern:"); [print("  ", b) for b in big]
