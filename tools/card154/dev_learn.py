"""dev: learn patterns from n training episodes and report their quality and the time"""
import sys, time, json, numpy as np
sys.path.insert(0, "tools/card154")
import feedback as Fb
W = Fb.W
seed, by, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
world = W.make_world(seed, by); t0 = time.time()
eps = Fb.episodes(world, n, seed * 100 + 10); print("episodes", round(time.time() - t0, 1), flush=True)
mem = Fb.Memory(); t0 = time.time(); tot = dict(stored=0, known=0, too_big=0); f0 = []; movedk = {}
for e, ep in enumerate(eps):
    L, rec, log = Fb.run_episode(ep, e, mem, seed=e)
    for k in tot: tot[k] += log[k]
    rows, sp = Fb.score_frame(L[0], rec, mem, ep["masks"][0], Fb.things_of(ep["objs"], e))
    for r in rows:
        r["known_truth"] = r["key"] in movedk; r["e"] = e; r["spurious"] = sp; f0.append(r)
    for i in [1] + [3 + j for j in range(len(ep["objs"]))]:
        pos = ep["hand"] if i == 1 else ep["pos"][:, i - 3]
        st = int((np.abs(np.diff(pos.astype(int), axis=0)).sum(1) > 0).sum())
        if st >= Fb.N_MIN: movedk[Fb.key_of(ep, e, i)] = movedk.get(Fb.key_of(ep, e, i), 0) + 1
    if e % 10 == 9: print(e + 1, "episodes", round(time.time() - t0, 1), "s", len(mem), "patterns", tot, flush=True)
iou = np.array([q["iou"] for q in mem.info]); keys = [q["key"] for q in mem.info]
print("patterns", len(mem), "iou quartiles", np.percentile(iou, [5, 25, 50, 75]).round(2), "on ground", sum(k is None for k in keys))
print("things that moved >=3 steps", len(movedk), "with a pattern", sum(k in keys for k in movedk), "with a pattern of iou>=.8", sum(any(q["key"] == k and q["iou"] >= .8 for q in mem.info) for k in movedk))
print("patterns per key (pool, hand)", sorted([(k, keys.count(k)) for k in set(keys) if k and not k.startswith("new")], key=lambda a: -a[1])[:12])
print("Jin, Jout medians", np.median(mem.Jin).round(1), np.median(mem.Jout).round(1))
kn = [r for r in f0 if r["known_truth"]]; late = [r for r in kn if r["e"] >= n // 2]
pc = lambda v: round(100 * float(np.mean(v)), 1) if len(v) else None
print("frame 0, things known by truth (moved before):", len(kn), "found", pc([r["found"] for r in kn]), "itself", pc([r["itself"] for r in kn]), "| second half", len(late), pc([r["found"] for r in late]), pc([r["itself"] for r in late]))
print("  hand only", pc([r["found"] for r in kn if r["key"] == "hand"]), "pool only", pc([r["found"] for r in kn if r["key"] != "hand"]), "n pool", sum(r["key"] != "hand" for r in kn))
print("  not known: claimed", pc([r["claimed"] for r in f0 if not r["known_truth"]]), "spurious labels per frame", np.mean([r["spurious"] for r in f0]).round(3))
