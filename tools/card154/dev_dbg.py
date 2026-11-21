"""which never-stored things are recognised as a stored pattern: dev_dbg.py <seed> <by>"""
import sys, numpy as np, torch, collections
sys.path.insert(0, "tools/card154")
import feedback as Fb
W = Fb.W
seed, by = int(sys.argv[1]), sys.argv[2]
world = W.make_world(seed, by); tr = Fb.episodes(world, 100, seed * 100 + 10)
mem = Fb.Memory(); steps = collections.Counter()
for e, ep in enumerate(tr):
    Fb.run_episode(ep, e, mem, seed=e)
    for i in [1] + [3 + j for j in range(len(ep["objs"]))]:
        pos = ep["hand"] if i == 1 else ep["pos"][:, i - 3]
        steps[Fb.key_of(ep, e, i)] += int((np.abs(np.diff(pos.astype(int), axis=0)).sum(1) > 0).sum())
known = set()
for e, ep in enumerate(tr):
    for i in [1] + [3 + j for j in range(len(ep["objs"]))]:
        pos = ep["hand"] if i == 1 else ep["pos"][:, i - 3]
        if (np.abs(np.diff(pos.astype(int), axis=0)).sum(1) > 0).sum() >= Fb.N_MIN: known.add(Fb.key_of(ep, e, i))
te = Fb.episodes(world, 100, seed * 100 + 30, "pool"); cnt = collections.Counter()
for j, ep in enumerate(te):
    lab, rec, _, _ = Fb.first_sight(ep["frames"][0], mem, True, 1000 + j); mask = ep["masks"][0]
    for i, key, full in Fb.things_of(ep["objs"], "a%d" % j):
        if key in known: continue
        m = mask == i
        for r in rec:
            if (r["pix"] & m).sum() >= 0.25 * m.sum():
                q = mem.info[r["m"]]; cnt[(key, "moved in training, steps: %d" % steps[key], "pattern of", q["key"], "pattern iou", q["iou"], "pattern pixels", q["n"], "pattern episode", q["episode"])] += 1
for k, v in cnt.items(): print(v, k)
print("patterns whose key is not known:", [(q["key"], q["iou"], q["n"], steps[q["key"]]) for q in mem.info if q["key"] not in known])
# the same count on both tests, split by whether the pattern was stored from the thing itself
def split(frames, masks, things, s0):
    n = own = other = 0
    for j, (fr, mk, th) in enumerate(zip(frames, masks, things)):
        lab, rec, _, _ = Fb.first_sight(fr, mem, True, s0 + j)
        for i, key, full in th:
            m = mk == i
            if key in known or m.sum() < 8: continue
            n += 1
            for r in rec:
                if (r["pix"] & m).sum() >= 0.25 * m.sum():
                    if mem.info[r["m"]]["key"] == key: own += 1
                    else: other += 1
                    break
    return dict(n=n, by_its_own_pattern=own, by_another_pattern=other)
print("first sight", split([ep["frames"][0] for ep in te], [ep["masks"][0] for ep in te], [Fb.things_of(ep["objs"], "a%d" % e) for e, ep in enumerate(te)], 1000))
sc = W.scenes(world, 300, seed * 100 + 50)
print("covered", split([s["frame"] for s in sc], [s["mask"].astype(int) for s in sc], [Fb.things_of(s["objs"], "b%d" % e) for e, s in enumerate(sc)], 2000))
