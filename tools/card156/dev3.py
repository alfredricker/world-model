"""dev: the agent's own experience.  dev3.py <seed> <by> <n>"""
import sys, time, numpy as np, torch
sys.path.insert(0, "tools/card156")
import agent as Ag
Le, Ap, Fb, W = Ag.Le, Ag.Ap, Ag.Fb, Ag.W
seed, by, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
world = W.make_world(seed, by); rng = np.random.default_rng(seed * 100 + 10)
mem = Fb.Memory(); net = Ap.Pairs(); t0 = time.time(); tot = {}
for e in range(n):
    ep, S, lg = Ag.live_episode(world, e, rng, mem, net, seed=e)
    for k, v in lg.items(): tot[k] = tot.get(k, 0) + v
    if e % 10 == 9:
        loss = net.fit(300); print(e + 1, "episodes", round(time.time() - t0), "s", len(mem), "patterns", tot, "loss", round(loss, 3), flush=True); tot = {}
pc = lambda v: round(100 * float(np.mean(v)), 1) if len(v) else None
for source, sizes in (("fresh", W.SIZES + W.NEW_SIZES), ("pool", W.SIZES)):
    te = Fb.episodes(world, 60, seed * 100 + (30 if source == "fresh" else 40), source, sizes); rows = []; sp = []
    for j, ep in enumerate(te):
        lab, rec, _, _ = Ap.first_sight(ep["frames"][0], mem, net, 1000 + j)
        r, s = Fb.score_frame(lab, rec, mem, ep["masks"][0], Fb.things_of(ep["objs"], "x%d" % j)); sp.append(s)
        for q, (i, key, full) in zip(r, Fb.things_of(ep["objs"], "x%d" % j)):
            q["kind"] = -1 if i == 1 else ep["objs"][i - 3]["kind"]; q["size"] = 7 if i == 1 else ep["objs"][i - 3]["size"]; rows.append(q)
    print(source, "found: hand", pc([q["found"] for q in rows if q["kind"] == -1]), *["%s %s (iou %.2f, n %d)" % (nm, pc([q["found"] for q in rows if q["kind"] == k]), np.mean([q["iou"] for q in rows if q["kind"] == k]), sum(q["kind"] == k for q in rows)) for k, nm in enumerate(("takeable", "fixed", "sticky"))],
          "| new widths", pc([q["found"] for q in rows if q["size"] in W.NEW_SIZES and q["kind"] != 1]), "| labels on the ground per scene", round(float(np.mean(sp)), 2))
