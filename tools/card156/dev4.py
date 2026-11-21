"""dev: the labels on the ground, with and without the prior put back.  dev4.py <seed> <by> <n>  (script experience)"""
import sys, numpy as np, torch
sys.path.insert(0, "tools/card156")
import learn as Le
Ap, Fb, W = Le.Ap, Le.Fb, Le.W
seed, by, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
world = W.make_world(seed, by); tr = Fb.episodes(world, n, seed * 100 + 10); mem = Fb.Memory(); net = Ap.Pairs()
for e, ep in enumerate(tr):
    Le.run_episode(ep, e, mem, net, seed=e)
    if e % 10 == 9: net.fit(300)
print("class shares among all pairs (none, together, apart)", (net.counts / net.counts.sum()).cpu().numpy().round(5))
te = Fb.episodes(world, 30, seed * 100 + 30, "fresh", W.SIZES + W.NEW_SIZES)
for prior, hard, base in ((True, True, False), (True, False, False), (True, False, True)):
    net.prior = prior; net.hard = hard; net.base = base; found = []; sp = []; nlab = []; agg = []
    for j, ep in enumerate(te):
        lab, rec, _, model = Ap.first_sight(ep["frames"][0], mem, net, 1000 + j); mask = ep["masks"][0]; th = Fb.things_of(ep["objs"], "x")
        rows, s = Fb.score_frame(lab, rec, mem, mask, th); sp.append(s); nlab.append(len(np.unique(lab)))
        found += [(r["found"], ep["objs"][i - 3]["kind"]) for r, (i, _, _) in zip(rows, th) if i >= 3]
        A = model.A[:Ap.HW, :Ap.HW]; g = torch.as_tensor((mask == 0).ravel(), device="cuda"); ob = torch.as_tensor((mask >= 3).ravel(), device="cuda")
        same = torch.as_tensor(mask.ravel(), device="cuda"); same = (same[:, None] == same[None]) & ob[:, None]
        agg.append((float(A[g][:, g].mean()), float(A[same].mean()), float(A[ob][:, g].mean())))
    print("prior put back:", prior, "likeliest state:", hard, "baseline:", base, "| found takeable, fixed, sticky", [round(100 * float(np.mean([f for f, k in found if k == b])), 1) for b in range(3)], "| labels per scene", np.mean(nlab), "on the ground", np.mean(sp),
          "| mean A: ground-ground %.3f, inside a thing %.2f, thing-ground %.2f" % tuple(np.mean(agg, 0)))
