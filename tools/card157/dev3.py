"""check: contacts by the true behaviour of the object, own token against the simulator's.  dev3.py <by> <seed>..."""
import sys, torch, numpy as np
import importlib.util
_spec = importlib.util.spec_from_file_location("classes157", "tools/card157/classes.py"); C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
W, T = C.W, C.T
by = sys.argv[1]
for seed in sys.argv[2:]:
    X = torch.load("runs/157/exp_%s_%s.pt" % (by, seed), weights_only=False); n = np.zeros((3, 4), int); objs = [set(), set(), set()], [set(), set(), set()]
    for e, ep in enumerate(X["tr"]):
        slot_of = {int(p): i for i, p in enumerate(X["own0"][e]) if p >= 0}
        for s in X["seen"][e]:
            q = X["mem"].info[s["p"]]; i = slot_of.get(s["p"], q["tid"] - 3 if q["episode"] == e and q.get("tid", 0) >= 3 else -1)
            if i < 0: continue
            o = ep["objs"][i]; k = o["kind"]
            for t in range(s["t0"], T):
                if s["n"][t] < 4 or s["code"][t] < 0: continue
                a = W.touching(ep["hand"][t], ep["pos"][t, i], o["size"]); b = C.touching(ep["hand"][t], s["box"][t])
                n[k, 0] += a and b; n[k, 1] += a and not b; n[k, 2] += b and not a; n[k, 3] += 1
                if a: objs[0][k].add((e, i))
                if b: objs[1][k].add((e, i))
    for k, nm in enumerate(("takeable", "fixed", "sticky")):
        print(by, seed, "%-8s contact in both %5d, simulator only %4d, own only %4d | objects ever in contact: simulator %3d, own %3d" % (nm, n[k, 0], n[k, 1], n[k, 2], len(objs[0][k]), len(objs[1][k])))
