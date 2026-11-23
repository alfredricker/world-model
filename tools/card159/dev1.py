"""development check: how the box around an object's points differs from the true box, and which relative places the
own boxes show at steps where the simulator reports contact.  dev1.py <by>"""
import sys, torch, numpy as np, importlib.util, collections
_s = importlib.util.spec_from_file_location("relation159", "tools/card159/relation.py"); Re = importlib.util.module_from_spec(_s); _s.loader.exec_module(Re)
W, T = Re.W, Re.T; by = sys.argv[1]; KEY = sys.argv[2]; X = torch.load("runs/159/exp_%s_0.pt" % by, weights_only=False); rows = Re.own_rows(X)
err = collections.Counter(); rel = collections.Counter(); kinds = collections.Counter(); n = 0
for r in rows:
    if r["obj"] < 0: continue
    ep = X["tr"][r["e"]]; o = ep["objs"][r["obj"]]; s = o["size"]; seen = next(q for q in X["seen"][r["e"]] if q["p"] == r["p"])
    for t in np.nonzero(r["valid"])[0]:
        y, x = ep["pos"][t, r["obj"]]; d = seen[KEY][t] - np.array([y, x, y + s, x + s]); n += 1
        big = np.abs(d).max(); out = (d[:2] < 0).any() or (d[2:] > 0).any()
        err["exact" if big == 0 else ("outside by %s" % ("1" if max(-d[:2].min(), d[2:].max()) == 1 else "2-3" if max(-d[:2].min(), d[2:].max()) <= 3 else "4+")) if out else "inside, short by %s" % ("1" if big == 1 else "2-3" if big <= 3 else "4+")] += 1
        if W.touching(ep["hand"][t], (y, x), s):
            held = ep["held"][t + 1] == r["obj"]; rel[("held" if held else "free"), Re.name(r["rel"][t])] += 1
            if not Re.TOUCH[r["rel"][t]]:
                kinds[("held" if held else "free"), "first sight pattern" if seen["t0"] == 0 else "stored this episode", "exact" if big == 0 else "outside" if out else "inside"] += 1
print(by, "steps", n, {k: "%.1f%%" % (100 * v / n) for k, v in sorted(err.items())})
for h in ("free", "held"):
    tot = sum(v for (a, _), v in rel.items() if a == h); print("  contact,", h, tot, {k: v for (a, k), v in sorted(rel.items(), key=lambda q: -q[1]) if a == h and v >= 0.02 * tot})
print("  missed by the rule:", dict(kinds))
