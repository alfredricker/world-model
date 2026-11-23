"""check before card 160: which objects have an inexact box, by whether motion could ever correct it, and by size.  dev3.py <by>"""
import sys, torch, numpy as np, importlib.util, collections
_s = importlib.util.spec_from_file_location("relation159", "tools/card159/relation.py"); Re = importlib.util.module_from_spec(_s); _s.loader.exec_module(Re)
T = Re.T; by = sys.argv[1]; X = torch.load("runs/159/exp_%s_0.pt" % by, weights_only=False); rows = Re.own_rows(X)
moved_before = set(); A = collections.Counter(); B = collections.Counter(); Sz = collections.defaultdict(list); side = collections.Counter()
order = sorted(range(len(rows)), key=lambda j: rows[j]["e"])
for j in order:
    r = rows[j]
    if r["obj"] < 0: continue
    ep = X["tr"][r["e"]]; o = ep["objs"][r["obj"]]; s = o["size"]; seen = next(q for q in X["seen"][r["e"]] if q["p"] == r["p"]); q = X["mem"].info[r["p"]]
    v = np.nonzero(r["valid"])[0]
    if not len(v): continue
    t = v[0]; y, x = ep["pos"][t, r["obj"]]; d = seen["box"][t] - np.array([y, x, y + s, x + s]); exact = not d.any()
    moves = bool(np.abs(np.diff(ep["pos"][:, r["obj"]], axis=0)).sum()); uid = o["uid"]
    origin = "still group" if q.get("still") else "moved group"
    can = "moves in this episode" if moves else ("moved in an earlier episode" if uid >= 0 and uid in moved_before else ("fixed" if o["kind"] == 1 else "movable, never moved so far"))
    A[origin, can] += 1; B[origin, can] += exact
    if q.get("still"):
        Sz[s].append((exact, seen["n"][t] / o["mask"].sum(), np.abs(d).max()))
        for k, nm in enumerate(("top", "left", "bottom", "right")):
            side["short" if (d[k] > 0 if k < 2 else d[k] < 0) else "over" if d[k] != 0 else "exact"] += 1
    if moves and uid >= 0: moved_before.add(uid)
for k in sorted(A): print(by, "%-12s %-30s sequences %4d, box exact at first step %5.1f%%" % (k[0], k[1], A[k], 100 * B[k] / A[k]))
for s in sorted(Sz): print("   still groups, object width %2d: n %3d, exact %5.1f%%, pixels covered (median) %.2f, worst side off by (median) %d" % (s, len(Sz[s]), 100 * np.mean([q[0] for q in Sz[s]]), np.median([q[1] for q in Sz[s]]), np.median([q[2] for q in Sz[s]])))
print("   still groups, sides:", dict(side))
