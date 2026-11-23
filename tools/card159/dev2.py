"""check: how exact an object's box is, by how the model came to hold the object.  dev2.py <by>"""
import sys, torch, numpy as np, importlib.util, collections
_s = importlib.util.spec_from_file_location("relation159", "tools/card159/relation.py"); Re = importlib.util.module_from_spec(_s); _s.loader.exec_module(Re)
T = Re.T; by = sys.argv[1]; X = torch.load("runs/159/exp_%s_0.pt" % by, weights_only=False); rows = Re.own_rows(X)
tot = collections.Counter(); ex = collections.Counter(); first = collections.Counter(); exf = collections.Counter(); cover = collections.defaultdict(list)
for r in rows:
    if r["obj"] < 0: continue
    ep = X["tr"][r["e"]]; o = ep["objs"][r["obj"]]; s = o["size"]; seen = next(q for q in X["seen"][r["e"]] if q["p"] == r["p"]); q = X["mem"].info[r["p"]]
    kind = ("stored this episode" if q["episode"] == r["e"] else "recognised, stored earlier") + (", as a still group" if q.get("still") else ", after it moved")
    v = np.nonzero(r["valid"])[0]
    for j, t in enumerate(v):
        y, x = ep["pos"][t, r["obj"]]; d = seen["box"][t] - np.array([y, x, y + s, x + s]); tot[kind] += 1; ex[kind] += int(not d.any())
        if j == 0:
            first[kind] += 1; exf[kind] += int(not d.any()); cover[kind].append(seen["n"][t] / o["mask"].sum())
for k in sorted(tot):
    print(by, "%-48s sequences %4d | box exact at its first step %5.1f%%, over all steps %5.1f%% | points seen at first step / object's pixels: median %.2f" % (k, first[k], 100 * exf[k] / first[k], 100 * ex[k] / tot[k], np.median(cover[k])))
