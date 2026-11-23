"""check before card 160: how good a still pattern is by the episode in which it was stored.  dev5.py <by>"""
import sys, torch, numpy as np, importlib.util
_s = importlib.util.spec_from_file_location("relation159", "tools/card159/relation.py"); Re = importlib.util.module_from_spec(_s); _s.loader.exec_module(Re)
by = sys.argv[1]; X = torch.load("runs/159/exp_%s_0.pt" % by, weights_only=False); info = X["mem"].info; rows = Re.own_rows(X)
use = np.zeros(len(info), int)
for r in rows: use[r["p"]] += r["obj"] >= 0
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 70), (70, 100)):
    q = [(i["iou"], use[p]) for p, i in enumerate(info) if i.get("still") and i.get("tid", 0) >= 3 and lo <= i["episode"] < hi]
    m = [(i["iou"], use[p]) for p, i in enumerate(info) if not i.get("still") and i.get("tid", 0) >= 3 and lo <= i["episode"] < hi]
    print(by, "stored in episodes %2d-%2d: still patterns on objects %3d, overlap with the object (mean) %.2f, at least 0.9 in %3.0f%%, sequences they were used in %3d | moved patterns %3d, overlap %.2f, at least 0.9 in %3.0f%%" % (
        lo, hi - 1, len(q), np.mean([a for a, _ in q]) if q else 0, 100 * np.mean([a >= 0.9 for a, _ in q]) if q else 0, sum(b for _, b in q), len(m), np.mean([a for a, _ in m]) if m else 0, 100 * np.mean([a >= 0.9 for a, _ in m]) if m else 0))
