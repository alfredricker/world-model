"""development check: where the own contact token misses a contact the simulator reports"""
import sys, torch, numpy as np, collections
import importlib.util
_spec = importlib.util.spec_from_file_location("classes157", "tools/card157/classes.py"); C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
W, T, HAND = C.W, C.T, C.HAND
for by in ("shape", "colour"):
    X = torch.load("runs/157/exp_%s_0.pt" % by, weights_only=False); tr = X["tr"]; why = collections.Counter(); came = collections.Counter()
    for e, ep in enumerate(tr):
        slot_of = {int(p): i for i, p in enumerate(X["own0"][e]) if p >= 0}
        for s in X["seen"][e]:
            q = X["mem"].info[s["p"]]; i = slot_of.get(s["p"], q["tid"] - 3 if q["episode"] == e and q.get("tid", 0) >= 3 else -1)
            if i < 0: continue
            o = ep["objs"][i]
            for t in range(s["t0"], T):
                if s["n"][t] < 4 or s["code"][t] < 0 or not W.touching(ep["hand"][t], ep["pos"][t, i], o["size"]): continue
                if C.touching(ep["hand"][t], s["box"][t]): why["caught"] += 1; continue
                y, x = ep["pos"][t, i]; tb = np.array([y, x, y + o["size"], x + o["size"]]); d = s["box"][t] - tb
                k = "box equals the true box" if not d.any() else "box inside the true box (short by %d)" % min(3, np.abs(d).max()) if (d[:2] >= 0).all() and (d[2:] <= 0).all() else "box reaches outside the true box"
                why[k] += 1; came[k] += int(ep["held"][t + 1] == i and (ep["hand"][t + 1] != ep["hand"][t]).any())
    print(by, dict(why), "| of the missed, the object came along:", dict(came))
