"""Card 098.6, section 5: files confirmed and born, by the evaluator's class, first steps of workers 0-7 (tokens).
  bin/prun python tools/card098.6/by_class.py"""
import sys
import time
import numpy as np
from collections import Counter
sys.argv = ["crafter.py", "track", "--mode", "tokens", "--workers", "0", "--steps", "1000"]
sys.path.insert(0, "tools/card098.6")
import crafter as CR
import files as FL
CR.MODE = "tokens"
NAMES = ["none", "water", "grass", "stone", "path", "sand", "tree", "lava", "coal", "iron", "diamond", "table",
         "furnace", "player", "cow", "zombie", "skeleton", "arrow", "plant"]
z = np.load("runs/098.5/play_w0.npz")
sem = z["sem"].reshape(len(z["act"]), 63)
light = z["light"]
cur = {}
conf, births, tent, byw = Counter(), Counter(), Counter(), Counter()
step0, confirm0, new0 = FL.Files.step, FL.Files._confirm, FL.Files._new
T = {"t": 0, "agent": None}


def cls_at(p):
    t, ag = T["t"], T["agent"]
    rel = np.round(np.asarray(p) - ag).astype(int)
    for i, r in enumerate(CR.REL.astype(int)):
        if r[0] == rel[0] and r[1] == rel[1]:
            return NAMES[sem[t, i]] if sem[t, i] >= 0 else "outside"
    return "?"


def step(self, t, P, V, B, known, region):
    global sem, light
    if t == 0:
        T["w"] = T.get("w", -1) + 1
        zz = np.load(f"runs/098.5/play_w{T['w']}.npz")
        sem, light = zz["sem"].reshape(len(zz["act"]), 63), zz["light"]
    T["t"] = t
    T["agent"] = (P - CR.REL[CR.VIEW]).mean(0); T["P"], T["known"], T["B"], T["V"] = P, np.asarray(known, bool), B, V
    t0 = time.time()
    out = step0(self, t, P, V, B, known, region)
    if False:
        ag = T["agent"]
        cl = [cls_at(p) for p in P]
        pi = self.individual(np.asarray(V, np.float32))
        R = self.Rt
        eps = (R.mu[:R.n].sum() + 0.5) / (R.k[:R.n].sum() + 1)
        g = [x for x, c in zip(pi, cl) if c == "grass"]
        cw = [x for x, c in zip(pi, cl) if c in ("cow", "zombie", "skeleton", "arrow")]
        print("probe w", T["w"], "t", t, "m0 %.3f eps %.4f pi0 %.3f |S| %d" % (self._m0(), eps, self.pi0, len(self.S)),
              "grass pi med/max", np.round([np.median(g), np.max(g)], 3) if g else None,
              "movers pi", np.round(cw, 3) if cw else None, flush=True)
        for x, c, v in zip(pi, cl, V):
            if c in ("cow", "zombie", "skeleton", "arrow"):
                K = self.kern(np.asarray(v, np.float32)[None], R.V[:R.n])[0]
                nb = K > 0
                kk, mm = R.k[:R.n][nb], R.mu[:R.n][nb]
                ev = self._evidence(self._m0(), np.flatnonzero(nb))
                print("   ", c, "pi %.3f" % x, "records", nb.sum(), "steps", kk.sum(), "moves %.1f" % mm.sum(),
                      "moved>0.5:", (mm > 0.5).sum(), "k of unmoved: q50/q90", np.percentile(kk[mm <= 0.5], [50, 90]) if (mm <= 0.5).any() else None,
                      "mean weight %.2f" % K[nb].mean(), "ev q10/q50/q90", np.percentile(ev, [10, 50, 90]).round(1) if nb.any() else None,
                      "dist to nearest kind", None if self.kinds is None else round(float(np.linalg.norm(self.kinds - v, axis=1).min()), 3))
                break
    cur["time"] = cur.get("time", 0) + time.time() - t0
    if False:
        print(t, "files", len(self.files), "confirmed", sum(f["confirmed"] for f in self.files), "kinds",
              0 if self.kinds is None else len(self.kinds), f"{cur['time']:.1f}s", dict(self.stats), flush=True)
    return out


def _confirm(self, f):
    conf[(cls_at(f["pos"]), "night" if light[T["t"]] < 0.3 else "day")] += 1
    byw[(T["w"], cls_at(f["pos"]))] += 1
    return confirm0(self, f)


def _new(self, t, v, pos, vel, known, r0, rec=None):
    i = int(np.argmin(np.abs(T["P"] - pos).sum(1)))
    where = "first seen" if not T["known"][i] else "known"
    births[(cls_at(pos), "pair" if known else "kind", where, "confirmed" if r0 >= FL.CONFIRM else "tentative")] += 1
    return new0(self, t, v, pos, vel, known, r0, rec)


FL.Files.step, FL.Files._confirm, FL.Files._new = step, _confirm, _new
CR.track([0])
print("confirmed by true class", conf.most_common(15))
print("births", births.most_common(20))
print("confirmed grass / cow by worker", [(w, byw[(w, "grass")], byw[(w, "cow")]) for w in range(8)])
