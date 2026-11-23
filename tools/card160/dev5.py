import sys, pickle, time, numpy as np
import grid, rules, energy
np.set_printoptions(legacy='1.25')
from grid import NA, NG, FAR
T = pickle.load(open("runs/160/exp_1.pkl", "rb")); codes = T.codes; t0 = time.time(); M = energy.Energy(T, sys.argv[1], l1=float(sys.argv[2])); print("fit %.0f s nats %.0f" % (time.time() - t0, M.nll), "objective", M.log)
pickle.dump(M, open("runs/160/dev_model_%s.pkl" % sys.argv[1], "wb"))
if "W0" in M.w:
    W0 = M.w["W0"]; nz = np.nonzero(np.abs(W0) > 1e-6)[0]
    def fn(f):
        if f < NA: return rules.ANAME[f]
        if f < M.FI: return "agent at " + (str(tuple(int(v) for v in grid.gid_inv(f - NA))) if f - NA < NG else "far")
        if f < M.FY: return "out of view: %s by %s" % (rules.name(codes, M.items[f - M.FI][0]), rules.ANAME[M.items[f - M.FI][1]])
        return "out of view by " + rules.ANAME[f - M.FY]
    print("shared couplings toward a change (non-zero %d of %d):" % (len(nz), len(W0))); print("  ", [(fn(f), round(float(W0[f]), 2)) for f in nz])
    print("non-zero couplings of a code's own difference: %d of %d" % ((np.abs(M.w["dW"]) > 1e-6).sum(), M.w["dW"].size))
    for nm in ("blue key", "red box", "closed green door", "locked green door", "empty"):
        c = [k for k in range(2, len(codes.triples)) if rules.name(codes, k) == nm][0]; outs = M.cell[c]; d = M.w["dW"][c]
        for o in outs:
            nzf = np.nonzero(np.abs(d[:, o]) > 1e-6)[0]
            if o != c and len(nzf) < 40: print("  %s -> %s: bias %.1f" % (nm, rules.name(codes, o), M.w["b"][c, o]), [(fn(f), round(float(d[f, o]), 2)) for f in nzf])
