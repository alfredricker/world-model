"""after the test: the couplings of the agent's own move, heading east.  usage: dev8.py <experience>"""
import sys, numpy as np
import grid, rules, evaluate
from grid import OFF8, NA
T = evaluate.load(sys.argv[1]); M = evaluate.model(sys.argv[1], "sparse", T); codes = T.codes; w = M.wa; d = 0; o = 1 * 4 + 0
def fn(f):
    if f < NA: return rules.ANAME[f]
    if f < M.AI: return "%s at %s" % (rules.name(codes, (f - NA) % M.NC), OFF8[(f - NA) // M.NC])
    if f < M.AY: return "out of view: %s by %s" % (rules.name(codes, M.items[f - M.AI][0]), rules.ANAME[M.items[f - M.AI][1]])
    return "out of view by " + rules.ANAME[f - M.AY]
print("objective", M.log["agent"]); print("heading east -> moved east: bias %.2f" % w["b"][d, o])
print(" own couplings:", sorted([(fn(f), round(float(w["dW"][d][f, o]), 2)) for f in np.nonzero(np.abs(w["dW"][d][:, o]) > 1e-6)[0]], key=lambda x: -abs(x[1]))[:60])
print(" shared couplings toward any change:", sorted([(fn(f), round(float(w["W0"][f]), 2)) for f in np.nonzero(np.abs(w["W0"]) > 1e-6)[0]], key=lambda x: -abs(x[1]))[:40])
