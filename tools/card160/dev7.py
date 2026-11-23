"""after the test: how often did play walk onto an open door of each colour, by heading?  usage: dev7.py <experience>"""
import sys, numpy as np
import grid, rules, evaluate
from grid import OFF8, DIRV
T = evaluate.load(sys.argv[1]); codes = T.codes; cnt = {}
for (d, a, nb, it, out), n in T.agent.items():
    f = nb[OFF8.index(DIRV[d])]; t = codes.triples[f]
    if a == 2 and t and t[0] == 4 and t[2] == 0:
        q = cnt.setdefault(rules.CNAME[t[1]], np.zeros((4, 2), int)); q[d, 0] += n; q[d, 1] += n * (out[0] != 0)
for c, q in sorted(cnt.items()):
    print("open %-7s door ahead, forward: tries by heading (east, south, west, north) %s, moved %s" % (c, q[:, 0].tolist(), q[:, 1].tolist()))
