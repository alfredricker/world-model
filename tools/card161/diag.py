"""Card 161, data check: where does card 160's agent with the perceptron first run out of plan?
usage: diag.py <experience> <kind> <task> <first seed> <episodes>     (seeds outside card 160's test seeds)"""
import sys, os, numpy as np
from collections import Counter
import model, grid, rules, agent, tasks, evaluate, agent161
from grid import DIRV
exp, kind, t, s0, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
T = evaluate.load(exp); M = evaluate.model(exp, kind, T); codes = T.codes; nm = lambda c: rules.name(codes, int(c))
ex, _ = evaluate.examples(t, codes); goals = agent.goal_events(ex); M._st = {}; A = (agent161.Agent if os.environ.get("AGENT") == "161" else agent.Agent)(M, goals); w = grid.World(tasks.ALL[t], codes); kinds = Counter(); ok = 0
gcodes = {g[1] for g in goals}
for seed in range(s0, s0 + n):
    g, p = w.reset(seed); A.reset(g, p); done = False; k = 0; first = None; lost = False
    while not done:
        before = A.noplan; items = A.bel.items(); fails = dict(A.fail); a = A.act()
        if A.noplan > before and first is None:
            x, y, d = A.bel.pose; fx, fy = x + DIRV[d][0], y + DIRV[d][1]; front = nm(A.bel.map[fx, fy])
            held = sorted(nm(c) + " by " + rules.ANAME[ak] for c, ak in items)
            failed = sorted(set(nm(A.bel.map[X]) + " -> " + nm(o) for (X, o, it), v in fails.items() if v >= 2))
            first = (k, tuple(held), front, tuple(failed))
        g, p, succ, done = w.step(a); ev = A.observe(a, g, p); k += 1
        if any(c in gcodes and a == 5 for _, _, c, c2 in ev):
            lost = True                                                   # the goal's token was changed by toggle: it cannot be picked up any more
    ok += succ
    kinds[("solved" if succ else "FAILED", "goal token destroyed" if lost else "", first[1:] if first else "always a plan")] += 1
print(exp, kind, t, "solved %d of %d" % (ok, n))
if os.environ.get("BRIEF") != "1":
    for key, v in sorted(kinds.items(), key=lambda q: -q[1]):
        print("  %3d  %s" % (v, key))
else:
    fl = [k for k in kinds if k[0] == "FAILED"]; print("   failed %d; of these the goal token destroyed %d" % (sum(kinds[k] for k in fl), sum(kinds[k] for k in fl if k[1])))
