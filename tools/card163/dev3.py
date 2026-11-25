"""how far the agent gets: the furthest link of the told solution it reached, per layout.  usage: dev3.py <stage> <experience> <task> [n]"""
import os, sys
os.environ["FIX"] = "1"
import numpy as np, evaluate, agent, grid, rules, tasks, act, ev
from collections import Counter
stage, exp, t = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
T, M = ev.load(stage, exp); w = grid.World(tasks.ALL[t], T.codes); ex, _ = evaluate.examples(t, T.codes); A = act.Agent(M, agent.goal_events(ex)); TA = ev.told(t, T.codes); nm = lambda c: rules.name(T.codes, c); far = Counter(); ok = 0
for seed in range(2000, 2000 + n):
    ep, _ = ev.told_episode(w, TA, seed); sol = [(x, y, c2) for st in ep for x, y, c, c2 in st[4]]
    g, p = w.reset(seed); A.reset(g, p); done = False; k = 0; best = 0
    while not done:
        a = A.act(); g, p, succ, done = w.step(a)
        for x, y, c, c2 in A.observe(a, g, p):
            if k < len(sol) and (x, y, c2) == sol[k]:
                k += 1; best = max(best, k)
    ok += succ; far["%d of %d links (%s)" % (best, len(sol), "; ".join("%s" % nm(c2) for x, y, c2 in sol))] += 1
print(stage, t, "solved", ok, "of", n)
for k, v in sorted(far.items()):
    print("   %3d  %s" % (v, k))
