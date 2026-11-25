"""trace the agent that acts by the prediction.  usage: dev1.py <stage> <experience> <task> <seed> [steps]"""
import os, sys
os.environ["FIX"] = "1"
import numpy as np, evaluate, agent, grid, rules, tasks, act, ev
stage, exp, t, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); cap = int(sys.argv[5]) if len(sys.argv) > 5 else 80
T, M = ev.load(stage, exp); w = grid.World(tasks.ALL[t], T.codes); ex, _ = evaluate.examples(t, T.codes); A = act.Agent(M, agent.goal_events(ex)); nm = lambda c: rules.name(T.codes, c)
g, p = w.reset(seed); A.reset(g, p); done = False; n = 0
print("goals", [(g[0], nm(g[1]), g[2] if g[0] == "arrive" else nm(g[2])) for g in A.goals][:8])
while not done and n < cap:
    G = A.goal_asks(A.bel.map); q = M.ask([(A.bel.map, A.bel.pose, A.bel.items(), X, o) for X, o in G]); a = A.act()
    g, p, succ, done = w.step(a); e = A.observe(a, g, p); n += 1
    print("%3d pose %s held %s  goal q by action %s  act %s%s" % (n, A.bel.pose, [nm(k) for k, _ in A.bel.items()], np.round(q.max(0), 2) if len(G) else None, rules.ANAME[a], "   EVENT " + "; ".join("%s -> %s at %s" % (nm(c), nm(c2), (x, y)) for x, y, c, c2 in e) if e else ""))
print("success", succ, "steps", n)
