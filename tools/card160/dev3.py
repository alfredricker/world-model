"""trace one episode of the rules agent.  usage: dev3.py <task> <seed> <steps>"""
import sys, numpy as np
import grid, rules, agent, tasks
t, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); codes = grid.Codes(); R = rules.Rules(codes); w = grid.World(tasks.ALL[t], codes); A = agent.Agent(R, tasks.told_goal(t, codes))
g, p = w.reset(seed); A.reset(g, p); print(w.u.mission, w.u.max_steps); print(str(w.u))
for i in range(n):
    a = A.act(); g, p, succ, done = w.step(a); ev = A.observe(a, g, p)
    print(i, rules.ANAME[a], p, [(x, y, rules.name(codes, c), rules.name(codes, c2)) for x, y, c, c2 in ev], [(rules.name(codes, k), rules.ANAME[b]) for k, b in A.bel.items()], "carrying", w.u.carrying, "noplan", A.noplan, "SUCCESS" if succ else "")
    if done: break
print(str(w.u))
