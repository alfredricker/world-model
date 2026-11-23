"""after the test: trace the learned agent.  usage: [FIX=1] dev11.py <experience> <kind> <task> <seed> <steps>"""
import sys, time, numpy as np
import grid, rules, agent, tasks, evaluate
exp, kind, t, seed, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); T = evaluate.load(exp); M = evaluate.model(exp, kind, T); codes = T.codes
ex, _ = evaluate.examples(t, codes); A = agent.Agent(M, agent.goal_events(ex)); w = grid.World(tasks.ALL[t], codes); g, p = w.reset(seed); A.reset(g, p); print("examples done", flush=True)
for i in range(n):
    t0 = time.time(); a = A.act(); g, p, succ, done = w.step(a); ev = A.observe(a, g, p)
    print(i, rules.ANAME[a], p, [(x, y, rules.name(codes, c), rules.name(codes, c2)) for x, y, c, c2 in ev], "no plan", A.noplan, "%.2f s" % (time.time() - t0), "SUCCESS" if succ else "", flush=True)
    if done: break
