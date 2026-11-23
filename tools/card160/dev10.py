"""after the test: a few episodes on layouts outside the test.  usage: [FIX=1] dev10.py <experience> <kind> <task> <first seed> <n>"""
import sys, time, numpy as np
import grid, rules, agent, tasks, evaluate
exp, kind, t, s0, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); T = evaluate.load(exp); M = evaluate.model(exp, kind, T)
ex, _ = evaluate.examples(t, T.codes); A = agent.Agent(M, agent.goal_events(ex)); w = grid.World(tasks.ALL[t], T.codes)
for s in range(s0, s0 + n):
    t0 = time.time(); r = agent.episode(w, s, A); print(t, s, "fix" if agent.AFTER_TEST_FIX else "frozen", "success", r[0], "steps", r[1], "no plan", r[2], "%.1f s" % (time.time() - t0), flush=True)
