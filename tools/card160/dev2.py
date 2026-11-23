"""the gate's upper bound: the agent with MiniGrid's rules written by hand, and random actions.  usage: dev2.py <first seed> <n> <tasks...>"""
import sys, time, numpy as np
import grid, rules, agent, tasks
s0, n = int(sys.argv[1]), int(sys.argv[2])
for t in sys.argv[3:]:
    codes = grid.Codes(); R = rules.Rules(codes); w = grid.World(tasks.ALL[t], codes); A = agent.Agent(R, tasks.told_goal(t, codes)); t0 = time.time()
    res = [agent.episode(w, s, A) for s in range(s0, s0 + n)]; rnd = [agent.episode(w, s, agent.Random(s)) for s in range(s0, s0 + n)]
    ok = [r for r in res if r[0]]
    print("%-20s rules agent %3d of %d, steps when solved %.1f, steps with no plan %.1f%%, %.1f ms a step | random %d of %d | failed seeds %s" % (
        t, len(ok), n, np.mean([r[1] for r in ok]) if ok else 0, 100 * sum(r[2] for r in res) / sum(r[1] for r in res), 1000 * (time.time() - t0) / sum(r[1] for r in res), sum(r[0] for r in rnd), n,
        [s0 + i for i, r in enumerate(res) if not r[0]][:8]), flush=True)
