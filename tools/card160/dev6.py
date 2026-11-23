"""after the test: why has the learned agent no plan in UnlockPickup?  usage: dev6.py <experience> <task> <seed>"""
import sys, pickle, numpy as np
import grid, rules, energy, agent, tasks, evaluate
from grid import OFF8, DIRV
exp, t, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); T = evaluate.load(exp); M = evaluate.model(exp, "sparse", T); codes = T.codes; nm = lambda c: rules.name(codes, int(c))
ex, _ = evaluate.examples(t, codes); goals = agent.goal_events(ex); print("goals", [(g[0], nm(g[1])) + tuple(nm(x) if i == 0 else rules.ANAME[x] for i, x in enumerate(g[2:])) for g in goals])
w = grid.World(tasks.ALL[t], codes); A = agent.Agent(M, goals); g, p = w.reset(seed); A.reset(g, p); mp = A.bel.map; print("pose", p); print("\n".join(" ".join("%-9s" % nm(mp[x, y])[:9] for x in range(mp.shape[0])) for y in range(mp.shape[1])))
for goal in goals:
    for X in A.cells(goal[1]):
        st = M.statement(int(mp[X]), goal[2], goal[3]); print("goal token", X, nm(mp[X]), "statement", st)
        tg = [(X[0] - dx, X[1] - dy, d) for dx, dy, d in map(grid.gid_inv, st["G"])]; W, H = mp.shape; dist, prev = A.paths(mp, A.bel.items(), (p[0] * H + p[1]) * 4 + p[2])
        print("  targets", tg, "distances", [dist.get((x * H + y) * 4 + d) for x, y, d in tg if 0 <= x < W and 0 <= y < H])
for c in np.unique(mp):
    c = int(c); t3 = codes.triples[c]
    if t3 and t3[0] == 4:
        print("door", nm(c), "known changes", [nm(o) for o in M.outcomes(c)], "all next codes seen", [nm(o) for o in M.cell.get(c, [])])
        x, y = [int(v[0]) for v in np.nonzero(mp == c)]; nb = list(grid.neighbours(mp, x - 1, y)); j = OFF8.index((1, 0))
        for o in M.cell.get(c, []):
            nb2 = list(nb); nb2[j] = o; print("   from the west heading east, forward, if the door were", nm(o), "->", M.agent_one(0, 2, tuple(nb2), A.bel.items()))
        nb2 = list(nb); nb2[j] = M.codes.of((1, 0, 0)); print("   ... if it were empty ->", M.agent_one(0, 2, tuple(nb2), A.bel.items()))
