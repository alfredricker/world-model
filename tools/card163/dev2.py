"""the goal's prediction along the told agent's episodes, by steps left to the goal, against GAMMA ** steps.  usage: dev2.py <stage> <experience> <task> [n]"""
import os, sys
os.environ["FIX"] = "1"
import numpy as np, evaluate, agent, grid, rules, tasks, act, ev, q as Q
stage, exp, t = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
T, M = ev.load(stage, exp); w = grid.World(tasks.ALL[t], T.codes); A = ev.told(t, T.codes); goal = tasks.told_goal(t, T.codes); by = {}
for seed in range(2000, 2000 + n):
    ep, succ = ev.told_episode(w, A, seed)
    if not succ:
        continue
    for i, (mp, pose, items, a, e, p2) in enumerate(ep):
        G = [((int(x), int(y)), M.ARR if g[0] == "arrive" else int(g[2])) for g in goal for x, y in zip(*np.nonzero(mp == g[1]))]
        if G:
            qq = M.ask([(mp, pose, items, X, o) for X, o in G]); j = int(qq.max(1).argmax()); left = len(ep) - 1 - i; b = by.setdefault(left // 4 * 4, [0, 0.0, 0, 0.0]); b[0] += 1; b[1] += float(qq[j].max()); b[2] += int(qq[j].argmax() == a); b[3] += float(qq[j].max() - np.sort(qq[j])[-2])
for k in sorted(by):
    v = by[k]; print("steps left %2d-%2d  n %3d  mean q %.3f   %.2f ** steps %.3f   best action is the told agent's %3.0f%%   gap to the second action %.3f" % (k, k + 3, v[0], v[1] / v[0], Q.GAMMA, Q.GAMMA ** (k + 1.5), 100 * v[2] / v[0], v[3] / v[0]))
