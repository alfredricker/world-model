"""a bound: the agent is handed the next event of the told agent's solution as the event to expect, and acts on the prediction for it.
If this solves the task, the prediction walks to and brings about every near event, and what fails is choosing the event.  usage: dev4.py <stage> <experience> <n> <tasks...>"""
import os, sys
os.environ["FIX"] = "1"
import numpy as np, evaluate, agent, grid, rules, tasks, act, ev
stage, exp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
T, M = ev.load(stage, exp)
class Handed(act.Agent):
    def expect(s):
        return (1.0, s.todo[s.k] if s.k < len(s.todo) else None, 1)
for t in sys.argv[4:]:
    w = grid.World(tasks.ALL[t], T.codes); A = Handed(M, []); TA = ev.told(t, T.codes); ok = 0; steps = []; told_steps = []; nb = 0; tot = 0
    for seed in range(2000, 2000 + n):
        ep, s0 = ev.told_episode(w, TA, seed); A.todo = [((x, y), c2) for st in ep for x, y, c, c2 in st[4]]
        if tasks.told_goal(t, T.codes)[0][0] == "arrive":
            A.todo.append((tuple(ep[-1][5][:2]), M.ARR))
        g, p = w.reset(seed); A.reset(g, p); A.k = 0; done = False; i = 0
        while not done:
            a = A.act(); g, p, succ, done = w.step(a); i += 1
            for x, y, c, c2 in A.observe(a, g, p):
                if A.k < len(A.todo) and ((x, y), c2) == A.todo[A.k]:
                    A.k += 1; A.held = None
        ok += succ; tot += i; nb += A.noplan
        if not succ and os.environ.get("WHY"):
            nm = lambda c: "arrive" if c == M.ARR else rules.name(T.codes, c); print("   failed", seed, "told ok", s0, "reached", A.k, "of", [(X, nm(o)) for X, o in A.todo], "pose", A.bel.pose, "held", [nm(k) for k, _ in A.bel.items()], flush=True)
        if succ:
            steps.append(i); told_steps.append(len(ep))
    print(stage, t, "handed the next event: solved %d of %d, mean steps %.1f (the told agent %.1f), next-best actions %.1f%%" % (ok, n, np.mean(steps) if steps else 0, np.mean(told_steps) if steps else 0, 100.0 * nb / tot), flush=True)
