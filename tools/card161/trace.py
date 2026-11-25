"""Card 161, after the test: why has the agent no plan in KeyCorridor?  The tree of wants at the first step without a plan.
usage: trace.py <experience> <task> <first seed> <episodes> <trees to print>     (layouts outside every test)"""
import os, sys
os.environ["FIX"] = "1"
import numpy as np
from collections import Counter
import model, evaluate, agent, agent161, grid, rules, tasks
from agent import FAIL, HERE
exp, t, s0, n, show = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
T = evaluate.load(exp); M = evaluate.model(exp, "mlp", T); M._st = {}; codes = T.codes; nm = lambda c: rules.name(codes, int(c))


class Tr(agent161.Agent):
    log = None

    def note(s, depth, text):
        if s.log is not None:
            s.log.append("  " * depth + text)

    def want_effect(s, X, o, a_req, stack, chain):
        c = int(s.bel.map[X]); items = s.bel.items(); d = len(chain); why = None
        if (X, o) in chain: why = "already in this chain"
        elif len(chain) > 8: why = "chain too long"
        elif s.fail[(X, o, items)] >= 2: why = "failed twice in this state"
        elif s.M.statement(c, o, a_req) is None: why = "no statement for this change" + ("" if a_req is None else " by " + rules.ANAME[a_req])
        s.note(d, "want %s at %s -> %s%s" % (nm(c), X, nm(o), "   FAIL: " + why if why else ""))
        r = super().want_effect(X, o, a_req, stack, chain)
        if why is None:
            s.note(d, "   = %s" % (rules.ANAME[r] if isinstance(r, int) else r))
        return r

    def want_in(s, f, stack, chain):
        s.note(len(chain), "needs out of view: %s" % ((nm(f[1][0]) + " by " + rules.ANAME[f[1][1]]) if f[0] == "it" else "anything by " + rules.ANAME[f[1]]))
        return super().want_in(f, stack, chain)

    def want_out(s, f, stack, chain):
        s.note(len(chain), "must come back into view: %s" % (nm(f[1][0]) if f[0] == "it" else "anything gone by " + rules.ANAME[f[1]]))
        return super().want_out(f, stack, chain)

    def go(s, targets, stack, chain):
        r = super().go(targets, stack, chain)
        if r == FAIL:
            s.note(len(chain), "no way to the place (%d target poses)" % len(targets))
        return r


ex, _ = evaluate.examples(t, codes); A = Tr(M, agent.goal_events(ex)); w = grid.World(tasks.ALL[t], codes); kinds = Counter(); shown = 0
for seed in range(s0, s0 + n):
    g, p = w.reset(seed); A.reset(g, p); done = False; k = 0; first = None
    while not done:
        if first is None:
            mp = A.bel.map.copy(); pose = A.bel.pose; A.log = []; r = A.plan(); lg = A.log; A.log = None
            if r is None:
                top = [l for l in lg if "FAIL" in l or "no way" in l]; first = (k, mp, pose, lg, sorted(nm(c) + " by " + rules.ANAME[a] for c, a in A.bel.items()))
        a = A.act(); g, p, succ, done = w.step(a); A.observe(a, g, p); k += 1
    if first is None:
        kinds[("solved" if succ else "FAILED", "always a plan")] += 1; continue
    leaf = [l.strip() for l in first[3] if "FAIL" in l or "no way" in l]; why = leaf[0].split("FAIL: ")[-1] if leaf else "?"
    why = "no way to the place" if why.startswith("no way") else why
    kinds[("solved" if succ else "FAILED", "first no plan at step %s" % ("0" if first[0] == 0 else ">0"), why)] += 1
    if not succ and shown < show:
        shown += 1; mp = first[1]; print("---- seed", seed, "step", first[0], "pose", first[2], "out of view:", first[4])
        print("\n".join(" ".join("%-7s" % nm(mp[x, y]).replace("closed ", "c.").replace("locked ", "L.").replace("open ", "o.")[:7] for x in range(mp.shape[0])) for y in range(mp.shape[1])))
        print("\n".join(first[3][:40]))
print(exp, t); [print("  %3d  %s" % (v, key)) for key, v in sorted(kinds.items(), key=lambda q: -q[1])]
