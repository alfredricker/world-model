"""Card 162: the evaluation.  Arms: base (card 161's arm A), A (the network for the agent's move), B (one network for every token).

usage: ev.py fit <arm> <experience>
       ev.py doors <arm> <experience> [layouts]            can a door imagined open be walked through, by heading (KeyCorridor maps outside every test)
       ev.py moves <arm> <experience>                      the agent's move against MiniGrid's rules, for every code ahead and heading
       ev.py run <arm> <experience> <split> <tasks...>     card 161's agent; 100 layouts a task
Every agent runs with card 160's planner repair (FIX=1)."""
import os, sys, json, time, pickle
os.environ["FIX"] = "1"
import numpy as np
import model, evaluate, agent, agent161, grid, rules, tasks, net
from grid import DIRV, OFF8, OUT
evaluate.SEEDS.update(fresh=range(6000, 6100), old=range(5000, 5100), gate=range(2000, 2040), smoke=range(2000, 2003))
FWD = 2


def fit(arm, exp):
    T = evaluate.load(exp); base = net.load(exp, "base", T); log = []; t0 = time.time(); w = net.fit(base, T, arm == "B", log=log)
    pickle.dump(w, open("runs/162/net_%s_%s.pkl" % (arm, exp), "wb")); print("fit", arm, exp, "rows", w["rows"], "%.0f s" % (time.time() - t0), "objective", log, flush=True)


def doors(arm, exp, n=40):
    T = evaluate.load(exp); M = net.load(exp, arm, T); codes = T.codes; w = grid.World(tasks.ALL["KeyCorridorS3R3"], codes); tot = np.zeros(4, int); ok = np.zeros(4, int); wall = codes.of((rules.WALL, 5, 0)) if hasattr(rules, "WALL") else None
    for seed in range(2000, 2000 + n):
        mp, pose = w.reset(seed)
        for c in np.unique(mp):
            t3 = codes.triples[int(c)]
            if not (t3 and t3[0] == rules.DOOR and t3[2] == 1):
                continue
            outs = M.outcomes(int(c))
            for x, y in zip(*np.nonzero(mp == c)):
                for d, (dx, dy) in enumerate(DIRV):
                    px, py = int(x) - dx, int(y) - dy
                    if rules.name(codes, mp[px, py]) == "wall":
                        continue
                    tot[d] += 1
                    if outs:
                        nb = list(grid.neighbours(mp, px, py)); nb[OFF8.index((dx, dy))] = outs[0]; a = M.statement(int(c), outs[0])["a"]
                        ok[d] += M.agent_one(d, FWD, tuple(nb), frozenset([(int(c), a)]))[0] != 0
    out = dict(arm=arm, experience=exp, approaches_by_heading=tot.tolist(), passable_by_heading=ok.tolist(), percent=100.0 * ok.sum() / tot.sum())
    json.dump(out, open("runs/162/doors_%s_%s.json" % (arm, exp), "w")); print(out, flush=True)


def moves(arm, exp):
    """forward with each known code ahead, walls on the other sides, nothing out of view: does the agent move, as MiniGrid's rules say?  by whether the experience holds that (code, heading)"""
    T = evaluate.load(exp); M = net.load(exp, arm, T); Rl = rules.Rules(T.codes); codes = T.codes; tried = set()
    for (d, a, nb, it, out), cnt in T.agent.items():
        if a == FWD:
            tried.add((nb[OFF8.index(DIRV[d])], d))
    wall = next(c for c in range(len(codes.triples)) if rules.name(codes, c) == "wall"); empty = codes.of((rules.EMPTY, 0, 0)); res = dict(tried=[0, 0], never=[0, 0]); wrong = []
    for c in range(2, M.NC):
        for d in range(4):
            nb = [empty] * 8; nb[OFF8.index(DIRV[d])] = c; nb = tuple(nb); truth = Rl.agent_one(d, FWD, nb, frozenset()); says = M.agent_one(d, FWD, nb, frozenset()); k = "tried" if (c, d) in tried else "never"
            res[k][0] += tuple(says) == tuple(truth); res[k][1] += 1
            if tuple(says) != tuple(truth):
                wrong.append((rules.name(codes, c), d, k))
    out = dict(arm=arm, experience=exp, right_of_tried=res["tried"], right_of_never_tried=res["never"], wrong=wrong); json.dump(out, open("runs/162/moves_%s_%s.json" % (arm, exp), "w")); print(out, flush=True)


def run(arm, exp, split, names):
    T = evaluate.load(exp); M = net.load(exp, arm, T)
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); ex, tried = evaluate.examples(t, T.codes); A = agent161.Agent(M, agent.goal_events(ex)); gc = {g[1] for g in A.goals}; out = []; lost = 0; t0 = time.time()
        for s in evaluate.SEEDS[split]:
            g, p = w.reset(s); A.reset(g, p); done = False; n = 0; gone = False
            while not done:
                a = A.act(); g, p, succ, done = w.step(a); ev = A.observe(a, g, p); n += 1
                gone |= any(c in gc and a == rules.TOGGLE for _, _, c, c2 in ev)
            out.append((succ, n, A.noplan)); lost += gone and not succ
        ok = [r for r in out if r[0]]; steps = sum(r[1] for r in out)
        res = dict(success=100.0 * len(ok) / len(out), layouts=len(out), steps_when_solved=float(np.mean([r[1] for r in ok])) if ok else None, no_plan_share=100.0 * sum(r[2] for r in out) / steps,
                   ms_per_step=1000 * (time.time() - t0) / steps, failed=[s for s, r in zip(evaluate.SEEDS[split], out) if not r[0]], failed_with_goal_token_destroyed=int(lost))
        print(arm, exp, split, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "failed"}, flush=True)
        json.dump(res, open("runs/162/run_%s_%s_%s_%s.json" % (arm, exp, split, t), "w"), indent=1)


if __name__ == "__main__":
    cmd, arm, exp = sys.argv[1:4]
    if cmd == "fit": fit(arm, exp)
    elif cmd == "doors": doors(arm, exp, int(sys.argv[4]) if len(sys.argv) > 4 else 40)
    elif cmd == "moves": moves(arm, exp)
    elif cmd == "run": run(arm, exp, sys.argv[4], sys.argv[5:])
