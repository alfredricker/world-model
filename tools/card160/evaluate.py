"""Card 160: the evaluation.

usage: evaluate.py fit <experience> <sparse|mlp>          fits and saves the model, scores its statements against MiniGrid's rules
       evaluate.py run <experience> <sparse|mlp|rules|random> <dev|test> [tasks...]     goal from 20 example success steps; 100 layouts a task
       evaluate.py effects <experience> <sparse|mlp> <dev|test>                        F1 of predicted events on trajectories of the tasks
"""
import sys, json, time, pickle, numpy as np
import grid, rules, energy, agent, tasks
from grid import NA, NG, FAR, R, gid, Belief

N_EX, SEEDS = 20, dict(dev=range(500, 600), test=range(1000, 1100))


def load(exp):
    return pickle.load(open("runs/160/exp_%s.pkl" % exp, "rb"))


def model(exp, kind, T):
    if kind == "rules":
        return rules.Rules(T.codes)
    M = pickle.load(open("runs/160/model_%s_%s.pkl" % (exp, kind), "rb")); M.codes = T.codes
    return M


def examples(task, codes, n=N_EX, s0=3000):
    """example success steps, from the agent that is told MiniGrid's rules, in layouts that are never tested"""
    w = grid.World(tasks.ALL[task], codes); A = agent.Agent(rules.Rules(codes), tasks.told_goal(task, codes)); out = []; tried = 0
    while len(out) < n and tried < 3 * n:
        ex = agent.episode(w, s0 + tried, A, want_example=True)[3]; tried += 1
        if ex is not None:
            out.append(ex)
    return out, tried


def statements(M, T):
    Rl = rules.Rules(T.codes); cnt = rules.effect_counts(T); ctx = rules.contexts(Rl); c_, a_, g_, it_, o_, n_ = T.rows(); rows = []
    for c, o, a in rules.truth_effects(T.codes, sorted(set(int(x) for x in np.unique(c_)))):
        if cnt.get((c, o, a), 0) < energy.SEEN:
            continue
        st = M.statement(c, o, a); m = c_ == c; ng = np.bincount(g_[m], weights=n_[m], minlength=NG + 1)
        bad = [x for x in ctx if ng[x[1]] >= energy.SEEN and rules.says(st, *x) != (x[0] == a and Rl.cell(c, *x) == o)]
        part = "right" if not bad else "none" if st is None else "place" if any(not x[2] for x in bad) else "hands"
        rows.append(dict(code=rules.name(T.codes, c), next=rules.name(T.codes, o), action=rules.ANAME[a], times=cnt[(c, o, a)], verdict=part))
    return rows


def run(exp, kind, split, names):
    T = load(exp); res = {}
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes)
        if kind == "random":
            A = None
        else:
            ex, tried = examples(t, T.codes); goals = agent.goal_events(ex); A = agent.Agent(model(exp, kind, T), goals)
        t0 = time.time(); out = [agent.episode(w, s, A if A is not None else agent.Random(s)) for s in SEEDS[split]]; ok = [r for r in out if r[0]]; steps = sum(r[1] for r in out)
        res[t] = dict(success=100.0 * len(ok) / len(out), steps_when_solved=float(np.mean([r[1] for r in ok])) if ok else None, no_plan_share=100.0 * sum(r[2] for r in out) / steps,
                      ms_per_step=1000 * (time.time() - t0) / steps, failed=[s for s, r in zip(SEEDS[split], out) if not r[0]][:20])
        if A is not None:
            res[t].update(examples=len(ex), example_tries=tried, goal_events=len(goals))
        print(exp, kind, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res[t].items()}, flush=True)
        json.dump(res, open("runs/160/run_%s_%s_%s%s.json" % (exp, kind, split, "_fix" if agent.AFTER_TEST_FIX else ""), "w"), indent=1)
    return res


def effects(exp, kind, split):
    """trajectories of the told agent with three actions in ten at random; predicted events against the ones that happened"""
    T = load(exp); M = model(exp, kind, T); res = {}; rng = np.random.default_rng(0)
    for t in (tasks.TEST if split == "test" else tasks.DEV):
        w = grid.World(tasks.ALL[t], T.codes); A = agent.Agent(rules.Rules(T.codes), tasks.told_goal(t, T.codes)); tp = fp = fn = 0; mv = []; n = 0
        for s in range(4000, 4020):
            g, p = w.reset(s); A.reset(g, p); done = False; k = 0
            while not done and k < 300:
                a = A.act(); a = int(rng.integers(NA)) if rng.random() < 0.3 else a; mp, pose, items = A.bel.map, A.bel.pose, A.bel.items()
                c, gg, _, xs, ys = grid.cell_rows(mp, pose, mp, pose); pred = set()
                for ci, gi, x, y in zip(c, gg, xs, ys):
                    q = M.p_cell(int(ci), a, int(gi), items) if hasattr(M, "p_cell") else None; o = max(q, key=q.get) if q else M.cell(int(ci), a, int(gi), items)
                    if o != ci:
                        pred.add((int(x), int(y), int(o)))
                pm = M.agent_one(pose[2], a, grid.neighbours(mp, pose[0], pose[1]), items)
                g, p, succ, done = w.step(a); ev = A.observe(a, g, p); true = {(x, y, c2) for x, y, _, c2 in ev}; k += 1; n += 1
                pred = {q for q in pred if (q[0], q[1]) != (p[0], p[1])}                     # the cell the agent steps onto is hidden afterwards
                tp += len(pred & true); fp += len(pred - true); fn += len(true - pred); mv.append(tuple(pm) == grid.agent_outcome(pose, p))
        res[t] = dict(f1=100.0 * 2 * tp / max(2 * tp + fp + fn, 1), events=tp + fn, predicted=tp + fp, right=tp, agent_move_right=100.0 * float(np.mean(mv)), steps=n)
        print(exp, kind, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res[t].items()}, flush=True)
    json.dump(res, open("runs/160/effects_%s_%s_%s.json" % (exp, kind, split), "w"), indent=1)


if __name__ == "__main__":
    cmd, exp, kind = sys.argv[1:4]
    if cmd == "fit":
        T = load(exp); t0 = time.time(); M = energy.Energy(T, kind); sec = time.time() - t0; rows = statements(M, T); pickle.dump(M, open("runs/160/model_%s_%s.pkl" % (exp, kind), "wb"))
        v = [r["verdict"] for r in rows]; out = dict(experience=exp, kind=kind, seconds=round(sec), nats=M.nll, objective=M.log, statements=len(rows), right=v.count("right"),
                                                     wrong_none=v.count("none"), wrong_place=v.count("place"), wrong_hands=v.count("hands"), rows=rows)
        if kind == "sparse":
            out["couplings_nonzero"] = dict(shared=int((np.abs(M.w["W0"]) > 1e-6).sum()), own=int((np.abs(M.w["dW"]) > 1e-6).sum()), of=int(M.w["dW"].size))
        json.dump(out, open("runs/160/fit_%s_%s.json" % (exp, kind), "w"), indent=1)
        print(exp, kind, "fitted in %.0f s; statements right %d of %d (no statement %d, place wrong %d, hands wrong %d)" % (sec, out["right"], len(rows), out["wrong_none"], out["wrong_place"], out["wrong_hands"]))
        for r in rows:
            if r["verdict"] != "right":
                print("   ", r)
    elif cmd == "run":
        split = sys.argv[4]; run(exp, kind, split, sys.argv[5:] or list(tasks.TEST if split == "test" else tasks.DEV))
    elif cmd == "effects":
        effects(exp, kind, sys.argv[4])
