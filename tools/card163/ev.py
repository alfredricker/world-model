"""Card 163: the evaluation.

usage: ev.py probe <stage> <experience> [tasks...]            along episodes of the agent told MiniGrid's rules: the prediction for the event that comes next,
                                                              by how far away it is, and whether the best action is the told agent's
       ev.py run <stage> <experience> <split> <tasks...>      the agent that acts by the prediction; success, the walk against MiniGrid's rules, the first expected event
"""
import os, sys, json, time, pickle
os.environ["FIX"] = "1"
import numpy as np
import evaluate, agent, grid, rules, tasks, act
from grid import DIRV, OFF8, NA, gid
evaluate.SEEDS.update(fresh=range(7000, 7100), gate=range(2000, 2040), smoke=range(2000, 2004))
BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]


def load(stage, exp):
    T = evaluate.load(exp)
    return T, act.Model("runs/163/q_%s_%s.pt" % (stage, exp), T)


def told(task, codes):
    return agent.Agent(rules.Rules(codes), tasks.told_goal(task, codes))


def told_episode(w, A, seed, cap=400):
    """the told agent's episode: (map, pose, tokens out of view, action, events, pose after) per step"""
    g, p = w.reset(seed); A.reset(g, p); done = False; out = []; succ = False
    while not done and len(out) < cap:
        a = A.act(); st = (A.bel.map, A.bel.pose, A.bel.items()); g, p, succ, done = w.step(a); ev = A.observe(a, g, p); out.append(st + (a, ev, p))
    return out, succ


def probe(stage, exp, names, n=20):
    T, M = load(stage, exp); res = {}
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); A = told(t, T.codes); by = {b: [0, 0.0, 0] for b in BUCKETS}; stages = {}
        goal = tasks.told_goal(t, T.codes)
        for seed in range(2000, 2000 + n):
            ep, succ = told_episode(w, A, seed)
            if not succ:
                continue
            nxt = [None] * len(ep); last = None
            if goal[0][0] == "arrive":
                last = (len(ep) - 1, ep[-1][5][:2], M.ARR)
            for i in range(len(ep) - 1, -1, -1):
                if ep[i][4]:
                    x, y, c, c2 = ep[i][4][0]; last = (i, (x, y), c2)
                nxt[i] = last
            rows = [(mp, pose, items, X, o) for (mp, pose, items, a, ev, p2), (j, X, o) in zip(ep, nxt)]; q = M.ask(rows); k = 0
            for i, ((mp, pose, items, a, ev, p2), (j, X, o)) in enumerate(zip(ep, nxt)):
                d = j - i; b = next(b for b in BUCKETS if b[0] <= d <= b[1]); by[b][0] += 1; by[b][1] += float(q[i].max()); by[b][2] += int(q[i].argmax() == a)
            # the goal's own prediction at the start and after each event
            asks = lambda mp: [((int(x), int(y)), M.ARR if g[0] == "arrive" else int(g[2])) for g in goal for x, y in zip(*np.nonzero(mp == g[1]))]
            marks = [0] + [i + 1 for i in range(len(ep) - 1) if ep[i][4]]
            for k, i in enumerate(marks):
                mp, pose, items = ep[i][:3]; G = asks(mp)
                if G:
                    stages.setdefault(k, []).append(float(M.ask([(mp, pose, items, X, o) for X, o in G]).max()))
        res[t] = dict(by_distance={"%d-%d" % b: dict(steps=v[0], mean_q=round(v[1] / max(v[0], 1), 3), same_action=round(100.0 * v[2] / max(v[0], 1), 1)) for b, v in by.items() if v[0]},
                      goal_q_after_k_events={k: round(float(np.mean(v)), 3) for k, v in stages.items()})
        print(stage, exp, t, flush=True)
        for b, v in res[t]["by_distance"].items():
            print("    steps to the event %-6s n %4d   mean q %.3f   best action is the told agent's %5.1f%%" % (b, v["steps"], v["mean_q"], v["same_action"]), flush=True)
        print("    q of the goal at the start and after each event:", res[t]["goal_q_after_k_events"], flush=True)
    json.dump(res, open("runs/163/probe_%s_%s.json" % (stage, exp), "w"), indent=1)


class Truth:
    """MiniGrid's rules, for scoring a step: how far the pose is from where an event can be brought about"""
    def __init__(s, codes):
        s.R = rules.Rules(codes); s.A = agent.Agent(s.R, []); s.A.G = {}; s.A.D = {}; s.NC_arr = None

    def targets(s, mp, items, X, o, arr):
        W, H = mp.shape
        if o == arr:
            return {(X[0] * H + X[1]) * 4 + d: None for d in range(4)}
        out = {}
        for d, (dx, dy) in enumerate(DIRV):
            px, py = X[0] - dx, X[1] - dy
            if 0 <= px < W and 0 <= py < H:
                acts = [a for a in range(NA) if s.R.cell(int(mp[X]), a, gid(dx, dy, d), items) == o]
                if acts:
                    out[(px * H + py) * 4 + d] = acts
        return out

    def dist(s, mp, items, pose, tg):
        H = mp.shape[1]; d, _ = s.A.paths(mp, items, (pose[0] * H + pose[1]) * 4 + pose[2])
        return min((d[t] for t in tg if t in d), default=1e9)

    def best_actions(s, mp, items, pose, X, o, arr):
        """None if the event is not within reach as things are; else the actions the rules rate as closest"""
        tg = s.targets(mp, items, X, o, arr); H = mp.shape[1]; here = (pose[0] * H + pose[1]) * 4 + pose[2]
        if not tg:
            return None
        if here in tg and tg[here] is not None:
            return set(tg[here])
        d0 = s.dist(mp, items, pose, tg)
        if d0 >= agent.C_BLOCK:
            return None
        good = set()
        for a in range(3):
            m, d2 = s.R.agent_one(pose[2], a, grid.neighbours(mp, pose[0], pose[1]), items); p2 = (pose[0] + DIRV[m - 1][0], pose[1] + DIRV[m - 1][1], d2) if m else (pose[0], pose[1], d2)
            if s.dist(mp, items, p2, tg) < d0:
                good.add(a)
        return good


def run(stage, exp, split, names):
    T, M = load(stage, exp); Tr = Truth(T.codes)
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); ex, tried = evaluate.examples(t, T.codes); A = act.Agent(M, agent.goal_events(ex)); TA = told(t, T.codes); out = []; t0 = time.time(); walk = [0, 0]; first = [0, 0]; links = []
        for s in evaluate.SEEDS[split]:
            ep, ok = told_episode(w, TA, s); e1 = next(((x, y), c2) for st in ep for x, y, c, c2 in st[4][:1]) if any(st[4] for st in ep) else None
            g, p = w.reset(s); A.reset(g, p); done = False; n = 0; Tr.A.G = {}; Tr.A.D = {}
            while not done:
                a = A.act()
                if A.held is not None:
                    good = Tr.best_actions(A.bel.map, A.bel.items(), A.bel.pose, A.held[0], A.held[1], M.ARR)
                    if good is not None:
                        walk[0] += a in good; walk[1] += 1
                g, p, succ, done = w.step(a); A.observe(a, g, p); n += 1
            out.append((succ, n, A.noplan)); links.append(A.first[1] if A.first else 0)
            if e1 is not None:
                first[0] += A.first is not None and A.first[0] == e1; first[1] += 1
        ok = [r for r in out if r[0]]; steps = sum(r[1] for r in out)
        res = dict(success=100.0 * len(ok) / len(out), layouts=len(out), steps_when_solved=float(np.mean([r[1] for r in ok])) if ok else None, walk_closest=100.0 * walk[0] / max(walk[1], 1), walk_steps=walk[1],
                   first_event_right=100.0 * first[0] / max(first[1], 1), first_chain_links=float(np.mean(links)), next_best_share=100.0 * sum(r[2] for r in out) / steps, ms_per_step=1000 * (time.time() - t0) / steps,
                   failed=[s for s, r in zip(evaluate.SEEDS[split], out) if not r[0]])
        print(stage, exp, split, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "failed"}, flush=True)
        json.dump(res, open("runs/163/run_%s_%s_%s_%s%s.json" % (stage, exp, split, t, os.environ.get("TAG", "")), "w"), indent=1)


if __name__ == "__main__":
    cmd, stage, exp = sys.argv[1:4]
    if cmd == "probe":
        probe(stage, exp, sys.argv[4:] or list(tasks.ALL))
    elif cmd == "run":
        run(stage, exp, sys.argv[4], sys.argv[5:])
