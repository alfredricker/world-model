"""Card 164: diagnostics and evaluation.

usage:
  ev.py bound <mode> <exp> <n> <tasks...>             handed-event upper bound (like card 163's dev4, now with action ea)
  ev.py reach <mode> <exp> <n> <tasks...>             probe V(s; E) for directly reachable vs blocked/unmet events
  ev.py run <arm> <mode> <exp> <split> <tasks...>     arm: A (card 162 System 2 + System 1 q), B (event-search System 2 + System 1 q)
"""
import os, sys, json, time
os.environ["FIX"] = "1"
import numpy as np
import evaluate, agent, agent161, grid, rules, tasks, net, act, q as Q
from grid import DIRV, OFF8, NA, gid

evaluate.SEEDS.update(fresh=range(8000, 8100), gate=range(2000, 2040), smoke=range(2000, 2005))


def told(task, codes):
    return agent.Agent(rules.Rules(codes), tasks.told_goal(task, codes))


def told_episode(w, A, seed, cap=400):
    g, p = w.reset(seed); A.reset(g, p); done = False; out = []; succ = False
    while not done and len(out) < cap:
        a = A.act(); st = (A.bel.map, A.bel.pose, A.bel.items()); g, p, succ, done = w.step(a); ev = A.observe(a, g, p); out.append(st + (a, ev, p))
    return out, succ


def load_all(mode, exp):
    T = evaluate.load(exp)
    QM = act.QModel("runs/164/q_%s_%s.pt" % (mode, exp))
    EM = net.load(exp, "A", T)
    return T, QM, EM


class Truth:
    """MiniGrid's rules, for scoring walk steps and direct reachability."""
    def __init__(s, codes):
        s.R = rules.Rules(codes); s.A = agent.Agent(s.R, []); s.A.G = {}; s.A.D = {}

    def targets(s, mp, items, X, o, ea, arr):
        W, H = mp.shape
        if o == arr:
            return {(X[0] * H + X[1]) * 4 + d: None for d in range(4)}
        out = {}
        for d, (dx, dy) in enumerate(DIRV):
            px, py = X[0] - dx, X[1] - dy
            if 0 <= px < W and 0 <= py < H:
                if s.R.cell(int(mp[X]), ea, gid(dx, dy, d), items) == o:
                    out[(px * H + py) * 4 + d] = [ea]
        return out

    def dist(s, mp, items, pose, tg):
        H = mp.shape[1]; d, _ = s.A.paths(mp, items, (pose[0] * H + pose[1]) * 4 + pose[2])
        return min((d[t] for t in tg if t in d), default=1e9)

    def best_actions(s, mp, items, pose, X, o, ea, arr):
        tg = s.targets(mp, items, X, o, ea, arr); H = mp.shape[1]; here = (pose[0] * H + pose[1]) * 4 + pose[2]
        if not tg:
            return None
        if here in tg and tg[here] is not None:
            return set(tg[here])
        d0 = s.dist(mp, items, pose, tg)
        if d0 >= agent.C_BLOCK:
            return None
        good = set()
        for a in range(3):
            m, d2 = s.R.agent_one(pose[2], a, grid.neighbours(mp, pose[0], pose[1]), items)
            p2 = (pose[0] + DIRV[m - 1][0], pose[1] + DIRV[m - 1][1], d2) if m else (pose[0], pose[1], d2)
            if s.dist(mp, items, p2, tg) < d0:
                good.add(a)
        return good


def bound(mode, exp, n, names):
    T, QM, _ = load_all(mode, exp)
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); A = act.Handed(QM, []); TA = told(t, T.codes)
        ok = 0; steps = []; told_steps = []; nb = 0; tot = 0
        for seed in range(2000, 2000 + n):
            ep, s0 = told_episode(w, TA, seed)
            A.todo = [((x, y), int(c2), int(st[3])) for st in ep for x, y, c, c2 in st[4]]
            if tasks.told_goal(t, T.codes)[0][0] == "arrive":
                A.todo.append((tuple(ep[-1][5][:2]), QM.ARR, QM.ARR_ACT))
            g, p = w.reset(seed); A.reset(g, p); done = False; i = 0
            while not done:
                a = A.act(); g, p, succ, done = w.step(a); A.observe(a, g, p); i += 1
            ok += succ; tot += i; nb += A.noplan
            if succ:
                steps.append(i); told_steps.append(len(ep))
            elif os.environ.get("WHY"):
                nm = lambda c: "arrive" if c == QM.ARR else rules.name(T.codes, c)
                print("   failed", seed, "reached", A.k, "of", [(X, nm(o), rules.ANAME[ea] if ea < NA else "arr") for X, o, ea in A.todo], "pose", A.bel.pose, flush=True)
        print(mode, exp, t, "handed: solved %d of %d, mean steps %.1f (told %.1f), next-best %.1f%%" %
              (ok, n, np.mean(steps) if steps else 0, np.mean(told_steps) if told_steps else 0, 100.0 * nb / max(tot, 1)), flush=True)


def reach(mode, exp, n, names):
    """Compare V(s; E) = max_a q(s, a; E) for truly directly reachable events vs blocked / unmet-precondition events."""
    T, QM, _ = load_all(mode, exp); Tr = Truth(T.codes)
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); TA = told(t, T.codes)
        pos_near, pos_mid, pos_far, neg_blk, neg_cond = [], [], [], [], []
        for seed in range(2000, 2000 + n):
            ep, ok = told_episode(w, TA, seed)
            if not ok:
                continue
            Tr.A.G = {}; Tr.A.D = {}
            evs = [((x, y), int(c2), int(st[3])) for st in ep for x, y, c, c2 in st[4]]
            if tasks.told_goal(t, T.codes)[0][0] == "arrive":
                evs.append((tuple(ep[-1][5][:2]), QM.ARR, QM.ARR_ACT))
            # check at start and after each event
            marks = [0] + [i + 1 for i in range(len(ep) - 1) if ep[i][4]]
            rows = []; meta = []
            for idx, i in enumerate(marks):
                mp, pose, items = ep[i][:3]
                for j in range(idx, len(evs)):
                    X, o, ea = evs[j]
                    if o != QM.ARR and mp[X] == o:
                        continue
                    tg = Tr.targets(mp, items, X, o, ea, QM.ARR)
                    rows.append((mp, pose, items, X, o, ea))
                    if not tg:
                        meta.append("cond")
                    else:
                        d = Tr.dist(mp, items, pose, tg)
                        if d >= agent.C_BLOCK:
                            meta.append("blk")
                        elif d <= 5:
                            meta.append("near")
                        elif d <= 12:
                            meta.append("mid")
                        else:
                            meta.append("far")
            if rows:
                vs = QM.ask(rows).max(1)
                for m, v in zip(meta, vs):
                    (pos_near if m == "near" else pos_mid if m == "mid" else pos_far if m == "far" else neg_blk if m == "blk" else neg_cond).append(float(v))
        fmt = lambda xs: "n=%3d med=%.3f p10=%.3f p90=%.3f" % (len(xs), np.median(xs), np.percentile(xs, 10), np.percentile(xs, 90)) if xs else "n=  0"
        print(mode, exp, t, flush=True)
        print("   reachable  <=5:", fmt(pos_near))
        print("   reachable 6-12:", fmt(pos_mid))
        print("   reachable  >12:", fmt(pos_far))
        print("   blocked       :", fmt(neg_blk))
        print("   unmet cond    :", fmt(neg_cond), flush=True)


class Base162(agent161.Agent):
    """Exact card 162 Arm A agent (comparison baseline on the same fresh layouts)."""
    def reset(s, g, p):
        super().reset(g, p); s.held = None; s.first = None; s.plans = 0; s.expansions = 0


class BaseSurprise(agent161.Agent):
    """Ablation: card 162's Dijkstra agent with the exact same P14 surprise-recovery rule as Card 164 Arm A."""
    def reset(s, g, p):
        super().reset(g, p); s.held = None; s.first = None; s.plans = 0; s.expansions = 0

    def want_effect(s, X, o, a_req, stack, chain):
        c = int(s.bel.map[X]); items = s.bel.items()
        if (X, o) in chain or len(chain) > 8 or s.fail[(X, o, items)] >= 2:
            return agent.FAIL
        st = s.M.statement(c, o, a_req)
        if st is None:
            return agent.FAIL
        free = any(g < 0 or g == grid.FAR for g in st["G"])
        targets = [] if free else [(X[0] - dx, X[1] - dy, d) for dx, dy, d in map(grid.gid_inv, st["G"])]
        stack2 = stack + (tuple(targets),) if targets else stack; chain2 = chain + ((X, o),)
        for f in st["need"]:
            if not agent.holds(f, items):
                return s.want_in(f, stack2, chain2)
        unmet = []
        refuted = s.fail[(X, o, items)] >= 1
        if s.p(c, o, st["a"], st["G"], items) < 0.5 or refuted:
            unmet = [("it", it) for it in sorted(items) if s.p(c, o, st["a"], st["G"], items - {it}) >= 0.5]
            if not refuted:
                unmet = unmet[:1]
        if unmet and (free or s.free_reach(targets)):
            for u in unmet:
                r = s.want_out(u, stack2, chain2)
                if isinstance(r, int):
                    return r
        r = agent.HERE if free else s.go(targets, stack, chain2)
        if r == agent.HERE:
            s.tried = (X, o, items)
            return int(st["a"])
        return r


def run(arm, mode, exp, split, names):
    T, QM, EM = load_all(mode, exp); Tr = Truth(T.codes)
    os.makedirs("runs/164", exist_ok=True)
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); ex, _ = evaluate.examples(t, T.codes); goals = agent.goal_events(ex)
        A = Base162(EM, goals) if arm == "162" else BaseSurprise(EM, goals) if arm == "162s" else act.ArmA(QM, EM, goals) if arm == "A" else act.ArmB(QM, EM, goals)
        TA = told(t, T.codes); out = []; walk = [0, 0]; first = [0, 0]; t0 = time.time(); lost = 0; gc = {g[1] for g in goals}
        for s in evaluate.SEEDS[split]:
            ep, _ = told_episode(w, TA, s)
            e1 = next((((x, y), int(c2), int(st[3])) for st in ep for x, y, c, c2 in st[4][:1]), None)
            g, p = w.reset(s); A.reset(g, p); done = False; n = 0; gone = False; Tr.A.G = {}; Tr.A.D = {}
            while not done:
                a = A.act()
                if A.held is not None:
                    good = Tr.best_actions(A.bel.map, A.bel.items(), A.bel.pose, A.held[0], A.held[1], A.held[2], QM.ARR)
                    if good is not None:
                        walk[0] += a in good; walk[1] += 1
                g, p, succ, done = w.step(a); ev = A.observe(a, g, p); n += 1
                gone |= any(c in gc and a == rules.TOGGLE for _, _, c, c2 in ev)
            out.append((succ, n, A.noplan, A.plans, A.expansions)); lost += gone and not succ
            if e1 is not None:
                first[0] += (A.first == e1); first[1] += 1
        ok = [r for r in out if r[0]]; steps = sum(r[1] for r in out)
        res = dict(success=100.0 * len(ok) / len(out), layouts=len(out),
                   steps_when_solved=float(np.mean([r[1] for r in ok])) if ok else None,
                   walk_closest=100.0 * walk[0] / max(walk[1], 1), walk_steps=walk[1],
                   first_event_right=100.0 * first[0] / max(first[1], 1),
                   next_best_share=100.0 * sum(r[2] for r in out) / max(steps, 1),
                   mean_plans=float(np.mean([r[3] for r in out])),
                   mean_expansions=float(np.mean([r[4] for r in out])),
                   failed_with_goal_token_destroyed=int(lost),
                   ms_per_step=1000.0 * (time.time() - t0) / max(steps, 1),
                   failed=[s for s, r in zip(evaluate.SEEDS[split], out) if not r[0]])
        print(arm, mode, exp, split, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "failed"}, flush=True)
        json.dump(res, open("runs/164/run_%s_%s_%s_%s_%s%s.json" % (arm, mode, exp, split, t, os.environ.get("TAG", "")), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bound":
        bound(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:])
    elif cmd == "reach":
        reach(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])
