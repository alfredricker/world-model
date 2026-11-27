"""Card 165: diagnostics and evaluation.

usage:
  ev.py check <exp> <n> <tasks...>                  score RNet reachability (r_arr and r_act) against MiniGrid rules
  ev.py run <exp> <split> <tasks...>                ZeroDijkstraAgent (RNet + cell MLP + frozen System 1 q_reach)
"""
import os, sys, json, time, pickle
os.environ["FIX"] = "1"
import numpy as np
import evaluate, agent, grid, rules, tasks, net, act as act164, plan
from grid import DIRV, NA, gid

evaluate.SEEDS.update(fresh=range(9000, 9100), fresh164=range(8000, 8100), gate=range(2000, 2040), smoke=range(2000, 2005))


def told(task, codes):
    return agent.Agent(rules.Rules(codes), tasks.told_goal(task, codes))


def told_episode(w, A, seed, cap=400):
    g, p = w.reset(seed); A.reset(g, p); done = False; out = []; succ = False
    while not done and len(out) < cap:
        a = A.act(); st = (A.bel.map, A.bel.pose, A.bel.items()); g, p, succ, done = w.step(a); ev = A.observe(a, g, p); out.append(st + (a, ev, p))
    return out, succ


class Truth:
    """MiniGrid's rules, for scoring walk steps and binary reachability."""
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


def load_all(exp):
    T = evaluate.load(exp)
    EM = pickle.load(open("runs/160/model_%s_mlp.pkl" % exp, "rb")); EM.codes = T.codes; EM._st = {}
    QM = act164.QModel("runs/164/q_reach_%s.pt" % exp)
    RM = plan.RModel("runs/165/reach_%s.pt" % exp)
    return T, QM, EM, RM


def check(exp, n, names, s0=2000):
    """Exact binary reachability accuracy of RNet (r_arr and r_act) vs MiniGrid rules on unseen layouts."""
    T, QM, _, RM = load_all(exp); Tr = Truth(T.codes); res = {}
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); TA = told(t, T.codes)
        arr_cm = [0, 0, 0, 0]; act_cm = [0, 0, 0, 0]  # tp, fp, fn, tn
        for seed in range(s0, s0 + n):
            ep, ok = told_episode(w, TA, seed)
            if not ok:
                continue
            marks = [0] + [i + 1 for i in range(len(ep) - 1) if ep[i][4]]
            for i in marks:
                mp, pose, items = ep[i][:3]
                Tr.A.G = {}; Tr.A.D = {}; H = mp.shape[1]
                d, _ = Tr.A.paths(mp, items, (pose[0] * H + pose[1]) * 4 + pose[2])
                ra, ract = RM.query([(mp, pose[:2])])[0]
                for x in range(mp.shape[0]):
                    for y in range(mp.shape[1]):
                        if (x, y) == pose[:2]:
                            continue
                        tr_arr = any((x * H + y) * 4 + dd in d and d[(x * H + y) * 4 + dd] < agent.C_BLOCK for dd in range(4))
                        pr_arr = bool(ra[x, y] >= 0.5)
                        arr_cm[(0 if tr_arr and pr_arr else 1 if (not tr_arr) and pr_arr else 2 if tr_arr else 3)] += 1
                        tr_act = any(0 <= x - dx < mp.shape[0] and 0 <= y - dy < mp.shape[1] and
                                     any(((x - dx) * H + (y - dy)) * 4 + dd in d and d[((x - dx) * H + (y - dy)) * 4 + dd] < agent.C_BLOCK for dd in range(4))
                                     for dx, dy in DIRV)
                        pr_act = bool(ract[x, y] >= 0.5)
                        act_cm[(0 if tr_act and pr_act else 1 if (not tr_act) and pr_act else 2 if tr_act else 3)] += 1
        acc = lambda cm: 100.0 * (cm[0] + cm[3]) / max(sum(cm), 1)
        res[t] = dict(r_arr_acc=round(acc(arr_cm), 4), r_arr_cm=arr_cm, r_act_acc=round(acc(act_cm), 4), r_act_cm=act_cm)
        print("exp %s s0=%d %-20s  r_arr acc=%.2f%% (tp=%d fp=%d fn=%d tn=%d)  |  r_act acc=%.2f%% (tp=%d fp=%d fn=%d tn=%d)" %
              (exp, s0, t, acc(arr_cm), *arr_cm, acc(act_cm), *act_cm), flush=True)
    json.dump(res, open("runs/165/check_%s_%d.json" % (exp, s0), "w"), indent=1)


def run(arm, exp, split, names):
    T, QM, EM, RM = load_all(exp); Tr = Truth(T.codes)
    EM164 = net.load(exp, "A", T) if arm == "164" else None
    os.makedirs("runs/165", exist_ok=True)
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); ex, _ = evaluate.examples(t, T.codes); goals = agent.goal_events(ex)
        if arm == "164":
            os.environ["SURPRISE"] = "1"; A = act164.ArmA(QM, EM164, goals)
        else:
            A = plan.ZeroDijkstraAgent(QM, EM, RM, goals)
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
        print(arm, exp, split, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "failed"}, flush=True)
        json.dump(res, open("runs/165/run_%s_%s_%s_%s%s.json" % (arm, exp, split, t, os.environ.get("TAG", "")), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        s0 = int(os.environ.get("S0", 2000))
        check(sys.argv[2], int(sys.argv[3]), sys.argv[4:], s0=s0)
    elif cmd == "run":
        if sys.argv[2] in ("164", "165"):
            run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
        else:
            run("165", sys.argv[2], sys.argv[3], sys.argv[4:])
