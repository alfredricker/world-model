"""Card 166: diagnostics and evaluation for Unified Factored Relational World Model (WNet).

usage:
  ev.py check <exp> <n> <tasks...>            score WNet reachability (r_arr/r_act) AND live cell-effect accuracy vs MiniGrid rules
  ev.py run <arm> <exp> <split> <tasks...>    arm in {166, 166_abl, 165}; split in {smoke, gate, fresh}
"""
import os, sys, json, time, pickle
os.environ["FIX"] = "1"
import numpy as np
import evaluate, agent, grid, rules, tasks, act as act164, plan as plan166
from grid import DIRV, NA, UNK, OUT, gid

evaluate.SEEDS.update(fresh=range(10000, 10100), fresh165=range(9000, 9100), gate=range(2000, 2040), smoke=range(2000, 2005))


def told(task, codes):
    return agent.Agent(rules.Rules(codes), tasks.told_goal(task, codes))


def told_episode(w, A, seed, cap=400):
    g, p = w.reset(seed); A.reset(g, p); done = False; out = []; succ = False
    while not done and len(out) < cap:
        a = A.act(); st = (A.bel.map, A.bel.pose, A.bel.items()); g, p, succ, done = w.step(a); ev = A.observe(a, g, p); out.append(st + (a, ev, p))
    return out, succ


class Truth:
    """MiniGrid's rules, for scoring walk steps, binary reachability, and 1-step cell effects."""
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


def load_166(exp, ablate=False):
    T = evaluate.load(exp)
    QM = act164.QModel("runs/164/q_reach_%s.pt" % exp)
    WM = plan166.WModel("runs/166/world_%s.pt" % exp, ablate_effects=ablate)
    return T, QM, WM


def check(exp, n, names, s0=2000):
    """Score WNet reachability (r_arr/r_act) AND live 1-step cell-effect accuracy (including counterfactual contrasts) vs MiniGrid rules."""
    T, QM, WM = load_166(exp); Tr = Truth(T.codes); res = {}
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); TA = told(t, T.codes)
        arr_cm = [0, 0, 0, 0]; act_cm = [0, 0, 0, 0]  # tp, fp, fn, tn
        cell_rows = []; cell_trues = []; cell_is_chg = []
        for seed in range(s0, s0 + n):
            ep, ok = told_episode(w, TA, seed)
            if not ok:
                continue
            marks = [0] + [i + 1 for i in range(len(ep) - 1) if ep[i][4]]
            for i in marks:
                mp, pose, items = ep[i][:3]
                Tr.A.G = {}; Tr.A.D = {}; H = mp.shape[1]
                d, _ = Tr.A.paths(mp, items, (pose[0] * H + pose[1]) * 4 + pose[2])
                ra, ract = WM.query_reach([(mp, pose[:2])])[0]
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
            # Collect live 1-step cell-effect queries at every step of the episode (front cell + counterfactual actions/items!)
            for mp, pose, items, a_taken, ev, pose2 in ep:
                W, H = mp.shape; px, py, pd = pose; fx, fy = px + DIRV[pd][0], py + DIRV[pd][1]
                if 0 <= fx < W and 0 <= fy < H and mp[fx, fy] not in (UNK, OUT):
                    c = int(mp[fx, fy])
                    # Test the action taken AND all interactive actions (pickup=3, drop=4, toggle=5) + counterfactual empty items
                    for a_q in (3, 4, 5):
                        for it_q in (items, frozenset()):
                            tr_o = Tr.R.cell(c, a_q, gid(fx - px, fy - py, pd), it_q)
                            cell_rows.append((c, a_q, fx - px, fy - py, pd, it_q))
                            cell_trues.append(int(tr_o))
                            cell_is_chg.append(int(tr_o != c))
        probs = WM.predict_batch(cell_rows)
        preds = probs.argmax(1)
        trues = np.array(cell_trues, np.int64)
        ischg = np.array(cell_is_chg, bool)
        cell_acc = 100.0 * (preds == trues).mean() if len(trues) else 0.0
        chg_rec = 100.0 * (preds[ischg] == trues[ischg]).mean() if ischg.any() else 100.0
        unchg_spec = 100.0 * (preds[~ischg] == trues[~ischg]).mean() if (~ischg).any() else 100.0
        acc = lambda cm: 100.0 * (cm[0] + cm[3]) / max(sum(cm), 1)
        res[t] = dict(
            r_arr_acc=round(acc(arr_cm), 4), r_arr_cm=arr_cm,
            r_act_acc=round(acc(act_cm), 4), r_act_cm=act_cm,
            cell_acc=round(float(cell_acc), 4),
            cell_chg_rec=round(float(chg_rec), 4), cell_n_chg=int(ischg.sum()),
            cell_unchg_spec=round(float(unchg_spec), 4), cell_n_unchg=int((~ischg).sum()),
        )
        print("exp %s s0=%d %-20s  r_arr=%.2f%% r_act=%.2f%% | cell=%.2f%% (chg=%.2f%% [%d], unchg=%.2f%% [%d])" %
              (exp, s0, t, acc(arr_cm), acc(act_cm), cell_acc, chg_rec, int(ischg.sum()), unchg_spec, int((~ischg).sum())), flush=True)
    os.makedirs("runs/166", exist_ok=True)
    json.dump(res, open("runs/166/check_%s_%d.json" % (exp, s0), "w"), indent=1)


def run(arm, exp, split, names):
    T = evaluate.load(exp); Tr = Truth(T.codes)
    QM = act164.QModel("runs/164/q_reach_%s.pt" % exp)
    if arm == "165":
        import importlib.util
        sys.path.insert(0, os.path.abspath("tools/card165"))
        sp = importlib.util.spec_from_file_location("plan165", os.path.abspath("tools/card165/plan.py"))
        plan165 = importlib.util.module_from_spec(sp); sp.loader.exec_module(plan165)
        EM = pickle.load(open("runs/160/model_%s_mlp.pkl" % exp, "rb")); EM.codes = T.codes; EM._st = {}
        RM = plan165.RModel("runs/165/reach_%s.pt" % exp)
        make_agent = lambda goals: plan165.ZeroDijkstraAgent(QM, EM, RM, goals)
    else:
        WM = plan166.WModel("runs/166/world_%s.pt" % exp, ablate_effects=(arm == "166_abl"))
        make_agent = lambda goals: plan166.ZeroDijkstraAgent(QM, WM, goals)
    os.makedirs("runs/166", exist_ok=True)
    for t in names:
        w = grid.World(tasks.ALL[t], T.codes); ex, _ = evaluate.examples(t, T.codes); goals = agent.goal_events(ex)
        A = make_agent(goals)
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
        json.dump(res, open("runs/166/run_%s_%s_%s_%s%s.json" % (arm, exp, split, t, os.environ.get("TAG", "")), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        s0 = int(os.environ.get("S0", 2000))
        check(sys.argv[2], int(sys.argv[3]), sys.argv[4:], s0=s0)
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
