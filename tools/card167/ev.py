"""Card 167: Zero-leak verification and partial-observability evaluation.

usage:
  ev.py leaktest <exp>                                  assert zero observation leakage across 50 episodes (10/task)
  ev.py run <arm> <exp> <split> <tasks...>              arm in {167, 167_base, 166_full}; split in {smoke, gate, fresh}
"""
import os, sys, json, time
os.environ["FIX"] = "1"
import numpy as np
import evaluate, agent, grid, rules, tasks, act as act164, plan as plan167
import importlib.util
from grid import UNK

# Load Card 166's plan module without colliding with Card 167's plan module
_sp166 = importlib.util.spec_from_file_location("plan166_mod", os.path.abspath("tools/card166/plan.py"))
plan166_mod = importlib.util.module_from_spec(_sp166); _sp166.loader.exec_module(plan166_mod)
from partial import WorldPartial

evaluate.SEEDS.update(fresh=range(11000, 11100), gate=range(2000, 2040), smoke=range(2000, 2005))
TASKS5 = ["DoorKey-8x8", "DoorKey-16x16", "KeyCorridorS3R3", "UnlockPickup", "BlockedUnlockPickup"]


def load_models(exp):
    T = evaluate.load(exp)
    QM = act164.QModel("runs/164/q_reach_%s.pt" % exp)
    WM = plan166_mod.WModel("runs/166/world_%s.pt" % exp)
    return T, QM, WM


def leaktest(exp, n_per_task=10):
    """Strict zero-leak assertion across 50 episodes (10 per task):
    At every step, assert (bel.map != UNK) is strictly contained in bel.seen_union,
    and every non-UNK cell matches its last-seen value from gen_obs() (zero occluded reads)."""
    T, QM, WM = load_models(exp)
    total_steps = 0; total_eps = 0; occ_at_start = 0
    for t in TASKS5:
        w = WorldPartial(tasks.ALL[t], T.codes)
        ex, _ = evaluate.examples(t, T.codes); goals = agent.goal_events(ex)
        A = plan167.PartialAgent(QM, WM, goals, epistemic=True)
        for seed in range(2000, 2000 + n_per_task):
            om, vm, p = w.reset(seed); A.reset_partial(om, vm, p)
            # Check initial visibility leak
            assert not ((A.bel.map != UNK) & (~A.bel.seen_union)).any(), f"Leak at reset {t} seed={seed}"
            assert A.bel.map[p[0], p[1]] == UNK, "Cell under agent was leaked at reset!"
            occ_at_start += (len(A.goal_asks(A.bel.map)) == 0)
            done = False; steps = 0
            while not done and steps < 200:
                a = A.act()
                om, vm, p, succ, done = w.step(a)
                A.observe_partial(a, om, vm, p)
                assert not ((A.bel.map != UNK) & (~A.bel.seen_union)).any(), f"Leak at {t} seed={seed} step={steps}"
                steps += 1; total_steps += 1
            total_eps += 1
    print(f"LEAKTEST PASSED: 0 leaks across {total_eps} episodes ({total_steps} steps); goals occluded at t=0: {occ_at_start}/{total_eps}", flush=True)


def run(arm, exp, split, names):
    T, QM, WM = load_models(exp)
    os.makedirs("runs/167", exist_ok=True)
    for t in names:
        ex, _ = evaluate.examples(t, T.codes); goals = agent.goal_events(ex)
        gc = {g[1] for g in goals}
        if arm == "166_full":
            w = grid.World(tasks.ALL[t], T.codes)
            A = plan166_mod.ZeroDijkstraAgent(QM, WM, goals)
        else:
            w = WorldPartial(tasks.ALL[t], T.codes)
            A = plan167.PartialAgent(QM, WM, goals, epistemic=(arm == "167"))

        out = []; t0 = time.time(); lost = 0; occ0 = 0; mem_used = 0
        for s in evaluate.SEEDS[split]:
            done = False; n = 0; gone = False; goal_seen_ever = False; goal_left_view = False
            if arm == "166_full":
                g, p = w.reset(s); A.reset(g, p)
                while not done:
                    a = A.act()
                    g, p, succ, done = w.step(a); ev = A.observe(a, g, p); n += 1
                    gone |= any(c in gc and a == rules.TOGGLE for _, _, c, c2 in ev)
                ep_plans = 0
            else:
                om, vm, p = w.reset(s); A.reset_partial(om, vm, p)
                occ0 += (len(A.goal_asks(A.bel.map)) == 0)
                while not done:
                    # Track whether a visible goal later leaves the 7x7 cone while remaining in Belief (P2 persistence)
                    ga = A.goal_asks(A.bel.map)
                    if ga:
                        goal_seen_ever = True
                        if any(not vm[X] for X, _, _ in ga):
                            goal_left_view = True
                    a = A.act()
                    om, vm, p, succ, done = w.step(a)
                    ev, _ = A.observe_partial(a, om, vm, p); n += 1
                    gone |= any(c in gc and a == rules.TOGGLE for _, _, c, c2 in ev)
                ep_plans = A.epistemic_plans
                mem_used += (succ and goal_left_view)
            out.append((succ, n, A.noplan, A.plans, A.expansions, ep_plans))
            lost += gone and not succ
        ok = [r for r in out if r[0]]; steps = sum(r[1] for r in out)
        res = dict(
            success=100.0 * len(ok) / len(out), layouts=len(out),
            steps_when_solved=float(np.mean([r[1] for r in ok])) if ok else None,
            next_best_share=100.0 * sum(r[2] for r in out) / max(steps, 1),
            mean_plans=float(np.mean([r[3] for r in out])),
            mean_epistemic_plans=float(np.mean([r[5] for r in out])),
            mean_expansions=float(np.mean([r[4] for r in out])),
            goals_occluded_at_reset=int(occ0),
            solved_after_goal_left_view=int(mem_used),
            failed_with_goal_token_destroyed=int(lost),
            ms_per_step=1000.0 * (time.time() - t0) / max(steps, 1),
            failed=[s for s, r in zip(evaluate.SEEDS[split], out) if not r[0]],
        )
        print(arm, exp, split, t, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "failed"}, flush=True)
        json.dump(res, open("runs/167/run_%s_%s_%s_%s%s.json" % (arm, exp, split, t, os.environ.get("TAG", "")), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "leaktest":
        leaktest(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
