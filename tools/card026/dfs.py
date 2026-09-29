"""Card 026: depth-first subgoals, exact check.

Card 025's conditions ("approachable by moving closer", card 024's step
measure) and acting, with the tree grown only where acting needs it. At each
move the search starts at the goal square: if one of the goal's ways is
possible now (its condition holds), in evidence order, act on the first;
otherwise go down into its ways one at a time, depth first, backing up when
nothing below a way is possible (to depth 6). A goal's ways are discovered
(card 012's evidence test and self-check) the first time the search reaches
it, then kept. No goal budget, no "on at start" leaf rule.

Batched in rounds: a round replays all layouts from the start with the goals
expanded so far, and a layout pauses at the first goal its search reaches
that has not been expanded. The paused goals are expanded from the
experience and the next round replays. Play up to a pause is exactly what
discovery on demand would do, so the goals expanded are exactly the ones it
would expand; the last round (no pauses) is its result.

Also reruns, on the same layouts, the control (card 012's conditions,
path-search walking) and card 025's tree, both rebuilt from
runs/025_tree.json, counting the conditions each move checks.

Run: bin/prun python tools/card026/dfs.py [--episodes 5000] [--layouts 500] [--out runs/026_dfs.json]
"""
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card023"))
import closer  # noqa: E402

from worldmodel import discover_logic as dl  # noqa: E402
from worldmodel.envs import logicdoor as ld  # noqa: E402

BUDGET = 200


def pool20():
    return mp.get_context("fork").Pool(20)


# ---------------------------------------------------------------- the search

def dfs(kids, depth, expanded, on, g, checks):
    """First way possible now at or below goal g, depth first.

    Returns (way, None), (None, goal) when the search reaches a goal not yet
    expanded, or (None, None) when nothing below g is possible."""
    if g not in expanded:
        return None, g
    ways = kids[g]
    for c in ways:
        checks[0] += 1
        if on(c):
            return c, None
    for c in ways:
        if depth[c] < dl.MAX_DEPTH:
            w, need = dfs(kids, depth, expanded, on, c, checks)
            if w is not None or need is not None:
                return w, need
    return None, None


def _dfs_job(job):
    idx, layouts, parent, action, depth, kids, expanded = job
    out = []
    for i, lay in zip(idx, layouts):
        rng = np.random.default_rng(1000 + int(i))
        ex = closer.Approach(lay, parent, action)
        s = ld.start_state(lay)
        rec = {"done": False, "steps": BUDGET, "pause": None, "checks": [], "used": Counter(),
               "random_no_way": 0, "random_no_move": 0}
        for t in range(BUDGET):
            checks = [0]
            c, need = dfs(kids, depth, expanded, lambda n: ex.holds(n, s), 0, checks)
            if need is not None:
                rec["pause"] = need
                break
            rec["checks"].append(checks[0])
            a = None
            if c is None:
                rec["random_no_way"] += 1
            else:
                rec["used"][c] += 1
                g, aw = parent[c], action[c]
                a = aw if ex.ready(g, aw, s) else closer.closer_move(ex, c, s)
                if a is None:
                    rec["random_no_move"] += 1
            if a is None:
                a = int(rng.integers(len(dl.ACTIONS)))
            s, end = ld.step(lay, s, a)
            if end:
                rec["done"], rec["steps"] = True, t + 1
                break
        out.append(rec)
    return out


def play(pool, test, tree, expanded):
    kids = {g: tree.children(g) for g in expanded}
    chunks = [c.tolist() for c in np.array_split(np.arange(len(test)), 40) if len(c)]
    parts = pool.map(_dfs_job, [(c, [test[i] for i in c], tree.parent, tree.action, tree.depth, kids,
                                 set(expanded)) for c in chunks])
    return [r for p in parts for r in p]


# ---------------------------------------------------------------- discovery on demand

def expand_goals(pool, tree, G, expanded, layouts, tr, seq, st0, log):
    """Card 012's evidence test and self-check for goals G (run_exact's steps, one batch)."""
    act, w0 = tr["act"], tr["w0"]
    L0, L1 = dl.exact_labels(pool, layouts, tree.parent, tree.action, tr["ep"], [tr["s0"], tr["s1"]], G)

    def gains_of(g):
        m = ~L0[:, g]
        return dl.way_gains(act[m], L1[m, g], L1[m, g], w0[m])

    new = dl.expand(tree, G, gains_of, log)
    expanded.update(G)
    if not new:
        return
    need = sorted(set(new) | {tree.parent[c] for c in new})
    Ls, = dl.exact_labels(pool, layouts, tree.parent, tree.action, seq["ep"], [seq["st"]], need)
    Lst, = dl.exact_labels(pool, layouts, tree.parent, tree.action, st0["ep"], [st0["st"]], new)

    def check(c):
        g, a = tree.parent[c], tree.action[c]
        y = dl.walk_then_act(seq["act"], seq["ep"], Ls[:, g], a)
        m = ~Ls[:, g]
        return dl.check_gain(Ls[m, c], y[m]), {"rate_on": round(float(y[m & Ls[:, c]].mean()), 4),
                                               "rate_off": round(float(y[m & ~Ls[:, c]].mean()), 4)}

    dl.settle(tree, new, lambda c: Lst[:, c].mean(), check, log)
    for c in new:
        if tree.status[c] == "frontier":
            tree.status[c] = "not expanded"


# ---------------------------------------------------------------- card 025's arms, rerun

def _act025_job(job):
    """closer._act_job with card 012's chooser counting the conditions it checks each move."""
    checks, orig = [], dl.choose

    def counting(tree, on, R, W, persist):
        n = [0]

        def on2(c):
            n[0] += 1
            return on(c)

        r = orig(tree, on2, R, W, persist)
        checks.append(n[0])
        return r

    dl.choose = counting
    try:
        r = closer._act_job(job)
    finally:
        dl.choose = orig
    r["checks_sum"], r["checks_max"], r["checks_n"] = sum(checks), max(checks, default=0), len(checks)
    return r


def rerun_025(old, name, layouts, seq, test, log):
    tk, cls, kind, walker = {"control": ("tree_path", closer.PathExact, "path", "path"),
                             "card 025": ("tree_closer", closer.Approach, "closer", "closer")}[name]
    rep = [{"path": p, "status": s, "way_evidence_nats": None, "self_check": None} for p, s in old[tk]["ways"]]
    tree = dl.Tree.from_report(rep)
    dl.Exact = cls                                  # the workers read the module's Exact
    pool = pool20()
    Ls, = dl.exact_labels(pool, layouts, tree.parent, tree.action, seq["ep"], [seq["st"]])
    persist = [dl.persistence(Ls[:, n], seq["ep"]) for n in range(len(tree.parent))]
    chunks = [c for c in np.array_split(np.arange(len(test)), 40) if len(c)]
    parts = pool.map(_act025_job, [([test[i] for i in c], rep, persist, kind, walker) for c in chunks])
    pool.close()
    dl.Exact = closer._Exact
    n = np.array([len(c) for c in chunks])
    succ = np.array([p["success"] for p in parts])
    ms = [(p["mean_steps_when_successful"], p["success"] * len(c)) for p, c in zip(parts, chunks) if p["success"]]
    r = {"success": round(float((succ * n).sum() / n.sum()), 4),
         "mean_steps_when_successful": round(sum(m * w for m, w in ms) / max(sum(w for _, w in ms), 1), 1),
         "random_steps": int(sum(p["random_steps"] for p in parts)),
         "moves": int(sum(p["moves"] for p in parts)),
         "tree_nodes": len(tree.parent),
         "checks_per_move_mean": round(sum(p["checks_sum"] for p in parts) / max(sum(p["checks_n"] for p in parts), 1), 2),
         "checks_per_move_max": int(max(p["checks_max"] for p in parts))}
    r["random_share"] = round(r["random_steps"] / max(r["moves"], 1), 4)
    log(f"RESULT acting, {name}: {json.dumps(r)}")
    return r


# ---------------------------------------------------------------- main

def summary(recs, tree):
    done = np.array([r["done"] for r in recs])
    steps = np.array([r["steps"] for r in recs])
    rnd = sum(r["random_no_way"] + r["random_no_move"] for r in recs)
    checks = [c for r in recs for c in r["checks"]]
    used = Counter()
    for r in recs:
        used.update(r["used"])
    return {"success": round(float(done.mean()), 4),
            "mean_steps_when_successful": round(float(steps[done].mean()), 1) if done.any() else None,
            "random_steps": int(rnd), "moves": int(steps.sum()), "random_share": round(rnd / max(int(steps.sum()), 1), 4),
            "random_no_way": int(sum(r["random_no_way"] for r in recs)),
            "random_way_but_no_move": int(sum(r["random_no_move"] for r in recs)),
            "layouts_with_random_moves": int(sum(r["random_no_way"] + r["random_no_move"] > 0 for r in recs)),
            "checks_per_move_mean": round(float(np.mean(checks)), 2) if checks else None,
            "checks_per_move_max": int(max(checks, default=0)),
            "ways_used_steps": {"/".join(tree.path(c)) or "(goal square)": int(k) for c, k in used.most_common()}}


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    episodes, n_test = int(args.get("--episodes", 5000)), int(args.get("--layouts", 500))
    out = Path(args.get("--out", "runs/026_dfs.json"))
    closer.MEASURE = "step"                          # card 024's measure, read by the workers
    dl.MAX_GOALS = 10 ** 9                           # no goal budget
    dl.START_SHARE = 2.0                             # no "on at start" leaf rule
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    res = {"note": "Card 026, tools/card026/dfs.py, exact computation, key world.",
           "episodes": episodes, "layouts": n_test, "max_depth": dl.MAX_DEPTH}

    pool = pool20()
    layouts, tr, seq, st0 = dl.collect(pool, "key", episodes, 11, 0.0, seq_episodes=300)
    pool.close()
    log(f"data: {len(layouts)} episodes, {len(tr['act'])} rows, {len(np.unique(seq['ep']))} sequence episodes")
    rng = np.random.default_rng(777)
    test = [ld.make_layout(8, "key", rng) for _ in range(n_test)]

    old = json.loads(Path("runs/025_tree.json").read_text())
    res["acting_rerun"] = {name: rerun_025(old, name, layouts, seq, test, log) for name in ("control", "card 025")}
    out.write_text(json.dumps(res, indent=1) + "\n")

    dl.Exact = closer.Approach                       # the workers read the module's Exact
    pool = pool20()
    tree, expanded, rounds = dl.Tree(), set(), []
    tree.status[0] = "not expanded"
    while True:
        t0 = time.monotonic()
        recs = play(pool, test, tree, expanded)
        paused = Counter(r["pause"] for r in recs if r["pause"] is not None)
        rnd = {"round": len(rounds), "goals_expanded_before": len(expanded), "tree_nodes_before": len(tree.parent),
               "layouts_paused": int(sum(paused.values())),
               "goals_needed": {"/".join(tree.path(g)) or "(goal square)": int(k) for g, k in sorted(paused.items())}}
        log(f"round {rnd['round']}: {json.dumps(rnd)}")
        if not paused or len(rounds) >= 40:
            rnd["seconds"] = round(time.monotonic() - t0, 1)
            rounds.append(rnd)
            break
        expand_goals(pool, tree, sorted(paused), expanded, layouts, tr, seq, st0, log)
        rnd["seconds"] = round(time.monotonic() - t0, 1)
        rounds.append(rnd)
    pool.close()

    rep = tree.report()
    res["rounds"] = rounds
    res["tree"] = rep
    res["tree_nodes"] = len(tree.parent)
    res["accepted_ways"] = sum(1 for n in range(1, len(tree.parent)) if tree.status[n] != "rejected by self-check")
    res["rejected_ways"] = sum(1 for n in range(1, len(tree.parent)) if tree.status[n] == "rejected by self-check")
    res["goals_expanded"] = len(expanded)
    res["acting"] = summary(recs, tree)
    res["complete"] = not any(r["pause"] is not None for r in recs)
    log("tree: " + json.dumps([(r["path"], r["status"]) for r in rep]))
    log(f"RESULT depth-first acting (last round): {json.dumps(res['acting'])}")
    log(f"RESULT conditions: {res['tree_nodes']} tree nodes ({res['accepted_ways']} accepted ways, "
        f"{res['rejected_ways']} rejected), {res['goals_expanded']} goals expanded, {len(rounds)} rounds")
    res["seconds"] = round(time.monotonic() - t00, 1)
    out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
