"""Card 023: closer in view, exact check.

Card 012's exact conditions with "reachable by walking" replaced by
"approachable by moving closer in view": from here, repeatedly take a move
whose exact effect brings a target closer, where the target is the tile the
way's action acts on (the tile in front of a ready pose). Closeness is the
target's distance from the agent (tiles), then whether it is ahead (0),
beside (1) or behind (2). No path search, no step counts.

Runs, on the same exact discovery data: the control (card 012's conditions,
path search), the new conditions (rediscovered tree), and acting in 500 new
layouts with (a) control tree + path-search walking, (b) control tree +
moving closer, (c) new tree + moving closer.

Card 024 reuses the pipeline with `--measure step`: the target is the
ready pose itself (the tile to stand on and the way to face), and closeness
is the distance to that tile, then how many turns until the agent faces a
direction in which one step forward really brings it closer (a free tile).
Where no such direction exists the agent is stuck: a real obstacle.

Run: bin/prun python tools/card023/closer.py [--measure view|step] [--goals 16] [--start-share 0.5] [--out runs/...json]
"""
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld
from worldmodel.envs.keydoor import DIR_VEC

_Exact = dl.Exact


class PathExact(_Exact):
    """Card 012's exact goals (a path exists), with the stores moving closer needs."""

    def __init__(self, lay, parent, action):
        super().__init__(lay, parent, action)
        self.rsets, self.appr = {}, {}

MOVE_ORDER = (ld.FORWARD, ld.LEFT, ld.RIGHT)
MEASURE = "view"          # card 023; "step" is card 024's measure


def _turns(d, e):
    return min((e - d) % 4, (d - e) % 4)


def _free(lay, s, pos):
    c = ld.cell(lay, s, pos)
    return c == "empty" or (c == "door" and bool(s[6]))


def closeness(ex, s, rset):
    """Lowest closeness over the way's targets, seen from state s (lower is closer).

    view (card 023): target tiles = the tiles in front of ready poses;
      (tiles away, 0 ahead / 1 beside / 2 behind).
    step (card 024): targets = the ready poses; (tiles away from the pose's tile,
      turns until facing a direction where a step forward onto a free tile brings
      the agent closer; at the tile, turns until facing the pose's direction; 3 = none).
    """
    poses, fronts = rset
    x, y, d = s[:3]
    best = None
    if MEASURE == "view":
        fx, fy = DIR_VEC[d]
        for tx, ty in fronts:
            vx, vy = tx - x, ty - y
            ahead = vx * fx + vy * fy
            k = (abs(vx) + abs(vy), 0 if ahead > 0 else 1 if ahead == 0 else 2)
            if best is None or k < best:
                best = k
        return best
    closer_dirs = {}
    for px, py, pd in poses:
        dist = abs(px - x) + abs(py - y)
        if dist == 0:
            k = (0, _turns(d, pd))
        else:
            if (px, py) not in closer_dirs:
                closer_dirs[(px, py)] = [e for e in range(4)
                                         if abs(px - x - DIR_VEC[e][0]) + abs(py - y - DIR_VEC[e][1]) < dist
                                         and _free(ex.lay, s, (x + DIR_VEC[e][0], y + DIR_VEC[e][1]))]
            k = (dist, min((_turns(d, e) for e in closer_dirs[(px, py)]), default=3))
        if best is None or k < best:
            best = k
    return best


def target_tiles(rset):
    """The tiles closeness measures distance to."""
    poses, fronts = rset
    return fronts if MEASURE == "view" else sorted({(x, y) for x, y, _ in poses})


def ready_set(ex, n, rest):
    """Poses where way n's action achieves its parent, and the tiles those actions act on."""
    key = (n, rest)
    if key not in ex.rsets:
        lay, size = ex.lay, ex.lay.size
        p, a = ex.parent[n], ex.action[n]
        dummy = (0, 0, 0, *rest)
        poses = set()
        for x in range(1, size - 1):
            for y in range(1, size - 1):
                c = ld.cell(lay, dummy, (x, y))
                if (x, y) == lay.goal or not (c == "empty" or (c == "door" and rest[3])):
                    continue
                for d in range(4):
                    if ex.ready(p, a, (x, y, d, *rest)):
                        poses.add((x, y, d))
        ex.rsets[key] = (poses, sorted({ld.front(u) for u in poses}))
    return ex.rsets[key]


def closer_move(ex, n, s):
    """The first move (forward, left, right) whose exact effect brings way n's target closer; None if none."""
    rset = ready_set(ex, n, s[3:])
    if not rset[0]:
        return None
    here = closeness(ex, s, rset)
    for a in MOVE_ORDER:
        t = ld.step(ex.lay, s, a)[0]
        if (t[0], t[1]) == ex.lay.goal and t[:2] != s[:2]:
            continue                                  # stepping onto the goal square is not walking
        if closeness(ex, t, rset) < here:
            return a
    return None


def approach(ex, n, s, trace=None):
    """Does moving closer in view reach a pose where way n's action works? Memoised along the path."""
    seen, cur = [], s
    while True:
        k = (n, cur)
        if k in ex.appr:
            out, cur = ex.appr[k]
            break
        seen.append(k)
        poses, _ = ready_set(ex, n, cur[3:])
        if cur[:3] in poses:
            out = True
            break
        a = closer_move(ex, n, cur)
        if a is None:
            out = False
            break
        cur = ld.step(ex.lay, cur, a)[0]
    for k in seen:
        ex.appr[k] = (out, cur)                        # the result and where the walk ended
    if trace is not None:
        trace.append(cur)
    return out


class Approach(_Exact):
    """Card 012's exact goals with "reachable by walking" = "approachable by moving closer in view"."""

    def __init__(self, lay, parent, action):
        super().__init__(lay, parent, action)
        self.rsets, self.appr = {}, {}

    def _holds(self, n, s):
        if n == 0:
            return (s[0], s[1]) == self.lay.goal
        if self.holds(self.parent[n], s):
            return True
        return approach(self, n, s)


def stuck_cause(lay, ex, n, s):
    """Where a path exists but moving closer gets stuck: what is in the way."""
    end = []
    approach(ex, n, s, end)
    cur = end[0]
    rset = ready_set(ex, n, cur[3:])
    if not rset[0]:
        return "no ready pose"
    t = min(target_tiles(rset), key=lambda t: abs(t[0] - cur[0]) + abs(t[1] - cur[1]))
    if (cur[0] - lay.wall_x) * (t[0] - lay.wall_x) < 0:
        return "wall between"
    return "in front: " + ld.cell(lay, cur, ld.front(cur))


# ---------------------------------------------------------------- acting

def act(layouts, tree, persist, exact_cls, walker, budget=200, seed=0):
    """Exact conditions choose the way (card 012's chooser); walker 'path' or 'closer' picks moves."""
    rng = np.random.default_rng(seed)
    states = [ld.start_state(l) for l in layouts]
    exs = [exact_cls(l, tree.parent, tree.action) for l in layouts]
    done = np.zeros(len(layouts), bool)
    steps = np.full(len(layouts), budget)
    random_steps = 0
    for i in range(len(layouts)):
        ex = wex = exs[i]
        for step_i in range(budget):
            s = states[i]

            def R(n):
                return 1.0

            def W(n):
                if walker == "path":
                    return -ex.distances(n, s).get(s[:3], 10 ** 6)
                rset = ready_set(wex, n, s[3:])
                return -closeness(wex, s, rset)[0] if rset[0] else -10 ** 6

            c = dl.choose(tree, lambda n: ex.holds(n, s), R, W, persist)
            a = None
            if c is not None:
                g, aw = tree.parent[c], tree.action[c]
                if ex.ready(g, aw, s):
                    a = aw
                elif walker == "path":
                    a = ex.move_towards(c, s)
                else:
                    a = closer_move(wex, c, s)
            if a is None:
                a = int(rng.integers(len(dl.ACTIONS)))
                random_steps += 1
            states[i], end = ld.step(layouts[i], s, a)
            if end:
                done[i] = True
                steps[i] = step_i + 1
                break
    return {"success": round(float(done.mean()), 4),
            "mean_steps_when_successful": round(float(steps[done].mean()), 1) if done.any() else None,
            "random_steps": int(random_steps), "moves": int(steps.sum())}


def _act_job(job):
    layouts, rep, persist, kind, walker = job
    tree = dl.Tree.from_report(rep)
    return act(layouts, tree, persist, {"path": PathExact, "closer": Approach}[kind], walker)


def _shortest(job):
    """Exact shortest step counts to the goal square from the start (breadth-first over full states)."""
    out = []
    for lay in job:
        s0 = ld.start_state(lay)
        seen, frontier, dist = {s0}, [s0], 0
        found = None
        while frontier and found is None and dist < 400:
            nxt = []
            for s in frontier:
                for a in range(len(dl.ACTIONS)):
                    t, end = ld.step(lay, s, a)
                    if end:
                        found = dist + 1
                        break
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
                if found is not None:
                    break
            frontier, dist = nxt, dist + 1
        out.append(found)
    return out


def _gap(job):
    """Rows where the path condition holds and approachability does not, by way and cause."""
    layouts, parent, action, ep, states = job
    tree_n = len(parent)
    counts = Counter()
    cur = None
    for e, st in zip(ep, states):
        if e != cur:
            cur = e
            pe, ae = PathExact(layouts[e], parent, action), Approach(layouts[e], parent, action)
        s = dl.to_tup(st)
        for n in range(1, tree_n):
            p = parent[n]
            if pe.holds(p, s) or ae.holds(p, s):
                continue
            old, new = pe.holds(n, s), ae.holds(n, s)
            counts[(n, "path holds")] += int(old)
            if old and not new:
                counts[(n, stuck_cause(layouts[e], ae, n, s))] += 1
    return counts


def main():
    global MEASURE
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    MEASURE = args.get("--measure", "view")
    dl.MAX_GOALS = int(args.get("--goals", dl.MAX_GOALS))
    dl.START_SHARE = float(args.get("--start-share", dl.START_SHARE))
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    out = Path(args.get("--out", "runs/023_closer.json"))
    res = {"measure": MEASURE, "goal_budget": dl.MAX_GOALS, "leaf_if_on_at_start_above": dl.START_SHARE}
    log(f"closeness measure: {MEASURE}; goal budget: {dl.MAX_GOALS}; leaf if on at start above {dl.START_SHARE}")
    pool = mp.get_context("fork").Pool(20)
    layouts, tr, seq, st0 = dl.collect(pool, "key", 5000, 11, 0.0, seq_episodes=300)
    log(f"data: {len(layouts)} episodes, {len(tr['act'])} rows, {len(np.unique(seq['ep']))} sequence episodes")
    pool.close()

    trees = {}
    for kind, cls in (("path", PathExact), ("closer", Approach)):
        dl.Exact = cls                                  # the workers read the module's Exact
        pool = mp.get_context("fork").Pool(20)
        t0 = time.monotonic()
        log(f"exact discovery, conditions: {kind}")
        tree, persist = dl.run_exact(pool, layouts, tr, seq, st0, log, 5000, 300)
        pool.close()
        dl.Exact = _Exact
        trees[kind] = (tree, persist)
        res[f"tree_{kind}"] = {"ways": [(r["path"], r["status"]) for r in tree.report()],
                               "structure": dl.structure_check("key", tree),
                               "seconds": round(time.monotonic() - t0, 1)}
        log(f"tree ({kind}): {json.dumps(res[f'tree_{kind}'])}")
        out.write_text(json.dumps(res, indent=1) + "\n")

    # data check: where the control tree's path condition holds but moving closer gets stuck
    pool = mp.get_context("fork").Pool(20)
    tree_p = trees["path"][0]
    ep, st = seq["ep"], seq["st"]
    chunks = np.array_split(np.arange(len(ep)), 40)
    parts = pool.map(_gap, [({int(e): layouts[e] for e in np.unique(ep[c])}, tree_p.parent, tree_p.action,
                             ep[c], st[c]) for c in chunks])
    gap = Counter()
    for p in parts:
        gap.update(p)
    res["data_check_gap"] = {"/".join(tree_p.path(n)): {k: v for (m, k), v in sorted(gap.items()) if m == n}
                             for n in range(1, len(tree_p.parent))}
    log(f"RESULT data check (sequence frames, parent condition off): {json.dumps(res['data_check_gap'])}")

    # acting: 500 new layouts
    rng = np.random.default_rng(777)
    test = [ld.make_layout(8, "key", rng) for _ in range(500)]
    parts = pool.map(_shortest, [list(c) for c in np.array_split(np.array(test, dtype=object), 40)])
    shortest = np.array([v if v is not None else -1 for p in parts for v in p])
    res["shortest_mean_steps"] = round(float(shortest[shortest > 0].mean()), 1)
    arms = {"control: path conditions, path-search walking": ("path", "path", "path"),
            "baseline: path conditions, moving closer": ("path", "path", "closer"),
            "new: approachable conditions (rediscovered), moving closer": ("closer", "closer", "closer")}
    res["acting"] = {}
    for name, (tk, kind, walker) in arms.items():
        tree, persist = trees[tk]
        rep = tree.report()
        chunks = np.array_split(np.arange(len(test)), 40)
        parts = pool.map(_act_job, [([test[i] for i in c], rep, list(persist), kind, walker) for c in chunks])
        n = np.array([len(c) for c in chunks])
        succ = np.array([p["success"] for p in parts])
        ms = [(p["mean_steps_when_successful"], p["success"] * len(c)) for p, c in zip(parts, chunks) if p["success"]]
        r = {"success": round(float((succ * n).sum() / n.sum()), 4),
             "mean_steps_when_successful": round(sum(m * w for m, w in ms) / max(sum(w for _, w in ms), 1), 1),
             "random_steps": int(sum(p["random_steps"] for p in parts)),
             "moves": int(sum(p["moves"] for p in parts))}
        r["random_share"] = round(r["random_steps"] / max(r["moves"], 1), 4)
        res["acting"][name] = r
        log(f"RESULT acting, {name}: {json.dumps(r)}")
        out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"shortest route to the goal from the start, mean over solvable layouts: {res['shortest_mean_steps']}")
    res["seconds"] = round(time.monotonic() - t00, 1)
    out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
