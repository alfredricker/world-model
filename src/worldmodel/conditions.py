"""Card 003: check the theory of conditions with exact computation.

Random play in the key-door room, recorded as symbolic states. Two ways of
finding a goal's conditions are compared:

- (b) achievement conditions: at the steps where an action makes the goal
  true, versus the same action failing, find the smallest set of variable
  values under which the action succeeds with probability >= 0.9, scored per
  member by removal (sequential covering gives one rule per way of
  succeeding);
- (a) long-range jumps: the exact chance that random play reaches the goal
  within H steps, followed along successful episodes; which kinds of step
  make it jump.

Evaluator machinery only (C1): the variables are the simulator's.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .envs.keydoor import (CARRY, PICKUP, TOGGLE, FORWARD, Layout, cell, front, make_layout,
                           start_state, step, variables)

GOALS = {
    "door_unlocked": lambda s: s[4] != 0,
    "door_open": lambda s: s[4] == 2,
    "holding_matching_key": lambda s: s[3] == 1,
    "in_doorway": None,       # positional goals, to follow the chain below the goal square
    "in_right_room": None,
    "on_goal": None,          # the episode ends on entering the goal square
}
ACTION_NAMES = ("left", "right", "forward", "pickup", "toggle")


def holds(goal: str, layout: Layout, state: tuple) -> bool:
    if goal == "on_goal":
        return (state[0], state[1]) == layout.goal
    if goal == "in_doorway":
        return state[0] == layout.wall_x
    if goal == "in_right_room":
        return state[0] > layout.wall_x
    return GOALS[goal](state)


# ---------------------------------------------------------------- data

def collect(size: int, episodes: int, max_steps: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        layout = make_layout(size, rng)
        state = start_state(layout)
        states, actions = [state], []
        for a in rng.integers(5, size=max_steps):
            state, done = step(layout, state, int(a))
            states.append(state)
            actions.append(int(a))
            if done:
                break
        out.append({"layout": layout, "states": states, "actions": actions})
    return out


def kind(layout: Layout, s: tuple, a: int, t: tuple) -> str:
    """What a transition did, for reporting."""
    if t[3] != s[3]:
        return "pickup_" + CARRY[t[3]]
    if t[4] != s[4]:
        return {(0, 2): "unlock", (1, 2): "door_reopen", (2, 1): "door_close"}[(s[4], t[4])]
    if (t[0], t[1]) == layout.goal:
        return "enter_goal"
    if t != s:
        return "move"
    if a == TOGGLE:
        return "toggle_failed_locked" if cell(layout, s, front(s)) == "door" else "toggle_nothing"
    if a == PICKUP:
        return "pickup_failed_hands_full" if cell(layout, s, front(s)) in ("key", "distractor") else "pickup_nothing"
    return "bump"


def data_check(data: list[dict]) -> dict:
    counts = Counter()
    firsts = Counter()
    for ep in data:
        lay, S, A = ep["layout"], ep["states"], ep["actions"]
        seen = set()
        for s, a, t in zip(S, A, S[1:]):
            k = kind(lay, s, a, t)
            counts[k] += 1
            if k not in seen:
                firsts[k] += 1
                seen.add(k)
    n = len(data)
    return {"episodes": n, "steps": int(sum(len(e["actions"]) for e in data)),
            "events_per_1000_episodes": {k: round(1000 * v / n, 1) for k, v in sorted(counts.items())},
            "episodes_with_event_per_1000": {k: round(1000 * v / n, 1) for k, v in sorted(firsts.items())}}


# ---------------------------------------------------------------- method (b)

def achievement_rows(data: list[dict], goal: str):
    """Every step where the goal did not hold before: (variables, action, achieved)."""
    rows = []
    for ep in data:
        lay, S, A = ep["layout"], ep["states"], ep["actions"]
        for s, a, t in zip(S, A, S[1:]):
            if not holds(goal, lay, s):
                rows.append((variables(lay, s), a, holds(goal, lay, t)))
    return rows


def find_rules(rows, precision_target=0.9, min_share=0.05, keep=0.95, min_removal=0.1,
               action_names=ACTION_NAMES) -> list[dict]:
    """Sequential covering over conjunctions of variable=value atoms."""
    total = sum(r[2] for r in rows)
    if total == 0:
        return []
    by_action = defaultdict(list)
    for v, a, y in rows:
        by_action[a].append((v, y))
    matrices = {}
    for a, rs in by_action.items():
        atoms = sorted({(k, val) for v, _ in rs for k, val in v.items()}, key=str)
        X = np.array([[v[k] == val for k, val in atoms] for v, _ in rs], bool)
        y = np.array([r[1] for r in rs], bool)
        matrices[a] = (atoms, X, y)
    remaining = {a: m[2].copy() for a, m in matrices.items()}
    rules = []
    while sum(r.sum() for r in remaining.values()) >= max(min_share * total, 1):
        a = max(remaining, key=lambda k: remaining[k].sum())   # action with most uncovered successes
        atoms, X, y = matrices[a]
        rem = remaining[a]
        chosen, mask = [], np.ones(len(y), bool)
        while y[mask].mean() < precision_target:
            base = (rem & mask).sum()
            best, best_prec = None, y[mask].mean()
            for j in range(len(atoms)):
                if j in chosen:
                    continue
                m = mask & X[:, j]
                if (rem & m).sum() < keep * base or m.sum() == 0:
                    continue
                p = y[m].mean()
                if p > best_prec + 1e-9:
                    best, best_prec = j, p
            if best is None:
                break
            chosen.append(best)
            mask &= X[:, best]

        def prec(members, absent=None):
            m = np.ones(len(y), bool)
            for j in members:
                m &= X[:, j]
            if absent is not None:
                m &= ~X[:, absent]
            return float(y[m].mean()) if m.any() else float("nan")

        def removal(j):
            """Achievement probability with every member, minus with j absent
            and the others present. No attempts with j absent: 0 (redundant
            or untestable), so pruning drops it."""
            without = prec([i for i in chosen if i != j], absent=j)
            return 0.0 if np.isnan(without) else prec(chosen) - without

        # Drop members whose absence barely changes the achievement probability.
        while chosen:
            scores = {j: removal(j) for j in chosen}
            j = min(scores, key=scores.get)
            if scores[j] >= min_removal:
                break
            chosen.remove(j)
        m = np.ones(len(y), bool)
        for j in chosen:
            m &= X[:, j]
        covered = rem & m
        if covered.sum() == 0:
            break
        rules.append({
            "action": action_names[a],
            "conditions": {f"{atoms[j][0]}={atoms[j][1]}": round(removal(j), 3) for j in chosen},
            "achievement_probability": round(prec(chosen), 3),
            "attempts": int(m.sum()),
            "share_of_successes": round(covered.sum() / total, 3),
            "without_any_condition": round(float(y.mean()), 4),
        })
        remaining[a] = rem & ~m
    return rules


def probability_under(rows, action: int, required: dict) -> tuple[float, int]:
    """Achievement probability of an action when the variables take the given values."""
    hits = [y for v, a, y in rows if a == action and all(v[k] == val for k, val in required.items())]
    return (float(np.mean(hits)) if hits else float("nan")), len(hits)


# ---------------------------------------------------------------- method (a)

@lru_cache(maxsize=4096)
def state_graph(layout: Layout):
    """Every state reachable from the layout's start, and the next-state table."""
    start = start_state(layout)
    index, states, nxt = {start: 0}, [start], []
    i = 0
    while i < len(states):
        row = []
        for a in range(5):
            t, _ = step(layout, states[i], a)
            if t not in index:
                index[t] = len(states)
                states.append(t)
            row.append(index[t])
        nxt.append(row)
        i += 1
    return index, states, np.array(nxt)


def exact_chance(layout: Layout, goal: str, horizon: int):
    """Chance that uniform random play reaches the goal within `horizon` steps,
    for every state reachable from the layout's start."""
    index, states, nxt = state_graph(layout)
    success = np.array([holds(goal, layout, s) for s in states])
    # Entering the goal square ends the episode: from there nothing else can be reached.
    dead = np.array([(s[0], s[1]) == layout.goal for s in states]) & ~success
    v = success.astype(float)
    for _ in range(horizon):
        v = np.where(success, 1.0, np.where(dead, 0.0, v[nxt].mean(axis=1)))
    return index, v


def jumps(data: list[dict], goal: str, horizon: int) -> dict:
    """Along episodes that reach the goal: rise in the exact chance per kind of step."""
    rise = defaultdict(list)
    top = Counter()
    start_chance = []
    for ep in data:
        lay, S, A = ep["layout"], ep["states"], ep["actions"]
        hit = next((t for t, s in enumerate(S) if holds(goal, lay, s)), None)
        if hit == 0:
            continue
        index, v = exact_chance(lay, goal, horizon)
        end = hit if hit is not None else len(S) - 1
        vals = [v[index[s]] for s in S[:end + 1]]
        start_chance.append(vals[0])
        best, best_kind = -1.0, None
        for t in range(end - 1 if hit is not None else end):   # exclude the achieving step itself
            k = kind(lay, S[t], A[t], S[t + 1])
            d = vals[t + 1] - vals[t]
            rise[k].append(d)
            if d > best:
                best, best_kind = d, k
        if hit is not None and best_kind is not None:
            top[best_kind] += 1
    n = sum(top.values())
    return {"horizon": horizon, "successful_episodes": n,
            "median_chance_at_start": round(float(np.median(start_chance)), 3) if start_chance else None,
            "mean_rise_by_kind": {k: {"mean": round(float(np.mean(d)), 4), "max": round(float(np.max(d)), 4), "n": len(d)}
                                  for k, d in sorted(rise.items())},
            "share_of_largest_rise_in_successful_episodes": {k: round(c / n, 3) for k, c in top.most_common()}}


# ---------------------------------------------------------------- baseline

def lift_baseline(data: list[dict], goal: str, window: int = 10, top: int = 6) -> list:
    """Variable values more frequent in the steps just before success than overall."""
    near, overall = Counter(), Counter()
    n_near = n_all = 0
    for ep in data:
        lay, S = ep["layout"], ep["states"]
        hit = next((t for t, s in enumerate(S) if holds(goal, lay, s)), None)
        for t, s in enumerate(S[:hit if hit is not None else len(S)]):
            items = variables(lay, s).items()
            overall.update(items)
            n_all += 1
            if hit is not None and hit - window <= t < hit:
                near.update(items)
                n_near += 1
    lift = {f"{k}={v}": (near[(k, v)] / n_near) / (overall[(k, v)] / n_all)
            for (k, v) in overall if near[(k, v)] >= 0.2 * n_near}
    return sorted(((k, round(l, 2)) for k, l in lift.items()), key=lambda x: -x[1])[:top]


# ---------------------------------------------------------------- sample size

def successes_curve(data, goal, expected: dict, counts=(10, 30, 100, 300), repeats=20, seed=0):
    """How often the first rule is exactly the expected one, with few successes."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in counts:
        right = done = 0
        for _ in range(repeats):
            order = rng.permutation(len(data))
            chosen, got = [], 0
            for i in order:
                chosen.append(data[i])
                got += any(holds(goal, data[i]["layout"], s) for s in data[i]["states"][1:])
                if got >= n:
                    break
            if got < n:
                continue
            rules = find_rules(achievement_rows(chosen, goal))
            done += 1
            right += bool(rules) and rules[0]["action"] == expected["action"] and \
                set(rules[0]["conditions"]) == set(expected["conditions"])
        out[n] = {"exact_recovery": round(right / done, 2) if done else None, "repeats": done}
    return out


# ---------------------------------------------------------------- main

EXPECTED = {
    "door_unlocked": {"action": "toggle", "conditions": ["front=door", "carrying_matches_door=True"]},
    # front_matches_door is only true for a key in front, so it implies front=key.
    "holding_matching_key": {"action": "pickup", "conditions": ["front_matches_door=True", "carrying=none"]},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps-factor", type=int, default=10)   # MiniGrid DoorKey: 10 * size^2
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--size", type=int, default=None, help="size for the full analysis")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    result = {"data_check": {}}
    for size in args.sizes:
        data = collect(size, args.episodes, args.max_steps_factor * size * size, args.seed)
        result["data_check"][size] = data_check(data)
        print(size, json.dumps(result["data_check"][size]["episodes_with_event_per_1000"]))
    if args.check_only:
        print(json.dumps(result, indent=1))
        return

    size = args.size or args.sizes[-1]
    data = collect(size, args.episodes, args.max_steps_factor * size * size, args.seed)
    result["analysis_size"] = size
    for goal in GOALS:
        rows = achievement_rows(data, goal)
        g = {"successes": int(sum(r[2] for r in rows)),
             "rules": find_rules(rows),
             "baseline_lift": lift_baseline(data, goal),
             "jumps": jumps(data, goal, args.horizon)}
        if goal in ("door_unlocked", "door_open"):
            g["toggle_facing_door_with_distractor"] = probability_under(
                rows, TOGGLE, {"front": "door", "carrying": "key", "carrying_matches_door": False})
            g["toggle_facing_door_empty_handed"] = probability_under(rows, TOGGLE, {"front": "door", "carrying": "none"})
        if goal in EXPECTED:
            g["expected"] = EXPECTED[goal]
            g["successes_curve"] = successes_curve(data, goal, EXPECTED[goal])
        result[goal] = g
        print(goal, json.dumps(g["rules"]))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
