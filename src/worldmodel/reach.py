"""Card 004: reach conditions. Walking to X as an action with conditions.

For each state and target, whether turns and forward steps alone reach the
target is computed exactly per layout. Card 003's condition finder is run on
those labels, with the hand-named place ("side") removed from the
vocabulary. A planner then chains the found rules backward from the goal
square and executes them. Evaluator machinery only (C1).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import numpy as np

from .conditions import achievement_rows, collect, find_rules, state_graph
from .envs.keydoor import (DIR_VEC, FORWARD, LEFT, PICKUP, RIGHT, TOGGLE, Layout, cell, front,
                           make_layout, start_state, step, variables)

TARGETS = ("goal", "door", "key", "distractor")


def at_target(layout: Layout, state: tuple, target: str) -> bool:
    if target == "goal":
        return (state[0], state[1]) == layout.goal
    return cell(layout, state, front(state)) == target


def target_exists(state: tuple, target: str) -> bool:
    return not ((target == "key" and state[3] == 1) or (target == "distractor" and state[3] == 2))


@lru_cache(maxsize=4096)
def walk_components(layout: Layout, carry: int, door: int) -> dict:
    """(x, y, dir) -> component id under turns and forward steps, for a fixed
    carry and door state. Walking is reversible (turn around, step back), so
    components are undirected; the goal square ends an episode but is still
    reached from its component."""
    s = layout.size
    nodes = [(x, y, d) for x in range(1, s - 1) for y in range(1, s - 1) for d in range(4)
             if cell(layout, (x, y, d, carry, door), (x, y)) in ("empty", "goal")
             or ((x, y) == layout.door and door == 2)]
    parent = {n: n for n in nodes}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for u in nodes:
        if (u[0], u[1]) == layout.goal:
            continue                          # the episode ends here; entering it is the edge
        for a in (LEFT, RIGHT, FORWARD):
            v = step(layout, (*u, carry, door), a)[0][:3]
            parent[root(u)] = root(v)
    return {n: root(n) for n in nodes}


@lru_cache(maxsize=4096)
def reachable_targets(layout: Layout, carry: int, door: int) -> dict:
    """Component id -> set of targets reachable by walking."""
    comp = walk_components(layout, carry, door)
    out = {}
    for n, c in comp.items():
        st = (*n, carry, door)
        for t in TARGETS:
            if at_target(layout, st, t):
                out.setdefault(c, set()).add(t)
    return out


def walk_reaches(layout: Layout, state: tuple, target: str) -> bool:
    comp = walk_components(layout, state[3], state[4])
    return target in reachable_targets(layout, state[3], state[4]).get(comp.get(state[:3]), set())


def walk_rows(data, target: str, stride: int = 4, drop=("side",)):
    rows = []
    for ep in data:
        lay = ep["layout"]
        for s in ep["states"][:-1:stride]:
            if at_target(lay, s, target) or not target_exists(s, target):
                continue
            v = {k: val for k, val in variables(lay, s).items() if k not in drop}
            rows.append((v, 0, walk_reaches(lay, s, target)))
    return rows


def walk_path(layout: Layout, state: tuple, target: str) -> list[int] | None:
    """Shortest turns-and-forward sequence to the target, or None."""
    prev = {state: None}
    todo = deque([state])
    while todo:
        u = todo.popleft()
        if at_target(layout, u, target):
            path = []
            while prev[u] is not None:
                u, a = prev[u]
                path.append(a)
            return path[::-1]
        if (u[0], u[1]) == layout.goal:
            continue
        for a in (LEFT, RIGHT, FORWARD):
            t, _ = step(layout, u, a)
            if t not in prev:
                prev[t] = (u, a)
                todo.append(t)
    return None


# ---------------------------------------------------------------- planner

def persistence(data, atoms: set) -> dict:
    """P(atom holds at t+1 | it holds at t) under random play."""
    held, kept = Counter(), Counter()
    for ep in data:
        lay, S = ep["layout"], ep["states"]
        prev = None
        for s in S:
            v = variables(lay, s)
            now = {a for a in atoms if str(v.get(a[0])) == a[1]}
            if prev is not None:
                held.update(prev)
                kept.update(prev & now)
            prev = now
    return {a: kept[a] / held[a] for a in atoms if held[a]}


def parse(cond: str) -> tuple[str, str]:
    k, _, v = cond.partition("=")
    return (k, v)


def holds_atom(layout, state, atom) -> bool:
    if atom == ("on_goal", "True"):
        return (state[0], state[1]) == layout.goal
    return str(variables(layout, state).get(atom[0])) == atom[1]


class Planner:
    """Chains found rules backward. abilities: atom -> (kind, detail, conditions)."""

    def __init__(self, abilities: dict, persist: dict):
        self.abilities = abilities
        self.persist = persist

    def achieve(self, layout, state, atom, trace, depth=0, budget=2000):
        if holds_atom(layout, state, atom):
            return state, True
        if atom not in self.abilities or depth > 8:
            return state, False
        kind, detail, conds = self.abilities[atom]
        conds = sorted(conds, key=lambda c: -self.persist.get(c, 0.0))
        for _ in range(3):                    # re-establish conditions a later one broke
            for c in conds:
                state, ok = self.achieve(layout, state, c, trace, depth + 1)
                if not ok:
                    return state, False
            if all(holds_atom(layout, state, c) for c in conds):
                break
        else:
            return state, False
        if kind == "walk":
            path = walk_path(layout, state, detail)
            if path is None:
                return state, False
            trace.append(("walk to " + detail, len(path)))
            for a in path:
                state, _ = step(layout, state, a)
        else:
            trace.append((kind, 1))
            state, _ = step(layout, state, detail)
        return state, holds_atom(layout, state, atom)


def shortest_solution(layout: Layout) -> int:
    index, states, nxt = state_graph(layout)
    dist = {0: 0}
    todo = deque([0])
    while todo:
        u = todo.popleft()
        s = states[u]
        if (s[0], s[1]) == layout.goal:
            return dist[u]
        for v in nxt[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                todo.append(v)
    return -1


def random_chance(layout: Layout, horizon: int) -> float:
    from .conditions import exact_chance
    index, v = exact_chance(layout, "on_goal", horizon)
    return float(v[index[start_state(layout)]])


# ---------------------------------------------------------------- main

WALK_ATOM = {"goal": ("on_goal", "True"), "door": ("front", "door"),
             "key": ("front_matches_door", "True")}
ACTION_GOALS = {"door_open": (("door", "open"), TOGGLE),
                "holding_matching_key": (("carrying_matches_door", "True"), PICKUP)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--test-layouts", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = collect(args.size, args.episodes, 10 * args.size ** 2, args.seed)
    result = {"walk": {}, "walk_with_side": {}}
    for target in TARGETS:
        for name, drop in (("walk", ("side",)), ("walk_with_side", ())):
            rows = walk_rows(data, target, drop=drop)
            result[name][target] = {"attempts": len(rows), "reached": int(sum(r[2] for r in rows)),
                                    "rules": find_rules(rows, action_names=("walk to " + target,))}
        print(target, json.dumps(result["walk"][target]["rules"]))

    # Abilities from the main rule (most successes) of each found rule set.
    abilities = {}
    for target, atom in WALK_ATOM.items():
        rule = result["walk"][target]["rules"][0]
        abilities[atom] = ("walk", target, [parse(c) for c in rule["conditions"]])
    action_rules = {}
    for goal, (atom, action) in ACTION_GOALS.items():
        rule = find_rules(achievement_rows(data, goal))[0]
        action_rules[goal] = rule
        abilities[atom] = (rule["action"], action, [parse(c) for c in rule["conditions"]])
    result["action_rules"] = action_rules
    atoms = {c for _, _, conds in abilities.values() for c in conds}
    persist = persistence(data[:500], atoms)
    result["persistence"] = {f"{k}={v}": round(p, 3) for (k, v), p in persist.items()}
    result["abilities"] = {f"{k}={v}": [kind, str(d), [f"{a}={b}" for a, b in c]]
                           for (k, v), (kind, d, c) in abilities.items()}
    planner = Planner(abilities, persist)

    rng = np.random.default_rng(12345)
    plans, ok, steps, shortest, chance, distractor = Counter(), 0, [], [], [], 0
    for _ in range(args.test_layouts):
        lay = make_layout(args.size, rng)
        trace = []
        state, success = planner.achieve(lay, start_state(lay), ("on_goal", "True"), trace)
        plans[" → ".join(t for t, _ in trace)] += 1
        distractor += state[3] == 2
        if success:
            ok += 1
            n = sum(k for _, k in trace)
            steps.append(n)
            shortest.append(shortest_solution(lay))
            chance.append(random_chance(lay, n))
    result["acting"] = {
        "layouts": args.test_layouts, "success": round(ok / args.test_layouts, 3),
        "distractor_picked": distractor,
        "plans": dict(plans.most_common()),
        "mean_steps": round(float(np.mean(steps)), 1), "mean_shortest": round(float(np.mean(shortest)), 1),
        "steps_over_shortest": round(float(np.mean(np.array(steps) / np.array(shortest))), 3),
        "random_play_chance_in_same_steps": round(float(np.mean(chance)), 4)}
    print(json.dumps(result["acting"], indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
