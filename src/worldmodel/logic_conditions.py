"""Card 010 part A: conditions by Bayesian evidence, no thresholds.

A goal's attempts (state variables, action, achieved) are explained by a
decision list: an ordered "or" of rules, each an "and" of atoms
(variable=value, including action=...), plus a default. Each rule's
success rate has a Beta(1, 1) prior and is integrated out; the structure
costs log 2 per rule and log 2 + log(number of atoms) per atom. A change is
kept only if it raises this evidence. Greedy search: add an atom to a rule
or start a new rule; if no single change helps, try pairs among the 20
best single atoms (conditions useful only together).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from math import lgamma


def betaln(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)

from .conditions import ACTION_NAMES as ACTIONS_CARD003
from .conditions import collect as collect_card003
from .conditions import find_rules
from .envs import keydoor as kd
from .envs import logicdoor as ld

KEYS = ("action", "front", "front_colour", "front_matches_door", "carrying", "carrying_colour",
        "carrying_matches_door", "door", "switch", "vase", "side", "dir", "x", "y")


# ---------------------------------------------------------------- data

class World:
    def __init__(self, name):
        self.name = name
        if name == "nodrop":
            self.actions = ACTIONS_CARD003
            self.variables = kd.variables
            self.goals = {"door_open": lambda lay, s: s[4] == 2,
                          "holding_matching_key": lambda lay, s: s[3] == 1,
                          "on_goal": lambda lay, s: (s[0], s[1]) == lay.goal}
        else:
            self.actions = ld.ACTION_NAMES
            self.variables = ld.variables
            self.goals = ld.GOALS

    def collect(self, episodes, seed):
        if self.name == "nodrop":
            return collect_card003(8, episodes, 640, seed)
        return ld.collect(self.name, episodes, seed)


def step_patterns(world: World, ep: dict) -> list[tuple]:
    """Atom values of every step (state variables and the action taken)."""
    lay, S, A = ep["layout"], ep["states"], ep["actions"]
    out = []
    for s, a in zip(S, A):
        v = world.variables(lay, s)
        v["action"] = world.actions[a]
        out.append(tuple(str(v.get(k, "none")) for k in KEYS))
    return out


def episode_patterns(world: World, ep: dict, pats: list[tuple], goal: str) -> Counter:
    """(pattern, achieved) -> count, for steps where the goal did not hold."""
    f = world.goals[goal]
    lay, S = ep["layout"], ep["states"]
    out = Counter()
    for p, s, t in zip(pats, S, S[1:]):
        if not f(lay, s):
            out[(p, bool(f(lay, t)))] += 1
    return out


class Data:
    """Unique patterns with trial and success counts, as an atom matrix."""

    def __init__(self, counts: Counter, keys: tuple = KEYS):
        rows = defaultdict(lambda: [0, 0])
        for (pat, y), c in counts.items():
            rows[pat][0] += c
            rows[pat][1] += c * y
        pats = list(rows)
        self.n = np.array([rows[p][0] for p in pats], float)
        self.k = np.array([rows[p][1] for p in pats], float)
        atoms = sorted({(keys[i], v) for p in pats for i, v in enumerate(p)})
        self.atoms = atoms
        index = {a: j for j, a in enumerate(atoms)}
        self.X = np.zeros((len(pats), len(atoms)), bool)
        for r, p in enumerate(pats):
            for i, v in enumerate(p):
                self.X[r, index[(keys[i], v)]] = True


# ---------------------------------------------------------------- evidence finder

def log_evidence(d: Data, rules: list[list[int]]) -> float:
    """Evidence of the rule list; -inf unless every rule is a route to
    success (succeeds more often than the default), as the definition says."""
    remaining = np.ones(len(d.n), bool)
    total, rates = 0.0, []
    for rule in rules:
        m = remaining & d.X[:, rule].all(1)
        N, K = d.n[m].sum(), d.k[m].sum()
        if N == 0:
            return -np.inf
        total += betaln(K + 1, N - K + 1)
        rates.append(K / N)
        remaining &= ~m
    N, K = d.n[remaining].sum(), d.k[remaining].sum()
    total += betaln(K + 1, N - K + 1)
    if rates and min(rates) <= (K / N if N else 0.0):
        return -np.inf
    cost = sum(np.log(2) + len(r) * (np.log(2) + np.log(len(d.atoms))) for r in rules)
    return total - cost


def evidence_rules(d: Data, top_pairs: int = 20, max_steps: int = 40):
    rules: list[list[int]] = []
    best = log_evidence(d, rules)
    trace = []
    for _ in range(max_steps):
        moves = []
        for j in range(len(d.atoms)):
            moves.append(((len(rules), (j,)), rules + [[j]]))
            for r in range(len(rules)):
                if j not in rules[r]:
                    new = [list(x) for x in rules]
                    new[r].append(j)
                    moves.append(((r, (j,)), new))
        # Also removals (an atom, or a whole rule), so the search can leave dead ends.
        for r in range(len(rules)):
            moves.append(((r, ("drop rule",)), [list(x) for i, x in enumerate(rules) if i != r]))
            if len(rules[r]) > 1:
                for j in rules[r]:
                    new = [list(x) for x in rules]
                    new[r] = [i for i in new[r] if i != j]
                    moves.append(((r, ("drop", j)), new))
        scored = [(log_evidence(d, new), tag, new) for tag, new in moves]
        score, tag, new = max(scored, key=lambda x: x[0])
        if score <= best:
            # Pairs: conditions that only help together.
            singles = sorted(scored, key=lambda x: -x[0])
            cand = []
            for _, (r, parts), _ in singles:
                if len(parts) != 1 or isinstance(parts[0], str):
                    continue
                j = parts[0]
                if j not in cand:
                    cand.append(j)
                if len(cand) == top_pairs:
                    break
            pair_moves = []
            for a_i in range(len(cand)):
                for b_i in range(a_i + 1, len(cand)):
                    j1, j2 = cand[a_i], cand[b_i]
                    pair_moves.append(((len(rules), (j1, j2)), rules + [[j1, j2]]))
                    for r in range(len(rules)):
                        if j1 not in rules[r] and j2 not in rules[r]:
                            new = [list(x) for x in rules]
                            new[r] += [j1, j2]
                            pair_moves.append(((r, (j1, j2)), new))
            scored = [(log_evidence(d, new), tag, new) for tag, new in pair_moves]
            if not scored:
                break
            score, tag, new = max(scored, key=lambda x: x[0])
            if score <= best:
                break
        trace.append({"move": [tag[0], [p if isinstance(p, str) else f"{d.atoms[p][0]}={d.atoms[p][1]}"
                                        for p in tag[1]]],
                      "evidence_gain_bits": round((score - best) / np.log(2), 1)})
        rules, best = new, score
    return rules, best, trace


def describe(d: Data, rules):
    out, remaining = [], np.ones(len(d.n), bool)
    for rule in rules:
        m = remaining & d.X[:, rule].all(1)
        entry = {"conditions": sorted(f"{d.atoms[j][0]}={d.atoms[j][1]}" for j in rule),
                 "attempts": int(d.n[m].sum()), "successes": int(d.k[m].sum()),
                 "success_rate": round(float(d.k[m].sum() / max(d.n[m].sum(), 1)), 4), "removal": {}}
        full = d.X[:, rule].all(1)
        for j in rule:
            others = d.X[:, [i for i in rule if i != j]].all(1) if len(rule) > 1 else np.ones(len(d.n), bool)
            without = others & ~d.X[:, j]
            p_with = d.k[full].sum() / max(d.n[full].sum(), 1)
            p_without = d.k[without].sum() / d.n[without].sum() if d.n[without].sum() else float("nan")
            entry["removal"][f"{d.atoms[j][0]}={d.atoms[j][1]}"] = round(float(p_with - p_without), 4)
        out.append(entry)
        remaining &= ~m
    out.append({"default": True, "attempts": int(d.n[remaining].sum()), "successes": int(d.k[remaining].sum())})
    return out


def rule_sets(described) -> set:
    return {frozenset(r["conditions"]) for r in described if "conditions" in r}


# ---------------------------------------------------------------- expected rules

DOOR = {"key": [["action=toggle", "front=door", "carrying_matches_door=True"]],
        "switch": [["action=toggle", "front=door", "switch=on"]],
        "either": [["action=toggle", "front=door", "carrying_matches_door=True"],
                   ["action=toggle", "front=door", "switch=on"]],
        "both": [["action=toggle", "front=door", "carrying_matches_door=True", "switch=on"]],
        "nodrop": [["action=toggle", "front=door", "carrying_matches_door=True"]]}
OTHER = {"holding_matching_key": [["action=pickup", "front_matches_door=True", "carrying=none"]],
         "switch_on": [["action=toggle", "front=switch"]],
         "on_goal": [["action=forward", "front=goal"]]}


def expected(world: str, goal: str) -> set:
    rules = DOOR[world] if goal == "door_open" else OTHER[goal]
    return {frozenset(r) for r in rules}


def rarest_route(world: str, lay, s) -> bool:
    """For the few-examples curve: did this door opening use the rarest rule?"""
    key = s[3] == 1
    sw = bool(s[7]) if world != "nodrop" else False
    return {"key": key, "switch": sw, "either": key and not sw, "both": key and sw, "nodrop": key}[world]


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    started = time.monotonic()
    result = {"episodes": args.episodes}
    for name in ("key", "switch", "either", "both", "nodrop"):
        world = World(name)
        data = world.collect(args.episodes, args.seed)
        res = {}
        pats = [step_patterns(world, ep) for ep in data]
        for goal in world.goals:
            per_ep = [episode_patterns(world, ep, p, goal) for ep, p in zip(data, pats)]
            total = Counter()
            for c in per_ep:
                total.update(c)
            d = Data(total)
            rules, score, trace = evidence_rules(d)
            desc = describe(d, rules)
            found = rule_sets(desc)
            entry = {"rules": desc, "trace": trace, "expected": sorted(map(sorted, expected(name, goal))),
                     "exact": found == expected(name, goal),
                     "vase_in_any_rule": any(c.startswith("vase=") or c == "front=vase"
                                             for r in found for c in r)}
            # Baseline: card 005's threshold finder on the same attempts.
            rows = []
            for ep in data[:1000]:
                lay, S, A = ep["layout"], ep["states"], ep["actions"]
                f = world.goals[goal]
                for s, a, t in zip(S, A, S[1:]):
                    if not f(lay, s):
                        rows.append((world.variables(lay, s), a, bool(f(lay, t))))
            base = find_rules(rows, action_names=world.actions)
            entry["threshold_finder_1000_episodes"] = [sorted([f"action={r['action']}"] + list(r["conditions"]))
                                                       for r in base]
            # Few examples: subsample episodes until n successes of the rarest rule.
            if goal == "door_open":
                routes = []
                for ep in data:
                    lay, S = ep["layout"], ep["states"]
                    routes.append(sum(1 for s, t in zip(S, S[1:]) if not s[_door(name)] == _open(name)
                                      and t[_door(name)] == _open(name) and rarest_route(name, lay, s)))
                curve = {}
                rng = np.random.default_rng(0)
                for n in (5, 10, 30, 100):
                    ok = 0
                    for _ in range(args.repeats):
                        got, chosen = 0, Counter()
                        for i in rng.permutation(len(data)):
                            chosen.update(per_ep[i])
                            got += routes[i]
                            if got >= n:
                                break
                        sub = Data(chosen)
                        r_sub, _, _ = evidence_rules(sub)
                        ok += rule_sets(describe(sub, r_sub)) == expected(name, goal)
                    curve[n] = round(ok / args.repeats, 2)
                entry["exact_recovery_by_rarest_rule_successes"] = curve
            res[goal] = entry
            print(name, goal, "exact" if entry["exact"] else "NOT exact", sorted(map(sorted, found)),
                  entry.get("exact_recovery_by_rarest_rule_successes", ""), flush=True)
        result[name] = res
        args.out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    result["seconds"] = round(time.monotonic() - started, 1)
    args.out.write_text(json.dumps(result, indent=1, default=str) + "\n")


def _door(name):
    return 4 if name == "nodrop" else 6


def _open(name):
    return 2 if name == "nodrop" else 1


if __name__ == "__main__":
    main()
