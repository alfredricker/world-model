"""Goal-conditioned fork evaluator for chained rooms (CHARTER rung 1).

At a probe step the learner is shown a goal condition as example frames and
must rank the five actions by the true fewest steps to the goal after each.
This module computes those true distances; they never reach the learner.

Searching by deep-copying the simulator is too slow, so the search runs on a
small exact re-implementation of the world's rules (``Layout`` + ``advance``).
``tests/test_rooms_goals.py`` checks it against the real environment step by
step. Distances ignore the ``max_steps`` truncation. States are Markov without
the step counter: a timed door carries its own elapsed time, capped at its
delay. Everything reachable from a later state of an episode is reachable
from its start, so one search per episode serves every fork in it.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from ..data import file_hash, write_json
from .rooms import ACTION_MAP, RoomsConfig, RoomsPort, Switch, SwitchDoor, TimedDoor, persistent_actions
from .rooms_data import world_seed
from .rooms_probes import FRONT_KINDS, _front

LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = range(5)   # opaque IDs, see ACTION_MAP
INTERACTIONS = (PICKUP, TOGGLE)
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))          # MiniGrid DIR_TO_VEC
INF = np.iinfo(np.int32).max
GOAL_KINDS = ("key_held", "door_open")
CLASSES = ("indifferent", "interaction", "movement", "mixed")


@dataclass(frozen=True)
class Layout:
    """The parts of an episode's world that never change."""
    blocked: frozenset        # walls and switches: never entered
    lava: frozenset
    goal: tuple[int, int]
    doors: tuple              # (pos, kind, colour, delay) per door index
    switches: tuple           # (pos, door index) per switch index
    consume_keys: bool

    @property
    def door_at(self) -> dict:
        return {d[0]: i for i, d in enumerate(self.doors)}

    @property
    def switch_at(self) -> dict:
        return {s[0]: i for i, s in enumerate(self.switches)}


# State: (x, y, dir, carrying, objects, doors, switches)
#   carrying: None | ("key", colour) | ("box", contents colour or None)
#   objects:  sorted tuple of (pos, item) for keys and boxes on the floor
#   doors:    per door (locked_or_enabled, open, elapsed); for a switch door
#             the first field is "enabled", for other kinds "locked"
#   switches: per switch, on


def layout_of(env) -> Layout:
    from minigrid.core.world_object import Lava, Wall
    blocked, lava = set(), set()
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if isinstance(obj, (Wall, Switch)):
                blocked.add((x, y))
            elif isinstance(obj, Lava):
                lava.add((x, y))
    doors = tuple((tuple(r.pos), r.kind, r.color, getattr(r.obj, "delay", 0)) for r in env.doors)
    door_index = {id(r.obj): i for i, r in enumerate(env.doors)}
    switches = tuple((tuple(r.switch_pos), door_index[id(r.obj)]) for r in env.doors if r.kind == "switch")
    return Layout(frozenset(blocked), frozenset(lava), tuple(env.goal_pos), doors, switches,
                  env.config.consume_keys)


def _item(obj):
    from minigrid.core.world_object import Box, Key
    if obj is None:
        return None
    if isinstance(obj, Key):
        return ("key", obj.color)
    if isinstance(obj, Box):
        return ("box", None if obj.contains is None else obj.contains.color)
    raise TypeError(f"Not a movable object: {obj!r}")


def state_of(env, layout: Layout) -> tuple:
    from minigrid.core.world_object import Box, Key
    objects = []
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if isinstance(obj, (Key, Box)):
                objects.append(((x, y), _item(obj)))
    doors = []
    for (pos, kind, colour, delay), r in zip(layout.doors, env.doors, strict=True):
        door = r.obj
        if isinstance(door, SwitchDoor):
            doors.append((bool(door.enabled), bool(door.is_open), 0))
        elif isinstance(door, TimedDoor):
            elapsed = 0
            if door.is_open and door.opened_at is not None:
                elapsed = min(env.step_count - door.opened_at, delay)
            doors.append((False, bool(door.is_open), elapsed))
        else:
            doors.append((bool(door.is_locked), bool(door.is_open), 0))
    switches = tuple(bool(env.grid.get(*pos).on) for pos, _ in layout.switches)
    x, y = (int(v) for v in env.agent_pos)
    return (x, y, int(env.agent_dir), _item(env.carrying), tuple(sorted(objects)), tuple(doors), switches)


def advance(layout: Layout, state: tuple, action: int) -> tuple[tuple, bool]:
    """One step of the world's rules; returns (next state, terminated)."""
    x, y, d, carrying, objects, doors, switches = state
    dx, dy = DIRS[d]
    front = (x + dx, y + dy)
    floor = dict(objects)
    doors = list(doors)
    switches = list(switches)
    before_doors = tuple(doors)
    door_at, switch_at = layout.door_at, layout.switch_at
    terminated = False
    if action == LEFT:
        d = (d - 1) % 4
    elif action == RIGHT:
        d = (d + 1) % 4
    elif action == FORWARD:
        if front in door_at:
            if doors[door_at[front]][1]:
                x, y = front
        elif front in layout.blocked or front in floor:
            pass
        else:
            x, y = front
            terminated = front in layout.lava or front == layout.goal
    elif action == PICKUP:
        item = floor.get(front)
        if item is not None:
            if carrying is not None and item[0] == "key":
                floor[front], carrying = carrying, item          # the port's swap
            elif carrying is None:
                del floor[front]
                carrying = item
    elif action == TOGGLE:
        if front in door_at:
            i = door_at[front]
            kind, colour = layout.doors[i][1], layout.doors[i][2]
            flag, is_open, elapsed = doors[i]
            if kind == "switch":
                if is_open:
                    is_open = False
                elif flag:
                    is_open = True
            elif kind == "locked" and flag:
                if carrying == ("key", colour):
                    flag, is_open = False, True
            else:
                is_open = not is_open
            doors[i] = (flag, is_open, elapsed)
        elif front in switch_at:
            i = switch_at[front]
            switches[i] = True
            j = layout.switches[i][1]
            doors[j] = (True, doors[j][1], doors[j][2])
        elif front in floor and floor[front][0] == "box":
            contents = floor[front][1]
            if contents is None:
                del floor[front]
            else:
                floor[front] = ("key", contents)
    # Post-step door bookkeeping, in door order, as in ChainedRoomsEnv.step.
    for i, (pos, kind, colour, delay) in enumerate(layout.doors):
        was_flag, was_open, was_elapsed = before_doors[i]
        flag, is_open, elapsed = doors[i]
        if kind == "locked" and was_flag and not flag and layout.consume_keys:
            carrying = None
        if kind == "timed":
            if is_open and not was_open:
                elapsed = 0
            elif is_open:
                elapsed = min(was_elapsed + 1, delay)
            else:
                elapsed = 0
            if is_open and elapsed >= delay and (x, y) != pos:
                is_open, elapsed = False, 0
            doors[i] = (flag, is_open, elapsed)
    state = (x, y, d, carrying, tuple(sorted(floor.items())), tuple(doors), tuple(switches))
    return state, terminated


def holds(layout: Layout, state: tuple, goal: tuple) -> bool:
    kind, colour = goal
    if kind == "key_held":
        return state[3] == ("key", colour)
    if kind == "door_open":
        return any(door[2] == colour and s[1] for door, s in zip(layout.doors, state[5], strict=True))
    raise ValueError(f"Unknown goal kind {kind}")


@dataclass
class Graph:
    index: dict               # state -> id
    states: list
    succ: np.ndarray          # (n, 5) next-state ids, -1 after termination
    terminal: np.ndarray      # (n,) bool: reached by a terminating step
    complete: bool            # False if the state cap stopped the search
    _pred: tuple | None = None


def reachable(layout: Layout, start: tuple, cap: int = 2_000_000) -> Graph:
    """Every state reachable from ``start``, by breadth-first search."""
    index, states, succ, terminal = {start: 0}, [start], [], [False]
    queue = deque([0])
    complete = True
    while queue:
        i = queue.popleft()
        row = [-1] * 5
        if not terminal[i]:
            for a in range(5):
                nxt, term = advance(layout, states[i], a)
                j = index.get(nxt)
                if j is None:
                    if len(states) >= cap:
                        complete = False
                        continue
                    j = len(states)
                    index[nxt] = j
                    states.append(nxt)
                    terminal.append(term)
                    queue.append(j)
                row[a] = j
        succ.append(row)
    return Graph(index, states, np.array(succ, np.int64).reshape(-1, 5), np.array(terminal), complete)


def predecessors(graph: Graph) -> tuple[np.ndarray, np.ndarray]:
    """Reverse edges in compressed form: preds of j are src[start[j]:start[j+1]]."""
    if getattr(graph, "_pred", None) is None:
        n = len(graph.states)
        src = np.repeat(np.arange(n), 5)
        dst = graph.succ.reshape(-1)
        keep = dst >= 0
        src, dst = src[keep], dst[keep]
        order = np.argsort(dst, kind="stable")
        start = np.zeros(n + 1, np.int64)
        np.add.at(start, dst + 1, 1)
        graph._pred = (np.cumsum(start), src[order])
    return graph._pred


def distances(layout: Layout, graph: Graph, goal: tuple) -> np.ndarray:
    """Fewest steps from every state to one where ``goal`` holds (INF if never)."""
    start, src = predecessors(graph)
    dist = np.full(len(graph.states), INF, np.int64)
    frontier = np.array([i for i, s in enumerate(graph.states) if holds(layout, s, goal)], np.int64)
    dist[frontier] = 0
    depth = 0
    while len(frontier):
        depth += 1
        lo, hi = start[frontier], start[frontier + 1]
        pred = np.concatenate([src[a:b] for a, b in zip(lo, hi)]) if len(frontier) else frontier
        pred = np.unique(pred[dist[pred] == INF])
        dist[pred] = depth
        frontier = pred
    return dist


def all_action_distances(graph: Graph, dist: np.ndarray) -> np.ndarray:
    """(n, 5): true steps to the goal after each action, for every state."""
    succ = graph.succ
    safe = np.where(succ >= 0, succ, 0)
    nxt = dist[safe]
    return np.where((succ >= 0) & (nxt != INF), 1 + nxt, INF)


def action_distances(graph: Graph, dist: np.ndarray, state_id: int) -> np.ndarray:
    """True steps to the goal after each action: 1 + distance of its successor."""
    out = np.full(5, INF, np.int64)
    for a, j in enumerate(graph.succ[state_id]):
        if j >= 0 and dist[j] != INF and not (graph.terminal[j] and dist[j] > 0):
            out[a] = 1 + dist[j]
    return out


def classify(action_dist: np.ndarray) -> tuple[str, np.ndarray]:
    """The fork's class and its set of best actions."""
    best_value = action_dist.min()
    best = action_dist == best_value
    if best.all() or best_value == INF:
        return "indifferent", best
    in_best = set(np.flatnonzero(best).tolist())
    if in_best <= set(INTERACTIONS):
        return "interaction", best
    if not in_best & set(INTERACTIONS):
        return "movement", best
    return "mixed", best


def no_effect(graph: Graph, state_id: int) -> np.ndarray:
    """Per action: True if it leaves the world state unchanged."""
    return graph.succ[state_id] == state_id


def goals_of(layout: Layout, state: tuple) -> list[tuple]:
    """Candidate goals in an episode: keys of every colour present, and doors."""
    colours = {c for _, item in state[4] for c in [item[1]] if c is not None}
    if state[3] is not None and state[3][1] is not None:
        colours.add(state[3][1])
    goals = [("key_held", c) for c in sorted(colours)]
    goals += [("door_open", c) for c in sorted({door[2] for door in layout.doors})]
    return goals


def visible_goals(env, layout: Layout, state: tuple) -> list[tuple]:
    """Goals that hold now and can be seen in the current frame: a carried key
    is drawn in the first-person view; an open door must be in view."""
    out = []
    if state[3] is not None and state[3][0] == "key":
        out.append(("key_held", state[3][1]))
    for (pos, kind, colour, _), s in zip(layout.doors, state[5], strict=True):
        if s[1] and env.agent_sees(*pos):
            out.append(("door_open", colour))
    return sorted(set(out))


def goal_probe(out: Path, config: RoomsConfig, episodes: int, seed: int, *, ordinary_rate: float = 0.05,
               repeat: float = 0.5, cap: int = 2_000_000) -> dict:
    """Episodes with goal-conditioned forks and a pool of goal-example frames.

    Forks are taken where probe() takes them: an object in front, or at
    ``ordinary_rate`` otherwise. For each fork and each goal of its episode the
    true steps to the goal after every action are stored with the fork's class.
    Every step records which goals hold and are visible, so example sets can
    be drawn from other episodes."""
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    goal_names = [f"{k}:{c}" for k in GOAL_KINDS for c in config.colours + ["grey"]]
    arrays, records = {}, []
    totals = {name: 0 for name in CLASSES}
    for index in range(episodes):
        port = RoomsPort(config)
        observation = port.reset(world_seed(seed + 700000, config.split, index))
        env = port._env
        layout = layout_of(env)
        start = state_of(env, layout)
        graph = reachable(layout, start, cap)
        goals = goals_of(layout, start)
        dist = {g: distances(layout, graph, g) for g in goals}
        frames, visible, forks = [observation.rgb], [], []
        policy = persistent_actions(rng, config.max_steps, port.action_count, repeat)
        for t, action in enumerate(policy):
            state = state_of(env, layout)
            sid = graph.index[state]
            visible.append([goal_names.index(f"{k}:{c}") for k, c in visible_goals(env, layout, state)])
            front = env.grid.get(*env.front_pos)
            interesting = (front is not None and front.type != "wall") or rng.random() < ordinary_rate
            if interesting:
                effect = no_effect(graph, sid)
                front_kind = FRONT_KINDS.index(_front(env)[0])
                carried = -1 if state[3] is None else ["key", "box"].index(state[3][0])
                for g in goals:
                    ad = action_distances(graph, dist[g], sid)
                    cls, best = classify(ad)
                    totals[cls] += 1
                    forks.append([t, goal_names.index(f"{g[0]}:{g[1]}"), CLASSES.index(cls), sid, front_kind, carried,
                                  *np.minimum(ad, INF).tolist(), *best.astype(int).tolist(),
                                  *effect.astype(int).tolist()])
            observation = port.step(int(action))
            frames.append(observation.rgb)
            if observation.terminated or observation.truncated:
                break
        state = state_of(env, layout)
        visible.append([goal_names.index(f"{k}:{c}") for k, c in visible_goals(env, layout, state)])
        port.close()
        vis = np.zeros((len(frames), len(goal_names)), np.bool_)
        for t, ids in enumerate(visible):
            vis[t, ids] = True
        arrays[f"e{index}_frames"] = np.stack(frames)
        arrays[f"e{index}_visible"] = vis
        arrays[f"e{index}_forks"] = np.array(forks, np.int64).reshape(-1, 6 + 15)
        records.append({"episode": index, "length": len(frames) - 1, "forks": len(forks),
                        "states": len(graph.states), "complete": graph.complete,
                        "goals": [f"{k}:{c}" for k, c in goals], "combination": env.combination()})
    np.savez_compressed(out / "goal_probe.npz", **arrays)
    metadata = {"format": 1, "world": asdict(config), "seed": seed, "episodes": records,
                "goal_names": goal_names, "classes": list(CLASSES), "totals": totals,
                "front_kinds": list(FRONT_KINDS), "carried_kinds": ["key", "box"],
                "fork_fields": ["step", "goal", "class", "state", "front_kind", "carried"] + [f"dist_{a}" for a in range(5)]
                + [f"best_{a}" for a in range(5)] + [f"no_effect_{a}" for a in range(5)],
                "inf": int(INF), "sha256": file_hash(out / "goal_probe.npz"),
                "seconds": time.monotonic() - started}
    write_json(out / "goal_probe.json", metadata)
    return metadata


def load_goal_probe(path: Path):
    metadata = json.loads((path / "goal_probe.json").read_text())
    if file_hash(path / "goal_probe.npz") != metadata["sha256"]:
        raise ValueError("Goal probe hash mismatch")
    archive = np.load(path / "goal_probe.npz", allow_pickle=False)
    episodes = [{k: archive[f"e{r['episode']}_{k}"] for k in ("frames", "visible", "forks")}
                for r in metadata["episodes"]]
    return episodes, metadata


def example_sets(episodes: list[dict], goal: int, exclude: int, count: int, rng: np.random.Generator):
    """``count`` frames in which ``goal`` holds and is visible, from episodes
    other than ``exclude``, as (episode, step) pairs; None if too few exist."""
    pool = [(e, t) for e, ep in enumerate(episodes) if e != exclude
            for t in np.flatnonzero(ep["visible"][:, goal]).tolist()]
    if len(pool) < count:
        return None
    picks = rng.choice(len(pool), size=count, replace=False)
    return [pool[i] for i in picks]


def top1(pred: np.ndarray, best: np.ndarray) -> float:
    """1 if the lowest predicted distance falls on a truly best action. Ties in
    the prediction share the credit."""
    low = pred == pred.min()
    return float((low & best.astype(bool)).sum() / low.sum())


def chance(best: np.ndarray) -> float:
    return float(best.sum() / len(best))


def front_kind_of(layout: Layout, state: tuple) -> str:
    """The FRONT_KINDS name of the cell in front, from abstract state (as _front)."""
    x, y, d = state[:3]
    front = (x + DIRS[d][0], y + DIRS[d][1])
    door_at = layout.door_at
    if front in door_at:
        i = door_at[front]
        kind = layout.doors[i][1]
        if kind == "locked":
            return "locked" if state[5][i][0] else "plain"
        return kind
    if front in layout.switch_at:
        return "switchobj"
    if front in layout.blocked:
        return "wall"
    floor = dict(state[4])
    if front in floor:
        return floor[front][0]
    if front in layout.lava:
        return "lava"
    if front == layout.goal:
        return "goal"
    return "none"


def set_state(env, layout: Layout, state: tuple) -> None:
    """Put a freshly reset environment of the same world into ``state``."""
    from minigrid.core.world_object import Box, Key
    x, y, d, carrying, objects, doors, switches = state
    for cx in range(env.width):
        for cy in range(env.height):
            if isinstance(env.grid.get(cx, cy), (Key, Box)):
                env.grid.set(cx, cy, None)

    def make(item):
        if item[0] == "key":
            return Key(item[1])
        return Box("grey", contains=None if item[1] is None else Key(item[1]))

    for pos, item in objects:
        obj = make(item)
        obj.cur_pos = np.array(pos)
        env.grid.set(*pos, obj)
    env.carrying = None if carrying is None else make(carrying)
    if env.carrying is not None:
        env.carrying.cur_pos = np.array([-1, -1])
    for (pos, kind, colour, delay), record, (flag, is_open, elapsed) in zip(layout.doors, env.doors, doors, strict=True):
        door = record.obj
        door.is_open = is_open
        if kind == "switch":
            door.enabled = flag
        elif kind == "timed":
            door.opened_at = env.step_count - elapsed if is_open else None
        else:
            door.is_locked = flag
    for (pos, _), on in zip(layout.switches, switches, strict=True):
        env.grid.get(*pos).on = on
    env.agent_pos = (x, y)
    env.agent_dir = d


def _frame(env, config: RoomsConfig) -> np.ndarray:
    return env.get_frame(tile_size=config.tile_size, agent_pov=True).copy()


def sampled_probe(out: Path, config: RoomsConfig, worlds: int, seed: int, *, per_stratum: int = 300,
                  movement: int = 1000, examples_per_goal: int = 400, cap: int = 2_000_000) -> dict:
    """Goal-conditioned forks at states chosen from each world's reachable set.

    Random-walk trajectories almost never reach the rare situations the test is
    about (for example, facing a locked door holding its key), so fork states
    are drawn from every reachable state of each development world, stratified
    by (goal kind, best action, front kind) for interaction-decisive forks,
    plus a sample of movement-decisive ones. Goal examples are frames of
    reachable states where the goal holds and is visible, drawn per world so
    that a fork's examples can come from other worlds."""
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 993]))
    goal_names = [f"{k}:{c}" for k in GOAL_KINDS for c in config.colours + ["grey"]]
    candidates, pools, records, available = [], {g: [] for g in goal_names}, [], {}
    worlds_data = []
    for w in range(worlds):
        port = RoomsPort(config)
        world = world_seed(seed + 700000, config.split, w)
        port.reset(world)
        env = port._env
        layout = layout_of(env)
        start = state_of(env, layout)
        graph = reachable(layout, start, cap)
        goals = goals_of(layout, start)
        live = ~graph.terminal
        fronts = np.array([FRONT_KINDS.index(front_kind_of(layout, st)) for st in graph.states])
        unchanged = graph.succ == np.arange(len(graph.states))[:, None]
        for g in goals:
            name = f"{g[0]}:{g[1]}"
            dist = distances(layout, graph, g)
            ad = all_action_distances(graph, dist)
            low = ad.min(1)
            best = ad == low[:, None]
            decided = live & (low != INF) & ~best.all(1)
            inter = decided & ~best[:, :3].any(1)
            move = decided & ~best[:, 3:].any(1)
            groups = {}
            for sid in np.flatnonzero(inter):
                act = "pickup" if best[sid, PICKUP] and not best[sid, TOGGLE] else "toggle" if not best[sid, PICKUP] else "both"
                groups.setdefault(f"{g[0]}|{act}|{FRONT_KINDS[fronts[sid]]}", []).append(int(sid))
            per_world_cap = max(5, 3 * per_stratum // worlds)
            for stratum, sids in groups.items():
                available[stratum] = available.get(stratum, 0) + len(sids)
                for sid in rng.choice(sids, size=min(len(sids), per_world_cap), replace=False):
                    candidates.append(("interaction", stratum, w, int(sid), name,
                                       ad[sid], best[sid], unchanged[sid], int(fronts[sid]), graph.states[sid]))
            msel = np.flatnonzero(move)
            for sid in rng.choice(msel, size=min(len(msel), max(1, movement // worlds)), replace=False) if len(msel) else []:
                candidates.append(("movement", f"{g[0]}|move", w, int(sid), name, ad[sid], best[sid], unchanged[sid],
                                   int(fronts[sid]), graph.states[sid]))
        worlds_data.append((world, layout))
        # Goal example pool: states where a goal holds and is visible (checked on the real env).
        order = rng.permutation(np.flatnonzero(live))
        wanted = {f"{k}:{c}" for k, c in goals}
        found = {n: 0 for n in wanted}
        per_world = max(1, examples_per_goal // worlds)
        for sid in order[:20000]:
            st = graph.states[sid]
            if not any(holds(layout, st, tuple(n.split(":"))) for n in wanted if found[n] < per_world):
                continue
            set_state(env, layout, st)
            for k, c in visible_goals(env, layout, st):
                n = f"{k}:{c}"
                if n in found and found[n] < per_world:
                    pools[n].append((w, int(sid), _frame(env, config)))
                    found[n] += 1
            if all(v >= per_world for v in found.values()):
                break
        port.close()
        n_states, complete = len(graph.states), graph.complete
        del graph
        records.append({"world": w, "states": n_states, "complete": complete,
                        "goals": sorted(wanted), "combination": env.combination()})
    # Stratified selection: at most per_stratum per stratum, spread over worlds.
    chosen = []
    by_stratum = {}
    for c in candidates:
        by_stratum.setdefault(c[1], []).append(c)
    strata_counts = {}
    for stratum, items in sorted(by_stratum.items()):
        limit = movement if stratum.endswith("|move") else per_stratum
        items = [items[i] for i in rng.permutation(len(items))]
        per_world, picked, taken = {}, [], set()
        # Round-robin over worlds so one large world cannot fill a stratum.
        for round_ in range(1 + max(1, limit)):
            for it in items:
                if len(picked) >= limit:
                    break
                key = (it[2], it[3], it[4])
                if key in taken or per_world.get(it[2], 0) > round_:
                    continue
                picked.append(it)
                taken.add(key)
                per_world[it[2]] = per_world.get(it[2], 0) + 1
            if len(picked) >= limit or len(taken) == len(items):
                break
        strata_counts[stratum] = {"available": available.get(stratum, len(items)), "chosen": len(picked),
                                  "worlds": len({it[2] for it in picked})}
        chosen += picked
    chosen.sort(key=lambda c: c[2])
    frames, rows, current = [], [], None
    for cls, stratum, w, sid, name, ad, best, unchanged, front, state in chosen:
        world, layout = worlds_data[w]
        if current != w:
            port = RoomsPort(config)
            port.reset(world)
            current = w
        set_state(port._env, layout, state)
        frames.append(_frame(port._env, config))
        rows.append([w, sid, goal_names.index(name), CLASSES.index(cls), front,
                     *np.minimum(ad, INF).tolist(), *best.astype(int).tolist(), *unchanged.astype(int).tolist()])
    arrays = {"fork_frames": np.stack(frames), "forks": np.array(rows, np.int64),
              "strata": np.array([c[1] for c in chosen])}
    worlds_data.clear()
    for n, items in pools.items():
        arrays[f"pool_{n}_world"] = np.array([i[0] for i in items], np.int64)
        arrays[f"pool_{n}_frames"] = (np.stack([i[2] for i in items]) if items
                                      else np.zeros((0, 42, 42, 3), np.uint8))
    np.savez_compressed(out / "sampled_probe.npz", **arrays)
    metadata = {"format": 1, "world": asdict(config), "seed": seed, "worlds": records, "goal_names": goal_names,
                "classes": list(CLASSES), "front_kinds": list(FRONT_KINDS), "strata": strata_counts,
                "pool_sizes": {n: len(v) for n, v in pools.items()},
                "fork_fields": ["world", "state", "goal", "class", "front_kind"] + [f"dist_{a}" for a in range(5)]
                + [f"best_{a}" for a in range(5)] + [f"no_effect_{a}" for a in range(5)],
                "inf": int(INF), "sha256": file_hash(out / "sampled_probe.npz"),
                "seconds": time.monotonic() - started}
    write_json(out / "sampled_probe.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development", "transfer_combo", "transfer_colour"),
                        default="development")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--play", type=float, default=0.0, help="play-start probability; 0 = ordinary starts")
    parser.add_argument("--sampled", action="store_true", help="stratified states from reachable sets")
    args = parser.parse_args()
    config = RoomsConfig(split=args.split, play_probability=args.play, max_steps=args.max_steps)
    if args.sampled:
        meta = sampled_probe(args.out, config, args.episodes, args.seed)
    else:
        meta = goal_probe(args.out, config, args.episodes, args.seed)
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
