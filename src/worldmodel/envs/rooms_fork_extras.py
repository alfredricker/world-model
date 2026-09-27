"""Extra evaluator material for the goal-conditioned fork (card 001).

Built beside an existing ``sampled_probe`` directory, from the same worlds:

- **Fork states** as simulator state, for the feasibility gate, and each
  fork's true steps-to-goal. Every fork frame is re-rendered from its state
  and checked against the stored frame.
- **Successors:** the observation after each of the 5 actions (the
  current one again when an action ends the episode), to score the
  reachability head on true next states apart from the transition model.
- **Decidability from the view** (card 001, user decision 2026-09-26): all
  live reachable states of all development worlds that give the same
  egocentric view (and carried object) are grouped. A fork is decidable
  when some action is best, for the fork's goal, in every state of its
  group, so a model without memory, which does not know which world it is
  in, can always be right. Where a goal is unreachable or absent, every
  action counts as best. ``decidable_within_world`` groups only the fork's
  own world (optimistic; reported for reference).
- **Exact goals:** for each fork, the first goal state on a shortest path
  (lowest action ID on ties), as a frame and as simulator state. The gate
  compares pooled 4-example goals from other worlds against these.
- **Goal pools** with both a frame and simulator state per example.
- **Matched condition pairs** for the condition-gap diagnostic: two states
  of one world identical except for one change, with the true steps to a
  goal from each. ``key``: holding the goal door's key, against the key
  back where it started (on the floor or in its box). ``box``: the key's
  box opened, against unopened, for the key-held goal. ``irrelevant``:
  a plain door open against closed, for a goal of another colour.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

from ..data import file_hash, write_json
from .rooms import RoomsConfig, RoomsPort
from .rooms_data import world_seed
from .rooms_goals import (INF, _frame, all_action_distances, distances, goals_of, holds, layout_of, reachable,
                          set_state, state_of, visible_goals)
from .rooms_symbolic import egocentric, static_grid, symbolic

PAIR_KINDS = ("key", "box", "irrelevant")


def shortest_path_goal(graph, dist: np.ndarray, sid: int) -> int:
    """The first state where the goal holds on a shortest path from ``sid``."""
    cur = sid
    while dist[cur] > 0:
        for a in range(5):
            j = graph.succ[cur, a]
            if j >= 0 and dist[j] == dist[cur] - 1:
                cur = int(j)
                break
        else:
            raise RuntimeError("No shortest-path successor")
    return cur


def _start_item(start: tuple, colour: str):
    """Where the key of ``colour`` starts: (pos, item) on the floor, key or box."""
    for pos, item in start[4]:
        if item[1] == colour and item[0] in ("key", "box"):
            return pos, item
    return None


def _pairs(layout, graph, start, dist, rng, per_kind: int, scan: int):
    """Up to ``per_kind`` matched pairs of each kind: (kind, goal, sid_a, sid_b),
    where a has the condition and b does not."""
    out = {k: [] for k in PAIR_KINDS}
    live = np.flatnonzero(~graph.terminal)
    doors = {d[2]: (i, d) for i, d in enumerate(layout.doors)}
    for sid in rng.permutation(live)[:scan]:
        if all(len(v) >= per_kind for v in out.values()):
            break
        st = graph.states[sid]
        x, y, d, carrying, objects, door_states, switches = st
        # key: holding the key of a locked door that is still locked
        if len(out["key"]) < per_kind and carrying is not None and carrying[0] == "key":
            c = carrying[1]
            if c in doors and doors[c][1][1] == "locked" and door_states[doors[c][0]][0]:
                origin = _start_item(start, c)
                if origin is not None and origin[0] not in dict(objects):
                    other = (x, y, d, None, tuple(sorted(objects + (origin,))), door_states, switches)
                    j = graph.index.get(other)
                    g = ("door_open", c)
                    if j is not None and g in dist and dist[g][sid] != INF:
                        out["key"].append(("key", g, int(sid), int(j)))
        # box: the key's box opened (key on the floor where the box was) or not
        if len(out["box"]) < per_kind:
            floor = dict(objects)
            for pos, item in objects:
                if item[0] != "key":
                    continue
                origin = _start_item(start, item[1])
                if origin is None or origin[1][0] != "box" or origin[0] != pos:
                    continue
                boxed = dict(floor)
                boxed[pos] = ("box", item[1])
                other = (x, y, d, carrying, tuple(sorted(boxed.items())), door_states, switches)
                j = graph.index.get(other)
                g = ("key_held", item[1])
                if j is not None and g in dist and dist[g][sid] != INF:
                    out["box"].append(("box", g, int(sid), int(j)))
                    break
        # irrelevant: a plain door of another colour than the goal, open vs closed
        if len(out["irrelevant"]) < per_kind:
            for i, (pos, kind, colour, _) in enumerate(layout.doors):
                if kind != "plain":
                    continue
                other_goals = [g for g in dist if g[1] != colour]
                if not other_goals:
                    continue
                flipped = list(door_states)
                flipped[i] = (door_states[i][0], not door_states[i][1], door_states[i][2])
                other = (x, y, d, carrying, objects, tuple(flipped), switches)
                j = graph.index.get(other)
                g = other_goals[int(rng.integers(len(other_goals)))]
                a_sid, b_sid = (sid, j) if door_states[i][1] else (j, sid)   # a: the door open
                if j is not None and dist[g][a_sid] != INF:
                    out["irrelevant"].append(("irrelevant", g, int(a_sid), int(b_sid)))
                    break
    return [p for k in PAIR_KINDS for p in out[k][:per_kind]]


def _world(args):
    (config_dict, seed, w, rows, frames, goal_names, per_goal, per_kind, scan, cap) = args
    config = RoomsConfig(**config_dict)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 997, w]))
    port = RoomsPort(config)
    port.reset(world_seed(seed + 700000, config.split, w))
    env = port._env
    layout = layout_of(env)
    base = static_grid(layout)
    start = state_of(env, layout)
    graph = reachable(layout, start, cap)
    goals = goals_of(layout, start)
    dist = {g: distances(layout, graph, g) for g in goals}
    # Egocentric view of every live state, grouped by identical view.
    live_ids = np.flatnonzero(~graph.terminal)
    groups = {}
    view_key = np.full(len(graph.states), -1, np.int64)
    for sid in live_ids:
        set_state(env, layout, graph.states[sid])
        view, view_carried = egocentric(env)
        key = view.tobytes() + view_carried.tobytes()
        view_key[sid] = groups.setdefault(key, len(groups))
    members = [[] for _ in range(len(groups))]
    for sid in live_ids:
        members[view_key[sid]].append(sid)
    members = [np.array(m) for m in members]
    best_rows = {}
    for g, dg in dist.items():
        ad = all_action_distances(graph, dg)
        low = ad.min(1, keepdims=True)
        best_rows[g] = (ad == low)          # all True where the goal is unreachable
    # Per view group, the actions best in every member, as 5-bit masks per goal
    # name (31 = no constraint: goal absent from this world).
    weights = 1 << np.arange(5)
    group_masks = np.full((len(groups), len(goal_names)), 31, np.uint8)
    for g, best_g in best_rows.items():
        gi = goal_names.index(f"{g[0]}:{g[1]}")
        for k, m in enumerate(members):
            group_masks[k, gi] = int((best_g[m].all(0) * weights).sum())
    import hashlib
    group_hash = np.zeros(len(groups), np.int64)
    for key, k in groups.items():
        group_hash[k] = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little", signed=True)

    def render(sid):
        st = graph.states[sid]
        set_state(env, layout, st)
        grid, carried = symbolic(layout, st, base)
        view, view_carried = egocentric(env)
        return _frame(env, config), grid, carried, st, view, view_carried

    fork = {"index": [], "grid": [], "carried": [], "steps": [], "goal_frame": [], "goal_grid": [],
            "goal_carried": [], "goal_visible": [], "frame_ok": [], "next_frame": [], "next_grid": [],
            "next_carried": [], "ego_grid": [], "ego_carried": [], "goal_ego_grid": [], "goal_ego_carried": [],
            "next_ego_grid": [], "next_ego_carried": [], "decidable_within_world": [], "view_group": [],
            "view_hash": []}
    for (index, sid, goal), stored in zip(rows, frames, strict=True):
        g = tuple(goal_names[goal].split(":"))
        frame, grid, carried, _, view, view_carried = render(sid)
        fork["index"].append(index)
        fork["frame_ok"].append(bool(np.array_equal(frame, stored)))
        fork["grid"].append(grid)
        fork["carried"].append(carried)
        fork["ego_grid"].append(view)
        fork["ego_carried"].append(view_carried)
        fork["steps"].append(int(dist[g][sid]))
        group = members[view_key[sid]]
        fork["decidable_within_world"].append(bool(best_rows[g][group].all(0).any()))
        fork["view_group"].append(len(group))
        fork["view_hash"].append(group_hash[view_key[sid]])
        succ = [render(int(j) if j >= 0 else sid) for j in graph.succ[sid]]
        fork["next_frame"].append(np.stack([s_[0] for s_ in succ]))
        fork["next_grid"].append(np.stack([s_[1] for s_ in succ]))
        fork["next_carried"].append(np.stack([s_[2] for s_ in succ]))
        fork["next_ego_grid"].append(np.stack([s_[4] for s_ in succ]))
        fork["next_ego_carried"].append(np.stack([s_[5] for s_ in succ]))
        target = shortest_path_goal(graph, dist[g], sid)
        gframe, ggrid, gcarried, gstate, gview, gview_carried = render(target)
        fork["goal_ego_grid"].append(gview)
        fork["goal_ego_carried"].append(gview_carried)
        fork["goal_frame"].append(gframe)
        fork["goal_grid"].append(ggrid)
        fork["goal_carried"].append(gcarried)
        fork["goal_visible"].append((g[0], g[1]) in visible_goals(env, layout, gstate))
    # Goal pools: states where a goal holds and is visible.
    pools = {}
    live = np.flatnonzero(~graph.terminal)
    found = {f"{k}:{c}": 0 for k, c in goals}
    for sid in rng.permutation(live)[:20000]:
        st = graph.states[sid]
        wanted = [n for n in found if found[n] < per_goal and holds(layout, st, tuple(n.split(":")))]
        if not wanted:
            if all(v >= per_goal for v in found.values()):
                break
            continue
        set_state(env, layout, st)
        visible = {f"{k}:{c}" for k, c in visible_goals(env, layout, st)}
        for n in wanted:
            if n in visible:
                grid, carried = symbolic(layout, st, base)
                view, view_carried = egocentric(env)
                pools.setdefault(n, []).append((_frame(env, config), grid, carried, view, view_carried))
                found[n] += 1
    pairs = []
    for kind, g, a, b in _pairs(layout, graph, start, dist, rng, per_kind, scan):
        fa, ga, ca, _, va, vca = render(a)
        fb, gb, cb, _, vb, vcb = render(b)
        pairs.append((PAIR_KINDS.index(kind), goal_names.index(f"{g[0]}:{g[1]}"),
                      int(dist[g][a]), int(dist[g][b]), fa, ga, ca, va, vca, fb, gb, cb, vb, vcb))
    port.close()
    return w, fork, pools, pairs, (group_hash, group_masks)


def build(probe: Path, out: Path, *, per_goal: int = 4, per_kind: int = 3, scan: int = 40000,
          cap: int = 2_000_000, workers: int = 12) -> dict:
    started = time.monotonic()
    meta = json.loads((probe / "sampled_probe.json").read_text())
    if file_hash(probe / "sampled_probe.npz") != meta["sha256"]:
        raise ValueError("Probe hash mismatch")
    with np.load(probe / "sampled_probe.npz", allow_pickle=False) as archive:
        forks = archive["forks"]
        fork_frames = archive["fork_frames"]
    config = dict(meta["world"])
    config["rooms"] = tuple(config["rooms"])
    config["lava_per_room"] = tuple(config["lava_per_room"])
    goal_names = meta["goal_names"]
    jobs = []
    for w in range(len(meta["worlds"])):
        idx = np.flatnonzero(forks[:, 0] == w)
        rows = [(int(i), int(forks[i, 1]), int(forks[i, 2])) for i in idx]
        jobs.append((config, meta["seed"], w, rows, fork_frames[idx], goal_names, per_goal, per_kind, scan, cap))
    n = len(forks)
    fork_out = {"grid": np.zeros((n, 8, 29, 6), np.uint8), "carried": np.zeros((n, 2), np.uint8),
                "steps": np.zeros(n, np.int64), "goal_frame": np.zeros((n, 42, 42, 3), np.uint8),
                "goal_grid": np.zeros((n, 8, 29, 6), np.uint8), "goal_carried": np.zeros((n, 2), np.uint8),
                "goal_visible": np.zeros(n, np.bool_), "next_frame": np.zeros((n, 5, 42, 42, 3), np.uint8),
                "next_grid": np.zeros((n, 5, 8, 29, 6), np.uint8), "next_carried": np.zeros((n, 5, 2), np.uint8),
                "ego_grid": np.zeros((n, 7, 7, 3), np.uint8), "ego_carried": np.zeros((n, 2), np.uint8),
                "goal_ego_grid": np.zeros((n, 7, 7, 3), np.uint8), "goal_ego_carried": np.zeros((n, 2), np.uint8),
                "next_ego_grid": np.zeros((n, 5, 7, 7, 3), np.uint8),
                "next_ego_carried": np.zeros((n, 5, 2), np.uint8),
                "decidable_within_world": np.zeros(n, np.bool_), "view_group": np.zeros(n, np.int64),
                "view_hash": np.zeros(n, np.int64)}
    pools = {name: [] for name in goal_names}
    pairs, frame_ok = [], np.zeros(n, np.bool_)
    with Pool(workers) as pool:
        view_masks = {}
        for w, fork, world_pools, world_pairs, (hashes, masks) in pool.imap_unordered(_world, jobs):
            for h, m in zip(hashes.tolist(), masks):
                view_masks[h] = view_masks[h] & m if h in view_masks else m.copy()
            idx = np.array(fork["index"], np.int64)
            if len(idx):
                frame_ok[idx] = fork["frame_ok"]
                for k in fork_out:
                    fork_out[k][idx] = np.array(fork[k])
            for name, items in world_pools.items():
                pools[name] += [(w, *item) for item in items]
            pairs += [(w, *p) for p in world_pairs]
    if not frame_ok.all():
        raise RuntimeError(f"{(~frame_ok).sum()} fork frames did not re-render identically")
    # Across all worlds: the actions best in every state that shows the fork's view.
    bits = np.array([view_masks[h][g] for h, g in zip(fork_out["view_hash"].tolist(), forks[:, 2])], np.uint8)
    fork_out["view_best"] = ((bits[:, None] >> np.arange(5)) & 1).astype(np.bool_)
    fork_out["decidable"] = fork_out["view_best"].any(1)
    fork_best = forks[:, 10:15].astype(bool)
    if (fork_out["view_best"] & ~fork_best).any():
        raise RuntimeError("An action best in every state with the view is not best at the fork")
    arrays = {f"fork_{k}": v for k, v in fork_out.items()}
    for name, items in pools.items():
        items.sort(key=lambda i: i[0])
        arrays[f"pool_{name}_world"] = np.array([i[0] for i in items], np.int64)
        arrays[f"pool_{name}_frames"] = (np.stack([i[1] for i in items]) if items else np.zeros((0, 42, 42, 3), np.uint8))
        arrays[f"pool_{name}_grid"] = (np.stack([i[2] for i in items]) if items else np.zeros((0, 8, 29, 6), np.uint8))
        arrays[f"pool_{name}_carried"] = (np.stack([i[3] for i in items]) if items else np.zeros((0, 2), np.uint8))
        arrays[f"pool_{name}_ego_grid"] = (np.stack([i[4] for i in items]) if items else np.zeros((0, 7, 7, 3), np.uint8))
        arrays[f"pool_{name}_ego_carried"] = (np.stack([i[5] for i in items]) if items else np.zeros((0, 2), np.uint8))
    pairs.sort(key=lambda p: (p[0], p[1]))
    arrays["pair_rows"] = np.array([[p[0], p[1], p[2], p[3], p[4]] for p in pairs], np.int64).reshape(-1, 5)
    for side, offset in (("a", 5), ("b", 10)):
        arrays[f"pair_{side}_frame"] = np.stack([p[offset] for p in pairs])
        arrays[f"pair_{side}_grid"] = np.stack([p[offset + 1] for p in pairs])
        arrays[f"pair_{side}_carried"] = np.stack([p[offset + 2] for p in pairs])
        arrays[f"pair_{side}_ego_grid"] = np.stack([p[offset + 3] for p in pairs])
        arrays[f"pair_{side}_ego_carried"] = np.stack([p[offset + 4] for p in pairs])
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "fork_extras.npz", **arrays)
    rows = arrays["pair_rows"]
    info = {"format": 1, "probe_sha256": meta["sha256"], "world": asdict(RoomsConfig(**config)),
            "pair_kinds": list(PAIR_KINDS), "pair_fields": ["world", "kind", "goal", "steps_a", "steps_b"],
            "pair_counts": {k: int((rows[:, 1] == i).sum()) for i, k in enumerate(PAIR_KINDS)},
            "pool_sizes": {k: len(v) for k, v in pools.items()},
            "exact_goal_visible": float(fork_out["goal_visible"].mean()),
            "decidable": float(fork_out["decidable"].mean()),
            "sha256": file_hash(out / "fork_extras.npz"), "seconds": time.monotonic() - started}
    write_json(out / "fork_extras.json", info)
    return info


def load_extras(path: Path) -> tuple[dict, dict]:
    info = json.loads((path / "fork_extras.json").read_text())
    if file_hash(path / "fork_extras.npz") != info["sha256"]:
        raise ValueError("Extras hash mismatch")
    with np.load(path / "fork_extras.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return arrays, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(build(args.probe, args.out, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
