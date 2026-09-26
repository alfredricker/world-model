"""Simulator-state observations for chained rooms. Evaluator side only.

The feasibility gate trains the hub with perception solved, to show the
rung-1 test can be passed. Two forms:

- **egocentric** (the gate's upper bound): MiniGrid's own encoding of the
  agent's 7x7 view (object type, colour, state per cell; unseen cells 0),
  the same cells the frames show, plus the carried object. It has the
  frames' geometry, so an action changes as much of it as of a frame.
- **allocentric**: the whole 8x29 world with hidden state (box contents,
  switch doors). Kept for reference; on it a step changes one cell and the
  transition model learned nothing (card 001 gate).

Frame learners never load these arrays.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

from ..data import file_hash, write_json
from .rooms import RoomsConfig, RoomsPort
from .rooms_data import world_seed
from .rooms_goals import Layout, layout_of, state_of

HEIGHT, WIDTH = 8, 29
CHANNELS = ("type", "colour", "door_kind", "door_status", "box_contents", "agent")
TYPES = ("empty", "wall", "lava", "goal", "door", "key", "box", "switch")
COLOURS = ("red", "green", "blue", "purple", "yellow", "grey")      # stored as index + 1; 0 = none
DOOR_KINDS = ("locked", "switch", "plain", "timed")                  # stored as index + 1
CARRY = ("none", "key", "box")
# Number of values each channel can take, for embedding tables.
SIZES = (len(TYPES), len(COLOURS) + 1, len(DOOR_KINDS) + 1, 4, len(COLOURS) + 1, 5)
CARRY_SIZES = (len(CARRY), len(COLOURS) + 1)


def _colour(c) -> int:
    return 0 if c is None else COLOURS.index(c) + 1


def static_grid(layout: Layout) -> np.ndarray:
    """The channels that never change within a world."""
    grid = np.zeros((HEIGHT, WIDTH, len(CHANNELS)), np.uint8)
    for x, y in layout.blocked:
        grid[y, x, 0] = TYPES.index("wall")
    for x, y in layout.lava:
        grid[y, x, 0] = TYPES.index("lava")
    gx, gy = layout.goal
    grid[gy, gx, 0] = TYPES.index("goal")
    for (x, y), door in layout.switches:
        grid[y, x, 0] = TYPES.index("switch")
        grid[y, x, 1] = _colour("grey")
    for (x, y), kind, colour, _ in layout.doors:
        grid[y, x, 0] = TYPES.index("door")
        grid[y, x, 1] = _colour(colour)
        grid[y, x, 2] = DOOR_KINDS.index(kind) + 1
    return grid


def symbolic(layout: Layout, state: tuple, base: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(grid (8, 29, 6) uint8, carried (2,) uint8) for one abstract state.

    door_status: 1 closed and locked (or a switch door not yet enabled),
    2 closed and openable, 3 open. Box contents are included: this is the
    full simulator state, not what the agent can see."""
    grid = (static_grid(layout) if base is None else base).copy()
    x, y, d, carrying, objects, doors, switches = state
    for ((px, py), kind, colour, _), (flag, is_open, _) in zip(layout.doors, doors, strict=True):
        # The first door field is "locked" for locked doors and "enabled" for switch doors.
        closed_blocked = flag if kind == "locked" else (not flag if kind == "switch" else False)
        grid[py, px, 3] = 3 if is_open else (1 if closed_blocked else 2)
    for ((sx, sy), _), on in zip(layout.switches, switches, strict=True):
        grid[sy, sx, 3] = 3 if on else 1
    for (ox, oy), item in objects:
        grid[oy, ox, 0] = TYPES.index(item[0])
        if item[0] == "key":
            grid[oy, ox, 1] = _colour(item[1])
        else:
            grid[oy, ox, 1] = _colour("grey")
            grid[oy, ox, 4] = _colour(item[1])
    grid[y, x, 5] = d + 1
    carried = np.zeros(2, np.uint8)
    if carrying is not None:
        carried[0] = CARRY.index(carrying[0])
        carried[1] = _colour(carrying[1])
    return grid, carried


EGO_SIZES = (11, 6, 3)          # MiniGrid OBJECT_TO_IDX, COLOR_TO_IDX, door state / switch on
EGO_CARRY_SIZES = (11, 6)


def egocentric(env) -> tuple[np.ndarray, np.ndarray]:
    """(view (7, 7, 3) uint8, carried (type, colour) uint8), as MiniGrid's
    gen_obs encodes it. The carried object also appears at the agent's cell."""
    grid, vis_mask = env.gen_obs_grid()
    view = grid.encode(vis_mask).astype(np.uint8)
    carried = np.zeros(2, np.uint8)
    if env.carrying is not None:
        carried[:] = env.carrying.encode()[:2]
    return view, carried


def _replay(args):
    config_dict, world, actions, frames_hash = args
    config = RoomsConfig(**config_dict)
    port = RoomsPort(config)
    rgb = [port.reset(world).rgb]
    env = port._env
    layout = layout_of(env)
    base = static_grid(layout)
    grids, carried, views, view_carried = [], [], [], []

    def record():
        g, c = symbolic(layout, state_of(env, layout), base)
        grids.append(g)
        carried.append(c)
        v, vc = egocentric(env)
        views.append(v)
        view_carried.append(vc)
    record()
    for a in actions:
        rgb.append(port.step(int(a)).rgb)
        record()
    port.close()
    import hashlib
    same = hashlib.sha256(np.stack(rgb).tobytes()).hexdigest() == frames_hash
    return np.stack(grids), np.stack(carried), np.stack(views), np.stack(view_carried), same


def annotate(dataset: Path, workers: int = 16) -> dict:
    """Replay every episode of a collected dataset and store its symbolic
    states in ``symbolic.npz``, checking the replayed frames match."""
    import hashlib
    started = time.monotonic()
    meta = json.loads((dataset / "dataset.json").read_text())
    config = dict(meta["world"])
    config["rooms"] = tuple(config["rooms"])
    config["lava_per_room"] = tuple(config["lava_per_room"])
    jobs = []
    with np.load(dataset / "episodes.npz", allow_pickle=False) as archive:
        for record in meta["episodes"]:
            i = record["episode"]
            frames = archive[f"e{i}_frames"]
            jobs.append((config, record["seed"], archive[f"e{i}_actions"],
                         hashlib.sha256(frames.tobytes()).hexdigest()))
    arrays, mismatched = {}, []
    with Pool(workers) as pool:
        for i, (grids, carried, views, view_carried, same) in enumerate(pool.imap(_replay, jobs, chunksize=8)):
            arrays[f"e{i}_grid"] = grids
            arrays[f"e{i}_carried"] = carried
            arrays[f"e{i}_ego_grid"] = views
            arrays[f"e{i}_ego_carried"] = view_carried
            if not same:
                mismatched.append(i)
    if mismatched:
        raise RuntimeError(f"Replayed frames differ in {len(mismatched)} episodes, e.g. {mismatched[:5]}")
    np.savez_compressed(dataset / "symbolic.npz", **arrays)
    info = {"format": 2, "channels": list(CHANNELS), "sizes": list(SIZES), "carry_sizes": list(CARRY_SIZES),
            "episodes": len(jobs), "sha256": file_hash(dataset / "symbolic.npz"),
            "dataset_sha256": meta["sha256"], "seconds": time.monotonic() - started}
    write_json(dataset / "symbolic.json", info)
    return info


def load_symbolic(dataset: Path, prefix: str = "") -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-episode (grid, carried) arrays; prefix "ego_" for the egocentric view."""
    info = json.loads((dataset / "symbolic.json").read_text())
    if file_hash(dataset / "symbolic.npz") != info["sha256"]:
        raise ValueError("Symbolic hash mismatch")
    with np.load(dataset / "symbolic.npz", allow_pickle=False) as archive:
        grids = [archive[f"e{i}_{prefix}grid"] for i in range(info["episodes"])]
        carried = [archive[f"e{i}_{prefix}carried"] for i in range(info["episodes"])]
    return grids, carried


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(annotate(args.dataset, args.workers), indent=2))


if __name__ == "__main__":
    main()
