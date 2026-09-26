"""True steps-to-condition for every frame of a collected dataset.

Evaluator side only. For each episode, the world's reachable states are
searched from the episode's start (as ``rooms_goals`` does for development
worlds), and each frame's state gets the true fewest steps to each of the 8
probe conditions (key held / door open x red, green, blue, grey; INF where
the condition cannot be reached or does not exist in that world). Used only
for oracle fits that show a test is passable (CHARTER rule 7), never as a
training signal for a result.
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
from .rooms_goals import GOAL_KINDS, INF, distances, goals_of, layout_of, reachable, state_of

COLOURS = ("red", "green", "blue", "grey")
CONDITIONS = tuple((k, c) for k in GOAL_KINDS for c in COLOURS)   # the probe's goal_names order


def _episode(args):
    config_dict, world, actions, cap = args
    config = RoomsConfig(**config_dict)
    port = RoomsPort(config)
    port.reset(world)
    env = port._env
    layout = layout_of(env)
    start = state_of(env, layout)
    graph = reachable(layout, start, cap)
    present = set(goals_of(layout, start))
    dist = {g: distances(layout, graph, g) for g in CONDITIONS if g in present}
    ids = [graph.index[start]]
    for a in actions:
        port.step(int(a))
        ids.append(graph.index[state_of(env, layout)])
    port.close()
    ids = np.array(ids)
    out = np.full((len(ids), len(CONDITIONS)), INF, np.int64)
    for j, g in enumerate(CONDITIONS):
        if g in dist:
            out[:, j] = dist[g][ids]
    return out.astype(np.int32), graph.complete, len(graph.states)


def annotate(dataset: Path, workers: int = 12, cap: int = 2_000_000) -> dict:
    started = time.monotonic()
    meta = json.loads((dataset / "dataset.json").read_text())
    config = dict(meta["world"])
    config["rooms"] = tuple(config["rooms"])
    config["lava_per_room"] = tuple(config["lava_per_room"])
    with np.load(dataset / "episodes.npz", allow_pickle=False) as archive:
        jobs = [(config, r["seed"], archive[f"e{r['episode']}_actions"], cap) for r in meta["episodes"]]
    arrays, incomplete, sizes = {}, 0, []
    with Pool(workers) as pool:
        for i, (d, complete, n) in enumerate(pool.imap(_episode, jobs, chunksize=4)):
            arrays[f"e{i}"] = d
            incomplete += not complete
            sizes.append(n)
    np.savez_compressed(dataset / "true_dist.npz", **arrays)
    info = {"format": 1, "conditions": [f"{k}:{c}" for k, c in CONDITIONS], "inf": int(INF),
            "episodes": len(jobs), "incomplete_searches": incomplete, "max_states": int(max(sizes)),
            "sha256": file_hash(dataset / "true_dist.npz"), "dataset_sha256": meta["sha256"],
            "seconds": time.monotonic() - started}
    write_json(dataset / "true_dist.json", info)
    return info


def load_true_distances(dataset: Path) -> list[np.ndarray]:
    info = json.loads((dataset / "true_dist.json").read_text())
    if file_hash(dataset / "true_dist.npz") != info["sha256"]:
        raise ValueError("True-distance hash mismatch")
    with np.load(dataset / "true_dist.npz", allow_pickle=False) as archive:
        return [archive[f"e{i}"] for i in range(info["episodes"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(annotate(args.dataset, args.workers), indent=2))


if __name__ == "__main__":
    main()
