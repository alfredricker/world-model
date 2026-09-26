"""Reward-free collection for chained rooms, with evaluator labels kept apart.

``episodes.npz`` holds only what the learner may see: RGB frames, opaque
action IDs, rewards and episode flags. ``labels.npz`` holds simulator-derived
per-step labels for the evaluator and is never loaded by a learner.
"""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np

from ..data import Episode, Replay, file_hash, write_json
from .rooms import EVENTS, FRONT_TYPES, DOOR_TYPES, RoomsConfig, RoomsPort, persistent_actions

MAX_DOORS, MAX_SWITCHES, MAX_BOXES = 3, 3, 5
DOOR_FIELDS = ("kind", "color", "open", "locked", "enabled", "in_view", "facing")
SWITCH_FIELDS = ("on", "in_view", "facing")
BOX_FIELDS = ("opened", "facing")


def _label_row(labels: dict) -> np.ndarray:
    """Flatten one label dict into a fixed-width int16 row."""
    row = [labels["room"], labels["rooms"], labels["x"], labels["y"], labels["dir"], labels["carrying"],
           FRONT_TYPES.index(labels["front_type"]), labels["front_color"], int(labels["play_start"]),
           labels["step"]]
    for i in range(MAX_DOORS):
        if i < len(labels["doors"]):
            d = labels["doors"][i]
            row += [DOOR_TYPES.index(d["kind"]), d["color"], int(d["open"]), int(d["locked"]),
                    int(d["enabled"]), int(d["in_view"]), int(d["facing"])]
        else:
            row += [-1] * len(DOOR_FIELDS)
    for i in range(MAX_SWITCHES):
        if i < len(labels["switches"]):
            s = labels["switches"][i]
            row += [int(s["on"]), int(s["in_view"]), int(s["facing"])]
        else:
            row += [-1] * len(SWITCH_FIELDS)
    for i in range(MAX_BOXES):
        if i < len(labels["boxes"]):
            b = labels["boxes"][i]
            row += [int(b["opened"]), int(b["facing"])]
        else:
            row += [-1] * len(BOX_FIELDS)
    return np.array(row, np.int16)


LABEL_COLUMNS = (["room", "rooms", "x", "y", "dir", "carrying", "front_type", "front_color", "play_start", "step"]
                 + [f"door{i}_{f}" for i in range(MAX_DOORS) for f in DOOR_FIELDS]
                 + [f"switch{i}_{f}" for i in range(MAX_SWITCHES) for f in SWITCH_FIELDS]
                 + [f"box{i}_{f}" for i in range(MAX_BOXES) for f in BOX_FIELDS])


def column(name: str) -> int:
    return LABEL_COLUMNS.index(name)


def world_seed(seed: int, split: str, index: int) -> int:
    split_id = {"train": 0, "development": 1, "transfer_combo": 2, "transfer_colour": 3}[split]
    return int(np.random.SeedSequence([seed, split_id, index]).generate_state(1)[0])


def run_episode(config: RoomsConfig, seed: int, index: int, repeat: float):
    """One reward-free episode with the persistent random policy."""
    port = RoomsPort(config)
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 173]))
    observation = port.reset(world_seed(seed, config.split, index))
    combination = port._env.combination()
    actions = persistent_actions(rng, config.max_steps, port.action_count, repeat)
    frames, rewards, terminated, truncated = [observation.rgb], [], [], []
    label_rows, event_rows = [_label_row(port.last_labels)], []
    taken = []
    for action in actions:
        observation = port.step(int(action))
        frames.append(observation.rgb)
        taken.append(int(action))
        rewards.append(observation.reward)
        terminated.append(observation.terminated)
        truncated.append(observation.truncated)
        label_rows.append(_label_row(port.last_labels))
        event_rows.append(np.array([port.last_events[e] for e in EVENTS], np.int8))
        if observation.terminated or observation.truncated:
            break
    port.close()
    episode = Episode(np.stack(frames), np.array(taken, np.int64), np.array(rewards, np.float32),
                      np.array(terminated, np.bool_), np.array(truncated, np.bool_))
    episode.validate(port.action_count)
    return episode, np.stack(label_rows), np.stack(event_rows), combination


def _worker(args):
    config_dict, seed, index, repeat = args
    config = RoomsConfig(**config_dict)
    episode, labels, events, combination = run_episode(config, seed, index, repeat)
    return index, episode, labels, events, combination


def collect(out: Path, config: RoomsConfig, episodes: int, seed: int, *, repeat: float = 0.5,
            workers: int = 8) -> dict:
    if episodes < 1 or seed < 0 or workers < 1:
        raise ValueError("Require positive episode count, non-negative seed, positive workers")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(out / "config.json", {"steps": episodes * config.max_steps, "phase": "collect",
                                    "episodes": episodes, "world": asdict(config), "seed": seed,
                                    "repeat": repeat, "policy": "uniform opaque IDs with geometric persistence"})
    source = Path(__file__).resolve().parent
    sources = {"data.py": source.parent / "data.py", "rooms.py": source / "rooms.py",
               "rooms_data.py": source / "rooms_data.py"}
    (out / "collector_source").mkdir()
    hashes = {}
    for name, path in sources.items():
        (out / "collector_source" / name).write_bytes(path.read_bytes())
        hashes[name] = file_hash(path)
    config_dict = asdict(config)
    config_dict["rooms"] = tuple(config_dict["rooms"])
    config_dict["lava_per_room"] = tuple(config_dict["lava_per_room"])
    jobs = [(config_dict, seed, i, repeat) for i in range(episodes)]
    arrays, label_arrays, records = {}, {}, []
    coverage = dict.fromkeys(EVENTS, 0)
    coverage_episodes = []
    try:
        with Pool(workers) as pool, (out / "metrics.jsonl").open("w") as stream:
            for index, episode, labels, events, combination in pool.imap(_worker, jobs, chunksize=4):
                for name in ("frames", "actions", "rewards", "terminated", "truncated"):
                    arrays[f"e{index}_{name}"] = getattr(episode, name)
                label_arrays[f"e{index}_labels"] = labels
                label_arrays[f"e{index}_events"] = events
                counts = {e: int(events[:, j].sum()) for j, e in enumerate(EVENTS)}
                for e, v in counts.items():
                    coverage[e] += v
                records.append({"episode": index, "seed": world_seed(seed, config.split, index),
                                "length": len(episode.actions), "reward": float(episode.rewards.sum()),
                                "play_start": bool(labels[0, column("play_start")])})
                coverage_episodes.append({"episode": index, "combination": combination, "events": counts,
                                          "length": len(episode.actions)})
                if len(records) % 50 == 0 or len(records) == episodes:
                    stream.write(json.dumps({"step": sum(r["length"] for r in records), "episodes": len(records),
                                             "seconds": time.monotonic() - started, "phase": "collect"}) + "\n")
                    stream.flush()
    except BaseException as error:
        write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error),
                                           "seconds": time.monotonic() - started})
        raise
    np.savez_compressed(out / "episodes.npz", **arrays)
    np.savez_compressed(out / "labels.npz", **label_arrays)
    verified = all(file_hash(sources[n]) == h for n, h in hashes.items())
    if not verified:
        raise RuntimeError("Collector source changed during collection")
    metadata = {"format": 1, "world": asdict(config), "seed": seed, "repeat": repeat,
                "policy": "uniform opaque action IDs with geometric persistence",
                "action_count": RoomsPort.action_count, "episodes": records,
                "transitions": sum(r["length"] for r in records), "sha256": file_hash(out / "episodes.npz"),
                "labels_sha256": file_hash(out / "labels.npz"), "label_columns": LABEL_COLUMNS,
                "events": list(EVENTS), "collector_sources": hashes,
                "dependencies": {n: version(n) for n in ("numpy", "minigrid", "gymnasium")}}
    write_json(out / "dataset.json", metadata)
    write_json(out / "coverage.json", coverage)
    write_json(out / "coverage_episodes.json", coverage_episodes)
    write_json(out / "metrics.json", {"status": "complete", "steps": metadata["transitions"],
                                       "episodes": episodes, "seconds": time.monotonic() - started,
                                       "coverage": coverage, "source_snapshot_verified": verified,
                                       "mean_length": metadata["transitions"] / episodes})
    return metadata


def load_labels(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Evaluator only. Returns per-episode label rows (T+1) and event rows (T)."""
    metadata = json.loads((path / "dataset.json").read_text())
    if file_hash(path / "labels.npz") != metadata["labels_sha256"]:
        raise ValueError("Label hash mismatch")
    with np.load(path / "labels.npz", allow_pickle=False) as archive:
        labels = [archive[f"e{i}_labels"] for i in range(len(metadata["episodes"]))]
        events = [archive[f"e{i}_events"] for i in range(len(metadata["episodes"]))]
    return labels, events


def load_rooms_replay(path: Path, *, training: bool = False) -> tuple[Replay, dict]:
    metadata = json.loads((path / "dataset.json").read_text())
    if metadata["format"] != 1 or "label_columns" not in metadata:
        raise ValueError("Not a chained-rooms replay")
    if training and metadata["world"]["split"] != "train":
        raise ValueError("Only the train split may enter training")
    if file_hash(path / "episodes.npz") != metadata["sha256"]:
        raise ValueError("Dataset hash mismatch")
    with np.load(path / "episodes.npz", allow_pickle=False) as archive:
        episodes = [Episode(**{n: archive[f"e{i}_{n}"] for n in ("frames", "actions", "rewards", "terminated", "truncated")})
                    for i in range(len(metadata["episodes"]))]
    return Replay(episodes, metadata["action_count"]), metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development", "transfer_combo", "transfer_colour"), default="train")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--play", type=float, default=0.5)
    parser.add_argument("--repeat", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = RoomsConfig(split=args.split, play_probability=args.play, max_steps=args.max_steps)
    metadata = collect(args.out, config, args.episodes, args.seed, repeat=args.repeat, workers=args.workers)
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("episodes", "label_columns")}, indent=2))


if __name__ == "__main__":
    main()
