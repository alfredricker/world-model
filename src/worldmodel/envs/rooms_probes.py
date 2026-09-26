"""Evaluator screens for chained rooms: consequence forks and feasibility.

``probe`` records ordinary-start episodes and, at evaluator-selected steps,
forks the simulator for every action to obtain the actual next frame of each.
Strata are evaluator labels derived from simulator state; they never enter the
learner.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from ..data import file_hash, write_json
from .rooms import ACTION_MAP, EVENTS, RoomsConfig, RoomsPort, persistent_actions
from .rooms_data import world_seed

FORK_FIELDS = ("front_type", "front_kind", "front_color", "carrying", "match", "door_open", "door_enabled",
               "switch_on", "switch_in_view", "since_switch", "box_opened", "rooms", "room", "step")
FRONT_KINDS = ("none", "wall", "locked", "switch", "plain", "timed", "key", "box", "switchobj", "lava", "goal")


def _front(env) -> tuple[str, int, dict]:
    """Describe the cell in front for stratification."""
    from minigrid.core.constants import COLOR_TO_IDX
    from minigrid.core.world_object import Box, Door, Key, Lava, Goal
    from .rooms import Switch, SwitchDoor, TimedDoor
    obj = env.grid.get(*env.front_pos)
    info = {"door_open": -1, "door_enabled": -1, "switch_on": -1, "switch_in_view": -1, "since_switch": -1, "box_opened": -1}
    if obj is None:
        return "none", -1, info
    color = COLOR_TO_IDX[obj.color]
    if isinstance(obj, Door):
        record = next(r for r in env.doors if r.pos == tuple(env.front_pos))
        info["door_open"] = int(obj.is_open)
        if isinstance(obj, SwitchDoor):
            switch = env.grid.get(*record.switch_pos)
            info["door_enabled"] = int(obj.enabled)
            info["switch_on"] = int(switch.on)
            info["switch_in_view"] = int(env.agent_sees(*record.switch_pos))
            return "switch", color, info
        if isinstance(obj, TimedDoor):
            return "timed", color, info
        return ("locked" if obj.is_locked else "plain") if record.kind == "locked" else record.kind, color, info
    if isinstance(obj, Key):
        return "key", color, info
    if isinstance(obj, Box):
        return "box", color, info
    if isinstance(obj, Switch):
        info["switch_on"] = int(obj.on)
        return "switchobj", color, info
    if isinstance(obj, Lava):
        return "lava", color, info
    if isinstance(obj, Goal):
        return "goal", color, info
    return "wall", color, info


def probe(out: Path, config: RoomsConfig, episodes: int, seed: int, *, ordinary_rate: float = 0.05,
          repeat: float = 0.5) -> dict:
    """Episodes with all-action forks at evaluator-chosen steps.

    Ordinary starts are the deployment condition; a play-start probe set is a
    declared diagnostic with more decisive events."""
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    from minigrid.core.constants import COLOR_TO_IDX
    rng = np.random.default_rng(np.random.SeedSequence([seed, 977]))
    arrays, records, fork_count = {}, [], 0
    switch_press_step: dict[int, int] = {}
    for index in range(episodes):
        port = RoomsPort(config)
        observation = port.reset(world_seed(seed + 500000, config.split, index))
        env = port._env
        frames, actions, forks, fork_steps, fork_labels, terminated = [observation.rgb], [], [], [], [], []
        switch_press_step.clear()
        policy = persistent_actions(rng, config.max_steps, port.action_count, repeat)
        for t, action in enumerate(policy):
            kind, color, info = _front(env)
            interesting = kind not in ("none", "wall") or rng.random() < ordinary_rate
            if interesting:
                outcomes, deaths = [], []
                for a in range(port.action_count):
                    branch = copy.deepcopy(env)
                    _, reward, term, _, _ = branch.step(ACTION_MAP[a])
                    outcomes.append(branch.get_frame(tile_size=config.tile_size, agent_pov=True).copy())
                    deaths.append(int(term and reward == 0))
                carrying = -1 if env.carrying is None else COLOR_TO_IDX[env.carrying.color]
                match = int(kind == "locked" and carrying == color)
                since = -1
                if kind == "switch" and info["switch_on"] == 1:
                    record = next(r for r in env.doors if r.pos == tuple(env.front_pos))
                    # -2: pressed before the episode began (play starts), delay unknown.
                    since = t - switch_press_step[record.index] if record.index in switch_press_step else -2
                labels = [FRONT_KINDS.index(kind), FRONT_KINDS.index(kind), color, carrying, match, info["door_open"],
                          info["door_enabled"], info["switch_on"], info["switch_in_view"], since, info["box_opened"],
                          env.rooms, env.room_of(env.agent_pos[0]), t]
                forks.append(np.stack(outcomes))
                fork_steps.append(t)
                fork_labels.append(np.array(labels + deaths, np.int32))
                fork_count += 1
            observation = port.step(int(action))
            if port.last_events["switch_press"]:
                for r in env.doors:
                    if r.kind == "switch" and env.grid.get(*r.switch_pos).on and r.index not in switch_press_step:
                        switch_press_step[r.index] = t + 1
            frames.append(observation.rgb)
            actions.append(int(action))
            terminated.append(observation.terminated)
            if observation.terminated or observation.truncated:
                break
        port.close()
        arrays[f"e{index}_frames"] = np.stack(frames)
        arrays[f"e{index}_actions"] = np.array(actions, np.int64)
        arrays[f"e{index}_terminated"] = np.array(terminated, np.bool_)
        arrays[f"e{index}_fork_frames"] = np.stack(forks) if forks else np.zeros((0, port.action_count) + frames[0].shape, np.uint8)
        arrays[f"e{index}_fork_steps"] = np.array(fork_steps, np.int64)
        arrays[f"e{index}_fork_labels"] = np.stack(fork_labels) if fork_labels else np.zeros((0, len(FORK_FIELDS) + port.action_count), np.int32)
        records.append({"episode": index, "length": len(actions), "forks": len(fork_steps), "combination": env.combination()})
    np.savez_compressed(out / "probe.npz", **arrays)
    metadata = {"format": 1, "world": asdict(config), "seed": seed, "episodes": records, "forks": fork_count,
                "fork_fields": list(FORK_FIELDS) + [f"death_{a}" for a in range(RoomsPort.action_count)],
                "front_kinds": list(FRONT_KINDS), "sha256": file_hash(out / "probe.npz"),
                "seconds": time.monotonic() - started}
    write_json(out / "probe.json", metadata)
    return metadata


def load_probe(path: Path):
    metadata = json.loads((path / "probe.json").read_text())
    if file_hash(path / "probe.npz") != metadata["sha256"]:
        raise ValueError("Probe hash mismatch")
    archive = np.load(path / "probe.npz", allow_pickle=False)
    episodes = []
    for r in metadata["episodes"]:
        i = r["episode"]
        episodes.append({k: archive[f"e{i}_{k}"] for k in ("frames", "actions", "terminated", "fork_frames", "fork_steps", "fork_labels")})
    return episodes, metadata


# `score` (fork identification by a frozen learner) is not ported: the new repo has not chosen a learner.


def stratify(rows: list[dict], kinds: list[str], action_count: int) -> dict:
    pickup, toggle = 3, action_count - 1

    def summary(selected):
        n = len(selected)
        return {"n": n, "fork_top1": float(np.mean([r["correct"] for r in selected])) if n else None}

    def death_summary(selected):
        pos = [r["death_prob"] for r in selected if r["death"]]
        neg = [r["death_prob"] for r in selected if not r["death"]]
        if not pos or not neg:
            return {"n_death": len(pos), "n_alive": len(neg), "auc": None}
        pos, neg = np.array(pos), np.array(neg)
        auc = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
        return {"n_death": len(pos), "n_alive": len(neg), "auc": auc}

    changed = [r for r in rows if r["changed"]]
    report = {"all": summary(rows), "changed_only": summary(changed),
              "by_action": {str(a): summary([r for r in changed if r["action"] == a]) for a in range(action_count)},
              "by_front": {}, "death": death_summary(rows)}
    for k, name in enumerate(kinds):
        sel = [r for r in changed if r["front_kind"] == k]
        if sel:
            report["by_front"][name] = summary(sel)
    lk, sw, bx, ky = (kinds.index(n) for n in ("locked", "switch", "box", "key"))
    toggles = [r for r in rows if r["action"] == toggle]
    # Relation screen: the decisive toggle at a locked door, with and without the matching key.
    report["toggle_at_locked_match"] = summary([r for r in toggles if r["front_kind"] == lk and r["match"] == 1])
    report["toggle_at_locked_mismatch"] = summary([r for r in toggles if r["front_kind"] == lk and r["match"] == 0])
    # Memory screen: the decisive toggle at a closed switch door; the frame is identical either way.
    closed = [r for r in toggles if r["front_kind"] == sw and r["door_open"] == 0]
    report["toggle_at_switch_enabled"] = summary([r for r in closed if r["door_enabled"] == 1])
    report["toggle_at_switch_disabled"] = summary([r for r in closed if r["door_enabled"] == 0])
    report["toggle_at_switch_enabled_switch_out_of_view"] = summary(
        [r for r in closed if r["door_enabled"] == 1 and r["switch_in_view"] == 0])
    for lo, hi, name in ((0, 16, "since_switch_0_16"), (17, 48, "since_switch_17_48"), (49, 10 ** 6, "since_switch_49_plus")):
        report[name] = summary([r for r in closed if r["door_enabled"] == 1 and lo <= r["since_switch"] <= hi])
    report["since_switch_before_episode"] = summary([r for r in closed if r["door_enabled"] == 1 and r["since_switch"] == -2])
    report["toggle_at_box"] = summary([r for r in toggles if r["front_kind"] == bx])
    report["pickup_at_key"] = summary([r for r in rows if r["action"] == pickup and r["front_kind"] == ky])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("train", "development", "transfer_combo", "transfer_colour"), default="development")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--play", type=float, default=0.0, help="play-start probability; 0 = ordinary starts")
    args = parser.parse_args()
    config = RoomsConfig(split=args.split, play_probability=args.play, max_steps=args.max_steps)
    meta = probe(args.out, config, args.episodes, args.seed)
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
