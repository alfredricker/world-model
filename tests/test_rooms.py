"""Chained rooms: mechanics, labels, learner information boundaries and probes."""
from pathlib import Path

import numpy as np
import pytest
from minigrid.core.constants import COLOR_TO_IDX
from minigrid.core.world_object import Door, Key

from worldmodel.envs.rooms import ChainedRoomsEnv, RoomsConfig, RoomsPort, Switch, SwitchDoor, TimedDoor, persistent_actions
from worldmodel.envs.rooms_data import collect, column, load_labels, load_rooms_replay
from worldmodel.envs.rooms_probes import load_probe, probe

ACTIONS = dict(left=0, right=1, forward=2, pickup=3, toggle=5)  # raw MiniGrid IDs for env.step


def _face(env, target):
    """Evaluator helper: place the agent adjacent to a cell, facing it."""
    x, y = target
    for dx, dy, d in ((-1, 0, 0), (0, -1, 1), (1, 0, 2), (0, 1, 3)):
        pos = (x + dx, y + dy)
        if env.grid.get(*pos) is None and 0 < pos[0] < env.width and 0 < pos[1] < env.height:
            env.agent_pos = np.array(pos)
            env.agent_dir = d
            return
    raise RuntimeError("No free adjacent cell")


def _env(split="train", seed=0, **kw):
    env = ChainedRoomsEnv(RoomsConfig(split=split, **kw))
    env.reset(seed=seed)
    return env


def test_switch_door_needs_its_switch():
    for seed in range(20):
        env = _env(seed=seed)
        records = [r for r in env.doors if r.kind == "switch"]
        if not records:
            continue
        r = records[0]
        _face(env, r.pos)
        assert isinstance(r.obj, SwitchDoor) and not r.obj.is_open
        env.step(ACTIONS["toggle"])
        assert not r.obj.is_open, "switch door opened without its switch"
        switch = env.grid.get(*r.switch_pos)
        assert isinstance(switch, Switch)
        _face(env, r.switch_pos)
        env.step(ACTIONS["toggle"])
        assert switch.on and r.obj.enabled
        _face(env, r.pos)
        env.step(ACTIONS["toggle"])
        assert r.obj.is_open
        return
    pytest.skip("no switch door in twenty seeds")


def test_switch_door_renders_identically_before_and_after_switch():
    door = SwitchDoor()
    before = door.encode()
    door.enabled = True
    assert door.encode() == before


def test_locked_door_needs_matching_key_and_consumes_it():
    for seed in range(30):
        env = _env(seed=seed)
        r = next(x for x in env.doors if x.kind == "locked")
        _face(env, r.pos)
        env.step(ACTIONS["toggle"])
        assert r.obj.is_locked
        other = next((c for c in env.config.colours if c != r.color), None)
        env.carrying = Key(other)
        env.step(ACTIONS["toggle"])
        assert r.obj.is_locked, "wrong colour unlocked the door"
        env.carrying = Key(r.color)
        _, _, _, _, info = env.step(ACTIONS["toggle"])
        assert not r.obj.is_locked and r.obj.is_open
        assert info["events"]["unlock"] and env.carrying is None, "key was not consumed"
        return


def test_timed_door_closes_after_delay():
    for seed in range(60):
        env = _env(seed=seed, timed_delay=3)
        records = [r for r in env.doors if r.kind == "timed"]
        if not records:
            continue
        r = records[0]
        _face(env, r.pos)
        env.step(ACTIONS["toggle"])
        assert r.obj.is_open
        env.agent_dir = (env.agent_dir + 2) % 4  # turn away, stay out of the doorway
        for _ in range(3):
            env.step(ACTIONS["left"])
        assert not r.obj.is_open
        return
    pytest.skip("no timed door in sixty seeds")


def test_colour_sets_and_combination_holdout():
    train = {(r.color, r.index) for s in range(40) for r in _env("train", s).doors if r.kind == "locked"}
    held = {(r.color, r.index) for s in range(40) for r in _env("transfer_combo", s).doors if r.kind == "locked"}
    assert train and held and not (train & held)
    assert {c for c, _ in train} <= {"red", "green", "blue"}
    colour = {r.color for s in range(20) for r in _env("transfer_colour", s).doors if r.kind == "locked"}
    assert colour <= {"purple", "yellow"}


def test_port_is_rgb_only_and_persistent_policy_is_generic():
    port = RoomsPort(RoomsConfig())
    step = port.reset(seed=3)
    assert step.rgb.dtype == np.uint8 and step.rgb.shape == (42, 42, 3)
    for name in ("grid", "labels", "events"):
        assert not hasattr(step, name)
    actions = persistent_actions(np.random.default_rng(0), 1000)
    assert set(actions.tolist()) <= set(range(5))
    assert (actions[1:] == actions[:-1]).mean() > 0.3


def test_collection_labels_align(tmp_path):
    out = tmp_path / "data"
    meta = collect(out, RoomsConfig(split="train", max_steps=40, play_probability=0.5), 6, 1, workers=2)
    replay, _ = load_rooms_replay(out, training=True)
    labels, events = load_labels(out)
    for e, l, ev in zip(replay.episodes, labels, events, strict=True):
        assert len(l) == len(e.frames) and len(ev) == len(e.actions)
        assert (l[:, column("step")] == np.arange(len(l))).all()
    assert meta["transitions"] == sum(len(e.actions) for e in replay.episodes)
    dev = tmp_path / "dev"
    collect(dev, RoomsConfig(split="development", max_steps=20), 2, 1, workers=1)
    with pytest.raises(ValueError):
        load_rooms_replay(dev, training=True)


def test_probe_cpu(tmp_path):
    out = tmp_path / "probe"
    meta = probe(out, RoomsConfig(split="development", max_steps=30), 2, 5, ordinary_rate=0.5)
    episodes, _ = load_probe(out)
    assert meta["forks"] == sum(len(e["fork_steps"]) for e in episodes)
    for e in episodes:
        assert e["fork_frames"].shape[1:] == (5, 42, 42, 3)
        assert (e["fork_steps"] < len(e["actions"])).all()
        # The fork for the recorded action equals the recorded next frame.
        for f, t in enumerate(e["fork_steps"]):
            a = e["actions"][t]
            assert (e["fork_frames"][f, a] == e["frames"][t + 1]).all()


def test_play_starts_can_hold_the_matching_key():
    holding = 0
    for seed in range(60):
        env = _env(seed=seed, play_probability=1.0)
        locked = [r for r in env.doors if r.kind == "locked" and r.obj.is_locked]
        if env.carrying is not None and any(r.color == env.carrying.color for r in locked):
            holding += 1
            # The POV frame shows the carried key in the agent's cell.
            assert env.gen_obs_grid()[0].get(env.agent_view_size // 2, env.agent_view_size - 1) is env.carrying
    assert holding >= 10
