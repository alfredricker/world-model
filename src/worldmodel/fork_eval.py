"""Score a model on the goal-conditioned fork (CHARTER rung 1, card 001).

The model is seen only through a scorer: given fork observations and goal
examples it returns predicted steps-to-goal after each action (B, 5) and
from the current state (B,). The evaluator's truth comes from
``runs/sampled_dev`` (``rooms_goals.sampled_probe``) and its extras
(``rooms_fork_extras``); the learner never sees either during training.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .envs.rooms_fork_extras import load_extras
from .envs.rooms_goals import INTERACTIONS

SCORED = ("door_open|toggle|locked", "door_open|pickup|key", "door_open|toggle|box",
          "door_open|toggle|plain", "key_held|pickup|key", "key_held|toggle|box")
MOVEMENT = ("door_open|move", "key_held|move")

Scorer = Callable[[object, object, int], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ForkSet:
    """Everything the harness needs, in one place, for one observation mode."""
    obs: object                  # frames (N, 42, 42, 3) or (grid, carried)
    goal: np.ndarray             # goal name index per fork
    world: np.ndarray
    strata: np.ndarray
    dist: np.ndarray             # (N, 5) true steps after each action
    best: np.ndarray             # (N, 5) bool
    no_effect: np.ndarray        # (N, 5) bool
    steps: np.ndarray            # (N,) true steps from the fork state
    exact: object                # exact goal observation per fork
    successors: object           # observation after each action, (N, 5, ...)
    pools: dict                  # goal index -> (world array, observations)
    pairs: dict                  # condition pairs
    goal_names: list


def _obs(arrays: dict, prefix: str, mode: str):
    """Observations stored under ``prefix`` in the extras, in one mode:
    frames, the allocentric simulator grid, or the egocentric view."""
    if mode == "frames":
        key = f"{prefix}frames" if f"{prefix}frames" in arrays else f"{prefix}frame"
        return arrays[key]
    ego = "ego_" if mode == "egocentric" else ""
    return arrays[f"{prefix}{ego}grid"], arrays[f"{prefix}{ego}carried"]


def take(obs, idx):
    if isinstance(obs, tuple):
        return tuple(o[idx] for o in obs)
    return obs[idx]


def to_device(obs, device):
    if isinstance(obs, tuple):
        return tuple(torch.from_numpy(np.ascontiguousarray(o)).to(device) for o in obs)
    return torch.from_numpy(np.ascontiguousarray(obs)).to(device)


def load_forks(probe: Path, extras: Path, mode: str = "frames") -> ForkSet:
    meta = json.loads((probe / "sampled_probe.json").read_text())
    with np.load(probe / "sampled_probe.npz", allow_pickle=False) as archive:
        forks, fork_frames, strata = archive["forks"], archive["fork_frames"], archive["strata"]
    ex, info = load_extras(extras)
    if info["probe_sha256"] != meta["sha256"]:
        raise ValueError("Extras were built from another probe")
    obs = fork_frames if mode == "frames" else _obs(ex, "fork_", mode)
    exact = _obs(ex, "fork_goal_", mode)
    successors = _obs(ex, "fork_next_", mode)
    names = meta["goal_names"]
    pools = {}
    for i, n in enumerate(names):
        if len(ex[f"pool_{n}_world"]):
            pools[i] = (ex[f"pool_{n}_world"], _obs(ex, f"pool_{n}_", mode))
    rows = ex["pair_rows"]
    pairs = {"world": rows[:, 0], "kind": rows[:, 1], "goal": rows[:, 2], "steps_a": rows[:, 3],
             "steps_b": rows[:, 4], "a": _obs(ex, "pair_a_", mode), "b": _obs(ex, "pair_b_", mode),
             "kinds": info["pair_kinds"]}
    return ForkSet(obs, forks[:, 2], forks[:, 0], strata, forks[:, 5:10], forks[:, 10:15].astype(bool),
                   forks[:, 15:20].astype(bool), ex["fork_steps"], exact, successors, pools, pairs, names)


def draw_examples(fs: ForkSet, goals: np.ndarray, worlds: np.ndarray, n: int, rng: np.random.Generator,
                  same_world: bool = False):
    """n pool indices per row, from worlds other than the row's world (the
    test), or from the row's own world (a diagnostic; with replacement when
    the world has fewer than n examples)."""
    out = np.zeros((len(goals), n), np.int64)
    for i, (g, w) in enumerate(zip(goals, worlds)):
        pool_world = fs.pools[int(g)][0]
        allowed = np.flatnonzero((pool_world == w) if same_world else (pool_world != w))
        out[i] = rng.choice(allowed, size=n, replace=len(allowed) < n)
    return out


def swapped_goals(fs: ForkSet, rng: np.random.Generator) -> np.ndarray:
    """Another goal with a pool for each fork (the goal-swapped control)."""
    have = np.array(sorted(g for g, (w, _) in fs.pools.items() if len(w) >= 8))
    out = np.empty(len(fs.goal), np.int64)
    for i, g in enumerate(fs.goal):
        choices = have[have != g]
        out[i] = choices[rng.integers(len(choices))]
    return out


def _gather_goals(fs: ForkSet, goals, picks):
    """Observations of the chosen pool examples, flattened to (B * n, ...)."""
    parts = []
    for g, row in zip(goals, picks):
        parts.append(take(fs.pools[int(g)][1], row))
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(len(parts[0])))
    return np.concatenate(parts)


def run_scorer(scorer: Scorer, obs, goal_obs, n: int, device, batch: int = 256):
    rows = len(obs[0]) if isinstance(obs, tuple) else len(obs)
    after, now = [], []
    for s in range(0, rows, batch):
        idx = np.arange(s, min(rows, s + batch))
        gidx = (idx[:, None] * n + np.arange(n)).reshape(-1)
        a, c = scorer(to_device(take(obs, idx), device), to_device(take(goal_obs, gidx), device), n)
        after.append(a.float().cpu().numpy())
        now.append(c.float().cpu().numpy())
    return np.concatenate(after), np.concatenate(now)


def top1_rows(pred: np.ndarray, best: np.ndarray) -> np.ndarray:
    """Per row: share of the predicted-lowest actions that are truly best."""
    low = pred <= pred.min(1, keepdims=True) + 1e-6
    return (low & best).sum(1) / low.sum(1)


def rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def summarise(fs: ForkSet, after: np.ndarray, rows: np.ndarray | None = None) -> dict:
    rows = np.arange(len(fs.goal)) if rows is None else rows
    t = top1_rows(after[rows], fs.best[rows])
    strata = fs.strata[rows]
    per = {str(s): float(t[strata == s].mean()) for s in np.unique(strata)}
    scored = [per[s] for s in SCORED if s in per]
    move = t[np.isin(strata, MOVEMENT)]
    low = after[rows] <= after[rows].min(1, keepdims=True) + 1e-6
    fake = low & fs.no_effect[rows] & np.isin(np.arange(5), INTERACTIONS)[None]
    return {"scored_top1": float(np.mean(scored)) if scored else float("nan"),
            "movement_top1": float(move.mean()) if len(move) else float("nan"),
            "no_effect_best": float((fake.sum(1) / low.sum(1)).mean()),
            "per_stratum": per}


def baselines(fs: ForkSet) -> dict:
    """Random ranking (chance from ties) and each fixed action preference."""
    chance = fs.best.sum(1) / 5
    out = {"random": float(np.mean([chance[fs.strata == s].mean() for s in SCORED]))}
    for a in range(5):
        pred = np.ones((len(fs.goal), 5))
        pred[:, a] = 0
        out[f"always_{a}"] = summarise(fs, pred)["scored_top1"]
    return out


def evaluate(scorer: Scorer, fs: ForkSet, device, *, n: int = 4, seed: int = 0, rows: np.ndarray | None = None,
             exact: bool = True, successor: Scorer | None = None, same_world: bool = False) -> dict:
    """Pooled n-example goals from other worlds (the test), the goal-swapped
    control, exact single-state goals (the gate's comparison), rank
    correlation of predicted and true steps, and the condition gap."""
    rng = np.random.default_rng(seed)
    rows = np.arange(len(fs.goal)) if rows is None else np.asarray(rows)
    obs = take(fs.obs, rows)
    picks = draw_examples(fs, fs.goal[rows], fs.world[rows], n, rng)
    after = np.full((len(fs.goal), 5), np.nan)
    now = np.full(len(fs.goal), np.nan)
    after[rows], now[rows] = run_scorer(scorer, obs, _gather_goals(fs, fs.goal[rows], picks), n, device)
    result = {"pooled": summarise(fs, after, rows)}
    scored_rows = rows[np.isin(fs.strata[rows], SCORED + MOVEMENT)]
    result["pooled"]["rank_correlation"] = spearman(now[scored_rows], fs.steps[scored_rows].astype(float))
    swap = swapped_goals(fs, rng)[rows]
    swap_picks = draw_examples(fs, swap, fs.world[rows], n, rng)
    swapped = np.full((len(fs.goal), 5), np.nan)
    swapped[rows], _ = run_scorer(scorer, obs, _gather_goals(fs, swap, swap_picks), n, device)
    result["swapped"] = summarise(fs, swapped, rows)
    if exact:
        ex_after = np.full((len(fs.goal), 5), np.nan)
        ex_now = np.full(len(fs.goal), np.nan)
        ex_after[rows], ex_now[rows] = run_scorer(scorer, obs, take(fs.exact, rows), 1, device)
        result["exact"] = summarise(fs, ex_after, rows)
        result["exact"]["rank_correlation"] = spearman(ex_now[scored_rows], fs.steps[scored_rows].astype(float))
    if same_world:
        keep = np.array([np.any(fs.pools[int(g)][0] == w) for g, w in zip(fs.goal[rows], fs.world[rows])])
        r2 = rows[keep]
        sw_picks = draw_examples(fs, fs.goal[r2], fs.world[r2], n, rng, same_world=True)
        sw_after = np.full((len(fs.goal), 5), np.nan)
        sw_now = np.full(len(fs.goal), np.nan)
        sw_after[r2], sw_now[r2] = run_scorer(scorer, take(fs.obs, r2), _gather_goals(fs, fs.goal[r2], sw_picks), n, device)
        result["same_world"] = summarise(fs, sw_after, r2)
        sr = r2[np.isin(fs.strata[r2], SCORED + MOVEMENT)]
        result["same_world"]["rank_correlation"] = spearman(sw_now[sr], fs.steps[sr].astype(float))
        result["same_world"]["rows"] = int(len(r2))
    if successor is not None:
        # The reachability head on the true observation after each action,
        # with exact goals: separates the head from the transition model.
        nxt = take(fs.successors, rows)
        flat = tuple(o.reshape(-1, *o.shape[2:]) for o in nxt) if isinstance(nxt, tuple) else nxt.reshape(-1, *nxt.shape[2:])
        goal5 = take(fs.exact, np.repeat(rows, 5))
        _, d = run_scorer(successor, flat, goal5, 1, device, batch=1280)
        su = np.full((len(fs.goal), 5), np.nan)
        su[rows] = d.reshape(-1, 5)
        result["successor_exact"] = summarise(fs, su, rows)
        # The same with the pooled goal examples the test uses.
        goal5 = _gather_goals(fs, np.repeat(fs.goal[rows], 5), np.repeat(picks, 5, axis=0))
        _, d = run_scorer(successor, flat, goal5, n, device, batch=1280)
        sp = np.full((len(fs.goal), 5), np.nan)
        sp[rows] = d.reshape(-1, 5)
        result["successor_pooled"] = summarise(fs, sp, rows)
    result["condition_gap"] = condition_gap(scorer, fs, device, n=n, seed=seed)
    return result


def condition_gap(scorer: Scorer, fs: ForkSet, device, *, n: int = 4, seed: int = 0) -> dict:
    """Predicted against true gap in steps-to-goal between matched states
    that differ in one condition (a has it, b does not)."""
    p = fs.pairs
    rng = np.random.default_rng(seed + 1)
    picks = draw_examples(fs, p["goal"], p["world"], n, rng)
    goals = _gather_goals(fs, p["goal"], picks)
    _, da = run_scorer(scorer, p["a"], goals, n, device)
    _, db = run_scorer(scorer, p["b"], goals, n, device)
    pred_gap = db - da
    true_gap = (p["steps_b"] - p["steps_a"]).astype(float)
    out = {"rank_correlation": spearman(pred_gap, true_gap)}
    for k, name in enumerate(p["kinds"]):
        m = p["kind"] == k
        if m.any():
            out[name] = {"n": int(m.sum()), "pred_gap": float(pred_gap[m].mean()),
                         "true_gap": float(true_gap[m].mean()),
                         "pred_gap_positive": float((pred_gap[m] > 0).mean())}
    return out
