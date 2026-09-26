"""Episode-safe RGB/action replay. Evaluator labels live in separate artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass
class Episode:
    frames: np.ndarray  # T+1; the final frame is the actual post-action observation
    actions: np.ndarray  # T
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def validate(self, action_count: int) -> None:
        n = len(self.actions)
        if n < 1 or len(self.frames) != n + 1:
            raise ValueError("Need exactly T actions and T+1 observations")
        if self.frames.dtype != np.uint8 or self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError("Frames must be uint8 THWC RGB")
        if self.actions.dtype.kind not in "iu" or np.any(self.actions < 0) or np.any(self.actions >= action_count):
            raise ValueError("Invalid opaque action IDs")
        if any(x.shape != (n,) for x in (self.rewards, self.terminated, self.truncated)):
            raise ValueError("Transition fields must have length T")
        if np.any(self.terminated[:-1] | self.truncated[:-1]):
            raise ValueError("Episode crosses a reset")
        if not np.isfinite(self.rewards).all():
            raise ValueError("Non-finite reward")


@dataclass
class Batch:
    frames: torch.Tensor  # B,T+1,C,H,W, floats in [0,1]
    actions: torch.Tensor  # B,T


class Replay:
    def __init__(self, episodes: list[Episode], action_count: int):
        if not episodes:
            raise ValueError("Empty replay")
        for episode in episodes:
            episode.validate(action_count)
        self.episodes = episodes
        self.action_count = action_count

    def sample(self, batch_size: int, transitions: int, rng: np.random.Generator,
               device: str | torch.device = "cpu") -> Batch:
        if batch_size < 1 or transitions < 1:
            raise ValueError("Batch size and window length must be positive")
        # Sample windows uniformly, not episodes uniformly; never cross a reset.
        counts = np.array([max(0, len(e.actions) - transitions + 1) for e in self.episodes])
        if not counts.sum():
            raise ValueError(f"No episode has {transitions} transitions")
        cumulative = np.cumsum(counts)
        frames, actions = [], []
        for draw in rng.integers(0, cumulative[-1], size=batch_size):
            index = int(np.searchsorted(cumulative, draw, side="right"))
            start = int(draw - (cumulative[index - 1] if index else 0))
            episode = self.episodes[index]
            frames.append(episode.frames[start:start + transitions + 1])
            actions.append(episode.actions[start:start + transitions])
        return Batch(
            torch.from_numpy(np.stack(frames)).to(device).permute(0, 1, 4, 2, 3).float() / 255,
            torch.from_numpy(np.stack(actions)).to(device).long(),
        )
