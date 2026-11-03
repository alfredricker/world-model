"""Card 116, gate step 1: pixel-level truth for scoring slots.

Picks (seed 0) 2,000 validation frames with at least 4 model steps of history in their episode, and 100 validation
clips of 21 model steps (frames s, s + 5, ..., s + 100) starting at least 20 frames into an episode. Each frame is
rendered by OGBench's simulator from the stored joint positions and button states as a segmentation; geoms are grouped
into labels: 0 background, 1 arm (all ur5e links), 2 cube, 3 button 0, 4 button 1, 5 drawer (moving part), 6 window
(moving part).

  MUJOCO_GL=egl ... bin/prun python tools/card116/truth.py → runs/116/truth.npz (frames, labels, clips, clip_labels)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
from masks import object_of_body  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/116'
EP, SKIP, HIST, CLIP = 1001, 5, 4, 21
NAMES = ['background', 'arm', 'cube', 'button0', 'button1', 'drawer', 'window']


def main():
    z = np.load(DATA / 'visual-scene-play-v0-val.npz')
    qpos, btn = z['qpos'], z['button_states']
    n = len(qpos)
    rng = np.random.default_rng(0)
    local = np.arange(n) % EP
    frames = np.sort(rng.choice(np.where(local >= HIST * SKIP)[0], 2000, replace=False))
    starts = np.sort(rng.choice(np.where((local >= HIST * SKIP) & (local + (CLIP - 1) * SKIP <= EP - 1))[0], 100,
                                replace=False))
    clips = starts[:, None] + SKIP * np.arange(CLIP)
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection')
    env.reset(seed=0)
    u = env.unwrapped
    m = u._model
    geom_lab = np.array([object_of_body(m.body(m.geom_bodyid[g]).name) + 1 for g in range(m.ngeom)])  # -1 → 0
    qvel0 = np.zeros(m.nv)

    def label(r):
        u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
        seg = u.render(segmentation=True)
        gid, typ = seg[..., 0], seg[..., 1]
        return np.where((typ == 5) & (gid >= 0), geom_lab[np.clip(gid, 0, m.ngeom - 1)], 0).astype(np.uint8)

    labels = np.stack([label(r) for r in frames])
    clip_labels = np.stack([np.stack([label(r) for r in c]) for c in clips])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / 'truth.npz', frames=frames, labels=labels, clips=clips, clip_labels=clip_labels)
    vis = [(labels == k).any((1, 2)).mean() for k in range(7)]
    px = [(labels == k).sum((1, 2))[(labels == k).any((1, 2))].mean() for k in range(1, 7)]
    print('visible share:', dict(zip(NAMES, np.round(vis, 3))))
    print('pixels when visible:', dict(zip(NAMES[1:], np.round(px, 1))))


if __name__ == '__main__':
    main()
