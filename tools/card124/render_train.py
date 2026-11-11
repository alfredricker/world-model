"""Card 124: 256 x 256 frames of OGBench's state dataset scene-play-v0 (which stores contact), every fifth frame.

  MUJOCO_GL=egl ... bin/prun python tools/card124/render_train.py [--episodes 450] [--first 0] [--name train]
  → runs/124/rgb_<name>.npy (uint8 [episodes, 201, 256, 256, 3]); the val set reads scene-play-v0-val.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/124'
EP, SKIP, R = 1001, 5, 256
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    name, neps, first = arg('--name', 'train'), int(arg('--episodes', '450')), int(arg('--first', '0'))
    z = np.load(DATA / ('scene-play-v0-val.npz' if name == 'val' else 'scene-play-v0.npz'))
    qpos, btn = z['qpos'], z['button_states']
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    steps = np.arange(0, EP, SKIP)
    out = np.lib.format.open_memmap(OUT / f'rgb_{name}.npy', 'w+', np.uint8, (neps, len(steps), R, R, 3))
    t0 = time.time()
    for e in range(neps):
        for k, t in enumerate(steps):
            r = (first + e) * EP + t
            u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
            out[e, k] = u.render()
        if e % 50 == 0:
            print(f'{name} episode {e}/{neps} {time.time() - t0:.0f}s', flush=True)
    out.flush()
    print(f'{name}: {neps} episodes in {time.time() - t0:.0f}s', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
