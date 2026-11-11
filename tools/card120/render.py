"""Card 120, gate step 1: frames and object masks for OGBench's state dataset scene-play-v0, which stores contact.

Each frame is rendered from the stored joint positions and button states (camera and size as visual-scene-v0,
64 x 64), in colour and as a segmentation, exactly as card 115 rendered masks for the visual dataset.

  MUJOCO_GL=egl ... bin/prun python tools/card120/render.py --set val|tail
  → runs/120/rgb_<set>.npy (uint8 [n, 64, 64, 3]), runs/120/masks_<set>.npy (uint8 [n, 6, 196])
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card115'))
from masks import OBJECTS, object_of_body  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/120'
EP = 1001
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    which = arg('--set', 'val')
    z = np.load(DATA / ('scene-play-v0-val.npz' if which == 'val' else 'scene-play-v0.npz'))
    rows = np.arange(len(z['qpos'])) if which == 'val' else np.arange(900 * EP, 1000 * EP)
    limit = int(arg('--limit', 0))
    if limit:
        rows = rows[:limit]
    qpos, btn = z['qpos'][rows], z['button_states'][rows]
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection')
    env.reset(seed=0)
    u = env.unwrapped
    m = u._model
    geom_obj = np.array([object_of_body(m.body(m.geom_bodyid[g]).name) for g in range(m.ngeom)])
    patch_of = (np.arange(64) * 14) // 64
    pid = patch_of[:, None] * 14 + patch_of[None, :]
    masks = np.zeros((len(rows), len(OBJECTS), 196), np.uint8)
    rgb = np.zeros((len(rows), 64, 64, 3), np.uint8)
    qvel0 = np.zeros(m.nv)
    t0 = time.time()
    for i in range(len(rows)):
        u.set_state(qpos[i].astype(np.float64), qvel0, btn[i].astype(np.int64))
        seg = u.render(segmentation=True)
        gid, typ = seg[..., 0], seg[..., 1]
        obj = np.where((typ == 5) & (gid >= 0), geom_obj[np.clip(gid, 0, m.ngeom - 1)], -1)
        for k in range(len(OBJECTS)):
            masks[i, k, np.unique(pid[obj == k])] = 1
        rgb[i] = u.render()
        if i % 10000 == 0:
            print(f'{which} {i}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    np.save(OUT / f'masks_{which}.npy', masks)
    np.save(OUT / f'rgb_{which}.npy', rgb)
    print(f'{which}: {len(rows)} frames in {time.time() - t0:.0f}s; mean patches per object:',
          dict(zip(OBJECTS, masks.sum(2).mean(0).round(1))), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
