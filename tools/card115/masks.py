"""Card 115, gate step 1 and scoring masks: which of DINOv2's 14 x 14 patches each object covers, in every held-out frame.

Each frame is rendered by OGBench's simulator from the stored joint positions and button states (camera
'front_pixels', 64 x 64), once in colour and once as a segmentation (geom ids). Geoms are grouped by body into six
objects: the arm (ur5e/*), the cube (object_0), button 0, button 1, the drawer's moving part (drawer_link*), the
window's moving part (windowb_*). A patch covers an object if any of its pixels do (pixel row r is patch row
floor(14 r / 64), as after the 196 x 196 upsampling). The colour render is compared with the dataset's frame.

  MUJOCO_GL=egl ... bin/prun python tools/card115/masks.py --set val|tail [--limit N]
  → runs/115/masks_<set>.npy (uint8 [n, 6, 196]) and the colour check
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
OUT = ROOT / 'runs/115'
EP = 1001
OBJECTS = ['arm', 'cube', 'button0', 'button1', 'drawer', 'window']
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def object_of_body(name):
    if name.startswith('ur5e/'):
        return 0
    if name == 'object_0':
        return 1
    if name == 'button_0':
        return 2
    if name == 'button_1':
        return 3
    if name.startswith('drawer_link'):
        return 4
    if name.startswith('windowb_'):
        return 5
    return -1


def main():
    which = arg('--set', 'val')
    limit = int(arg('--limit', 0))
    if which == 'val':
        z = np.load(DATA / 'visual-scene-play-v0-val.npz')
        qpos, btn = z['qpos'], z['button_states']
        obs = np.load(ROOT / 'runs/114/obs_visual-scene-play-v0-val.npy', mmap_mode='r')
        rows = np.arange(len(qpos))
    else:
        z = np.load(DATA / 'visual-scene-play-v0.npz')
        qpos, btn = z['qpos'], z['button_states']
        obs = np.load(ROOT / 'runs/114/obs_visual-scene-play-v0.npy', mmap_mode='r')
        rows = np.arange(900 * EP, 1000 * EP)
    if limit:
        rows = rows[:limit]
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection')
    env.reset(seed=0)
    u = env.unwrapped
    m = u._model
    geom_obj = np.array([object_of_body(m.body(m.geom_bodyid[g]).name) for g in range(m.ngeom)])
    patch_of = (np.arange(64) * 14) // 64                                   # pixel row/col → patch row/col
    pid = patch_of[:, None] * 14 + patch_of[None, :]                        # [64, 64] patch index
    masks = np.zeros((len(rows), len(OBJECTS), 196), np.uint8)
    diffs = []
    qvel0 = np.zeros(m.nv)
    t0 = time.time()
    for i, r in enumerate(rows):
        u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
        seg = u.render(segmentation=True)
        gid, typ = seg[..., 0], seg[..., 1]
        obj = np.where((typ == 5) & (gid >= 0), geom_obj[np.clip(gid, 0, m.ngeom - 1)], -1)  # 5 = mjOBJ_GEOM
        for k in range(len(OBJECTS)):
            masks[i, k, np.unique(pid[obj == k])] = 1
        if i % 100 == 0:
            rgb = u.render()
            diffs.append(np.abs(rgb.astype(np.int16) - np.asarray(obs[r]).astype(np.int16)).mean())
        if i % 20000 == 0:
            print(f'{which} {i}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f'masks_{which}.npy', masks)
    d = np.array(diffs)
    print(f'{which}: {len(rows)} frames in {time.time() - t0:.0f}s; colour render vs dataset on {len(d)} frames: '
          f'mean |diff| {d.mean():.3f} of 255, largest {d.max():.3f}')
    print('mean patches per object:', dict(zip(OBJECTS, masks.sum(2).mean(0).round(1))))


if __name__ == '__main__':
    main()
