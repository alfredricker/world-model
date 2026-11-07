"""Card 116.3, gate step 1: re-render frames at 256 x 256 with OGBench's own camera from the stored joint positions and
button states (velocities zero; they do not change a still frame).

--set train: the frames of frames448_train.npy (every fifth frame of training episodes 0-449).
--set val:   the frames of frames448_val.npy (truth frames with history, clips with lead-in).

  MUJOCO_GL=egl ... bin/prun python tools/card116/render256.py --set train|val → runs/116/rgb256_<set>.npy (uint8)
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
OUT = ROOT / 'runs/116'
R = 256
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    which = arg('--set', 'val')
    idx = np.load(OUT / f'frames448_{which}.npy')
    z = np.load(DATA / ('visual-scene-play-v0' + ('-val' if which == 'val' else '') + '.npz'))
    qpos, btn = z['qpos'], z['button_states']
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    out = np.lib.format.open_memmap(OUT / f'rgb256_{which}.npy', 'w+', np.uint8, (len(idx), R, R, 3))
    t0 = time.time()
    for i, r in enumerate(idx):
        u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
        out[i] = u.render()
        if i % 20000 == 0:
            print(f'{which} {i}/{len(idx)} {time.time() - t0:.0f}s', flush=True)
    out.flush()
    print(f'{which}: {len(idx)} frames in {time.time() - t0:.0f}s', flush=True)
    os._exit(0)                                                    # skip EGL teardown errors


if __name__ == '__main__':
    main()
