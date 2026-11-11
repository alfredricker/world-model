"""Card 125: the memory of touches. For every touch of a set (card 122's list), the scene just before it:
the mean of blocks 2 and 12 of the frozen encoder over 3 frames of the before window (ending 2 frames before
contact), at 256 px, 32 x 32 patches. → runs/125/before_<set>.npy, fp16 [touches, 1024, 768].

  MUJOCO_GL=egl ... bin/prun python tools/card125/memory.py --set tail|val
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'card124'))
from train import DATA, EP, ROOT, features  # noqa: E402
from encode import load_encoder  # noqa: E402

R, CLEAR, BACK = 256, 2, [0, 4, 9]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


def main():
    name = arg('--set', 'val')
    z = np.load(DATA / ('scene-play-v0-val.npz' if name == 'val' else 'scene-play-v0.npz'))
    cut = 0 if name == 'val' else int(arg('--first', '900')) * EP
    qpos, btn = z['qpos'][cut:cut + 100 * EP], z['button_states'][cut:cut + 100 * EP]
    st = np.load(ROOT / f'runs/122/computed_{name}.npz')['st']
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    enc = load_encoder('cuda')
    out = np.lib.format.open_memmap(ROOT / f'runs/125/before_{name}.npy', 'w+', np.float16, (len(st), 1024, 768))
    for i in range(0, len(st), 32):
        rows = np.concatenate([[s - CLEAR - b for b in BACK] for s in st[i:i + 32]])
        fr = np.zeros((len(rows), R, R, 3), np.uint8)
        for j, r in enumerate(rows):
            u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
            fr[j] = u.render()
        f = features(enc, torch.from_numpy(fr).to('cuda'))
        out[i:i + 32] = f.reshape(-1, len(BACK), 1024, 768).mean(1).half().cpu().numpy()
    out.flush()
    print(name, len(st), 'touches')
    os._exit(0)


if __name__ == '__main__':
    main()
