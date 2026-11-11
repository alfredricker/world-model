"""Card 125.1: how steady each patch is just before each touch.
For every touch of a set: the 10 frames of the before window (ending 2 frames before contact) are rendered at
256 px and encoded (blocks 2 and 12); per patch, the mean over frames of the squared distance to the window's mean.
→ runs/125.1/var_<set>.npy, float32 [touches, 1024].

  MUJOCO_GL=egl ... bin/prun python tools/card125.1/steady.py --set m600 --first 600
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

R, CLEAR, W = 256, 2, 10
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
    out = np.zeros((len(st), 1024), np.float32)
    for i in range(0, len(st), 8):
        rows = np.concatenate([np.arange(s - CLEAR - W + 1, s - CLEAR + 1) for s in st[i:i + 8]])
        fr = np.zeros((len(rows), R, R, 3), np.uint8)
        for j, r in enumerate(rows):
            u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
            fr[j] = u.render()
        f = features(enc, torch.from_numpy(fr).to('cuda')).reshape(-1, W, 1024, 768)
        out[i:i + 8] = ((f - f.mean(1, keepdim=True)) ** 2).sum(-1).mean(1).cpu().numpy()
    np.save(ROOT / f'runs/125.1/var_{name}.npy', out)
    print(name, len(st), 'touches; median variance', float(np.median(out)))
    os._exit(0)


if __name__ == '__main__':
    main()
