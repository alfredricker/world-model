"""Card 172: the training data. Recorded play (OGBench's state dataset), every fifth frame rendered at 256 px, encoded by
the frozen DINOv2 ViT-S/14 (blocks 2 and 12 per patch) and cut to card 168's 64 components (runs/132/pca.pt), so the
frames read here are in the same space as card 141's test frames and og5's input.

  build.py meta          -> runs/172/meta_<set>.npz   the arm's pose per step; for scoring only: cube, drawer, window, locks
  build.py train A B     -> runs/172/feat_train.npy   fp16 [450, 201, 1024, 64], episodes A to B-1 (paced writes)
  build.py val A B       -> runs/172/feat_val.npy     fp16 [100, 201, 1024, 64]
The contact value is not stored: nothing in this card reads it.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = Path.home() / '.ogbench/data', ROOT / 'runs/172'
EP, SKIP, R, K, NEP = 1001, 5, 256, 64, dict(train=450, val=100)
STEPS = np.arange(0, EP, SKIP)
FILE = dict(train='scene-play-v0.npz', val='scene-play-v0-val.npz')


def renderer():
    import gymnasium as gym
    import ogbench  # noqa: F401
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)

    def render(qpos, btn):
        out = np.zeros((len(qpos), R, R, 3), np.uint8)
        for i in range(len(qpos)):
            u.set_state(qpos[i].astype(np.float64), qvel0, btn[i].astype(np.int64))
            out[i] = u.render()
        return out
    return render


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1]
    if what == 'meta':
        for name, n in NEP.items():
            z = np.load(DATA / FILE[name])
            rows = (np.arange(n)[:, None] * EP + STEPS[None]).reshape(-1)
            o, q = z['observations'][rows], z['qpos'][rows]
            np.savez(OUT / f'meta_{name}.npz', pose=np.c_[o[:, 0:6], o[:, 12:18]].reshape(n, len(STEPS), 12).astype(np.float32),
                     cube=q[:, 14:17].reshape(n, len(STEPS), 3), drawer=np.abs(q[:, 23]).reshape(n, -1), window=np.abs(q[:, 24]).reshape(n, -1),
                     locks=z['button_states'][rows].reshape(n, len(STEPS), 2))
            print(name, 'steps', o.shape[0], flush=True)
        return
    from train import features          # card 124
    from encode import load_encoder
    render, enc, t0 = renderer(), load_encoder('cuda'), time.time()
    a, b = int(sys.argv[2]), int(sys.argv[3])
    p = torch.load(ROOT / 'runs/132/pca.pt', map_location='cuda', weights_only=False)
    mu, sd, V, scale = (p[k].float() for k in ('mu', 'sd', 'V', 'scale'))
    z = np.load(DATA / FILE[what])
    qpos, btn = z['qpos'], z['button_states']
    path = OUT / f'feat_{what}.npy'
    X = np.load(path, mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path, 'w+', np.float16, (NEP[what], len(STEPS), 1024, K))
    for e in range(a, b):
        rows = e * EP + STEPS
        fr = torch.from_numpy(render(qpos[rows], btn[rows])).cuda()
        with torch.no_grad():
            f = torch.cat([(((features(enc, fr[k:k + 67]) - mu) / sd) @ V) / scale for k in range(0, len(rows), 67)])
        X[e] = f.half().cpu().numpy()
        if e % 10 == 9:
            X.flush(); time.sleep(1)                                        # pace the write
        if e % 25 == 0:
            print(what, 'episode', e, f'{time.time() - t0:.0f}s', flush=True)
    X.flush()
    print(what, a, b, 'done', f'{time.time() - t0:.0f}s', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
