"""Card 145: the training data. Recorded play (OGBench's state dataset scene-play-v0), every fifth frame rendered at
256 px, encoded by the frozen DINOv2 ViT-S/14 (blocks 2 and 12 per patch, 768 numbers), each patch cut to 64
principal components.

  build.py pca            → runs/145/pca.npz        the components, from 400 random training frames
  build.py meta           → runs/145/meta_<set>.npz the arm's own pose per step (joint angles, gripper position, yaw,
                            opening; observation columns 0-5 and 12-17), and for scoring only: cube pose, drawer and
                            window opening, the two locks, the contact value
  build.py train A B      → runs/145/feat_train.npy fp16 [450, 201, 1024, 64], episodes A to B-1 (paced writes)
  build.py val A B        → runs/145/feat_val.npy   fp16 [100, 201, 1024, 64] (scene-play-v0-val)

  MUJOCO_GL=egl ... bin/prun python tools/card145/build.py train 0 150
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/card124'))
DATA = Path.home() / '.ogbench/data'
OUT = ROOT / 'runs/145'
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
    what = sys.argv[1]
    if what == 'meta':
        for name, n in NEP.items():
            z = np.load(DATA / FILE[name])
            rows = (np.arange(n)[:, None] * EP + STEPS[None]).reshape(-1)
            o, q = z['observations'][rows], z['qpos'][rows]
            contact = (z['observations'][:n * EP, 18].reshape(n, EP)[:, 1:] > 0.05).reshape(n, 200, SKIP).any(-1)   # contact during the step that ends here
            np.savez(OUT / f'meta_{name}.npz', pose=np.c_[o[:, 0:6], o[:, 12:18]].reshape(n, len(STEPS), 12).astype(np.float32),
                     cube=q[:, 14:21].reshape(n, len(STEPS), 7), drawer=np.abs(q[:, 23]).reshape(n, -1), window=np.abs(q[:, 24]).reshape(n, -1),
                     locks=z['button_states'][rows].reshape(n, len(STEPS), 2), contact=contact)
            print(name, 'steps', o.shape[0], flush=True)
        return
    from train import features
    from encode import load_encoder
    render, enc, t0 = renderer(), load_encoder('cuda'), time.time()
    if what == 'pca':
        z = np.load(DATA / FILE['train'])
        rng = np.random.default_rng(0)
        rows = rng.integers(0, NEP['train'], 400) * EP + rng.choice(STEPS, 400)
        fr = render(z['qpos'][rows], z['button_states'][rows])
        S = torch.cat([features(enc, torch.from_numpy(fr[a:a + 50]).cuda()) for a in range(0, 400, 50)]).reshape(-1, 768)
        mean = S.mean(0)
        ev, V = torch.linalg.eigh(torch.cov((S - mean).T))
        ev, V = ev.flip(0), V.flip(1)
        np.savez(OUT / 'pca.npz', mean=mean.cpu().numpy(), V=V[:, :K].cpu().numpy(), ev=ev.cpu().numpy())
        print('variance kept by', K, 'components', round(float(ev[:K].sum() / ev.sum()), 4), f'{time.time() - t0:.0f}s', flush=True)
        os._exit(0)
    a, b = int(sys.argv[2]), int(sys.argv[3])
    p = np.load(OUT / 'pca.npz')
    mean, V = torch.from_numpy(p['mean']).cuda(), torch.from_numpy(p['V']).cuda()
    z = np.load(DATA / FILE[what])
    qpos, btn = z['qpos'], z['button_states']
    path = OUT / f'feat_{what}.npy'
    X = np.load(path, mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path, 'w+', np.float16, (NEP[what], len(STEPS), 1024, K))
    for e in range(a, b):
        rows = e * EP + STEPS
        fr = torch.from_numpy(render(qpos[rows], btn[rows])).cuda()
        with torch.no_grad():
            f = torch.cat([(features(enc, fr[k:k + 67]) - mean) @ V for k in range(0, len(rows), 67)])
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
