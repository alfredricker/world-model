"""Card 132, step 1: the stream of events of every stored set, in a compact form.

One event = one touch. For each touch of a set (runs/122/computed_<set>.npz) the frame before it is already
stored as frozen DINOv2 patch features (runs/125/before_<set>.npy, [touches, 1024, 768]). Here each patch's 768
numbers are standardised and projected to 64 by PCA (fitted on 300 frames of m400), and the touch is described by
proprioception: gripper position at contact and at release, gripper opening at both, its length in steps.
Truth kept for scoring only: cube position, drawer and window positions, locks, at the frame before the touch.

  bin/prun python tools/card132/build.py → runs/132/z_<set>.npy (fp16, about 260 MB each), ev_<set>.npz, pca.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT, DATA, EP, D = ROOT / 'runs/132', Path.home() / '.ogbench/data', 1001, 64
SETS = {'m400': 400, 'm600': 600, 'm800': 800, 'tail': 900, 't500': 500, 't700': 700, 'val': None}


def main():
    X = np.load(ROOT / 'runs/125/before_m400.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    S = torch.from_numpy(np.asarray(X[np.sort(rng.choice(len(X), 300, replace=False))], dtype=np.float32)).cuda().reshape(-1, 768)
    mu, sd = S.mean(0), S.std(0) + 1e-6
    _, s, V = torch.pca_lowrank((S - mu) / sd, q=D, center=False, niter=4)
    var = float((s ** 2).sum() / (((S - mu) / sd) ** 2).sum())
    scale = s / (len(S) - 1) ** 0.5                                         # unit variance per kept component
    torch.save(dict(mu=mu.cpu(), sd=sd.cpu(), V=V.cpu(), scale=scale.cpu()), OUT / 'pca.pt')
    print('PCA keeps', round(var, 3), 'of the variance', flush=True)
    del S
    for name, first in SETS.items():
        z = np.load(DATA / ('scene-play-v0-val.npz' if first is None else 'scene-play-v0.npz'))
        sl = slice(0, None) if first is None else slice(first * EP, (first + 100) * EP)
        obs, qpos = z['observations'][sl], z['qpos'][sl]
        c = np.load(ROOT / f'runs/122/computed_{name}.npz')
        st, en = c['st'], c['en']
        X = np.load(ROOT / f'runs/125/before_{name}.npy', mmap_mode='r')
        assert len(X) == len(st), (name, len(X), len(st))
        Z = np.zeros((len(st), 1024, D), np.float16)
        for i in range(0, len(st), 64):
            f = torch.from_numpy(np.asarray(X[i:i + 64], dtype=np.float32)).cuda()
            Z[i:i + 64] = ((((f - mu) / sd) @ V) / scale).cpu().numpy()
        np.save(OUT / f'z_{name}.npy', Z)
        t0 = np.maximum(st - 1, 0)
        touch = np.c_[obs[st + 2, 12:15], obs[en, 12:15], obs[st + 2, 17], obs[en, 17], (en - st) / 100.0].astype(np.float32)
        np.savez(OUT / f'ev_{name}.npz', touch=touch, episode=st // EP, cube=qpos[t0, 14:17], drawer=qpos[t0, 23], window=qpos[t0, 24],
                 locks=z['button_states'][sl][t0])
        print(name, 'events', len(st), 'episodes', len(set(st // EP)), flush=True)


if __name__ == '__main__':
    main()
