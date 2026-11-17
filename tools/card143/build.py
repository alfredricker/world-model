"""Card 143: the added training frames, one per release.

For every touch of sets m400, m600, m800 across which a change was seen (card 125's outcome, as
tools/card129/readout.py) and whose place belongs to a group with the aim reading switched on (runs/129.1/aim.json;
no truth): one frame from after the touch's release and before the next touch, with the gripper at least 15 cm from
the release point, drawn at random among such frames; 256 px, blocks 2 and 12 per patch. Target: the gripper's
position at release (proprioception; observation units, 1 = 10 cm).

  MUJOCO_GL=egl ... bin/prun python tools/card143/build.py [--per 4] → runs/143/feats.npy, meta.npy (feats4.npy, meta4.npy: up to 4 frames per release)
  meta columns: release 3, contact 3, true cube at the frame 3 (scores only), group, episode, set, true drawer opening (m)
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'card129'))
from readout import CEN, DATA, EP, FAR, R, ROOT, SETS, otsu  # noqa: E402

OUT = ROOT / 'runs/143'
PER = int(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == '--per'), 1))   # frames drawn per release
TAG = '' if PER == 1 else str(PER)


def releases():
    k = np.load(ROOT / 'runs/127/know.npz')
    z = np.load(DATA / 'scene-play-v0.npz')
    obs400 = z['observations'][400 * EP:500 * EP]
    centres = np.stack([obs400[k['st'][k['place'] == c] + 2, 12:15].mean(0) for c in range(int(k['place'].max()) + 1)])
    G = torch.load(ROOT / 'runs/129/readout_q.pt', weights_only=False)['group']
    on = [int(g) for g, v in json.load(open(ROOT / 'runs/129.1/aim.json')).items() if v['informative']]
    print('groups with the aim reading on', on, flush=True)
    rng = np.random.default_rng(0)
    out = {}
    for name, first in SETS.items():
        sl = slice(first * EP, (first + 100) * EP)
        obs, qpos = z['observations'][sl], z['qpos'][sl]
        c = np.load(ROOT / f'runs/122/computed_{name}.npz')
        st, en = c['st'], c['en']
        s = np.log(np.sort(c['tnum'] / (c['tden'] + np.median(c['tden'])), 1)[:, -4:].mean(1))
        ym = s > otsu(s)
        pos, rel = obs[st + 2, 12:15], obs[en, 12:15]
        grp = G[np.linalg.norm(pos[:, None] - centres[None], axis=-1).argmin(1)]
        rows = []
        for i in np.where(ym & np.isin(grp, on))[0]:
            hi = st[i + 1] - 2 if i + 1 < len(st) and st[i + 1] // EP == st[i] // EP else (st[i] // EP + 1) * EP
            t = np.arange(en[i] + 3, hi)
            t = t[np.linalg.norm(obs[t, 12:15] - rel[i], axis=1) >= FAR] if len(t) else t
            if len(t):
                for r in rng.choice(t, min(PER, len(t)), replace=False):
                    rows.append((r, *rel[i], *pos[i], *((qpos[r, 14:17] - CEN) * 10), grp[i], st[i] // EP, first, abs(qpos[r, 23])))
        out[name] = np.array(rows)
        print(name, 'releases', len(rows), 'of', int((ym & np.isin(grp, on)).sum()), flush=True)
    return out


def build():
    import gymnasium as gym
    import ogbench  # noqa: F401
    from train import features
    from encode import load_encoder
    T = releases()
    env = gym.make('visual-scene-v0', terminate_at_goal=False, mode='data_collection', width=R, height=R)
    env.reset(seed=0)
    u = env.unwrapped
    qvel0 = np.zeros(u._model.nv)
    enc = load_encoder('cuda')
    z = np.load(DATA / 'scene-play-v0.npz')
    n = sum(len(v) for v in T.values())
    X = np.lib.format.open_memmap(OUT / f'feats{TAG}.npy', 'w+', np.float16, (n, 1024, 768))
    j = 0
    for name, first in SETS.items():
        qpos, btn = z['qpos'][first * EP:(first + 100) * EP], z['button_states'][first * EP:(first + 100) * EP]
        for a in range(0, len(T[name]), 32):
            rows = T[name][a:a + 32, 0].astype(int)
            fr = np.zeros((len(rows), R, R, 3), np.uint8)
            for i, r in enumerate(rows):
                u.set_state(qpos[r].astype(np.float64), qvel0, btn[r].astype(np.int64))
                fr[i] = u.render()
            f = features(enc, torch.from_numpy(fr).to('cuda'))
            X[j:j + len(f)] = f.half().cpu().numpy()
            j += len(f)
            if (a // 32) % 4 == 3:                                         # pace the write
                X.flush(); time.sleep(2)
    X.flush()
    np.save(OUT / f'meta{TAG}.npy', np.concatenate([v[:, 1:] for v in T.values()]))
    print('frames', n, flush=True)
    os._exit(0)


if __name__ == '__main__':
    build()
