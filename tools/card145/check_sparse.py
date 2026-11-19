"""Card 145, data check 1: how many of the scene's true variables change in one model step?

From the recorded play's stored state (scoring only), episodes 0-449, one model step = 5 frames. A variable changes
in a step if the cube moves over 0.3 cm, a slide over 0.3 cm, or a lock flips. Reported: the share of steps in which
0, 1, 2 or more variables change (the lowest rate at which any state that follows the scene must change), how often
each variable changes, which pairs change together, how long a stretch of change and a stretch of rest last, and how
often the gripper itself moves (the self).

  bin/prun python tools/card145/check_sparse.py → runs/145/check_sparse.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EP, SKIP, N = 1001, 5, 450
z = np.load(Path.home() / '.ogbench/data/scene-play-v0.npz')
q = z['qpos'][:N * EP].reshape(N, EP, -1)[:, ::SKIP].astype(np.float64)
b = z['button_states'][:N * EP].reshape(N, EP, 2)[:, ::SKIP]
o = z['observations'][:N * EP].reshape(N, EP, -1)[:, ::SKIP]
names = ['cube', 'drawer', 'window', 'lock0', 'lock1']
ch = np.stack([np.linalg.norm(np.diff(q[:, :, 14:17], axis=1), axis=-1) > 0.003, np.abs(np.diff(q[:, :, 23], axis=1)) > 0.003,
               np.abs(np.diff(q[:, :, 24], axis=1)) > 0.003, np.diff(b[:, :, 0], axis=1) != 0, np.diff(b[:, :, 1], axis=1) != 0], -1)   # [N, 200, 5]
n = ch.sum(-1)
out = {'steps': int(n.size), 'share of steps with k variables changing': {str(k): round(float((n == k).mean()), 4) for k in range(4)},
       'mean variables changing per step': round(float(n.mean()), 4),
       'share of steps each variable changes': {m: round(float(ch[..., i].mean()), 4) for i, m in enumerate(names)},
       'changing together, share of steps': {f'{names[i]}+{names[j]}': round(float((ch[..., i] & ch[..., j]).mean()), 4)
                                               for i in range(5) for j in range(i + 1, 5) if (ch[..., i] & ch[..., j]).mean() > 0.0005},
       'gripper moves over 0.3 cm, share of steps': round(float((np.linalg.norm(np.diff(o[:, :, 12:15], axis=1), axis=-1) > 0.03).mean()), 4)}
runs, rests = {m: [] for m in names}, {m: [] for m in names}
for i, m in enumerate(names):
    for e in range(N):
        x = ch[e, :, i].astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, x, 0]))
        runs[m] += list(edges[1::2] - edges[::2])
        y = 1 - x
        edges = np.flatnonzero(np.diff(np.r_[0, y, 0]))
        rests[m] += list(edges[1::2] - edges[::2])
out['stretches of change per episode'] = {m: round(len(v) / N, 1) for m, v in runs.items()}
out['median length of a stretch of change, steps'] = {m: float(np.median(v)) for m, v in runs.items()}
out['median length of a stretch of rest, steps'] = {m: float(np.median(v)) for m, v in rests.items()}
json.dump(out, open(ROOT / 'runs/145/check_sparse.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
