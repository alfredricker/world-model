"""Card 145, data check 3: how well does og4's event signal, the contact sensor, mark the steps in which the scene
truly changes? Episodes 0-449, one model step = 5 frames; a step is "in contact" if contact > 0.05 in any of its
frames. Scored against "some true variable changes in this step" (check_sparse.py's rule), exactly and with one step
of slack on either side.

  bin/prun python tools/card145/check_contact.py → runs/145/check_contact.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EP, SKIP, N = 1001, 5, 450
z = np.load(Path.home() / '.ogbench/data/scene-play-v0.npz')
q = z['qpos'][:N * EP].reshape(N, EP, -1)[:, ::SKIP].astype(np.float64)
b = z['button_states'][:N * EP].reshape(N, EP, 2)[:, ::SKIP]
c = (z['observations'][:N * EP, 18].reshape(N, EP)[:, 1:] > 0.05).reshape(N, 200, SKIP).any(-1)
names = ['cube', 'drawer', 'window', 'lock0', 'lock1']
ch = np.stack([np.linalg.norm(np.diff(q[:, :, 14:17], axis=1), axis=-1) > 0.003, np.abs(np.diff(q[:, :, 23], axis=1)) > 0.003,
               np.abs(np.diff(q[:, :, 24], axis=1)) > 0.003, np.diff(b[:, :, 0], axis=1) != 0, np.diff(b[:, :, 1], axis=1) != 0], -1)
any_ch = ch.any(-1)


def wide(x):
    y = x.copy(); y[:, 1:] |= x[:, :-1]; y[:, :-1] |= x[:, 1:]
    return y


def f1(pred, true, slack):
    p = (pred & (wide(true) if slack else true)).sum() / max(pred.sum(), 1)
    r = (true & (wide(pred) if slack else pred)).sum() / max(true.sum(), 1)
    return dict(precision=round(float(p), 3), recall=round(float(r), 3), f1=round(float(2 * p * r / (p + r)), 3))


out = {'share of steps in contact': round(float(c.mean()), 3), 'share of steps with a true change': round(float(any_ch.mean()), 3),
       'contact against any change, exact': f1(c, any_ch, 0), 'contact against any change, one step of slack': f1(c, any_ch, 1),
       'share of each variable\'s change steps that are in contact': {m: round(float(c[ch[..., i]].mean()), 3) for i, m in enumerate(names)},
       'always says change': f1(np.ones_like(c), any_ch, 0)}
json.dump(out, open(ROOT / 'runs/145/check_contact.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
