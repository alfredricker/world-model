"""Card 132, data check: how often an event's outcome depends on something the agent cannot see at that moment.

Two kinds of hiding, from the true state stored with each event (runs/132/ev_<set>.npz, the state before each touch):
  natural      the cube inside the drawer while the drawer is closed (under 3 cm). A reveal is a touch that takes the
               drawer from under 3 cm to over 10 cm; its outcome depends on whether the cube is inside.
  narrow view  the agent sees only the variable it touches (before and after the touch). The variable the next touch
               acts on was last in view some touches ago, or never in this episode, and may have changed since.
A touch's target is the variable it changed; a touch that changed nothing goes to the nearest target by where the
hand made contact (centroid of the contacts that did change it; for the cube, the cube itself), within 6 cm.

  bin/prun python tools/card132/check.py → runs/132/check.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
MEM, TEST = ['m400', 'm600', 'm800', 'tail'], ['t500', 't700', 'val']
NAMES = ['cube', 'drawer', 'window', 'lock0', 'lock1']


def load(names):
    out = []
    for n in names:
        d = np.load(ROOT / f'runs/132/ev_{n}.npz')
        s = np.c_[d['cube'], d['drawer'], d['window'], d['locks']].astype(np.float64)       # cube 0:3, drawer 3, window 4, locks 5:7
        hand = d['touch'][:, :3] * 0.1 + np.array([0.425, 0, 0])
        for e in np.unique(d['episode']):
            m = d['episode'] == e
            out.append((s[m], hand[m]))
    return out


def changed(a, b):
    return np.array([np.linalg.norm(a[:3] - b[:3]) > 0.01, abs(a[3] - b[3]) > 0.01, abs(a[4] - b[4]) > 0.01, a[5] != b[5], a[6] != b[6]])


def inside(s):
    return (np.abs(s[:, 0] - 0.33) < 0.12) & (s[:, 2] > 0.04) & (s[:, 1] < -0.1)                 # as tools/card132/gate.py


def targets(eps):
    """Per episode, the target of each touch but the last (-1 unknown), and what each touch changed."""
    ch = [np.array([changed(s[i], s[i + 1]) for i in range(len(s) - 1)]).reshape(-1, 5) for s, _ in eps]
    cen = {}
    for k in range(1, 5):
        pts = np.concatenate([h[:-1][c[:, k] & (c.sum(1) == 1)] for (s, h), c in zip(eps, ch)])
        cen[k] = pts.mean(0)
    tg = []
    for (s, h), c in zip(eps, ch):
        t = np.full(len(c), -1)
        for i in range(len(c)):
            if c[i].any():
                t[i] = 1 if c[i, 1] else 2 if c[i, 2] else 3 if c[i, 3] else 4 if c[i, 4] else 0     # a drawer carrying the cube is a drawer touch
            else:
                dist = {0: np.linalg.norm(h[i] - s[i, :3]), **{k: np.linalg.norm(h[i] - cen[k]) for k in cen}}
                k = min(dist, key=dist.get)
                t[i] = k if dist[k] < 0.06 else -1
        tg.append(t)
    return tg, ch, cen


def stats(eps):
    tg, ch, cen = targets(eps)
    r = dict(episodes=len(eps), touches=int(sum(len(t) for t in tg)), by_target={}, no_target=int(sum((t == -1).sum() for t in tg)))
    hidden = reveal_in = reveal_out = 0
    delays, stale = [], {k: dict(n=0, never=0, same=0, changed=0, gap=[]) for k in range(5)}
    lockflip = dict(n=0, flipped=0, failed=0)
    for (s, h), t, c in zip(eps, tg, ch):
        hid = inside(s) & (np.abs(s[:, 3]) < 0.03)
        hidden += int(hid.sum())
        since = 0
        for i in range(len(t)):
            since = since + 1 if hid[i] else 0
            if abs(s[i, 3]) < 0.03 and abs(s[i + 1, 3]) > 0.10:
                if hid[i]:
                    reveal_in += 1; delays.append(since)
                else:
                    reveal_out += 1
        last = {}                                                           # target → (index it was last in view, its state then)
        for i in range(len(t)):
            k = t[i]
            if k < 0:
                continue
            val = lambda j, k=k: s[j, :3] if k == 0 else s[j, 2 + k]
            d = stale[k]; d['n'] += 1
            if k not in last:
                d['never'] += 1
            else:
                j, v = last[k]
                same = np.linalg.norm(np.atleast_1d(val(i) - v)) <= 0.01
                d['same' if same else 'changed'] += 1; d['gap'].append(i - j)
            if k in (1, 2):                                                 # a pull: was a lock flipped since this slide was last touched?
                lockflip['n'] += 1
                j0 = last[k][0] if k in last else 0
                lockflip['flipped'] += int((s[i, 5:7] != s[j0, 5:7]).any())
                lockflip['failed'] += int(not c[i].any())
            last[k] = (i + 1, val(i + 1))
    for k in range(5):
        d = stale[k]
        r['by_target'][NAMES[k]] = dict(touches=d['n'], never_seen=d['never'], seen_unchanged=d['same'], seen_but_changed=d['changed'],
                                        median_gap=float(np.median(d['gap'])) if d['gap'] else None)
    r['natural'] = dict(events_cube_hidden=hidden, reveals_with_cube=reveal_in, reveals_empty=reveal_out,
                        touches_hidden_before_reveal=dict(median=float(np.median(delays)) if delays else None, max=int(max(delays)) if delays else None))
    r['pulls'] = lockflip
    r['centroids'] = {NAMES[k]: np.round(v, 3).tolist() for k, v in cen.items()}
    return r


if __name__ == '__main__':
    out = dict(memory=stats(load(MEM)), test=stats(load(TEST)))
    json.dump(out, open(ROOT / 'runs/132/check.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
