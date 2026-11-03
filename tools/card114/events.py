"""Card 114, gate step 1: the held-out events, defined by what the input shows, never by the outcome.

- Pull (drawer or window) starting at frame t: the gripper is within 3 cm of the handle and closed (opening > 0.5),
  the commanded motion along the slide has one sign and magnitude > 0.5 on each of steps t..t+4, and the slide has
  at least 4 cm of room in that direction. The first t of each run of such frames is the event. Outcome (truth,
  for scoring): the slide moved at least 4 cm in the commanded direction by t + 20.
- Press: `button_states` flips between frame f-1 and f. The rollout starts at f - 10 (two model steps before), so
  the flip shows within k <= 4 model steps.
- Quiet windows (for false flips): start frames whose next 20 steps hold no flip.
Every event needs frames t-10 .. t+20 inside one episode (history of 3 model steps, rollout of 4).

  bin/prun python tools/card114/events.py [--data state] → runs/114/events.npz (sevents.npz), and the counts
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EP = 1001          # frames per episode
SKIP, K, H = 5, 4, 3
NEAR, ROOM, MOVE = 0.03, 0.04, 0.04
# slide axis of the effector command, and the sign linking command to joint motion (from the replay:
# drawer handle y = -drawer joint, window handle x = +window joint)
SLIDES = {'drawer': (1, -1.0, (-0.16, 0.0), 0), 'window': (0, 1.0, (0.0, 0.2), 1)}


def episode_ok(t):
    e0 = (t // EP) * EP
    return (t - (H - 1) * SKIP >= e0) & (t + K * SKIP <= e0 + EP - 1)


def find(s):
    out = {}
    eff, grip, act, btn = s['effector'], s['gripper'], s['actions'], s['buttons']
    n = len(grip)
    for name, (ax, sign, (lo, hi), bi) in SLIDES.items():
        q, handle = s[name], s[f'{name}_handle']
        near = (np.linalg.norm(eff - handle, axis=1) < NEAR) & (grip > 0.5)
        cmd = act[:, ax]
        win = np.lib.stride_tricks.sliding_window_view(np.pad(cmd, (0, 4)), 5)[:n]
        pos, neg = (win > 0.5).all(1), (win < -0.5).all(1)
        direction = np.where(pos, 1, np.where(neg, -1, 0)) * sign      # direction of joint motion commanded
        room = np.where(direction > 0, hi - q, q - lo) >= ROOM
        cand = near & (direction != 0) & room
        start = cand & ~np.r_[False, cand[:-1]]
        t = np.where(start)[0]
        t = t[episode_ok(t)]
        moved = (q[t + K * SKIP] - q[t]) * direction[t]
        # moved at any step up to t + 20, in the commanded direction
        best = np.stack([(q[t + j] - q[t]) * direction[t] for j in range(1, K * SKIP + 1)], 1).max(1)
        out[name] = dict(t=t, locked=(btn[t, bi] == 0), dir=direction[t], opened=best >= MOVE, final=moved)
    flip = np.where((btn[1:] != btn[:-1]).any(1))[0] + 1
    which = np.argmax(btn[flip] != btn[flip - 1], 1)
    t = flip - 2 * SKIP
    ok = episode_ok(t) & (t // EP == flip // EP)
    out['press'] = dict(t=t[ok], flip=flip[ok], button=which[ok])
    anyflip = np.zeros(n, bool)
    anyflip[flip] = True
    c = np.r_[0, np.cumsum(anyflip)]
    tq = np.arange(0, n, SKIP)
    tq = tq[episode_ok(tq)]
    out['quiet'] = dict(t=tq[(c[tq + K * SKIP + 1] - c[tq + 1]) == 0])
    return out


def report(tag, ev):
    print(tag)
    for name in SLIDES:
        e = ev[name]
        for lk in (True, False):
            m = e['locked'] == lk
            print(f"  {name:6s} {'locked  ' if lk else 'unlocked'} pulls {m.sum():5d}  moved>=4cm {e['opened'][m].mean():.3f}")
    p = ev['press']
    print(f"  presses button0 {(p['button'] == 0).sum()}  button1 {(p['button'] == 1).sum()}  quiet windows {len(ev['quiet']['t'])}")


def main():
    pre = 'sstate' if 'state' in sys.argv else 'state'
    sv = dict(np.load(ROOT / f'runs/114/{pre}_val.npz'))
    st = dict(np.load(ROOT / f'runs/114/{pre}_train.npz'))
    ev_val = find(sv)
    report('validation (100 episodes)', ev_val)
    # the last 100 training episodes, held out if validation falls short
    cut = 900 * EP
    tail = {k: v[cut:] for k, v in st.items()}
    ev_tail = find(tail)
    report('training episodes 900-999', ev_tail)
    flat = {}
    for tag, ev in [('val', ev_val), ('tail', ev_tail)]:
        for name, d in ev.items():
            for k, v in d.items():
                flat[f'{tag}/{name}/{k}'] = v
    np.savez(ROOT / f"runs/114/{'s' if pre == 'sstate' else ''}events.npz", **flat)


if __name__ == '__main__':
    main()
