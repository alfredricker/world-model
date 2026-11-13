"""Card 132: the memory test bed on true visible state, and its bounds.

Before each touch the agent predicts the touch's outcome by recall: the 5 nearest stored events by a key, majority
outcome. Every arm uses the same recall; only the key differs. Scored on the test episodes' memory-dependent events:
  reopen   a drawer under 3 cm is taken past 10 cm: is the cube inside?
  pull     a touch on the drawer or the window: does it move? Events in which a lock also flipped are left out of
           the store and the score: they are two touches recorded as one (card 120: 18% of events are not single).
Views. A (camera): every variable, the cube absent while it is inside the closed drawer. B (narrow): only the
variable the hand touches, seen after the touch (a slide shows its own lock too).
Keys (contact point and the touched object's identity always: in 14% of pulls the recorded contact point is not at
the handle; lengths in units of 2 cm, a flag counts as 3 units = 6 cm):
  floor_A    + the present camera view            floor_B   the contact point alone
  last_seen  + each variable's last value in view this episode, with a flag for "not seen yet" (view B)
  ceiling    + the true locks and whether the cube is truly in the drawer
Stores of 10, 30, 100 episodes are drawn 10 times from the 400 and averaged.

  bin/prun python tools/card132/bed.py → runs/132/bed.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check as C

ROOT, U, F, K = C.ROOT, 0.02, 3.0, 5


def lock_of(eps, tg, ch):
    """Which lock goes with the drawer (1) and the window (2): the one whose value best splits failed pulls."""
    out, table = {}, {}
    for k in (1, 2):
        rows = np.array([(s[i, 5], s[i, 6], not c[i, k]) for (s, _), t, c in zip(eps, tg, ch) for i in range(len(t)) if t[i] == k])
        diff = [abs(rows[rows[:, j] == 1, 2].mean() - rows[rows[:, j] == 0, 2].mean()) for j in (0, 1)]
        out[k] = int(np.argmax(diff))
        table[C.NAMES[k]] = {f'lock{j}={v}': round(float(rows[rows[:, j] == v, 2].mean()), 3) for j in (0, 1) for v in (0, 1)}
    return out, table


def events(eps, tg, ch, lk):
    """Every touch with a known outcome → keys per arm, outcomes, and which scored set it belongs to."""
    rows = dict(floor_A=[], floor_B=[], last_seen=[], ceiling=[])
    moved, cube_in, pull, reopen, ep, lock_belief, cube_belief, mixed = [], [], [], [], [], [], [], []
    for e, ((s, h), t, c) in enumerate(zip(eps, tg, ch)):
        ins = C.inside(s)
        b = dict(cube=None, drawer=None, window=None, l0=None, l1=None)
        for i in range(len(t)):
            con = np.r_[h[i] / U, 10.0 * (np.arange(-1, 5) == t[i])]            # where the hand touches, and which object (perception's job)
            hid = ins[i] and abs(s[i, 3]) < 0.03
            rows['floor_B'].append(con)
            rows['floor_A'].append(np.r_[con, F * (not hid), (0 * s[i, :3] if hid else s[i, :3]) / U, s[i, 3:5] / U, F * s[i, 5:7]])
            rows['ceiling'].append(np.r_[con, F * s[i, 5:7], F * ins[i]])
            f = []
            for name, n in (('cube', 3), ('drawer', 1), ('window', 1)):
                f += [F * (b[name] is not None), *(np.zeros(n) if b[name] is None else np.atleast_1d(b[name]) / U)]
            for name in ('l0', 'l1'):
                f += [F * (b[name] is not None), 0.0 if b[name] is None else F * b[name]]
            rows['last_seen'].append(np.r_[con, f])
            k = t[i]
            mixed.append(k in (1, 2) and bool(c[i, 3] or c[i, 4]))    # a lock flipped inside the same event: two touches recorded as one
            pull.append(k in (1, 2) and not mixed[-1]); moved.append(bool(c[i, 1] or c[i, 2]))
            reopen.append(abs(s[i, 3]) < 0.03 and abs(s[i + 1, 3]) > 0.10); cube_in.append(bool(ins[i + 1])); ep.append(e)
            lb = b[f'l{lk[k]}'] if k in (1, 2) else None
            lock_belief.append(-1 if lb is None else int(lb == s[i, 5 + lk[k]]))
            cube_belief.append(int(b['cube'] is not None))
            if k == 0:
                b['cube'] = s[i + 1, :3]
            elif k in (1, 2):
                b['drawer' if k == 1 else 'window'] = s[i + 1, 2 + k]; b[f'l{lk[k]}'] = s[i + 1, 5 + lk[k]]
            elif k in (3, 4):
                b[f'l{k - 3}'] = s[i + 1, 2 + k]
    A = lambda x: np.array(x)
    return ({a: torch.tensor(np.array(v), dtype=torch.float32) for a, v in rows.items()},
            dict(moved=A(moved), cube_in=A(cube_in), pull=A(pull), reopen=A(reopen), ep=A(ep), lock_belief=A(lock_belief), cube_belief=A(cube_belief), mixed=A(mixed)))


def recall(ks, ys, kq):
    idx = torch.cdist(kq, ks).topk(min(K, len(ks)), largest=False).indices.numpy()
    return ys[idx].mean(1) > 0.5


def bal(pred, y):
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2) if y.any() and (~y).any() else None


def main():
    mem, test = C.load(C.MEM), C.load(C.TEST)
    tgm, chm, _ = C.targets(mem)
    tgt, cht, _ = C.targets(test)
    lk, table = lock_of(mem, tgm, chm)
    Km, m = events(mem, tgm, chm, lk)
    Kt, t = events(test, tgt, cht, lk)
    sets = dict(reopen=(t['reopen'], 'cube_in'), pull=(t['pull'], 'moved'))
    rng, out = np.random.default_rng(0), dict(lock_of={C.NAMES[k]: f'lock{j}' for k, j in lk.items()}, fail_rate_by_lock=table,
                                              n={k: dict(events=int(q.sum()), positive=int(t[y][q].sum())) for k, (q, y) in sets.items()}, arms={})
    for arm in Km:
        out['arms'][arm] = {}
        for n in (10, 30, 100, 400):
            acc = {k: [] for k in sets}
            for _ in range(1 if n == 400 else 10):
                keep = np.isin(m['ep'], rng.choice(400, n, replace=False)) & ~m['mixed']
                for k, (q, y) in sets.items():
                    acc[k].append(bal(recall(Km[arm][keep], m[y][keep].astype(np.float32), Kt[arm][q]), t[y][q]))
            out['arms'][arm][n] = {k: round(100 * float(np.mean([a for a in v if a is not None])), 1) for k, v in acc.items()}
    # the last-seen arm at the full store, split by what the agent believed
    split = {}
    ok = ~m['mixed']
    p = recall(Km['last_seen'][ok], m['moved'][ok].astype(np.float32), Kt['last_seen'][t['pull']])
    y, lb = t['moved'][t['pull']], t['lock_belief'][t['pull']]
    for name, sel in (('lock not seen yet', lb == -1), ('lock belief right', lb == 1), ('lock belief wrong', lb == 0)):
        split['pull, ' + name] = dict(n=int(sel.sum()), failed=int((~y[sel]).sum()), balanced=bal(p[sel], y[sel]), plain=float((p[sel] == y[sel]).mean()) if sel.any() else None)
    p = recall(Km['last_seen'], m['cube_in'].astype(np.float32), Kt['last_seen'][t['reopen']])
    y, cb = t['cube_in'][t['reopen']], t['cube_belief'][t['reopen']]
    for name, sel in (('cube not seen yet', cb == 0), ('cube seen before', cb == 1)):
        split['reopen, ' + name] = dict(n=int(sel.sum()), with_cube=int(y[sel].sum()), balanced=bal(p[sel], y[sel]), plain=float((p[sel] == y[sel]).mean()) if sel.any() else None)
    out['last_seen_split'] = split
    json.dump(out, open(ROOT / 'runs/132/bed.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
