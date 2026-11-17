"""Card 143, data check before drafting: what the hand's own pose could label, beyond the moment of contact.

Bed frames (card 141): the frame before each touch, with truth for scoring (runs/132/ev_<set>.npz). Touch i has a
contact point and a release point (proprioception). Two labels for the cube and the slides:
  forward   the frame before touch i, labelled with touch i's contact point (og4's training signal)
  backward  the frame before touch i+1, labelled with touch i's release point (new)
Which object a touch moved is taken from the truth here (the agent's own grouping is card 129's); counts and
errors by where the cube is, and for each slide the opening read from the hand's position by a linear fit.

  bin/prun python tools/card143/check.py → runs/143/check.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'tools/card141'))
import common as C


def load(sets):
    t = C.truth(sets)
    d = [np.load(ROOT / f'runs/132/ev_{n}.npz') for n in sets]
    touch = np.concatenate([x['touch'] for x in d]).astype(np.float64)
    t['contact'], t['release'] = touch[:, :3] / 10 + C.CEN, touch[:, 3:6] / 10 + C.CEN
    n = len(t['ep'])
    t['next'] = np.r_[t['ep'][1:] == t['ep'][:-1], False]                  # touch i+1 is in the same episode
    return t, n


def med(x):
    return round(float(np.median(x)) * 100, 2) if len(x) else None


def within(x, r):
    return round(100 * float((x < r).mean()), 1) if len(x) else None


def cube(t, n):
    i = np.where(t['next'])[0]
    moved = np.linalg.norm(t['cube'][i + 1] - t['cube'][i], axis=1) > 0.01
    out = {}
    # forward: the touch is on the cube (contact within 3 cm of the true cube) and moved it
    on = np.linalg.norm(t['contact'][i] - t['cube'][i], axis=1) < 0.03
    for wh in ('table', 'open', 'hidden'):
        k = moved & on & (t['where'][i] == wh)
        e = np.linalg.norm((t['contact'][i] - t['cube'][i])[k][:, :2], axis=1)
        out[f'forward {wh}'] = dict(frames=int(k.sum()), median_cm=med(e), within_1cm=within(e, 0.01))
    # backward: touch i moved the cube; the frame before touch i+1 is labelled with touch i's release point
    for wh in ('table', 'open', 'hidden'):
        k = moved & (t['where'][i + 1] == wh)
        e = np.linalg.norm((t['release'][i] - t['cube'][i + 1])[k][:, :2], axis=1)
        e3 = np.linalg.norm((t['release'][i] - t['cube'][i + 1])[k], axis=1)
        out[f'backward {wh}'] = dict(frames=int(k.sum()), median_cm=med(e), within_1cm=within(e, 0.01), within_3cm=within(e, 0.03),
                                     median_3d_cm=med(e3))
    # how the cube moved without being in the hand at release: carried by the drawer, or pushed
    k = moved & (np.linalg.norm((t['release'][i] - t['cube'][i + 1]), axis=1) > 0.05)
    out['moved but not released at the hand (over 5 cm)'] = dict(
        frames=int(k.sum()), of_them_cube_in_drawer_after=int((t['where'][i + 1][k] != 'table').sum()))
    # the label kept while the cube stays where it was released (truth says when): frames it would cover
    keep = {wh: 0 for wh in ('table', 'open', 'hidden')}
    last = None
    for j in range(n):
        if j == 0 or t['ep'][j] != t['ep'][j - 1]:
            last = None
        elif np.linalg.norm(t['cube'][j] - t['cube'][j - 1]) > 0.01:
            last = j if np.linalg.norm(t['release'][j - 1] - t['cube'][j]) < 0.03 else None
        if last is not None and np.linalg.norm(t['cube'][j] - t['cube'][last]) < 0.01:
            keep[t['where'][j]] += 1
    out['frames covered if the release label is kept until the cube next moves'] = keep
    out['frames by where the cube is'] = {wh: int((t['where'] == wh).sum()) for wh in ('table', 'open', 'hidden')}
    wa, wb = t['where'][i], t['where'][i + 1]
    out['touches that hide the cube (open to hidden)'] = int(((wa == 'open') & (wb == 'hidden')).sum())
    out['touches that reveal it (hidden to open)'] = int(((wa == 'hidden') & (wb == 'open')).sum())
    out['touches that put it in the drawer (table to open)'] = int(((wa == 'table') & (wb == 'open')).sum())
    out['touches that take it out (open to table)'] = int(((wa == 'open') & (wb == 'table')).sum())
    return out


def slide_rows(t, s):
    i = np.where(t['next'])[0]
    k = np.abs(t[s][i + 1] - t[s][i]) > 0.01
    i = i[k]
    return dict(contact=t['contact'][i], before=t[s][i], release=t['release'][i], after=t[s][i + 1])


def slides(tl, tt):
    out = {}
    for s in ('drawer', 'window'):
        L, T = slide_rows(tl, s), slide_rows(tt, s)
        res = dict(touches_learn=int(len(L['after'])), touches_test=int(len(T['after'])))
        for name, p, y in (('backward (release point, opening after)', 'release', 'after'), ('forward (contact point, opening before)', 'contact', 'before')):
            A = lambda x: np.c_[x, np.ones(len(x))]
            B = np.linalg.lstsq(A(L[p]), L[y], rcond=None)[0]
            e = np.abs(A(T[p]) @ B - T[y])
            res[name] = dict(median_cm=med(e), within_1cm=within(e, 0.01), within_3cm=within(e, 0.03),
                             mean_value_cm=med(np.abs(T[y] - L[y].mean())))
        # the hand's travel during the touch against the slide's travel
        dl, dt = (L['release'] - L['contact']), (T['release'] - T['contact'])
        B = np.linalg.lstsq(dl, L['after'] - L['before'], rcond=None)[0]
        e = np.abs(dt @ B - (T['after'] - T['before']))
        res['travel (release minus contact, change of opening)'] = dict(median_cm=med(e), within_1cm=within(e, 0.01), within_3cm=within(e, 0.03))
        res['opening after, spread'] = {q: round(float(np.quantile(T['after'], q / 100)) * 100, 1) for q in (5, 25, 50, 75, 95)}
        out[s] = res
    return out


def main():
    tl, nl = load(C.LEARN)
    tt, nt = load(C.TEST)
    out = dict(learn=dict(frames=nl, cube=cube(tl, nl)), test=dict(frames=nt, cube=cube(tt, nt)), slides=slides(tl, tt))
    json.dump(out, open(ROOT / 'runs/143/check.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
