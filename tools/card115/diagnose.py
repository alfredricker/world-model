"""Card 115, after the run (diagnostics, not criteria): what a revised test should measure.

D1: surprise with the arm's patches left out (as the change upper bound does).
D2: matching in time and place: a boundary counts for an event only if its most surprising (or most changed) patch
    lies on the changing object, or one patch from it. The random baseline draws random frames and random patches.
Thresholds as in score.py (best F1 on episodes 900-999, under the same matching rule).

  bin/prun python tools/card115/diagnose.py → runs/115/diagnose.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from masks import OBJECTS  # noqa: E402
from score import EP, OUT, ROOT, SKIP, TOL, frame_score, gripper_events, match, peaks, true_events, zscore  # noqa: E402,F401


def dilated(masks_tau, masks_prev, obj):
    m = (masks_tau[obj].any(0) | masks_prev[obj].any(0)).reshape(14, 14)
    d = m.copy()
    d[1:] |= m[:-1]; d[:-1] |= m[1:]
    e = d.copy()
    e[:, 1:] |= d[:, :-1]; e[:, :-1] |= d[:, 1:]
    return e.reshape(-1)


def match_place(S, bd, top):
    """Time-and-place matching: candidate pairs need the boundary's top patch on the event's object."""
    ev = S['T']
    pairs = []
    lo = np.searchsorted(bd, ev - TOL)
    hi = np.searchsorted(bd, ev + TOL, side='right')
    for i, (a, b) in enumerate(zip(lo, hi)):
        for j in range(a, b):
            tau = bd[j]
            if tau // EP == ev[i] // EP and dilated(S['masks'][tau], S['masks'][tau - SKIP], S['O'][i])[top[j]]:
                pairs.append((abs(int(tau) - int(ev[i])), i, j))
    pairs.sort()
    me, mb = np.zeros(len(ev), bool), np.zeros(len(bd), bool)
    for _, i, j in pairs:
        if not me[i] and not mb[j]:
            me[i] = mb[j] = True
    p = mb.mean() if len(bd) else 0.0
    r = me.mean()
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, me


def main():
    sets = {}
    for name, sfile, cut in [('tail', 'state_train.npz', 900 * EP), ('val', 'state_val.npz', 0)]:
        s = dict(np.load(ROOT / 'runs/114' / sfile))
        s = {k: v[cut:cut + 100 * EP] for k, v in s.items() if len(v) == len(s['buttons'])}
        sp = np.load(OUT / f'surprise_{name}.npz')
        T, Kd, O = true_events(s)
        ok = (T % EP >= 3 * SKIP)
        sets[name] = dict(err=sp['err'].astype(np.float32), chg=sp['chg'].astype(np.float32),
                          masks=np.load(OUT / f'masks_{name}.npy').astype(bool), T=T[ok], K=Kd[ok], O=O[ok])
    tail = sets['tail']
    arm = OBJECTS.index('arm')
    res = {}
    for d, sig, mask_arm in [('surprise', 'err', False), ('surprise_arm_masked', 'err', True),
                             ('change', 'chg', False), ('change_arm_masked', 'chg', True)]:
        out = {}
        for nm, S in sets.items():
            z = zscore(S[sig], tail[sig])
            allowed = None
            if mask_arm:
                am = S['masks'][:, arm]
                allowed = ~(am | np.r_[np.zeros((SKIP, 196), bool), am[:-SKIP]])
                z = np.where(allowed, z, -np.inf)
            sc = frame_score(z, allowed)
            pk = peaks(sc)
            top = np.argmax(np.nan_to_num(z[pk], nan=-np.inf), 1)
            out[nm] = (sc, pk, top)
        # threshold by time-only F1 (D1) and by time-and-place F1 (D2), each on episodes 900-999
        sc, pk, top = out['tail']
        cands = np.quantile(sc[pk], np.linspace(0, 0.99, 60))

        def time_f1(S, bd):
            me, mb, _ = match(S['T'], bd)
            p, r = (mb.mean() if len(bd) else 0), me.mean()
            return 2 * p * r / (p + r) if p + r else 0

        thr_t = max((time_f1(tail, pk[sc[pk] > c]), c) for c in cands)[1]
        thr_p = max((match_place(tail, pk[sc[pk] > c], top[sc[pk] > c])[0], c) for c in cands)[1]
        sc, pk, top = out['val']
        val = sets['val']
        sel_t = sc[pk] > thr_t
        sel_p = sc[pk] > thr_p
        F, P, R, me = match_place(val, pk[sel_p], top[sel_p])
        res[d] = dict(time_only_f1=round(time_f1(val, pk[sel_t]), 4),
                      time_and_place=dict(f1=round(F, 4), precision=round(P, 4), recall=round(R, 4),
                                          boundaries=int(sel_p.sum()),
                                          recall_by_kind={k: round(float(me[val['K'] == k].mean()), 4)
                                                          for k in np.unique(val['K'])}))
        if d == 'surprise':
            rng = np.random.default_rng(0)
            valid = np.where(np.isfinite(sc))[0]
            n = int(sel_p.sum())
            rf = [match_place(val, np.sort(rng.choice(valid, n, replace=False)), rng.integers(0, 196, n))[0]
                  for _ in range(3)]
            res['random_time_and_place_f1'] = round(float(np.mean(rf)), 4)
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'diagnose.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
