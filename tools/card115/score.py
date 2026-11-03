"""Card 115: score boundaries from surprise against the simulator's condition changes.

Detectors (same rules, different per-patch signal):
- surprise: the predictor's error;
- change: plain frame-to-frame change of the patch vectors (no prediction);
- change, arm masked (upper bound): plain change with patches on the arm (true mask at t or t + 5) left out.
Each patch's signal is turned into (x - median) / MAD, both over episodes 900-999. Frame score: mean of the 4
largest. Boundaries: frames that are the highest within +-5 steps (same episode) and above a threshold. The
threshold maximises F1 on episodes 900-999; validation episodes are the test. Matching: one to one, within +-5 steps,
closest pairs first. Random baseline: as many boundaries as the surprise detector, at random valid frames.

  bin/prun python tools/card115/score.py → runs/115/score.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from masks import OBJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/115'
EP, SKIP, TOL, TOPK = 1001, 5, 5, 4


def runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def true_events(s):
    """→ times [m], kinds [m], objects [m, 6] (which objects show the change)."""
    n = len(s['buttons'])
    ep = np.arange(n) // EP
    same = ep[1:] == ep[:-1]
    T, Kd, O = [], [], []
    flip = np.where((s['buttons'][1:] != s['buttons'][:-1]).any(1) & same)[0] + 1
    for f in flip:
        b = int(np.argmax(s['buttons'][f] != s['buttons'][f - 1]))
        o = np.zeros(6, bool)
        o[OBJECTS.index(f'button{b}')] = True
        o[OBJECTS.index('drawer' if b == 0 else 'window')] = True       # the handle changes colour too
        T.append(f); Kd.append('lock flip'); O.append(o)
    for name, v in [('drawer', s['drawer'][:, None]), ('window', s['window'][:, None]), ('cube', s['cube'])]:
        moving = (np.linalg.norm(np.diff(v, axis=0), axis=1) > 0.002) & same
        a, b = runs(moving)
        st, en = [], []
        for x, y in zip(a, b):
            if st and x - en[-1] < 3 and x // EP == st[-1] // EP:
                en[-1] = y
            else:
                st.append(x); en.append(y)
        for x, y in zip(st, en):
            if y - x < 3:
                continue
            o = np.zeros(6, bool)
            o[OBJECTS.index(name)] = True
            T += [x + 1, y]; Kd += [f'{name} start', f'{name} stop']; O += [o, o]
    order = np.argsort(T)
    return np.array(T)[order], np.array(Kd)[order], np.array(O)[order]


def gripper_events(s):
    n = len(s['gripper'])
    ep = np.arange(n) // EP
    g = s['gripper'] > 0.5
    return np.where((g[1:] != g[:-1]) & (ep[1:] == ep[:-1]))[0] + 1


def zscore(x, ref):
    med = np.nanmedian(ref, 0)
    mad = np.nanmedian(np.abs(ref - med), 0) + 1e-6
    return (x - med) / mad


def frame_score(z, allowed=None):
    if allowed is not None:
        z = np.where(allowed, z, -np.inf)
    top = -np.sort(-np.nan_to_num(z, nan=-np.inf), 1)[:, :TOPK]
    sc = np.where(np.isfinite(top).all(1), top.mean(1), -np.inf)
    return sc


def peaks(sc):
    n = len(sc)
    ep = np.arange(n) // EP
    pad = np.pad(sc, TOL, constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * TOL + 1)
    epad = np.pad(ep, TOL, constant_values=-1)
    ewin = np.lib.stride_tricks.sliding_window_view(epad, 2 * TOL + 1)
    w = np.where(ewin == ep[:, None], win, -np.inf)
    is_peak = np.isfinite(sc) & (sc >= w.max(1))
    # one peak per plateau
    is_peak &= ~np.r_[False, is_peak[:-1] & (sc[1:] == sc[:-1])]
    return np.where(is_peak)[0]


def match(ev, bd):
    """One-to-one, closest first, within TOL and the same episode → matched event mask, matched boundary mask, pairs."""
    if len(bd) == 0:
        return np.zeros(len(ev), bool), np.zeros(0, bool), []
    pairs = []
    lo = np.searchsorted(bd, ev - TOL)
    hi = np.searchsorted(bd, ev + TOL, side='right')
    for i, (a, b) in enumerate(zip(lo, hi)):
        for j in range(a, b):
            if bd[j] // EP == ev[i] // EP:
                pairs.append((abs(int(bd[j]) - int(ev[i])), i, j))
    pairs.sort()
    me, mb, out = np.zeros(len(ev), bool), np.zeros(len(bd), bool), []
    for _, i, j in pairs:
        if not me[i] and not mb[j]:
            me[i] = mb[j] = True
            out.append((i, j))
    return me, mb, out


def f1(ev, bd):
    me, mb, _ = match(ev, bd)
    p = mb.mean() if len(bd) else 0.0
    r = me.mean()
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def best_threshold(sc, pk, ev):
    cands = np.quantile(sc[pk], np.linspace(0.0, 0.99, 100))
    res = [(f1(ev, pk[sc[pk] > c])[0], c) for c in cands]
    return max(res)[1]


def main():
    sets = {}
    for name, sfile, cut in [('tail', 'state_train.npz', 900 * EP), ('val', 'state_val.npz', 0)]:
        s = dict(np.load(ROOT / 'runs/114' / sfile))
        s = {k: v[cut:cut + 100 * EP] for k, v in s.items() if len(v) == len(s['buttons'])}
        sp = np.load(OUT / f'surprise_{name}.npz')
        masks = np.load(OUT / f'masks_{name}.npy').astype(bool)
        T, Kd, O = true_events(s)
        local = T % EP
        ok = (local >= 3 * SKIP) & (local <= EP - 1)                      # a score exists there
        sets[name] = dict(err=sp['err'].astype(np.float32), chg=sp['chg'].astype(np.float32), masks=masks,
                          T=T[ok], K=Kd[ok], O=O[ok], dropped=int((~ok).sum()), grip=gripper_events(s))
    tail, val = sets['tail'], sets['val']
    arm = OBJECTS.index('arm')
    res = {'events': {k: int((val['K'] == k).sum()) for k in np.unique(val['K'])},
           'events_dropped_no_score': val['dropped']}
    det = {}
    for d in ['surprise', 'change', 'change_arm_masked']:
        sig = 'err' if d == 'surprise' else 'chg'
        out = {}
        for nm, S in sets.items():
            z = zscore(S[sig], tail[sig])
            allowed = None
            if d == 'change_arm_masked':
                am = S['masks'][:, arm]
                allowed = ~(am | np.r_[np.zeros((SKIP, 196), bool), am[:-SKIP]])
            sc = frame_score(z, allowed)
            out[nm] = dict(z=z, sc=sc, pk=peaks(sc))
        thr = best_threshold(out['tail']['sc'], out['tail']['pk'], tail['T'])
        pk = out['val']['pk']
        bd = pk[out['val']['sc'][pk] > thr]
        F, P, R = f1(val['T'], bd)
        lf_thr = np.quantile(out['tail']['sc'][np.isfinite(out['tail']['sc'])], 0.97)
        F2 = f1(val['T'], pk[out['val']['sc'][pk] > lf_thr])[0]
        me, mb, pairs = match(val['T'], bd)
        entry = dict(f1=round(F, 4), precision=round(P, 4), recall=round(R, 4), boundaries=int(len(bd)),
                     f1_label_free_top3pct=round(F2, 4),
                     recall_by_kind={k: round(float(me[val['K'] == k].mean()), 4) for k in np.unique(val['K'])})
        # unmatched boundaries near the gripper closing or opening
        g = val['grip']
        um = bd[~mb]
        near = np.abs(um[:, None] - g[None, :]).min(1) <= TOL if len(um) and len(g) else np.zeros(0, bool)
        entry['unmatched_near_gripper_change'] = round(float(near.mean()), 4) if len(um) else 0.0
        if d == 'surprise':
            z = out['val']['z']
            hits, kinds = [], []
            for i, j in pairs:
                tau = bd[j]
                top = int(np.nanargmax(z[tau]))
                obj = val['O'][i]
                m = (val['masks'][tau][obj].any(0) | val['masks'][tau - SKIP][obj].any(0)).reshape(14, 14)
                dil = m.copy()
                dil[1:] |= m[:-1]; dil[:-1] |= m[1:]
                d2 = dil.copy()
                d2[:, 1:] |= dil[:, :-1]; d2[:, :-1] |= dil[:, 1:]
                hits.append(bool(d2.reshape(-1)[top])); kinds.append(val['K'][i])
            hits, kinds = np.array(hits), np.array(kinds)
            entry['localised'] = round(float(hits.mean()), 4)
            entry['localised_by_kind'] = {k: round(float(hits[kinds == k].mean()), 4) for k in np.unique(kinds)}
            rng = np.random.default_rng(0)
            valid = np.where(np.isfinite(out['val']['sc']))[0]
            rf = [f1(val['T'], np.sort(rng.choice(valid, len(bd), replace=False)))[0] for _ in range(5)]
            res['random'] = dict(f1=round(float(np.mean(rf)), 4), boundaries=int(len(bd)))
        det[d] = entry
    res.update(det)
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / 'score.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
