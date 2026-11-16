"""Card 134.1: recall keyed per object.

Test bed and view of card 134 (true visible state; a lock is seen only when the lock itself is touched). Memory of
the episode is one record per object: seen flag, last seen value, touched flag, effect of the last touch on it, and
whether that touch changed it. Recall answers the effect on each object B of touching object A separately, among
stored touches of A, with a key made of A's record plus the records that the store shows to matter for (A, B).
Those are chosen by contrast: a record (or the contact point) is added while it raises, by at least GAIN points,
the balanced accuracy of "B changed or not" recalled for episodes of the store held out in turn. Nothing is
trained; items are scaled to unit spread, equal weight.

  bin/prun python tools/card134.1/factored.py gate   → runs/134.1/gate.json   (store only, episodes held out)
  bin/prun python tools/card134.1/factored.py        → runs/134.1/factored.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card134'))
import effects as E
from effects import B, C

K, KP, GAIN, FOLDS, MIN = 5, 25, 1.0, 5, 10
NAMES = ('cube', 'drawer', 'window', 'l0', 'l1')
SEEN = {0: slice(9, 13), 1: slice(13, 15), 2: slice(15, 17), 3: slice(17, 19), 4: slice(19, 21)}
EFF = {0: slice(21, 25), 1: slice(25, 27), 2: slice(27, 29), 3: slice(29, 31), 4: slice(31, 33)}
YS = {0: slice(0, 3), 1: slice(3, 4), 2: slice(4, 5), 3: slice(5, 6), 4: slice(6, 7)}
THR = {0: 2.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}


def records(X, flag=True):
    """contents of card 134 → dict: 'contact' and one record per object."""
    X = X.numpy()
    r = {'contact': X[:, :3]}
    for o, n in enumerate(NAMES):
        e = X[:, EFF[o]]
        parts = [X[:, SEEN[o]], e]
        if flag:
            parts.append((np.linalg.norm(e[:, 1:], axis=1, keepdims=True) > 0.5).astype(np.float32))
        r[n] = np.concatenate(parts, 1)
    return r, X[:, 3:9].argmax(1) - 1


def key(r, rows, a, extra):
    cols = ([r[NAMES[a]][rows]] if a >= 0 else []) + [r[c][rows] for c in extra]
    return np.concatenate(cols, 1) if cols else np.zeros((len(rows), 1), np.float32)


def knn(A, Ya, Q, k):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    d = torch.cdist(torch.tensor((Q - mu) / sd), torch.tensor((A - mu) / sd))
    return Ya[d.topk(min(k, len(A)), largest=False).indices.numpy()]                      # [Q,k,dim]


def changed(y, b):
    return np.linalg.norm(y, axis=-1) > THR[b]


def held_out(r, Y, ep, rows, a, b, extra):
    Kx, Yb = key(r, np.arange(len(Y)), a, extra)[rows], Y[rows][:, YS[b]]
    pred = np.zeros(len(rows), bool)
    for f in range(FOLDS):
        q = ep[rows] % FOLDS == f
        if q.any() and (~q).any():
            pred[q] = changed(np.median(knn(Kx[~q], Yb[~q], Kx[q], K), 1), b)
    return 100 * B.bal(pred, changed(Yb, b))


def select(r, tgt, Y, ep, log=None):
    """for each (touched a, affected b): the records added to a's own by contrast over the store."""
    sel = {}
    for a in range(-1, 5):
        rows = np.where(tgt == a)[0]
        for b in range(5):
            c = changed(Y[rows][:, YS[b]], b)
            if min(c.sum(), (~c).sum()) < MIN:
                continue
            extra, best = [], held_out(r, Y, ep, rows, a, b, [])
            start = best
            while True:
                cand = [x for x in ('contact',) + NAMES if x not in extra and (a < 0 or x != NAMES[a])]
                sc = {x: held_out(r, Y, ep, rows, a, b, extra + [x]) for x in cand}
                x = max(sc, key=sc.get)
                if sc[x] < best + GAIN:
                    break
                extra.append(x); best = sc[x]
            sel[(a, b)] = extra
            if log is not None:
                log[f'{"none" if a < 0 else NAMES[a]}->{NAMES[b]}'] = dict(own=round(start, 1), added=extra, held_out=round(best, 1), n=[int(len(rows)), int(c.sum())])
    return sel


def answer(sel, rs, tgs, Ys, rq, tgq, slide):
    """per query: does the slide move, does the cube move, probability that the slide moves."""
    n = len(tgq)
    pull, cube, prob = np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan)
    for a in range(-1, 5):
        S, Q = np.where(tgs == a)[0], np.where(tgq == a)[0]
        if not len(Q):
            continue
        for b in range(5):
            if (a, b) not in sel:
                continue
            A, Qk = key(rs, S, a, sel[(a, b)]), key(rq, Q, a, sel[(a, b)])
            Yb = Ys[S][:, YS[b]]
            if b == 0:
                cube[Q] = changed(np.median(knn(A, Yb, Qk, K), 1), 0)
            elif b in (1, 2):
                m = slide[Q] == b
                pull[Q[m]] = changed(np.median(knn(A, Yb, Qk[m], K), 1), b) if m.any() else False
                if m.any():
                    prob[Q[m]] = changed(knn(A, Yb, Qk[m], KP), b).mean(1)
    return pull, cube, np.nan_to_num(prob, nan=0.0)


def main():
    gate = len(sys.argv) > 1 and sys.argv[1] == 'gate'
    mem = C.load(C.MEM)
    tgm, chm, _ = C.targets(mem)
    lk, _ = B.lock_of(mem, tgm, chm)
    Xm, Ym, mm = E.events(mem, tgm, chm, lk)
    Ym = Ym.numpy()
    out = {}
    if gate:                                                                             # store only: fifth of the episodes held out in turn
        for name, flag in (('record', False), ('record_with_changed_flag', True)):
            r, tg = records(Xm['effects'], flag)
            pull, cube, prob = (np.zeros(len(tg), bool), np.zeros(len(tg), bool), np.zeros(len(tg)))
            log = {}
            for f in range(FOLDS):
                tr = np.where(~mm['mixed'] & (mm['ep'] % FOLDS != f))[0]
                te = np.where(mm['ep'] % FOLDS == f)[0]
                sub = lambda rows: {k: v[rows] for k, v in r.items()}
                sel = select(sub(tr), tg[tr], Ym[tr], mm['ep'][tr], log if f == 0 else None)
                for nm, v in zip(('p', 'c', 'q'), answer(sel, sub(tr), tg[tr], Ym[tr], sub(te), tg[te], mm['slide'][te])):
                    {'p': pull, 'c': cube, 'q': prob}[nm][te] = v
            out[name] = dict(score=E.score(pull, cube, prob, mm, None), selection_fold0=log)
            print(name, json.dumps(out[name]['score']), flush=True)
            for k, v in log.items():
                print('  ', k, v, flush=True)
        json.dump(out, open(C.ROOT / 'runs/134.1/gate.json', 'w'), indent=1)
        return
    test = C.load(C.TEST)
    tgt, cht, _ = C.targets(test)
    Xt, Yt, mt = E.events(test, tgt, cht, lk)
    rq, tq = records(Xt['effects'])
    r, tg = records(Xm['effects'])
    out['n'] = {g: [int((mt['pull'] & (mt['group'] == i)).sum()), int((mt['pull'] & (mt['group'] == i) & ~mt['moved']).sum())] for i, g in enumerate(E.GROUPS)}
    for n, draws in ((400, 1), (100, 3)):
        rows_ = []
        for s in range(draws):
            keep = np.where(~mm['mixed'] & np.isin(mm['ep'], np.random.default_rng(s).choice(400, n, replace=False)))[0]
            sub = {k: v[keep] for k, v in r.items()}
            log = {}
            sel = select(sub, tg[keep], Ym[keep], mm['ep'][keep], log)
            rows_.append(E.score(*answer(sel, sub, tg[keep], Ym[keep], rq, tq, mt['slide']), mt, None))
            if s == 0:
                out[f'selection_{n}'] = log
        out[n] = {k: round(float(np.mean([x[k] for x in rows_])), 3 if 'logloss' in k or 'prob' in k else 1) for k in rows_[0]}
        print(n, json.dumps(out[n]), flush=True)
        for k, v in out[f'selection_{n}'].items():
            print('  ', k, v, flush=True)
    json.dump(out, open(C.ROOT / 'runs/134.1/factored.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
