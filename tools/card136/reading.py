"""Card 136: the gate and a first reading of the current memory on the memory world.

For each world: the observer's probability (world.py), the base rate (share of the object's stored touches that
changed it; no memory within the episode), and the current recall (card 134.1): one record per object within the
episode (seen flag, last seen value, whether the last touch changed it, log of 1 + time since seen), key = the
touched object's record + the records chosen by contrast over the store, probability = share of the 25 nearest
stored touches of that object that changed it. Nothing is kept from one episode to the next, as in the model now.
Episodes 300-499 are scored in blocks of 20; the store is every episode before the block.

  bin/prun python tools/card136/reading.py gate → runs/136/gate.json   (observer and base rate only)
  bin/prun python tools/card136/reading.py      → runs/136/reading.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world as W

ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS, START, BLOCK, K, KP, GAIN, FOLDS, MIN = (0, 1, 2), 300, 20, 5, 25, 1.0, 5, 10


def logloss(y, p):
    p = np.clip(p, 0.02, 0.98)
    return float(-(y * np.log(p) + (~y) * np.log(1 - p)).mean())


def calibration(y, p):
    b = np.minimum((p * 10).astype(int), 9)
    return float(sum((b == i).mean() * abs(p[b == i].mean() - y[b == i].mean()) for i in range(10) if (b == i).any()))


def bal(pred, y):
    return 0.5 * (pred[y].mean() + (~pred[~y]).mean()) if y.any() and (~y).any() else float((pred == y).mean())


def records(ev):
    R = np.zeros((len(ev['ep']), W.N, 4), np.float32)
    rec, when, ep = np.zeros((W.N, 3), np.float32), np.zeros(W.N), -1
    for i, (e, t, o, c, v) in enumerate(zip(ev['ep'], ev['t'], ev['obj'], ev['changed'], ev['value'])):
        if e != ep:
            rec[:], ep = 0, e
        R[i, :, :3] = rec
        R[i, :, 3] = np.where(rec[:, 0] > 0, np.log1p(np.maximum(t - when, 0)), 0)
        rec[o] = (1, v, c); when[o] = t
    return R


def knn(A, Ya, Q, k):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    d = torch.cdist(torch.tensor((Q - mu) / sd), torch.tensor((A - mu) / sd))
    return Ya[d.topk(min(k, len(A)), largest=False).indices.numpy()]


def key(R, rows, o, extra):
    return R[rows][:, [o] + extra].reshape(len(rows), -1)


def select(R, y, ep, rows, o):
    def held(extra):
        Kx, pred = key(R, rows, o, extra), np.zeros(len(rows), bool)
        for f in range(FOLDS):
            q = ep[rows] % FOLDS == f
            if q.any() and (~q).any():
                pred[q] = knn(Kx[~q], y[rows][~q], Kx[q], K).mean(1) > 0.5
        return 100 * bal(pred, y[rows])
    if min(y[rows].sum(), (~y[rows]).sum()) < MIN:
        return []
    extra, best = [], held([])
    while True:
        sc = {x: held(extra + [x]) for x in range(W.N) if x != o and x not in extra}
        x = max(sc, key=sc.get)
        if sc[x] < best + GAIN:
            return extra
        extra.append(x); best = sc[x]


def main():
    gate = len(sys.argv) > 1 and sys.argv[1] == 'gate'
    out = {}
    for seed in SEEDS:
        ev = W.stream(seed)
        y, ep, obj = ev['changed'].astype(bool), ev['ep'], ev['obj']
        test = ep >= START
        base, rec = np.zeros(len(y)), np.zeros(len(y))
        R = None if gate else records(ev)
        chosen = {}
        for b0 in range(START, W.EPS, BLOCK):
            st, blk = ep < b0, (ep >= b0) & (ep < b0 + BLOCK)
            for o in range(W.N):
                S, Q = np.where(st & (obj == o))[0], np.where(blk & (obj == o))[0]
                base[Q] = y[S].mean()
                if not gate and len(Q):
                    extra = select(R, y, ep, S, o)
                    chosen.setdefault(W.ROLES[ev['role'][S[0]]], []).append([W.ROLES[int(np.where(np.array(ev['world']['obj']) == x)[0][0])] for x in extra])
                    rec[Q] = knn(key(R, S, o, extra), y[S], key(R, Q, o, extra), KP).mean(1)
        o_ = dict(world=ev['world'], observer_calibration_gap=round(100 * calibration(y[test], ev['p_obs'][test]), 2), roles={})
        for r, name in enumerate(W.ROLES):
            q = test & (ev['role'] == r)
            d = dict(n=int(q.sum()), changed=round(float(y[q].mean()), 3), base=round(logloss(y[q], base[q]), 3), observer=round(logloss(y[q], ev['p_obs'][q]), 3))
            if not gate:
                d['recall'] = round(logloss(y[q], rec[q]), 3)
                d['gap_closed'] = round((d['base'] - d['recall']) / max(d['base'] - d['observer'], 1e-6), 2)
                d['chosen_last_block'] = chosen[name][-1]
                if r in W.PARENTS:                                                       # cold touches: neither the child nor its parents seen yet in this episode
                    wo = np.array(ev['world']['obj'])
                    cold = q & (R[:, wo[r], 0] == 0) & np.all(R[:, wo[W.PARENTS[r] + ([7] if r == 16 else [])], 0] == 0, axis=1)
                    d['cold'] = dict(n=int(cold.sum()), base=round(logloss(y[cold], base[cold]), 3), observer=round(logloss(y[cold], ev['p_obs'][cold]), 3), recall=round(logloss(y[cold], rec[cold]), 3))
                    d['cold']['gap_closed'] = round((d['cold']['base'] - d['cold']['recall']) / max(d['cold']['base'] - d['cold']['observer'], 1e-6), 2) if d['cold']['base'] - d['cold']['observer'] > 0.05 else None
            o_['roles'][name] = d
        if not gate:
            q = test & (ev['role'] == 16)
            o_['rewired_by_episodes_since'] = {f'{lo}-{hi}': dict(recall=round(logloss(y[m], rec[m]), 3), observer=round(logloss(y[m], ev['p_obs'][m]), 3), base=round(logloss(y[m], base[m]), 3))
                                               for lo, hi in ((0, 19), (20, 49), (50, 149)) for m in [q & (ev['since_rewire'] >= lo) & (ev['since_rewire'] <= hi)] if m.any()}
        out[seed] = o_
        print('world', seed, 'observer calibration gap', o_['observer_calibration_gap'], flush=True)
        for name, d in o_['roles'].items():
            if name.startswith('ch_'):
                print(f'  {name:14s}', d, flush=True)
        if not gate:
            print('  rewired', o_['rewired_by_episodes_since'], flush=True)
    ch = [n for n in W.ROLES if n.startswith('ch_')]
    out['mean'] = {n: {k: round(float(np.mean([out[s]['roles'][n][k] for s in SEEDS])), 3) for k in ('base', 'observer') + (() if gate else ('recall', 'gap_closed'))} for n in ch}
    print('mean'); [print(f'  {n:14s}', v) for n, v in out['mean'].items()]
    json.dump(out, open(ROOT / f'runs/136/{"gate" if gate else "reading"}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
