"""Card 140: which period of my past is the present like?

Memory world, records never emptied, key = prior x learned weights, answer = share of the 25 nearest (card 138's kept
arm). New: the store is also read period by period (stored episodes in stretches of PERIOD), and the answer is a
mixture:
  experts   "all" (the whole store, card 138's answer) and one per stored period with at least 25 touches of the
            object: the share of the 25 nearest stored touches within that period
  weights   posterior over experts given how each answered the object's last WINDOW touches (whose outcomes the
            agent has seen):  weight = prior x exp(-sum of their log losses);  prior: half on "all", half spread
            over the periods
  answer    the weighted mean of the experts' probabilities
Where the world has not changed every period answers alike and "all", with the most cases, keeps the weight; after a
change the periods that resemble the present take over, old or recent.
Arms: k25 (card 138), periods (this card), hand (reference, rewired child only: the store cut down, using the world's
truth, to the episodes wired as now).

  bin/prun python tools/card140/periods.py world=N   → runs/140/periods_N.json
  bin/prun python tools/card140/periods.py merge     → runs/140/periods.json
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card139'))
import sure as C9
from sure import C8, R0, R1, W

ROOT, PERIOD, WINDOW = R0.ROOT, 80, 20
ARMS = ('k25', 'periods', 'hand')
ll1 = lambda y, p: -np.log(np.clip(p if y else 1 - p, 0.02, 0.98))


def run(seed):
    ev = W.stream(seed, EPS=R1.EPS, LEN=R1.LEN, REWIRE=R1.REWIRE, back=R1.BACK)
    y, ep, obj, wo = ev['changed'].astype(bool), ev['ep'], ev['obj'], np.array(ev['world']['obj'])
    test = ep >= R1.START
    Rp, cold_obj = R1.records(ev, True)
    prob, base = {a: np.zeros(len(y)) for a in ARMS}, np.zeros(len(y))
    maps, hist, wall = {}, defaultdict(lambda: defaultdict(lambda: deque(maxlen=WINDOW))), np.zeros(len(y))
    for bi, b0 in enumerate(range(R1.START, R1.EPS, R1.BLOCK)):
        st, blk = ep < b0, (ep >= b0) & (ep < b0 + R1.BLOCK)
        for o in range(W.N):
            S, Q = np.where(st & (obj == o))[0], np.where(blk & (obj == o))[0]
            base[Q] = y[S].mean()
            if not len(Q):
                continue
            ex = R0.select(Rp, y, ep, S, o)
            Ka, Kq = R0.key(Rp, S, o, ex), R0.key(Rp, Q, o, ex)
            if bi % C8.REFIT == 0 or o not in maps or maps[o][1] != ex:
                maps[o] = (C8.fit(Ka, y[S], ep[S], 'weights', C9.BETA[seed]), list(ex))
            g = maps[o][0]
            E, Eq = g(Ka), g(Kq)
            P = {'all': C9.share(E, y[S], Eq, 25)}
            for j in np.unique(ep[S] // PERIOD):
                m = ep[S] // PERIOD == j
                if m.sum() >= 25:
                    P[int(j)] = C9.share(E[m], y[S][m], Eq, 25)
            prob['k25'][Q] = P['all']
            same = (ep[S] // R1.REWIRE) % 2 == (b0 // R1.REWIRE) % 2
            prob['hand'][Q] = C9.share(E[same], y[S][same], Eq, 25) if ev['role'][Q[0]] == 16 and same.sum() >= 25 else P['all']
            names = list(P)
            prior = np.array([0.5 if n == 'all' else 0.5 / max(len(names) - 1, 1) for n in names])
            for i, q in enumerate(Q):                                                    # in time order: the outcomes of earlier touches are known
                w = prior * np.exp(-np.array([sum(hist[o][n]) for n in names]))
                w = w / w.sum()
                prob['periods'][q] = float(sum(w[k] * P[n][i] for k, n in enumerate(names)))
                wall[q] = w[0]
                for n in names:
                    hist[o][n].append(ll1(y[q], P[n][i]))
    out = dict(roles={}, calibration_gap={a: round(100 * R0.calibration(y[test], prob[a][test]), 2) for a in ARMS})
    for r, name in enumerate(W.ROLES):
        if r not in W.PARENTS:
            continue
        q = test & (ev['role'] == r)
        cold = cold_obj[:, wo[r]] & np.all(cold_obj[:, wo[W.PARENTS[r] + ([7] if r == 16 else [])]], axis=1)
        d = {}
        parts = [('cold', q & cold), ('warm', q & ~cold)]
        if r == 16:
            parts += [('warm_first_40_after_rewiring', q & ~cold & (ev['since_rewire'] < 40)), ('warm_later', q & ~cold & (ev['since_rewire'] >= 40))]
        for part, m in parts:
            e = dict(n=int(m.sum()), base=round(R0.logloss(y[m], base[m]), 3), observer=round(R0.logloss(y[m], ev['p_obs'][m]), 3), weight_on_all=round(float(wall[m].mean()), 2))
            room = e['base'] - e['observer']
            for a in ARMS:
                e[a] = round(R0.logloss(y[m], prob[a][m]), 3)
                e[a + '_gap'] = round((e['base'] - e[a]) / room, 3) if room >= 0.1 else None
            d[part] = e
        out['roles'][name] = d
    return out


def main():
    if 'merge' in sys.argv:
        d = {s: json.load(open(ROOT / f'runs/140/periods_{s}.json')) for s in R1.SEEDS}
        mean = lambda v: None if any(x is None for x in v) else round(float(np.mean(v)), 3)
        out = dict(worlds=d, mean={n: {part: {k: mean([d[s]['roles'][n][part][k] for s in R1.SEEDS]) for k in d[0]['roles'][n][part]} for part in d[0]['roles'][n]} for n in R1.CH},
                   calibration_gap={a: mean([d[s]['calibration_gap'][a] for s in R1.SEEDS]) for a in ARMS})
        for n in R1.CH:
            for part in out['mean'][n]:
                m = out['mean'][n][part]
                print(f'  {n:14s} {part:28s} base {m["base"]} obs {m["observer"]} | ' + ' '.join(f'{a} {m[a]} ({m[a + "_gap"]})' for a in ARMS) + f' | on all {m["weight_on_all"]}')
        print(out['calibration_gap'], {n: [d[s]['roles']['ch_rewired']['warm'][n + '_gap'] for s in R1.SEEDS] for n in ARMS})
        json.dump(out, open(ROOT / 'runs/140/periods.json', 'w'), indent=1)
        return
    seed = [int(a[6:]) for a in sys.argv if a.startswith('world=')][0]
    json.dump(run(seed), open(ROOT / f'runs/140/periods_{seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
