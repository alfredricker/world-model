"""Card 140, after the run (not a declared arm): where does the rewired child's loss sit? Warm touches, three worlds.
  a  card 138's answer (key chosen and weighted on the whole store, whole store answers)
  b  the same key, only stored episodes wired as now answer            (card 140's reference)
  c  key = own record + the record of the true parent now, equal weight, whole store answers
  d  key as c, only stored episodes wired as now answer
  bin/prun python tools/card140/why.py → runs/140/why.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import periods as P
from periods import C8, C9, R0, R1, W

out = {}
for seed in R1.SEEDS:
    ev = W.stream(seed, EPS=R1.EPS, LEN=R1.LEN, REWIRE=R1.REWIRE, back=R1.BACK)
    y, ep, obj, wo = ev['changed'].astype(bool), ev['ep'], ev['obj'], np.array(ev['world']['obj'])
    Rp, cold_obj = R1.records(ev, True)
    o = wo[16]
    prob = {k: np.zeros(len(y)) for k in 'abcd'}
    base = np.zeros(len(y))
    for b0 in range(R1.START, R1.EPS, R1.BLOCK):
        S, Q = np.where((ep < b0) & (obj == o))[0], np.where((ep >= b0) & (ep < b0 + R1.BLOCK) & (obj == o))[0]
        base[Q] = y[S].mean()
        same = (ep[S] // R1.REWIRE) % 2 == (b0 // R1.REWIRE) % 2
        ex = R0.select(Rp, y, ep, S, o)
        g = C8.fit(R0.key(Rp, S, o, ex), y[S], ep[S], 'weights', C9.BETA[seed])
        E, Eq = g(R0.key(Rp, S, o, ex)), g(R0.key(Rp, Q, o, ex))
        prob['a'][Q] = C9.share(E, y[S], Eq, 25); prob['b'][Q] = C9.share(E[same], y[S][same], Eq, 25)
        par = [int(wo[7 if (b0 // R1.REWIRE) % 2 else 1])]
        Ka, Kq = R0.key(Rp, S, o, par), R0.key(Rp, Q, o, par)
        prob['c'][Q] = R0.knn(Ka, y[S], Kq, 25).mean(1); prob['d'][Q] = R0.knn(Ka[same], y[S][same], Kq, 25).mean(1)
    q = (ep >= R1.START) & (obj == o)
    cold = cold_obj[:, o] & cold_obj[:, wo[1]] & cold_obj[:, wo[7]]
    m = q & ~cold
    b, ob = R0.logloss(y[m], base[m]), R0.logloss(y[m], ev['p_obs'][m])
    out[seed] = {k: round((b - R0.logloss(y[m], prob[k][m])) / (b - ob), 3) for k in 'abcd'}
    print(seed, out[seed], flush=True)
json.dump(out, open(R0.ROOT / 'runs/140/why.json', 'w'), indent=1)
