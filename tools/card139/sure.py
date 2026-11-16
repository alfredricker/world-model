"""Card 139: an answer that knows when not to trust itself.

Memory world, records never emptied, key = prior x learned weights (card 138, the kept arm; beta as chosen there).
Only the answer changes:
  k25      the share of the 25 nearest stored touches that changed the object (cards 134 to 138)
  k200     the share of the 200 nearest (gate only: is the loss where nothing is known the noise of 25 samples?)
  trust    recall judges its own answer from its own past errors. On the store, each fifth of the episodes is
           answered from the others (k25). A small read-out is then fitted, per touched object, from that answer and
           the state of the records in the key (for each: seen at all, seen in this episode, log time since seen) to
           what happened:   logit p = a L + b + c.s + L d.s,   L = logit of the k25 share, s = the state items.
           It starts at a = 1 and the rest 0 (trust recall as it is) and pays 10 / (stored touches) x its squared
           distance from that start. Where the records' state has gone with wrong answers, the answer is pulled
           toward what happened in such cases.

  bin/prun python tools/card139/sure.py gate world=N → runs/139/gate_N.json     (k25, k200)
  bin/prun python tools/card139/sure.py world=N      → runs/139/sure_N.json     (k25, trust)
  bin/prun python tools/card139/sure.py merge [gate] → runs/139/sure.json | gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card138'))
import correct as C8
from correct import R0, R1, W

ROOT = R0.ROOT
BETA = {int(k): v['weights'] for k, v in json.load(open(ROOT / 'runs/138/correct.json'))['beta'].items()}
logit = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))


def state(R, rows, o, extra):
    return R[rows][:, [o] + extra][:, :, [0, 4, 3]].reshape(len(rows), -1)                # seen, seen in this episode, log time since seen


def trust_fit(E, y, ep, S_state):
    """→ a function (k25 share, state items) → probability."""
    p = np.zeros(len(y))
    for f in range(5):
        q = ep % 5 == f
        p[q] = share(E[~q], y[~q], E[q], 25)
    mu, sd = S_state.mean(0), S_state.std(0) + 1e-6
    feats = lambda p, s: torch.tensor(np.c_[logit(p), (s - mu) / sd, logit(p)[:, None] * (s - mu) / sd], dtype=torch.float32)
    F, Y = feats(p, S_state), torch.tensor(y, dtype=torch.float32)
    w0 = torch.zeros(F.shape[1]); w0[0] = 1
    w, b = w0.clone().requires_grad_(True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(F @ w + b, Y) + 10 / len(y) * (((w - w0) ** 2).sum() + b ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    w, b = w.detach(), b.detach()
    return lambda p, s: torch.sigmoid(feats(p, s) @ w + b).numpy()


def share(E, y, Eq, k):
    return y[torch.cdist(Eq, E).topk(min(k, len(E)), largest=False).indices.numpy()].mean(1)


def run(seed, gate):
    arms = ('k25', 'k200') if gate else ('k25', 'trust')
    ev = W.stream(seed, EPS=R1.EPS, LEN=R1.LEN, REWIRE=R1.REWIRE, back=R1.BACK)
    y, ep, obj, wo = ev['changed'].astype(bool), ev['ep'], ev['obj'], np.array(ev['world']['obj'])
    test = ep >= R1.START
    Rp, cold_obj = R1.records(ev, True)
    prob, base = {a: np.zeros(len(y)) for a in arms}, np.zeros(len(y))
    maps = {}
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
                g = C8.fit(Ka, y[S], ep[S], 'weights', BETA[seed])
                maps[o] = (g, list(ex), None if gate else trust_fit(g(Ka), y[S], ep[S], state(Rp, S, o, ex)))
            g, _, tm = maps[o]
            E, Eq = g(Ka), g(Kq)
            prob['k25'][Q] = share(E, y[S], Eq, 25)
            if gate:
                prob['k200'][Q] = share(E, y[S], Eq, 200)
            else:
                prob['trust'][Q] = tm(prob['k25'][Q], state(Rp, Q, o, ex))
    out = dict(roles={}, calibration_gap={a: round(100 * R0.calibration(y[test], prob[a][test]), 2) for a in arms})
    for r, name in enumerate(W.ROLES):
        if r not in W.PARENTS:
            continue
        q = test & (ev['role'] == r)
        cold = cold_obj[:, wo[r]] & np.all(cold_obj[:, wo[W.PARENTS[r] + ([7] if r == 16 else [])]], axis=1)
        d = {}
        for part, m in (('cold', q & cold), ('warm', q & ~cold)):
            e = dict(n=int(m.sum()), base=round(R0.logloss(y[m], base[m]), 3), observer=round(R0.logloss(y[m], ev['p_obs'][m]), 3))
            room = e['base'] - e['observer']
            for a in arms:
                e[a] = round(R0.logloss(y[m], prob[a][m]), 3)
                e[a + '_gap'] = round((e['base'] - e[a]) / room, 3) if room >= 0.1 else None
            d[part] = e
        out['roles'][name] = d
    return out


def main():
    gate = 'gate' in sys.argv
    stem = 'gate' if gate else 'sure'
    if 'merge' in sys.argv:
        d = {s: json.load(open(ROOT / f'runs/139/{stem}_{s}.json')) for s in R1.SEEDS}
        mean = lambda v: None if any(x is None for x in v) else round(float(np.mean(v)), 3)
        out = dict(worlds=d, mean={n: {part: {k: mean([d[s]['roles'][n][part][k] for s in R1.SEEDS]) for k in d[0]['roles'][n][part]} for part in ('cold', 'warm')} for n in R1.CH},
                   calibration_gap={a: mean([d[s]['calibration_gap'][a] for s in R1.SEEDS]) for a in d[0]['calibration_gap']})
        for n in R1.CH:
            for part in ('cold', 'warm'):
                print(f'  {n:14s} {part}', out['mean'][n][part])
        print(out['calibration_gap'])
        json.dump(out, open(ROOT / f'runs/139/{stem}.json', 'w'), indent=1)
        return
    seed = [int(a[6:]) for a in sys.argv if a.startswith('world=')][0]
    json.dump(run(seed, gate), open(ROOT / f'runs/139/{stem}_{seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
