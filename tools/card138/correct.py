"""Card 138: the key's learned correction on the memory world, records never emptied (card 137).

For each touched object the key of card 134.1 (its own record + the records chosen by contrast; items scaled to unit
spread, equal weight) is the prior. Arms:
  prior     card 137 as it is
  weights   prior x exp(delta), one delta per item, starting at 0                       size = sum of delta^2
  network   the same weights, plus 8 more dimensions from a small network over the key's items, starting near 0
                                                                                        size = sum delta^2 + mean squared length of the new dimensions
  hand      reference, uses the world's truth: every record but the carry switch's is emptied at the start of each
            episode (what a key would do if it knew which old values count)
Training (store only, per touched object): each stored touch's outcome is predicted from the stored touches of other
episodes as the softmax(-distance^2 / tau)-weighted share that changed (tau = the median squared distance to the
25th nearest under the prior); loss = log loss / the prior's log loss + beta / (stored touches) x size. beta is one
number per world and arm, chosen among BETAS on a fifth of the stored episodes held out, at the first block.
Scoring as card 136.1 (cold and warm; probability = share of the 25 nearest); the store is refreshed every 40
episodes and the correction refitted every REFIT blocks.

  bin/prun python tools/card138/correct.py gate    → runs/138/gate.json     (prior and hand)
  bin/prun python tools/card138/correct.py         → runs/138/correct.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card136.1'))
import reading as R1
from reading import R0, W

ROOT, BETAS, STEPS, REFIT = R0.ROOT, (0.1, 1.0, 10.0, 100.0), 150, 2
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def fit(X, y, ep, kind, beta, seed=0):
    """→ a function from raw keys (numpy) to embedded keys (torch, cpu)."""
    torch.manual_seed(seed)
    X, Y, ept = torch.tensor(X, device=DEV), torch.tensor(y, dtype=torch.float32, device=DEV), torch.tensor(ep, device=DEV)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    z = lambda x: (x - mu) / sd
    if kind == 'prior':
        return lambda x: z(torch.tensor(x, device=DEV)).cpu()
    Z, same = z(X), ept[:, None] == ept[None]
    d0 = (torch.cdist(Z, Z) ** 2).masked_fill(same, float('inf'))
    tau = d0.topk(min(25, len(X) - 1), largest=False).values[:, -1].median().clamp(min=1e-3)
    ll = lambda p: torch.nn.functional.binary_cross_entropy(p.clamp(0.02, 0.98), Y)
    err0 = ll(torch.softmax(-d0 / tau, 1) @ Y)
    delta = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    if kind == 'weights':
        params, f, size = [delta], (lambda x: z(x) * delta.exp()), (lambda e: (delta ** 2).sum())
    else:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)).to(DEV)
        torch.nn.init.normal_(net[-1].weight, std=1e-2); torch.nn.init.zeros_(net[-1].bias)
        params, f = [delta] + list(net.parameters()), (lambda x: torch.cat([z(x) * delta.exp(), net(z(x))], 1))
        size = lambda e: (delta ** 2).sum() + (e[:, X.shape[1]:] ** 2).sum(1).mean()
    opt = torch.optim.Adam(params, lr=1e-2 if kind == 'weights' else 3e-3)
    for _ in range(STEPS):
        e = f(X)
        d = (torch.cdist(e, e) ** 2).masked_fill(same, float('inf'))
        loss = ll(torch.softmax(-d / tau, 1) @ Y) / err0 + beta / len(X) * size(e)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda x: f(torch.tensor(x, device=DEV)).detach().cpu()


def share(g, Ka, ya, Kq):
    d = torch.cdist(g(Kq), g(Ka))
    return ya[d.topk(min(R0.KP, len(Ka)), largest=False).indices.numpy()].mean(1)


def hand_records(ev):
    Rp, cold = R1.records(ev, True)
    Re, _ = R1.records(ev, False)
    Rh = np.concatenate([Re, Re[:, :, :1]], 2)                                           # emptied each episode; "seen in this episode" = seen
    c = ev['world']['obj'][2]
    Rh[:, c] = Rp[:, c]
    return Rp, Rh, cold


def main():
    gate = 'gate' in sys.argv
    arms = ('prior', 'hand') if gate else ('prior', 'weights', 'network')
    out, t0 = {}, time.time()
    only = [int(a[6:]) for a in sys.argv if a.startswith('world=')]
    for seed in (only or R1.SEEDS):
        ev = W.stream(seed, EPS=R1.EPS, LEN=R1.LEN, REWIRE=R1.REWIRE, back=R1.BACK)
        y, ep, obj, wo = ev['changed'].astype(bool), ev['ep'], ev['obj'], np.array(ev['world']['obj'])
        test = ep >= R1.START
        Rp, Rh, cold_obj = hand_records(ev)
        prob = {a: np.zeros(len(y)) for a in arms}
        base = np.zeros(len(y))
        beta, maps, sizes = {}, {}, {a: [] for a in arms}
        for bi, b0 in enumerate(range(R1.START, R1.EPS, R1.BLOCK)):
            st, blk = ep < b0, (ep >= b0) & (ep < b0 + R1.BLOCK)
            sel = {}
            for o in range(W.N):
                S = np.where(st & (obj == o))[0]
                sel[o] = (S, R0.select(Rp, y, ep, S, o), R0.select(Rh, y, ep, S, o) if gate else None)
            if bi == 0 and not gate:                                                     # beta: per world and arm, on stored episodes held out
                for a in ('weights', 'network'):
                    errs = {}
                    for bt in BETAS:
                        tot = 0.0
                        for o, (S, extra, _) in sel.items():
                            Kx, held = R0.key(Rp, S, o, extra), ep[S] % 5 == 0
                            g = fit(Kx[~held], y[S][~held], ep[S][~held], a, bt)
                            tot += R0.logloss(y[S][held], share(g, Kx[~held], y[S][~held], Kx[held])) * held.sum()
                        errs[bt] = tot
                    beta[a] = min(errs, key=errs.get)
                    print('world', seed, a, 'held-out by beta', {k: round(v, 1) for k, v in errs.items()}, '→', beta[a], f'{time.time() - t0:.0f}s', flush=True)
            for o, (S, extra, extra_h) in sel.items():
                Q = np.where(blk & (obj == o))[0]
                base[Q] = y[S].mean()
                if not len(Q):
                    continue
                for a in arms:
                    R, ex = (Rh, extra_h) if a == 'hand' else (Rp, extra)
                    Ka, Kq = R0.key(R, S, o, ex), R0.key(R, Q, o, ex)
                    if a in ('prior', 'hand'):
                        g = fit(Ka, y[S], ep[S], 'prior', 0)
                    else:
                        if bi % REFIT == 0 or (a, o) not in maps or maps[(a, o)][1] != ex:
                            maps[(a, o)] = (fit(Ka, y[S], ep[S], a, beta[a]), list(ex))
                        g = maps[(a, o)][0]
                    prob[a][Q] = share(g, Ka, y[S], Kq)
        o_ = dict(beta=beta, roles={})
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
            o_['roles'][name] = d
        out[seed] = o_
        print('world', seed, 'done', f'{time.time() - t0:.0f}s', flush=True)
    if only:
        json.dump(out, open(ROOT / f'runs/138/correct_{only[0]}.json', 'w'), indent=1)
        return
    mean = lambda v: None if any(x is None for x in v) else round(float(np.mean(v)), 3)
    out['mean'] = {n: {part: {k: mean([out[s]['roles'][n][part][k] for s in R1.SEEDS]) for k in out[R1.SEEDS[0]]['roles'][n][part]} for part in ('cold', 'warm')} for n in R1.CH}
    for n in R1.CH:
        for part in ('cold', 'warm'):
            m = out['mean'][n][part]
            print(f'  {n:14s} {part} base {m["base"]} obs {m["observer"]} ' + ' '.join(f'{a} {m[a]} ({m[a + "_gap"]})' for a in arms), flush=True)
    json.dump(out, open(ROOT / f'runs/138/{"gate" if gate else "correct"}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
