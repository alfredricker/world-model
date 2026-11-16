"""Quick check (2026-11-13, no card): card 138's learned weight per key item on the OGBench test bed of card 134.1.

Same store (400 episodes), same test pulls, same keys chosen by contrast. Arms: prior (card 134.1 as it is) and
weights (prior x exp(delta) per item, card 138). beta is chosen among BETAS on a fifth of the stored episodes held out.

  bin/prun python tools/card138/ogbench.py   → runs/138/ogbench.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import correct as K8
spec = importlib.util.spec_from_file_location('factored', HERE.parent / 'card134.1/factored.py')
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
E, B, C = F.E, F.B, F.C


def near(g, A, Q, k):
    return torch.cdist(g(Q), g(A)).topk(min(k, len(A)), largest=False).indices.numpy()


def answer(sel, maps, rs, tgs, Ys, rq, tgq, slide):
    n = len(tgq)
    pull, cube, prob = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n)
    for (a, b), extra in sel.items():
        S, Q = np.where(tgs == a)[0], np.where(tgq == a)[0]
        if not len(Q):
            continue
        A, Qk, Yb, g = F.key(rs, S, a, extra), F.key(rq, Q, a, extra), Ys[S][:, F.YS[b]], maps[(a, b)]
        if b == 0:
            cube[Q] = F.changed(np.median(Yb[near(g, A, Qk, F.K)], 1), 0)
        elif b in (1, 2):
            m = slide[Q] == b
            if m.any():
                pull[Q[m]] = F.changed(np.median(Yb[near(g, A, Qk[m], F.K)], 1), b)
                prob[Q[m]] = F.changed(Yb[near(g, A, Qk[m], F.KP)], b).mean(1)
    return pull, cube, prob


def main():
    mem, test = C.load(C.MEM), C.load(C.TEST)
    tgm, chm, _ = C.targets(mem)
    tgt, cht, _ = C.targets(test)
    lk, _ = B.lock_of(mem, tgm, chm)
    Xm, Ym, mm = E.events(mem, tgm, chm, lk)
    Xt, _, mt = E.events(test, tgt, cht, lk)
    Ym = Ym.numpy()
    rq, tq = F.records(Xt['effects'])
    r, tg = F.records(Xm['effects'])
    out = {}
    for n, draws in ((400, 1), (100, 3)):
        res = {'prior': [], 'weights': []}
        for s in range(draws):
            keep = np.where(~mm['mixed'] & np.isin(mm['ep'], np.random.default_rng(s).choice(400, n, replace=False)))[0]
            sub, tgs, Ys, ep = {k: v[keep] for k, v in r.items()}, tg[keep], Ym[keep], mm['ep'][keep]
            sel = F.select(sub, tgs, Ys, ep)
            data = {}
            for (a, b), extra in sel.items():
                S = np.where(tgs == a)[0]
                data[(a, b)] = (F.key(sub, S, a, extra), F.changed(Ys[S][:, F.YS[b]], b), ep[S])
            errs = {}
            for bt in K8.BETAS:
                tot = 0.0
                for Kx, y, e in data.values():
                    held = e % 5 == 0
                    g = K8.fit(Kx[~held], y[~held], e[~held], 'weights', bt)
                    p = y[~held][near(g, Kx[~held], Kx[held], F.KP)].mean(1)
                    tot += E.logloss(y[held], p) * held.sum()
                errs[bt] = tot
            beta = min(errs, key=errs.get)
            print(n, s, 'held-out by beta', {k: round(float(v), 1) for k, v in errs.items()}, '→', beta, flush=True)
            for arm in res:
                maps = {ab: K8.fit(Kx, y, e, arm, beta) for ab, (Kx, y, e) in data.items()}
                res[arm].append(E.score(*answer(sel, maps, sub, tgs, Ys, rq, tq, mt['slide']), mt, None))
        out[n] = {arm: {k: round(float(np.mean([x[k] for x in v])), 3 if 'logloss' in k or 'prob' in k else 1) for k in v[0]} for arm, v in res.items()}
        for arm in res:
            print(n, arm, json.dumps(out[n][arm]), flush=True)
    json.dump(out, open(C.ROOT / 'runs/138/ogbench.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
