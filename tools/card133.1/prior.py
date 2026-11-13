"""Card 133.1: the recall key as a prior plus a correction that the stored evidence has to pay for.

Card 133's test (true visible state, narrow view; recall of a touch's effect; 5 nearest, median).
  prior    every item of the key (contact point, touched object, last seen values and "seen" flags) scaled to unit
           spread over the store, equal weight
  weights  prior x exp(delta), one delta per item, starting at 0;          size = sum of delta^2
  network  prior with 16 more dimensions from a small network, starting near 0; size = mean squared length of them
Training (store only): the effect of each stored event is predicted from the other episodes as the
softmax(-distance^2 / tau)-weighted mean (tau = the median squared distance to the nearest stored event under the
prior); loss = error / the prior's error + beta / (stored events) x size.
beta is chosen among BETAS by the error of 5-nearest recall on a fifth of the stored episodes held out from
training (seed 0), then every seed trains on the whole store.

  bin/prun python tools/card133.1/prior.py --gate
  bin/prun python tools/card133.1/prior.py --episodes 100      → runs/133.1/prior_<episodes>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card133'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'card132'))
import bed as B
import check as C
import key as K133

ROOT, BETAS, STEPS, SEEDS = C.ROOT, [0.1, 1.0, 10.0, 100.0, 1000.0], 2000, 3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def fit(X, Y, ep, kind, beta, seed):
    """→ a function from raw keys to embedded keys."""
    torch.manual_seed(seed)
    X, Y, ept = X.to(DEV), Y.to(DEV), torch.tensor(ep, device=DEV)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    z = lambda x: (x - mu) / sd
    if kind == 'prior':
        return lambda x: z(x.to(DEV)).cpu()
    Z = z(X)
    same = ept[:, None] == ept[None]
    d0 = (torch.cdist(Z, Z) ** 2).masked_fill(same, float('inf'))
    tau = d0.min(1).values.median()
    err0 = ((torch.softmax(-d0 / tau, 1) @ Y - Y) ** 2).mean()
    if kind == 'weights':
        delta = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
        params, f, size = [delta], (lambda x: z(x) * delta.exp()), (lambda e: (delta ** 2).sum())
        lr = 1e-2
    else:
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 16)).to(DEV)
        torch.nn.init.normal_(net[-1].weight, std=1e-2); torch.nn.init.zeros_(net[-1].bias)   # exactly zero gives no gradient: every pair is then at distance 0 in the new dimensions
        params, f, size = list(net.parameters()), (lambda x: torch.cat([z(x), net(z(x))], 1)), (lambda e: (e[:, X.shape[1]:] ** 2).sum(1).mean())
        lr = 1e-3
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(STEPS):
        q = torch.randint(0, len(X), (min(512, len(X)),), device=DEV)
        eq = f(X[q])
        d = (torch.cdist(eq, f(X)) ** 2).masked_fill(ept[q][:, None] == ept[None], float('inf'))
        loss = ((torch.softmax(-d / tau, 1) @ Y - Y[q]) ** 2).mean() / err0 + beta / len(X) * size(eq)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda x: f(x.to(DEV)).detach().cpu()


def held_out_error(X, Y, ep, kind, beta):
    eps = np.unique(ep)
    out = np.isin(ep, np.random.default_rng(0).choice(eps, max(1, len(eps) // 5), replace=False))
    g = fit(X[~out], Y[~out], ep[~out], kind, beta, 0)
    return float(((K133.recall(g(X[~out]), Y[~out], g(X[out])) - Y[out]) ** 2).mean())


def main():
    n_ep, gate = K133.arg('--episodes', 400), '--gate' in sys.argv
    mem, test = C.load(C.MEM), C.load(C.TEST)
    tgm, chm, _ = C.targets(mem)
    tgt, cht, _ = C.targets(test)
    lk, _ = B.lock_of(mem, tgm, chm)
    Xa, _, Ya, mm = K133.events(mem, tgm, chm, lk)
    Xt, _, Yt, mt = K133.events(test, tgt, cht, lk)
    wf = torch.tensor(K133.W_FIXED, dtype=torch.float32)

    def store(n, seed):
        keep = ~mm['mixed'] & np.isin(mm['ep'], np.random.default_rng(seed).choice(400, n, replace=False))
        return Xa[keep], Ya[keep], mm['ep'][keep]

    def mean(rows):
        return {k: round(float(np.mean([r[k] for r in rows])), 1) for k in rows[0] if rows[0][k] is not None}

    if gate:
        out = {}
        for n in (30, 100, 400):
            pr, hs = [], []
            for s in range(SEEDS):
                X, Y, ep = store(n, s)
                g = fit(X, Y, ep, 'prior', 0, s)
                pr.append(K133.score(K133.recall(g(X), Y, g(Xt)), mt)); hs.append(K133.score(K133.recall(X * wf, Y, Xt * wf), mt))
            out[n] = dict(prior=mean(pr), hand_set=mean(hs))
            print(n, out[n], flush=True)
        json.dump(out, open(ROOT / 'runs/133.1/gate.json', 'w'), indent=1)
        return
    out = dict(episodes=n_ep, arms={})
    rng = np.random.default_rng(1)
    for kind in ('prior', 'weights', 'network'):
        beta = 0.0
        if kind != 'prior':
            X, Y, ep = store(n_ep, 0)
            errs = {b: held_out_error(X, Y, ep, kind, b) for b in BETAS}
            beta = min(errs, key=errs.get)
            print(kind, 'held-out error by beta', {b: round(e, 3) for b, e in errs.items()}, '→', beta, flush=True)
        rows = []
        for s in range(SEEDS):
            X, Y, ep = store(n_ep, s)
            g = fit(X, Y, ep, kind, beta, s)
            Es = g(X)
            r = K133.score(K133.recall(Es, Y, g(Xt)), mt)
            for name, sl in (('lock_shuffled', K133.LOCK), ('cube_shuffled', K133.CUBE)):
                Xs = Xt.clone(); Xs[:, sl] = Xt[torch.tensor(rng.permutation(len(Xt))), sl]
                r[name] = K133.score(K133.recall(Es, Y, g(Xs)), mt)['pull']
            rows.append(r)
        out['arms'][kind] = dict(mean(rows), beta=beta, seeds=rows, **({'held_out': errs} if kind != 'prior' else {}))
        print(kind, out['arms'][kind] | dict(seeds=None), flush=True)
    json.dump(out, open(ROOT / f'runs/133.1/prior_{n_ep}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
