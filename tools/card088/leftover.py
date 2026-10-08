"""Card 088: the relation's projection from the directions card 087's properties ignore; its data check and gate.

  G = mean over property-play tiles and ensemble members of J(z)^T J(z)     what the properties read (J: the five
                                                                            outputs' Jacobian with respect to z)
  C = covariance of z within each property profile, pooled                  what varies among tiles of one kind
  P = the top r solutions of C v = mu (G + eps I) v                         within-kind, read by no property

Scored with card 069's probe (locked door and key pairs, MiniGrid's six colours left out one at a time, and the three
unseen hues; only the threshold fitted), at r = 1, 2, 4, 8, against card 070's trained P, random directions of the
same rank, and G's own top directions.

  bin/prun python tools/card088/leftover.py          writes runs/088/gate.json and runs/088/P.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import gate as G69                                     # noqa: E402  (card 069's probe: tiles, pairs, learned)
sys.path.insert(0, str(ROOT / "tools" / "card087"))
import props as PR                                     # noqa: E402

OUT = ROOT / "runs" / "088"
RANKS = (1, 2, 4, 8)


def ensemble():
    nets = []
    for sd in torch.load(PR.OUT / "ensemble.pt"):
        n = PR.Net().cuda()
        n.load_state_dict(sd)
        nets.append(n.eval())
    return nets


def sensitivity(nets, Z):
    """G: mean J^T J over tiles and members, the five outputs' logits."""
    Zt = torch.as_tensor(Z, dtype=torch.float32, device="cuda")
    G = torch.zeros(32, 32, dtype=torch.float64, device="cuda")
    for net in nets:
        f = lambda z: torch.cat([o.reshape(o.shape[0], -1) for o in net(z)], 1)
        for i in range(0, len(Zt), 1024):
            z = Zt[i:i + 1024].clone().requires_grad_(True)
            out = f(z)
            for k in range(out.shape[1]):
                g, = torch.autograd.grad(out[:, k].sum(), z, retain_graph=True)
                g = g.double()
                G += g.T @ g
    return (G / (len(nets) * len(Zt))).cpu().numpy()


def within(Z, groups):
    C = np.zeros((32, 32))
    n = 0
    for g in np.unique(groups):
        X = Z[groups == g]
        if len(X) > 1:
            X = X - X.mean(0)
            C += X.T @ X
            n += len(X) - 1
    return C / n


def leftover(C, G, eps_rel=1e-3):
    eps = eps_rel * np.trace(G) / 32
    L = np.linalg.cholesky(G + eps * np.eye(32))       # C v = mu B v, as a symmetric problem in L^T v
    Li = np.linalg.inv(L)
    mu, U = np.linalg.eigh(Li @ C @ Li.T)
    V = Li.T @ U
    order = np.argsort(mu)[::-1]
    return mu[order], V[:, order]


def probe(P, Z6, Z9):
    return G69.learned(Z6, Z9, np.asarray(P, np.float64))


def main():
    rng = np.random.default_rng(880)
    D = PR.play(20000, rng)
    Z = PR.encode(D["tiles"])
    nets = ensemble()
    W, Pp, Tp = PR.predict(nets, Z)
    profile = np.array([f"{w}{int(p > 0.5)}{int(t > 0.5)}" for w, p, t in
                        zip(W.mean(0).argmax(1), Pp.mean(0), Tp.mean(0))])
    G = sensitivity(nets, Z)
    C = within(Z, profile)
    mu, V = leftover(C, G)
    Z6 = PR.encode(G69.tiles(G69.SIX))
    Z9 = PR.encode(G69.tiles(G69.SIX + G69.UNSEEN))
    rep = {"note": "Card 088 gate, tools/card088/leftover.py", "tiles": len(Z),
           "profiles": {p: int((profile == p).sum()) for p in np.unique(profile)},
           "leftover eigenvalues (top 10)": [round(float(m), 4) for m in mu[:10]]}
    ck = torch.load(PR.ENCODER)
    rep["card 070's trained P (upper bound)"] = probe(ck["P"].double().numpy(), Z6, Z9)
    gev, gV = np.linalg.eigh(G)
    gtop = gV[:, np.argsort(gev)[::-1]]
    rand = np.random.default_rng(88)
    for r in RANKS:
        rep[f"leftover, rank {r}"] = probe(V[:, :r].T, Z6, Z9)
        rep[f"random directions, rank {r}"] = probe(rand.normal(size=(r, 32)), Z6, Z9)
        rep[f"G's top directions, rank {r}"] = probe(gtop[:, :r].T, Z6, Z9)
    # data check (report only): how much of the leftover the hue explains; keys' and doors' leftovers alike?
    col = np.isin(D["kind"], PR.COLOURED)
    Y = Z[col] @ V[:, :4]
    H = np.c_[D["hue"][col] / 255.0, np.ones(col.sum())]
    for kind in ("key", "locked"):
        m = D["kind"][col] == kind
        beta, *_ = np.linalg.lstsq(H[m], Y[m], rcond=None)
        res = Y[m] - H[m] @ beta
        rep[f"hue's share of the top-4 leftover's variance, {kind}"] = round(float(1 - res.var(0).sum() / Y[m].var(0).sum()), 4)
    ang = {}
    for kind in ("key", "locked"):
        m = D["kind"] == kind
        _, Vk = leftover(within(Z[m], profile[m]), G)
        ang[kind] = Vk[:, :2]
    qa, _ = np.linalg.qr(ang["key"])
    qb, _ = np.linalg.qr(ang["locked"])
    rep["principal angles (degrees), keys' and locked doors' top-2 leftover"] = [
        round(float(np.degrees(np.arccos(np.clip(s, -1, 1)))), 1) for s in np.linalg.svd(qa.T @ qb)[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "P.npz", V=V, mu=mu, G=G, C=C)
    (OUT / "gate.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
