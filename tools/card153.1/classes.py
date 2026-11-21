"""Card 153.1: dynamics classes inferred from appearance and transitions together.

Terms.  Object i has appearance features a_i (the attractor indices of its two appearance memories: shape and colour
in the sprite world, suffix and onset in the word stream) and observed transitions D_i.  Its dynamics class is a latent
variable z_i in {1..K}; class k is one energy function over the tokens of a transition, with pairwise couplings
J_k = J_0 + Delta_k (card 152).  The appearance prior p(z | a) is a pairwise energy between a and z (card 153's
fit_looks).  Card 153 inferred z under a Chinese-restaurant prior that ignores a; here:

  posterior    p(z_i | a_i, D_i)  is proportional to  p(z_i | a_i) p(D_i | z_i)      in learning as at test
  proposals    a new class is proposed from the residual couplings of one object, or of all the objects that share
               an appearance attractor (their transitions pooled)
  acceptance   a proposal is fitted (couplings, appearance prior and assignments alternating), and kept only if it
               predicts held-out transitions better:
                 S = - sum_i log sum_k p(z_i = k | a_i, D_i^fit) p(D_i^held-out | k)
               with the couplings and the prior fitted on one half of the transitions and scored on the other, both ways

No prior on the number of classes and no threshold: a class that is not there does not predict held-out transitions.
Compute caps: 3 proposals tried a round, 3 alternations a fit.
"""
import sys, numpy as np, torch
sys.path.insert(0, "tools/card153")
import kinds as Kd

dev, M = Kd.dev, Kd.M


def residual(model, r, base, grp, n_grp, l1, steps=300, seed=0):
    """Sparse residual couplings of every group at once.  grp [R]: the group of each transition; base [n_grp]: the class
    whose couplings the group's residual is added to."""
    torch.manual_seed(seed); Dg, Do = model.GO[-1], model.OO[-1]; Mo = M.block_mask(model.OV)
    g = torch.as_tensor(grp, device=dev); kt = torch.as_tensor(base, device=dev)
    f0 = model.fields(r)[torch.arange(len(g), device=dev), kt[g]]; pair0 = model.pair(model.Wo, r)[kt]
    P = [torch.zeros(s, device=dev, requires_grad=True) for s in ((n_grp, Do), (n_grp, Do, Do), (n_grp, Dg, Do))]
    opt = torch.optim.Adam(P, lr=0.05)
    for _ in range(steps):
        Ws = (P[1] + P[1].transpose(1, 2)) / 2 * Mo
        f = f0 + P[0][g] + Ws[g[:, None], r["io"]].sum(1) + P[2][g[:, None], r["ig"]].sum(1)
        loss = model.nll(f, (pair0 + model.pair(Ws, r))[g], r).sum() / r["N"] + l1 * (P[0].abs().sum() + Ws.abs().sum() + P[2].abs().sum())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return P[0].clone(), ((P[1] + P[1].transpose(1, 2)) / 2 * Mo).clone(), P[2].clone()


@torch.no_grad()
def candidate_energy(model, r, base, n_ind, delta, chunk=8):
    """E [objects, candidates]: the energy of each object's transitions under (couplings of class base[c]) + residual c"""
    n = len(base); fk = model.fields(r); out = np.zeros((n_ind, n)); kt = torch.as_tensor(base, device=dev); pair0 = model.pair(model.Wo, r)
    for s in range(0, n, chunk):
        c = torch.arange(s, min(s + chunk, n), device=dev)
        f = fk[:, kt[c]].transpose(0, 1) + delta[0][c][:, None] + torch.einsum("re,ced->crd", r["xo"], delta[1][c]) + torch.einsum("rg,cgd->crd", r["xg"], delta[2][c])
        out[:, s:s + len(c)] = torch.zeros(len(c), n_ind, device=dev).index_add_(1, r["ind"], model.nll(f, pair0[kt[c]] + model.pair(delta[1][c], r), r)).T.cpu().numpy()
    return out


def logsumexp(x):
    m = x.max(1, keepdims=True); return (m + np.log(np.exp(x - m).sum(1, keepdims=True)))[:, 0]


def learn_classes(make, G, O, m, ind, n_ind, codes, V, l1_obj=1e-3, tries=3, rounds=8, log=None):
    """codes: two arrays [n_ind] of appearance attractor indices, V their sizes.  Returns the couplings, each object's
    class, and E [objects, classes] = -log p(D_i | k) in nats."""
    log = [] if log is None else log; S, C = codes
    rows_of = lambda z: np.where(m, z[ind.clip(0)], 0)
    fold = np.random.default_rng(0).random(len(G)) < 0.5; halves = (m & fold[:, None], m & ~fold[:, None])
    probe = make(); r = probe.rows(G, O, m, ind); rh = [probe.rows(G, O, h, ind) for h in halves]
    oh = lambda x, v: torch.nn.functional.one_hot(torch.as_tensor(x), v).float()

    def log_prior(E, K):                                              # log p(z | a), the appearance prior fitted to these energies
        if K == 1:
            return np.zeros((n_ind, 1))
        looks, _ = Kd.fit_looks(S, C, E, [V[0], V[1], K], rounds=1)
        return torch.log_softmax(looks.field(oh(S, V[0]), oh(C, V[1])), -1).numpy().astype(np.float64)

    def fit(z, K):                                                    # couplings, prior and assignments, alternating
        for it in range(4):
            model = make().fit(G, O, m, rows_of(z), K); E = model.energy(r, n_ind)
            if it == 3:
                break
            z2, K2 = Kd.compact((log_prior(E, K) - E).argmax(1))
            if K2 == K and (z2 == z).all():
                break
            z, K = z2, K2
        return model, z, K, E

    def held_out(z, K):                                               # S: both halves, each scored by the other's fit
        s = 0.0
        for a, b in ((0, 1), (1, 0)):
            mf = make().fit(G, O, halves[a], rows_of(z), K); Ea, Eb = mf.energy(rh[a], n_ind), mf.energy(rh[b], n_ind)
            lq = log_prior(Ea, K) - Ea; lq = lq - logsumexp(lq)[:, None]
            s -= float(logsumexp(lq - Eb).sum())
        return s

    model, z, K, E = fit(np.zeros(n_ind, int), 1); score = held_out(z, K)
    for rnd in range(rounds):
        entry = dict(round=rnd, classes=K, held_out=round(score, 1), tried=[]); log.append(entry)
        here = E[np.arange(n_ind), z]; cands = []
        for name, grp, n_grp in (("object", np.arange(n_ind), n_ind), ("first appearance code", S, V[0]), ("second appearance code", C, V[1])):
            base = np.array([np.bincount(z[grp == g_], minlength=K).argmax() if (grp == g_).any() else 0 for g_ in range(n_grp)])
            Ec = candidate_energy(model, r, base, n_ind, residual(model, r, base, grp[r["ind"].cpu().numpy()], n_grp, l1_obj))
            gain = np.maximum(here[:, None] - Ec, 0)
            if name == "object":
                gain[np.arange(n_ind), np.arange(n_ind)] = 0          # an object's own transitions do not count for its proposal
            cands += [(float(gain[:, c].sum()), name, c, gain[:, c] > 0) for c in range(n_grp)]
        cands.sort(key=lambda t: -t[0]); done = False
        for gsum, name, c, join in cands[:tries]:
            if gsum <= 0:
                break
            z0 = z.copy(); z0[join] = K
            model2, z2, K2, E2 = fit(z0, K + 1); s2 = held_out(z2, K2)
            entry["tried"].append(dict(proposed_by=name, index=int(c), joined=int(join.sum()), classes=K2, held_out=round(s2, 1), kept=bool(s2 < score)))
            if s2 < score:
                model, z, K, E, score, done = model2, z2, K2, E2, s2, True
                break
        if not done:
            break
    return model, z, E
