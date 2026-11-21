"""Card 153: kinds of behaviour found by one rule, for any items that leave records (objects, words).

An item is an individual (one object, one word); a record is one step in which it took part, written as tokens:
G tokens belong to the whole record, O tokens to one item, and some O tokens are the item's effect tokens (what it
did there).  A kind is a set of pairwise couplings between those tokens (a shared set plus the kind's own sparse
difference, card 152).  One energy is lowered throughout:

    L = sum over items of E(item, its kind)  -  log P(the grouping)

    E(item, kind)    how badly the kind's couplings predict what the item did, over all its records, in nats: minus
                     the log probability of its effect tokens taken together, given the record's other tokens
    P(the grouping)  the Chinese-restaurant prior of latent-cause models: a kind is likelier the more items it has

The rule (card 151's "store what nothing fits, merge what fits each other", with "fits" measured by L):
  1. every item learns its own sparse difference from the couplings of its kind;
  2. an item's difference is a candidate kind.  The candidate that lowers L most, counting only what it explains of
     OTHER items' records, is stored as a kind; repeated until no candidate lowers L;
  3. the kinds are fitted again on their members, items take the kind that lowers L, two kinds are merged when
     that lowers L, and a kind left with one member is given up (fitted to one item, it is judged only by the
     records it was fitted to); kept only if L came out lower than before; then 1 again, until no candidate is stored.
The only number set by hand is the prior's concentration, alpha = 1.
"""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "tools/card152")
import modules as M

dev = M.dev
ALPHA = 1.0


def onehot(x, V):
    return torch.cat([F.one_hot(torch.as_tensor(x[..., i]), v) for i, v in enumerate(V)], -1).float().to(dev)


class Couplings(M.Dyn):
    """card 152's Dyn for any token sizes.  With the sprite world's sizes, M.Dyn.forward, M.vote and M.propose apply."""
    def __init__(self, GV, OV, effect):
        self.GV, self.OV, self.effect = list(GV), list(OV), list(effect)
        self.GO, self.OO = np.cumsum([0] + self.GV), np.cumsum([0] + self.OV)

    def fit(self, G, O, m, kind, K, l1=1e-3, steps=500, seed=0):
        torch.manual_seed(seed); self.K = K; GV, OV, GO, OO = self.GV, self.OV, self.GO, self.OO
        Dg, Do = GO[-1], OO[-1]; ng, no = len(GV), len(OV)
        mt = torch.as_tensor(m, device=dev); xg, xo = onehot(G, GV), onehot(O, OV) * mt[..., None]
        kh = F.one_hot(torch.as_tensor(kind, device=dev).clamp(min=0), K).float() * mt[..., None]
        P = [torch.zeros(s, device=dev, requires_grad=True) for s in ((Dg,), (Dg, Dg), (K, Do), (K, Do, Do), (K, Dg, Do), (Do,), (Do, Do), (Dg, Do))]
        opt = torch.optim.Adam(P, lr=0.05); Mg, Mo = M.block_mask(GV), M.block_mask(OV)
        Gt, Ot = torch.as_tensor(G, device=dev), torch.as_tensor(O, device=dev); ce = F.cross_entropy

        def parts():
            Wg = (P[1] + P[1].T) / 2 * Mg
            dWo = (P[3] + P[3].transpose(1, 2)) / 2 * Mo; W0 = (P[6] + P[6].T) / 2 * Mo
            return P[0], Wg, P[5] + P[2], W0 + dWo, P[7] + P[4], Wg.abs().sum() + W0.abs().sum() + P[7].abs().sum() + dWo.abs().sum() + P[4].abs().sum()
        for _ in range(steps):
            bg, Wg, bo, Wo, Wgo, reg = parts()
            fg = bg + xg @ Wg + torch.einsum("nok,nod,kgd->ng", kh, xo, Wgo)
            fo = torch.einsum("nok,kd->nod", kh, bo) + torch.einsum("nok,noe,ked->nod", kh, xo, Wo) + torch.einsum("nok,ng,kgd->nod", kh, xg, Wgo)
            loss = sum(ce(fg[:, GO[i]:GO[i + 1]], Gt[:, i]) for i in range(ng))
            loss = loss + sum((ce(fo[..., OO[i]:OO[i + 1]].reshape(-1, OV[i]), Ot[..., i].reshape(-1), reduction="none") * mt.reshape(-1)).sum() / len(G) for i in range(no))
            loss = loss + l1 * reg
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            bg, Wg, bo, Wo, Wgo, _ = parts()
            self.bg, self.Wg, self.bo, self.Wo, self.Wgo = bg.clone(), Wg.clone(), bo.clone(), Wo.clone(), Wgo.clone()
        return self

    # ---- the records of valid items, flattened: R rows
    def rows(self, G, O, m, ind):
        """Also every joint value of the effect tokens (X, one row per combination) and which one each record shows."""
        n, o = np.nonzero(m); OV, OO, eff = self.OV, self.OO, self.effect
        grid = np.stack(np.meshgrid(*[np.arange(OV[t]) for t in eff], indexing="ij"), -1).reshape(-1, len(eff))
        X = torch.zeros(len(grid), OO[-1], device=dev)
        for j, t in enumerate(eff):
            X[torch.arange(len(grid)), torch.as_tensor(OO[t] + grid[:, j])] = 1
        tgt = np.ravel_multi_index([O[n, o][:, t] for t in eff], [OV[t] for t in eff])
        keep = torch.ones(OO[-1], device=dev)
        for t in eff:
            keep[OO[t]:OO[t + 1]] = 0
        xo = onehot(O[n, o], OV) * keep                                                       # the item's tokens that are not effects
        io = xo.nonzero()[:, 1].reshape(len(n), -1) if keep.sum() > 0 else torch.zeros(len(n), 0, dtype=torch.long, device=dev)
        xg = onehot(G[n], self.GV)
        return dict(xg=xg, xo=xo, io=io, ig=xg.nonzero()[:, 1].reshape(len(n), -1), X=X, tgt=torch.as_tensor(tgt, device=dev),
                    ind=torch.as_tensor(ind[n, o], device=dev), N=len(G))

    def nll(self, f, pair, r):
        """The cost in nats of what the item did: minus the log of the probability of its effect tokens taken together,
        given every other token of the record.  f [..., R, Do]: the field from the other tokens; pair [..., C] or
        [..., R, C]: the couplings among the effect tokens, for every combination of their values."""
        s = f @ r["X"].T + (pair if pair.dim() == f.dim() else pair[..., None, :])
        return -torch.log_softmax(s, -1).gather(-1, r["tgt"].expand(s.shape[:-1])[..., None])[..., 0]

    def pair(self, Wo, r):
        return torch.einsum("ce,...ed,cd->...c", r["X"], Wo, r["X"]) / 2

    @torch.no_grad()
    def fields(self, r):
        """[R, K, Do]: the field on each item's tokens under every kind, from the tokens that are not its effects"""
        return self.bo[None] + torch.einsum("re,ked->rkd", r["xo"], self.Wo) + torch.einsum("rg,kgd->rkd", r["xg"], self.Wgo)

    @torch.no_grad()
    def energy(self, r, n_ind):
        """E [individuals, kinds] in nats"""
        e = self.nll(self.fields(r).transpose(0, 1), self.pair(self.Wo, r), r)                  # [K, R]
        return torch.zeros(self.K, n_ind, device=dev).index_add_(1, r["ind"], e).T.cpu().numpy().astype(np.float64)

    def residual(self, r, kind, n_ind, l1, steps=300, seed=0):
        """every individual's own sparse difference from the couplings of its kind, all fitted at once"""
        torch.manual_seed(seed); Dg, Do = self.GO[-1], self.OO[-1]; Mo = M.block_mask(self.OV); i = r["ind"]; kt = torch.as_tensor(kind, device=dev)
        base = self.fields(r)[torch.arange(len(i), device=dev), kt[i]]; pair0 = self.pair(self.Wo, r)[kt]            # [R, Do], [n_ind, C]
        P = [torch.zeros(s, device=dev, requires_grad=True) for s in ((n_ind, Do), (n_ind, Do, Do), (n_ind, Dg, Do))]
        opt = torch.optim.Adam(P, lr=0.05)
        for _ in range(steps):
            Ws = (P[1] + P[1].transpose(1, 2)) / 2 * Mo
            f = base + P[0][i] + Ws[i[:, None], r["io"]].sum(1) + P[2][i[:, None], r["ig"]].sum(1)
            loss = self.nll(f, (pair0 + self.pair(Ws, r))[i], r).sum() / r["N"] + l1 * (P[0].abs().sum() + Ws.abs().sum() + P[2].abs().sum())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return P[0].clone(), ((P[1] + P[1].transpose(1, 2)) / 2 * Mo).clone(), P[2].clone()

    @torch.no_grad()
    def candidate_energy(self, r, kind, n_ind, delta, chunk=8):
        """E [individuals, candidates]: candidate c is the couplings of individual c's kind plus c's own difference"""
        fk = self.fields(r); out = np.zeros((n_ind, n_ind)); kt = torch.as_tensor(kind, device=dev); pair0 = self.pair(self.Wo, r)
        for s in range(0, n_ind, chunk):
            c = torch.arange(s, min(s + chunk, n_ind), device=dev)
            f = fk[:, kt[c]].transpose(0, 1) + delta[0][c][:, None] + torch.einsum("re,ced->crd", r["xo"], delta[1][c]) + torch.einsum("rg,cgd->crd", r["xg"], delta[2][c])
            e = self.nll(f, pair0[kt[c]] + self.pair(delta[1][c], r), r)
            out[:, s:s + len(c)] = torch.zeros(len(c), n_ind, device=dev).index_add_(1, r["ind"], e).T.cpu().numpy()
        return out


# ---------------------------------------------------------------- the grouping's energy
def prior(n):
    n = n[n > 0]
    return len(n) * math.log(ALPHA) + sum(math.lgamma(v) for v in n)


def total(E, a):
    return float(E[np.arange(len(a)), a].sum() - prior(np.bincount(a, minlength=E.shape[1])))


def assign(E, a, alone=True, sweeps=10):
    """Each individual takes the kind that lowers L, the others held (all at once, a few sweeps); then two kinds are
    merged while that lowers L.  alone=False: a kind left with one member is given up, because a kind fitted to one
    individual is judged only by the records it was fitted to.  Returns the grouping and L."""
    a = a.copy(); K = E.shape[1]; best = (total(E, a), a.copy())
    for _ in range(sweeps):
        n = np.bincount(a, minlength=K)[None] - (np.arange(K)[None] == a[:, None])
        c = E - np.log(np.maximum(n, ALPHA)); new = np.where(c[np.arange(len(a)), a] <= c.min(1) + 1e-9, a, c.argmin(1))
        if (new == a).all():
            break
        a = new; L = total(E, a)
        if L < best[0]:
            best = (L, a.copy())
    L, a = best
    while True:
        n = np.bincount(a, minlength=K); move = None
        for x in np.nonzero(n)[0]:
            for y in np.nonzero(n)[0]:
                if x != y:
                    b = np.where(a == x, y, a); Lb = total(E, b)
                    if Lb < L - 1e-9:
                        L, move = Lb, b
        if move is None:
            break
        a = move
    while not alone:
        n = np.bincount(a, minlength=K); one = np.nonzero(n == 1)[0]
        if not len(one) or (n > 0).sum() == 1:
            break
        j = int(np.nonzero(a == one[0])[0][0]); c = np.where(n > 1, E[j] - np.log(np.maximum(n, 1)), np.inf)
        if not np.isfinite(c).any():
            c = np.where((n > 0) & (np.arange(K) != one[0]), E[j], np.inf)
        a[j] = int(c.argmin())
    return a, total(E, a)


def choose(E, Ec, kind, log):
    """store candidates as kinds, the one that lowers L most first, each judged without what it explains of its own seed"""
    a, L = assign(E, kind); cols = E; chosen = []
    while True:
        here = cols[np.arange(len(a)), a]; best = None
        for c in range(Ec.shape[1]):
            if c in chosen or not (Ec[:, c] < here - 1e-6).any():
                continue
            col = Ec[:, c].copy(); col[c] = cols[c].min()
            b, Lb = assign(np.concatenate([cols, col[:, None]], 1), a)
            if Lb < L - 1e-6 and (best is None or Lb < best[0]):
                best = (Lb, c, b, col)
        if best is None:
            return a, cols.shape[1], chosen
        log.append(dict(stored_candidate=int(best[1]), joined=int((best[2] == cols.shape[1]).sum()), energy_before=round(L, 1), energy_after=round(best[0], 1)))
        L, a = best[0], best[2]; chosen.append(best[1]); cols = np.concatenate([cols, best[3][:, None]], 1)


def compact(a):
    used = np.unique(a); return np.searchsorted(used, a), len(used)


def learn_kinds(make, G, O, m, ind, n_ind, l1_obj, rounds=4, refits=4, log=None):
    """make(): a new Couplings.  ind [N, M]: which individual each (record, item) is.  Returns the couplings, each
    individual's kind, and E [individuals, kinds] in nats."""
    log = [] if log is None else log
    rows_of = lambda kind: np.where(m, kind[ind.clip(0)], 0)
    kind = np.zeros(n_ind, int); K = 1
    model = make().fit(G, O, m, rows_of(kind), K); r = model.rows(G, O, m, ind)
    E = model.energy(r, n_ind); L = total(E, kind)
    for rnd in range(rounds):
        entry = dict(round=rnd, kinds=K, energy=round(L, 1), stored=[]); log.append(entry)
        delta = model.residual(r, kind, n_ind, l1_obj)
        a, K2, chosen = choose(E, model.candidate_energy(r, kind, n_ind, delta), kind, entry["stored"])
        if not chosen:
            break
        a, K2 = compact(a)
        for it in range(refits):
            model2 = make().fit(G, O, m, rows_of(a), K2); E2 = model2.energy(r, n_ind)
            b, L2 = assign(E2, a, alone=False)
            if (b == a).all():
                break
            a, K2 = compact(b)
            if it == refits - 1:
                model2 = make().fit(G, O, m, rows_of(a), K2); E2 = model2.energy(r, n_ind); L2 = total(E2, a)
        same = K2 == K and (a == kind).all()                                    # the stored kinds did not survive the fit
        entry.update(kinds_after=K2, energy_after=round(L2, 1), kept=bool(L2 < L and not same))
        if L2 >= L or same:
            break
        model, kind, K, E, L = model2, a, K2, E2, L2
    return model, kind, E


# ---------------------------------------------------------------- looks -> does, taught by every individual in
# proportion to what its records say about its kind
class SoftLooks(M.Looks):
    def fit(self, S, C, P, w, V, l1=1e-2, steps=600, seed=0):
        """S, C [n]: two codes of the looks; P [n, K]: belief over kinds from the records alone; w [n]: its weight"""
        torch.manual_seed(seed); self.V = V; off = np.cumsum([0] + V); self.off = off
        P = torch.as_tensor(P, device=dev).float(); w = torch.as_tensor(w, device=dev).float(); w = w / w.sum()
        St, Ct = torch.as_tensor(S, device=dev), torch.as_tensor(C, device=dev)
        x = torch.cat([F.one_hot(St, V[0]).float(), F.one_hot(Ct, V[1]).float(), P], -1); Mb = M.block_mask(V)
        b = torch.zeros(off[-1], device=dev, requires_grad=True); Wr = torch.zeros(off[-1], off[-1], device=dev, requires_grad=True)
        opt = torch.optim.Adam([b, Wr], lr=0.05)
        for _ in range(steps):
            W = (Wr + Wr.T) / 2 * Mb; f = b + x @ W
            loss = (w * (F.cross_entropy(f[:, off[0]:off[1]], St, reduction="none") + F.cross_entropy(f[:, off[1]:off[2]], Ct, reduction="none")
                         - (P * torch.log_softmax(f[:, off[2]:off[3]], -1)).sum(-1))).sum() + l1 * W.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        b = b.detach(); W = ((Wr + Wr.T) / 2 * Mb).detach()
        # The sparsity penalty decides which couplings exist and shrinks them all; their common strength is then fitted
        # without it (one number), so that the belief over kinds read from the looks is as sure as the records warrant.
        ls = torch.zeros((), device=dev, requires_grad=True); opt = torch.optim.Adam([ls], lr=0.05); fk = (b + x @ W)[:, off[2]:off[3]]
        for _ in range(300):
            loss = -(w * (P * torch.log_softmax(ls.exp() * fk, -1)).sum(-1)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        self.strength = float(ls.detach().exp()); self.b = b * self.strength; self.W = W * self.strength
        return self


def records_belief(E):
    """belief over kinds from an individual's records alone, and how much it says (1 - entropy / log K)"""
    p = np.exp(-(E - E.min(1, keepdims=True))); p = p / p.sum(1, keepdims=True)
    h = -(p * np.log(p + 1e-12)).sum(1) / max(math.log(E.shape[1]), 1e-9)
    return p, 1 - h


def fit_looks(S, C, E, V, rounds=3):
    """Looks-to-kind couplings by expectation-maximisation: first taught by each individual's records alone, weighted
    by what they say; then, a few times, by each individual's belief from its looks and its records together."""
    P, w = records_belief(E); looks = SoftLooks().fit(S, C, P, w, V)
    oh = lambda x, v: F.one_hot(torch.as_tensor(x), v).float()
    for _ in range(rounds):
        P = torch.softmax(looks.field(oh(S, V[0]), oh(C, V[1])) - torch.as_tensor(E - E.min(1, keepdims=True), dtype=torch.float32), -1).numpy()
        looks = SoftLooks().fit(S, C, P, np.ones(len(S)), V)
    return looks, w
