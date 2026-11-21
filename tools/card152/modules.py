"""Card 152: the object module.  One per object; all of them share their memories and are run as one batch.

  where     the place of the object's own frame in the image (its top-left corner)
  what      three memories of stored patterns, each learned by storing an input that nothing stored fits:
              shape    where the object's points lie in its frame, scaled to the frame's extent (8 x 8)
              colour   the mean colour seen at its points
              itself   the full pattern seen in its frame, pixel by pixel: one stored pattern per individual object,
                       stored at first clear sight.  What and where settle together: the energy of (pattern, shift)
                       is how badly the pattern, placed at that shift, explains what is seen; a pixel of the pattern
                       hidden by another object costs nothing, a pixel seen as background costs as much as a wrong one.
  does      a belief over kinds of behaviour.  A kind is a set of pairwise couplings between the tokens of a step
            (card 148's energy, one set per kind); the kind of an object is the set that predicts its records.
            A new kind is opened when several objects are predicted by no kind (as object files were in card 147).
  looks -> does   pairwise couplings between shape code, colour code and kind, learned by pseudolikelihood with L1.
"""
import numpy as np, torch, torch.nn.functional as F

dev = "cuda"
PS, C0, CS = 16, 16, 48                         # pattern size; where the seen box sits on the canvas; canvas size
CU = 0.05                                       # the cost of a pixel the pattern does not explain (about a wrong texture's)
GV, OV = [9, 9, 9, 9, 2], [2, 2, 9, 9, 2]
GO, OO = np.cumsum([0] + GV), np.cumsum([0] + OV)


# ---------------------------------------------------------------- what the module is given: its points in one frame
def views(frame, mask, n):
    img = frame.astype(np.float32) / 255; out = []
    for i in range(n):
        V = mask == 3 + i
        if not V.any():
            out.append(None); continue
        ys, xs = np.nonzero(V); y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        other = (mask != 0) & ~V
        cut = lambda a, v: np.pad(a, ((C0, CS), (C0, CS)), constant_values=v)[y0:y0 + CS, x0:x0 + CS]
        Vp = cut(V.astype(np.float32), 0); Fp = cut((~V & ~other).astype(np.float32), 1)
        Ip = np.stack([cut(img[..., c], 0) for c in range(3)], -1) * Vp[..., None]
        can = np.concatenate([(Ip ** 2).sum(-1, keepdims=True), Ip, Vp[..., None], Fp[..., None]], -1).transpose(2, 0, 1)
        grown = np.zeros_like(V); grown[max(y0 - 1, 0):y1 + 1, max(x0 - 1, 0):x1 + 1] = True
        sil = F.adaptive_avg_pool2d(torch.tensor(V[y0:y1, x0:x1], dtype=torch.float32)[None], 8)[0].reshape(-1)
        out.append(dict(box=(y0, x0, y1, x1), canvas=torch.tensor(can), n=int(V.sum()), clear=not (grown & other).any(),
                        sil=sil, colour=torch.tensor(img[V].mean(0)), crop_m=V[y0:y1, x0:x1], crop=img[y0:y1, x0:x1]))
    return out


# ---------------------------------------------------------------- memories of stored patterns
class Proto:
    """stores an input that nothing fits (energy over theta); keeps a pattern met twice; merges two that come to fit each other"""
    def __init__(self, theta):
        self.theta = theta

    def learn(self, X):
        protos, counts = [], []
        for p in X:
            if protos:
                d = 0.5 * ((torch.stack(protos) - p) ** 2).sum(-1); k = int(d.argmin())
            if not protos or d[k] > self.theta:
                protos.append(p.clone()); counts.append(1)
            else:
                counts[k] += 1; protos[k] += (p - protos[k]) / counts[k]
        keep = []
        for k in sorted(range(len(protos)), key=lambda k: -counts[k]):
            if counts[k] >= 2 and all(0.5 * ((protos[k] - protos[j]) ** 2).sum() > self.theta for j in keep):
                keep.append(k)
        self.P = torch.stack([protos[k] for k in keep])
        self.scale = max(float(self.energy(X).min(1).values.mean()), 1e-4)        # how badly the right pattern usually fits
        return self

    def energy(self, X):
        return 0.5 * ((X[:, None] - self.P[None]) ** 2).sum(-1)

    def belief(self, X):
        return torch.softmax(-self.energy(X) / self.scale, -1)


class Itself:
    """one stored pattern per individual object, in the object's frame"""
    def __init__(self, theta):
        self.theta = theta; self.W = torch.zeros(0, 6, PS, PS, device=dev); self.where = []

    def __len__(self):
        return len(self.W) // 3

    def store(self, v):
        h, w = v["crop_m"].shape
        M = torch.zeros(PS, PS); Tx = torch.zeros(PS, PS, 3)
        M[:h, :w] = torch.tensor(v["crop_m"], dtype=torch.float32); Tx[:h, :w] = torch.tensor(v["crop"]) * M[:h, :w, None]
        k = torch.zeros(3, 6, PS, PS)
        k[0, 0] = M; k[0, 1:4] = -2 * Tx.permute(2, 0, 1); k[0, 4] = (Tx ** 2).sum(-1)       # squared error over explained pixels
        k[1, 4] = M                                                                           # pixels explained
        k[2, 5] = M                                                                           # pattern pixels seen as background
        self.W = torch.cat([self.W, k.to(dev)]); self.where.append(v["box"][:2])
        return len(self) - 1

    def energy(self, vs):
        """[views, patterns, 33, 33]: every stored pattern at every shift of its frame against the seen box"""
        can = torch.stack([v["canvas"] for v in vs]).to(dev); n = torch.tensor([v["n"] for v in vs], device=dev).float()
        r = F.conv2d(can, self.W).reshape(len(vs), len(self), 3, CS - PS + 1, CS - PS + 1)
        return (r[:, :, 0] + CU * ((n[:, None, None, None] - r[:, :, 1]) + r[:, :, 2])) / n[:, None, None, None]

    def settle(self, vs, chunk=40):
        """returns for each view: lowest energy, its pattern, its shift; and the lowest energy with the frame left at
        the seen box (no settling of where), and with each pattern left where it was stored in the image (no object frame)"""
        out = []
        for s in range(0, len(vs), chunk):
            E = self.energy(vs[s:s + chunk]); B, K = E.shape[:2]
            e, idx = E.reshape(B, -1).min(1)
            k = idx // (33 * 33); sh = idx % (33 * 33)
            e0, k0 = E[:, :, C0, C0].min(1)
            box = torch.tensor([v["box"][:2] for v in vs[s:s + chunk]], device=dev); st = torch.tensor(np.array(self.where), device=dev)
            d = st[None] - box[:, None] + C0                                                   # the shift that puts pattern k back where it was stored
            ok = ((d >= 0) & (d <= CS - PS)).all(-1); d = d.clamp(0, CS - PS)
            ea = torch.where(ok, E[torch.arange(B, device=dev)[:, None], torch.arange(K, device=dev)[None], d[..., 0], d[..., 1]], torch.full((B, K), 9.0, device=dev))
            ea, ka = ea.min(1)
            for b in range(B):
                out.append(dict(e=float(e[b]), k=int(k[b]), shift=(int(sh[b]) // 33 - C0, int(sh[b]) % 33 - C0), e0=float(e0[b]), k0=int(k0[b]), ea=float(ea[b]), ka=int(ka[b])))
        return out


# ---------------------------------------------------------------- what an object does: one set of couplings per kind
def onehot(x, V):
    return torch.cat([F.one_hot(torch.as_tensor(x[..., i]), v) for i, v in enumerate(V)], -1).float().to(dev)


def block_mask(V):
    off = np.cumsum([0] + V); M = torch.ones(off[-1], off[-1])
    for i in range(len(V)):
        M[off[i]:off[i + 1], off[i]:off[i + 1]] = 0
    return M.to(dev)


class Dyn:
    def fit(self, G, O, m, kind, K, l1=1e-3, steps=500, seed=0):
        """pseudolikelihood, as card 148; the couplings of the hand and action are shared, an object's are its kind's"""
        torch.manual_seed(seed); self.K = K
        Dg, Do = GO[-1], OO[-1]
        mt = torch.as_tensor(m, device=dev); xg, xo = onehot(G, GV), onehot(O, OV) * mt[..., None]
        kh = F.one_hot(torch.as_tensor(kind, device=dev).clamp(min=0), K).float() * mt[..., None]            # [N,4,K]
        P = [torch.zeros(s, device=dev, requires_grad=True) for s in ((Dg,), (Dg, Dg), (K, Do), (K, Do, Do), (K, Dg, Do), (Do,), (Do, Do), (Dg, Do))]
        opt = torch.optim.Adam(P, lr=0.05); Mg, Mo = block_mask(GV), block_mask(OV)
        Gt, Ot = torch.as_tensor(G, device=dev), torch.as_tensor(O, device=dev)
        ce = F.cross_entropy

        def parts():                                   # a kind's couplings are a shared set plus its own difference, both kept sparse
            Wg = (P[1] + P[1].T) / 2 * Mg
            dWo = (P[3] + P[3].transpose(1, 2)) / 2 * Mo; W0 = (P[6] + P[6].T) / 2 * Mo
            return P[0], Wg, P[5] + P[2], W0 + dWo, P[7] + P[4], Wg.abs().sum() + W0.abs().sum() + P[7].abs().sum() + dWo.abs().sum() + P[4].abs().sum()
        for _ in range(steps):
            bg, Wg, bo, Wo, Wgo, reg = parts()
            fg = bg + xg @ Wg + torch.einsum("nok,nod,kgd->ng", kh, xo, Wgo)
            fo = torch.einsum("nok,kd->nod", kh, bo) + torch.einsum("nok,noe,ked->nod", kh, xo, Wo) + torch.einsum("nok,ng,kgd->nod", kh, xg, Wgo)
            loss = sum(ce(fg[:, GO[i]:GO[i + 1]], Gt[:, i]) for i in range(5))
            loss = loss + sum((ce(fo[..., OO[i]:OO[i + 1]].reshape(-1, OV[i]), Ot[..., i].reshape(-1), reduction="none") * mt.reshape(-1)).sum() / len(G) for i in range(5))
            loss = loss + l1 * reg
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            bg, Wg, bo, Wo, Wgo, _ = parts()
            self.bg, self.Wg, self.bo, self.Wo, self.Wgo = bg.clone(), Wg.clone(), bo.clone(), Wo.clone(), Wgo.clone()
        return self

    @torch.no_grad()
    def forward(self, G, O, m, w, iters=30, seen_hand=False):
        """The state before and the action are clamped; the hand's effect and each object's displacement and
        held-after settle (mean field, temperature falling).  w [N,4,K]: each object's belief over kinds.
        seen_hand: the hand's own effect is clamped too (reading a record afterwards, not predicting).
        Returns each object's settled (o_y, o_x, held after) [N,4,3]."""
        mt = torch.as_tensor(m, device=dev); w = torch.as_tensor(w, device=dev).float() * mt[..., None]
        pg, po = onehot(G, GV), onehot(O, OV) * mt[..., None]
        po[..., OO[2]:] = 0
        for i in range(0 if seen_hand else 2):
            pg[:, GO[i]:GO[i + 1]] = 1.0 / GV[i]
        for i in range(2, 5):
            po[..., OO[i]:OO[i + 1]] = 1.0 / OV[i]
        po = po * mt[..., None]
        bo = torch.einsum("nok,kd->nod", w, self.bo)
        for t in range(iters):
            beta = 0.25 * (8 / 0.25) ** (t / (iters - 1))
            for i in range(0 if seen_hand else 2):
                fg = self.bg + pg @ self.Wg + torch.einsum("nok,nod,kgd->ng", w, po, self.Wgo)
                pg[:, GO[i]:GO[i + 1]] = torch.softmax(beta * fg[:, GO[i]:GO[i + 1]], -1)
            for i in (4, 2, 3):
                fo = bo + torch.einsum("nok,noe,ked->nod", w, po, self.Wo) + torch.einsum("nok,ng,kgd->nod", w, pg, self.Wgo)
                po[..., OO[i]:OO[i + 1]] = torch.softmax(beta * fo[..., OO[i]:OO[i + 1]], -1) * mt[..., None]
        self.hand = torch.stack([pg[:, GO[i]:GO[i + 1]].argmax(-1) for i in range(2)], -1).cpu().numpy()      # the hand's settled effect
        return torch.stack([po[..., OO[i]:OO[i + 1]].argmax(-1) for i in range(2, 5)], -1).cpu().numpy()


def vote(model, G, O, m, w):
    """A belief over kinds is a belief over which energy applies: each kind settles its own answer for the object and
    the answers are weighted by the belief (a mixture), instead of settling once under the averaged couplings.
    The hand's own effect is settled once, with every object at its likeliest kind."""
    K = model.K; N = len(G); score = np.zeros((N, 4, 9 * 9 * 2), np.float32); w = np.asarray(w)
    model.forward(G, O, m, np.eye(K, dtype=np.float32)[w.argmax(-1)]); G = G.copy(); G[:, :2] = model.hand
    for k in range(K):
        one = np.zeros((N, 4, K), np.float32); one[..., k] = 1
        p = model.forward(G, O, m, one, seen_hand=True); code = p[..., 0] * 18 + p[..., 1] * 2 + p[..., 2]
        np.add.at(score, (np.arange(N)[:, None], np.arange(4)[None], code), w[..., k])
    c = score.argmax(-1)
    return np.stack([c // 18, (c // 2) % 9, c % 2], -1)


def propose(model, G, O, m, k, iters=30):
    """the action as the output token (card 148): the state and the wanted effect are clamped, the action settles,
    under kind k's couplings for every object.  Returns (a_y, a_x, grip) as token values."""
    with torch.no_grad():
        mt = torch.as_tensor(m, device=dev); pg, po = onehot(G, GV), onehot(O, OV) * mt[..., None]
        for i in range(2, 5):
            pg[:, GO[i]:GO[i + 1]] = 1.0 / GV[i]
        f0 = torch.einsum("nod,gd->ng", po, model.Wgo[k])
        for t in range(iters):
            beta = 0.25 * (8 / 0.25) ** (t / (iters - 1))
            for i in range(2, 5):
                fg = model.bg + pg @ model.Wg + f0
                pg[:, GO[i]:GO[i + 1]] = torch.softmax(beta * fg[:, GO[i]:GO[i + 1]], -1)
        return torch.stack([pg[:, GO[i]:GO[i + 1]].argmax(-1) for i in range(2, 5)], -1).cpu().numpy()


def informative(O, m):
    return ((O[..., 0] == 1) | (O[..., 1] == 1)) & m


def wrong(model, G, O, m, w, seen_hand=False):
    """per (step, object): was the settled effect not what happened"""
    return (model.forward(G, O, m, w, seen_hand=seen_hand) != O[..., 2:]).any(-1) & m


def learn_kinds(G, O, m, ind, n_ind, kmax=6, n_min=3, rounds=40, log=None):
    """ind [N,4]: which individual object each (step, object) is.  Start with one kind.  Repeat: fit the kinds'
    couplings; give each individual the kind that mispredicts the fewest of its records.  When nobody moves any more:
      a kind with fewer than n_min members is dissolved (it takes several objects to make a kind);
      else, if at least n_min individuals are predicted by no kind (two surprises or more), a kind is opened for the
        one with the most surprises, and the others join it or not in the rounds that follow (as object files were
        opened in card 147);
      else a kind whose members another kind predicts as well is merged into it (as stored patterns that came to fit
        each other were in card 151);
      else stop.
    A kind's couplings are a shared set plus its own sparse difference (Dyn.fit), so a kind opened for one object
    starts from what all objects do.  Opening stops after three kinds in a row had to be dissolved."""
    kind = np.zeros(n_ind, int); K = 1; info = informative(O, m); idx = ind[info]
    n_info = np.bincount(idx, minlength=n_ind); closed = np.zeros(n_ind, bool); failed = 0; moves = np.zeros(n_ind, int)

    def drop(kind, a, to):
        kind = kind.copy(); kind[kind == a] = to[kind == a]; kind[kind > a] -= 1
        return kind
    for r in range(rounds):
        model = Dyn().fit(G, O, m, np.where(m, kind[ind.clip(0)], 0), K)
        mis = np.zeros((n_ind, K))
        for k in range(K):
            w = np.zeros((len(G), 4, K), np.float32); w[..., k] = 1
            mis[:, k] = np.bincount(idx, weights=wrong(model, G, O, m, w, seen_hand=True)[info], minlength=n_ind)
        best = mis.min(1); new = np.where((mis[np.arange(n_ind), kind] > best) & (moves < 4), mis.argmin(1), kind)   # stay unless another kind is better;
        moves += new != kind                                                                                        # an individual that keeps changing sides stays put
        lost = (best >= 2) & (best >= 0.05 * n_info) & ~closed                                             # two surprises that no kind predicts
        entry = dict(round=r, kinds=K, moved=int((new != kind).sum()), unexplained=int(lost.sum()), wrong_under_own_kind=round(float(best.sum() / max(n_info.sum(), 1)), 4))
        if log is not None:
            log.append(entry)
        if r == rounds - 1:
            break
        if (new != kind).any():
            kind = new; continue
        size = np.bincount(kind, minlength=K); small = [a for a in range(K) if size[a] < n_min]
        if small and K > 1:
            a = small[0]; other = np.where(np.arange(K)[None] == a, np.inf, mis)
            to = other.argmin(1); to = to - (to > a)
            closed |= kind == a; kind = drop(kind, a, to + (to >= a)); K -= 1; failed += 1; entry["did"] = "dissolved"; continue
        if lost.sum() >= n_min and K < kmax and failed < 3:
            kind = kind.copy(); kind[np.argmax(np.where(lost, best + 1e-3 * n_info, -1))] = K; K += 1; entry["did"] = "opened"; continue
        gone = None
        for a in sorted(range(K), key=lambda k: size[k]):
            fits = [b for b in range(K) if b != a and mis[kind == a, b].sum() <= mis[kind == a, a].sum()]
            if fits:
                gone = (a, max(fits, key=lambda b: size[b])); break
        if gone is None:
            break
        kind = drop(kind, gone[0], np.full(n_ind, gone[1])); K -= 1; entry["did"] = "merged"
    eps = float(np.clip(best.sum() / max(n_info.sum(), 1), 1e-3, 0.2))
    return model, kind, mis, n_info, eps


# ---------------------------------------------------------------- looks -> does
class Looks:
    def fit(self, X, V, l1=1e-2, steps=600, seed=0):
        """X [n,3]: shape code, colour code, kind of each individual whose records settle its kind.  Pairwise couplings by pseudolikelihood."""
        torch.manual_seed(seed); self.V = V; off = np.cumsum([0] + V); self.off = off
        x = onehot(X, V); Xt = torch.as_tensor(X, device=dev); Mb = block_mask(V)
        b = torch.zeros(off[-1], device=dev, requires_grad=True); Wr = torch.zeros(off[-1], off[-1], device=dev, requires_grad=True)
        opt = torch.optim.Adam([b, Wr], lr=0.05)
        for _ in range(steps):
            W = (Wr + Wr.T) / 2 * Mb; f = b + x @ W
            loss = sum(F.cross_entropy(f[:, off[i]:off[i + 1]], Xt[:, i]) for i in range(3)) + l1 * W.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            self.b = b.clone(); self.W = ((Wr + Wr.T) / 2 * Mb).clone()
        return self

    def field(self, ps, pc):
        """what beliefs over shape code and colour code say about the kind [n,K]"""
        o = self.off; k = slice(o[2], o[3])
        return (self.b[k] + ps.to(dev) @ self.W[o[0]:o[1], k] + pc.to(dev) @ self.W[o[1]:o[2], k]).cpu()

    def share(self):
        o = self.off; k = slice(o[2], o[3])
        s, c = float(self.W[o[0]:o[1], k].abs().sum()), float(self.W[o[1]:o[2], k].abs().sum())
        return dict(shape=round(s / max(s + c, 1e-9), 3), colour=round(c / max(s + c, 1e-9), 3))
