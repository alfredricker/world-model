"""Card 042: card 039's planner with recall in two levels. The same-thing level reads the codes (arm A's
codebooks, card 035's reading with card 036's fresh codes); the similar-thing level is card 039's recall over
the vectors.

  same thing:    k1_ij = [codes of front equal] [codes of held equal] exp(-chamfer_lambda1v(view_i, view_j))
  similar thing: k2_ij = exp(-d_lambda2(i, j)) over the vectors (front, held, view), card 039's
  P(class) = (N + beta s) / (N + beta),  N = sum_j k1 n_j,  s = (sum_j k2 share_j + 1/L) / (sum_j k2 + 1)
  outcome classes: which places changed, and into which codes.

lambda1v = lambda2v + softplus(delta); lambda2, delta and log beta are fitted together by leaving out one
stored key at a time. A code tuple with a "new" piece matches only its own vector. Where the same-thing level
carries a prediction, the result is what the thing became (the stored result of the best supported outcome
class); otherwise it is imagined as in card 038. Draw and undraw (the agent's own place) read the codes the
same way. Moves keep card 038's recall.

  bin/prun python tools/card042/code_recall.py --dev --arm A --seeds 399-399 --layouts 30 --out runs/042_dev.json
  bin/prun python tools/card042/code_recall.py --arm A --seeds 400-404 --layouts-b 100 --out runs/042_armA.json
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
for d in ("card039", "card035", "card036"):
    sys.path.insert(0, str(TOOLS / d))
import numpy as np                                    # noqa: E402
import slot_planner as SP                             # noqa: E402

VP = SP.VP
import novelty as NV                                  # noqa: E402
import fresh as FR                                    # noqa: E402
import torch                                          # noqa: E402

ALPHA = 6.0
CC = {}                                               # the codebooks of the arm in use


# ---------------------------------------------------------------- codes

_vectors_of = VP.vectors_of


def vectors_of(arm, seed, tiles, pairs, groups, dev, log):
    """Card 038's vectors; for arm A also the codes (card 035's reading, card 036's fresh codes)."""
    assert arm == "A", "the same-thing level needs codebooks (arm A)"
    VP.N.ENC = VP.ENC_A
    e = VP.N.encoder(VP.MU, seed, tiles, pairs, groups, dev, log)
    z = np.asarray(e["z"], np.float64)
    ca, vec, rp = FR.fresh_codes(e["z"], e["books"], tiles, ALPHA)
    K = NV.K
    zz = z.reshape(len(z), K, -1)
    used, rad = [], []
    for k in range(K):
        u, r = NV.code_radii(zz[np.asarray(tiles), k], e["books"][k])
        used.append(np.asarray(u))
        rad.append(np.asarray(r))
    CC.clear()
    CC.update({"z": z, "books": e["books"], "ca": ca, "fresh": vec, "used": used, "rad": rad,
               "cut": {k: rp[k]["cut"] for k in rp}, "hcodes": {}, "ids": {}})
    rep = dict(e["rep"])
    rep["codes"] = {"fresh": {int(k): v for k, v in rp.items()}, "tiles_named_apart": FR.named_apart(ca, tiles)}
    return z, [slice(8 * k, 8 * k + 8) for k in range(4)], rep


def code_of_vec(v):
    K = NV.K
    p = np.asarray(v).reshape(K, -1)
    out = []
    for k in range(K):
        books, used, rad = CC["books"][k], CC["used"][k], CC["rad"][k]
        d = np.sqrt(((books[used] - p[k]) ** 2).sum(1))
        j = int(d.argmin())
        if d[j] <= ALPHA * rad[j]:
            out.append(int(used[j]))
            continue
        best, bd = NV.NEW, float("inf")
        for (kk, code), fv in CC["fresh"].items():
            if kk == k:
                dd = float(np.sqrt(((fv - p[k]) ** 2).sum()))
                if dd <= CC["cut"][k] and dd < bd:
                    best, bd = code, dd
        out.append(int(best))
    return tuple(out)


def code_id(S, h):
    """One integer per code tuple; a tuple with a "new" piece stands only for its own vector."""
    h = int(h)
    r = CC["hcodes"].get(h)
    if r is None:
        inv = CC.setdefault("inv", {})
        if not inv:
            for t, hh in enumerate(S.hof.tolist()):
                inv.setdefault(int(hh), t)
        t = inv.get(h)
        tup = tuple(int(x) for x in CC["ca"][t]) if t is not None else code_of_vec(S.arr[h])
        if NV.NEW in tup:
            r = -1 - h
        else:
            r = CC["ids"].setdefault(tup, len(CC["ids"]))
        CC["hcodes"][h] = r
    return r


# ---------------------------------------------------------------- the fit

def isp(x):
    return float(np.log(np.expm1(x)))


def fit_codes(S, Xfh, kpos, ulist, counts, same, group, steps=2000, lr=0.03, sharp=8.0):
    dev = "cuda"
    f32 = dict(dtype=torch.float32, device=dev)
    n, L = counts.shape
    Ct = torch.as_tensor(counts, **f32)
    Rt = Ct / Ct.sum(1, keepdim=True)
    Xt = torch.as_tensor(Xfh, **f32)
    Dfh = torch.abs(Xt[:, None] - Xt[None])
    Sm = torch.as_tensor(same, **f32)
    m = max(len(SP.SETS[s]) for s in ulist)
    ns, Dv = len(ulist), S.arr.shape[1]
    E = np.zeros((ns, m, Dv))
    mk = np.zeros((ns, m), bool)
    for i, s in enumerate(ulist):
        E[i, :len(SP.SETS[s])] = S.arr[SP.SETS[s]]
        mk[i, :len(SP.SETS[s])] = True
    Et, mt = torch.as_tensor(E, **f32), torch.as_tensor(mk, device=dev)
    kp = torch.as_tensor(np.asarray(kpos), device=dev)
    D2 = Xfh.shape[1]

    def setdist(lamv):
        a2b = SP.chamfer_a2b(Et, mt, lamv)
        return (a2b + a2b.T)[kp][:, kp]

    with torch.no_grad():
        dv = setdist(torch.ones(Dv, **f32))
        d1 = Dfh @ torch.ones(D2, **f32) + dv
        med = float(d1[d1 > 0].median())
        pv = dv[dv > 0]
        p1 = float(torch.quantile(pv[:2_000_000], 0.01)) if len(pv) else 1.0
    lam2_0 = 1.0 / med
    lam1_0 = sharp / p1
    g = torch.as_tensor(np.asarray(group), device=dev)
    keep = (g[:, None] != g[None]).float()
    th2 = torch.nn.Parameter(torch.full((D2 + Dv,), isp(lam2_0), **f32))
    dl = torch.nn.Parameter(torch.full((Dv,), isp(max(lam1_0 - lam2_0, 1e-3)), **f32))
    lb = torch.nn.Parameter(torch.tensor(0.0, **f32))

    def predict():
        lam2 = torch.nn.functional.softplus(th2)
        lam1v = lam2[D2:] + torch.nn.functional.softplus(dl)
        k1 = Sm * torch.exp(-setdist(lam1v)) * keep
        k2 = torch.exp(-(Dfh @ lam2[:D2] + setdist(lam2[D2:]))) * keep
        N = k1 @ Ct
        s = (k2 @ Rt + 1.0 / L) / (k2.sum(-1, keepdim=True) + 1.0)
        b = torch.exp(lb)
        return (N + b * s) / (N.sum(-1, keepdim=True) + b)

    rows = lambda P: (Rt * torch.log(P.clamp_min(1e-12))).sum(-1)
    with torch.no_grad():
        l0 = float(rows(predict()).mean())
    opt = torch.optim.Adam([th2, dl, lb], lr=lr)
    for _ in range(steps):
        loss = -rows(predict()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = predict()
        r = rows(P).cpu().numpy()
        lam2 = torch.nn.functional.softplus(th2).double().cpu().numpy()
        lam1v = lam2[D2:] + torch.nn.functional.softplus(dl).double().cpu().numpy()
    return dict(lam2=lam2, lam1v=lam1v, beta=float(np.exp(float(lb.detach()))), l0=l0, rows=r, P=P.cpu().numpy())


# ---------------------------------------------------------------- the kind

class CodeKind(SP.SlotKind):
    def __init__(self, name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
        super().__init__(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)
        self.c1, self.c2 = {}, {}
        self.fid = np.array([code_id(S, k[0]) for k in self.keys], np.int64)
        self.hid = np.array([code_id(S, k[1]) for k in self.keys], np.int64)
        keys = np.asarray(keys, np.int64).reshape(len(w), nkey)
        cats, after = np.asarray(cats, np.int64), np.asarray(after, np.int64)
        self.lab, self.lcat, self.lafter = {}, [], []
        rows = np.array([self.index[tuple(int(v) for v in r)] for r in keys])
        lid = np.array([self._label(int(c), af) for c, af in zip(cats, after)])
        self.lc = np.zeros((len(self.keys), len(self.lab)))
        np.add.at(self.lc, (rows, lid), w)
        self.prior = 1.0 / len(self.lab)
        D = self.D
        if len(self.keys) > 2:
            same = (self.fid[:, None] == self.fid[None]) & (self.hid[:, None] == self.hid[None])
            fit = fit_codes(S, self.X, self.kpos, self.ulist, self.lc, same, list(range(len(self.keys))))
            self.lam2, self.beta = fit["lam2"], fit["beta"]
            self.lam1 = self.lam2.copy()
            self.lam1[2 * D:] = fit["lam1v"]
            R = self.lc / self.lc.sum(1, keepdims=True)
            door = np.array([VP.CO.name_of_code(self._tile(k[0])).startswith("closed door") for k in self.keys])
            M = np.zeros((len(self.lcat), ncat))
            M[np.arange(len(self.lcat)), self.lcat] = 1
            Pc, Rc = fit["P"] @ M, R @ M
            right = (Pc.argmax(1) == Rc.argmax(1)) & (Pc.max(1) >= 0.5)
            ps = lambda lam: round(float(lam.sum()), 2)
            self.report["code_level"] = {
                "loo": round(float(fit["rows"].mean()), 5), "loo_start": round(fit["l0"], 4),
                "door_keys": int(door.sum()), "right_door": round(float(right[door].mean()), 4) if door.any() else None,
                "right_other": round(float(right[~door].mean()), 4), "beta": round(self.beta, 4),
                "outcome_classes": len(self.lab), "code_tuples_front": int(len(set(self.fid.tolist()))),
                "lambda_same_view": ps(self.lam1[2 * D:]),
                "lambda_similar": {"front": ps(self.lam2[:D]), "held": ps(self.lam2[D:2 * D]), "view": ps(self.lam2[2 * D:])}}
        else:
            self.lam1, self.lam2, self.beta = self.lam * 10, self.lam, 1.0
        self.base2 = (self.lc.copy(), dict(self.lab), list(self.lcat), list(self.lafter), self.fid.copy(), self.hid.copy())

    def _tile(self, h):
        inv = CC.setdefault("inv", {})
        if not inv:
            for t, hh in enumerate(self.S.hof.tolist()):
                inv.setdefault(int(hh), t)
        return inv.get(int(h), 0)

    def _label(self, c, af):
        k = (c,) + tuple(code_id(self.S, af[p]) if (c >> p) & 1 else -10 ** 9 for p in range(len(af)))
        i = self.lab.get(k)
        if i is None:
            i = self.lab[k] = len(self.lcat)
            self.lcat.append(c)
            self.lafter.append(tuple(int(af[p]) if (c >> p) & 1 else None for p in range(len(af))))
            if hasattr(self, "lc"):
                self.lc = np.hstack([self.lc, np.zeros((len(self.lc), 1))])
        return i

    def add(self, key, cat, after=None, w=1.0):
        super().add(key, cat, after, w)
        key = tuple(int(h) for h in key)
        i = self.index[key]
        if len(self.fid) < len(self.keys):
            self.fid = np.append(self.fid, code_id(self.S, key[0]))
            self.hid = np.append(self.hid, code_id(self.S, key[1]))
        if len(self.lc) < len(self.keys):
            self.lc = np.vstack([self.lc, np.zeros((len(self.keys) - len(self.lc), self.lc.shape[1]))])
        j = self._label(int(cat), np.asarray(after, np.int64))          # may add a column to lc
        self.lc[i, j] += w

    def reset(self):
        super().reset()
        n = len(self.ulist)
        for c in (getattr(self, "c1", {}), getattr(self, "c2", {})):
            for s in list(c):
                c[s] = c[s][:n]
        if getattr(self, "base2", None) is not None:
            lc, lab, lcat, lafter, fid, hid = self.base2
            self.lc, self.lab, self.lcat, self.lafter = lc.copy(), dict(lab), list(lcat), list(lafter)
            self.fid, self.hid = fid.copy(), hid.copy()

    def _set_dists(self, sid, lam, cache):
        r = cache.get(sid)
        if r is None or len(r) < len(self.ulist):
            have = 0 if r is None else len(r)
            new = SP.chamfer_to(self.S, sid, self.ulist[have:], lam[2 * self.D:])
            r = cache[sid] = new if r is None else np.concatenate([r, new])
        return r

    def weights(self, qs):
        D = self.D
        same = np.stack([(self.fid == code_id(self.S, q[0])) & (self.hid == code_id(self.S, q[1])) for q in qs])
        dv1 = np.stack([self._set_dists(int(q[2]), self.lam1, self.c1)[:len(self.ulist)][self.kpos] for q in qs])
        k1 = same * np.exp(-dv1)
        A = np.stack([self.key_vec(q) for q in qs])
        d2 = VP.wl1(A, self.X, self.lam2[:2 * D])
        d2 += np.stack([self._set_dists(int(q[2]), self.lam2, self.c2)[:len(self.ulist)][self.kpos] for q in qs])
        k2 = np.exp(-d2)
        k2[k2 < VP.KMIN] = 0.0
        share = self.lc / self.lc.sum(1, keepdims=True)
        N = k1 @ self.lc
        z = k2.sum(1, keepdims=True) + 1.0
        s = (k2 @ share + self.prior) / z
        return k1, k2, z, share, N, s

    def predict(self, qs):
        _, _, _, _, N, s = self.weights(qs)
        P = (N + self.beta * s) / (N.sum(1, keepdims=True) + self.beta)
        M = np.zeros((P.shape[1], self.ncat))
        M[np.arange(P.shape[1]), self.lcat[:P.shape[1]]] = 1.0
        return P @ M

    def cat_of(self, qs):
        todo = [q for q in dict.fromkeys(qs) if q not in self.ccache]
        if todo:
            for q, p in zip(todo, self.predict(todo)):
                self.ccache[q] = int(p.argmax()) if p.max() >= 0.5 else None
        return [self.ccache[q] for q in qs]

    def templates(self, u):
        """The distinct (held, view) pairs of the stored tries on the same thing as u (equal codes); when there
        are none, on things like u (card 039's rule, k >= KMIN on the similar-thing level's front part)."""
        r = self.tcache.get(u)
        if r is None:
            idx = np.flatnonzero(self.fid == code_id(self.S, u))
            if not len(idx):
                D = self.D
                kf = np.exp(-(np.abs(self.X[:, :D] - self.S.arr[int(u)]) @ self.lam2[:D]))
                idx = np.flatnonzero(kf >= VP.KMIN)
            r = self.tcache[u] = sorted({self.keys[t][1:] for t in idx})
        return r

    def result(self, q, c):
        """What the things become in category c. Where the same-thing level carries the prediction (its tries
        in c outweigh the similar-thing level's beta), what they became: the stored result of c's most supported
        outcome class. Otherwise imagined as card 038 does, each stored key weighted by its share."""
        r = self.rcache.get((q, c))
        if r is None:
            k1, k2, z, share, N, _ = self.weights([q])
            inc = np.asarray(self.lcat) == c
            if N[0, inc].sum() > self.beta:
                j = int(np.flatnonzero(inc)[np.argmax(N[0, inc])])
                r = self.rcache[(q, c)] = self.lafter[j]
                return r
            wk = k1[0] * self.lc[:, inc].sum(1) + self.beta * k2[0] * share[:, inc].sum(1) / z[0, 0]
            r = self.rcache[(q, c)] = tuple(self.transport(q, c, wk))
        return r


class CodeLook(SP._Kind):
    """Draw and undraw (how the agent's place looks after a step onto or off a thing): what the most tried
    thing with the same codes became, when there is one (the same-thing level; the fitted weight of the
    similar-thing level is near zero wherever codes match, as for pick up, toggle and drop); otherwise card
    038's result, imagined from the neighbours over the vectors."""

    def same(self, q):
        fid = getattr(self, "_fid", None)
        if fid is not None and len(fid) > len(self.keys):          # memory was reset to the stored tries
            fid = self._fid = fid[:len(self.keys)]
        if fid is None or len(fid) < len(self.keys):
            have = 0 if fid is None else len(fid)
            new = np.array([code_id(self.S, k[0]) for k in self.keys[have:]], np.int64)
            fid = self._fid = new if fid is None else np.concatenate([fid, new])
        return fid == code_id(self.S, q[0])

    def look(self, q):
        r = self.rcache.get(q)
        if r is None:
            m = self.same(q) & (self.counts[:, 0] > 0)
            if not m.any():
                return super().look(q)
            t = int(np.flatnonzero(m)[np.argmax(self.counts[m, 0])])
            hs = self.aft.get((t, 0))
            r = self.rcache[q] = hs[0] if hs is not None else self.S.add(self.after_mat(0, 0)[t])
        return r


def kind(name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
    cls = CodeKind if nkey == 3 else (CodeLook if name in ("draw", "undraw") else SP._Kind)
    return cls(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)


def main():
    if "--out" not in sys.argv:
        sys.exit("--out is required (card 038's default path would be overwritten)")
    SP.install()
    VP.Kind = kind
    VP.vectors_of = vectors_of
    VP.main()


if __name__ == "__main__":
    main()
