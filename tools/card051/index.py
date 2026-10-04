"""Card 051, step 1: card 050's recall and card 047's situations, indexed by situation.

Card 050's prediction for a pick up, toggle or drop depends on a stored key (front, held, view) only through
  - the front and held tiles (their vectors, for the admitted front, held and relation conditions; their code
    tuples, for the query's own situation), and
  - the admitted view conditions ("a token with this code tuple is in view").
Keys equal on those are one group, with summed outcome counts, and a query is compared with each group once:
the same numbers as card 050, at a cost that grows with the distinct situations, not with memory. Likewise:
  - the query's own situation is a hash lookup (code tuples of front and held, admitted view values);
  - admission (card 049) runs over groups of keys equal on every candidate condition; a key's leave-one-out
    prediction is its group's row minus its own counts (a key's distance to its own group is zero);
  - card 047's situations for the planner (the (held, view) of stored tries on a tile, and of tries that made
    a tile walkable) keep one situation per class (held tile, admitted view values): recall gives every
    member of a class the same prediction.

  bin/prun python tools/card051/index.py --clutter --seed 399 --n 30 --out runs/051/s1_clutter_index_399.json
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card050"))
import own as OWN                                      # noqa: E402

CD = OWN.CD
CR, SP, VP = CD.CR, CD.SP, CD.VP
S7 = CD.S7
SIT = S7                                               # card 047's module
GRID, MAX_ADMIT = CD.GRID, CD.MAX_ADMIT
WAYS_MAX = 1000                                        # keys per outcome class for fitting card 038's ways


def wmedian(vals, wts):
    """np.median of the multiset in which value vals[k] occurs wts[k] times."""
    o = np.argsort(vals, kind="stable")
    v, c = vals[o], np.cumsum(wts[o])
    T = c[-1]
    at = lambda r: v[int(np.searchsorted(c, r, side="right"))]      # the r-th element, 0-based
    return float(at(T // 2)) if T % 2 else 0.5 * (float(at(T // 2 - 1)) + float(at(T // 2)))


class IndexKind(OWN.OwnKind):

    # ------------------------------------------------------------ admission on groups

    def admit(self):
        """Card 049's admission and card 050's alpha, exactly, over groups. While candidate c is weighed, the
        likelihood depends on the admitted conditions and c only, so keys equal on the front and held tiles, the
        admitted view conditions and c (when c reads the view) are one group: a key's leave-one-out prediction
        is its group's row less its own counts."""
        t0 = time.monotonic()
        views = [ci for ci, c in enumerate(self.cand) if c[0] == "view"]
        self.vidx = {ci: j for j, ci in enumerate(views)}
        n = len(self.keys)
        lc = self.lc[:n]
        L = lc.shape[1]
        Fk = self.feats(self.keys)
        fh = np.array([[int(k[0]), int(k[1])] for k in self.keys], np.int64)
        Vb = Fk[3].astype(np.int64)
        dev = "cuda"
        f64 = dict(dtype=torch.float64, device=dev)
        Ct = torch.as_tensor(lc, **f64)
        tot = Ct.sum(1)
        # scales: the median of the positive key-pair distances (card 049), from (front, held) groups; a view
        # candidate's positive distances are all 1
        _, first, inv = np.unique(fh, axis=0, return_index=True, return_inverse=True)
        mfh = np.bincount(inv.ravel()).astype(np.float64)
        Ffh = tuple(x[first] for x in Fk)
        mm = (mfh[:, None] * mfh[None]).ravel()
        scale = []
        for ci, c in enumerate(self.cand):
            if c[0] == "view":
                scale.append(1.0)
                continue
            d = self.cand_dist(ci, Ffh, Ffh).ravel()
            pos = d > 0
            scale.append(wmedian(d[pos], mm[pos]) if pos.any() else 1.0)
        self.scale = np.array(scale)

        def part(adm, extra=None):
            cols = [self.vidx[ci] for ci in adm if ci in self.vidx]
            if extra is not None and extra in self.vidx:
                cols.append(self.vidx[extra])
            M = np.concatenate([fh, Vb[:, cols]], 1)
            _, fi, gi = np.unique(M, axis=0, return_index=True, return_inverse=True)
            gt = torch.as_tensor(gi.ravel(), device=dev)
            Cg = torch.zeros((len(fi), L), **f64).index_add_(0, gt, Ct)
            return tuple(x[fi] for x in Fk), gt, Cg, Cg.sum(1)

        def Dist(ci, Fg):
            return torch.as_tensor(self.cand_dist(ci, Fg, Fg) / self.scale[ci], **f64)

        def P_of(dist, a, gt, Cg, Tg):
            w = torch.exp(-dist)                                   # group to group; a group's own entry is 1
            return ((w @ Cg)[gt] - Ct + a / L) / ((w @ Tg)[gt] - tot + a)[:, None]

        def ll(dist, a, pg):
            return float((Ct * torch.log(P_of(dist, a, *pg[1:]).clamp_min(1e-300))).sum())

        adm = [ci for ci, c in enumerate(self.cand) if c[0] == "front"]
        pg = part(adm)
        base = sum(Dist(ci, pg[0]) for ci in adm)
        best = max(GRID, key=lambda g: ll(g * base, 1.0, pg))
        lam = {ci: float(best) for ci in adm}
        cur = ll(best * base, 1.0, pg)
        start = cur
        cost = math.log(len(self.cand))
        trace, sizes = [], [len(pg[2])]
        while len(adm) < 4 + MAX_ADMIT:
            top = None
            p0 = part(adm)
            d0 = sum(lam[ci] * Dist(ci, p0[0]) for ci in adm)
            for ci in range(len(self.cand)):
                if ci in adm:
                    continue
                if ci in self.vidx:
                    p = part(adm, ci)
                    d = sum(lam[c] * Dist(c, p[0]) for c in adm)
                else:
                    p, d = p0, d0
                Dc = Dist(ci, p[0])
                for g in GRID:
                    v = ll(d + g * Dc, 1.0, p)
                    if top is None or v > top[0]:
                        top = (v, ci, float(g))
            if top is None or top[0] - cur <= cost:
                break
            v, ci, g = top
            trace.append({"cond": self.cname(ci), "gain": round(v - cur, 2)})
            adm.append(ci)
            lam[ci] = g
            cur = v
            sizes.append(len(part(adm)[2]))
        pg = part(adm)
        th = torch.nn.Parameter(torch.tensor([math.log(lam[ci]) for ci in adm], **f64))
        la = torch.nn.Parameter(torch.tensor(0.0, **f64))
        Ds = torch.stack([Dist(ci, pg[0]) for ci in adm])
        opt = torch.optim.Adam([th, la], lr=0.05)
        for _ in range(300):
            P = P_of((torch.exp(th)[:, None, None] * Ds).sum(0), torch.exp(la), *pg[1:])
            loss = -(Ct * torch.log(P.clamp_min(1e-300))).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.adm = adm
        self.lamc = torch.exp(th).detach().cpu().numpy() / self.scale[adm]
        self.a = float(torch.exp(la).detach())                    # card 049's prior strength among neighbours
        self.base3 = (self.lc.copy(),)
        self.report["conditions"] = {
            "admitted": [self.cname(ci) for ci in adm], "trace": trace, "candidates": len(self.cand),
            "cost_nats": round(cost, 2), "ll_front_only": round(start, 2), "ll_greedy": round(cur, 2),
            "ll_refit": round(-float(loss.detach()), 2), "a": round(self.a, 4),
            "keys": n, "groups_while_admitting": sizes, "seconds": round(time.monotonic() - t0, 1)}
        print(self.name, "admitted", self.report["conditions"]["admitted"], "trace", trace,
              "keys", n, "groups", sizes, flush=True)
        # card 050's alpha, over the same groups
        with torch.no_grad():
            Pn = P_of((torch.exp(th)[:, None, None] * Ds).sum(0), self.a, *pg[1:]).cpu().numpy()
        cols = self.view_cols()
        osig = [(int(self.fid[i]), int(self.hid[i]), tuple(Fk[3][i, cols].tolist())) for i in range(n)]
        oi = {}
        oof = np.array([oi.setdefault(s, len(oi)) for s in osig])
        Og = np.zeros((len(oi), L))
        np.add.at(Og, oof, lc)
        Gk = Og[oof]
        Gt = Gk.sum(1, keepdims=True)
        msk = lc > 0

        def lla(alpha):
            P = (Gk - 1.0 + alpha * Pn) / (Gt - 1.0 + alpha)
            return float((lc[msk] * np.log(np.maximum(P[msk], 1e-300))).sum())

        vals = [lla(al) for al in OWN.ALPHAS]
        j = int(np.argmax(vals))
        self.beta = float(OWN.ALPHAS[j])
        self.report["conditions"]["own"] = {"alpha": round(self.beta, 6), "ll": round(vals[j], 2),
                                            "ll_neighbours_only": round(lla(1e9), 2), "situations": len(oi)}
        print(self.name, "own tries first: alpha", self.report["conditions"]["own"],
              "admission seconds", self.report["conditions"]["seconds"], flush=True)
        self.build_index()

    # ------------------------------------------------------------ the index

    def vbits(self, sid):
        return tuple(self.view_feats(int(sid))[self.vcols].tolist())

    def build_index(self):
        self.vcols = self.view_cols()
        self.ix = {"g": {}, "rep": [], "Gc": np.zeros((0, self.lc.shape[1])), "gof": [],
                   "o": {}, "oof": [], "cls": {}, "fcode": {}, "Gaw": np.zeros((0, self.ncat)),
                   "Gsum": np.zeros((0, self.ncat, self.nres, self.D)), "Gaft": {}, "feats": None}
        for i, k in enumerate(self.keys):
            self._index_key(i, k)
        lc = self.lc[:len(self.keys)]
        gof = np.array(self.ix["gof"])
        np.add.at(self.ix["Gc"], gof, lc)
        np.add.at(self.ix["Gaw"], gof, self.aw[:len(self.keys)])
        np.add.at(self.ix["Gsum"], gof, self.sums[:len(self.keys)])
        for (t, c), hs in self.aft.items():
            self._agree(int(gof[t]), c, hs)
        oof = np.array(self.ix["oof"])
        Oc = np.zeros((len(self.ix["o"]), lc.shape[1]))
        np.add.at(Oc, oof, lc)
        for o, r in self.ix["o"].items():
            self.ix["o"][o] = Oc[r[1]].copy(), r[1]
        self.ix["feats"] = None
        self.base_ix = self._copy_ix()
        # card 038's ways (keep, copy, shift or set, per part), fitted once on the stored memory like recall's
        # weights, not after every try; refitted with admission
        self.ways_keep = {}
        for c in range(self.ncat):
            if (self.aw[:, c] > 0).any():
                self.ways(c)

    def forget(self):
        super().forget()
        if getattr(self, "ways_keep", None) is not None:
            self.ways_cache = self.ways_keep

    def ways(self, c):
        keep = getattr(self, "ways_keep", None)
        if keep is None:
            return super().ways(c)
        r = keep.get(c)
        if r is not None:
            return r
        idx = np.flatnonzero(self.aw[:, c] > 0)
        if len(idx) > WAYS_MAX:
            idx = np.sort(np.random.default_rng(c).choice(idx, WAYS_MAX, replace=False))
        grp = np.asarray(self.group)[idx]
        r = {}
        for p in self.changed(c):
            if len(set(grp.tolist())) < 2:
                r[p] = [VP.SHIFT] * len(self.parts)
                continue
            A = self.after_mat(c, p)[idx]
            B, O = self.before_rows(idx, p), self.other_rows(idx, p, c)
            share = self.counts[idx, c] / self.counts[idx].sum(1)
            logw = -self.key_dists(idx) + np.log(share)[None]
            logw[grp[:, None] == grp[None]] = -np.inf
            wgt = np.exp(logw - logw.max(1, keepdims=True))
            wgt /= wgt.sum(1, keepdims=True)
            choice = []
            for sl in self.parts:
                a, b, o = A[:, sl], B[:, sl], O[:, sl]
                err = [np.abs(b - a).sum(), np.abs(o - a).sum(),
                       np.abs(b + wgt @ (a - b) - a).sum(), np.abs(wgt @ a - a).sum()]
                choice.append(int(np.argmin(err)))
            r[p] = choice
        keep[c] = r
        return r

    def _agree(self, g, c, hs):
        A = self.ix["Gaft"]
        if (g, c) not in A:
            A[(g, c)] = hs
        elif A[(g, c)] != hs:
            A[(g, c)] = None

    def _index_key(self, i, k):
        """Place stored key i in its group, its own situation and its class (no counts)."""
        ix = self.ix
        vb = self.vbits(k[2])
        gk = (int(k[0]), int(k[1]), vb)
        g = ix["g"].get(gk)
        if g is None:
            g = ix["g"][gk] = len(ix["rep"])
            ix["rep"].append(i)
            ix["Gc"] = np.vstack([ix["Gc"], np.zeros((1, ix["Gc"].shape[1]))])
            ix["Gaw"] = np.vstack([ix["Gaw"], np.zeros((1, self.ncat))])
            ix["Gsum"] = np.concatenate([ix["Gsum"], np.zeros((1, self.ncat, self.nres, self.D))])
            ix["feats"] = None
        ix["gof"].append(g)
        ok = (CR.code_id(self.S, k[0]), CR.code_id(self.S, k[1]), vb)
        o = ix["o"].get(ok)
        if o is None:
            o = ix["o"][ok] = (np.zeros(self.lc.shape[1]), len(ix["o"]))
        ix["oof"].append(o[1])
        f = int(k[0])
        ix["fcode"].setdefault(f, CR.code_id(self.S, f))
        cls = ix["cls"].setdefault(f, {})
        ck = (int(k[1]), vb)
        sit = (int(k[1]), int(k[2]))
        if ck not in cls or sit < cls[ck]:
            cls[ck] = sit
        return g, ok

    def _copy_ix(self):
        ix = self.ix
        return {"g": dict(ix["g"]), "rep": list(ix["rep"]), "Gc": ix["Gc"].copy(), "gof": list(ix["gof"]),
                "o": {k: (v[0].copy(), v[1]) for k, v in ix["o"].items()}, "oof": list(ix["oof"]),
                "cls": {f: dict(c) for f, c in ix["cls"].items()}, "fcode": dict(ix["fcode"]),
                "Gaw": ix["Gaw"].copy(), "Gsum": ix["Gsum"].copy(), "Gaft": dict(ix["Gaft"]), "feats": None}

    def snapshot(self):
        """Make the present memory the one reset() returns to (memory grown for a timing test)."""
        self.base = (list(self.keys), dict(self.gid), list(self.group), self.X.copy(), self.counts.copy(),
                     dict(self.aft), (self.sums.copy(), self.aw.copy()), self.kpos.copy(), list(self.ulist))
        self.base2 = (self.lc.copy(), dict(self.lab), list(self.lcat), list(self.lafter), self.fid.copy(),
                      self.hid.copy())
        self.base_ix = self._copy_ix()

    def reset(self):
        super().reset()
        if getattr(self, "base_ix", None) is not None:
            b = self.base_ix
            self.ix = {"g": dict(b["g"]), "rep": list(b["rep"]), "Gc": b["Gc"].copy(), "gof": list(b["gof"]),
                       "o": {k: (v[0].copy(), v[1]) for k, v in b["o"].items()}, "oof": list(b["oof"]),
                       "cls": {f: dict(c) for f, c in b["cls"].items()}, "fcode": dict(b["fcode"]),
                       "Gaw": b["Gaw"].copy(), "Gsum": b["Gsum"].copy(), "Gaft": dict(b["Gaft"]), "feats": None}

    def add(self, key, cat, after=None, w=1.0):
        key = tuple(int(h) for h in key)
        if getattr(self, "ix", None) is None:
            return super().add(key, cat, after, w)
        i = self.index.get(key)
        old = None if i is None else self.lc[i].copy()
        oaw = None if i is None else self.aw[i].copy()
        osum = None if i is None else self.sums[i].copy()
        super().add(key, cat, after, w)
        i = self.index[key]
        ix = self.ix
        if old is None:
            g, ok = self._index_key(i, key)
        else:
            g = ix["gof"][i]
            ok = (CR.code_id(self.S, key[0]), CR.code_id(self.S, key[1]), self.vbits(key[2]))
        L = self.lc.shape[1]
        if ix["Gc"].shape[1] < L:
            ix["Gc"] = np.hstack([ix["Gc"], np.zeros((len(ix["Gc"]), L - ix["Gc"].shape[1]))])
        new = self.lc[i]
        delta = new - (np.pad(old, (0, L - len(old))) if old is not None else 0.0)
        ix["Gc"][g] += delta
        vec, o = ix["o"][ok]
        if len(vec) < L:
            vec = np.pad(vec, (0, L - len(vec)))
        ix["o"][ok] = (vec + delta, o)
        ix["Gaw"][g] += self.aw[i] - (oaw if oaw is not None else 0.0)
        ix["Gsum"][g] += self.sums[i] - (osum if osum is not None else 0.0)
        if after is not None:
            self._agree(g, int(cat), tuple(int(h) for h in after))

    # ------------------------------------------------------------ prediction over groups

    def group_feats(self):
        ix = self.ix
        if ix["feats"] is None or len(ix["feats"][0]) < len(ix["rep"]):
            ix["feats"] = self.feats([self.keys[i] for i in ix["rep"]])
        return ix["feats"]

    def group_weights(self, qs):
        Fg, Fq = self.group_feats(), self.feats(qs)
        d = np.zeros((len(qs), len(Fg[0])))
        for ci, l in zip(self.adm, self.lamc):
            d += l * self.cand_dist(ci, Fq, Fg)
        return np.exp(-d)

    def own_counts(self, q):
        L = self.lc.shape[1]
        r = self.ix["o"].get((CR.code_id(self.S, q[0]), CR.code_id(self.S, q[1]), self.vbits(q[2])))
        if r is None:
            return np.zeros(L)
        v = r[0]
        return np.pad(v, (0, L - len(v))) if len(v) < L else v

    def neighbours(self, qs):
        wG = self.group_weights(qs)
        Gc = self.ix["Gc"]
        L = self.lc.shape[1]
        if Gc.shape[1] < L:
            Gc = np.hstack([Gc, np.zeros((len(Gc), L - Gc.shape[1]))])
        Nn = wG @ Gc
        return wG, (Nn + self.a / L) / (Nn.sum(1, keepdims=True) + self.a)

    def predict(self, qs):
        _, s = self.neighbours(qs)
        N = np.stack([self.own_counts(q) for q in qs])
        P = (N + self.beta * s) / (N.sum(1, keepdims=True) + self.beta)
        M = np.zeros((P.shape[1], self.ncat))
        M[np.arange(P.shape[1]), self.lcat[:P.shape[1]]] = 1.0
        return P @ M

    def weights(self, qs):
        """Card 050's per-key weights, expanded from the groups (for callers that need them per key)."""
        wG, s = self.neighbours(qs)
        n = len(self.keys)
        gof, oof = np.array(self.ix["gof"][:n]), np.array(self.ix["oof"][:n])
        w = wG[:, gof]
        own = np.stack([oof == self.ix["o"].get((CR.code_id(self.S, q[0]), CR.code_id(self.S, q[1]),
                                                 self.vbits(q[2])), (None, -1))[1] for q in qs]).astype(np.float64)
        lc = self.lc[:n]
        share = lc / lc.sum(1, keepdims=True)
        N = np.stack([self.own_counts(q) for q in qs])
        z = w.sum(1, keepdims=True) + 1.0
        return own, w, z, share, N, s

    def result(self, q, c):
        r = self.rcache.get((q, c))
        if r is None:
            N = self.own_counts(q)
            inc = np.asarray(self.lcat) == c
            if N[inc].sum() > self.beta:
                j = int(np.flatnonzero(inc)[np.argmax(N[inc])])
                r = self.rcache[(q, c)] = self.lafter[j]
                return r
            own, w, z, share, _, _ = self.weights([q])
            wk = own[0] * self.lc[:, inc].sum(1) + self.beta * w[0] * share[:, inc].sum(1) / z[0, 0]
            r = self.rcache[(q, c)] = tuple(self.transport(q, c, wk))
        return r

    # ------------------------------------------------------------ situations by class (card 047's readers)

    def templates(self, u):
        r = self.tcache.get(u)
        if r is None:
            ix = self.ix
            fs = list(ix["cls"])
            A = self.S.arr
            cu = CR.code_id(self.S, u)
            kf = np.exp(-(np.abs(A[fs] - A[int(u)]) @ self.lam2[:self.D])) if fs else np.zeros(0)
            best = {}
            for f, k in zip(fs, kf):
                if ix["fcode"][f] == cu or k >= VP.KMIN:
                    for ck, sit in ix["cls"][f].items():
                        if ck not in best or sit < best[ck]:
                            best[ck] = sit
            r = self.tcache[u] = sorted(best.values())
        return r

    def opened(self, free):
        """Card 047's opened_in for this action, over groups: the classes of stored situations in which the
        action made a tile walkable, one situation each."""
        ix = self.ix
        best = {}
        for c in (1, 3):
            for g in np.flatnonzero(ix["Gaw"][:, c] > 0):
                i = ix["rep"][g]
                f = self.keys[i][0]
                if free[f]:
                    continue
                hs = ix["Gaft"].get((int(g), c))
                h0 = hs[0] if hs is not None else self.S.add(ix["Gsum"][g, c, 0] / ix["Gaw"][g, c])
                if not free[int(h0)]:
                    continue
                ck = (int(self.keys[i][1]), self.vbits(self.keys[i][2]))
                sit = (int(self.keys[i][1]), int(self.keys[i][2]))
                if ck not in best or sit < best[ck]:
                    best[ck] = sit
        return sorted(best.values())


_opened_in = SIT.opened_in


def opened_in(self, a):
    kd = self.kinds[a]
    if not isinstance(kd, IndexKind):
        return _opened_in(self, a)
    k = ("situations", a)
    r = self.openedc.get(k)
    if r is None:
        r = self.openedc[k] = kd.opened(self.M.free)
    return r


_kind = CR.kind


def index_kind(name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
    if nkey == 3:
        return IndexKind(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)
    return _kind(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)


def install():
    CR.kind = index_kind
    SIT.opened_in = opened_in


def main():
    install()
    sys.argv[1:1] = ["--recall", "own"]                # card 049's main leaves CR.kind alone for "own"
    CD.main()


if __name__ == "__main__":
    main()
