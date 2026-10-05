"""Card 039: card 038's planner with what is in view as a set of things instead of one merged vector.

Card 038 (tools/card038/vector_planner.py, imported unchanged) merges the things in view into one vector, the
largest value per dimension, and recall compares those vectors. Here the view is the set of things itself, and
recall compares two views by chamfer matching under its weights: each thing is matched with the most similar thing
in the other view, both ways, and the weighted L1 mismatches are summed. The weights (lambda, one per dimension of
the front, held and view parts) are fitted as before, by leaving out one (front, held) pair at a time, now through
the set distance. Nothing names a dimension, part or attribute to compare. Moves, draw and undraw keep card 038's
recall (their keys have no view part). Everything else is card 038's.

  python tools/card039/slot_planner.py --arm B --seeds 400-404 --layouts-b 100 --out runs/039_armB.json
  python tools/card039/slot_planner.py --dev --arm B --seeds 399-399 --layouts 100 --out runs/039_dev.json
  python tools/card039/slot_planner.py --check --arm B --seeds 400-404 --out runs/039_check.json   (criterion 3)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card038"))
import vector_planner as VP                                    # noqa: E402

SETS = []          # set id -> the things' handles (sorted int64)
SETID = {}         # frozenset of handles -> set id


def set_of(hs):
    key = frozenset(int(h) for h in hs)
    i = SETID.get(key)
    if i is None:
        i = SETID[key] = len(SETS)
        SETS.append(np.array(sorted(key), np.int64))
    return i


def chamfer_to(S, q, sids, lamv):
    """Chamfer distances from set q to each set in sids, under the view weights lamv."""
    if len(sids) == 0:
        return np.zeros(0)
    sizes = np.array([len(SETS[s]) for s in sids])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    Wd = VP.wl1(S.arr[SETS[q]], S.arr[np.concatenate([SETS[s] for s in sids])], lamv)    # (|q|, all things)
    return np.minimum.reduceat(Wd, starts, axis=1).sum(0) + np.add.reduceat(Wd.min(0), starts)


_WL1 = {}


def _wl1():
    """Weighted L1 between every row of A and every row of B as one fused kernel (torch.compile): the (|A|, |B|, D)
    differences are never made (card 068)."""
    if not _WL1:
        import torch

        def wl1(A, B, lam):
            return ((A[:, None, :] - B[None, :, :]).abs() * lam).sum(-1)

        _WL1["f"] = torch.compile(wl1, dynamic=True)
    return _WL1["f"]


def chamfer_a2b(Et, mt, lamv, budget=2 ** 28):
    """For every pair of sets (A, B): the sum over A's things of the weighted L1 to B's nearest thing, (ns, ns),
    for Et (ns, m, Dv) with mask mt (ns, m). The same numbers and gradient as
      Wd = (|Et[:, None, :, None] - Et[None, :, None]| @ lamv).masked_fill(~mt[None, :, None, :], inf)
      Wd.min(-1).values.masked_fill(~mt[:, None, :], 0).sum(-1)
    (card 068), up to ties for the nearest thing: the minimum passes its gradient to the nearest thing only, so the
    backward pass needs the differences to the nearest things alone, made a slice of sets at a time."""
    import torch
    global _ChamferA2B
    if "_ChamferA2B" not in globals():
        class _ChamferA2B(torch.autograd.Function):
            @staticmethod
            def forward(ctx, E, M, lam):
                ns, m, Dv = E.shape
                X = E.reshape(ns * m, Dv)
                C = _wl1()(X, X, lam).reshape(ns, m, ns, m).permute(0, 2, 1, 3)        # (A, B, a, b)
                C = C.masked_fill(~M[None, :, None, :], float("inf"))
                v, j = C.min(-1)                                                          # (A, B, a)
                del C
                v = v.masked_fill(~M[:, None, :], 0.0)
                ctx.save_for_backward(E, M, j)
                ctx.r = max(1, budget // max(1, ns * m * Dv))
                return v.sum(-1)

            @staticmethod
            def backward(ctx, g):
                E, M, j = ctx.saved_tensors
                ns, m, Dv = E.shape
                gl = torch.zeros(Dv, dtype=E.dtype, device=E.device)
                B = torch.arange(ns, device=E.device)[None, :, None]
                for i in range(0, ns, ctx.r):
                    near = E[B, j[i:i + ctx.r]]                                           # (r, B, a, Dv)
                    d = (E[i:i + ctx.r, None, :, :] - near).abs_()
                    w = g[i:i + ctx.r, :, None] * M[i:i + ctx.r, None, :]
                    gl += w.reshape(-1) @ d.reshape(-1, Dv)
                return None, None, gl
    return _ChamferA2B.apply(Et, mt, lamv)


def fit_lambda_sets(S, Xfh, kpos, ulist, Rk, group, steps=1500, lr=0.02):
    """Card 037's fit (fit_lambda in card 038) with the view part's distance the chamfer matching between the
    keys' sets. Every lambda equal at the start, with the median distance between keys 1."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f32 = dict(dtype=torch.float32, device=dev)
    Xt = torch.as_tensor(Xfh, **f32)
    Rt = torch.as_tensor(Rk, **f32)
    n, L = Rt.shape
    Dfh = torch.abs(Xt[:, None] - Xt[None])                                  # (n, n, 2D)
    m = max(len(SETS[s]) for s in ulist)
    ns, Dv = len(ulist), S.arr.shape[1]
    E = np.zeros((ns, m, Dv))
    mk = np.zeros((ns, m), bool)
    for i, s in enumerate(ulist):
        E[i, :len(SETS[s])] = S.arr[SETS[s]]
        mk[i, :len(SETS[s])] = True
    Et, mt = torch.as_tensor(E, **f32), torch.as_tensor(mk, device=dev)
    kp = torch.as_tensor(np.asarray(kpos), device=dev)

    def setdist(lamv):
        a2b = chamfer_a2b(Et, mt, lamv)                                         # (ns, ns): A's things to B
        return a2b + a2b.T

    def dist(lam):
        D2 = Xfh.shape[1]
        return Dfh @ lam[:D2] + setdist(lam[D2:])[kp][:, kp]

    with torch.no_grad():
        d1 = dist(torch.ones(Xfh.shape[1] + Dv, **f32))
        med = float(d1[d1 > 0].median()) if bool((d1 > 0).any()) else 1.0
    g = torch.as_tensor(np.asarray(group), device=dev)
    keep = (g[:, None] != g[None]).float()

    def ll(theta):
        lam = torch.nn.functional.softplus(theta)
        k = torch.exp(-dist(lam)) * keep
        P = (k @ Rt + 1.0 / L) / (k.sum(-1, keepdim=True) + 1.0)
        return (Rt * torch.log(P)).sum(-1).mean()

    lam0 = 1.0 / max(med, 1e-6)
    th = torch.nn.Parameter(torch.full((Xfh.shape[1] + Dv,), float(np.log(np.expm1(lam0))), **f32))
    with torch.no_grad():
        l0 = float(ll(th))
    if n > 2 and bool(keep.any()):
        opt = torch.optim.Adam([th], lr=lr)
        for _ in range(steps):
            loss = -ll(th)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        l1 = float(ll(th))
        lam = torch.nn.functional.softplus(th).double().cpu().numpy()
    return lam, l0, l1


def fit_alpha_dist(Dist, counts):
    """card 038's fit_alpha, given the distances between keys."""
    k = np.exp(-Dist)
    k[k < VP.KMIN] = 0.0
    np.fill_diagonal(k, 0.0)
    share = counts / counts.sum(1, keepdims=True)
    wn = k @ share
    q = (wn + VP.PRIOR) / (wn.sum(1, keepdims=True) + 1.0)
    n = counts.sum(1, keepdims=True)
    msk = counts > 0
    rest = np.maximum(counts - 1.0, 0.0)

    def ll(alpha):
        a = np.broadcast_to(alpha, n.shape)
        P = (rest + a * q) / np.maximum(n - 1.0 + a, 1e-12)
        return float((counts[msk] * np.log(P[msk])).sum() / counts.sum())

    grid = np.exp(np.linspace(-9.0, 9.0, 181))
    vals = [ll(a) for a in grid]
    i = int(np.argmax(vals))
    return float(grid[i]), vals[i], ll(wn.sum(1, keepdims=True) + 1.0)


_Kind = VP.Kind


class SlotKind(_Kind):
    """Card 038's Kind for pick up, toggle and drop, with the view part a set of things. X holds the front and
    held vectors; kpos maps each key to its set's place in ulist (the distinct sets among the keys)."""

    def __init__(self, name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
        assert nkey == 3 and after is not None and lam is None
        self.name, self.S, self.parts, self.nkey, self.ncat = name, S, parts, nkey, ncat
        self.D = S.arr.shape[1]
        self.pair, self.moves = True, False
        self.nres = after.shape[1]
        keys = np.asarray(keys, np.int64).reshape(len(w), nkey)
        cats = np.asarray(cats, np.int64)
        uk, kinv = np.unique(keys, axis=0, return_inverse=True)
        kinv = kinv.ravel()
        self.keys = [tuple(int(v) for v in r) for r in uk]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.gid = {}
        self.group = [self.gid.setdefault(k[:2], len(self.gid)) for k in self.keys]
        if HOLDOUT == "key":                    # leave out one stored key at a time, not every key of its pair
            self.group = list(range(len(self.keys)))
        nk = len(self.keys)
        self.counts = np.zeros((nk, ncat))
        np.add.at(self.counts, (kinv, cats), w)
        self.rep, self.aft = {}, {}
        self.sums = np.zeros((nk, ncat, self.nres, self.D))
        self.aw = np.zeros((nk, ncat))
        grp = np.stack([kinv, cats] + [np.asarray(after[:, p], np.int64) for p in range(self.nres)], 1)
        ug, ginv = np.unique(grp, axis=0, return_inverse=True)
        gw = np.bincount(ginv.ravel(), w, len(ug))
        for g, row in enumerate(ug):
            self._acc(int(row[0]), int(row[1]), tuple(int(v) for v in row[2:]), gw[g])
        self.X = np.stack([self.key_vec(k) for k in self.keys])
        self.ulist, self.upos = [], {}
        self.kpos = np.array([self._pos(k[2]) for k in self.keys], np.int64)
        self.qcache = {}
        self.report = {"keys": nk, "pairs": len(self.gid), "sets": len(self.ulist),
                       "categories": int((self.counts.sum(0) > 0).sum())}
        if len(self.gid) >= 2:
            Rk = self.counts / self.counts.sum(1, keepdims=True)
            self.lam, l0, l1 = fit_lambda_sets(S, self.X, self.kpos, self.ulist, Rk, self.group)
            self.report.update({"loo_start": round(l0, 4), "loo_fitted": round(l1, 4)})
        else:
            self.lam = np.ones(3 * self.D)
        self.lam = np.asarray(self.lam, np.float64)
        D = self.D
        self.report["lambda_sums"] = {"front": round(float(self.lam[:D].sum()), 2),
                                      "held": round(float(self.lam[D:2 * D].sum()), 2),
                                      "view": round(float(self.lam[2 * D:].sum()), 2)}
        self.alpha, la, l37 = fit_alpha_dist(self.key_dists(np.arange(nk)), self.counts)
        self.report.update({"alpha": round(self.alpha, 5), "loo_per_try_alpha": round(la, 5),
                            "loo_per_try_card037": round(l37, 5)})
        self.base = (list(self.keys), dict(self.gid), list(self.group), self.X.copy(), self.counts.copy(),
                     dict(self.aft), (self.sums.copy(), self.aw.copy()), self.kpos.copy(), list(self.ulist))
        self.forget()

    # -- the sets
    def _pos(self, sid):
        p = self.upos.get(sid)
        if p is None:
            p = self.upos[sid] = len(self.ulist)
            self.ulist.append(sid)
        return p

    def set_dists(self, sid):
        """Chamfer distances from set sid to every distinct stored set (cached; extended as sets are added)."""
        r = self.qcache.get(sid)
        if r is None or len(r) < len(self.ulist):
            have = 0 if r is None else len(r)
            new = chamfer_to(self.S, sid, self.ulist[have:], self.lam[2 * self.D:])
            r = self.qcache[sid] = new if r is None else np.concatenate([r, new])
        return r

    def dists(self, qs):
        """Recall's distances from each query key to every stored key."""
        D = self.D
        A = np.stack([self.key_vec(q) for q in qs])
        out = VP.wl1(A, self.X, self.lam[:2 * D])
        for i, q in enumerate(qs):
            out[i] += self.set_dists(int(q[2]))[:len(self.ulist)][self.kpos]
        return out

    def key_dists(self, idx):
        return self.dists([self.keys[t] for t in idx])[:, idx]

    # -- card 038's Kind, with the distances above
    def key_vec(self, q):
        return np.concatenate([self.S.arr[int(q[0])], self.S.arr[int(q[1])]])

    def reset(self):
        keys, gid, group, X, counts, aft, rest, kpos, ulist = self.base
        self.keys, self.gid, self.group = list(keys), dict(gid), list(group)
        self.X, self.counts, self.aft = X.copy(), counts.copy(), dict(aft)
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.sums, self.aw = rest[0].copy(), rest[1].copy()
        self.kpos, self.ulist = kpos.copy(), list(ulist)
        self.upos = {s: i for i, s in enumerate(self.ulist)}
        n = len(self.ulist)
        self.qcache = {s: r[:n] for s, r in self.qcache.items()}
        self.forget()

    def add(self, key, cat, after=None, w=1.0):
        key = tuple(int(h) for h in key)
        if key not in self.index:
            self.kpos = np.append(self.kpos, self._pos(key[2]))
        super().add(key, cat, after, w)

    def kernel(self, q):
        return np.exp(-self.dists([q])[0])

    def cat_of(self, qs):
        todo = [q for q in dict.fromkeys(qs) if q not in self.ccache]
        if todo:
            share = self.counts / self.counts.sum(1, keepdims=True)
            k = np.exp(-self.dists(todo))
            k[k < VP.KMIN] = 0.0
            for i, q in enumerate(todo):
                own = self.index.get(q)
                P, _ = VP.neighbour_vote(k[i], share, own)
                if own is not None:
                    P = (self.counts[own] + self.alpha * P) / (self.counts[own].sum() + self.alpha)
                self.ccache[q] = int(P.argmax()) if P.max() >= 0.5 else None
        return [self.ccache[q] for q in qs]

    def pair_distance(self, p, q):
        D = self.D
        held = float(np.abs(self.S.arr[int(p[0])] - self.S.arr[int(q[0])]) @ self.lam[D:2 * D])
        return held + float(chamfer_to(self.S, int(p[1]), [int(q[1])], self.lam[2 * D:])[0])

    def ways(self, c):
        r = self.ways_cache.get(c)
        if r is not None:
            return r
        idx = np.flatnonzero(self.aw[:, c] > 0)
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
        self.ways_cache[c] = r
        return r


def kind(name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
    cls = SlotKind if nkey == 3 else _Kind
    return cls(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)


class SlotWorld(VP.World):
    """Card 038's World; what is in view is a set's id instead of a merged vector's handle."""

    def ctx_of(self, hs):
        return set_of(hs)


HOLDOUT = "pair"                              # recall's weights: leave out a (front, held) pair, or one key


def install():
    VP.Kind, VP.World = kind, SlotWorld


# ---------------------------------------------------------------- criterion 3: the view comparison alone

RULE = {"key": lambda held_fits, on: held_fits, "switch": lambda held_fits, on: on,
        "either": lambda held_fits, on: held_fits or on, "both": lambda held_fits, on: held_fits and on}


def check(arm, seeds, worlds, n, out, log):
    """Toggling the closed door holding nothing, in the view as it is and after each imagined change (each key
    picked up; the switch turned on), against the world's rule (the evaluator's): from memory before acting."""
    T, F, ld = VP.T, VP.F, VP.ld
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pickle.loads(T.DATA.read_bytes())
    cache = pickle.loads(T.MEMORY.read_bytes())
    groups = T.use_groups(cache["mem"], dev)
    res = {"note": "Card 039 criterion 3, tools/card039/slot_planner.py --check", "arm": arm, "seeds": []}
    for seed in seeds:
        z, parts, _ = VP.vectors_of(arm, seed, cache["tiles"], cache["pairs"], groups, dev, log)
        S = VP.Store(z)
        lut = np.zeros(256, np.uint8)
        lut[:len(S.hof)] = S.hof
        VP.Kd.APP = lut
        floor = None
        rs = {"seed": seed, "worlds": {}}
        for world in worlds:
            W = SlotWorld(S, parts, d["data"][world], dev, lambda m: None)
            VP.WORLD, F.M = W, W.M
            nm = lambda h: W.judge.name(int(h))
            floor = next(h for h in range(S.nobs) if nm(h) == "floor")
            kd = W.kinds[VP.TOG]
            right, total, wrong = 0, 0, []
            for li in range(n):
                lay = d["data"][world]["test"][li]
                W.reset()
                pl = VP.VPlan(W)
                f, st, _, _ = pl.observe(None, VP.see(lay, ld.start_state(lay)))
                things = np.unique(f[:VP.NV]).tolist()
                door = [u for u in things if nm(u).startswith("closed door")]
                if not door:
                    continue
                door = door[0]
                on0 = any(nm(u).startswith("switch on") for u in things)
                cases = [("as now", st, on0)]
                cases += [(f"pick up {nm(k)}", pl.imagine(st, VP.PICK, int(k), None), on0)
                          for k in things if nm(k).startswith("key")]
                cases += [("switch on", pl.imagine(st, VP.TOG, int(u), None), True)
                          for u in things if nm(u).startswith("switch off")]
                for name, s2, on in cases:
                    total += 1
                    if s2 is None:
                        wrong.append((li, name, "no change predicted"))
                        continue
                    pred = kd.cat_of([(int(door), floor, pl.ctx_id(s2[0], s2[1]))])[0]
                    truth = RULE[world](False, on)
                    if bool(pred and pred & 1) == truth:
                        right += 1
                    else:
                        wrong.append((li, name, f"predicted {pred}, rule {'opens' if truth else 'stays'}"))
            rs["worlds"][world] = {"cases": total, "right_share": round(right / max(total, 1), 4),
                                   "wrong_first": wrong[:8], "toggle_kind": kd.report}
            log(f"  check arm {arm} seed {seed} {world}: {right}/{total} right; toggle {kd.report['lambda_sums']}")
        res["seeds"].append(rs)
        out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def main():
    global HOLDOUT
    install()
    args = sys.argv[1:]
    get = lambda k, dflt: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), dflt)
    HOLDOUT = get("--holdout", HOLDOUT)
    assert HOLDOUT in ("pair", "key")
    if "--check" in args:
        VP.T.configure()
        t00 = time.monotonic()
        log = lambda msg: print(f"[{time.monotonic() - t00:6.0f}s] {msg}", flush=True)
        a_, b_ = get("--seeds", "400-404").split("-")
        check(get("--arm", "B"), range(int(a_), int(b_) + 1), tuple(get("--worlds", "switch,either").split(",")),
              int(get("--layouts", "50")), Path(get("--out", "runs/039_check.json")), log)
        return
    if "--out" not in args:
        sys.exit("--out is required (card 038's default path would be overwritten)")
    VP.main()
    out = Path(get("--out", ""))
    if out.exists() and "--dev" not in args:
        r = json.loads(out.read_text())
        r["note"] = "Card 039, tools/card039/slot_planner.py (card 038's planner, view as a set of things)"
        out.write_text(json.dumps(r, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
