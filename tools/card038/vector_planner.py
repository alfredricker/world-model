"""Card 038: planning on vectors (Appendix A as revised on 2026-09-30, after the shakedown).

The planner holds what it sees as the encoder's vectors and never labels them. Each distinct set of numbers
is stored once (a handle), so search recognises a state it has already seen only by exactly the same
numbers; nothing is merged by similarity.

Recall predicts every effect. A kind's key is the thing in front (left, right, forward, and the agent drawn
onto and undrawn from a tile), or the thing in front, the thing held and what is in view (pick up, toggle,
drop). What is in view is one vector: the largest value, per dimension, among the things in view, the
agent's own place aside (a set pooled into one vector, 1703_06114). The other keys vote with weight
k = exp(-sum_j lambda_j |x_j - x'_j|), a key's tries k / n each (card 037): q_c = (W_c + 1/4) / (W + 1).
Lambda is fitted per world and kind by card 037's leave-one-out, with every key of the same thing in front
and held (a pair) left out together. The key's own tries n_c join with that vote as a prior of strength
alpha (revision, agreed with the user on 2026-09-30; hierarchical Dirichlet smoothing, MacKay & Peto 1995):
P_c = (n_c + alpha q_c) / (n + alpha), alpha fitted per world and kind by leave-one-try-out, not refitted
while acting. A key with no tries of its own gets q, as in card 037, whose rule is alpha = W + 1. An outcome
category (which places change; for moves: moved, blocked, ended) is predicted when P_c >= 1/2. What
results is carried over part by part: kept, copied from the other changed place, shifted by the
neighbours' weighted mean change, or set to their weighted mean result; the way per part is chosen by
leave-one-pair-out over the category's keys. Forward decides whether the agent enters a tile; draw and
undraw supply only how the agent looks on it (on forward's lambda).

Working backward keeps card 029's search, walking and order, and every condition is a prediction:
"episode ended"; "place j can be walked onto" (forward's prediction for its thing); "facing"; and part
conditions, inferred from memory (section 2, agreed with the user on 2026-09-30). When doing a on thing u
does not make condition c true with what is held and in view now, the (held, view) pairs of the kind's
stored tries on things like u are split into successes (with that pair, recall predicts that doing a on u
makes c true) and failures. A part (the thing held, what is in view) is a condition when doing a on u fails
with that part as now and the other part as in a success. It is met when recall predicts success with that
part as it is and the other part as in the success, and achieved by a pick up, toggle or drop whose
predicted effect changes that part. Each condition is one prediction; nothing searches over sequences of
imagined actions (GOAL.md P21). Online learning: every real try joins its
kind's memory and that kind's cached predictions are dropped (lambda fixed); memory carries across the
layouts of one worker's chunk. The evaluator judges a predicted place by the real tile nearest its vector
(L1, among the 52 tiles the view can show).

Steps:
  bin/prun python tools/card038/vector_planner.py --encoders --seeds 399-409        arm B's encoders
  bin/prun python tools/card038/vector_planner.py --gate                            arm B's encoders checked
  bin/prun python tools/card038/vector_planner.py --dev --arm A --seeds 399-399 [--worlds key] [--layouts 40]
  bin/prun python tools/card038/vector_planner.py --arm B --seeds 400-409 [--layouts-b 100] --out runs/038_armB.json
Arms: 1 oracle vectors (one-hot kind, colour, state, agent), seed-independent; A card 037's encoders (the
codebooks exist but are never read); B the same recipe without the codebook terms.
"""
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card037"))
import recall_planner as RP  # noqa: E402  (card 037 -> 036 -> 035 -> 034 -> 033 -> 031 -> 029 -> 028)

N, T, CO = RP.N, RP.T, RP.CO
F, G, dl, ld, Kd, closer = CO.F, CO.G, CO.dl, CO.ld, CO.Kd, CO.closer
KP, R33 = N.KP, T.R
NV, NPL, W13 = CO.NV, CO.NPL, CO.W13
CENTRE, FRONT, HELD, BEHIND = CO.CENTRE, CO.FRONT, CO.HELD, CO.BEHIND
LEFT, RIGHT, FWD, PICK, DROP, TOG = range(6)
MOVES = (LEFT, RIGHT, FWD)
INTER = (PICK, TOG, DROP)
CHANGED, UNCHANGED, ENDED = CO.CHANGED, CO.UNCHANGED, CO.ENDED
APP0 = CO.APP0
WORLDS = CO.WORLDS
PRIOR, KMIN = 0.25, 0.01                     # card 037
HELDP, VIEWP = 0, 1                          # the parts of a pick up, toggle or drop key a condition is about
BUDGET = 200
ROOT = Path(__file__).resolve().parents[2]
ENC_A = ROOT / "runs" / "037_enc"
ENC_B = ROOT / "runs" / "038_encB"
MU = 0.01
VIEW = np.array([int(c) for c in CO.VIEW_CODES])      # the 52 tiles the view can show
WAYS = ("keep", "copy", "shift", "set")
KEEP, COPY, SHIFT, SET = range(4)
KNAME = {LEFT: "left", RIGHT: "right", FWD: "forward", PICK: "pickup", DROP: "drop", TOG: "toggle"}
WORLD = None                                           # the model the workers use (set before forking)
LAYOUTS_B = None                                       # --layouts-b: acting layouts in world (b), reported only
PROG = None                                            # main runs: the progress file and its contents


def progress(**kw):
    """Main runs: merge kw into the progress file next to the results (read by monitor.py)."""
    if PROG is None:
        return
    PROG["data"].update(kw, updated=time.time())
    tmp = PROG["path"].with_suffix(".tmp")
    tmp.write_text(json.dumps(PROG["data"], default=str) + "\n")
    tmp.replace(PROG["path"])


# ---------------------------------------------------------------- vectors

class Store:
    """Every vector seen or made, each distinct set of numbers once. hof: tile code -> handle."""

    def __init__(self, z_codes):
        self.idx, self.n = {}, 0
        self.arr = np.zeros((512, z_codes.shape[1]))
        self.hof = np.array([self.add(v) for v in z_codes], np.int64)
        self.nobs = int(self.hof.max()) + 1          # handles below this are tiles the renderer can draw

    def add(self, v):
        v = np.round(np.asarray(v, np.float64), 9) + 0.0      # float noise below 1e-9 (and -0.0) is not a new vector
        b = v.tobytes()
        h = self.idx.get(b)
        if h is None:
            if self.n == len(self.arr):
                self.arr = np.concatenate([self.arr, np.zeros_like(self.arr)])
            h = self.idx[b] = self.n
            self.arr[h] = v
            self.n += 1
        return h


class Judge:
    """Evaluator: a predicted handle is right at a place when the real tile there is (one of) the tiles
    nearest its vector (L1, among the 52 the view can show)."""

    def __init__(self, S):
        self.S = S
        self.Zv = S.arr[S.hof[VIEW]].copy()
        self.cache = {}

    def right(self, h, true):
        h, true = int(h), int(true)
        if h == true:
            return True
        k = (h, true)
        r = self.cache.get(k)
        if r is None:
            v = self.S.arr[h]
            d = np.abs(self.Zv - v).sum(1)
            r = self.cache[k] = bool(np.abs(self.S.arr[true] - v).sum() <= d.min() + 1e-9)
        return r

    def name(self, h):
        v = self.S.arr[int(h)]
        d = np.abs(self.Zv - v).sum(1)
        i = int(d.argmin())
        return CO.name_of_code(VIEW[i]) + ("" if d[i] < 1e-9 else f" (~{d[i]:.2f})")


def oracle_vectors():
    """Arm 1: four one-hot parts from the evaluator's tile names (kind, colour, state, agent). The switch's
    colour is its state (on, off); a door's shut and closed states are both closed."""
    labs = [CO.sim_labels(c) for c in range(len(CO.TILES))]
    for lab in labs:
        if lab["object"] == "switch":
            lab["colour"] = "none"
    fields = ("object", "colour", "state", "agent on it")
    vals = [sorted({str(lab[f]) for lab in labs}) for f in fields]
    z = np.zeros((len(labs), sum(len(v) for v in vals)))
    parts, o = [], 0
    for v in vals:
        parts.append(slice(o, o + len(v)))
        o += len(v)
    for i, lab in enumerate(labs):
        for f, v, p in zip(fields, vals, parts):
            z[i, p.start + v.index(str(lab[f]))] = 1.0
    return z, parts


def wl1(A, B, lam):
    """Weighted L1 distances between the rows of A and the rows of B. Beyond two rows of A, summed one
    dimension at a time: the same numbers (to about 1e-14) without a len(A) x len(B) x d temporary."""
    if len(A) <= 2:
        return np.abs(A[:, None] - B[None]) @ lam
    out = np.zeros((len(A), len(B)))
    Bt = np.ascontiguousarray(B.T)
    for j in range(A.shape[1]):
        out += lam[j] * np.abs(A[:, j, None] - Bt[j][None])
    return out


def fit_lambda(X, Rk, group, steps=1500, lr=0.02):
    """Card 037's fit: the mean over keys of the leave-one-out log-likelihood of each key's outcome shares,
    every lambda equal at the start with the median distance between keys 1. Keys of the same group (the
    same pair) are left out together. On the GPU. Returns lambda and the objective at the start and after."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Rt = torch.as_tensor(Rk, dtype=torch.float32, device=dev)
    n, L = Rt.shape
    D = torch.abs(Xt[:, None] - Xt[None])                           # (n, n, d)
    dsum = D.sum(-1)
    med = float(dsum[dsum > 0].median()) if bool((dsum > 0).any()) else 1.0
    lam0 = 1.0 / max(med, 1e-6)
    g = torch.as_tensor(np.asarray(group), device=dev)
    keep = (g[:, None] != g[None]).float()
    th = torch.nn.Parameter(torch.full((X.shape[1],), float(np.log(np.expm1(lam0))), device=dev))

    def ll(theta):
        lam = torch.nn.functional.softplus(theta)
        k = torch.exp(-(D @ lam)) * keep
        P = (k @ Rt + 1.0 / L) / (k.sum(-1, keepdim=True) + 1.0)
        return (Rt * torch.log(P)).sum(-1).mean()

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


def neighbour_vote(k, share, own):
    """Card 037's vote from the other keys, with its prior: q_c = (W_c + 1/4) / (W + 1), W_c = sum over the
    other keys of k x share; and W."""
    wn = k @ share if own is None else k @ share - k[own] * share[own]
    return (wn + PRIOR) / (wn.sum() + 1.0), float(wn.sum())


def fit_alpha(X, counts, lam):
    """The strength of the neighbours' vote as a prior (hierarchical Dirichlet smoothing, MacKay & Peto
    1995): P_c = (n_c + alpha q_c) / (n + alpha), n_c a key's own tries. alpha maximises the leave-one-try-
    out log-likelihood of the stored tries (weighted tries count as that many tries), on a grid of log alpha.
    Also: the same objective under card 037's rule, which is alpha = W + 1 for each key."""
    k = np.exp(-wl1(X, X, lam))
    k[k < KMIN] = 0.0
    np.fill_diagonal(k, 0.0)
    share = counts / counts.sum(1, keepdims=True)
    wn = k @ share
    q = (wn + PRIOR) / (wn.sum(1, keepdims=True) + 1.0)
    n = counts.sum(1, keepdims=True)
    m = counts > 0
    rest = np.maximum(counts - 1.0, 0.0)

    def ll(alpha):
        a = np.broadcast_to(alpha, n.shape)
        P = (rest + a * q) / np.maximum(n - 1.0 + a, 1e-12)
        return float((counts[m] * np.log(P[m])).sum() / counts.sum())

    grid = np.exp(np.linspace(-9.0, 9.0, 181))
    vals = [ll(a) for a in grid]
    i = int(np.argmax(vals))
    return float(grid[i]), vals[i], ll(wn.sum(1, keepdims=True) + 1.0)


# ---------------------------------------------------------------- one kind's memory

class Kind:
    """One kind's stored tries merged by key (exact handles) and category, with recall's metric. Keys:
    (front,) for moves, draw and undraw; (front, held, view) for pick up, toggle and drop. Result places:
    front and held (pick up, toggle, drop); one place (draw, undraw: its before is the key, and its other
    place is the category's most common one there)."""

    def __init__(self, name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
        self.name, self.S, self.parts, self.nkey, self.ncat = name, S, parts, nkey, ncat
        self.D = S.arr.shape[1]
        self.pair = nkey >= 2
        self.moves = after is None
        self.nres = 0 if after is None else after.shape[1]
        keys = np.asarray(keys, np.int64).reshape(len(w), nkey)
        cats = np.asarray(cats, np.int64)
        uk, kinv = np.unique(keys, axis=0, return_inverse=True)
        kinv = kinv.ravel()
        self.keys = [tuple(int(v) for v in r) for r in uk]
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.gid = {}
        self.group = [self.gid.setdefault(k[:2], len(self.gid)) for k in self.keys]
        nk = len(self.keys)
        self.counts = np.zeros((nk, ncat))
        np.add.at(self.counts, (kinv, cats), w)
        self.rep, self.aft = {}, {}                             # aft: (key, cat) -> after handles when all tries agree
        if not self.moves:
            self.sums = np.zeros((nk, ncat, self.nres, self.D))
            self.aw = np.zeros((nk, ncat))
            grp = np.stack([kinv, cats] + [np.asarray(after[:, p], np.int64) for p in range(self.nres)], 1)
            ug, ginv = np.unique(grp, axis=0, return_inverse=True)
            gw = np.bincount(ginv.ravel(), w, len(ug))
            for g, row in enumerate(ug):
                self._acc(int(row[0]), int(row[1]), tuple(int(v) for v in row[2:]), gw[g])
            if other is not None:
                for c in range(ncat):
                    m = cats == c
                    if m.any():
                        u, inv = np.unique(other[m], return_inverse=True)
                        self.rep[c] = int(u[np.bincount(inv.ravel(), w[m]).argmax()])
        self.X = np.stack([self.key_vec(k) for k in self.keys])
        if lam is None:
            self.report = {"keys": nk, "pairs": len(self.gid), "categories": int((self.counts.sum(0) > 0).sum())}
            if len(self.gid) >= 2:
                Rk = self.counts / self.counts.sum(1, keepdims=True)
                self.lam, l0, l1 = fit_lambda(self.X, Rk, self.group)
                self.report.update({"loo_start": round(l0, 4), "loo_fitted": round(l1, 4)})
            else:
                self.lam = np.ones(self.X.shape[1])
        else:
            self.lam = lam
            self.report = {"keys": nk, "lambda": "forward's"}
        self.lam = np.asarray(self.lam, np.float64)
        self.alpha = None                                      # draw and undraw: look() weighs as before
        if lam is None:
            self.alpha, la, l37 = fit_alpha(self.X, self.counts, self.lam)
            self.report.update({"alpha": round(self.alpha, 5), "loo_per_try_alpha": round(la, 5),
                                "loo_per_try_card037": round(l37, 5)})
        self.base = (list(self.keys), dict(self.gid), list(self.group), self.X.copy(), self.counts.copy(),
                     dict(self.aft), None if self.moves else (self.sums.copy(), self.aw.copy()))
        self.forget()

    def _acc(self, t, c, hs, w):
        self.aw[t, c] += w
        for p in range(self.nres):
            self.sums[t, c, p] += w * self.S.arr[hs[p]]
        if (t, c) not in self.aft:
            self.aft[(t, c)] = hs
        elif self.aft[(t, c)] != hs:
            self.aft[(t, c)] = None

    # -- memory
    def key_vec(self, q):
        return np.concatenate([self.S.arr[int(h)] for h in q])

    def reset(self):
        keys, gid, group, X, counts, aft, rest = self.base
        self.keys, self.gid, self.group = list(keys), dict(gid), list(group)
        self.X, self.counts, self.aft = X.copy(), counts.copy(), dict(aft)
        self.index = {k: i for i, k in enumerate(self.keys)}
        if rest is not None:
            self.sums, self.aw = rest[0].copy(), rest[1].copy()
        self.forget()

    def forget(self):
        self.ccache, self.rcache, self.ways_cache, self.amat, self.tcache = {}, {}, {}, {}, {}

    def add(self, key, cat, after=None, w=1.0):
        key = tuple(int(h) for h in key)
        i = self.index.get(key)
        if i is None:
            i = self.index[key] = len(self.keys)
            self.keys.append(key)
            self.group.append(self.gid.setdefault(key[:2], len(self.gid)))
            self.X = np.vstack([self.X, self.key_vec(key)[None]])
            self.counts = np.vstack([self.counts, np.zeros((1, self.ncat))])
            if not self.moves:
                self.sums = np.concatenate([self.sums, np.zeros((1,) + self.sums.shape[1:])])
                self.aw = np.vstack([self.aw, np.zeros((1, self.ncat))])
        self.counts[i, cat] += w
        if not self.moves:
            self._acc(i, cat, tuple(int(h) for h in after), w)
        self.forget()

    def after_mat(self, c, p):
        """Every key's result at changed place p in category c (the stored vector when its tries agree)."""
        m = self.amat.get((c, p))
        if m is None:
            m = self.sums[:, c, p] / np.maximum(self.aw[:, c], 1e-300)[:, None]
            for (t, cc), hs in self.aft.items():
                if cc == c and hs is not None:
                    m[t] = self.S.arr[hs[p]]
            self.amat[(c, p)] = m
        return m

    # -- recall
    def kernel(self, q):
        return np.exp(-(np.abs(self.X - self.key_vec(q)) @ self.lam))

    def vote(self, q):
        k = self.kernel(q)
        own = self.index.get(q)
        return float(k.sum() - (k[own] if own is not None else 0.0))

    def cat_of(self, qs):
        """Per query key: the predicted category, or None when none reaches one half."""
        todo = [q for q in dict.fromkeys(qs) if q not in self.ccache]
        if todo:
            share = self.counts / self.counts.sum(1, keepdims=True)
            k = np.exp(-wl1(np.stack([self.key_vec(q) for q in todo]), self.X, self.lam))
            k[k < KMIN] = 0.0
            for i, q in enumerate(todo):
                own = self.index.get(q)
                P, _ = neighbour_vote(k[i], share, own)
                if own is not None:                                 # the vote as a prior of strength alpha
                    P = (self.counts[own] + self.alpha * P) / (self.counts[own].sum() + self.alpha)
                self.ccache[q] = int(P.argmax()) if P.max() >= 0.5 else None
        return [self.ccache[q] for q in qs]

    def move(self, q):
        return self.cat_of([q])[0]

    def templates(self, u):
        """The distinct (held, view) pairs of the stored tries on things like u (k >= KMIN on the front part
        of the metric)."""
        r = self.tcache.get(u)
        if r is None:
            D = self.D
            kf = np.exp(-(np.abs(self.X[:, :D] - self.S.arr[int(u)]) @ self.lam[:D]))
            r = self.tcache[u] = sorted({self.keys[t][1:] for t in np.flatnonzero(kf >= KMIN)})
        return r

    def pair_distance(self, p, q):
        """Recall's distance between two (held, view) pairs (the metric's held and view parts)."""
        D = self.D
        return float(np.abs(np.concatenate([self.S.arr[p[0]] - self.S.arr[q[0]], self.S.arr[p[1]] - self.S.arr[q[1]]]))
                     @ self.lam[D:3 * D])

    def result(self, q, c):
        """Category c's result for key q: handles per changed place (None where unchanged)."""
        r = self.rcache.get((q, c))
        if r is None:
            k = self.kernel(q)
            k[k < KMIN] = 0.0
            wk = k * self.counts[:, c] / self.counts.sum(1)
            own = self.index.get(q)
            if own is not None:                                     # the others' results as the prior's share
                wk *= self.alpha / (k.sum() - k[own] + 1.0)
                wk[own] = self.counts[own, c]
            r = self.rcache[(q, c)] = tuple(self.transport(q, c, wk))
        return r

    def look(self, q):
        """Draw and undraw: the result, carried over from every stored key (no threshold: forward decides
        whether the agent enters; this only says how it looks)."""
        r = self.rcache.get(q)
        if r is None:
            logw = -(np.abs(self.X - self.key_vec(q)) @ self.lam)
            own = self.index.get(q)
            if own is not None:
                logw[own] = np.log(self.counts[own, 0])
            wk = np.exp(logw - logw.max())
            r = self.rcache[q] = self.transport(q, 0, wk)[0]
        return r

    # -- what results
    def changed(self, c):
        if not self.pair:
            return [0]
        return [p for p in range(2) if (c >> p) & 1]

    def before(self, q, p):
        return self.S.arr[int(q[p if self.pair else 0])]

    def other(self, q, p, c):
        return self.S.arr[int(q[1 - p])] if self.pair else self.S.arr[self.rep[c]]

    def before_rows(self, idx, p):
        s = p if self.pair else 0
        return self.X[idx, s * self.D:(s + 1) * self.D]

    def other_rows(self, idx, p, c):
        if self.pair:
            return self.X[idx, (1 - p) * self.D:(2 - p) * self.D]
        return np.repeat(self.S.arr[self.rep[c]][None], len(idx), 0)

    def ways(self, c):
        """Per changed place, per part: keep, copy, shift or set, by leave-one-pair-out over the category's
        keys (equal weight per key; each predicted from the other pairs' keys, weighted by k x share). Fewer
        than two pairs: shift."""
        r = self.ways_cache.get(c)
        if r is not None:
            return r
        idx = np.flatnonzero(self.aw[:, c] > 0)
        grp = np.asarray(self.group)[idx]
        r = {}
        for p in self.changed(c):
            if len(set(grp.tolist())) < 2:
                r[p] = [SHIFT] * len(self.parts)
                continue
            A = self.after_mat(c, p)[idx]
            B, O = self.before_rows(idx, p), self.other_rows(idx, p, c)
            share = self.counts[idx, c] / self.counts[idx].sum(1)
            logw = -wl1(self.X[idx], self.X[idx], self.lam) + np.log(share)[None]
            logw[grp[:, None] == grp[None]] = -np.inf
            w = np.exp(logw - logw.max(1, keepdims=True))
            w /= w.sum(1, keepdims=True)
            choice = []
            for sl in self.parts:
                a, b, o = A[:, sl], B[:, sl], O[:, sl]
                err = [np.abs(b - a).sum(), np.abs(o - a).sum(),
                       np.abs(b + w @ (a - b) - a).sum(), np.abs(w @ a - a).sum()]
                choice.append(int(np.argmin(err)))
            r[p] = choice
        self.ways_cache[c] = r
        return r

    def transport(self, q, c, wk):
        """The category's result at each changed place (handles; None where unchanged)."""
        idx = np.flatnonzero((wk > 0) & (self.aw[:, c] > 0))
        out = [None] * max(self.nres, 1)
        if len(idx) == 0:
            return out
        ww = wk[idx] / wk[idx].sum()
        ways = self.ways(c)
        for p in self.changed(c):
            bq, oq = self.before(q, p), self.other(q, p, c)
            A, B = self.after_mat(c, p)[idx], self.before_rows(idx, p)
            res = np.empty(self.D)
            for sl, way in zip(self.parts, ways[p]):
                if way == KEEP:
                    res[sl] = bq[sl]
                elif way == COPY:
                    res[sl] = oq[sl]
                elif way == SHIFT:
                    res[sl] = bq[sl] + ww @ (A[:, sl] - B[:, sl])
                else:
                    res[sl] = ww @ A[:, sl]
            out[p] = self.S.add(res)
        return out


# ---------------------------------------------------------------- the world's model

class Lazy:
    def __init__(self, fn):
        self.fn, self.c = fn, {}

    def __getitem__(self, h):
        h = int(h)
        v = self.c.get(h)
        if v is None:
            v = self.c[h] = self.fn(h)
        return v

    def clear(self):
        self.c.clear()


class World:
    """One world's learned model: card 028's move maps and poses (fitted on exact repeats of the vectors),
    and recall's kinds."""

    def __init__(self, S, parts, D, dev, log):
        t0 = time.monotonic()
        self.S, self.parts, self.judge = S, parts, Judge(S)
        self.keep = np.ones(NV, bool)
        self.keep[CENTRE] = False                              # what is in view: the agent's own place aside
        self.ctxc, self.openc, self.openedc = {}, {}, {}
        lut = S.hof.astype(np.uint8)
        V0, V1 = lut[D["ego0"]], lut[D["ego1"]]
        act, w, term = D["act"], D["w"], D["term1"]
        vc = (V0[:, :NV] != V1[:, :NV]).any(1)
        out = np.where(term, ENDED, np.where(vc, CHANGED, UNCHANGED))
        # moves: card 028's maps and poses
        m = F.Model()
        rng = np.random.default_rng(0)
        m.src, m.ent = {}, {}
        for a in MOVES:
            rows = np.flatnonzero((act == a) & (out != UNCHANGED))
            if len(rows) > 40000:
                rows = np.sort(rng.choice(rows, 40000, replace=False))
            src, ent, _, _, _, _ = F.fit_map(V0[rows], V1[rows], dev)
            m.src[a], m.ent[a] = src, ent
        m.change_places = np.array([FRONT, HELD])
        F.build_poses(m)
        assert all(h == HELD for h in m.hidx), "the held place moves with the pose"
        self.M = m
        # what is in view, per stored try
        sel = np.flatnonzero(np.isin(act, INTER))
        pres = np.zeros((len(sel), S.nobs), bool)
        pres[np.arange(len(sel))[:, None], V0[sel, :NV][:, self.keep]] = True
        up, pinv = np.unique(pres, axis=0, return_inverse=True)
        pc = np.array([self.ctx_of(np.flatnonzero(r)) for r in up], np.int64)
        cid = np.zeros(len(act), np.int64)
        cid[sel] = pc[pinv.ravel()]
        # kinds
        k = {}
        for a in MOVES:
            s = np.flatnonzero(act == a)
            k[a] = Kind(KNAME[a], S, parts, 1, 3, V0[s, FRONT][:, None], out[s], w[s])
        fw = np.flatnonzero((act == FWD) & (out != UNCHANGED))
        lamf = k[FWD].lam
        k["draw"] = Kind("draw", S, parts, 1, 1, V0[fw, FRONT][:, None], np.zeros(len(fw), np.int64), w[fw],
                         after=V1[fw, CENTRE][:, None], other=V0[fw, CENTRE], lam=lamf)
        k["undraw"] = Kind("undraw", S, parts, 1, 1, V0[fw, CENTRE][:, None], np.zeros(len(fw), np.int64), w[fw],
                           after=V1[fw, BEHIND][:, None], other=V0[fw, FRONT], lam=lamf)
        for a in INTER:
            s = np.flatnonzero(act == a)
            b, af = V0[s][:, [FRONT, HELD]].astype(np.int64), V1[s][:, [FRONT, HELD]].astype(np.int64)
            cat = (af[:, 0] != b[:, 0]).astype(np.int64) + 2 * (af[:, 1] != b[:, 1])
            k[a] = Kind(KNAME[a], S, parts, 3, 4, np.concatenate([b, cid[s][:, None]], 1), cat, w[s], after=af)
        self.kinds = k
        m.move_out = {a: Lazy(lambda h, a=a: self.move_cat(a, h)) for a in MOVES}
        m.free = Lazy(lambda h: self.move_cat(FWD, h) == CHANGED)
        m.draw = Lazy(lambda h: self.kinds["draw"].look((h,)))
        m.undraw = Lazy(lambda h: self.kinds["undraw"].look((h,)))
        self.seconds = round(time.monotonic() - t0, 1)
        self.report = {"kinds": {kd.name: kd.report for kd in k.values()}, "poses": int(len(m.A)),
                       "pose_route_conflicts": m.pose_conflicts, "views_in_memory": int(len(up)),
                       "view_vectors_in_memory": int(len(set(pc.tolist()))), "seconds": self.seconds}

    # -- what is in view
    def ctx_of(self, hs):
        key = frozenset(int(h) for h in hs)
        r = self.ctxc.get(key)
        if r is None:
            r = self.ctxc[key] = self.S.add(self.S.arr[sorted(key)].max(0))
        return r

    def ctx_of_view(self, V):
        return self.ctx_of(np.unique(np.asarray(V)[:NV][self.keep]))

    def openable(self, u):
        """Memory holds a pick up or toggle that made a thing like u (k >= KMIN on the front part of that
        kind's metric) into one to walk onto: places showing u are checked as things in the way."""
        r = self.openc.get(u)
        if r is None:
            r = False
            for a in (PICK, TOG):
                kd = self.kinds[a]
                fr = self.openedc.get(a)
                if fr is None:
                    out = set()
                    for c in (1, 3):
                        for t in np.flatnonzero(kd.aw[:, c] > 0):
                            b = kd.keys[t][0]
                            if not self.M.free[b] and self.M.free[self.S.add(kd.after_mat(c, 0)[t])]:
                                out.add(b)
                    fr = self.openedc[a] = np.array(sorted(out), np.int64)
                if len(fr) and np.exp(-(np.abs(self.S.arr[fr] - self.S.arr[u]) @ kd.lam[:kd.D])).max() >= KMIN:
                    r = True
                    break
            self.openc[u] = r
        return r

    # -- predictions
    def move_cat(self, a, h):
        c = self.kinds[a].move((int(h),))
        return UNCHANGED if c is None else c                  # declared: unknown -> blocked, unchanged

    def outcome(self, a, u, h, cid):
        """(category, (front after, held after)); None where unchanged."""
        q = (int(u), int(h), int(cid))
        c = self.kinds[a].cat_of([q])[0]
        if not c:
            return 0, (None, None)
        return c, self.kinds[a].result(q, c)

    # -- online learning
    def reset(self):
        for kd in self.kinds.values():
            kd.reset()
        self.clear_lazy()

    def clear_lazy(self):
        self.openc.clear()
        self.openedc.clear()
        for a in MOVES:
            self.M.move_out[a].clear()
        self.M.free.clear()
        self.M.draw.clear()
        self.M.undraw.clear()

    def learn_try(self, V, a, V2, end):
        """Add one real try to its kind's memory (and forward's draw and undraw), from the real views."""
        if a in MOVES:
            cat = ENDED if end else (CHANGED if (V[:NV] != V2[:NV]).any() else UNCHANGED)
            self.kinds[a].add((V[FRONT],), cat)
            if a == FWD and cat != UNCHANGED:
                self.kinds["draw"].add((V[FRONT],), 0, after=(V2[CENTRE],))
                self.kinds["undraw"].add((V[CENTRE],), 0, after=(V2[BEHIND],))
            self.clear_lazy()
            return cat
        f0, h0, f1, h1 = int(V[FRONT]), int(V[HELD]), int(V2[FRONT]), int(V2[HELD])
        cat = int(f1 != f0) + 2 * int(h1 != h0)
        self.kinds[a].add((f0, h0, self.ctx_of_view(V)), cat, after=(f1, h1))
        self.openc.clear()
        self.openedc.clear()
        return cat


# ---------------------------------------------------------------- the planner (card 029's, on vectors)

class Ach:
    __slots__ = ("a", "u", "j")

    def __init__(self, a, u, j):
        self.a, self.u, self.j = a, u, j


class VPlan(G.Plan):
    """Card 029's working backward, walking and revision, with the facts as handles of vectors, every
    condition a prediction, and conditions of an action inferred from memory (conditions())."""

    def __init__(self, W):
        F.Think.__init__(self, [-1], [-1], True)
        self.W = W
        self.failed, self.refused, self.version = set(), set(), 0
        self.psets, self.pids = [], {}
        self.fmemo, self.amemo, self.rmemo, self.cmemo, self.canmemo, self.premo = {}, {}, {}, {}, {}, {}
        self.achmemo, self.condmemo, self.imemo, self.ctxmemo = {}, {}, {}, {}
        self.n_refused, self.walk_kind = 0, Counter()

    def forget(self):
        """The model changed (a try was added): drop everything derived from its predictions."""
        for d in (self.steps, self.memo, self.rsets, self.appr, self.pres, self.fmemo, self.amemo, self.rmemo,
                  self.cmemo, self.canmemo, self.premo, self.achmemo, self.condmemo, self.imemo):
            d.clear()

    def intern(self, f):
        b = f.tobytes()
        i = self.fid.get(b)
        if i is None:
            i = self.fid[b] = len(self.facts)
            f = np.array(f, np.int32)
            f.flags.writeable = False
            self.facts.append(f)
        return i

    # -- seeing
    def observe(self, world, V, ended=False, prefer=None):
        """Place the view: the pose whose predicted view has the least total L1 distance to it (the
        agent's own place aside); then the facts from it, the agent's own tile undrawn."""
        M = F.M
        V = np.asarray(V, np.int32)
        if world is None:
            f = V.copy()
            f[CENTRE] = M.undraw[V[CENTRE]]
            pose, miss, ties = 0, 0.0, 1
        else:
            Gv = np.where(M.A >= 0, world[np.maximum(M.A, 0)], M.Ent).astype(np.int64)
            u, inv = np.unique(np.concatenate([Gv.ravel(), V]), return_inverse=True)
            inv = inv.ravel()
            Z = self.W.S.arr[u]
            Dm = np.abs(Z[:, None] - Z[None]).sum(-1)
            d = Dm[inv[:Gv.size].reshape(Gv.shape), inv[Gv.size:][None]]
            d[:, CENTRE] = 0.0
            dist = d.sum(1)
            best = dist.min()
            tied = np.flatnonzero(dist <= best + 1e-9)
            pose = prefer if prefer is not None and (tied == prefer).any() else int(tied[0])
            f = np.array(world, np.int32)
            Ap = M.A[pose]
            sel = Ap >= 0
            sel[CENTRE] = False
            f[Ap[sel]] = V[sel]
            f[Ap[CENTRE]] = M.undraw[V[CENTRE]]
            miss, ties = float(best), len(tied)
        return f, (self.intern(f), int(pose), bool(ended)), miss, ties

    def view(self, st):
        M = F.M
        f = self.facts[st[0]]
        A = M.A[st[1]]
        v = np.where(A >= 0, f[np.maximum(A, 0)], M.Ent[st[1]]).astype(np.int32)
        v[CENTRE] = M.draw[v[CENTRE]]
        return v

    def ctx_id(self, fid, pose):
        """What is in view from this pose (the agent's own place aside), as one vector's handle."""
        k = (fid, pose)
        r = self.ctxmemo.get(k)
        if r is None:
            M = F.M
            A = M.A[pose]
            v = np.where(A >= 0, self.facts[fid][np.maximum(A, 0)], M.Ent[pose])[:NV]
            r = self.ctxmemo[k] = self.W.ctx_of(np.unique(v[self.W.keep]))
        return r

    # -- the learned effects
    def front_held(self, st):
        M = F.M
        f = self.facts[st[0]]
        fi = M.fidx[st[1]]
        return (int(f[fi]) if fi >= 0 else int(M.efront[st[1]])), int(f[HELD])

    def step(self, st, a):
        k = (st, a)
        r = self.steps.get(k)
        if r is not None:
            return r
        M = F.M
        fid, pose, ended = st
        if ended:
            r = st
        elif a in MOVES:
            u, _ = self.front_held(st)
            o = M.move_out[a][u]
            p2 = M.nxt[pose][a]
            r = st if o == UNCHANGED or p2 < 0 else (fid, p2, o == ENDED)
        else:
            u, h = self.front_held(st)
            c, (fa, ha) = self.W.outcome(a, u, h, self.ctx_id(fid, pose))
            if c == 0:
                r = st
            else:
                b = self.facts[fid].copy()
                fi = M.fidx[pose]
                if fa is not None and fi >= 0:
                    b[fi] = fa
                if ha is not None:
                    b[HELD] = ha
                r = (self.intern(b), pose, False)
        self.steps[k] = r
        return r

    def with_tile(self, st, i, u):
        b = self.facts[st[0]].copy()
        b[i] = u
        return (self.intern(b), st[1], st[2])

    def imagine_place(self, st, u, j):
        """Place j, else the place in front if it shows u, else the first place showing u (None: none)."""
        if j is not None:
            return j
        f = self.facts[st[0]]
        fi = F.M.fidx[st[1]]
        if 0 <= fi < NV and f[fi] == u:
            return fi
        where = np.flatnonzero(f[:NV] == u)
        return int(where[0]) if len(where) else None

    def imagine(self, st, a, u, j, h=None, v=None):
        """The facts after doing a on thing u (at place j, else at a place showing it), with h held and v
        in view (None: as now); None when nothing is predicted to change."""
        k = (st, a, u, j, h, v)
        if k in self.imemo:
            return self.imemo[k]
        f = self.facts[st[0]]
        j = self.imagine_place(st, u, j)
        r = None
        if j is not None:
            c, (fa, ha) = self.W.outcome(a, int(f[j]), int(f[HELD]) if h is None else h,
                                         self.ctx_id(st[0], st[1]) if v is None else v)
            if c:
                b = f.copy()
                if fa is not None:
                    b[j] = fa
                if ha is not None:
                    b[HELD] = ha
                r = (self.intern(b), st[1], False)
        self.imemo[k] = r
        return r

    # -- conditions
    def hold(self, c, st):
        self.computed += 1
        k = c[0]
        if k == "end":
            return st[2]
        if k == "face":
            return st[1] in self.psets[self.face_pid(st[0], c)]
        if k == "walk":
            return bool(F.M.free[self.facts[st[0]][c[1]]])
        if k == "part":
            _, p, a, u, j, c2, o = c
            return self.does(st, a, u, j, c2, None, o) if p == HELDP else self.does(st, a, u, j, c2, o, None)
        raise ValueError(c)

    def face_pid(self, fid, need):
        """Poses facing a place the need names (place j, or any place showing exactly u), standing on a
        place one can walk onto, without refused places or failed tries (with the same held thing)."""
        k = (fid, need, self.version)
        r = self.fmemo.get(k)
        if r is None:
            _, a, u, j = need
            M = F.M
            f = self.facts[fid]
            held = int(f[HELD])
            T_ = [j] if j is not None else np.flatnonzero(f[:NV] == u).tolist()
            P = [p for i in T_ if (fid, a, i) not in self.refused for p in M.facing[i]
                 if M.free[f[M.cidx[p]]] and (a, i, p, held) not in self.failed]
            r = self.fmemo[k] = self.intern_p(P)
        return r

    def standing(self, fid):
        k = ("standing", fid)
        r = self.canmemo.get(k)
        if r is None:
            f = self.facts[fid]
            M = F.M
            r = self.canmemo[k] = [p for p in M.all_poses.tolist() if M.free[f[M.cidx[p]]]]
        return r

    def does(self, st, a, u, j, c, h=None, v=None):
        """Doing a on thing u (at place j) makes c true, with h held and v in view (None: as now)."""
        s2 = self.imagine(st, a, u, j, h, v)
        return s2 is not None and self.hold(c, s2)

    def conditions(self, a, u, j, c, st, bit):
        """The ways doing a on thing u (at place j) can make the unmet condition c true, each a list of
        needs. If it does so with what is held and in view now: one way, no needs. Otherwise from memory:
        the (held, view) pairs of the kind's stored tries on things like u (k >= KMIN on the front part)
        are split into successes (doing a on u with that pair makes c true) and failures. For a success,
        a part is a condition when doing a on u fails with that part as now and the other part as in the
        pair: ("part", p, a, u, j, c, the pair's other part). Successes needing the same parts are one way,
        the pair nearest the present one (recall's metric) standing for them; a success needing no part
        explains nothing and is left out. bit: the place c depends on (1 front, 2 held); a pair whose
        predicted category leaves it unchanged cannot make c true and is not checked."""
        k = (st[0], st[1], a, u, j, c)
        r = self.condmemo.get(k)
        if r is not None:
            return r
        r = []
        if self.does(st, a, u, j, c):
            r = [[]]
        elif self.imagine_place(st, u, j) is not None:
            kd = self.W.kinds[a]
            hc, vc = int(self.facts[st[0]][HELD]), self.ctx_id(st[0], st[1])
            pairs = kd.templates(u)
            cats = kd.cat_of([(u, h, v) for h, v in pairs])
            best = {}
            for (h, v), cat in zip(pairs, cats):
                if not cat or not cat & bit or not self.does(st, a, u, j, c, h, v):
                    continue
                needs = []
                if not self.does(st, a, u, j, c, None, v):
                    needs.append(("part", HELDP, a, u, j, c, v))
                if not self.does(st, a, u, j, c, h, None):
                    needs.append(("part", VIEWP, a, u, j, c, h))
                if not needs:
                    continue
                sig = tuple(n[1] for n in needs)
                d = kd.pair_distance((h, v), (hc, vc))
                if sig not in best or d < best[sig][0]:
                    best[sig] = (d, needs)
            r = [best[s][1] for s in sorted(best)]
        self.condmemo[k] = r
        return r

    def achievers(self, c, st):
        """The episode ends: forward onto a thing predicted to end it. "Place j can be walked onto": pick up
        or toggle the thing at j. A part condition: a pick up, toggle or drop on a thing in the facts (other
        than the action and thing the condition is about) whose predicted effect changes that part (the held
        thing, or a thing in view) so that the condition holds. Each achiever's needs are its conditions
        (conditions()), then facing its thing."""
        key = (c, st[0], st[1])
        r = self.achmemo.get(key)
        if r is not None:
            return r
        f = self.facts[st[0]]
        if c[0] == "end":
            r = [(Ach(FWD, u, None), [("face", FWD, u, None)])
                 for u in np.unique(f[:NV]).tolist() if self.W.move_cat(FWD, u) == ENDED]
        else:
            if c[0] == "walk":
                cands, bit = [(a, int(f[c[1]]), c[1]) for a in (PICK, TOG)], 1
            else:
                cands = [(a, u, None) for a in INTER for u in np.unique(f[:NV]).tolist() if (a, u) != (c[2], c[3])]
                bit = 2 if c[1] == HELDP else 1
            r = []
            for a, u, j in cands:
                for needs in self.conditions(a, u, j, c, st, bit):
                    r.append((Ach(a, u, j), needs + [("face", a, u, j)]))
        self.achmemo[key] = r
        return r

    def walk(self, need, st, depth, chain, protect, faces):
        pid = self.face_pid(st[0], need)
        r, kind = self.reach(st, pid)
        if r is not None and r != G.HERE:
            return G.Result(r, [need, ("walk", kind)], self.closeness(st, self.psets[pid]))
        if depth + 1 > G.MAXD:
            return None
        # a place in the way, showing a thing memory has seen made into one to walk onto: would walking
        # reach the need if it could be walked onto? (checked with the thing the agent stands on there)
        f = self.facts[st[0]]
        M = F.M
        stand = int(f[M.cidx[st[1]]])
        if not M.free[stand]:
            return None
        cands = []
        for u in np.unique(f[:NV]).tolist():
            if M.free[u] or not self.W.openable(u):
                continue
            for j in np.flatnonzero(f[:NV] == u).tolist():
                st2 = self.with_tile(st, j, stand)
                pid2 = self.face_pid(st2[0], need)
                if not (self.connected(st2) & self.psets[pid2]):
                    continue
                if self.reach(st2, pid2)[0] is None:
                    continue
                cands.append((self.closeness(st, M.facing[j]) or (99, 9), j))
        for _, j in sorted(set(cands)):
            c = ("walk", j)
            if c in chain:
                continue
            res = self.solve(c, st, depth + 1, chain + [need], protect, faces + [need])
            if res is not None:
                res.trace = [need, ("walk", "blocked")] + res.trace
                return res
        return None


# ---------------------------------------------------------------- evaluation of effects

def _eval_job(job):
    V0, V1, act, term1 = job
    W = WORLD
    th = VPlan(W)
    ok = np.zeros(len(act), bool)
    errs = Counter()
    for i in range(len(act)):
        _, st, _, _ = th.observe(None, V0[i])
        nxt = th.step(st, int(act[i]))
        pv = th.view(nxt)
        bad = [q for q in np.flatnonzero(pv != V1[i]) if not W.judge.right(pv[q], V1[i, q])]
        ok[i] = not bad and nxt[2] == bool(term1[i])
        if not ok[i]:
            errs[(KNAME[int(act[i])], W.judge.name(V0[i, FRONT]), W.judge.name(V0[i, HELD]),
                  tuple((int(q), W.judge.name(pv[q]), W.judge.name(V1[i, q])) for q in bad[:3]),
                  nxt[2], bool(term1[i]))] += 1
        if len(th.facts) > 20000:
            th = VPlan(W)
    return ok, errs


def eval_rows(pool, lut, E0, E1, act, term1):
    if len(act) == 0:
        return np.zeros(0, bool), Counter()
    V0, V1 = lut[E0].astype(np.int32), lut[E1].astype(np.int32)
    chunks = [c for c in np.array_split(np.arange(len(act)), 40) if len(c)]
    parts = pool.map(_eval_job, [(V0[c], V1[c], act[c], term1[c]) for c in chunks])
    errs = Counter()
    for _, e in parts:
        errs.update(e)
    return np.concatenate([p[0] for p in parts]), errs


# ---------------------------------------------------------------- acting, with online learning

def see(lay, s):
    return F.see(lay, s).astype(np.int32)


def _act_job(job):
    idx, layouts, online = job
    W = WORLD
    M = F.M
    W.reset()                                   # every chunk starts from the memory before acting
    out = []
    for pos, (i, lay) in enumerate(zip(idx, layouts)):
        t0 = time.monotonic()
        rng = np.random.default_rng(1000 + int(i))
        pl = VPlan(W)
        s = ld.start_state(lay)
        V = see(lay, s)
        facts, st, _, _ = pl.observe(None, V)
        rec = {"position": pos, "done": False, "steps": BUDGET, "random": 0, "pred_wrong": 0, "failed_acts": 0,
               "max_mismatch": 0.0, "walk": Counter()}
        for t in range(BUDGET):
            res = pl.choose(st)
            if res is None:
                a = int(rng.integers(6))
                rec["random"] += 1
            else:
                a = res.action
                rec["walk"]["act" if res.trace[-1][0] == "act" else res.trace[-1][1]] += 1
            pred = pl.step(st, a)
            u, held = pl.front_held(st)
            if a in MOVES:
                pcat = M.move_out[a][u]
            else:
                pcat = W.outcome(a, u, held, pl.ctx_id(st[0], st[1]))[0]
            s2, end = ld.step(lay, s, a)
            V2 = see(lay, s2)
            facts, st2, miss, _ = pl.observe(facts, V2, end, prefer=pred[1])
            rec["max_mismatch"] = max(rec["max_mismatch"], miss)
            if a in MOVES:
                rcat = ENDED if end else (CHANGED if (V[:NV] != V2[:NV]).any() else UNCHANGED)
            else:
                rcat = int(V[FRONT] != V2[FRONT]) + 2 * int(V[HELD] != V2[HELD])
            if rcat != pcat:
                rec["pred_wrong"] += 1
                if a not in MOVES:
                    pl.failed.add((a, M.fidx[st[1]], st[1], held))      # card 029's rule, with the held thing
                    pl.version += 1
                    rec["failed_acts"] += 1
            if online:
                W.learn_try(V, a, V2, end)
                pl.forget()
            s, st, V = s2, st2, V2
            if end:
                rec["done"], rec["steps"] = True, t + 1
                break
        rec["refused"] = pl.n_refused
        rec["computed"] = pl.computed
        rec["seconds"] = time.monotonic() - t0
        out.append(rec)
    return out


def _act_indexed(job):
    return job[0], _act_job(job[1])


def act_arm(pool, test, online=True):
    chunks = [c.tolist() for c in np.array_split(np.arange(len(test)), 40) if len(c)]
    parts, n = [None] * len(chunks), 0
    progress(layouts_done=0, layouts_total=len(test))
    jobs = [(k, (c, [test[i] for i in c], online)) for k, c in enumerate(chunks)]
    for k, p in pool.imap_unordered(_act_indexed, jobs):          # progress as chunks finish; order kept
        parts[k] = p
        n += len(p)
        progress(layouts_done=n, goal_so_far=sum(r["done"] for q in parts if q for r in q))
    recs = [r for p in parts for r in p]
    done = np.array([r["done"] for r in recs])
    steps = np.array([r["steps"] for r in recs])
    moves = int(steps.sum())
    walk = Counter()
    curve = {}
    for r in recs:
        walk.update(r["walk"])
        c = curve.setdefault(r["position"], [0, 0, 0])
        c[0] += 1
        c[1] += r["done"]
        c[2] += r["pred_wrong"]
    res = {"success": round(float(done.mean()), 4),
           "mean_steps_when_successful": round(float(steps[done].mean()), 2) if done.any() else None,
           "random_share": round(sum(r["random"] for r in recs) / max(moves, 1), 4), "moves": moves,
           "predictions_wrong": int(sum(r["pred_wrong"] for r in recs)),
           "failed_acts": int(sum(r["failed_acts"] for r in recs)),
           "actions_refused": int(sum(r["refused"] for r in recs)),
           "steps_by_kind": dict(walk),
           "largest_view_distance_placing_the_agent": round(float(max(r["max_mismatch"] for r in recs)), 4),
           "conditions_computed_per_move": round(sum(r["computed"] for r in recs) / max(moves, 1), 1),
           "seconds_per_layout_mean": round(float(np.mean([r["seconds"] for r in recs])), 3),
           "seconds_per_layout_max": round(float(np.max([r["seconds"] for r in recs])), 2),
           "tries_curve": {int(p): {"layouts": v[0], "success": round(v[1] / v[0], 4),
                                    "wrong_predictions_per_layout": round(v[2] / v[0], 3)}
                           for p, v in sorted(curve.items())}}
    return res, recs


# ---------------------------------------------------------------- reports

def ways_report(W):
    out = {}
    for name in ("draw", "undraw", PICK, TOG, DROP):
        kd = W.kinds[name]
        cats = [0] if not kd.pair else [c for c in range(1, 4) if (kd.aw[:, c] > 0).any()]
        for c in cats:
            out[f"{kd.name} {c}"] = {("result" if not kd.pair else "front" if p == 0 else "held"):
                                     "".join(WAYS[x][0] for x in v) for p, v in kd.ways(c).items()}
    return out


def new_tile_report(W):
    """The new tiles' forward votes and looks, from memory before acting (evaluator names)."""
    J = W.judge
    out = {}
    for nmk, c in KP.new_tiles().items():
        h = int(W.S.hof[c])
        kd = W.kinds[FWD]
        out[nmk] = {"forward": {"vote": round(kd.vote((h,)), 3),
                                "category": {CHANGED: "moved", UNCHANGED: "blocked", ENDED: "ended",
                                             None: "unknown"}[kd.move((h,))]},
                    "draw": J.name(W.M.draw[h]), "undraw": J.name(W.M.undraw[h])}
    return out


# ---------------------------------------------------------------- one arm (one set of vectors) in every world

def run_vectors(label, z, parts, data, tests, dev, log, ref, worlds=WORLDS, layouts=None, online=True):
    """Learn, check and act in the familiar worlds and in the new-colour worlds."""
    global WORLD
    S = Store(z)
    lut = np.zeros(256, np.uint8)
    lut[:len(S.hof)] = S.hof
    Kd.APP = lut
    out = {"handles_of_tiles": int(S.nobs)}
    try:
        for world in worlds:
            t0 = time.monotonic()
            D = data[world]
            progress(world=world, stage="model and held-out effects", layouts_done=None, layouts_total=None)
            r = out[world] = {}
            W = World(S, parts, D, dev, log)
            WORLD, F.M = W, W.M
            r["model"] = W.report
            pool = F.pool20()
            ok, errs = eval_rows(pool, lut, D["hego0"], D["hego1"], D["hact"], D["hterm1"])
            log(f"  {label} {world}: model {W.seconds}s, held-out effects {ok.mean():.5f} "
                f"({time.monotonic() - t0:.0f}s)")
            pa = CO.per_action(ok, D["hact"])
            r["heldout_effects"] = {"per_action": pa, "all": round(float(ok.mean()), 6),
                                    "lowest": min(v["exact_share"] for v in pa.values()),
                                    "errors_top": [[list(map(str, k)), v] for k, v in errs.most_common(6)]}
            test = D["test"] if layouts is None else D["test"][:layouts]
            progress(stage="acting, familiar layouts")
            res, _ = act_arm(pool, test, online)
            pool.close()
            r["acting"] = res
            r["acting"]["steps_ratio_to_shortest"] = round((res["mean_steps_when_successful"] or 1e9) / D["shortest"], 3)
            r["acting"]["steps_ratio_to_card029"] = round((res["mean_steps_when_successful"] or 1e9) / ref[world], 3)
            r["criterion_1"] = {"effects_right": r["heldout_effects"]["lowest"] >= 0.999,
                                "success_99": res["success"] >= 0.99,
                                "steps_within_5pct": r["acting"]["steps_ratio_to_card029"] <= 1.05}
            r["criterion_1"]["pass"] = all(r["criterion_1"].values())
            if world == "key":
                r["ways"] = ways_report(W)
                if CO.TESTS:
                    r["new_tiles"] = new_tile_report(W)
            for tk, (tw, door_open) in CO.TESTS.items():
                if tw != world:
                    continue
                X = tests[tk]
                q = r[f"test_{tk}"] = {}
                CO.patch(door_open=bool(door_open))
                pool = F.pool20()
                progress(stage=f"test {tk}: effects", layouts_done=None, layouts_total=None)
                ok, errs = eval_rows(pool, lut, X["ego0"], X["ego1"], X["act"], X["term1"])
                progress(stage=f"test {tk}: acting")
                cap = LAYOUTS_B if tw == "switch" and LAYOUTS_B else layouts          # world (b): reported only
                res, _ = act_arm(pool, X["test"] if cap is None else X["test"][:cap], online)
                pool.close()
                CO.patch()
                q["cases"] = {k: {"rows": int(len(rows)),
                                  "right_share": round(float(ok[rows].mean()), 4) if len(rows) else None}
                              for k, rows in X["cases"].items()}
                q["errors_top"] = [[list(map(str, k)), v] for k, v in errs.most_common(8)]
                q["acting"] = res
                q["acting"]["steps_ratio_to_shortest"] = round((res["mean_steps_when_successful"] or 1e9) / X["shortest"], 3)
                crit = [v["right_share"] for k, v in q["cases"].items() if k in X["criterion_cases"]]
                q["criterion_2"] = {"lowest_case": min((x if x is not None else 0.0 for x in crit), default=None),
                                    "pass": bool(crit) and all(x is not None and x >= 0.99 for x in crit)}
                q["criterion_3"] = {"success_98": res["success"] >= 0.98,
                                    "steps_within_1_15x": q["acting"]["steps_ratio_to_shortest"] <= 1.15}
                q["criterion_3"]["pass"] = all(q["criterion_3"].values())
            r["seconds"] = round(time.monotonic() - t0, 1)
            log(f"  {label} {world}: model {W.seconds}s; effects lowest {r['heldout_effects']['lowest']}, acting "
                f"{r['acting']['success']} ({r['acting']['mean_steps_when_successful']} steps, "
                f"{r['acting']['seconds_per_layout_mean']}s/layout); "
                + "; ".join(f"test {tk}: cases lowest {r[f'test_{tk}']['criterion_2']['lowest_case']}, acting "
                            f"{r[f'test_{tk}']['acting']['success']}" for tk in CO.TESTS if f"test_{tk}" in r)
                + f" ({r['seconds']}s)")
    finally:
        Kd.APP = APP0
    return out


# ---------------------------------------------------------------- arm B's encoder: card 035's recipe without codebooks

def train_b(tiles, pairs, groups, mu, seed, dev, K=4, dim=8, pair_w=0.1, updates=5000):
    """Card 035's encoder (card 033's train_encoder, mode "own") without the codebook, commitment and
    re-seeding terms: the decoder reads the unit-length pieces. Rebuild + pair term (adaptive, 2002_02886)
    + mu x recall's leave-one-out term, lambda learned alongside."""
    import torch
    import torch.nn as nn
    fn = torch.nn.functional
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.manual_seed(seed)
    x_all = torch.as_tensor(CO.TILES / 255.0, dtype=torch.float32, device=dev).permute(0, 3, 1, 2)
    enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                        nn.ReLU(), nn.Flatten(), nn.Linear(32 * 16, K * dim)).to(dev)
    dec = nn.Sequential(nn.Linear(K * dim, 32 * 16), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)),
                        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()).to(dev)
    params = list(enc.parameters()) + list(dec.parameters())
    tc = torch.as_tensor(tiles, device=dev)
    pa = torch.as_tensor([p[0] for p in pairs], device=dev, dtype=torch.long)
    pb = torch.as_tensor([p[1] for p in pairs], device=dev, dtype=torch.long)

    def pieces(x):
        return fn.normalize(enc(x).reshape(len(x), K, dim), dim=-1)

    with torch.no_grad():
        X0, _, M0 = R33.stack(pieces(x_all).reshape(len(x_all), -1), groups, "own")
        theta = nn.Parameter(R33.init_theta(X0, M0))
    opt = torch.optim.Adam(params + [theta], lr=1e-3)
    t0 = time.monotonic()
    for _ in range(updates):
        x = x_all[tc]
        z = pieces(x)
        rebuild = fn.mse_loss(dec(z.reshape(len(x), -1)), x)
        za, zb = pieces(x_all[pa]), pieces(x_all[pb])
        d = (za - zb).norm(dim=-1)
        mid = (d.amax(1, keepdim=True) + d.amin(1, keepdim=True)).detach() / 2
        pair = (d * (d < mid)).sum(1).mean()
        X, Rr, Mk = R33.stack(pieces(x_all).reshape(len(x_all), -1), groups, "own")
        replay = -R33.loo_loglik(X, Rr, Mk, fn.softplus(theta)).sum()
        loss = rebuild + pair_w * pair + mu * replay
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z_all = pieces(x_all).reshape(len(x_all), -1).cpu().numpy()
        rep = {"rebuild_mse": round(float(rebuild), 6), "pair_term": round(float(pair), 4),
               "replay_term": round(float(replay), 4), "seconds": round(time.monotonic() - t0, 1)}
    return z_all, theta.detach().cpu().numpy(), rep


def encoder_b(seed, tiles, pairs, groups, dev, log):
    f = ENC_B / f"mu{MU}_s{seed}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {"z": d["z"], "theta": d["theta"], "rep": json.loads(str(d["rep"]))}
    ENC_B.mkdir(parents=True, exist_ok=True)
    z, th, rep = train_b(tiles, pairs, groups, MU, seed, dev)
    np.savez(f, z=z, theta=th, rep=json.dumps(rep))
    log(f"encoder B {seed}: {rep}")
    return {"z": z, "theta": th, "rep": rep}


def vectors_of(arm, seed, tiles, pairs, groups, dev, log):
    """(z for every tile code, parts, encoder report) for an arm."""
    if arm == "1":
        z, parts = oracle_vectors()
        return z, parts, {"oracle": True}
    parts = [slice(8 * k, 8 * k + 8) for k in range(4)]
    if arm == "A":
        N.ENC = ENC_A
        e = N.encoder(MU, seed, tiles, pairs, groups, dev, log)
    else:
        e = encoder_b(seed, tiles, pairs, groups, dev, log)
    return np.asarray(e["z"], np.float64), parts, e["rep"]


# ---------------------------------------------------------------- main

def main():
    args = sys.argv[1:]
    flag = lambda f: f in args
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    T.configure()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda msg: print(f"[{time.monotonic() - t00:6.0f}s] {msg}", flush=True)
    cache = pickle.loads(T.MEMORY.read_bytes())
    mem, tiles, pairs = cache["mem"], cache["tiles"], cache["pairs"]
    groups = T.use_groups(mem, dev)
    a_, b_ = get("--seeds", "400-409").split("-")
    seeds = tuple(range(int(a_), int(b_) + 1))

    if flag("--encoders"):
        for seed in seeds:
            encoder_b(seed, tiles, pairs, groups, dev, log)
        return

    if flag("--gate"):
        out = ROOT / "runs" / "038_gate.json"
        res = {"seeds": []}
        N.ENC = ENC_A
        a_mse = [N.encoder(MU, s, tiles, pairs, groups, dev, log)["rep"]["rebuild_mse"] for s in range(400, 410)]
        lim = 2 * float(np.median(a_mse))
        app = APP0[np.asarray(tiles)]
        for seed in range(400, 410):
            e = encoder_b(seed, tiles, pairs, groups, dev, log)
            zt = e["z"][np.asarray(tiles)]
            d = np.abs(zt[:, None] - zt[None]).sum(-1)
            d[app[:, None] == app[None]] = np.inf
            res["seeds"].append({"seed": seed, "encoder": e["rep"],
                                 "smallest_distance_between_tiles": round(float(d.min()), 4),
                                 "distinct": bool(d.min() > 1e-6), "rebuild_ok": e["rep"]["rebuild_mse"] <= lim})
        res["arm_a_median_rebuild"], res["limit"] = float(np.median(a_mse)), lim
        res["pass"] = all(s["distinct"] and s["rebuild_ok"] for s in res["seeds"])
        out.write_text(json.dumps(res, indent=1) + "\n")
        log(f"gate: {json.dumps(res)}")
        return

    d = pickle.loads(T.DATA.read_bytes())
    data, tests = d["data"], d["tests"]
    ref = json.loads(T.REF.read_text())
    arm = get("--arm", "A")
    layouts = int(get("--layouts", "0")) or None
    global LAYOUTS_B
    LAYOUTS_B = int(get("--layouts-b", "0")) or None

    if flag("--dev"):                   # shakedown on a spare seed: familiar worlds only
        CO.TESTS = {}
        z, parts, _ = vectors_of(arm, seeds[0], tiles, pairs, groups, dev, log)
        worlds = tuple(get("--worlds", ",".join(WORLDS)).split(","))
        o = run_vectors(f"dev arm {arm} seed {seeds[0]}", z, parts, data, tests, dev, log, ref, worlds, layouts)
        dump = Path(get("--out", "runs/038_dev.json"))
        dump.write_text(json.dumps(o, indent=1, default=str) + "\n")
        for w in worlds:
            x = o[w]
            print(json.dumps({"world": w, "criterion_1": x["criterion_1"], "effects": x["heldout_effects"]["per_action"],
                              "errors_top": x["heldout_effects"]["errors_top"][:4],
                              "acting": {k: v for k, v in x["acting"].items() if k != "tries_curve"},
                              "kinds": x["model"]["kinds"]}, default=str))
        return

    out = Path(get("--out", f"runs/038_arm{arm}.json"))
    res = {"note": "Card 038, tools/card038/vector_planner.py", "arm": arm, "seeds": []}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    global PROG
    run_seeds = seeds[:1] if arm == "1" else seeds
    PROG = {"path": out.with_suffix(".progress.json"), "data": {}}
    progress(arm=arm, seeds=list(run_seeds), out=str(out), pid=os.getpid(),
             started=time.time(), layouts_b=LAYOUTS_B, seeds_done=[], seed_seconds=[], finished=False)
    for seed in run_seeds:
        ts = time.monotonic()
        progress(seed=seed, seed_started=time.time())
        z, parts, erep = vectors_of(arm, seed, tiles, pairs, groups, dev, log)
        oc = run_vectors(f"arm {arm} seed {seed}", z, parts, data, tests, dev, log, ref, WORLDS, layouts)
        o = {"seed": seed, "encoder": erep, "worlds": oc, "verdicts": T.verdicts(oc)}
        res["seeds"].append(o)
        log(f"arm {arm} seed {seed}: verdicts {o['verdicts']}")
        save()
        progress(seeds_done=PROG["data"]["seeds_done"] + [seed],
                 seed_seconds=PROG["data"]["seed_seconds"] + [round(time.monotonic() - ts, 1)])
    res["seeds_passing"] = {c: sum(s["verdicts"][c] for s in res["seeds"]) for c in ("criterion_1", "criterion_2", "criterion_3")}
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    progress(finished=True, stage="finished")
    log(f"done: {res['seeds_passing']} -> {out}")


if __name__ == "__main__":
    main()
