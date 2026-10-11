"""Card 098.5: object files, the second level of the placing step. Tokens are (vector, position); positions are
continuous, in the frame placing gives (level 1: the agent's own motion), so nothing here reads a grid.

Per step, after level 1 has placed the view:
  - changed places: observed places whose vector is beyond the identity radius of what belief holds there (or where
    belief holds nothing);
  - each file takes one changed place showing a token within the identity radius of its own, among the positions
    where it could be, in one assignment for all files (cost in nats: vector distance / radius plus the cost of the
    position);
  - where a file could be (revision, after the gate):
      seen last step: its last position plus each displacement it can make, at cost
      -log P(displacement | its kind, the displacement it last made), counted from every file's history (long-term,
      like the kinds; add-one over the displacements it can make). It can stay, move as the agent's own steps move
      (the prior before its kind's motion is learned), or make any displacement its kind has been seen to make;
      unseen: a probability over where it is and the displacement it last made, carried forward each step by its
      kind's motion (the same counts) and set to zero at every place the view shows without it, then renormalised
      (a grid Bayes filter with negative information, as in search theory); cost -log P(here). It is held at half a
      step length's resolution, within 20 step lengths of where the file was last seen. A file with no probability
      left is dropped: every place it could be was seen without it;
  - or an object not yet seen: each changed place can instead go to a new object, at cost -log(the density of
    unseen objects of its kind there). Per episode and kind, that density is one more belief like a file's: it starts
    at how often a place shows a token of the kind when first seen (long-term; add-one), is carried forward by the
    kind's motion, and is set to zero wherever the view shows a place (an object there is seen; one that is new gets
    a file), plus the rate at which tokens of the kind appear out of nothing (an arrow loosed, a zombie spawned): how
    often a place shows the kind where, the step before, the view showed every place within the kind's reach and
    none of the kind (long-term; add-one; read from the view, not from the files). This is the intensity of
    undetected objects, with births, in multi-target tracking (Mahler's PHD filter; Reid 1979): a file is taken up
    where it is likelier to be than an object not yet seen;
  - births: a changed place whose token vanished (belief held a token, the view shows another) and a changed place
    within a step length showing a token like the vanished one: the token moved by itself (pair birth); a changed
    place showing a token like one that has had a file, with no source: a file with no motion yet (kind birth);
  - figure and ground (accretion and deletion of texture at occluding edges: Gibson 1966, Kaplan 1969): a move is
    locally the same as the ground moving the other way (a cow steps onto grass: "cow here" or "grass there"). A pair
    counts only when the token that arrived does not continue the surface around it (most of its unchanged
    neighbours within a step length unlike it) and the place it left now shows a token that does (the ground revealed);
  - places a file occupies are "explained": belief keeps what it held there (what lies under the moving token).

The first build (WM0985_FIRST=1, the gate's first run): inertia as the expected position (cost: distance from it in
step lengths), reach growing by a step length per step unseen, dropped when the whole reach lies in the view.
The agent's own place and carried tokens are not given to this level.
"""
import os
from collections import Counter, defaultdict

import numpy as np
from assign import linear_sum_assignment

BIG = 1e9
FIRST = os.environ.get("WM0985_FIRST") == "1"
FAR = 20                                                # step lengths: how far belief about an unseen file extends
WIDE = 48                                               # step lengths: the unseen objects' density, from the start


class Files:
    def __init__(self, dist, radius, step_len, own=()):
        self.dist, self.radius, self.step_len = dist, float(radius), float(step_len)
        self.h = self.step_len / 2                      # the resolution of positions in belief about unseen files
        self.R = 2 * FAR
        self.own = {self.key(np.zeros(2))} | {self.key(d) for d in own}
        self.kinds = None                               # long-term: vectors of tokens that have had a file (leader)
        self.motion = defaultdict(Counter)              # long-term: (kind, last displacement or None) -> next one
        self.fresh, self.kind_fresh = 0, Counter()      # long-term: places first seen, and those showing each kind
        self.could, self.appeared = Counter(), Counter()   # long-term, per kind: places it could appear from nothing
        self.U = 2 * WIDE
        self.next_id = 0
        self.stats = {"pair_births": 0, "kind_births": 0, "matched": 0, "dropped": 0}
        self.reset()

    def reset(self):
        self.files = []                                 # one episode's files
        self.unseen, self.anchor = {}, None             # one episode's density of unseen objects per kind
        self.last = None                                # last step's places (keys, kinds)

    def birth(self, kind):
        return (self.appeared[kind] + 1) / (self.could[kind] + 2)

    def density0(self, kind):
        return (self.kind_fresh[kind] + 1) / (self.fresh + 2)

    def unseen_map(self, kind):
        if kind not in self.unseen:
            self.unseen[kind] = np.full((2 * self.U + 1,) * 2, self.density0(kind), np.float32)
        return self.unseen[kind]

    # ------------------------------------------------------------------ kinds and their motion
    def is_kind(self, V):
        if self.kinds is None or not len(V):
            return np.zeros(len(V), bool)
        return self.dist(V, self.kinds).min(1) <= self.radius

    def kind_of(self, v, add=False):
        """The kind a token belongs to (the nearest within the identity radius); a new kind if add, else -1."""
        if self.kinds is not None:
            d = self.dist(v[None], self.kinds)[0]
            if d.min() <= self.radius:
                return int(d.argmin())
        if not add:
            return -1
        self.kinds = v[None].copy() if self.kinds is None else np.concatenate([self.kinds, v[None]])
        return len(self.kinds) - 1

    def key(self, d):
        """A displacement in half step lengths (integers)."""
        return tuple(np.round(np.asarray(d, np.float64) / self.h).astype(int).tolist())

    def keys(self, D):
        return np.round(np.asarray(D, np.float64).reshape(-1, 2) / self.h).astype(np.int64)

    def support(self, kind):
        """The displacements a file of this kind can make, in half step lengths."""
        return sorted(self.own | set(self.motion.get((kind, None)) or ()))

    def chain(self, kind):
        """The kind's displacements S and P(next | last) over S, add-one: a row per last displacement in S, and a last
        row for an unknown one (the kind's marginal), which is also the row for a last displacement with no counts."""
        S = self.support(kind)
        marg = self.motion.get((kind, None)) or {}
        T = np.empty((len(S) + 1, len(S)))
        for a, d in enumerate(S + [None]):
            m = (self.motion.get((kind, d)) if d is not None else None) or marg
            T[a] = [m.get(e, 0) + 1 for e in S]
        return S, T / T.sum(1, keepdims=True)

    def shift(self, M, d, fill=0.0):
        out = np.full_like(M, fill)
        n = M.shape[-1]
        dx, dy = d
        if abs(dx) < n and abs(dy) < n:
            out[max(dx, 0):n + min(dx, 0), max(dy, 0):n + min(dy, 0)] = M[max(-dx, 0):n + min(-dx, 0),
                                                                          max(-dy, 0):n + min(-dy, 0)]
        return out

    def where(self, f, t):
        """Where the file could be now: ("near", S, P(d)) if seen last step, else ("map", S, P(cell, last d))."""
        S, T = self.chain(f["kind"])
        if t - f["seen"] == 1:
            d = self.key(f["vel"]) if f["vel_known"] else None
            return "near", S, T[S.index(d) if d in S else len(S)]
        S0, maps = f["possible"]
        if S0 != S:                                     # the kind has made a new displacement: S grows
            m = np.zeros((len(S),) + maps.shape[1:], maps.dtype)
            m[[S.index(d) for d in S0]] = maps
            maps = m
        mixed = np.tensordot(T[:len(S)].astype(maps.dtype), maps, axes=([0], [0]))
        return "map", S, np.stack([self.shift(mixed[i], d) for i, d in enumerate(S)])

    def cells(self, f, P):
        """Places P as cells of the file's map (centred where it was last seen); inside (bool)."""
        c = self.keys(P - f["pos"]) + self.R
        return c, np.all((c >= 0) & (c <= 2 * self.R), axis=1)

    def _new(self, t, v, pos, vel, kind, known):
        f = {"id": self.next_id, "v": v.copy(), "pos": pos.astype(np.float64), "vel": vel.astype(np.float64),
             "seen": t, "born": t, "kind": kind, "vel_known": known, "possible": None}
        self.next_id += 1
        self.files.append(f)
        return f

    # ------------------------------------------------------------------ one step
    def step(self, t, P, V, B, known, region):
        """P (n, 2) positions of the observed places, V (n, d) observed vectors, B (n, d) believed vectors there before
        this step, known (n,) whether belief held a token there; region: (lo, hi) corners of the observed region
        (the first build only). Returns explained (n,) and the files seen this step [(id, position)]."""
        n = len(P)
        known = np.asarray(known, bool)
        same = np.zeros(n, bool)
        if known.any():
            same[known] = self.dist(V[known], B[known]).diagonal() <= self.radius
        changed = ~same
        explained = np.zeros(n, bool)
        seen = []
        D = self.dist(V, V)
        nb = (np.linalg.norm(P[:, None] - P[None], axis=-1) <= self.step_len + 1e-6) & ~np.eye(n, dtype=bool)
        nb &= same[None]                                # unchanged neighbours: the surface around a place
        alike = (nb & (D <= self.radius)).sum(1)
        ground = alike * 2 > nb.sum(1)                  # continues the surface around it
        W = {} if FIRST else {f["id"]: self.where(f, t) for f in self.files}
        kind_at = np.full(n, -1)
        if self.kinds is not None:
            dk = self.dist(V, self.kinds)
            kind_at = np.where(dk.min(1) <= self.radius, dk.argmin(1), -1)
        if not FIRST:                                   # unseen objects move as their kind moves
            if self.anchor is None:
                self.anchor = P.mean(0)
            for kd in list(self.unseen):
                S, T = self.chain(kd)
                u = self.unseen[kd]
                self.unseen[kd] = sum(T[-1, i] * self.shift(u, d, self.density0(kd)) for i, d in enumerate(S))
            ucl = self.keys(P - self.anchor) + self.U
            uin = np.all((ucl >= 0) & (ucl <= 2 * self.U), axis=1)
        # files: one assignment to changed places; each place may instead go to an object not yet seen
        if self.files and changed.any():
            C = np.flatnonzero(changed)
            dv = self.dist(np.stack([f["v"] for f in self.files]), V[C])
            cost = np.full((len(self.files) + (0 if FIRST else len(C)), len(C)), BIG)
            if not FIRST:
                for j, k in enumerate(C):
                    kd = kind_at[k]
                    if kd < 0:
                        lam = 1.0
                    elif uin[k]:
                        lam = float(self.unseen_map(kd)[ucl[k, 0], ucl[k, 1]]) + self.birth(kd)
                    else:
                        lam = self.density0(kd) + self.birth(kd)
                    cost[len(self.files) + j, j] = -np.log(lam)
            for i, f in enumerate(self.files):
                like = dv[i] <= self.radius
                if not like.any():
                    continue
                if FIRST:
                    reach = self.step_len * (t - f["seen"])
                    dp = np.linalg.norm(P[C] - f["pos"], axis=1)
                    de = np.linalg.norm(P[C] - (f["pos"] + f["vel"]), axis=1)
                    ok = like & (dp <= reach + 1e-6)
                    cost[i] = np.where(ok, dv[i] / self.radius + de / self.step_len, BIG)
                    continue
                how, S, w = W[f["id"]]
                if how == "near":
                    for j in np.flatnonzero(like):
                        d = self.key(P[C[j]] - f["pos"])
                        if d in S:
                            cost[i, j] = dv[i, j] / self.radius - np.log(w[S.index(d)])
                else:
                    cl, inside = self.cells(f, P[C])
                    here = np.zeros(len(C))
                    here[inside] = w[:, cl[inside, 0], cl[inside, 1]].sum(0)
                    ok = like & (here > 0)
                    cost[i, ok] = dv[i, ok] / self.radius - np.log(here[ok])
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] >= BIG or i >= len(self.files):
                    continue
                f, k = self.files[i], C[j]
                new = P[k] - f["pos"]
                if t - f["seen"] == 1:                  # a displacement in one step: learned per kind
                    if f["vel_known"]:
                        self.motion[(f["kind"], self.key(f["vel"]))][self.key(new)] += 1
                    self.motion[(f["kind"], None)][self.key(new)] += 1
                f["vel"], f["vel_known"] = new, t - f["seen"] == 1
                f["pos"], f["v"], f["seen"], f["possible"] = P[k].astype(np.float64), V[k].copy(), t, None
                explained[k] = True
                self.stats["matched"] += 1
        # pair births: a token vanished here and one like it appeared within a step length
        app = np.flatnonzero(changed & ~explained)
        van = np.flatnonzero(changed & known)
        if len(app) and len(van):
            dv = self.dist(V[app], B[van])
            dp = np.linalg.norm(P[app][:, None] - P[van][None], axis=-1)
            ok = (dv <= self.radius) & (dp <= self.step_len + 1e-6) & (app[:, None] != van[None])
            ok &= ~ground[app][:, None] & ground[van][None]     # a figure arrived; the place it left shows ground
            if ok.any():
                cost = np.where(ok, dv / self.radius + dp / self.step_len, BIG)
                r, c = linear_sum_assignment(cost)
                for i, j in zip(r, c):
                    if cost[i, j] >= BIG:
                        continue
                    k, u = app[i], van[j]
                    kind = self.kind_of(V[k], add=True)
                    self.motion[(kind, None)][self.key(P[k] - P[u])] += 1
                    self._new(t, V[k], P[k], P[k] - P[u], kind, True)
                    explained[k] = True
                    self.stats["pair_births"] += 1
        # kind births: a token like one that has had a file, with no source
        rest = np.flatnonzero(changed & ~explained)
        if len(rest):
            for k in rest[self.is_kind(V[rest])]:
                self._new(t, V[k], P[k], np.zeros(2), self.kind_of(V[k]), False)
                explained[k] = True
                self.stats["kind_births"] += 1
        if not FIRST and self.kinds is not None:      # appearances out of nothing, read from the view
            K = self.keys(P)
            enc = K[:, 0] * 1000003 + K[:, 1]
            if self.last is not None:
                lk, lkind = self.last
                order = np.argsort(lk)
                lk, lkind = lk[order], lkind[order]
                for kd in range(len(self.kinds)):
                    ok = np.ones(n, bool)
                    for d in self.support(kd):
                        src = (K[:, 0] - d[0]) * 1000003 + (K[:, 1] - d[1])
                        i = np.minimum(np.searchsorted(lk, src), len(lk) - 1)
                        ok &= (lk[i] == src) & (lkind[i] != kd)
                    self.could[kd] += int(ok.sum())
                    self.appeared[kd] += int((ok & (kind_at == kd)).sum())
            self.last = (enc, kind_at.copy())
        self.fresh += int((~known).sum())
        self.kind_fresh.update(kind_at[(kind_at >= 0) & ~known].tolist())
        if not FIRST and self.kinds is not None:
            for kd in range(len(self.kinds)):
                self.unseen_map(kd)[ucl[uin, 0], ucl[uin, 1]] = 0
        # files not seen: where they could be, less the places the view shows; dropped when nothing is left
        lo, hi = region
        keep = []
        for f in self.files:
            if f["seen"] == t:
                seen.append((f["id"], f["pos"].copy()))
                keep.append(f)
                continue
            if FIRST:
                reach = self.step_len * (t - f["seen"])
                if np.all(f["pos"] - reach >= lo) and np.all(f["pos"] + reach <= hi):
                    self.stats["dropped"] += 1
                    continue
                keep.append(f)
                continue
            how, S, w = W[f["id"]]
            if how == "near":
                M = np.zeros((len(S),) + (2 * self.R + 1,) * 2, np.float32)
                for i, (dx, dy) in enumerate(S):
                    if abs(dx) <= self.R and abs(dy) <= self.R:
                        M[i, dx + self.R, dy + self.R] = w[i]
            else:
                M = w
            cl, inside = self.cells(f, P)
            M[:, cl[inside, 0], cl[inside, 1]] = 0
            z = float(M.sum())
            if z <= 0:
                self.stats["dropped"] += 1
                continue
            f["possible"] = (S, M / z)
            keep.append(f)
        self.files = keep
        return explained, seen
