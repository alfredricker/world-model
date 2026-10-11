"""Card 098.7: card 098.6's files level (tools/card098.6/files.py) with one change: the width of "tokens like this"
(the kernel of recall, drift and the records' share) is the candidate that best predicts held-out stored episodes'
next moves, each from all the others (leave-one-out log-likelihood weighted as the stream; Duin 1976, Silverman 1986),
in place of the identity radius. Identity (unchanged, same token, the pair test, kinds) keeps the radius.

Card 098.6's description follows.

Card 098.6: what a followed token does next. The files level of card 098.5's revision (tools/card098.5/files.py),
with its per-kind motion tallies replaced by a bank of predictors per file and its births decided by explanations.
Tokens are (vector, position); positions are continuous, so nothing here reads a grid.

Per step, after level 1 has placed the view:
  - each file predicts its next move with a bank of two members, weighted by how likely each made the file's last
    observed moves (IMM, Blom and Bar-Shalom 1988; MOSAIC's responsibilities, Wolpert and Kawato 1998):
      prior:  the move distribution of all stored episodes pooled (most things stay; a move is one of the agent's
              own steps before any move is seen);
      recall: stored episodes of tokens like the file's (compact kernel at the identity radius on the encoder's
              vectors), given its last move; the prior joins with one episode's weight (Kemp, Perfors and
              Tenenbaum 2007), so a token like nothing stored gets the prior;
  - recall runs only on a trigger (Zacks et al. 2007; Yu and Dayan 2005): a new file, a file found again after being
    unseen, a look moved past the identity radius since the last recall, or surprise: the observed move less likely
    under the bank than ALPHA times the broad predictive 1/|moves| (AXIOM's rule for a new component);
  - unseen files carry a probability over places (098.5's grid Bayes filter with negative information), moved by the
    bank's mixture;
  - one assignment of files to changed places (cost in nats: vector distance / radius - log P(here | file)
    - log existence), each place may instead go to "not a file": its token drifted or a new object;
  - a token gets a file when tokens like it have needed memory of their own (revision, with the user): one record per
    token followed by place or by a file counts the steps it was seen and how often it did what the map at its place
    and recall over its look did not predict (here: it moved by itself); the share of records like it that needed
    their own history is a two-part mixture fitted by EM, with records seen for few steps counted by how much they
    show (Kemp, Perfors and Tenenbaum 2007: how much individuals of a kind differ). A new file's existence is that
    share, at a place first seen or at a changed place no pair explains;
  - a pair (a token vanished a step away and one like it appeared) has three explanations (PMBM's birth versus update):
      moved:  a token vanished a step away (098.5's pair test, figure and ground) and recall gives its move;
      drift:  the token believed there drifted past the radius, at this scan's rate for tokens like it (the last
              RECENT frames pooled with one pseudo-frame of the long-run rate; Mahler, Vo and Vo 2011, clutter rate
              estimated per scan), times how often such drift goes at least as far. Each change counts toward these
              rates by its probability of being drift (Chen et al. 2013: clutter estimated jointly with tracking,
              each measurement weighted by its probability of being clutter);
      new:    098.5's density of unseen objects (and appearance rate) for the kind of a confirmed file it is like;
      plus an unmodelled cause at the probability the prior gives the same move;
    a new file's existence is P(moved or new);
  - existence (IPDA; the Bernoulli part of PMBM) rises on each hit by the likelihood ratio of the file against the
    other explanations and falls on each miss by the file's probability inside the view; a file is confirmed at
    CONFIRM, deleted below DELETE, and a tentative file not confirmed within WINDOW steps is deleted (M-of-N);
  - only confirmed files keep belief under them ("explained"), become kinds, store their moves as episodes, stay out
    of the placing of the agent's own motion and are reported.

Declared exceptions (card 098.6, rule 8): 098.5's unseen-object densities and appearance rates stay tallies per kind
(a leader cluster of confirmed files' vectors); similarity is the encoder's distance at its identity radius.
"""
from collections import Counter, defaultdict, deque

import numpy as np
from assign import linear_sum_assignment

BIG = 1e9
FAR = 20                    # step lengths: how far belief about an unseen file extends
WIDE = 48                   # step lengths: the unseen objects' density, from the episode's start
RES = 50000                 # episodes per reservoir
VIEW_EVERY = 50             # one in 50 views of tokens that stayed is offered to the view reservoir
SWITCH = 0.05               # IMM switching probability
ALPHA = 0.1                 # AXIOM's concentration: surprise below ALPHA x the broad predictive
CONFIRM, DELETE = 0.95, 0.05
WINDOW = 3                  # steps a tentative file has to be confirmed
EM = 5                      # iterations fitting the share of tokens that need memory of their own
WIDTHS = [(2 / 3) ** i for i in range(10)]   # candidate widths of "like this", times the identity radius (1 to 1/40)
HELD = 500                  # held-out stored episodes per store at each refit
RECENT = 10                 # frames pooled for this scan's drift
REFRESH = 1000              # steps between refreshes of the pooled prior and the kinds' motion


class Reservoir:
    """A uniform sample of a stream of episodes; each stored episode stands for seen / stored of them."""

    def __init__(self, d, seed, n=RES):
        self.V = np.zeros((n, d), np.float32)
        self.lmove = np.zeros((n, 2), np.int64)         # the last move, in half step lengths
        self.has_last = np.zeros(n, bool)
        self.last = np.full(n, -1, np.int64)            # its index in the support, -1 unknown
        self.mi = np.full(n, -1, np.int64)              # the move's index in the support
        self.move = np.zeros((n, 2), np.int64)          # move in half step lengths
        self.look = np.zeros(n, np.float32)             # how far the look changed
        self.w = np.zeros(n, np.float32)                # soft weight (P(moved) for pair events)
        self.n, self.seen, self.cap = 0, 0, n
        self.rng = np.random.default_rng(seed)

    def add(self, v, last, move, look, w, sidx):
        """last: the last move (half step lengths) or None; sidx: the support's index of each move."""
        self.seen += 1
        if self.n < self.cap:
            i = self.n
            self.n += 1
        else:
            i = int(self.rng.integers(self.seen))
            if i >= self.cap:
                return
        self.V[i], self.move[i], self.look[i], self.w[i] = v, move, look, w
        self.has_last[i] = last is not None
        self.lmove[i] = last if last is not None else (0, 0)
        self.last[i] = sidx.get(tuple(last), -1) if last is not None else -1
        self.mi[i] = sidx.get(tuple(move), -1)

    def reindex(self, sidx):
        for arr, src, ok in ((self.mi, self.move, None), (self.last, self.lmove, self.has_last)):
            out = np.full(self.n, -1, np.int64)
            for k, j in sidx.items():
                sel = np.all(src[:self.n] == np.array(k), axis=1)
                out[sel] = j
            if ok is not None:
                out[~ok[:self.n]] = -1
            arr[:self.n] = out

    def weights(self, mult=1.0):
        return self.w[:self.n] * (self.seen / max(self.n, 1)) * mult


def lgamma(x):
    """log Gamma for positive x: shifted up by the recurrence, then Stirling's series."""
    y = np.array(x, np.float64)
    s = np.zeros_like(y)
    for _ in range(7):
        small = y < 7
        s -= np.where(small, np.log(y), 0)
        y = np.where(small, y + 1, y)
    return s + (y - 0.5) * np.log(y) - y + 0.5 * np.log(2 * np.pi) + 1 / (12 * y) - 1 / (360 * y ** 3) + \
        1 / (1260 * y ** 5)


def betaln(a, b):
    return lgamma(a) + lgamma(b) - lgamma(np.asarray(a) + np.asarray(b))


class Records:
    """One record per token followed by place or by a file: its vector when first seen, the steps it was seen after
    that, and how many of them it moved by itself (soft). A uniform sample of all records; each is updated while its
    token is followed."""

    def __init__(self, d, seed, n=RES):
        self.V = np.zeros((n, d), np.float32)
        self.k = np.zeros(n, np.float32)
        self.mu = np.zeros(n, np.float32)
        self.rid = np.full(n, -1, np.int64)
        self.n, self.seen, self.cap = 0, 0, n
        self.rng = np.random.default_rng(seed)

    def new(self, v):
        """A new record; its link (slot, id), or None when the sample does not keep it."""
        rid = self.seen
        self.seen += 1
        if self.n < self.cap:
            i = self.n
            self.n += 1
        else:
            i = int(self.rng.integers(self.seen))
            if i >= self.cap:
                return None
        self.V[i], self.k[i], self.mu[i], self.rid[i] = v, 0, 0, rid
        return i, rid

    def add(self, link, k, mu=0.0):
        if link is not None and self.rid[link[0]] == link[1]:
            self.k[link[0]] += k
            self.mu[link[0]] += mu


class Files:
    def __init__(self, dist, radius, step_len, own=(), dim=None, seed=0):
        self.dist, self.radius, self.step_len = dist, float(radius), float(step_len)
        self.h = float(radius)                          # the width of "like this", refitted by held-out prediction
        self.widths = []                                # (step, chosen width, held-out log-likelihood per candidate)
        self.next_fit = REFRESH
        self.rng_w = np.random.default_rng(seed + 11)   # held-out samples (apart from the views' sampling)
        self.h = self.step_len / 2
        self.R, self.U = 2 * FAR, 2 * WIDE
        self.S = sorted({self.key(np.zeros(2))} | {self.key(d) for d in own})   # the moves a token can make
        self.dim = dim
        self.Rf = self.Rp = self.Rv = None              # long-term: file steps, pair events, views
        self.Rt = None                                  # long-term: one record per token followed
        self.pi0 = None
        self.seed = seed
        self.rng = np.random.default_rng(seed + 7)
        self.kinds = None                               # long-term: confirmed files' vectors (leader)
        self.kind_T = {}                                # kind -> its leader's recall marginal (refreshed)
        self.fresh, self.kind_fresh = 0, Counter()      # long-term: places first seen, and those showing each kind
        self.could, self.appeared = Counter(), Counter()
        self.T_prior = None
        self.looks = np.zeros(0, np.float32)            # moved files' look changes (sorted, refreshed)
        self.steps = 0
        self.next_id = 0
        self.stats = Counter()
        self.log, self.log_pairs, self.log_moves, self.log_births = False, [], [], []   # the gate's check
        self.reset()

    def reset(self):
        self.files = []
        self.unseen, self.anchor = {}, None
        self.place_rec = {}                             # place -> the record of the token believed there
        self.last = None                                # last step's places (keys, kinds)
        self.prev = None                                # last step's places (keys, positions) for the view stream
        self.recent = deque(maxlen=RECENT)              # this scan's drift: (vectors, changed, distance)

    # ------------------------------------------------------------------ helpers
    def key(self, d):
        return tuple(np.round(np.asarray(d, np.float64) / self.h).astype(int).tolist())

    def keys(self, D):
        return np.round(np.asarray(D, np.float64).reshape(-1, 2) / self.h).astype(np.int64)

    def enc(self, K):
        return K[:, 0] * 1000003 + K[:, 1]

    def kern(self, Q, X):
        """Compact kernel of width h, the width of "like this": 1 - (d / h)^2 inside, 0 outside."""
        if len(Q) == 0 or len(X) == 0:
            return np.zeros((len(Q), len(X)), np.float32)
        Q, X = np.asarray(Q, np.float32), np.asarray(X, np.float32)
        d2 = (Q * Q).sum(1)[:, None] + (X * X).sum(1)[None] - 2 * Q @ X.T
        return np.clip(1 - d2 / (self.h ** 2), 0, None)

    def choose_width(self):
        """The candidate width that best predicts held-out stored episodes' next moves, each from all the others with
        the prior at one episode's weight, as recall predicts (leave-one-out log-likelihood; HELD episodes per store,
        each weighted by the share of the stream its store stands for)."""
        parts = self._res()
        if len(parts) < 2:
            return
        X = np.concatenate([R.V[:R.n] for R, m in parts]).astype(np.float32)
        M = np.concatenate([R.mi[:R.n] for R, m in parts])
        W = np.concatenate([R.weights(m) for R, m in parts]).astype(np.float64)
        store = np.concatenate([np.full(R.n, i) for i, (R, m) in enumerate(parts)])
        ok = (M >= 0) & (W > 0)
        X, M, W, store = X[ok], M[ok], W[ok], store[ok]
        Q, qw = [], []
        for i in range(len(parts)):
            idx = np.flatnonzero(store == i)
            if not len(idx):
                continue
            pick = self.rng_w.choice(idx, min(HELD, len(idx)), replace=False)
            Q.append(pick)
            qw.append(W[pick] * len(idx) / len(pick))   # each stands for its share of its store's stream
        Q, qw = np.concatenate(Q), np.concatenate(qw)
        S = len(self.S)
        P = self.prior()[-1]
        Y = np.zeros((len(X), S))
        Y[np.arange(len(X)), M] = W
        xx = (X * X).sum(1)
        ll = np.zeros(len(WIDTHS))
        for a in range(0, len(Q), 250):
            q = Q[a:a + 250]
            d2 = np.maximum(xx[q][:, None] + xx[None] - 2 * X[q] @ X.T, 0)
            d2[np.arange(len(q)), q] = np.inf           # leave the episode itself out
            for j, f in enumerate(WIDTHS):
                K = np.clip(1 - d2 / (f * self.radius) ** 2, 0, None)
                C = K @ Y
                T = (C[np.arange(len(q)), M[q]] + P[M[q]]) / (C.sum(1) + 1)
                ll[j] += float((qw[a:a + 250] * np.log(T)).sum())
        ll /= qw.sum()
        self.h = WIDTHS[int(np.argmax(ll))] * self.radius
        self.widths.append((self.steps, round(self.h, 4), [round(float(x), 4) for x in ll]))

    def _stores(self, d):
        if self.Rf is None:
            self.dim = d
            self.Rf, self.Rp, self.Rv = (Reservoir(d, self.seed + i) for i in range(3))
        if getattr(self, "Rt", None) is None:
            self.Rt = Records(d, self.seed + 3)

    def sidx(self):
        return {k: i for i, k in enumerate(self.S)}

    def s_index(self, d):
        k = self.key(d) if not isinstance(d, tuple) else d
        if k not in self.S:
            self.S = sorted(set(self.S) | {k})
            for f in self.files:                        # the support grew: every file's recall is stale
                f["need"] = True
            self.T_prior = None
            for R, m in self._res():
                R.reindex(self.sidx())
        return self.S.index(k)

    def is_kind(self, V):
        if self.kinds is None or not len(V):
            return np.zeros(len(V), bool)
        return self.dist(V, self.kinds).min(1) <= self.radius

    def kind_of(self, v, add=False):
        if self.kinds is not None:
            d = self.dist(v[None], self.kinds)[0]
            if d.min() <= self.radius:
                return int(d.argmin())
        if not add:
            return -1
        self.kinds = v[None].copy() if self.kinds is None else np.concatenate([self.kinds, v[None]])
        return len(self.kinds) - 1

    # ------------------------------------------------------------------ prediction
    def prior(self):
        """The move distribution of all stored episodes pooled (rows: last move, then unknown; all alike)."""
        if self.T_prior is None or self.T_prior.shape[1] != len(self.S):
            c = np.ones(len(self.S))                    # add-one
            for R, mult in self._res():
                mi = R.mi[:R.n]
                ok = mi >= 0
                np.add.at(c, mi[ok], R.weights(mult)[ok])
            self.T_prior = np.tile(c / c.sum(), (len(self.S) + 1, 1))
        return self.T_prior

    def _res(self):
        return [(R, m) for R, m in ((self.Rf, 1.0), (self.Rp, 1.0), (self.Rv, VIEW_EVERY)) if R is not None and R.n]

    def recall(self, Q):
        """For each query vector, P(next move | last move) as (m, |S|+1, |S|): rows per last move, last row unknown."""
        self.stats["recall_calls"] += len(Q)
        m, s = len(Q), len(self.S)
        C = np.zeros((m, s + 1, s))
        for R, mult in self._res():
            K = self.kern(Q, R.V[:R.n]) * R.weights(mult)[None]
            mi = R.mi[:R.n]
            ok = mi >= 0
            M = np.zeros((R.n, s), np.float32)
            M[np.flatnonzero(ok), mi[ok]] = 1
            C[:, s] += K @ M
            for a in range(s):
                sel = R.last[:R.n] == a
                if sel.any():
                    C[:, a] += K[:, sel] @ M[sel]
        P = self.prior()[0]
        tot = C.sum(2, keepdims=True)
        T = (C + P[None, None]) / (tot + 1)             # the prior with one episode's weight
        thin = tot[..., 0] < 1                          # less than one episode for this last move: use the marginal
        T[thin] = np.repeat(T[:, s:s + 1], s + 1, axis=1)[thin]
        return T

    def bank(self, f):
        return f["w"][0] * self.prior() + f["w"][1] * f["T"]

    def last_row(self, f):
        if f["vel_known"]:
            k = self.key(f["vel"])
            if k in self.S:
                return self.S.index(k)
        return len(self.S)

    def refresh(self, files):
        """Recall for the files whose trigger fired."""
        if not files:
            return
        T = self.recall(np.stack([f["v"] for f in files]))
        for f, t in zip(files, T):
            f["T"], f["v_recall"], f["need"] = t, f["v"].copy(), False

    # ------------------------------------------------------------------ which tokens need memory of their own
    def _evidence(self, m, idx=None):
        """Per record, log P(its history | it needs its own memory) - log P(its history | the map predicts it).
        A token with its own memory moves at a rate of its own, drawn around the rate m of all followed tokens with
        one episode's weight (a beta prior: individuals differ, Kemp, Perfors and Tenenbaum 2007); a token the map
        predicts is seen to move only at the rate any token is. (Centred on the rate of followed tokens like it,
        the two parts coincide for a look whose few followed tokens never moved, and the share is undetermined.)"""
        R = self.Rt
        eps = (R.mu[:R.n].sum() + 0.5) / (R.k[:R.n].sum() + 1)
        k, mu = (R.k[:R.n], R.mu[:R.n]) if idx is None else (R.k[idx], R.mu[idx])
        a, b = np.clip(m, 1e-3, 1 - 1e-3), 1 - np.clip(m, 1e-3, 1 - 1e-3)
        stay = np.maximum(k - mu, 0)
        return betaln(a + mu, b + stay) - betaln(a, b) - mu * np.log(eps) - stay * np.log1p(-eps)

    def individual(self, Q):
        """For each query vector, how often tokens like it have needed memory of their own: of the records within
        the identity radius, the share whose history the map does not predict, fitted by EM (a record seen for
        few steps counts by how much it shows); all records pooled join with one record's weight."""
        R = self.Rt
        if R is None or not R.n:
            return np.full(len(Q), 0.5)
        if self.pi0 is None or self.steps % 100 == 0:   # all records pooled, at the pooled move rate
            ev = self._evidence(self._m0())
            pi = 0.5
            for _ in range(EM):
                z = 1 / (1 + np.exp(-np.clip(np.log(pi / (1 - pi)) + ev, -50, 50)))
                pi = float((z.sum() + 0.5) / (len(z) + 1))
            self.pi0 = pi
        K = self.kern(Q, R.V[:R.n])
        cols = np.flatnonzero(K.sum(0) > 0)
        pi = np.full(len(Q), self.pi0)
        if not len(cols):
            return pi
        K = K[:, cols]
        ev = self._evidence(self._m0(), cols)[None]     # the same rate for every query
        for _ in range(EM):
            z = 1 / (1 + np.exp(-np.clip(np.log(pi / (1 - pi))[:, None] + ev, -50, 50)))
            pi = np.clip(((K * z).sum(1) + self.pi0) / (K.sum(1) + 1), 1e-6, 1 - 1e-6)
        return pi

    def own_move(self, dj):
        """P(a followed token makes move dj in a step): the pooled rate of followed tokens' moves, times the share
        of their moves that went that way (add-one)."""
        R = self.Rf
        mi = R.mi[:R.n] if R is not None else np.zeros(0, np.int64)
        stay = self.S.index((0, 0))
        moved = mi[(mi >= 0) & (mi != stay)]
        return self._m0() * (np.sum(moved == dj) + 1) / (len(moved) + len(self.S) - 1)

    def scenery_move(self, dj):
        """P(a token the map predicts is seen to make move dj): the rate any token is seen to move, times the
        prior's share of that move among moves."""
        R = self.Rt
        eps = (R.mu[:R.n].sum() + 0.5) / (R.k[:R.n].sum() + 1)
        row = self.prior()[-1]
        stay = self.S.index((0, 0))
        return eps * row[dj] / max(1 - row[stay], 1e-12)

    def _m0(self):
        """How often a followed token moves in a step (confirmed files' steps, pooled)."""
        R = self.Rf
        if R is None or not R.n:
            return 0.5
        return float((np.any(R.move[:R.n] != 0, axis=1).sum() + 0.5) / (R.n + 1))

    # ------------------------------------------------------------------ explanations
    def drift(self, B, W):
        """P(the token believed as B drifted to W): this scan's rate of drift past the radius for tokens like B,
        pooled with one pseudo-frame of the long-run rate, times how often such drift goes at least as far."""
        m = len(B)
        if m == 0:
            return np.zeros(0)
        dist = np.linalg.norm(np.asarray(W, np.float32) - np.asarray(B, np.float32), axis=1)

        def tally(K, u, ch, far):                       # places whose token stayed; drifted; as far or farther
            return K @ u, K @ ch, (K * (ch[None] * far)).sum(1)

        rk = rc = rf = np.zeros(m)
        if self.recent:
            Xb = np.concatenate([r[0] for r in self.recent])
            Xc = np.concatenate([r[1] for r in self.recent]).astype(np.float32)
            Xd = np.concatenate([r[2] for r in self.recent])
            Xu = np.concatenate([r[3] for r in self.recent]).astype(np.float32)
            rk, rc, rf = tally(self.kern(B, Xb), Xu, Xc, Xd[None] >= dist[:, None])
        lk = lc = lf = np.zeros(m)
        if self.Rv is not None and self.Rv.n:
            look = self.Rv.look[:self.Rv.n]
            wv = self.Rv.w[:self.Rv.n]
            lk, lc, lf = tally(self.kern(B, self.Rv.V[:self.Rv.n]), wv, wv * (look > self.radius),
                               look[None] >= dist[:, None])
        p_long, s_long = (lc + 0.5) / (lk + 1), (lf + 0.5) / (lc + 1)
        frame, cframe = np.maximum(rk / RECENT, 1.0), np.maximum(rc / RECENT, 1.0)   # one pseudo-frame each
        p_change = (rc + frame * p_long) / (rk + frame)
        s_far = (rf + cframe * s_long) / (rc + cframe)
        return p_change * s_far

    def look_match(self, dist):
        """How often a moved token's look changes at least this far (confirmed files' moves)."""
        n = len(self.looks)
        if n == 0:
            return np.ones(len(np.atleast_1d(dist)))
        return (n - np.searchsorted(self.looks, np.atleast_1d(dist)) + 0.5) / (n + 1)

    # ------------------------------------------------------------------ unseen objects (098.5, declared exception)
    def density0(self, kind):
        return (self.kind_fresh[kind] + 1) / (self.fresh + 2)

    def birth(self, kind):
        return (self.appeared[kind] + 1) / (self.could[kind] + 2)

    def unseen_map(self, kind):
        if kind not in self.unseen:
            self.unseen[kind] = np.full((2 * self.U + 1,) * 2, self.density0(kind), np.float32)
        return self.unseen[kind]

    def shift(self, M, d, fill=0.0):
        out = np.full_like(M, fill)
        n = M.shape[-1]
        dx, dy = d
        if abs(dx) < n and abs(dy) < n:
            out[max(dx, 0):n + min(dx, 0), max(dy, 0):n + min(dy, 0)] = M[max(-dx, 0):n + min(-dx, 0),
                                                                          max(-dy, 0):n + min(-dy, 0)]
        return out

    def cells(self, f, P):
        c = self.keys(P - f["pos"]) + self.R
        return c, np.all((c >= 0) & (c <= 2 * self.R), axis=1)

    # ------------------------------------------------------------------ where a file could be
    def where(self, f, t):
        """Where file f could be now, with the moves its prediction is over (the support may grow within a step)."""
        T = self.bank(f)
        if t - f["seen"] == 1:
            return "near", T[self.last_row(f)], list(self.S)
        S0, maps = f["possible"]
        if S0 != self.S:
            m = np.zeros((len(self.S),) + maps.shape[1:], maps.dtype)
            m[[self.S.index(d) for d in S0]] = maps
            maps = m
        mixed = np.tensordot(T[:len(self.S)].astype(maps.dtype), maps, axes=([0], [0]))
        return "map", np.stack([self.shift(mixed[i], d) for i, d in enumerate(self.S)]), list(self.S)

    def _new(self, t, v, pos, vel, known, r0, rec=None):
        f = {"id": self.next_id, "rec": rec, "v": v.copy(), "pos": pos.astype(np.float64),
             "vel": vel.astype(np.float64),
             "seen": t, "born": t, "kind": -1, "vel_known": known, "possible": None, "r": float(r0),
             "confirmed": False, "w": np.array([0.5, 0.5]), "T": None, "need": True, "v_recall": None}
        self.next_id += 1
        self.files.append(f)
        self.stats["births"] += 1
        if r0 >= CONFIRM:
            self._confirm(f)
        if f["confirmed"]:                              # the belief under a confirmed file is not its token's
            self.place_rec.pop(self.key(pos), None)
        return f

    def _confirm(self, f):
        f["confirmed"] = True
        f["kind"] = self.kind_of(f["v"], add=True)
        self.stats["confirmed"] += 1

    # ------------------------------------------------------------------ one step
    def step(self, t, P, V, B, known, region):
        n = len(P)
        V, B = np.asarray(V, np.float32), np.asarray(B, np.float32)
        self._stores(V.shape[1])
        self.steps += 1
        if self.steps >= self.next_fit:                 # refitted at doubling intervals: 1,000, 2,000, 4,000 ... steps
            h0 = self.h
            self.choose_width()
            self.next_fit *= 2
            if self.h != h0:
                for f in self.files:                    # recall's last answers used the old width
                    f["need"] = True
        if self.steps % REFRESH == 0:
            self.T_prior = None
            self.looks = np.sort(self.Rf.look[:self.Rf.n][np.any(self.Rf.move[:self.Rf.n] != 0, axis=1)])
            if self.kinds is not None:
                T = self.recall(self.kinds)
                self.kind_T = {k: T[k][-1] for k in range(len(self.kinds))}
        known = np.asarray(known, bool)
        same = np.zeros(n, bool)
        if known.any():
            same[known] = np.linalg.norm(V[known] - B[known], axis=1) <= self.radius
        changed = ~same
        explained = np.zeros(n, bool)
        taken = np.zeros(n, bool)                       # by any file this step
        seen = []
        D = self.dist(V, V)
        nb = (np.linalg.norm(P[:, None] - P[None], axis=-1) <= self.step_len + 1e-6) & ~np.eye(n, dtype=bool)
        nb &= same[None]
        alike = (nb & (D <= self.radius)).sum(1)
        ground = alike * 2 > nb.sum(1)
        pk = self.keys(P)
        penc = self.enc(pk)
        kind_at = np.full(n, -1)
        if self.kinds is not None:
            dk = self.dist(V, self.kinds)
            kind_at = np.where(dk.min(1) <= self.radius, dk.argmin(1), -1)
        if self.anchor is None:
            self.anchor = P.mean(0)
        for kd in list(self.unseen):                    # unseen objects move as their kind moves (098.5)
            row = self.kind_T.get(kd, self.prior()[-1])
            if len(row) != len(self.S):
                row = self.prior()[-1]
            u = self.unseen[kd]
            self.unseen[kd] = sum(row[i] * self.shift(u, d, self.density0(kd)) for i, d in enumerate(self.S))
        ucl = self.keys(P - self.anchor) + self.U
        uin = np.all((ucl >= 0) & (ucl <= 2 * self.U), axis=1)

        def lam_new(k):
            kd = kind_at[k]
            if kd < 0:
                return 0.0
            u = float(self.unseen_map(kd)[ucl[k, 0], ucl[k, 1]]) if uin[k] else self.density0(kd)
            return u + self.birth(kd)

        self.refresh([f for f in self.files if f["need"]])
        W = {f["id"]: self.where(f, t) for f in self.files}
        C = np.flatnonzero(changed)
        # a changed place without this file: its believed token drifted; at a place first seen, the token is one the
        # map predicts, as often as tokens like it are (the same share decides births there)
        alt = np.zeros(n)
        pi_at = np.full(n, np.nan)
        kc = C[known[C]]
        if len(kc):
            alt[kc] = self.drift(B[kc], V[kc])
        uc = C[~known[C]]
        if len(uc):
            pi_at[uc] = self.individual(V[uc])
            alt[uc] = 1 - pi_at[uc]
        lam = np.array([lam_new(k) for k in range(n)])
        # files: one assignment to changed places; each place may instead go to "not a file"
        if self.files and len(C):
            nf = len(self.files)
            dv = self.dist(np.stack([f["v"] for f in self.files]), V[C])
            cost = np.full((nf + len(C), len(C)), BIG)
            for j, k in enumerate(C):
                cost[nf + j, j] = -np.log(max(alt[k] + lam[k], 1e-12))
            here = np.zeros((nf, len(C)))
            for i, f in enumerate(self.files):
                like = dv[i] <= self.radius
                if not like.any():
                    continue
                how, w, S0 = W[f["id"]]
                if how == "near":
                    for j in np.flatnonzero(like):
                        d = self.key(P[C[j]] - f["pos"])
                        if d in S0:
                            here[i, j] = w[S0.index(d)]
                else:
                    cl, inside = self.cells(f, P[C])
                    h = np.zeros(len(C))
                    h[inside] = w[:, cl[inside, 0], cl[inside, 1]].sum(0)
                    here[i] = np.where(like, h, 0)
                ok = here[i] > 0
                cost[i, ok] = dv[i, ok] / self.radius - np.log(here[i, ok]) - np.log(max(f["r"], 1e-12))
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] >= BIG or i >= nf:
                    continue
                f, k = self.files[i], C[j]
                new = P[k] - f["pos"]
                look = float(np.linalg.norm(V[k] - f["v"]))
                found_again = t - f["seen"] > 1
                own = here[i, j] * float(self.look_match(look)[0])
                moved = self.key(new) != (0, 0)
                other = alt[k] + lam[k]
                L = own / max(other, 1e-12)
                f["r"] = f["r"] * L / (1 - f["r"] + f["r"] * L)
                self.Rt.add(f["rec"], 1, own / (own + other) if moved else 0.0)
                if not f["confirmed"]:                  # a tentative file's token is believed at its place
                    if self.key(f["pos"]) in self.place_rec and self.place_rec[self.key(f["pos"])] == f["rec"]:
                        self.place_rec.pop(self.key(f["pos"]))
                    self.place_rec[self.key(P[k])] = f["rec"]
                if not found_again:                     # one step: the bank learns, the move is an episode
                    dj = self.s_index(new)
                    a = self.last_row(f)
                    T0, T1 = self.prior(), f["T"]
                    if T1 is not None and T1.shape[1] == len(self.S):
                        p = np.array([T0[a, dj], T1[a, dj]])
                        w = (f["w"] * (1 - SWITCH) + SWITCH / 2) * p
                        f["w"] = w / w.sum() if w.sum() > 0 else np.array([0.5, 0.5])
                        if float(f["w"] @ p) < ALPHA / len(self.S):
                            f["need"] = True            # surprise
                            self.stats["surprise"] += 1
                        if self.log and f["confirmed"]:
                            self.log_moves.append((t, tuple(f["pos"]), tuple(new), float(np.log(T1[a, dj])),
                                                   float(np.log(T0[a, dj]))))
                    if f["confirmed"]:
                        lm = self.key(f["vel"]) if f["vel_known"] else None
                        self.Rf.add(f["v"], lm, self.key(new), look, 1.0, self.sidx())
                else:
                    f["need"] = True                    # found again
                if f["confirmed"] and f["v_recall"] is not None and \
                        np.linalg.norm(V[k] - f["v_recall"]) > self.radius:
                    f["need"] = True                    # its look moved past the radius
                f["vel"], f["vel_known"] = new, not found_again
                f["pos"], f["v"], f["seen"], f["possible"] = P[k].astype(np.float64), V[k].copy(), t, None
                if not f["confirmed"] and f["r"] >= CONFIRM:
                    self._confirm(f)
                    self.place_rec.pop(self.key(P[k]), None)
                taken[k] = True
                explained[k] = f["confirmed"]
                self.stats["matched"] += 1
        # births from pairs: a token vanished a step away and one like it appeared here
        app = np.flatnonzero(changed & ~taken)
        van = np.flatnonzero(changed & known & ~taken)
        paired = np.zeros(n, bool)
        pdrift = np.zeros(n)                            # each change's probability of being drift
        if len(app) and len(van):
            dv = self.dist(V[app], B[van])
            dp = np.linalg.norm(P[app][:, None] - P[van][None], axis=-1)
            ok = (dv <= self.radius) & (dp <= self.step_len + 1e-6) & (app[:, None] != van[None])
            ok &= ~ground[app][:, None] & ground[van][None]
            if ok.any():
                cost = np.where(ok, dv / self.radius + dp / self.step_len, BIG)
                r, c = linear_sum_assignment(cost)
                pairs = [(app[i], van[j]) for i, j in zip(r, c) if cost[i, j] < BIG]
                if pairs:
                    Pi = self.individual(B[[u for k, u in pairs]])
                    Dy = self.drift(B[[u for k, u in pairs]], V[[u for k, u in pairs]])
                    for (k, u), pi, dy in zip(pairs, Pi, Dy):
                        d = self.key(P[k] - P[u])
                        dj = self.s_index(d)
                        # moved: the token has memory of its own (the share of tokens like it that do) and made this
                        # move; or it is one the map predicts, and both places drifted or an unmodelled cause moved it
                        lm = pi * self.own_move(dj) * float(self.look_match(np.linalg.norm(V[k] - B[u]))[0])
                        ld = (1 - pi) * dy * alt[k]
                        lo = (1 - pi) * self.scenery_move(dj)
                        ln = lam[k] * dy
                        tot = lm + ld + ln + lo
                        r0 = (lm + ln) / tot if tot > 0 else 0.0
                        self.Rp.add(B[u], None, d, float(np.linalg.norm(V[k] - B[u])),
                                    lm / tot if tot > 0 else 0.0, self.sidx())
                        link = self.place_rec.pop(self.key(P[u])) if self.key(P[u]) in self.place_rec \
                            else self.Rt.new(B[u])
                        self.Rt.add(link, 1, lm / tot if tot > 0 else 0.0)
                        self.place_rec[self.key(P[k])] = link
                        paired[k] = paired[u] = True
                        pdrift[k] = pdrift[u] = ld / tot if tot > 0 else 1.0
                        self.stats["pair_events"] += 1
                        if self.log:
                            self.log_pairs.append((t, tuple(P[u]), tuple(P[k]), r0, lm / tot if tot > 0 else 0.0))
                        if r0 >= DELETE:
                            f = self._new(t, V[k], P[k], P[k] - P[u], True, r0, link)
                            taken[k] = True
                            explained[k] = f["confirmed"]
                            self.stats["pair_births"] += 1
        # a token no file or pair explains, at a place first seen or changed: a new record; a file when tokens like
        # it have needed memory of their own
        rest = np.flatnonzero(changed & ~taken & ~paired)
        if len(rest):
            pi = pi_at[rest]
            ask = np.isnan(pi)
            if ask.any():
                pi[ask] = self.individual(V[rest[ask]])
            for k, r0 in zip(rest, pi):
                link = self.Rt.new(V[k])
                self.place_rec[self.key(P[k])] = link
                if self.log:
                    self.log_births.append((t, tuple(P[k]), float(r0), bool(known[k])))
                if r0 >= DELETE:
                    f = self._new(t, V[k], P[k], np.zeros(2), False, float(r0), link)
                    taken[k] = True
                    explained[k] = f["confirmed"]
                    self.stats["token_births"] += 1
        # tokens that stayed: one more step in their records
        for k in np.flatnonzero(same & ~taken):
            key = self.key(P[k])
            if key in self.place_rec:                   # (None: a record the sample did not keep)
                self.Rt.add(self.place_rec[key], 1)
            else:
                self.place_rec[key] = self.Rt.new(V[k])
        # views: this scan's drift, and one in VIEW_EVERY of the tokens that stayed, as episodes; a change counts by
        # its probability of being drift (against a new object, or an unmodelled arrival at the prior's move rate)
        conf_here = explained.copy()
        kv = np.flatnonzero(known & ~taken & ~conf_here)
        if len(kv):
            dist = np.linalg.norm(V[kv] - B[kv], axis=1)
            ch = dist > self.radius
            arrive = 1.0 - self.prior()[-1][self.S.index((0, 0))]
            pd = np.where(paired[kv], pdrift[kv], alt[kv] / np.maximum(alt[kv] + lam[kv] + arrive, 1e-12))
            cw = np.where(ch, pd, 0.0)
            uw = np.where(ch, pd, 1.0)
            self.recent.append((B[kv].copy(), cw, dist, uw))
            if self.prev is not None:
                pe = self.prev
                full = np.ones(len(kv), bool)
                for d in self.S:
                    if d == (0, 0) or max(abs(d[0]), abs(d[1])) > 2:
                        continue
                    src = self.enc(pk[kv] + np.array(d))
                    full &= np.isin(src, pe) & np.isin(src, penc)
                for k, ds, ok, u in zip(kv, dist, full, uw):
                    if ok and u > 0 and self.rng.random() < 1 / VIEW_EVERY:
                        self.Rv.add(B[k], None, (0, 0), ds, float(u), self.sidx())
        self.prev = penc.copy()
        # appearances out of nothing (098.5), read from the view
        if self.kinds is not None:
            if self.last is not None:
                lk, lkind = self.last
                order = np.argsort(lk)
                lk, lkind = lk[order], lkind[order]
                for kd in range(len(self.kinds)):
                    ok = np.ones(n, bool)
                    for d in self.S:
                        src = (pk[:, 0] - d[0]) * 1000003 + (pk[:, 1] - d[1])
                        i = np.minimum(np.searchsorted(lk, src), len(lk) - 1)
                        ok &= (lk[i] == src) & (lkind[i] != kd)
                    self.could[kd] += int(ok.sum())
                    self.appeared[kd] += int((ok & (kind_at == kd)).sum())
            self.last = (penc, kind_at.copy())
        self.fresh += int((~known).sum())
        self.kind_fresh.update(kind_at[(kind_at >= 0) & ~known].tolist())
        if self.kinds is not None:
            for kd in range(len(self.kinds)):
                self.unseen_map(kd)[ucl[uin, 0], ucl[uin, 1]] = 0
        # files not found: existence falls by their probability inside the view; tentative ones have a window
        keep = []
        for f in self.files:
            if f["seen"] == t:
                if f["confirmed"]:
                    seen.append((f["id"], f["pos"].copy()))
                if f["confirmed"] or t - f["born"] < WINDOW:
                    keep.append(f)
                else:
                    self.stats["deleted"] += 1
                continue
            how, w, S0 = W[f["id"]]
            if how == "near":
                M = np.zeros((len(S0),) + (2 * self.R + 1,) * 2, np.float32)
                for i, (dx, dy) in enumerate(S0):
                    if abs(dx) <= self.R and abs(dy) <= self.R:
                        M[i, dx + self.R, dy + self.R] = w[i]
            else:
                M = w
            cl, inside = self.cells(f, P)
            q = float(M.sum(0)[cl[inside, 0], cl[inside, 1]].sum()) / max(float(M.sum()), 1e-12)
            f["r"] = f["r"] * (1 - q) / max(1 - f["r"] * q, 1e-12)
            M[:, cl[inside, 0], cl[inside, 1]] = 0
            z = float(M.sum())
            if z <= 0 or f["r"] < DELETE or (not f["confirmed"] and t - f["born"] >= WINDOW):
                self.stats["deleted"] += 1
                continue
            f["possible"] = (S0, M / z)
            keep.append(f)
        self.files = keep
        return explained, seen
