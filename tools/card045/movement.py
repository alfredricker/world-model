"""Card 045: card 044's planner (tokens) with walking through conditions.

System 1, the approach. A small network Q(g, m) predicts how many steps until the agent stands at a placement
g after move m, g given in the agent's frame (the placement's offset from the agent, x right and y ahead, and its
heading relative to the agent's, a unit vector). It is trained by fitted Q-iteration, Q(g, m) = 1 + min Q(T_m g,
.) and 0 at the target, where T_m is card 044's learned transformation of move m; every place within 6 tiles
and every heading is a target (hindsight), and only moves that have their effect are used, so V = min Q is how
soon in open space. V is read in whole steps; among moves with the least Q the order is forward, left, right
(card 023's).

The route. The tokens the approach steps onto, from its own moves and the transformations, with no recall.
It depends only on the two placements, so it is worked out once for every pair of placements (as are V and the
first move, the network's outputs) and kept.

System 2, the approach's conditions. The approach works when every token on its route is walkable (recall's
forward kind, per token). Between the placements the agent can stand at (on a walkable token), the clear
approaches are the edges of a chain: H0(q) = least V(q, p) over the need's placements p with a clear route, and
H_k(q) = min(H_{k-1}(q), least over q2 of V(q, q2) + H_{k-1}(q2) with a clear route to q2), up to card 029's
depth. The agent takes the first move of the first approach of a least chain: to p directly, else to the first
waypoint. When no chain exists, tokens that memory has seen made walkable (a door) are counted walkable; the
ones a least chain then crosses become conditions ("walk", j), pursued like any condition (cards 029, 038, 043).
Choosing between alternatives uses the chain's predicted steps. Nothing in walking calls recall on an imagined
move. Everything else is card 044's.

Walking modes (--walking): 045 (this card), 044 (card 044's walking, the baseline), exact (upper bound: the
fewest steps over placements by breadth-first search on the agent's own tokens, in place of V and the chains).

  bin/prun python tools/card045/movement.py --unseen                                  unseen rooms, shortest routes
  bin/prun python tools/card045/movement.py --dev --arm A --seeds 399-399 --layouts 30 --out runs/045_dev.json
  bin/prun python tools/card045/movement.py --arm A --seeds 399-399 --layouts 30 --layouts-b 30 --out runs/045_shakedown_399.json
  bin/prun python tools/card045/movement.py --arm A --seeds 400-404 --layouts-b 100 --out runs/045_armA.json
"""
import dataclasses
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card044"))
import tokens as TK                                    # noqa: E402  (card 044 -> 043 -> 042 -> 039 -> 038)

CS, CR, SP, VP, F, G = TK.CS, TK.CR, TK.SP, TK.VP, TK.F, TK.G
ld, closer = VP.ld, F.closer
NV, NPL, CENTRE, FRONT, HELD, R = TK.NV, TK.NPL, TK.CENTRE, TK.FRONT, TK.HELD, TK.R
MOVES, LEFT, RIGHT, FWD = VP.MOVES, VP.LEFT, VP.RIGHT, VP.FWD
ORDER = tuple(int(a) for a in closer.MOVE_ORDER)       # forward, left, right
HEADS = ((0, 1), (1, 0), (0, -1), (-1, 0))
INF = 10 ** 6
ROOT = Path(__file__).resolve().parents[2]
UNSEEN = ROOT / "runs" / "045_unseen.pkl"
ROOMS = ("room6", "room7", "mirror8")
WALKING = "045"
PLANS = []                                             # the planners made in this process (for their counts)


# ---------------------------------------------------------------- System 1: how soon, learned

def targets(r=R):
    return np.array([(x, y, hx, hy) for x in range(-r, r + 1) for y in range(-r, r + 1) for hx, hy in HEADS],
                    np.float32)


def moved(T, X):
    """Target placements after each move (card 044's transformations)."""
    return {a: np.concatenate([X[:, :2] @ T[a][0].T + T[a][1], X[:, 2:] @ T[a][0].T], 1).astype(np.float32)
            for a in MOVES}


QCACHE = {}


def train_q(T, dev, margin=R, iters=80, steps=600, hidden=256, lr=1e-3, seed=0, held=None):
    """Fitted Q-iteration over every target placement within 12 tiles (6 beyond the view, so that moves
    leaving the view are bootstrapped from trained values) and every move. Squared error; the rate falls tenfold
    for the last 30% of iterations. Q depends only on the transformations, so it is trained once per run.
    held (the gate's check): target placements left out of the loss (still read where moves lead to them)."""
    key = tuple((tuple(T[a][0].ravel().tolist()), tuple(T[a][1].tolist())) for a in MOVES)
    if key in QCACHE and held is None:
        return QCACHE[key]
    import torch
    torch.manual_seed(seed)
    f32 = dict(dtype=torch.float32, device=dev)
    X = targets(R + margin)
    N = moved(T, X)
    goal = np.array([0, 0, 0, 1], np.float32)
    net = torch.nn.Sequential(torch.nn.Linear(4, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, len(MOVES))).to(dev)
    scale = torch.tensor([1.0 / R, 1.0 / R, 1.0, 1.0], **f32)
    Xt = torch.as_tensor(X, **f32) * scale
    Nt = {a: torch.as_tensor(N[a], **f32) * scale for a in MOVES}
    Dn = {a: torch.as_tensor((N[a] == goal).all(1), device=dev) for a in MOVES}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    keep = torch.as_tensor(np.ones(len(X), bool) if held is None else ~held, device=dev)
    loss = None
    for it in range(iters):
        with torch.no_grad():
            Y = torch.stack([1.0 + torch.where(Dn[a], torch.zeros((), **f32), net(Nt[a]).min(1).values)
                             for a in MOVES], 1)
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 if it >= 0.7 * iters else 1.0)
        for _ in range(steps):
            loss = ((net(Xt) - Y) ** 2)[keep].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net = net.cpu().eval()
    if held is not None:
        return net, float(loss.detach())
    QCACHE[key] = (net, float(loss.detach()))
    return QCACHE[key]


def q_of(net, X):
    import torch
    with torch.no_grad():
        Xs = torch.as_tensor(X, dtype=torch.float32) * torch.tensor([1.0 / R, 1.0 / R, 1.0, 1.0])
        return np.concatenate([net(Xs[s:s + 65536]).numpy() for s in range(0, len(Xs), 65536)])


def exact_open(T, X, wide=3 * R):
    """Evaluator for the gate: the fewest moves to each target placement in open space (breadth first over a
    wider square with the same transformations)."""
    W = np.array([(x, y, hx, hy) for x in range(-wide, wide + 1) for y in range(-wide, wide + 1)
                  for hx, hy in HEADS], np.int64)
    index = {tuple(r): i for i, r in enumerate(W.tolist())}
    N = moved(T, W.astype(np.float32))
    succ = np.stack([[index.get(tuple(int(v) for v in r), -1) for r in N[a]] for a in MOVES], 1)
    V = np.full(len(W), INF)
    V[index[(0, 0, 0, 1)]] = 0
    for _ in range(8 * wide):
        nv = V.copy()
        for a in range(len(MOVES)):
            ok = succ[:, a] >= 0
            nv[ok] = np.minimum(nv[ok], 1 + V[succ[ok, a]])
        if (nv == V).all():
            break
        V = nv
    return np.array([V[index[tuple(int(v) for v in r)]] for r in X])


def read_q(Q, X=None):
    """V in whole steps (0 at the target itself), and the first of forward, left, right whose Q is V."""
    Qr = np.rint(Q).astype(np.int64)
    V = Qr.min(1)
    first = np.full(len(Q), -1, np.int64)
    for a in reversed(ORDER):
        first[Qr[:, a] == V] = a
    if X is not None:
        at = (X == np.array([0, 0, 0, 1])).all(1)
        V[at], first[at] = 0, -1
    return V, first


# ---------------------------------------------------------------- placements: V, first moves and routes

def geometry(m, net):
    """For every pair of placements (c, p): V, the first move and the route (the tokens forward steps onto)
    of the approach from c to p. Kept on the model; the workers read them."""
    n = len(m.A)
    pos = np.array([(px - R, R - py) for px, py in zip(m.px, m.py)], np.int64)
    hd = np.array([(TK.DIRS[d][0], -TK.DIRS[d][1]) for d in m.pd], np.int64)
    rt = np.stack([hd[:, 1], -hd[:, 0]], 1)
    d = pos[None, :, :] - pos[:, None, :]
    X = np.stack([(d * rt[:, None]).sum(-1), (d * hd[:, None]).sum(-1),
                  (hd[None] * rt[:, None]).sum(-1), (hd[None] * hd[:, None]).sum(-1)], -1).reshape(-1, 4)
    V, first = read_q(q_of(net, X.astype(np.float32)))
    V, first = V.reshape(n, n), first.reshape(n, n)
    np.fill_diagonal(V, 0)
    np.fill_diagonal(first, -1)
    nxt = np.array(m.nxt, np.int64)[:, :len(MOVES)]
    fidx = np.array(m.fidx, np.int64)
    cur, tgt = np.repeat(np.arange(n), n), np.tile(np.arange(n), n)
    L = int(V.max()) + 1
    toks = np.full((n * n, L), -1, np.int16)
    k = np.zeros(n * n, np.int64)
    ok = np.ones(n * n, bool)
    for _ in range(L + 1):
        act = np.flatnonzero((cur != tgt) & ok)
        if not len(act):
            break
        mv = first[cur[act], tgt[act]]
        bad = mv < 0
        ok[act[bad]] = False
        act, mv = act[~bad], mv[~bad]
        fw = mv == FWD
        toks[act[fw], k[act[fw]]] = fidx[cur[act[fw]]]
        k[act[fw]] += 1
        nc = nxt[cur[act], mv]
        ok[act[nc < 0]] = False
        cur[act[nc >= 0]] = nc[nc >= 0]
    ok &= cur == tgt
    m.V, m.first, m.routes, m.route_ok = V, first, toks.reshape(n, n, L), ok.reshape(n, n)
    m.nxt_arr, m.fidx_arr, m.cidx_arr = nxt, fidx, np.array(m.cidx, np.int64)
    return {"placement_pairs": int(n * n), "routes_reaching_their_target": round(float(ok.mean()), 6),
            "longest_route_forward_steps": int(k.max()), "largest_V": int(V.max())}


def build(m):
    """Card 044's tokens, then System 1's network and the placements' V, first moves and routes."""
    TK.build_tokens(m)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = {a: (np.array(v["M"], np.int64), np.array(v["b"], np.int64))
         for a, v in zip(MOVES, (m.tokens_report["moves"][VP.KNAME[a]] for a in MOVES))}
    t0 = time.monotonic()
    net, loss = train_q(T, dev)
    rep = {"q_seconds": round(time.monotonic() - t0, 1), "q_final_loss": round(loss, 6)}
    for name, X in (("view", targets()), ("trained", targets(2 * R))):
        Q = q_of(net, X)
        V, _ = read_q(Q, X)
        Ve = exact_open(T, X)
        far = ~(X == np.array([0, 0, 0, 1])).all(1)
        rep[f"V_exact_share_{name}"] = round(float((V == Ve).mean()), 6)
        rep[f"V_largest_error_{name}"] = int(np.abs(V - Ve).max())
        rep[f"Q_min_largest_error_{name}"] = round(float(np.abs(Q.min(1) - Ve)[far].max()), 4)
    rep.update(geometry(m, net))
    m.tokens_report["walking"] = rep


# ---------------------------------------------------------------- System 2: chains of clear approaches

class Chains:
    """Least predicted steps from each standing placement (S) to the need's placements through clear approaches
    (W: V where the route is clear, else INF), with at most MAXD waypoints."""

    def __init__(self, S, W, Pm):
        self.S, self.W = S, W
        H0 = np.where(Pm, 0, W[:, Pm].min(1) if Pm.any() else INF)
        self.Pm, self.Hs = Pm, [np.minimum(H0, INF)]
        for _ in range(G.MAXD):
            H = self.Hs[-1]
            Hn = np.minimum(H, (W + H[None]).min(1))
            if (Hn == H).all():
                break
            self.Hs.append(Hn)

    def cost(self, i):
        return int(self.Hs[-1][i])

    def hop(self, i, k=None):
        """The next placement on a least chain from S[i] (index into S) with at most k waypoints: (its index,
        whether it is a need's placement, the waypoints left). The chain takes as few waypoints as its cost
        allows, so it has at most len(Hs) hops."""
        k = len(self.Hs) - 1 if k is None else k
        h = self.Hs[k][i]
        k = next(q for q in range(k + 1) if self.Hs[q][i] == h)
        if k == 0:
            return int(np.flatnonzero(self.Pm & (self.W[i] == h))[0]), True, 0
        return int(np.flatnonzero(self.W[i] + self.Hs[k - 1] == h)[0]), False, k - 1


class Walk(TK.TPlan):
    """Walking through conditions (card 045)."""

    KIND = ("approach", "waypoint")

    def __init__(self, W):
        super().__init__(W)
        self.wmemo, self.hmemo = {}, {}

    def forget(self):
        super().forget()
        self.wmemo.clear()
        self.hmemo.clear()

    def walkable(self, fid):
        """Per token id: recall's forward kind predicts that stepping onto it moves the agent."""
        r = self.wmemo.get(("w", fid))
        if r is None:
            f = self.facts[fid]
            u, inv = np.unique(f, return_inverse=True)
            r = self.wmemo[("w", fid)] = np.array([TK.free_of(int(h)) for h in u])[inv.ravel()]
        return r

    def openable(self, fid):
        """Per token id: not walkable, and memory has seen a thing like it made walkable."""
        r = self.wmemo.get(("o", fid))
        if r is None:
            f = self.facts[fid]
            u, inv = np.unique(f, return_inverse=True)
            r = self.wmemo[("o", fid)] = np.array([self.openable_h(int(h)) for h in u])[inv.ravel()]
        return r

    def openable_h(self, h):
        """Per thing, until memory changes."""
        r = self.wmemo.get(("oh", h))
        if r is None:
            r = self.wmemo[("oh", h)] = h != TK.ABSENT and not TK.free_of(h) and bool(self.W.openable(h))
        return r

    def face_open(self, fid, need):
        """The need's placements whatever they stand on; a chain keeps those standing on a token walkable in
        the situation it assumes (with the door open, the door's tile)."""
        k = ("open", fid, need, self.version)
        r = self.fmemo.get(k)
        if r is None:
            _, a, u, j = need
            M = F.M
            f = self.facts[fid]
            held = int(f[HELD])
            T_ = [j] if j is not None else self.showing(f, u)
            r = self.fmemo[k] = self.intern_p([p for i in T_ if (fid, a, i) not in self.refused for p in M.facing[i]
                                               if (a, i, p, held) not in self.failed])
        return r

    def routes(self, fid):
        """Per situation, over the placements standing on a token that is walkable or could be opened: each
        approach's route, how many of its tokens are not walkable now, and the first of them."""
        k = ("routes", fid)
        r = self.hmemo.get(k)
        if r is None:
            M = F.M
            w, op = self.walkable(fid), self.openable(fid)
            S = np.flatnonzero((w | op)[M.cidx_arr])
            R_ = M.routes[np.ix_(S, S)]
            nb = ~np.append(w, True)[R_]                                     # -1 (no token) reads walkable
            first = np.take_along_axis(R_, nb.argmax(-1)[..., None], -1)[..., 0]
            r = self.hmemo[k] = (S, M.route_ok[np.ix_(S, S)], nb.sum(-1), first, M.V[np.ix_(S, S)],
                                 {int(p): i for i, p in enumerate(S)})
        return r

    def clear(self, fid, key):
        """The clear approaches (V where every token on the route is walkable, else INF) between standing
        placements, now (key "now") or with one token j counted walkable (key ("opened", j)); shared by every
        need in the same situation."""
        k = ("clear", fid, key)
        r = self.hmemo.get(k)
        if r is None:
            M = F.M
            S, ok, nb, first, V, ix = self.routes(fid)
            w = self.walkable(fid)
            if key == "now":
                stand, ok = w[M.cidx_arr[S]], ok & (nb == 0)
            else:
                j = key[1]
                stand, ok = w[M.cidx_arr[S]] | (M.cidx_arr[S] == j), ok & ((nb == 0) | ((nb == 1) & (first == j)))
            ok = ok & stand[:, None] & stand[None, :]
            Wm = np.where(ok, V, INF)
            np.fill_diagonal(Wm, INF)
            r = self.hmemo[k] = (S, Wm, stand, ix)
        return r

    def _plan(self, fid, pid, w, key):
        S, Wm, stand, ix = self.clear(fid, key)
        P = self.psets[pid]
        ch = Chains(S, Wm, np.isin(S, np.fromiter(P, np.int64, len(P))) & stand)
        ch.ix = ix
        return ch

    def plan(self, fid, pid, w=None, key="now"):
        k = (fid, pid, key)
        r = self.hmemo.get(k)
        if r is None:
            r = self.hmemo[k] = self._plan(fid, pid, self.walkable(fid) if w is None else w, key)
        return r

    def cost_of(self, ch, c):
        i = ch.ix.get(int(c))
        return INF if i is None else ch.cost(i)

    def step_of(self, ch, c):
        """(move, kind, cost) of the first approach of a least chain from placement c; None if none."""
        i = ch.ix.get(int(c))
        if i is None or ch.cost(i) >= INF:
            return None
        j, direct, _ = ch.hop(i)
        return int(F.M.first[c, ch.S[j]]), self.KIND[0] if direct else self.KIND[1], ch.cost(i)

    def stepped(self, ch, c):
        """The tokens the least chain from c steps onto, in order."""
        out, i, k = [], ch.ix.get(int(c)), None
        if i is None or ch.cost(i) >= INF:
            return out
        while not ch.Pm[i]:
            j, direct, k = ch.hop(i, k)
            out += [int(t) for t in F.M.routes[ch.S[i], ch.S[j]] if t >= 0]
            i = j
            if direct:
                break
        return out

    def reach(self, st, pid):
        k = (st, pid)
        if k in self.rmemo:
            return self.rmemo[k]
        P = self.psets[pid]
        r = (None, None)
        if not P:
            pass
        elif st[1] in P:
            r = (G.HERE, "here")
        else:
            s = self.step_of(self.plan(st[0], pid), st[1])
            if s is not None:
                r = (s[0], s[1])
        self.rmemo[k] = r
        return r

    def cost(self, st, pid):
        return self.cost_of(self.plan(st[0], pid), st[1])

    def walk(self, need, st, depth, chain, protect, faces):
        pid = self.face_pid(st[0], need)
        r, kind = self.reach(st, pid)
        if r is not None and r != G.HERE:
            return G.Result(r, [need, ("walk", kind)], (self.cost(st, pid), 0))
        if r == G.HERE or depth + 1 > G.MAXD:
            return None
        # no chain: a token memory has seen things like made walkable (a door), which alone, counted walkable,
        # gives a chain that crosses it, becomes the condition ("walk", j); the least such chain first
        fid = st[0]
        w, op = self.walkable(fid), self.openable(fid)
        if not op.any() or not w[F.M.cidx[st[1]]]:
            return None
        pid = self.face_open(fid, need)
        cands = []
        for j in np.flatnonzero(op).tolist():
            w2 = w.copy()
            w2[j] = True
            ch = self.plan(fid, pid, w2, ("opened", j))
            n = self.cost_of(ch, st[1])
            if n < INF and j in self.stepped(ch, st[1]):
                cands.append((n, j))
        for _, j in sorted(cands):
            c = ("walk", j)
            if c in chain:
                continue
            res = self.solve(c, st, depth + 1, chain + [need], protect, faces + [need])
            if res is not None:
                res.trace = [need, ("walk", "blocked")] + res.trace
                return res
        return None


class Random(Walk):
    """The floor: the agent's move toward any placement is one of forward, left, right at random."""

    KIND = ("random", "random")

    def step_of(self, ch, c):
        n = self.cost_of(ch, c)
        if n >= INF:
            return None
        return int(self.rng.choice(ORDER)), "random", n

    @property
    def rng(self):
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(45)
        return self._rng


class Exact(Walk):
    """Upper bound: the fewest steps over placements by breadth-first search on the agent's own tokens (forward
    only onto walkable tokens), in place of V and the chains."""

    KIND = ("exact", "exact")

    def _plan(self, fid, pid, w, key):
        M = F.M
        n = len(M.A)
        succ = M.nxt_arr.copy()
        ahead = M.fidx_arr
        succ[:, FWD] = np.where((ahead >= 0) & w[np.maximum(ahead, 0)], succ[:, FWD], -1)
        stand = w[M.cidx_arr]
        succ[~stand] = -1
        dist = np.full(n, INF)
        P = np.fromiter(self.psets[pid], np.int64, len(self.psets[pid]))
        dist[P[stand[P]]] = 0
        for _ in range(4 * n):
            cand = np.where(succ >= 0, dist[np.maximum(succ, 0)] + 1, INF).min(1)
            nd = np.where(stand, np.minimum(dist, cand), INF)
            if (nd == dist).all():
                break
            dist = nd
        ex = dataclasses.make_dataclass("ExactPlan", ["dist", "succ"])(dist, succ)
        return ex

    def step_of(self, ex, c):
        d = ex.dist[c]
        if d >= INF or d == 0:
            return None
        for a in ORDER:
            y = ex.succ[c, a]
            if y >= 0 and ex.dist[y] == d - 1:
                return a, "exact", int(d)
        return None

    def stepped(self, ex, c):
        out = []
        for _ in range(4 * len(ex.dist)):
            s = self.step_of(ex, c)
            if s is None:
                break
            if s[0] == FWD:
                out.append(int(F.M.fidx_arr[c]))
            c = ex.succ[c, s[0]]
        return out

    def cost_of(self, ex, c):
        return int(ex.dist[c])


# ---------------------------------------------------------------- counts: imagined moves and time in walking

class Counted:
    """Counts the moves walking imagines (fresh step predictions inside reach and connected) and its time."""

    def __init__(self, W):
        super().__init__(W)
        self.walk_moves, self.walk_seconds, self.plan_seconds, self._walking = 0, 0.0, 0.0, 0
        PLANS.append(self)

    def step(self, st, a):
        if self._walking and a in MOVES and (st, a) not in self.steps:
            self.walk_moves += 1
        return super().step(st, a)

    def _timed(self, fn, *a, **k):
        if self._walking:
            return fn(*a, **k)
        self._walking += 1
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            self.walk_seconds += time.perf_counter() - t0
            self._walking -= 1

    def reach(self, st, pid):
        return self._timed(super().reach, st, pid)

    def connected(self, st):
        return self._timed(super().connected, st)

    def plan(self, *a, **k):
        return self._timed(super().plan, *a, **k)

    def choose(self, st):
        t0 = time.perf_counter()
        try:
            return super().choose(st)
        finally:
            self.plan_seconds += time.perf_counter() - t0


class Plan045(Counted, Walk):
    pass


class PlanExact(Counted, Exact):
    pass


class PlanRandom(Counted, Random):
    pass


class Plan044(Counted, TK.TPlan):
    pass


PLAN = {"045": Plan045, "exact": PlanExact, "044": Plan044, "random": PlanRandom}


# ---------------------------------------------------------------- the evaluator's how soon (criterion 3)

HOME = np.zeros((TK.NT, 2), np.int64)
for (hx_, hy_), i_ in TK.ID_AT.items():
    HOME[i_] = (hx_, hy_)


def cell_of(lay, j):
    """The simulator's cell of token j (evaluator only)."""
    x0, y0, d0 = lay.start
    a, r = ld.DIR_VEC[d0], ld.DIR_VEC[(d0 + 1) % 4]
    hx, hy = HOME[j]
    return (x0 + hy * a[0] + hx * r[0], y0 + hy * a[1] + hx * r[1])


def true_steps(lay, s, target):
    """Fewest moves until the agent faces the cell, stepping only onto empty cells and the open door."""
    start = (s[0], s[1], s[2])
    seen, frontier, dist = {start}, [start], 0
    while frontier:
        nxt = []
        for x, y, d in frontier:
            dx, dy = ld.DIR_VEC[d]
            if (x + dx, y + dy) == target:
                return dist
            c = ld.cell(lay, s, (x + dx, y + dy))
            moves = [(x, y, (d - 1) % 4), (x, y, (d + 1) % 4)]
            if c == "empty" or (c == "door" and s[6]):
                moves.append((x + dx, y + dy, d))
            for t in moves:
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier, dist = nxt, dist + 1
    return None


def howsoon(pl, st, lay, s, out):
    """For every thing in view (a token whose tile is neither floor nor wall, evaluator names): the predicted
    steps until it is faced, against the evaluator's."""
    W = VP.WORLD
    f = pl.facts[st[0]]
    A = F.M.A[st[1]][:NV]
    for i in np.unique(A[(A >= 0) & (np.arange(NV) != CENTRE)]).tolist():
        h = int(f[i])
        if h == TK.ABSENT or W.judge.name(h) in ("floor", "wall"):
            continue
        truth = true_steps(lay, s, cell_of(lay, i))
        if truth is None:
            continue
        pl._walking += 1                              # not counted as walking's time
        try:
            pred = pl.cost(st, pl.face_pid(st[0], ("face", -1, h, i)))
        finally:
            pl._walking -= 1
        out[(int(min(pred, 999)), int(truth))] += 1


def unseen(lay):
    return lay.size != 8 or lay.goal[0] < lay.wall_x


def spearman(pairs):
    """Rank correlation of weighted (predicted, true) pairs, average ranks for ties."""
    if not pairs:
        return None
    p = np.repeat([a for a, _, _ in pairs], [n for _, _, n in pairs]).astype(np.float64)
    t = np.repeat([b for _, b, _ in pairs], [n for _, _, n in pairs]).astype(np.float64)

    def rank(v):
        u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        return (start + (cnt - 1) / 2.0)[inv]
    rp, rt = rank(p), rank(t)
    if rp.std() == 0 or rt.std() == 0:
        return None
    return float(np.corrcoef(rp, rt)[0, 1])


# ---------------------------------------------------------------- acting: card 038's loop with the counts

def _act_job(job):
    """Card 038's _act_job, recording walking's imagined moves and time, steps through doors as conditions,
    and on unseen rooms the how-soon pairs."""
    idx, layouts, online = job
    W = VP.WORLD
    M = F.M
    W.reset()
    out = []
    for pos, (i, lay) in enumerate(zip(idx, layouts)):
        t0 = time.monotonic()
        rng = np.random.default_rng(1000 + int(i))
        pl = VP.VPlan(W)
        s = ld.start_state(lay)
        V = VP.see(lay, s)
        facts, st, _, _ = pl.observe(None, V)
        rec = {"position": pos, "done": False, "steps": VP.BUDGET, "random": 0, "pred_wrong": 0, "failed_acts": 0,
               "max_mismatch": 0.0, "walk": Counter(), "door_steps": 0}
        hs, t_eval = Counter(), 0.0
        check = unseen(lay) and hasattr(pl, "cost")
        for t in range(VP.BUDGET):
            if check:
                te = time.monotonic()
                howsoon(pl, st, lay, s, hs)
                t_eval += time.monotonic() - te
            res = pl.choose(st)
            if res is None:
                a = int(rng.integers(6))
                rec["random"] += 1
            else:
                a = res.action
                rec["walk"]["act" if res.trace[-1][0] == "act" else res.trace[-1][1]] += 1
                rec["door_steps"] += ("walk", "blocked") in res.trace
            pred = pl.step(st, a)
            u, held = pl.front_held(st)
            if a in MOVES:
                pcat = M.move_out[a][u]
            else:
                pcat = W.outcome(a, u, held, pl.ctx_id(st[0], st[1]))[0]
            s2, end = ld.step(lay, s, a)
            V2 = VP.see(lay, s2)
            facts, st2, miss, _ = pl.observe(facts, V2, end, prefer=pred[1])
            rec["max_mismatch"] = max(rec["max_mismatch"], miss)
            if a in MOVES:
                rcat = VP.ENDED if end else (VP.CHANGED if (V[:NV] != V2[:NV]).any() else VP.UNCHANGED)
            else:
                rcat = int(V[FRONT] != V2[FRONT]) + 2 * int(V[HELD] != V2[HELD])
            if rcat != pcat:
                rec["pred_wrong"] += 1
                if a not in MOVES:
                    pl.failed.add((a, M.fidx[st[1]], st[1], held))
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
        rec["seconds"] = time.monotonic() - t0 - t_eval            # the evaluator's how-soon check excluded
        rec["walk_moves"], rec["walk_seconds"], rec["plan_seconds"] = pl.walk_moves, pl.walk_seconds, pl.plan_seconds
        rec["howsoon"] = [[p, q, n] for (p, q), n in hs.items()]
        out.append(rec)
    return out


_act_arm = VP.act_arm


def act_arm(pool, test, online=True):
    res, recs = _act_arm(pool, test, online)
    hs = Counter()
    for r in recs:
        for p, q, n in r["howsoon"]:
            hs[(p, q)] += n
    pairs = [[p, q, n] for (p, q), n in sorted(hs.items())]
    plan = sum(r["plan_seconds"] for r in recs)
    res["walking"] = {
        "imagined_moves_per_move": round(sum(r["walk_moves"] for r in recs) / max(res["moves"], 1), 3),
        "share_of_planning_time": round(sum(r["walk_seconds"] for r in recs) / plan, 4) if plan else None,
        "steps_through_door_conditions": int(sum(r["door_steps"] for r in recs)),
        "howsoon_pairs": int(sum(n for _, _, n in pairs)),
        "howsoon_rank_correlation": None if not pairs else round(spearman(pairs), 4),
        "howsoon": pairs}
    return res, recs


# ---------------------------------------------------------------- unseen rooms

_ego = VP.Kd.ego


def ego(codes, st):
    """Card 027's view for rooms smaller than 8: the top-down codes padded with wall to 8 x 8 (the view shows
    wall beyond the grid), the held row kept."""
    if codes.shape[1:] != (9, 8):
        n = codes.shape[2]
        pad = np.full((len(codes), 9, 8), VP.Kd.WALL, codes.dtype)
        pad[:, :n, :n] = codes[:, :n, :n]
        pad[:, 8, 0] = codes[:, n, 0]
        codes = pad
    return _ego(codes, st)


VP.Kd.ego = ego


def mirror(lay):
    n = lay.size
    fx = lambda p: (n - 1 - p[0], p[1])
    x, y, d = lay.start
    return dataclasses.replace(lay, wall_x=n - 1 - lay.wall_x, key=fx(lay.key), distractor=fx(lay.distractor),
                               switch=fx(lay.switch), vase=fx(lay.vase), goal=fx(lay.goal),
                               start=(n - 1 - x, y, {0: 2, 2: 0}.get(d, d)))


def _shortest(job):
    return closer._shortest(job)


def make_unseen(n=100):
    """Per world and room: n layouts (seeded apart from the training and test layouts) and their shortest
    routes (the evaluator's breadth-first search over full states)."""
    out = {}
    pool = F.pool20()
    for wi, world in enumerate(VP.WORLDS):
        out[world] = {}
        for ri, room in enumerate(ROOMS):
            rng = np.random.default_rng(45000 + 10 * wi + ri)
            size = {"room6": 6, "room7": 7, "mirror8": 8}[room]
            lays = [ld.make_layout(size, world, rng) for _ in range(n)]
            if room == "mirror8":
                lays = [mirror(lay) for lay in lays]
            chunks = [list(c) for c in np.array_split(np.array(lays, dtype=object), 40) if len(c)]
            sh = [x for p in pool.map(_shortest, chunks) for x in p]
            assert all(x is not None for x in sh), (world, room)
            out[world][room] = {"test": lays, "shortest_each": sh, "shortest": float(np.mean(sh))}
            print(world, room, "shortest mean", round(float(np.mean(sh)), 2), flush=True)
    pool.close()
    UNSEEN.write_bytes(pickle.dumps(out))


def inject_unseen(layouts=None, layouts_b=None):
    """The unseen rooms as tests of card 038's run (acting only; nothing in them is for held-out effects).
    Card 038's run acts on the first --layouts of each test (--layouts-b in the switch world); the shortest
    route is averaged over the same ones."""
    U = pickle.loads(UNSEEN.read_bytes())
    _rv = VP.run_vectors

    def run_vectors(label, z, parts, data, tests, *a, **k):
        tests = dict(tests)
        for world in U:
            cap = layouts_b if world == "switch" and layouts_b else layouts
            for room in ROOMS:
                X = U[world][room]
                tests[f"{room}_{world}"] = {
                    "test": X["test"], "shortest": float(np.mean(X["shortest_each"][:cap])),
                    "ego0": np.zeros((0, NPL), np.uint8),
                    "ego1": np.zeros((0, NPL), np.uint8), "act": np.zeros(0, np.int64),
                    "term1": np.zeros(0, bool), "cases": {}, "criterion_cases": []}
        return _rv(label, z, parts, data, tests, *a, **k)
    VP.run_vectors = run_vectors
    for world in U:
        for room in ROOMS:
            VP.CO.TESTS[f"{room}_{world}"] = (world, 0)


# ---------------------------------------------------------------- main

def install(walking):
    TK.install()
    F.build_poses = build
    VP.VPlan = PLAN[walking]
    VP._act_job = _act_job
    VP.act_arm = act_arm


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    if "--unseen" in args:
        VP.T.configure()
        make_unseen()
        return
    walking = get("--walking", WALKING)
    assert walking in PLAN
    if "--walking" in args:
        i = args.index("--walking")
        del sys.argv[i + 1:i + 3]
    CS.install()
    install(walking)
    if "--dev" not in args:
        inject_unseen(int(get("--layouts", "0")) or None, int(get("--layouts-b", "0")) or None)
    CR.main()
    out = Path(get("--out", ""))
    if out.is_file() and "--dev" not in args:
        r = json.loads(out.read_text())
        r["note"] = f"Card 045, tools/card045/movement.py, walking {walking}"
        out.write_text(json.dumps(r, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
