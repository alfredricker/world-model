"""Card 067: closeness by local propagation over the believed map, in place of card 045's route and chain tables.

Card 045 walked by System 1, a network giving how many steps from one placement to another in open space, and
System 2, chains of approaches whose routes (the tokens each approach steps onto) are clear; both were worked out
for every pair of placements of a fixed lattice and kept (676 placements, about 457,000 pairs). Here walking keeps
nothing between pairs of placements. For one situation and one need:

  closeness. Each placement the agent could stand at (on a token recall's forward kind says moves the agent) takes
             its closeness from its neighbours only: 0 at the need's placements; otherwise one step more than the
             least of the placements its three moves lead to, a forward step counting only onto a walkable token
             (card 044's learned transformations give the neighbours). Iterated to a fixed point (value iteration,
             Tamar et al. 2016's VIN on the agent's own map; Bellman 1957). The agent takes the move that brings it
             closer: forward, then left, then right among equals (card 023's order).
  crossing.  When no placement of the need can be reached, tokens memory has seen made walkable (a door, a thing
             to pick up) count as crossable at one step more (the pick up or toggle). On a least route, the first
             token not walkable now becomes the condition ("walk", j), pursued like any other condition; if it
             cannot be met, the token is left out and the next least route is taken. This replaces card 045's
             search over single tokens and card 051's over pairs.

Everything else is version 14's. Run through card 066's harness:

  bin/prun python tools/card066/tiers.py --tier 1 --n 200 --walking 067 --out runs/067/tier1.json
"""
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card057"))
import partial as PV                                   # noqa: E402

MV = PV.MV
F, G, TK = MV.F, MV.G, MV.TK
INF, ORDER, FWD = MV.INF, MV.ORDER, MV.FWD
CROSS = 1                                              # the pick up or toggle that makes a crossable token walkable
STATS = {"fields": 0, "rounds": 0, "relaxed": 0, "edges": 0, "placements": 0, "crossings_tried": 0,
         "field_seconds": 0.0}


class Field:
    """Closeness to one need over the placements one could stand at in one situation."""

    __slots__ = ("S", "ix", "succ", "cost", "dist")

    def __init__(self, S, ix, succ, cost, dist):
        self.S, self.ix, self.succ, self.cost, self.dist = S, ix, succ, cost, dist


def field(w, cross, P):
    """w: walkable per token; cross: crossable per token (None: none); P: the need's placements."""
    t0 = time.perf_counter()
    M = F.M
    on = M.cidx_arr
    stand = w[on] if cross is None else (w[on] | cross[on])
    S = np.flatnonzero(stand)
    ix = np.full(len(on), -1, np.int64)
    ix[S] = np.arange(len(S))
    nx = M.nxt_arr[S]
    succ = np.where(nx >= 0, ix[np.maximum(nx, 0)], -1)
    ahead = M.fidx_arr[S]
    known = ahead >= 0
    fw_w = known & w[np.maximum(ahead, 0)]
    cost = np.ones(succ.shape, np.int64)
    if cross is None:
        succ[~fw_w, FWD] = -1
    else:
        fw_c = known & cross[np.maximum(ahead, 0)] & ~fw_w
        succ[~(fw_w | fw_c), FWD] = -1
        cost[fw_c, FWD] += CROSS
    dist = np.full(len(S), INF, np.int64)
    p = ix[np.asarray(P, np.int64)] if len(P) else np.zeros(0, np.int64)
    p = np.unique(p[p >= 0])
    dist[p] = 0
    # the moves into each placement (reverse edges), so that only the neighbours of a placement whose closeness
    # just changed are updated: each edge is relaxed a few times, not once per sweep of the whole map
    u, m = np.nonzero(succ >= 0)
    v = succ[u, m]
    order = np.argsort(v, kind="stable")
    u, m, v = u[order], m[order], v[order]
    start = np.searchsorted(v, np.arange(len(S) + 1))
    c_um = cost[u, m]
    front, rounds, relaxed = p, 0, 0
    while len(front):
        lo, hi = start[front], start[front + 1]
        n = hi - lo
        if not n.sum():
            break
        e = np.repeat(lo - np.cumsum(np.r_[0, n[:-1]]), n) + np.arange(n.sum())
        nd = dist[v[e]] + c_um[e]
        tgt = u[e]
        relaxed += len(e)
        before = dist[tgt]
        np.minimum.at(dist, tgt, nd)
        front = np.unique(tgt[dist[tgt] < before])
        rounds += 1
    STATS["fields"] += 1
    STATS["rounds"] += rounds
    STATS["relaxed"] += relaxed
    STATS["edges"] += len(u)
    STATS["placements"] += len(S)
    STATS["field_seconds"] += time.perf_counter() - t0
    return Field(S, ix, succ, cost, dist)


# ---------------------------------------------------------------- Walk's methods, on fields

def _plan(self, fid, pid, w, key):
    P = self.psets[pid]
    P = np.fromiter(P, np.int64, len(P))
    cross = None
    if isinstance(key, tuple) and key[0] == "cross":
        cross = self.openable(fid) & ~w
        if key[1]:
            cross = cross.copy()
            cross[list(key[1])] = False
    return field(w, cross, P)


def cost_of(self, fl, c):
    i = fl.ix[int(c)]
    return INF if i < 0 else int(fl.dist[i])


def step_of(self, fl, c):
    """(move, kind, cost): a move from placement c that brings it closer; None at the need or where none does."""
    i = fl.ix[int(c)]
    if i < 0:
        return None
    d = fl.dist[i]
    if d >= INF or d == 0:
        return None
    for a in ORDER:
        y = fl.succ[i, a]
        if y >= 0 and fl.dist[y] + fl.cost[i, a] == d:
            return a, "closer", int(d)
    return None


def stepped(self, fl, c):
    """The tokens a least route from placement c steps onto, in order."""
    out = []
    M = F.M
    for _ in range(4 * len(fl.S) + 1):
        s = step_of(self, fl, c)
        if s is None:
            break
        if s[0] == FWD:
            out.append(int(M.fidx_arr[c]))
        c = int(fl.S[fl.succ[fl.ix[c], s[0]]])
    return out


def walk(self, need, st, depth, chain, protect, faces):
    pid = self.face_pid(st[0], need)
    r, kind = self.reach(st, pid)
    if r is not None and r != G.HERE:
        return G.Result(r, [need, ("walk", kind)], (self.cost(st, pid), 0))
    if r == G.HERE or depth + 1 > G.MAXD:
        return None
    fid = st[0]
    w, op = self.walkable(fid), self.openable(fid)
    if not op.any() or not w[F.M.cidx[st[1]]]:
        return None
    pid = self.face_open(fid, need)
    left_out = frozenset()
    for _ in range(int(op.sum())):
        fl = self.plan(fid, pid, w, ("cross", left_out))
        if cost_of(self, fl, st[1]) >= INF:
            return None
        crossed = [j for j in stepped(self, fl, st[1]) if not w[j]]
        if not crossed:
            return None
        j = crossed[0]
        c = ("walk", j)
        STATS["crossings_tried"] += 1
        if c not in chain:
            res = self.solve(c, st, depth + 1, chain + [need], protect, faces + [need])
            if res is not None:
                res.trace = [need, ("walk", "blocked")] + res.trace
                return res
        left_out = left_out | {j}
    return None


def build(m):
    """Card 044's tokens and placements; no network and nothing per pair of placements."""
    TK.build_tokens(m)
    m.nxt_arr = np.array(m.nxt, np.int64)[:, :len(MV.MOVES)]
    m.fidx_arr, m.cidx_arr = np.array(m.fidx, np.int64), np.array(m.cidx, np.int64)
    m.tokens_report["walking"] = {"kind": "card 067: closeness by propagation", "placements": int(len(m.A))}


def install():
    F.build_poses = build
    MV.Walk._plan, MV.Walk.cost_of, MV.Walk.step_of, MV.Walk.stepped = _plan, cost_of, step_of, stepped
    MV.Walk.walk = walk
