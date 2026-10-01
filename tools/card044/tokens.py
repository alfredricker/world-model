"""Card 044: card 043's planner (with card 042's recall) on the state as tokens.

A token is a what (a tile's vector handle; recall reads its codes from it) and a where: its offset from the
agent (x to the right, y ahead; the place ahead is (0, 1)), or the hand. Every tile the agent has seen is a
token, floor and walls included, and so is the held thing; the held row's other places are tokens fixed beside
the hand. A token exists once seen; a where no token holds shows what the move maps learned that places entering
the view show (wall here; card 028's entering appearance).

A move m sends every where but the hand's to M_m where + b_m, fitted by least squares to the place
correspondences card 028's move maps count (card 044's gate: exact). Since a move moves every token alike, a
situation stores the what of each token (by id) and one transformation g: a token's where is g applied to its
where when first seen. The placements are the transformations reachable by composing the moves' maps, the agent
standing within the first view's 13 x 13 places; nothing is read from card 028's pose table, which is not built.

Readers address tokens by where: ahead is the token at (0, 1), held is the hand's, in view are the tokens whose
where lies within 6 tiles. Recall still compares what is in view as the set of distinct appearances (declared
exception: comparing token sets with positions is the later memory card). Walking is not yet conditional: it
keeps card 024's hand-set closeness, computed from the placements, and still checks moving closer and the move
ways by stepping moves in imagination, now moving tokens (cards 041 and 045). Everything else is card 043's.

Token ids: 0..168 the first view's places (their where at first sight is that place's), 169..181 the held row
(the hand at 169), 182.. the other places within 12 tiles of the first view's centre (a declared bound; the
agent never stands farther than 6 tiles from it here).

  bin/prun python tools/card044/tokens.py --dev --arm A --seeds 399-399 --layouts 30 --out runs/044_dev.json
  bin/prun python tools/card044/tokens.py --arm A --seeds 399-399 --layouts 30 --layouts-b 30 --out runs/044_shakedown_399.json
  bin/prun python tools/card044/tokens.py --arm A --seeds 400-404 --layouts-b 100 --out runs/044_armA.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card043"))
import consistent as CS                                # noqa: E402  (card 043 -> 042 -> 039 -> 038)

CR, SP, VP = CS.CR, CS.SP, CS.VP
F, G = VP.F, VP.G
NV, NPL, W13, CENTRE, FRONT, HELD = VP.NV, VP.NPL, VP.W13, VP.CENTRE, VP.FRONT, VP.HELD
MOVES = VP.MOVES
R = (W13 - 1) // 2
LR = 2 * R
ABSENT = -1                                            # no token: never seen
DIRS = list(F.DIRS)                                    # (column, row) steps in turning order


def where_of(i):
    """A view place as a where: x to the right, y ahead."""
    r, c = divmod(int(i), W13)
    return (c - R, R - r)


WH = np.array([where_of(i) for i in range(NV)], np.int64)


def _lattice():
    at = {tuple(int(v) for v in WH[i]): i for i in range(NV)}
    n = NPL
    for y in range(LR, -LR - 1, -1):
        for x in range(-LR, LR + 1):
            if (x, y) not in at:
                at[(x, y)] = n
                n += 1
    return at, n


ID_AT, NT = _lattice()                                 # where at first sight -> token id; number of ids
LAT = np.array(sorted(ID_AT.values()), np.int64)       # every token id but the held row's


# ---------------------------------------------------------------- moves as transformations; the placements

def fit_moves(m):
    """Per move: where after = M where before + b, by least squares over the map's correspondences."""
    T, rep = {}, {}
    for a in MOVES:
        src = np.asarray(m.src[a])
        q = np.flatnonzero(src[:NV] >= 0)
        q = q[src[q] < NV]
        X = np.concatenate([WH[src[q]], np.ones((len(q), 1))], 1).astype(np.float64)
        sol = np.linalg.lstsq(X, WH[q].astype(np.float64), rcond=None)[0]
        resid = float(np.abs(X @ sol - WH[q]).max())
        Ma, ba = np.rint(sol[:2].T).astype(np.int64), np.rint(sol[2]).astype(np.int64)    # wheres lie on the grid
        T[a] = (Ma, ba)
        rep[VP.KNAME[a]] = {"M": Ma.tolist(), "b": ba.tolist(), "residual": round(resid, 9),
                            "correspondences": int(len(q))}
    return T, rep


def _inv(Rm):
    return np.rint(np.linalg.inv(Rm)).astype(np.int64)


def build_tokens(m):
    """In place of card 028's build_poses: the placements from the moves' transformations, with the
    attributes card 028's poses gave the planner (A: the token each view place shows; cidx, fidx: the token
    stood on and the token ahead; nxt; facing; px, py, pd: the agent's place and heading at first sight)."""
    T, rep = fit_moves(m)
    ents = np.concatenate([np.asarray(m.ent[a])[:NV][np.asarray(m.src[a])[:NV] < 0] for a in MOVES])
    u, n = np.unique(ents, return_counts=True)
    m.unseen = int(u[n.argmax()]) if len(u) else 0
    I = (np.eye(2, dtype=np.int64), np.zeros(2, np.int64))
    key = lambda g: (tuple(g[0].ravel().tolist()), tuple(g[1].tolist()))
    home = lambda g: _inv(g[0]) @ (-g[1])               # where the agent stands, in first-sight wheres
    gs, index, nxt = [I], {key(I): 0}, []
    i = 0
    while i < len(gs):
        Rm, t = gs[i]
        row = [-1] * 6
        for a in MOVES:
            Ma, ba = T[a]
            g2 = (Ma @ Rm, Ma @ t + ba)
            if np.abs(home(g2)).max() > LR - R:
                continue
            k = key(g2)
            j = index.get(k)
            if j is None:
                j = index[k] = len(gs)
                gs.append(g2)
            row[a] = j
        nxt.append(row)
        i += 1
    As, px, py, pd = [], [], [], []
    for Rm, t in gs:
        Ri = _inv(Rm)
        hw = (WH - t) @ Ri.T                            # each view place's where at first sight
        A = np.full(NPL, -1, np.int64)
        A[:NV] = [ID_AT.get((int(x), int(y)), -1) for x, y in hw]
        A[NV:] = np.arange(NV, NPL)
        As.append(A)
        hx, hy = home((Rm, t))
        ax, ay = Ri @ np.array([0, 1])
        px.append(int(hx) + R)
        py.append(R - int(hy))
        pd.append(DIRS.index((int(ax), -int(ay))))
    m.A = np.stack(As)
    m.Ent = np.full(m.A.shape, m.unseen, np.int64)
    m.nxt = nxt
    m.cidx = [int(A[CENTRE]) for A in As]
    m.fidx = [int(A[FRONT]) for A in As]
    m.efront = [m.unseen] * len(As)
    m.hidx = [HELD] * len(As)
    m.chg = [tuple(int(A[q]) for q in m.change_places) for A in As]
    m.px, m.py, m.pd = px, py, pd
    m.poses_at = [[] for _ in range(NT)]
    m.facing = [[] for _ in range(NT)]
    for p, (c, f) in enumerate(zip(m.cidx, m.fidx)):
        m.poses_at[c].append(p)
        if f >= 0:
            m.facing[f].append(p)
    m.all_poses = np.arange(len(As))
    m.pose_conflicts, m.pose_sweeps = 0, 0
    m.tokens_report = {"moves": rep, "placements": len(As), "token_ids": NT, "unseen_where_shows": m.unseen}


# ---------------------------------------------------------------- the planner on tokens

def free_of(h):
    return h != ABSENT and bool(F.M.free[h])


class TPlan(VP.VPlan):
    """Card 043's planner. A situation is (whats, placement, ended): the what of every token by id (ABSENT
    where none has been seen) and the transformation that gives every token's where."""

    def shown(self, f, ids):
        """What the tokens ids show: their what, or the unseen appearance where there is no token."""
        ids = np.asarray(ids)
        v = np.where(ids >= 0, f[np.maximum(ids, 0)], ABSENT)
        return np.where(v == ABSENT, F.M.unseen, v)

    def things(self, f):
        v = np.unique(f[LAT])
        return v[v != ABSENT].tolist()

    def showing(self, f, u):
        """The tokens whose what is u."""
        return LAT[f[LAT] == u].tolist()

    # -- seeing
    def observe(self, world, V, ended=False, prefer=None):
        """Place the view: the placement whose predicted view has the least total L1 distance to it (the
        agent's own place aside); then the tokens in view take the view's whats, the agent's own undrawn."""
        M = F.M
        V = np.asarray(V, np.int32)
        if world is None:
            f = np.full(NT, ABSENT, np.int32)
            f[:NPL] = V
            f[CENTRE] = M.undraw[V[CENTRE]]
            pose, miss, ties = 0, 0.0, 1
        else:
            Gv = self.shown(world, M.A).astype(np.int64)
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
        v = self.shown(self.facts[st[0]], F.M.A[st[1]]).astype(np.int32)
        v[CENTRE] = F.M.draw[v[CENTRE]]
        return v

    def ctx_id(self, fid, pose):
        """What is in view (the tokens within 6 tiles, the agent's own aside), as recall reads it."""
        k = (fid, pose)
        r = self.ctxmemo.get(k)
        if r is None:
            v = self.shown(self.facts[fid], F.M.A[pose][:NV])
            r = self.ctxmemo[k] = self.W.ctx_of(np.unique(v[self.W.keep]))
        return r

    def front_held(self, st):
        """The what of the token ahead and of the hand's."""
        f = self.facts[st[0]]
        fi = F.M.fidx[st[1]]
        u = int(f[fi]) if fi >= 0 else ABSENT
        return (F.M.unseen if u == ABSENT else u), int(f[HELD])

    # -- things named by token
    def imagine_place(self, st, u, j):
        """Token j, else the token ahead if it shows u, else the first token showing u (None: none)."""
        if j is not None:
            return j
        f = self.facts[st[0]]
        fi = F.M.fidx[st[1]]
        if fi >= 0 and f[fi] == u:
            return fi
        where = self.showing(f, u)
        return int(where[0]) if where else None

    def face_pid(self, fid, need):
        """Placements with the named token (or any token showing u) ahead, standing on a token one can walk
        onto, without refused tokens or failed tries (with the same held thing)."""
        k = (fid, need, self.version)
        r = self.fmemo.get(k)
        if r is None:
            _, a, u, j = need
            M = F.M
            f = self.facts[fid]
            held = int(f[HELD])
            T_ = [j] if j is not None else self.showing(f, u)
            P = [p for i in T_ if (fid, a, i) not in self.refused for p in M.facing[i]
                 if free_of(f[M.cidx[p]]) and (a, i, p, held) not in self.failed]
            r = self.fmemo[k] = self.intern_p(P)
        return r

    def standing(self, fid):
        k = ("standing", fid)
        r = self.canmemo.get(k)
        if r is None:
            f = self.facts[fid]
            r = self.canmemo[k] = [p for p in F.M.all_poses.tolist() if free_of(f[F.M.cidx[p]])]
        return r

    def free_at(self, f, x, y):
        """Card 024's closeness asks whether a place (column x, row y of the first view's grid) can be
        walked onto: the token whose where at first sight is there."""
        i = ID_AT.get((x - R, R - y))
        return i is not None and free_of(f[i])

    def achievers(self, c, st):
        """Card 038's achievers, over every token."""
        key = (c, st[0], st[1])
        r = self.achmemo.get(key)
        if r is not None:
            return r
        f = self.facts[st[0]]
        things = self.things(f)
        if c[0] == "end":
            r = [(VP.Ach(VP.FWD, u, None), [("face", VP.FWD, u, None)])
                 for u in things if self.W.move_cat(VP.FWD, u) == VP.ENDED]
        else:
            if c[0] == "walk":
                cands, bit = [(a, int(f[c[1]]), c[1]) for a in (VP.PICK, VP.TOG)], 1
            else:
                cands = [(a, u, None) for a in VP.INTER for u in things if (a, u) != (c[2], c[3])]
                bit = 2 if c[1] == VP.HELDP else 1
            r = []
            for a, u, j in cands:
                for needs in self.conditions(a, u, j, c, st, bit):
                    r.append((VP.Ach(a, u, j), needs + [("face", a, u, j)]))
        self.achmemo[key] = r
        return r

    def walk(self, need, st, depth, chain, protect, faces):
        """Card 038's walking, over every token."""
        pid = self.face_pid(st[0], need)
        r, kind = self.reach(st, pid)
        if r is not None and r != G.HERE:
            return G.Result(r, [need, ("walk", kind)], self.closeness(st, self.psets[pid]))
        if depth + 1 > G.MAXD:
            return None
        f = self.facts[st[0]]
        M = F.M
        stand = int(f[M.cidx[st[1]]])
        if not free_of(stand):
            return None
        cands = []
        for u in self.things(f):
            if M.free[u] or not self.W.openable(u):
                continue
            for j in self.showing(f, u):
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


_world_init = SP.SlotWorld.__init__


def _world_init_tokens(self, *a, **k):
    _world_init(self, *a, **k)
    self.report["tokens"] = self.M.tokens_report


def install():
    F.build_poses = build_tokens
    VP.VPlan = TPlan
    SP.SlotWorld.__init__ = _world_init_tokens


def main():
    CS.install()
    install()
    CR.main()
    args = sys.argv[1:]
    out = Path(next((args[i + 1] for i in range(len(args) - 1) if args[i] == "--out"), ""))
    if out.is_file() and "--dev" not in args:
        r = json.loads(out.read_text())
        r["note"] = "Card 044, tools/card044/tokens.py (card 043's planner on the state as tokens)"
        out.write_text(json.dumps(r, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
