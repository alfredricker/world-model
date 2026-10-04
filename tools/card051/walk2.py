"""Card 051, step 4c: a route may need two tokens made walkable, not only one.

Card 045's walking, when no route faces the target, counts one token at a time walkable (one that recall says a
pick up or toggle can make walkable) and takes the cheapest route that crosses it as the condition ("walk", j).
Card 048's two doors, layout 11 (traced 2026-10-04, seed 400): the cell before door green is reachable only
through the cell where key green lies, so reaching key red needs key green picked up and door green opened; no
single token gives a route, and the planner found no chain at all (random actions for the whole episode).

Here, when no single token gives a route, pairs of such tokens are tried (cheapest route first), and the token the
route steps onto first becomes the condition; the next step finds the other as a single token. Everything else is
step 4b's agent.

  bin/prun python tools/card051/walk2.py --chain two_doors --seed 400 --n 30 --out runs/051/s4c_two_400.json
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import commit as CM                                    # noqa: E402

S7 = CM.S7
MV, G = S7.MV, CM.G
F = MV.F
STATS = {"pairs_used": 0}
_walk = MV.Walk.walk


def walk(self, need, st, depth, chain, protect, faces):
    res = _walk(self, need, st, depth, chain, protect, faces)
    if res is not None:
        return res
    pid = self.face_pid(st[0], need)
    r, _ = self.reach(st, pid)
    if r is not None or depth + 1 > G.MAXD:
        return None
    fid = st[0]
    w, op = self.walkable(fid), self.openable(fid)
    if not op.any() or not w[F.M.cidx[st[1]]]:
        return None
    pid = self.face_open(fid, need)
    may = self.may_open(fid, pid, st[1])
    js = [j for j in np.flatnonzero(op).tolist() if j in may]
    cands = []
    for j1, j2 in combinations(js, 2):
        w2 = w.copy()
        w2[j1] = w2[j2] = True
        ch = self.plan(fid, pid, w2, ("opened2", j1, j2))
        n = self.cost_of(ch, st[1])
        if n < MV.INF:
            stepped = self.stepped(ch, st[1])
            on = [j for j in stepped if j in (j1, j2)]
            if len(set(on)) == 2:
                cands.append((n, on[0]))
    for _, j in sorted(set(cands)):
        c = ("walk", j)
        if c in chain:
            continue
        res = self.solve(c, st, depth + 1, chain + [need], protect, faces + [need])
        if res is not None:
            STATS["pairs_used"] += 1
            res.trace = [need, ("walk", "blocked twice")] + res.trace
            return res
    return None


_clear = MV.Walk.clear


def clear(self, fid, key):
    """Card 045's clear approaches, also with two tokens counted walkable (key ("opened2", j1, j2)): each
    approach may cross one of them; a chain through waypoints can cross both."""
    if not (isinstance(key, tuple) and key[0] == "opened2"):
        return _clear(self, fid, key)
    k = ("clear", fid, key)
    r = self.hmemo.get(k)
    if r is None:
        M = F.M
        S, ok, nb, first, V, ix = self.routes(fid)
        js = np.array(key[1:])
        w = self.walkable(fid)
        stand = w[M.cidx_arr[S]] | np.isin(M.cidx_arr[S], js)
        ok = ok & ((nb == 0) | ((nb == 1) & np.isin(first, js)))
        ok = ok & stand[:, None] & stand[None, :]
        Wm = np.where(ok, V, MV.INF)
        np.fill_diagonal(Wm, MV.INF)
        r = self.hmemo[k] = (S, Wm, stand, ix)
    return r


def install():
    CM.install()
    MV.Walk.walk = walk
    MV.Walk.clear = clear


def main():
    install()
    sys.argv[1:1] = ["--recall", "own"]
    CM.TH.IX.CD.main()


if __name__ == "__main__":
    main()
