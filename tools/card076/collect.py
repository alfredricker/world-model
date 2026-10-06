"""Card 076: card 074's stored weighings, each with the situation as tokens (report only; nothing the agent decides).

Every weighing card 074's store keeps (a new key in its episode) also records:
- the query: the two needs' kinds and actions, and whether A holds now (a link the release test weighs) or not (an
  unmet sibling the order test weighs);
- every token within the 13 x 13 window around the agent that has been seen, as (handle, dx, dy, marks): its where
  minus the where of the tile the achiever acts on (its j, else the nearest token showing its u), in the agent's
  frame (x to the right, y ahead); marks: 1 the hand (the held row; where the agent stands), 2 named by A, 4 named by
  n, 8 the achiever's target;
- per episode, each handle's code and vector (`handles`), so the scorer needs no encoder.

Nothing here changes a decision: the store's own record is untouched, the hold() it calls is pure.
"""
import sys

import numpy as np

T = sys.modules["tiers"]
VP, TK = T.VP, T.TK
C74 = sys.modules["conflicts"]
TI = sys.modules["ties"]
CR = sys.modules["code_recall"]
STACK = []                                             # the achievers being pursued (innermost last)
SITS = []                                              # per episode: one record per stored weighing
HANDLES = {}
STATS = {"sits_stored": 0}


def _places(self, c, f):
    """The token ids a need names: its place j, else every token showing its tile (the hand for a held tile)."""
    k = c[0]
    j = u = None
    if k == "part":
        u, j = c[3], c[4]
    elif k in ("face", "does"):
        u, j = c[2], c[3]
    elif k == "walk" and isinstance(c[1], (int, np.integer)):
        j = int(c[1])
    elif k in ("has", "hand"):
        return {VP.HELD} | set(np.flatnonzero(f[:TK.NT] == c[1]).tolist())
    if j is not None:
        return {int(j)}
    if u is not None:
        return set(np.flatnonzero(f[:TK.NT] == u).tolist())
    return set()


def kind_act(c):
    """A need's kind and action (card 074's record took a face need's tile for its action)."""
    k = c[0]
    if k == "part":
        return ["part-held" if c[1] == VP.HELDP else "part-view", int(c[2])]
    if k in ("face", "does"):
        return [k, int(c[1])]
    return [k, -1]


def _desc(self, h):
    h = int(h)
    if h not in HANDLES:
        S = self.W.S
        HANDLES[h] = [int(CR.code_id(S, h)), [round(float(x), 4) for x in S.arr[h]]]
    return h


def situation(self, A, n, st, op):
    f = self.facts[st[0]]
    Arow = T.F.M.A[st[1]][:VP.NV]
    seen = {}
    for k in np.flatnonzero(Arow >= 0).tolist():
        i = int(Arow[k])
        if i < len(f) and f[i] != TK.ABSENT:
            seen[i] = (int(TK.WH[k][0]), int(TK.WH[k][1]))
    a, u, j = op
    tgt = None
    if j is not None and int(j) in seen:
        tgt = seen[int(j)]
    elif u is not None:
        cand = [seen[i] for i in seen if f[i] == u]
        if cand:
            tgt = min(cand, key=lambda w: abs(w[0]) + abs(w[1]))
    tx, ty = tgt if tgt is not None else (0, 0)
    pa, pn = _places(self, A, f), _places(self, n, f)
    tid = {int(j)} if j is not None else {i for i in seen if f[i] == u}
    toks = []
    for i, (x, y) in seen.items():
        m = (2 if i in pa else 0) | (4 if i in pn else 0) | (8 if i in tid else 0)
        toks.append([_desc(self, f[i]), x - tx, y - ty, m])
    held = int(f[VP.HELD])
    toks.append([_desc(self, held), -tx, -ty, 1 | (2 if VP.HELD in pa else 0) | (4 if VP.HELD in pn else 0)])
    return toks, tgt is not None


def install():
    store0, pursue0 = C74.store, TI.pursue

    def store(self, A, n, st, outcome):
        k0 = len(C74.WEIGHINGS)
        store0(self, A, n, st, outcome)
        if len(C74.WEIGHINGS) == k0:
            return
        op = STACK[-1] if STACK else (None, None, None)
        toks, found = situation(self, A, n, st, op)
        SITS.append({"A": kind_act(A), "n": kind_act(n), "A_holds": bool(self.hold(A, st)), "outcome": outcome,
                     "fid": int(st[0]), "target_seen": found, "tokens": toks})
        STATS["sits_stored"] += 1

    def pursue(self, c, op, needs, st, depth, chain, protect, faces):
        STACK.append((getattr(op, "a", None), getattr(op, "u", None), getattr(op, "j", None)))
        try:
            return pursue0(self, c, op, needs, st, depth, chain, protect, faces)
        finally:
            STACK.pop()

    C74.store = store
    TI.S7.Plan047.pursue = pursue
    World = VP.World
    _reset = World.reset

    def reset(self):
        _reset(self)
        SITS.clear()
        HANDLES.clear()
        STACK.clear()

    World.reset = reset
