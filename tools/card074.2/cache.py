"""Card 074.2: card 074.1's arm B with recall's pair distance cached (the same decisions, faster).

Card 043's conditions rank a stored success (held h, view v) by recall's distance to the situation's (held, view),
pair_distance((h, v), (hc, vc)). It reads only four interned handles and set ids and the kind's weights, so it is
cached per kind by those four, and the cache is dropped whenever the weights change. On card 074.1's slowest
episode (tier 2 seed 1002078) the 607,298 calls of 35 steps had 352 distinct keys.

Also records each episode's actions (World.learn_try is called once per step with the action taken), for the
exactness check against card 074.1.
"""
import sys

import numpy as np

T = sys.modules["tiers"]
VP = T.VP
STATS = {"pd_calls": 0, "pd_computed": 0, "hold_calls": 0, "hold_computed": 0}
_FACE = {}


def _has_face(c):
    """A condition nesting a facing need reads refused places (the planner's version): never cached."""
    r = _FACE.get(c)
    if r is None:
        r = _FACE[c] = c[0] == "face" or any(isinstance(x, tuple) and x and _has_face(x) for x in c
                                             if isinstance(x, tuple) and x and isinstance(x[0], str))
    return r
ACTIONS = []


CACHING = __import__("os").environ.get("WM_CACHE") == "1"   # "0": record actions only (the exactness baseline)


def install():
    _record()
    if not CACHING:
        return
    SK = sys.modules["slot_planner"].SlotKind              # the one class that defines pair_distance (card 039)
    f0 = SK.pair_distance

    def pair_distance(self, p, q):
        STATS["pd_calls"] += 1
        lam = self.lam
        memo = self.__dict__.get("_pd_memo")
        if memo is None or memo[0] is not lam or not np.array_equal(memo[1], lam):
            memo = self.__dict__["_pd_memo"] = (lam, np.array(lam, copy=True), {})
        key = (int(p[0]), int(p[1]), int(q[0]), int(q[1]))
        r = memo[2].get(key)
        if r is None:
            STATS["pd_computed"] += 1
            r = memo[2][key] = f0(self, p, q)
        return r

    SK.pair_distance = pair_distance

    # hold for a part condition: doing an act on a tile makes a condition true, checked in the situation its
    # imagined effect produces, recursively. It reads the planner's imagined effects (imemo) only, so it is cached
    # by (condition, situation) and dropped with them in forget().
    P = sys.modules["situations"].Plan047 if "situations" in sys.modules else T.S7.Plan047
    hold0, forget0 = P.hold, P.forget

    def hold(self, c, st):
        if c[0] != "part" or _has_face(c):
            return hold0(self, c, st)
        STATS["hold_calls"] += 1
        memo = self.__dict__.setdefault("_c0742_hold", {})
        k = (c, st)
        r = memo.get(k)
        if r is None:
            STATS["hold_computed"] += 1
            r = memo[k] = hold0(self, c, st)
        return r

    def forget(self):
        forget0(self)
        self.__dict__.pop("_c0742_hold", None)

    P.hold, P.forget = hold, forget


def _record():
    World = VP.World
    _reset, _learn = World.reset, World.learn_try

    def reset(self):
        _reset(self)
        ACTIONS.clear()

    def learn_try(self, Vb, a, *rest, **kw):
        ACTIONS.append(int(a))
        return _learn(self, Vb, a, *rest, **kw)

    World.reset, World.learn_try = reset, learn_try
