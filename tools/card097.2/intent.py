"""Card 097.2: an intention held between steps (WM_INTENT=1). Card 093's design, taken up by the procedural series.

A deliberation (the planner's choose, or the fallback's door next to unseen places) leaves a chain of conditions,
from its top (the goal, or "place j can be walked onto") down to the need its action serves. At the next step,
unless a trigger fires, the agent continues: it plans only the deepest condition of the chain that does not hold yet
(the conditions below it are met), with the planner's own solve. Triggers (any one: deliberate from the top):
  1. the chain is done: its top condition holds;
  2. surprise: the pose, the token in front or the hand differs from the last action's prediction;
  3. newly seen: a token the agent can act on (not walkable) appears that was not in the facts at deliberation;
  4. cannot continue: planning the deepest unmet condition finds nothing.
"""
import os
import sys

import numpy as np

DEBUG = os.environ.get("WM_SC_DEBUG") == "1"

STATS = {"intent_deliberated": 0, "intent_continued": 0, "trig_done": 0, "trig_surprise": 0, "trig_seen": 0,
         "trig_cannot": 0}


def chain_of(trace):
    """The conditions of a plan's trace, top first (the markers ("walk", kind) and ("act", a) left out)."""
    out = []
    for c in trace:
        if c[0] == "act" or (c[0] == "walk" and isinstance(c[1], str)):
            continue
        out.append(c)
    return out


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        VP, F = T.VP, T.VP.F
        TRY = sys.modules.get("trying")
        PV = sys.modules["partial"]
        cls = T.PLANNER["cls"] or T.S7.Plan047
        choose0, solve0, fallback0 = cls.choose, cls.solve, PV.fallback

        def actionable(pl, st):
            f = pl.facts[st[0]]
            return {int(u) for u in np.unique(f[:VP.NV]).tolist() if u >= 0 and not F.M.free[int(u)]}

        def expect(pl, st, a):
            s2 = pl.step(st, a)
            return (s2[1],) + tuple(pl.front_held(s2))

        def remember(pl, st, res):
            ch = chain_of(res.trace)
            pl._intent = {"chain": ch, "seen": actionable(pl, st)} if ch else None
            if ch:
                pl._intent["expect"] = expect(pl, st, res.action)

        def triggered(pl, st, I):
            if (st[1],) + tuple(pl.front_held(st)) != I["expect"]:
                STATS["trig_surprise"] += 1
                return True
            if actionable(pl, st) - I["seen"]:
                STATS["trig_seen"] += 1
                return True
            if pl.hold(I["chain"][0], st):
                STATS["trig_done"] += 1
                return True
            return False

        def choose(self, st):
            if getattr(self, "_in_fallback", False):    # card 072's guesses plan inside the fallback: as before
                return choose0(self, st)
            I = getattr(self, "_intent", None)
            if I is not None and not triggered(self, st, I):
                ch = I["chain"]
                for i in range(len(ch) - 1, -1, -1):
                    c = ch[i]
                    if c[0] == "face" or self.hold(c, st):
                        continue
                    prot = [x for x in ch[i + 1:] if x[0] != "face" and self.hold(x, st)]   # met needs below it
                    res = solve0(self, c, st, 1, list(ch[:i]), prot, [])
                    if res is not None:
                        I["chain"] = ch[:i] + chain_of(res.trace)            # the new plan below c
                        if DEBUG:
                            print(f"   continue {str(c)[:60]} chain {[str(x)[:30] for x in I['chain'][:5]]}", flush=True)
                        STATS["intent_continued"] += 1
                        I["expect"] = expect(self, st, res.action)
                        return res
                    break
                STATS["trig_cannot"] += 1
            self._intent = None
            STATS["intent_deliberated"] += 1
            self._capture = None
            res = choose0(self, st)
            if res is not None:
                remember(self, st, res)
                if DEBUG:
                    print(f"   deliberated: chain {[str(x)[:40] for x in chain_of(res.trace)]}", flush=True)
            return res

        def solve(self, c, st, depth, chain, protect, faces):
            res = solve0(self, c, st, depth, chain, protect, faces)
            if res is not None and getattr(self, "_capture", 0) is None and c[0] == "walk" and depth == 1:
                self._capture = res                    # the fallback's door next to unseen places (card 060)
            return res

        def fallback(pl, st, rng, rec):
            pl._capture, pl._in_fallback = None, True
            try:
                a = fallback0(pl, st, rng, rec)
            finally:
                pl._in_fallback = False
            res = pl._capture
            pl._capture = 0
            if res is not None and res.action == a and (TRY is None or TRY.STATE["hyp"] is None):
                remember(pl, st, res)
            return a

        cls.choose, cls.solve, PV.fallback = choose, solve, fallback
        info["intent"] = {"triggers": ["done", "surprise", "newly seen", "cannot continue"]}
        log("card 097.2: an intention held between steps")
        return W, info

    T.setup = setup
