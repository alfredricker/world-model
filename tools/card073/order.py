"""Card 073: order needs by the states they conflict over (Koehler and Hoffmann's reasonable goal orderings).

Card 051's pursue imagined each unmet need's plan and called a need a threat when its act broke what another need's
plan relied on; when two needs both fill the hand, each threatens the other and card 029's fixed order decides
(card 072's loop: 224 of 508 weighings, 0 reorders). Here, for an achiever's needs:

- **Reasonable order** (Definition 8, tested on conditions as in their Definition 10): a condition B is ordered before
  a state A when, in the situation where A holds, B does not hold and either B is on a part A fixes, or every way to
  B (the planner's achievers for B there) has an unmet need on a part A fixes. Parts: the held tile, the view, a tile
  j ("walk", j). A part need fixes its part; ("does", ...) and ("hand", h) fix the held tile and the view; ("walk", j)
  fixes tile j; a face need fixes no part. B is looked for in the whole chain of a sibling need's plan, at any depth.
- **Order between unmet needs:** a need whose chain holds a B ordered before a sibling need A (judged in the
  situation A's own achiever produces, from the present) is pursued first; the first need, in card 029's order,
  that no sibling must precede is pursued; if every need must be preceded, card 029's order.
- **Protection:** the states the achiever relies on that hold now (its met needs, and the link "doing a on u works
  with the hand and view as they are", card 051's "does") are protected while its unmet needs are pursued, unless a
  B in that need's chain is ordered before them (then they may be undone and re-achieved). Card 029's refusal does
  the rest: a drop onto a tile a face need above it relies on is refused, and the drop is planned elsewhere.

What the agent knows about hands stays in recall (a pick up needs the held part changed; a toggle opens a door with
its key); nothing about hands is written here.
"""
import sys

import numpy as np

T = sys.modules["tiers"]
VP = T.VP
TH = sys.modules["threats"]
S7 = TH.S7
G = S7.MV.G
STATS = {"weighed": 0, "ordered": 0, "reordered": 0, "cycles": 0, "released": 0, "drop_refused": 0}
DEPTHS = {}
_hold = None


def parts_fixed(A):
    k = A[0]
    if k == "part":
        return {"held"} if A[1] == VP.HELDP else {"view"}
    if k in ("does", "hand"):
        return {"held", "view"}
    if k == "walk" and isinstance(A[1], (int, np.integer)):
        return {("tile", int(A[1]))}
    return set()


def part_of(m):
    k = m[0]
    if k == "part":
        return {"held"} if m[1] == VP.HELDP else {"view"}
    if k == "hand":
        return {"held"}
    if k == "walk" and isinstance(m[1], (int, np.integer)):
        return {("tile", int(m[1]))}
    return set()


def conditions_in(res):
    """The conditions of a plan's chain (markers such as ("walk", "blocked") and acts left out)."""
    out = []
    for c in getattr(res, "trace", []):
        if not isinstance(c, tuple) or not c:
            continue
        if c[0] == "part" or (c[0] == "walk" and isinstance(c[1], (int, np.integer))):
            out.append(c)
    return out


def before(self, B, A, stA):
    """B is reasonably ordered before A: where A holds (stA), B does not, and achieving B changes a part A fixes."""
    fixed = parts_fixed(A)
    if not fixed or B == A or stA is None or B[0] not in ("part", "walk"):
        return False
    if self.hold(B, stA):
        return False
    if part_of(B) & fixed:
        return True
    achs = self.achievers(B, stA)
    if not achs:
        return False
    return all(any(part_of(m) & fixed and not self.hold(m, stA) for m in needs if m[0] != "face")
               for _, needs in achs)


def produced(self, n, res, st):
    """The situation n's own achiever produces from the present (n's plan's first recorded choice)."""
    for c, k in getattr(res, "ops", []):
        if c == n and k[0] is not None:
            return self.imagine(st, k[0], k[1], k[2])
    return None


def hold(self, c, st):
    if c[0] == "does":
        return self.does(st, *c[1:])
    return _hold(self, c, st)


def pursue(self, c, op, needs, st, depth, chain, protect, faces):
    met = [n for n in needs if n[0] != "face" and self.hold(n, st)]
    links = met + ([("does", op.a, op.u, op.j, c)] if getattr(op, "a", None) in TH.ACTS else [])
    prot = protect + met
    if TH.HAND_LINK and any(n[0] == "part" and n[1] == VP.VIEWP and n[-1] is None and not self.hold(n, st)
                            for n in needs):
        h = int(self.front_held(st)[1])
        prot = prot + [("hand", h)]
        links = links + [("hand", h)]
    mine = [n for n in needs if n[0] == "face"]
    unmet = [n for n in needs if not self.hold(n, st)]
    if not unmet:                                      # every need met: card 029's act, with its refusal
        n_ref = self.n_refused
        res = G.Plan.pursue(self, c, op, needs, st, depth, chain, protect, faces)
        if res is None and self.n_refused > n_ref and getattr(op, "a", None) == VP.DROP:
            STATS["drop_refused"] += 1
        return None if res is None else TH._uses(res, links, op)
    # each need planned with none of this achiever's states protected, to read its chain; then the states held now
    # (met needs, the hand, the "does" link) are protected while it is pursued, unless its chain holds a condition
    # ordered before them
    held_now = list(dict.fromkeys(prot[len(protect):] + links))
    plans = {}
    for n in unmet:
        r = TH._one(self, c, n, st, depth, chain, protect, faces, mine)
        if r is None:
            plans[n] = None
            continue
        bs = conditions_in(r)
        keep = [A for A in held_now if not any(before(self, B, A, st) for B in bs)]
        STATS["released"] += len(held_now) - len(keep)
        plans[n] = TH._one(self, c, n, st, depth, chain, protect + keep, faces, mine) if keep else r
    if len(unmet) == 1:
        r = plans[unmet[0]]
        return None if r is None else TH._uses(r, links, op)
    STATS["weighed"] += 1
    if plans[unmet[0]] is None:                        # card 029's order: the first unmet need decides
        return None
    after = {n: False for n in unmet}                  # n must wait for a sibling
    for A in unmet:
        rA = plans[A]
        if rA is None:
            continue
        stA = produced(self, A, rA, st)
        for n in unmet:
            if n is A or plans[n] is None:
                continue
            if any(before(self, B, A, stA) for B in [n] + conditions_in(plans[n])):
                after[A] = True
                STATS["ordered"] += 1
                DEPTHS[depth] = DEPTHS.get(depth, 0) + 1
                break
    pick = next((k for k, n in enumerate(unmet) if not after[n] and plans[n] is not None), None)
    if pick is None:
        STATS["cycles"] += 1
        pick = 0
    elif pick != 0:
        STATS["reordered"] += 1
    return TH._uses(plans[unmet[pick]], links, op)


def install():
    global _hold
    _hold = S7.Plan047.hold
    S7.Plan047.pursue = pursue
    S7.Plan047.hold = hold
