"""Card 074: conflicts and orders read from plans, not from a list of parts.

Koehler and Hoffmann's Definition 8, with "there is no plan" answered by the agent's own planner:

- **Order.** Need n comes before need A when, in the situation A's chain produces (its pick ups, toggles and drops
  imagined by recall, in the order they would be done, from the present), n does not hold, the planner finds no plan
  for n with A protected, and finds one when A may be undone. Pursued first: the first need, in card 029's order,
  that no sibling must precede; if every one must be preceded, card 029's order.
- **Release.** A state the achiever relies on that holds now (a met need; "doing a on u works with the hand and view
  as they are"; card 051's hand link) is protected while a need is pursued, unless that need comes before it by the
  same test, judged in the present.
- **Protection of what comes after.** When need N is pursued before sibling b, an act in N's chain is refused when,
  in the situation it produces followed by N's own achiever, b holds not and can be planned only by undoing N. Card
  029's refusal then has the planner take another way (another place, another achiever).

An order, once chosen for an achiever, is kept between steps while its need is unmet and still gives a plan (LESSONS:
keep a choice between steps): recall's conditions can depend on what is in view, so a test can turn with the agent.

Each test plans in the planner itself; its refusals and version are restored afterwards, so a test leaves nothing in
the plan. Definition 8 asks only whether a plan exists, in any order, so inside a test the planner orders needs as
version 16 did (card 051's threats) and runs no tests of its own. Nothing here names a hand, a view, a tile or a route: what undoes what comes from recall's learned effects.

Every weighing is also stored (report only: here the derivation decides): the two conditions' kinds and actions, the
codes and vectors of the tiles they name and of the tile held, and the outcome.
"""
import sys

import numpy as np

T = sys.modules["tiers"]
VP = T.VP
TH = sys.modules["threats"]
S7 = TH.S7
G = S7.MV.G
F = S7.MV.F
CR = sys.modules["code_recall"]
STATS = {"weighed": 0, "ordered": 0, "reordered": 0, "cycles": 0, "released": 0, "refused_after": 0,
         "plan_tests": 0, "plan_tests_cached": 0, "order_kept": 0}
WEIGHINGS = []                                         # per episode (reset by World.reset)
_DEPTH = [0]                                           # > 0 inside a test plan: no nested tests
_hold = None
_SEEN = set()


# ---------------------------------------------------------------- plans as tests

def _test_plan(self, n, st, protect):
    """A plan for n from st with protect kept, or None; the planner's refusals and version are restored."""
    key = (n, st, tuple(protect))
    memo = self.__dict__.setdefault("_c074", {})
    if memo.get("_v") != self.version or memo.get("_n") != len(self.refused):
        memo.clear()
        memo["_v"], memo["_n"] = self.version, len(self.refused)
    if key in memo:
        STATS["plan_tests_cached"] += 1
        return memo[key]
    STATS["plan_tests"] += 1
    refused, version, nref = set(self.refused), self.version, self.n_refused
    _DEPTH[0] += 1
    try:
        if n[0] == "face":
            r = self.walk(n, st, 1, [], list(protect), [])
        else:
            r = self.solve(n, st, 1, [], list(protect), [])
    finally:
        _DEPTH[0] -= 1
        # caches keyed by the version saw the test's refusals: a fresh version keeps them from being reused
        self.version = version if self.version == version else self.version + 1
        self.refused, self.n_refused = refused, nref
    memo["_v"], memo["_n"] = self.version, len(self.refused)
    memo[key] = r is not None
    return memo[key]


def before(self, n, A, stA):
    """n is reasonably ordered before A: where A holds (stA), n does not, and n can be planned only by undoing A."""
    if stA is None or n == A or not self.hold(A, stA) or self.hold(n, stA):
        return False
    if _test_plan(self, n, stA, [A]):
        return False
    return _test_plan(self, n, stA, [])


def produced(self, n, res, st):
    """The situation n's chain produces from st: its interaction acts imagined by recall, deepest first."""
    ops = [k for c, k in getattr(res, "ops", []) if k[0] is not None and k[0] in TH.ACTS]
    if not ops:
        return None
    s = st
    for a, u, j in reversed(ops):
        s2 = self.imagine(s, a, u, j)
        if s2 is None:
            return None
        s = s2
    return s


def own_op(n, res):
    return next((k for c, k in getattr(res, "ops", []) if c == n and k[0] is not None), None)


def hold(self, c, st):
    if c[0] == "after":
        return False
    if c[0] == "does":
        return self.does(st, *c[1:])
    return _hold(self, c, st)


# ---------------------------------------------------------------- weighings stored (report only)

def _token(self, c, st):
    f = self.facts[st[0]]
    k = c[0]
    if k == "part":
        return int(c[3])
    if k in ("face", "does"):
        return int(c[2])
    if k == "hand":
        return int(c[1])
    if k == "walk" and isinstance(c[1], (int, np.integer)):
        return int(f[c[1]])
    return None


def _kind(c):
    if c[0] == "part":
        return "part-held" if c[1] == VP.HELDP else "part-view"
    return c[0]


def _act(c):
    return int(c[2]) if c[0] in ("part", "face", "does") else -1


def _desc(self, h):
    if h is None:
        return None
    S = self.W.S
    return [int(CR.code_id(S, h)), [round(float(x), 4) for x in S.arr[int(h)]]]


def store(self, A, n, st, outcome):
    held = int(self.facts[st[0]][VP.HELD])
    key = (_kind(A), _act(A), _kind(n), _act(n), _token(self, A, st), _token(self, n, st), held, st[0], outcome)
    if key in _SEEN:
        return
    _SEEN.add(key)
    WEIGHINGS.append({"A": [_kind(A), _act(A)], "n": [_kind(n), _act(n)], "uA": _desc(self, _token(self, A, st)),
                      "un": _desc(self, _token(self, n, st)), "held": _desc(self, held), "outcome": outcome})


# ---------------------------------------------------------------- the planner's pursue

def pursue(self, c, op, needs, st, depth, chain, protect, faces):
    if _DEPTH[0] > 0:                                  # inside a test: whether a plan exists, in version 16's order
        return TH.pursue(self, c, op, needs, st, depth, chain, protect, faces)
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
    if not unmet:                                      # every need met: act, unless it breaks what comes after
        a = getattr(op, "a", None)
        if a in TH.ACTS:
            for p in protect:
                if p[0] != "after":
                    continue
                _, N, opN, b = p
                s2 = self.step(st, a)
                sN = self.imagine(s2, *opN) if s2 is not None else None
                if sN is not None and before(self, b, N, sN):
                    self.refused.add((st[0], a, F.M.fidx[st[1]]))
                    self.version += 1
                    self.n_refused += 1
                    STATS["refused_after"] += 1
                    return None
        res = G.Plan.pursue(self, c, op, needs, st, depth, chain, protect, faces)
        return None if res is None else TH._uses(res, links, op)
    held_now = list(dict.fromkeys(prot[len(protect):] + links))
    plans = {}
    for n in unmet:
        keep = []
        for A in held_now:
            rel = before(self, n, A, st)
            store(self, A, n, st, "n first" if rel else "none")
            if rel:
                STATS["released"] += 1
            else:
                keep.append(A)
        plans[n] = TH._one(self, c, n, st, depth, chain, protect + keep, faces, mine)
    if len(unmet) == 1:
        r = plans[unmet[0]]
        return None if r is None else TH._uses(r, links, op)
    STATS["weighed"] += 1
    if plans[unmet[0]] is None:                        # card 029's order: the first unmet need decides
        return None
    waits = {n: False for n in unmet}
    states = {A: produced(self, A, plans[A], st) for A in unmet if plans[A] is not None}
    for A in unmet:
        for n in unmet:
            if n is A or plans.get(n) is None or A not in states:
                continue
            if before(self, n, A, states[A]):
                waits[A] = True
                STATS["ordered"] += 1
                store(self, A, n, st, "n first")
            else:
                store(self, A, n, st, "none")
    pick = next((k for k, n in enumerate(unmet) if not waits[n] and plans[n] is not None), None)
    if pick is None:
        STATS["cycles"] += 1
        pick = 0
    elif pick != 0:
        STATS["reordered"] += 1
    keep_key = (c, (getattr(op, "a", None), getattr(op, "u", None), getattr(op, "j", None)))
    kept = self.__dict__.setdefault("_order_keep", {})
    prev = kept.get(keep_key)
    if prev is not None and prev in unmet and plans.get(prev) is not None and unmet.index(prev) != pick:
        pick = unmet.index(prev)                       # the last step's order, while it still gives a plan
        STATS["order_kept"] += 1
    kept[keep_key] = unmet[pick]
    first = unmet[pick]
    opN = own_op(first, plans[first])
    after = [("after", first, opN, b) for b in unmet if b is not first and plans.get(b) is not None] if opN else []
    if after:                                          # what comes after is kept while `first` is pursued
        keep = [A for A in held_now if not before(self, first, A, st)]
        r = TH._one(self, c, first, st, depth, chain, protect + keep + after, faces, mine)
        if r is not None:
            return TH._uses(r, links, op)
    return TH._uses(plans[first], links, op)


def install():
    global _hold
    _hold = S7.Plan047.hold
    S7.Plan047.pursue = pursue
    S7.Plan047.hold = hold
    World = VP.World
    _reset = World.reset

    def reset(self):
        _reset(self)
        WEIGHINGS.clear()
        _SEEN.clear()

    World.reset = reset
