"""Card 074.1: found orders always followed, ties kept, and two-part needs split.

On card 074's conflicts (tools/card074/conflicts.py, WM_CONFLICTS=1), which this module replaces in the planner's
pursue:

- **An order the test finds is followed.** Of an achiever's unmet needs with a plan, those no sibling must precede
  are the candidates. One candidate: it is pursued, whatever was kept from the last step.
- **Ties.** Several candidates (no order found among them, or every need must be preceded): the last step's choice for
  this achiever, while its need is unmet, still gives a plan and is a candidate; else the cheaper need: fewer pick
  ups, toggles and drops in its plan, then fewer steps to its first, then card 029's order. Cost changes with every
  step the agent takes, so it decides only fresh choices.
- **Two-part needs split** (WM_SPLIT=1). Card 043 states a stored success (held h, view v) that needs both the hand
  and the view changed as two spliced needs, each checked with the other part fixed to the stored try's value. Here
  it is two independent conditions: ("has", h), card 056's condition (achieved by any act whose learned effect puts h
  in the hand), and card 043's one-part view need, checked with the situation's own hand and view. Both are ordered
  with the achiever's other needs by card 074's rule. Card 051's hand link (the present hand kept when only the view
  is asked for) is not added when the achiever also asks for the hand: the hand is then a need, not a link.

WM_TIES=074 keeps card 074's choice (the kept order overrides the test) and only counts, for the data check.
Nothing here names a hand, a ball or a door: h is the stored success's own held tile.
"""
import os
import sys

T = sys.modules["tiers"]
VP = T.VP
TH = sys.modules["threats"]
C74 = sys.modules["conflicts"]
S7 = TH.S7
G = S7.MV.G
HELDP, VIEWP, HELD = VP.HELDP, VP.VIEWP, VP.HELD
MODE = os.environ.get("WM_TIES", "1")                  # "1": this card's choice; "074": card 074's, counted
SPLIT = os.environ.get("WM_SPLIT") == "1"
STATS = {"weighed": 0, "found": 0, "followed": 0, "overridden": 0, "tie_kept": 0, "tie_cost": 0, "tie_order": 0, "single": 0,
         "cycles": 0, "flips": 0, "steps": 0, "steps_spliced": 0, "split_hand_met": 0, "split_view_held": 0}
_conditions0 = None


def spliced(n):
    return n[0] == "part" and n[-1] is not None


def _has_need(needs):
    return any(n[0] == "has" for n in needs)


def _test_pursue(self, c, op, needs, st, depth, chain, protect, faces):
    """Inside a test plan: version 16's order (card 074), without the hand link when the hand is itself a need."""
    if SPLIT and _has_need(needs):
        TH.HAND_LINK = False
        try:
            return TH.pursue(self, c, op, needs, st, depth, chain, protect, faces)
        finally:
            TH.HAND_LINK = True
    return TH.pursue(self, c, op, needs, st, depth, chain, protect, faces)


def cost(res):
    """Fewer pick ups, toggles and drops in the plan, then fewer steps to its first."""
    acts = sum(1 for _, k in getattr(res, "ops", []) if k[0] in TH.ACTS)
    near = res.near[0] if isinstance(res.near, tuple) else res.near
    return acts, float(near) if near is not None else float("inf")


def _note_order(self, key, n, A):
    """Card 074's test found n before A for this achiever on this step; a flip when the last step found A before n."""
    cur = self.__dict__.setdefault("_c0741_cur", {})
    prev = self.__dict__.get("_c0741_prev", {})
    pairs = cur.setdefault(key, set())
    if (n, A) in pairs:
        return
    pairs.add((n, A))
    if (A, n) in prev.get(key, ()):
        STATS["flips"] += 1


def _note_split(self, needs, st):
    """B's report: where the hand condition holds, does the view need hold already (once per situation and need)?"""
    hs = [n for n in needs if n[0] == "has"]
    vs = [n for n in needs if n[0] == "part" and n[1] == VIEWP and n[-1] is None]
    if not hs or not vs or not all(self.hold(h, st) for h in hs):
        return
    seen = self.__dict__.setdefault("_c0741_split", set())
    for v in vs:
        if (st[0], v) in seen:
            continue
        seen.add((st[0], v))
        STATS["split_hand_met"] += 1
        STATS["split_view_held"] += bool(self.hold(v, st))


def pursue(self, c, op, needs, st, depth, chain, protect, faces):
    if C74._DEPTH[0] > 0:
        return _test_pursue(self, c, op, needs, st, depth, chain, protect, faces)
    if SPLIT:
        _note_split(self, needs, st)
    before, store = C74.before, C74.store
    met = [n for n in needs if n[0] != "face" and self.hold(n, st)]
    links = met + ([("does", op.a, op.u, op.j, c)] if getattr(op, "a", None) in TH.ACTS else [])
    prot = protect + met
    if TH.HAND_LINK and not (SPLIT and _has_need(needs)) and any(
            n[0] == "part" and n[1] == VIEWP and n[-1] is None and not self.hold(n, st) for n in needs):
        h = int(self.front_held(st)[1])
        prot = prot + [("hand", h)]
        links = links + [("hand", h)]
    mine = [n for n in needs if n[0] == "face"]
    unmet = [n for n in needs if not self.hold(n, st)]
    if not unmet:                                      # card 074: act, unless it breaks what comes after
        a = getattr(op, "a", None)
        if a in TH.ACTS:
            for p in protect:
                if p[0] != "after":
                    continue
                _, N, opN, b = p
                s2 = self.step(st, a)
                sN = self.imagine(s2, *opN) if s2 is not None else None
                if sN is not None and before(self, b, N, sN):
                    self.refused.add((st[0], a, C74.F.M.fidx[st[1]]))
                    self.version += 1
                    self.n_refused += 1
                    C74.STATS["refused_after"] += 1
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
                C74.STATS["released"] += 1
            else:
                keep.append(A)
        plans[n] = TH._one(self, c, n, st, depth, chain, protect + keep, faces, mine)
    if len(unmet) == 1:
        r = plans[unmet[0]]
        return None if r is None else TH._uses(r, links, op)
    STATS["weighed"] += 1
    C74.STATS["weighed"] += 1
    if plans[unmet[0]] is None:                        # card 029's order: the first unmet need decides
        return None
    key = (c, (getattr(op, "a", None), getattr(op, "u", None), getattr(op, "j", None)))
    waits = {n: False for n in unmet}
    states = {A: C74.produced(self, A, plans[A], st) for A in unmet if plans[A] is not None}
    for A in unmet:
        for n in unmet:
            if n is A or plans.get(n) is None or A not in states:
                continue
            if before(self, n, A, states[A]):
                waits[A] = True
                C74.STATS["ordered"] += 1
                store(self, A, n, st, "n first")
                _note_order(self, key, n, A)
            else:
                store(self, A, n, st, "none")
    viable = [n for n in unmet if plans[n] is not None]
    cands = [n for n in viable if not waits[n]]
    kept = self.__dict__.setdefault("_order_keep", {})
    prev = kept.get(key)
    if MODE == "074":                                  # card 074's choice, for the data check
        pick = next((n for n in unmet if not waits[n] and plans[n] is not None), None)
        if pick is None:
            C74.STATS["cycles"] += 1
            pick = unmet[0]
        if len(cands) < len(viable):
            STATS["found"] += 1
        if prev is not None and prev in viable and prev != pick:
            if len(cands) < len(viable):
                STATS["overridden"] += 1
            pick = prev
            C74.STATS["order_kept"] += 1
        first = pick
    else:
        if not cands:                                  # every need must be preceded: a tie among all
            STATS["cycles"] += 1
            C74.STATS["cycles"] += 1
            cands = viable
        elif len(cands) < len(viable):
            STATS["found"] += 1
        if len(cands) == 1:
            first = cands[0]
            STATS["followed" if len(cands) < len(viable) else "single"] += 1
        elif prev is not None and prev in cands:
            first = prev
            STATS["tie_kept"] += 1
        else:
            costs = {n: cost(plans[n]) for n in cands}
            first = min(cands, key=lambda n: (costs[n], unmet.index(n)))
            STATS["tie_cost" if len(set(costs.values())) > 1 else "tie_order"] += 1
        if first is not unmet[0]:
            C74.STATS["reordered"] += 1
    kept[key] = first
    opN = C74.own_op(first, plans[first])
    after = [("after", first, opN, b) for b in viable if b is not first] if opN else []
    if after:                                          # card 074: what comes after is kept while `first` is pursued
        keep = [A for A in held_now if not before(self, first, A, st)]
        r = TH._one(self, c, first, st, depth, chain, protect + keep + after, faces, mine)
        if r is not None:
            return TH._uses(r, links, op)
    return TH._uses(plans[first], links, op)


def conditions(self, a, u, j, c, st, bit):
    """Card 043's conditions, except that a success needing both parts changed gives two independent conditions:
    ("has", h), h the stored try's held tile, and the one-part view need checked with the situation's own hand and
    view."""
    k = (st[0], st[1], a, u, j, c)
    r = self.condmemo.get(k)
    if r is not None:
        return r
    r = []
    if self.does(st, a, u, j, c):
        r = [[]]
    elif self.imagine_place(st, u, j) is not None:
        kd = self.W.kinds[a]
        hc, vc = int(self.facts[st[0]][HELD]), self.ctx_id(st[0], st[1])
        pairs = kd.templates(u)
        cats = kd.cat_of([(u, h, v) for h, v in pairs])
        best = {}
        for (h, v), cat in zip(pairs, cats):
            if not cat or not cat & bit or not self.does(st, a, u, j, c, h, v):
                continue
            held_ok = self.does(st, a, u, j, c, None, v)
            view_ok = self.does(st, a, u, j, c, h, None)
            if held_ok and view_ok:
                continue
            if view_ok:
                needs = [("part", HELDP, a, u, j, c, None)]
            elif held_ok:
                needs = [("part", VIEWP, a, u, j, c, None)]
            else:                                      # both: the hand holds h, then the view as it then is
                needs = [("has", int(h)), ("part", VIEWP, a, u, j, c, None)]
            sig = tuple(HELDP if n[0] == "has" else n[1] for n in needs)
            d = kd.pair_distance((h, v), (hc, vc))
            if sig not in best or d < best[sig][0]:
                best[sig] = (d, needs)
        r = [best[s][1] for s in sorted(best)]
    self.condmemo[k] = r
    return r


def install():
    global _conditions0
    S7.Plan047.pursue = pursue
    if SPLIT:
        _conditions0 = VP.VPlan.conditions
        VP.VPlan.conditions = conditions
        for cls in S7.Plan047.__mro__:                 # a subclass that copied card 043's conditions takes this one
            if "conditions" in cls.__dict__ and cls is not VP.VPlan:
                cls.conditions = conditions
    choose0 = S7.Plan047.choose

    def choose(self, st):
        self._c0741_prev = getattr(self, "_c0741_cur", {})
        self._c0741_cur = {}
        res = choose0(self, st)
        STATS["steps"] += 1
        if res is not None and any(spliced(n) for n in getattr(res, "trace", [])):
            STATS["steps_spliced"] += 1
        return res

    S7.Plan047.choose = choose
