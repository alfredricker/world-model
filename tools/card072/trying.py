"""Card 072: try the likeliest untried way, and learn the relation from each try.

When the planner finds no plan (version 15's fallback would explore the unseen, then act at random), the ways recall
does not predict to work are ranked: for pick up and toggle, every tile in the believed map that cannot be walked
onto (u), with every tile that could be held (the empty hand, what is held now, and every tile recall predicts a pick
up takes), recall's probability that the action changes u (the front-changing classes, summed). For the likeliest
MAX_CANDIDATES not tried in this episode, the planner is asked for a plan under a hypothesis: recall's readers treat
"the action on u's code, holding h's code" as working, with its likeliest changing class (the class predicted, the
situation added to the openable test's and the condition search's stored situations) and with the effect the search
needs from it, the tile it makes walkable: recall's after-tile for the pair in the present view, taken as walkable
while the hypothesis lasts (for a door never seen opened, recall's after-tile is a fresh token near the open door that
walking recall does not call walkable, card 068's finding; that prediction is not changed). The one with the greatest
probability / (1 + steps to the plan's first need) is taken (the planner holds no plan length: card 067 walking
knows only its next step). The hypothesis lasts until the try is made (the action done with u's code in front and
h's code held) or MAX_STEPS pass; memory then holds the try (card 050: a pair's own tries first, so a failed pair
predicts failure from then on).

Order when the planner finds no plan: an active hypothesis is continued (version 15's exploration under it: a door
next to unseen places, made walkable by the try, card 057's open_to_see); else version 15's exploration (the needed
things not yet seen); else the candidates, each asked for a plan to the goal and, failing that, for exploration under
it; else random actions.

After every try made on purpose, success or failure, the relation is refitted (card 069's refit, the weight on rel:P
refitted beside P: relation.ONLINE["lam"]).
"""
import sys

import numpy as np

T = sys.modules["tiers"]
R = sys.modules["relation"]
VP, PV, TK = T.VP, T.PV, T.TK
MAX_CANDIDATES = 6
MAX_STEPS = 100
STATS = {"tries_planned": 0, "tries_made": 0, "tries_succeeded": 0, "hypotheses_without_plan": 0,
         "tries_abandoned": 0, "try_refits": 0}
STATE = {"hyp": None, "tried": set(), "noplan": set(), "steps": 0, "log": []}
NAMES = {}


def code(W, h):
    return int(sys.modules["code_recall"].code_id(W.S, int(h)))


def name(h):
    if not NAMES:
        for c in range(len(T.Kd.APP)):
            NAMES.setdefault(int(T.Kd.APP[c]), T.KR.OBJECTS[c // 5] if c // 5 < len(T.KR.OBJECTS) else c)
    o = NAMES.get(int(h), int(h))
    return "empty" if o is None else (" ".join(str(x) for x in o) if isinstance(o, tuple) else str(o))


def _clear(W, pl):
    R._clear(W)
    pl.forget()


def _hyp_on(a, W, u, h):
    hyp = STATE["hyp"]
    return hyp is not None and hyp["a"] == a and code(W, u) == hyp["cu"] and code(W, h) == hyp["ch"]


def _wrap_kind(W, a):
    """Recall's readers for action a, with the hypothesis: its class predicted, its situation stored."""
    kd = W.kinds[a]
    if getattr(kd, "_try_wrapped", False):
        return
    cat_plain, tmpl_plain, result_plain = kd.cat_of, kd.templates, kd.result

    def result(q, c):
        hyp = STATE["hyp"]
        if hyp is not None and c == hyp["c"] and _hyp_on(a, W, q[0], q[1]):
            return hyp["after"]
        return result_plain(q, c)

    def cat_of(qs):
        out = cat_plain(qs)
        if STATE["hyp"] is None:
            return out
        return [STATE["hyp"]["c"] if _hyp_on(a, W, q[0], q[1]) else c for q, c in zip(qs, out)]

    def templates(u):
        r = tmpl_plain(u)
        hyp = STATE["hyp"]
        if hyp is not None and hyp["a"] == a and code(W, u) == hyp["cu"] and hyp["sit"] not in r:
            r = sorted(r + [hyp["sit"]])
        return r

    kd.cat_of, kd.templates, kd.result = cat_of, templates, result
    kd._try_wrapped = True


class HypFree:
    """Walking recall's answer, except for the hypothesis's after-tile, which the hypothesis takes as walkable."""

    def __init__(self, base):
        self.base = base

    def __getitem__(self, h):
        hyp = STATE["hyp"]
        if hyp is not None and hyp["after"][0] is not None and int(h) == int(hyp["after"][0]):
            return True
        return self.base[h]

    def clear(self):
        self.base.clear()


def _wrap_free(W):
    for M in {id(W.M): W.M, id(T.F.M): T.F.M}.values():
        if not isinstance(M.free, HypFree):
            M.free = HypFree(M.free)


def candidates(W, pl, st):
    """[(probability, hypothesis)], likeliest first: ways recall does not predict to work, not tried."""
    f = pl.facts[st[0]]
    seen = sorted({int(x) for x in np.unique(f) if x != TK.ABSENT})
    ctx = int(pl.ctx_id(st[0], st[1]))
    empty = int(T.Kd.APP[0])
    nonwalk = [u for u in seen if not TK.free_of(u)]
    cats = W.kinds[VP.PICK].cat_of([(u, empty, ctx) for u in nonwalk]) if nonwalk else []
    holds = sorted({empty, int(f[VP.HELD])} | {u for u, c in zip(nonwalk, cats) if c and c & 2})
    out = {}
    for a in (VP.PICK, VP.TOG):
        kd = W.kinds[a]
        qs = [(u, h, ctx) for u in nonwalk for h in holds]
        if not qs:
            continue
        plain = kd.cat_of(qs)
        P = kd.predict(qs)
        change = np.array([c & 1 for c in range(P.shape[1])], bool)
        for q, c0, p in zip(qs, plain, P):
            if c0 and c0 & 1:                          # recall predicts it works: already in the search
                continue
            pc = float(p[change].sum())
            k = (a, code(W, q[0]), code(W, q[1]))
            if pc <= 0 or k in STATE["tried"] or (k in out and out[k][0] >= pc):
                continue
            c = int(np.flatnonzero(change)[np.argmax(p[change])])
            out[k] = (pc, {"a": a, "cu": k[1], "ch": k[2], "c": c, "sit": (q[1], ctx), "u": q[0], "h": q[1],
                           "q": q})
    return sorted(out.values(), key=lambda x: -x[0])


def _steps(res):
    try:
        s = float(res.near[0] if isinstance(res.near, (tuple, list)) else res.near)
        return s if np.isfinite(s) else 0.0
    except (TypeError, ValueError, IndexError):
        return 0.0


def _act(pl, st):
    """Under the present hypothesis: (action, steps to the first need) from a plan to the goal, else from exploration
    (a door next to unseen places, which the hypothesis can make walkable); None when neither gives one."""
    res = pl.choose(st)
    if res is None and PV.ARM["arm"] == "B":
        a = PV.explore(pl, st)
        if a is None:
            return None
        j = getattr(pl, "look_j", None)
        res = pl.solve(("walk", j), st, 1, [], [], []) if j is not None else None
        return a, (_steps(res) if res is not None else 0.0)
    return None if res is None else (res.action, _steps(res))


def fallback(pl, st, rng, rec):
    W = pl.W
    for a in (VP.PICK, VP.TOG):
        _wrap_kind(W, a)
    _wrap_free(W)
    if STATE["hyp"] is not None:                       # continue the hypothesis (the goal not yet seen)
        r = _act(pl, st)
        if r is not None:
            return r[0]
        STATE["tried"].add((STATE["hyp"]["a"], STATE["hyp"]["cu"], STATE["hyp"]["ch"]))
        STATE["hyp"] = None
        STATS["tries_abandoned"] += 1
        _clear(W, pl)
    if PV.ARM["arm"] == "B":                           # version 15's exploration first
        a = PV.explore(pl, st)
        if a is not None:
            rec["explore"] += 1
            return a
    best, asked = None, 0
    for p, hyp in candidates(W, pl, st):
        k = (hyp["a"], hyp["cu"], hyp["ch"], st[0])
        if k in STATE["noplan"]:
            continue
        if asked >= MAX_CANDIDATES:
            break
        asked += 1
        hyp["after"] = tuple(W.kinds[hyp["a"]].result(hyp["q"], hyp["c"]))     # STATE["hyp"] is None: recall's own
        STATE["hyp"] = hyp
        _clear(W, pl)
        r = _act(pl, st)
        STATE["hyp"] = None
        if r is None:
            STATE["noplan"].add(k)
            STATS["hypotheses_without_plan"] += 1
            continue
        if best is None or p / (1 + r[1]) > best[0]:
            best = (p / (1 + r[1]), p, r[1], hyp)
    _clear(W, pl)
    if best is not None:
        STATE["hyp"], STATE["steps"] = best[3], 0
        _clear(W, pl)
        r = _act(pl, st)
        if r is not None:
            hyp = best[3]
            STATS["tries_planned"] += 1
            STATE["log"].append({"planned": [VP.KNAME.get(hyp["a"], hyp["a"]), name(hyp["u"]), name(hyp["h"])],
                                 "p": round(best[1], 4), "steps_to_first_need": best[2]})
            return r[0]
        STATE["hyp"] = None
        _clear(W, pl)
    rec["random"] += 1
    return int(rng.integers(6))


def install():
    PV.fallback = fallback
    R.ONLINE["lam"] = True
    SIT = T.S7
    opened_plain = SIT.opened_in

    def opened_in(self, a):
        r = opened_plain(self, a)
        hyp = STATE["hyp"]
        if hyp is not None and hyp["a"] == a and hyp["sit"] not in r:
            r = sorted(list(r) + [hyp["sit"]])
        return r

    SIT.opened_in = opened_in
    World = VP.World
    _learn, _reset = World.learn_try, World.reset

    def learn_try(self, V, a, V2, end):
        hyp = STATE["hyp"]
        refits = R.STATS["refits"]
        cat = _learn(self, V, a, V2, end)
        if hyp is None:
            return cat
        STATE["steps"] += 1
        if a == hyp["a"] and code(self, V[VP.FRONT]) == hyp["cu"] and code(self, V[VP.HELD]) == hyp["ch"]:
            ok = bool(cat & 1)
            STATS["tries_made"] += 1
            STATS["tries_succeeded"] += ok
            STATE["log"].append({"tried": [VP.KNAME.get(a, a), name(V[VP.FRONT]), name(V[VP.HELD])], "changed": ok})
            STATE["tried"].add((a, hyp["cu"], hyp["ch"]))
            STATE["hyp"] = None
            if R.ONLINE["on"] and R.STATS["refits"] == refits:   # card 069's refit did not run (prediction right)
                R.refit(self, self.kinds[a])
                STATS["try_refits"] += 1
            R._clear(self)
        elif STATE["steps"] > MAX_STEPS:
            STATE["tried"].add((hyp["a"], hyp["cu"], hyp["ch"]))
            STATE["hyp"] = None
            STATS["tries_abandoned"] += 1
            R._clear(self)
        return cat

    def reset(self):
        _reset(self)
        STATE.update(hyp=None, tried=set(), noplan=set(), steps=0, log=[])

    World.learn_try, World.reset = learn_try, reset
