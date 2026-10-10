"""Card 101: one rule for acting with no plan (WM_NOPLAN=1; run without WM_INFORM and WM_TRYING).
WM_NOPLAN=measure: version 26 unchanged, only criterion 2's count (the card's baseline).

When the planner finds no plan and nothing unseen is left to walk to (cards 057 and 060, unchanged), the agent acts
on the (action, token, hand) whose outcome arm C knows least, U = 1 - max(P(no change), P(change) * c):
  - P: arm C's prediction that the token in front changes, with this episode's own tries (095.3; beta, card 099);
  - c: 095.7's share, from the store, of what the token becomes (the same tokens can give different results:
    a box's contents), so own tries count for whether it changes, not for what it becomes.
Candidates: pick up and toggle, every non-walkable catalogue token in the believed 13 x 13 window, under card 072's
hands (empty, the present one, tokens a pick up can put in the hand), except the token's own appearance (getting
that hand picks up the token acted on: the act vanishes unmade, seed 1003184). Order: U / (1 + walking steps to face
the token); the first of at most six with a plan is taken, except that the act held before this step's deliberation
comes first while it is not made, still uncertain and still planned (the intention as the default when
reconsidering, as card 100 for ties; without it, "newly seen" triggers while turning switched between two acts, seed
1003184). The act is the condition ("inform", a, u, j, h): held once the act is made or place j no longer shows u; its
achiever is a on u at j, needing ("has", h) and facing; once they hold, the act is taken without card 029's check
that the model predicts it achieves its condition (being made is its success: an act chosen because its prediction
is uncertain is often one predicted not to work, which card 072 met by planning under a hypothesis). The intention
(card 097.2) holds it until a trigger. Nothing uncertain with a plan: a random action, as before.

Criterion 2: a belief state (every token, the placement, the hand) counted on each step whose action this fallback
chose (its step, or the held act's continuation); a count above 50 adds to noplan_recur_over.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for c in ("card095.3", "card095.7", "card097.2"):
    sys.path.insert(0, str(ROOT / "tools" / c))
import statechange as SC                               # noqa: E402
import inform as INF                                   # noqa: E402  (095.7's certainty only; its hook is not installed)
import intent as INT                                   # noqa: E402  (chain_of)

MODE = os.environ.get("WM_NOPLAN", "1")
MAX_PLANNED = 6
RECUR = 50
STATS = {"noplan_acts": 0, "noplan_kept": 0, "noplan_targets": 0, "noplan_planned": 0, "noplan_random": 0,
         "noplan_made": 0, "noplan_steps": 0, "noplan_recur_over": 0}
STATE = {"made": set(), "recur": {}, "pl": None, "held": None}
T = None


def note(pl, st):
    """Criterion 2: one more step from this fallback in this belief state."""
    STATS["noplan_steps"] += 1
    k = (pl.facts[st[0]].tobytes(), int(st[1]))
    n = STATE["recur"][k] = STATE["recur"].get(k, 0) + 1
    if n > RECUR:
        STATS["noplan_recur_over"] += 1


def uncertainty(W, a, u, h):
    P = W.kinds[a].predict([(u, h, 0)])[0]                # four categories: arm C with own tries
    p = float(P[1] + P[3])                               # the front changes
    c = INF.certainty(W, a, u, h)[1]
    return 1.0 - max(1.0 - p, p * c)


def candidates(pl, st):
    VP = T.VP
    W, M = pl.W, VP.F.M
    f = pl.facts[st[0]]
    held, empty = int(f[VP.HELD]), int(T.Kd.APP[0])
    known = SC.STATE["known_set"]
    toks = sorted({int(u) for u in f[:VP.NV].tolist() if u >= 0 and int(u) in known and not M.free[int(u)]})
    cats = W.kinds[VP.PICK].cat_of([(u, empty, 0) for u in toks]) if toks else []
    holds = sorted({empty, held} | {u for u, c in zip(toks, cats) if c and c & 2})
    out = []
    for u in toks:
        js = np.flatnonzero(f[:VP.NV] == u).tolist()
        d, j = min(((pl.closeness(st, M.facing[j]) or (99, 9))[0], j) for j in js)
        for a in (VP.PICK, VP.TOG):
            for h in holds:
                if h == u:                              # getting this hand would pick up the token acted on
                    continue
                U = uncertainty(W, a, u, h)
                if U > 1e-6:
                    out.append((U / (1.0 + d), U, a, u, int(j), h))
    STATS["noplan_targets"] += len(out)
    return sorted(out, key=lambda x: (-x[0], -x[1], x[2], x[3], x[4], x[5]))


def hold_it(pl, st, res):
    """The chosen act becomes the intention (card 097.2's form: chain, actionable tokens seen, expected outcome)."""
    VP = T.VP
    f = pl.facts[st[0]]
    seen = {int(u) for u in np.unique(f[:VP.NV]).tolist() if u >= 0 and not VP.F.M.free[int(u)]}
    s2 = pl.step(st, res.action)
    pl._intent = {"chain": INT.chain_of(res.trace), "seen": seen, "expect": (s2[1],) + tuple(pl.front_held(s2))}


def fallback(pl, st, rng, rec):
    PV = T.PV
    STATE["pl"] = pl
    note(pl, st)
    a = PV.explore(pl, st)                              # unseen places, then a door next to them (cards 057, 060)
    if a is not None:
        rec["explore"] += 1
        return a
    order = [(a, u, j, h) for _, _, a, u, j, h in candidates(pl, st)[:MAX_PLANNED]]
    held = STATE["held"]
    if held is not None and not pl.hold(held, st) and uncertainty(pl.W, *held[1:3], held[4]) > 1e-6:
        order = [held[1:]] + [x for x in order if x != held[1:]]
        STATS["noplan_kept"] += 1
    for a, u, j, h in order:
        STATS["noplan_planned"] += 1
        res = pl.solve(("inform", a, u, j, h), st, 1, [], [], [])
        if res is not None:
            hold_it(pl, st, res)
            STATE["held"] = ("inform", a, u, j, h)
            STATS["noplan_acts"] += 1
            return int(res.action)
    rec["random"] += 1
    STATS["noplan_random"] += 1
    return int(rng.integers(6))


def install():
    global T
    T = sys.modules["tiers"]
    if MODE == "1":
        assert not os.environ.get("WM_INFORM"), "card 101 replaces 095.7 (no WM_INFORM)"
        assert not os.environ.get("WM_TRYING"), "card 101 replaces card 072 (no WM_TRYING)"
        T.PV.fallback = fallback


def hook(Tm):
    setup0 = Tm.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        VP = Tm.VP
        cls = Tm.PLANNER["cls"] or Tm.S7.Plan047
        World = type(W)
        learn0, reset0 = World.learn_try, World.reset

        def reset(self):
            STATE["made"].clear()
            STATE["recur"].clear()
            STATE["pl"] = None
            STATE["held"] = None
            return reset0(self)

        World.reset = reset
        if MODE == "measure":
            fb0 = Tm.PV.fallback

            def fallback_measured(pl, st, rng, rec):
                note(pl, st)
                return fb0(pl, st, rng, rec)

            Tm.PV.fallback = fallback_measured
            info["noplan"] = {"mode": "measure"}
            log("card 101: version 26's fallback, criterion 2 counted")
            return W, info
        hold0, ach0, choose0, pursue0 = cls.hold, cls.achievers, cls.choose, cls.pursue
        G = sys.modules["subgoals"]

        def hold(self, c, st):
            if c[0] == "inform":
                _, a, u, j, h = c
                return int(self.facts[st[0]][j]) != u or c in STATE["made"]
            return hold0(self, c, st)

        def achievers(self, c, st):
            if c[0] == "inform":
                _, a, u, j, h = c
                return [(VP.Ach(a, u, j), [("has", h), ("face", a, u, j)])]
            return ach0(self, c, st)

        def pursue(self, c, op, needs, st, depth, chain, protect, faces):
            if c[0] == "inform" and all(self.hold(n, st) for n in needs):
                return G.Result(op.a, [("act", op.a)], (0, 0))   # made is its success, whatever is predicted
            return pursue0(self, c, op, needs, st, depth, chain, protect, faces)

        def choose(self, st):
            res = choose0(self, st)
            I = getattr(self, "_intent", None)
            if res is not None and I is not None and I["chain"] and I["chain"][0][0] == "inform":
                note(self, st)                          # the held act's continuation is this fallback's step
            return res

        def learn_try(self, V, a, V2, end):
            pl = STATE["pl"]
            I = getattr(pl, "_intent", None) if pl is not None else None
            if I is not None and I["chain"] and I["chain"][0][0] == "inform":
                top = I["chain"][0]
                if a == top[1] and int(V[VP.FRONT]) == top[2]:
                    STATE["made"].add(top)
                    STATS["noplan_made"] += 1
            return learn0(self, V, a, V2, end)

        cls.hold, cls.achievers, cls.choose, cls.pursue = hold, achievers, choose, pursue
        World.learn_try = learn_try
        info["noplan"] = {"mode": "one rule", "max_planned": MAX_PLANNED, "recur": RECUR}
        log("card 101: one rule for acting with no plan (least known outcome, held by the intention)")
        return W, info

    Tm.setup = setup
