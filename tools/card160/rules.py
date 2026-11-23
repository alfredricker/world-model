"""Card 160: MiniGrid's rules written by hand.  For the evaluator only: the gate's upper bound, the example success
steps that show a goal, the data check, and the truth that statements and predictions are scored against."""
import numpy as np
from collections import Counter
from grid import DIRV, OFF8, NA, NG, FAR, UNK, OUT, gid, gid_inv, unpack

EMPTY, WALL, DOOR, KEY, BALL, BOX, GOAL, LAVA = 1, 2, 4, 5, 6, 7, 8, 9
LEFT, RIGHT, FWD, PICK, DROP, TOGGLE, DONE = range(7)
ANAME = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
TNAME = {1: "empty", 2: "wall", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava"}
CNAME = ["red", "green", "blue", "purple", "yellow", "grey"]
SNAME = ["open", "closed", "locked"]


def name(codes, c):
    t = codes.triples[c] if c < len(codes.triples) else None
    if t is None:
        return "unknown" if c == UNK else "outside"
    if t[0] == DOOR:
        return "%s %s door" % (SNAME[t[2]], CNAME[t[1]])
    return TNAME.get(t[0], "?") if t[0] in (EMPTY, WALL, GOAL, LAVA) else "%s %s" % (CNAME[t[1]], TNAME[t[0]])


class Rules:
    kind = "rules"

    def __init__(s, codes):
        s.codes = codes; s.empty = codes.of((EMPTY, 0, 0)); s.move_actions = [FWD]

    def tri(s, c):
        return s.codes.triples[c] if c < len(s.codes.triples) else None

    def held(s, items):
        return [k for k, a in items if a == PICK and s.tri(k) is not None and s.tri(k)[0] in (KEY, BALL, BOX)]

    def cell(s, c, a, g, items):
        t = s.tri(c)
        if t is None or g == FAR:
            return c
        dx, dy, d = gid_inv(g)
        if (dx, dy) != DIRV[d]:
            return c
        h = s.held(items)
        if a == PICK and t[0] in (KEY, BALL, BOX) and not h:
            return s.empty
        if a == DROP and t[0] == EMPTY and h:
            return h[0]
        if a == TOGGLE and t[0] == BOX:
            return s.empty
        if a == TOGGLE and t[0] == DOOR:
            if t[2] == 0:
                return s.codes.of((DOOR, t[1], 1))
            if t[2] == 1 or any(s.tri(k)[0] == KEY and s.tri(k)[1] == t[1] for k in h):
                return s.codes.of((DOOR, t[1], 0))
        return c

    def passable(s, c):
        t = s.tri(c)
        return c == UNK or (t is not None and (t[0] in (EMPTY, GOAL, LAVA) or (t[0] == DOOR and t[2] == 0)))

    def agent(s, d, a, neigh, items):
        if a == LEFT:
            return (0, (d - 1) % 4)
        if a == RIGHT:
            return (0, (d + 1) % 4)
        if a == FWD and s.passable(neigh[OFF8.index(DIRV[d])]):
            return (1 + d, d)
        return (0, d)

    # ---- the interface the agent uses (energy.py's models have the same)
    def agent_pred(s, d, neigh_rows, items):
        return [[s.agent(d, a, nb, items) for a in range(NA)] for nb in neigh_rows]

    def agent_one(s, d, a, neigh, items):
        return s.agent(d, a, neigh, items)

    def carry(s):
        return [c for c, t in enumerate(s.codes.triples) if t is not None and t[0] in (KEY, BALL, BOX)]

    def outcomes(s, c, a_req=None):
        """the codes c can turn into (by action a_req)"""
        out = set()
        for a in ([a_req] if a_req is not None else range(NA)):
            for d in range(4):
                for h in [()] + [((k, PICK),) for k in s.carry()]:
                    o = s.cell(c, a, gid(*DIRV[d], d), frozenset(h))
                    if o != c:
                        out.add(o)
        return sorted(out)

    def statement(s, c, o, a_req=None):
        ok = [(a, d, h) for a in ([a_req] if a_req is not None else range(NA)) for d in range(4) for h in [None] + s.carry()
              if s.cell(c, a, gid(*DIRV[d], d), frozenset([(h, PICK)] if h is not None else [])) == o]
        if not ok or o == c:
            return None
        a = ok[0][0]; hs = {h for a2, d, h in ok if a2 == a}; G = sorted({gid(*DIRV[d], d) for a2, d, h in ok if a2 == a})
        need = [("it", (h, PICK)) for h in hs] if None not in hs else []
        avoid = [("any", PICK)] if hs == {None} else []
        return dict(a=a, G=G, need=need[:1], avoid=avoid, n=0)


def truth_effects(codes, codes_seen):
    """every effect MiniGrid's rules allow on the codes the experience holds: (code, next code, action)"""
    R = Rules(codes); out = []
    for c in codes_seen:
        for a in range(NA):
            for o in R.outcomes(c, a):
                out.append((c, o, a))
    return out


def contexts(R):
    """every combination of action, place of the agent and one thing carried (or nothing)"""
    return [(a, g, frozenset([(h, PICK)] if h is not None else [])) for a in range(NA) for g in range(NG + 1) for h in [None] + R.carry()]


def says(st, a, g, items):
    """does a statement allow its effect in this context?"""
    def holds(f):
        return (f[1] in items) if f[0] == "it" else any(it[1] == f[1] for it in items)
    return st is not None and a == st["a"] and g in st["G"] and all(holds(f) for f in st["need"]) and not any(holds(f) for f in st["avoid"])


def effect_counts(T):
    c, a, g, it, o, n = T.rows(); m = c != o; cnt = Counter()
    for ci, oi, ai, ni in zip(c[m], o[m], a[m], n[m]):
        cnt[(int(ci), int(oi), int(ai))] += int(ni)
    return cnt


def data_check(T):
    codes = T.codes; cnt = effect_counts(T); R = Rules(codes)
    seen = sorted(set(int(x) for x in np.unique(T.rows()[0])))
    truth = truth_effects(codes, seen); by = {}
    for c, o, a in truth:
        t = codes.triples[c]; kind = "%s %s%s" % (ANAME[a], (SNAME[t[2]] + " ") if t[0] == DOOR else "", TNAME[t[0]]) if t[0] != EMPTY else "drop"
        by.setdefault(kind, []).append(cnt.get((c, o, a), 0))
    print("effects in MiniGrid's rules on the codes seen: %d; experienced at least 3 times: %d" % (len(truth), sum(cnt.get(e, 0) >= 3 for e in truth)))
    for k, v in sorted(by.items()):
        print("   %-22s codes %2d   times, fewest to most: %s" % (k, len(v), sorted(v)))
    extra = [(name(codes, c), name(codes, o), ANAME[a], n) for (c, o, a), n in cnt.items() if (c, o, a) not in set(truth)]
    print("   changes seen that the rules do not list:", extra[:10])
    return dict(effects=len(truth), experienced=sum(cnt.get(e, 0) >= 3 for e in truth), by_kind={k: sorted(v) for k, v in by.items()})
