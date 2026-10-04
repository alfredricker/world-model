"""Card 051, step 4a: threats between the needs of one achiever.

Card 029's pursue() takes an achiever's first unmet need and works on it. When an achiever has two unmet needs,
working on one can undo what the other's plan relies on. Card 048's two doors, traced (2026-10-04, seed 399,
layout 0): "pick up key red" needs an empty hand and to face key red; facing key red needs door blue open, which
needs key blue in hand. The hand comes first in the order, so the agent drops key blue, then picks it up again
to reach key red, and so on for the whole episode.

Here, when an achiever has two or more unmet needs, each is planned from the present, and every plan records
the conditions it relies on that hold now (the needs met along its chain: its causal links). A need whose first
action would break a condition another need's plan relies on threatens it (SNLP's threat, McAllester and
Rosenblitt 1991); the first need that threatens none is pursued. If every one threatens another, card 029's order
stands. Everything else is card 050's agent with step 1's index.

  bin/prun python tools/card051/threats.py --chain two_doors --seed 399 --n 30 --out runs/051/s4_two_doors_399.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index as IX                                     # noqa: E402

S7 = IX.S7
G = S7.MV.G
DEBUG = False
ACTS = (3, 4, 5)                                       # pick up, drop, toggle
STATS = {"orders_weighed": 0, "reordered": 0, "all_threaten": 0}


def _uses(res, met, op=None):
    res.uses = list(getattr(res, "uses", [])) + list(met)
    if not hasattr(res, "act") and op is not None and getattr(op, "a", None) in ACTS:
        res.act = (op.a, op.u, op.j)                   # the act this plan works toward (its deepest achiever)
    return res


def _one(self, c, n, st, depth, chain, prot, faces, mine):
    if n[0] == "face":
        return self.walk(n, st, depth, chain + [c], prot, faces)
    if depth + 1 > G.MAXD:
        return None
    return self.solve(n, st, depth + 1, chain + [c], prot, faces + mine)


def _holds(self, p, st):
    """A condition a plan relies on: a need, or ("does", a, u, j, c): doing a on u makes c true with the hand
    and view of st (an achiever that works as things are has no needs, card 043, but still relies on them)."""
    if p[0] == "does":
        return self.does(st, *p[1:])
    return self.hold(p, st)


def pursue(self, c, op, needs, st, depth, chain, protect, faces):
    met = [n for n in needs if n[0] != "face" and self.hold(n, st)]
    links = met + ([("does", op.a, op.u, op.j, c)] if getattr(op, "a", None) in ACTS else [])
    prot = protect + met
    mine = [n for n in needs if n[0] == "face"]
    unmet = [n for n in needs if not self.hold(n, st)]
    if len(unmet) == 1:
        res = _one(self, c, unmet[0], st, depth, chain, prot, faces, mine)
        return None if res is None else _uses(res, links, op)
    if len(unmet) >= 2:
        STATS["orders_weighed"] += 1
        plans = [(n, _one(self, c, n, st, depth, chain, prot, faces, mine)) for n in unmet]
        if plans[0][1] is None:                        # card 029's order: the first unmet need decides
            return None
        threat = []
        for n, r in plans:
            if r is None:
                threat.append(True)
                continue
            st2 = None
            if hasattr(r, "act"):                      # the state after the act the plan works toward
                st2 = self.imagine(st, *r.act)
            if st2 is None:
                st2 = self.step(st, r.action)
            bad = False
            for m, rm in plans:
                if m is n or rm is None:
                    continue
                if any(_holds(self, p, st) and not _holds(self, p, st2) for p in getattr(rm, "uses", [])):
                    bad = True
                    break
            threat.append(bad)
        if DEBUG:
            import trace as TR
            W = self.W
            print("   weigh", TR.fmt(W, c), "| unmet", [TR.fmt(W, n) for n in unmet], "| actions",
                  [None if r is None else r.action for _, r in plans], "| uses",
                  [None if r is None else [TR.fmt(W, p) if p[0] != "does" else f"does({p[1]} {W.judge.name(int(p[2]))})" for p in getattr(r, "uses", [])] for _, r in plans],
                  "| threat", threat, flush=True)
        pick = next((k for k, t in enumerate(threat) if not t), None)
        if pick is None:
            STATS["all_threaten"] += 1
            pick = 0
        elif pick != 0:
            STATS["reordered"] += 1
        return _uses(plans[pick][1], links, op)
    # every need met: card 029's act, with its refusal
    res = G.Plan.pursue(self, c, op, needs, st, depth, chain, protect, faces)
    return None if res is None else _uses(res, links, op)


def install():
    IX.install()
    S7.Plan047.pursue = pursue


def main():
    install()
    sys.argv[1:1] = ["--recall", "own"]
    IX.CD.main()


if __name__ == "__main__":
    main()
