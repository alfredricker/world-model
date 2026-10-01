"""Card 043: card 042's planner and recall, with a condition checked only in situations the learned effects
produce from the present.

Card 038's conditions split a stored success (held h, view v) into parts. A part need was then checked with the
other part fixed to the stored try's: "holding X, in view v". That spliced situation can be one that cannot
occur (holding the green key while it also lies in v). Here, when a success needs one part changed and the
other part already works as it is now, the need is checked in the situation itself: an achiever (say, pick up
the green key) is imagined with its learned effect on the present, which moves the thing into the hand and
changes the view with it, and the action is checked there. Stored tries still say which part to change and
order the alternatives; they no longer supply the situation that is checked. Successes that need both parts
changed keep card 038's check (declared exception).

  bin/prun python tools/card043/consistent.py --dev --arm A --seeds 399-399 --layouts 30 --out runs/043_dev.json
  bin/prun python tools/card043/consistent.py --arm A --seeds 400-404 --layouts-b 100 --out runs/043_armA.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card042"))
import code_recall as CR                              # noqa: E402

SP, VP = CR.SP, CR.VP
HELDP, VIEWP, HELD = VP.HELDP, VP.VIEWP, VP.HELD


def conditions(self, a, u, j, c, st, bit):
    """Card 038's conditions (VPlan.conditions), except that a success needing one part changed gives the
    need ("part", p, a, u, j, c, None): met in a situation where doing a on u makes c true with that
    situation's own hand and view."""
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
            held_ok = self.does(st, a, u, j, c, None, v)        # the hand as now would do, with v
            view_ok = self.does(st, a, u, j, c, h, None)        # the view as now would do, with h
            if held_ok and view_ok:
                continue
            if view_ok:                                         # change the hand only: checked as it will be
                needs = [("part", HELDP, a, u, j, c, None)]
            elif held_ok:                                       # change the view only
                needs = [("part", VIEWP, a, u, j, c, None)]
            else:                                               # both: card 038's check (declared exception)
                needs = [("part", HELDP, a, u, j, c, v), ("part", VIEWP, a, u, j, c, h)]
            sig = tuple(n[1] for n in needs)
            d = kd.pair_distance((h, v), (hc, vc))
            if sig not in best or d < best[sig][0]:
                best[sig] = (d, needs)
        r = [best[s][1] for s in sorted(best)]
    self.condmemo[k] = r
    return r


def install():
    VP.VPlan.conditions = conditions


def main():
    install()
    CR.main()


if __name__ == "__main__":
    main()
