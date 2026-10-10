"""Card 100: one held intention, the only memory of choices between steps (WM_HELD=A or B).

Version 25 keeps choices between steps in four places: card 051's achiever per condition (`commit`), card 074.1's
first need per (condition, achiever) (`_order_keep`), card 060's door to look behind (`look_j` first) and card
097.2's intention. Arm A removes the first three: on a trigger the planner deliberates by its own ranking alone.
`look_j` stays as what it also is, this step's chosen door (card 072's fallback reads it in the same step); only its
use as the next step's first choice goes.

Arm B (after arm A's diagnosis: on tier 2 only the kept ties mattered): as arm A, and where card 074.1 breaks a tie
among an achiever's needs, the need that the intention's chain pursued after that condition comes first. The chain
is the intention held now, or the one just dropped while the agent deliberates (a chain records each condition
followed by the need pursued first, even when it was alone, as card 074.1's store did).

  WM_HELD=A|B <version 25's flags> bin/prun python tools/card069/run.py tiers ...
  diagnosis: WM_HELD_KEEP=commit|ties|door leaves that old keeper in place (arm A)
"""
import os
import sys

ARM = os.environ.get("WM_HELD", "A")
KEEP = set(filter(None, os.environ.get("WM_HELD_KEEP", "").split(",")))   # diagnosis: old keepers left in place
STATS = {}


class Forgetful(dict):
    """A store of choices that keeps none: every read finds nothing."""

    def __setitem__(self, k, v):
        pass

    def get(self, k, default=None):
        return default

    def setdefault(self, k, default=None):
        return default


class FromIntention(Forgetful):
    """Card 074.1's kept ties read from the intention's chain: key (condition, achiever) -> the need it pursued."""

    def __init__(self, pl):
        super().__init__()
        self.pl = pl

    def get(self, k, default=None):
        return self.pl.__dict__.get("_held_next", {}).get(k[0], default)


def next_of(chain):
    """Each condition of a chain (top first) -> the condition pursued below it."""
    return {chain[i]: chain[i + 1] for i in range(len(chain) - 1)}


def install():
    assert ARM in ("A", "B"), ARM
    T = sys.modules["tiers"]
    PV = sys.modules["partial"]
    cls = T.PLANNER["cls"] or T.S7.Plan047
    if "commit" not in KEEP:
        cls.commit = property(lambda self: {}, lambda self, v: None)   # card 051's kept achievers: none
    if "ties" not in KEEP:
        init0 = cls.__init__

        def __init__(self, *a, **k):
            init0(self, *a, **k)
            self.__dict__["_order_keep"] = Forgetful() if ARM == "A" else FromIntention(self)   # card 074.1's ties

        cls.__init__ = __init__
    if "door" not in KEEP:
        PV.KEEP_LOOK = False                                            # card 060's kept door: not preferred


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        if ARM == "B":
            cls = T.PLANNER["cls"] or T.S7.Plan047
            choose0 = cls.choose                        # card 097.2's (installed in an inner setup)

            def choose(self, st):
                I = getattr(self, "_intent", None)      # held now; dropped if this step deliberates
                if I is not None:
                    self.__dict__["_held_next"] = next_of(I["chain"])
                res = choose0(self, st)
                I = getattr(self, "_intent", None)
                if I is not None:
                    self.__dict__["_held_next"] = next_of(I["chain"])
                return res

            cls.choose = choose
        info["held"] = {"arm": ARM, "kept between steps": "the intention only"}
        log(f"card 100: arm {ARM}, the intention the only memory of choices between steps")
        return W, info

    T.setup = setup
