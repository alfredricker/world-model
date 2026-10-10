"""Card 095.5: the planner's conditions and "can this token be made walkable" read from arm C's store
(WM_COND=1; needs WM_STATECHANGE=1, WM_TIES and WM_SPLIT=1). WM_COND=shadow: version 23 acts, and at every call
both answers are computed and compared.

Version 23 still asks version 20's stored situations (the stores of pick up, toggle and drop) for one thing: which
hands to try. Card 074.1's conditions take the (held, view) situations of stored tries on tokens like u (templates),
card 047's openable those of tries that made a token walkable (opened); arm C then judges each. Here the hands to
try are those under which arm C's store saw the action change a token (its changed instances), this episode's own
tries that changed something, and card 072's guess. Arm C reads only the front and the hand (095.3's gate kept no
other place), so a condition is a part of the hand, ("part", HELDP, ...), and card 043's view part does not arise.
"""
import os
import sys

MODE = os.environ.get("WM_COND", "1")
STATS = {"cond_calls": 0, "cond_differ": 0, "open_calls": 0, "open_differ": 0, "old_reads": 0}
STATE = {"cand": {}, "differ": []}
DEBUG = os.environ.get("WM_SC_DEBUG") == "1"


def candidates(a):
    """The hands under which action a changed a token: arm C's store, this episode's own tries, card 072's guess."""
    SC = sys.modules["statechange"]
    base = STATE["cand"].get(a)
    if base is None:
        F = SC.STATE["stores"][a].Fch
        base = STATE["cand"][a] = {int(h) for h in F[:, 14].tolist() if h >= 0}
    out = set(base)
    for (a2, _, h), N in SC.STATE["own"].items():
        if a2 == a and N[1:].sum() > 0:
            out.add(int(h))
    TRY = sys.modules.get("trying")
    hyp = TRY.STATE["hyp"] if TRY is not None else None
    if hyp is not None and hyp["a"] == a:
        out.add(int(hyp["q"][1]))
    return sorted(out)


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        VP = T.VP
        HELDP = VP.HELDP
        TIES = sys.modules["ties"]
        SIT = sys.modules["situations"]
        assert TIES.SPLIT or os.environ.get("WM_LEAN") == "1", \
            "card 095.5 replaces card 074.1's split conditions (WM_SPLIT=1; card 099 drops the split)"
        cond_v23 = TIES.conditions
        opened_v23 = SIT.opened_in

        def conditions_c(self, a, u, j, c, st, bit):
            r = []
            if self.does(st, a, u, j, c):
                r = [[]]
            elif self.imagine_place(st, u, j) is not None:
                kd = self.W.kinds[a]
                hs = candidates(a)
                vc = self.ctx_id(st[0], st[1])
                for h, cat in zip(hs, kd.cat_of([(u, h, vc) for h in hs])):
                    if cat and cat & bit and self.does(st, a, u, j, c, h, None):
                        r = [[("part", HELDP, a, u, j, c, None)]]
                        break
            return r

        def openable_c(W_, u):
            for a in (VP.PICK, VP.TOG):
                kd = W_.kinds[a]
                qs = [(int(u), h, 0) for h in candidates(a)]
                for q, c in zip(qs, kd.cat_of(qs) if qs else []):
                    if c and c & 1:
                        f = kd.result(q, c)[0]
                        if f is not None and W_.M.free[int(f)]:
                            return True
            return False

        if MODE == "shadow":
            def conditions(self, a, u, j, c, st, bit):
                k = (st[0], st[1], a, u, j, c)
                fresh = k not in self.condmemo
                r0 = cond_v23(self, a, u, j, c, st, bit)
                if fresh:
                    STATS["cond_calls"] += 1
                    r1 = conditions_c(self, a, u, j, c, st, bit)
                    if sorted(map(tuple, r0)) != sorted(map(tuple, r1)):
                        STATS["cond_differ"] += 1
                        nm = W.judge.name
                        d = (VP.KNAME.get(a, a), nm(int(u)), str(c)[:40], nm(int(self.facts[st[0]][VP.HELD])),
                             str(r0)[:90], str(r1)[:90])
                        if len(STATE["differ"]) < 40:
                            STATE["differ"].append(d)
                            print(f"   conditions differ: {d}", flush=True)
                return r0

            World = type(W)
            openable0 = World.openable

            def openable(self, u):
                fresh = u not in self.openc
                r0 = openable0(self, u)
                if fresh:
                    STATS["open_calls"] += 1
                    r1 = openable_c(self, u)
                    if bool(r0) != r1:
                        STATS["open_differ"] += 1
                        if len(STATE["differ"]) < 40:
                            d = ("openable", self.judge.name(int(u)), bool(r0), r1)
                            STATE["differ"].append(d)
                            print(f"   openable differs: {d}", flush=True)
                return r0

            World.openable = openable
        else:
            def conditions(self, a, u, j, c, st, bit):
                k = (st[0], st[1], a, u, j, c)
                r = self.condmemo.get(k)
                if r is None:
                    STATS["cond_calls"] += 1
                    r = self.condmemo[k] = conditions_c(self, a, u, j, c, st, bit)
                return r

            def opened_in(self, a):
                STATS["open_calls"] += 1
                return [(h, 0) for h in candidates(a)]

            SIT.opened_in = opened_in
            for a in VP.INTER:                          # version 20's situations: counted, should not be read
                kd = W.kinds[a]
                t0 = kd.templates

                def templates(u, t0=t0):
                    STATS["old_reads"] += 1
                    return t0(u)

                kd.templates = templates
            IK = type(W.kinds[VP.PICK])
            o0 = IK.opened

            def opened(self, free):
                STATS["old_reads"] += 1
                return o0(self, free)

            IK.opened = opened
        TIES.conditions = conditions
        VP.VPlan.conditions = conditions
        for cls in TIES.S7.Plan047.__mro__:
            if "conditions" in cls.__dict__ and cls is not VP.VPlan:
                cls.conditions = conditions
        W.openc.clear()
        info["conditions"] = {"mode": MODE, "candidates": {VP.KNAME.get(a, a): len(candidates(a)) for a in VP.INTER}}
        log(f"card 095.5: conditions from arm C's store ({MODE}); {info['conditions']}")
        return W, info

    T.setup = setup

