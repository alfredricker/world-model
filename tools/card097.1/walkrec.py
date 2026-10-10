"""Card 097.1: the agent's own moves recorded for learning to walk (WM_WALKREC=1), and the field's choices at walking
decisions (the evaluation's reference).

Every step: the believed 13 x 13 view in the agent's frame (pl.view: token handles, the agent at the centre facing
ahead), the hand, the action taken and the placement (its place and heading in the first view's frame, card 044).
At a step whose action walks toward a need (a plan's trace [need, ("walk", "closer")]): the need's target
placements relative to the agent (x right, y ahead, heading turned) and the moves card 067's field rates closest.
WM_WALK_EPS=e: with probability e a step's action is replaced by a random move (data collection only).
"""
import os
import sys

import numpy as np

STATS = {"wr_steps": 0, "wr_walk_decisions": 0, "wr_random": 0}
STATE = {"rec": None}
EPS = float(os.environ.get("WM_WALK_EPS", "0"))


def rel(M, c, t):
    """Placement t relative to placement c: (x right, y ahead, heading turned right 0-3)."""
    DIRS = STATE["DIRS"]
    ahead, right = DIRS[M.pd[c]], DIRS[(M.pd[c] + 1) % 4]
    d = (M.px[t] - M.px[c], M.py[t] - M.py[c])
    return (int(d[0] * right[0] + d[1] * right[1]), int(d[0] * ahead[0] + d[1] * ahead[1]), int((M.pd[t] - M.pd[c]) % 4))


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        VP = T.VP
        F = VP.F
        STATE["DIRS"] = [tuple(d) for d in F.DIRS]
        cls = T.PLANNER["cls"] or T.S7.Plan047
        choose0 = cls.choose
        PV = sys.modules["partial"]
        fallback0 = PV.fallback
        World = type(W)
        learn0 = World.learn_try
        moves = (VP.LEFT, VP.RIGHT, VP.FWD)
        rng = np.random.default_rng(int(os.environ.get("WM_WALK_SEED", "97")))

        def note(pl, st, res):
            rec = STATE["rec"]
            if rec is None:
                return
            M = F.M
            rec["pending"] = {"pose": int(st[1]), "walk": None}
            tr = list(getattr(res, "trace", None) or [])
            k = next((i for i, c in enumerate(tr) if c == ("walk", "closer")), -1)
            if res is not None and k >= 1 and isinstance(tr[k - 1], tuple) and tr[k - 1][0] == "face":
                pid = pl.face_pid(st[0], tr[k - 1])
                fl = pl.plan(st[0], pid)
                i = fl.ix[int(st[1])]
                if i >= 0:
                    d = fl.dist[i]
                    best = [int(a) for a in moves if fl.succ[i, a] >= 0 and fl.dist[fl.succ[i, a]] + fl.cost[i, a] == d]
                    tg = sorted({rel(M, int(st[1]), int(p)) for p in pl.psets[pid]})
                    rec["pending"]["walk"] = {"best": best, "dist": int(d), "targets": tg, "action": int(res.action)}

        def choose(self, st):
            res = choose0(self, st)
            if not getattr(self, "_in_fallback", False):
                note(self, st, res)
            return res

        def fallback(pl, st, rng_, rec_):
            a = fallback0(pl, st, rng_, rec_)
            if STATE["rec"] is not None and STATE["rec"].get("pending") is None:
                STATE["rec"]["pending"] = {"pose": int(st[1]), "walk": None}
            return a

        def learn_try(self, V, a, V2, end):
            rec = STATE["rec"]
            if rec is not None:
                p = rec.pop("pending", None) or {"pose": -1, "walk": None}
                M = F.M
                c = p["pose"]
                rec["steps"].append({"view": np.asarray(V[:VP.NV], np.int32).tolist(), "held": int(V[VP.HELD]),
                                     "a": int(a), "pose": [int(M.px[c]), int(M.py[c]), int(M.pd[c])] if c >= 0 else None,
                                     "walk": p["walk"], "random": rec.pop("was_random", False)})
                STATS["wr_steps"] += 1
                STATS["wr_walk_decisions"] += p["walk"] is not None
            return learn0(self, V, a, V2, end)

        cls.choose = choose
        PV.fallback = fallback
        World.learn_try = learn_try
        if EPS > 0:                                    # data collection: some actions replaced by a random move
            Res = T.S7.MV.G.Result

            def choose_eps(self, st, _c=cls.choose):
                res = _c(self, st)
                if STATE["rec"] is not None and rng.random() < EPS:
                    STATS["wr_random"] += 1
                    STATE["rec"]["was_random"] = True
                    return Res(int(rng.choice(moves)), [("random",)], (0, 0))
                return res

            cls.choose = choose_eps
        STATE["episode0"] = T.episode
        T.episode = episode
        log(f"card 097.1: moves recorded (random share {EPS})")
        return W, info

    T.setup = setup


def episode(job):
    STATE["rec"] = {"steps": []}
    try:
        r = STATE["episode0"](job)
    finally:
        rec, STATE["rec"] = STATE["rec"], None
    r["walkrec"] = rec["steps"]
    return r
