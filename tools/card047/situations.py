"""Card 047: card 045's planner, with the situations for an action on a thing taken from both of recall's levels.

Wherever the planner asks in what situations (held thing, what is in view) an action could work on a thing u,
it now considers the situations of the stored tries on the same thing (equal codes) and on similar things
(k >= KMIN), never one level instead of the other, and recall's own prediction (both levels mixed, card 042)
judges each. Two readers ask this:
- what can be made walkable (card 038's World.openable): u is openable when recall predicts that a pick up or
  toggle makes u walkable in one of the situations where memory saw a thing made walkable;
- the search for conditions (card 042's templates): the situations whose parts become needs.
A failed try then rules out its own situation, not the thing. Card 045's filter on door candidates is removed.
Everything else is card 045's.

Reported (evaluator names): toggles the planner chose on a switch that was off ("switch_by_plan").

  bin/prun python tools/card047/situations.py --check --seeds 399-404 --worlds key --out runs/047_check.json
  bin/prun python tools/card047/situations.py --arm A --seeds 399-399 --layouts 30 --layouts-b 30 --out runs/047_shakedown_399.json
  bin/prun python tools/card047/situations.py --arm A --seeds 400-404 --layouts-b 100 --out runs/047_armA.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card045"))
import movement as MV                                  # noqa: E402

VP, CR, F = MV.VP, MV.CR, MV.F
SP = CR.SP
EVERY = frozenset(range(MV.TK.NT))
_templates = CR.CodeKind.templates
_openable = SP.SlotWorld.openable


def templates(self, u):
    """The distinct (held, view) situations of the stored tries on the same thing as u and on things like u."""
    r = self.tcache.get(u)
    if r is None:
        D = self.D
        n = len(self.keys)
        same = np.flatnonzero(self.fid[:n] == CR.code_id(self.S, u))
        kf = np.exp(-(np.abs(self.X[:n, :D] - self.S.arr[int(u)]) @ self.lam2[:D]))
        idx = set(same.tolist()) | set(np.flatnonzero(kf >= VP.KMIN).tolist())
        r = self.tcache[u] = sorted({self.keys[t][1:] for t in idx})
    return r


def opened_in(self, a):
    """The (held, view) situations of the stored tries of action a that made a thing walkable."""
    k = ("situations", a)
    r = self.openedc.get(k)
    if r is None:
        kd = self.kinds[a]
        out = set()
        for c in (1, 3):
            for t in np.flatnonzero(kd.aw[:, c] > 0):
                if not self.M.free[kd.keys[t][0]] and self.M.free[self.S.add(kd.after_mat(c, 0)[t])]:
                    out.add(tuple(int(x) for x in kd.keys[t][1:]))
        r = self.openedc[k] = sorted(out)
    return r


def openable(self, u):
    """Recall predicts that a pick up or toggle makes u walkable in a situation where memory saw a thing made
    walkable."""
    r = self.openc.get(u)
    if r is None:
        r = False
        for a in (VP.PICK, VP.TOG):
            kd = self.kinds[a]
            qs = [(int(u), h, v) for h, v in opened_in(self, a)]
            for q, c in zip(qs, kd.cat_of(qs) if qs else []):
                if c and c & 1:
                    f = kd.result(q, c)[0]
                    if f is not None and self.M.free[int(f)]:
                        r = True
                        break
            if r:
                break
        self.openc[u] = r
    return r


class Plan047(MV.Plan045):
    """Card 045's planner, counting the toggles it chooses on a switch that is off (evaluator names)."""

    def __init__(self, W):
        super().__init__(W)
        self.switch_by_plan = 0

    def choose(self, st):
        res = super().choose(st)
        if res is not None and res.action == VP.TOG:
            u, _ = self.front_held(st)
            if self.W.judge.name(int(u)) == "switch off":
                self.switch_by_plan += 1
        return res


_act_job, _act_arm = MV._act_job, MV.act_arm


def act_job(job):
    n0 = len(MV.PLANS)
    out = _act_job(job)
    for rec, pl in zip(out, MV.PLANS[n0:]):
        rec["switch_by_plan"] = getattr(pl, "switch_by_plan", 0)
    return out


def act_arm(pool, test, online=True):
    res, recs = _act_arm(pool, test, online)
    res["walking"]["switch_toggles_by_plan"] = int(sum(r.get("switch_by_plan", 0) for r in recs))
    res["walking"]["layouts_with_switch_by_plan"] = int(sum(r.get("switch_by_plan", 0) > 0 for r in recs))
    return res, recs


def install():
    CR.CodeKind.templates = templates
    SP.SlotWorld.openable = openable
    MV.Walk.may_open = lambda self, fid, pid, c: EVERY          # card 045's filter removed
    MV.PLAN["045"] = Plan047
    MV._act_job, MV.act_arm = act_job, act_arm


def check(seeds, worlds, out):
    """Per seed and world: what is openable (card 038's rule, card 047's), and how many situations the search
    for conditions considers per thing (card 042's rule, card 047's)."""
    MV.CS.install()
    MV.install("045")
    SP.install()
    VP.Kind, VP.vectors_of = CR.kind, CR.vectors_of
    VP.T.configure()
    cache = pickle.loads(VP.T.MEMORY.read_bytes())
    groups = VP.T.use_groups(cache["mem"], "cuda")
    data = pickle.loads(VP.T.DATA.read_bytes())["data"]
    res = {"note": "Card 047's gate: openable and situations per thing, before and after", "seeds": {}}
    for seed in seeds:
        z, parts, _ = VP.vectors_of("A", seed, cache["tiles"], cache["pairs"], groups, "cuda", lambda *a: None)
        S = VP.Store(z)
        lut = np.zeros(256, np.uint8)
        lut[:len(S.hof)] = S.hof
        VP.Kd.APP = lut
        res["seeds"][seed] = {}
        for world in worlds:
            W = VP.World(S, parts, data[world], "cuda", lambda *a: None)
            VP.WORLD, F.M = W, W.M
            things = sorted({int(k[0]) for a in (VP.PICK, VP.TOG) for k in W.kinds[a].keys})
            rows = {}
            for u in things:
                if W.M.free[u]:
                    continue
                W.openc.clear()
                o = bool(_openable(W, u))
                W.openc.clear()
                n = bool(openable(W, u))
                W.openc.clear()
                sit = {}
                for a in (VP.PICK, VP.TOG):
                    kd = W.kinds[a]
                    kd.tcache.clear()
                    s0 = len(_templates(kd, u))
                    kd.tcache.clear()
                    sit[VP.KNAME[a]] = [s0, len(templates(kd, u))]
                    kd.tcache.clear()
                rows[W.judge.name(u)] = {"openable_card038": o, "openable_card047": n, "situations_before_after": sit}
            res["seeds"][seed][world] = rows
            print(seed, world, {k: (v["openable_card038"], v["openable_card047"], v["situations_before_after"])
                                for k, v in rows.items()}, flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    if "--check" in args:
        a, b = get("--seeds", "399-404").split("-")
        check(range(int(a), int(b) + 1), get("--worlds", ",".join(VP.WORLDS)).split(","),
              get("--out", "runs/047_check.json"))
        return
    install()
    MV.main()
    out = Path(get("--out", ""))
    if out.is_file() and "--dev" not in args:
        r = json.loads(out.read_text())
        r["note"] = "Card 047, tools/card047/situations.py (card 045's walking; situations from both levels)"
        out.write_text(json.dumps(r, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
