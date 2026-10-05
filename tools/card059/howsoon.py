"""Card 059: how soon a goal is, from the chain of conditions.

Rung 1's second criterion: from any state, the predicted number of steps to a goal, against the evaluator's
fewest steps. Version 10's planner gives only the next action. Here the chain is followed in imagination one
condition at a time, never one step at a time: the chain's next act (pick up, drop or toggle on a thing), the
walking cost to face that thing (card 045's learned approach), one step for the act, then the imagined situation
after the act with the agent where walking would bring it; repeat until the goal holds. No action sequence is
searched: each round asks the planner for its chain, as acting does.

Goals and their frames are card 056's (one frame per episode); states come from random play on the test layouts.
Arms: frames, supplied (the goal written in), swapped (another goal's frames, scored against this goal's truth).

  bin/prun python tools/card059/howsoon.py runs/054/b_m0.5_399.pt --out runs/059/main_399.json --one-per-episode
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card056"))
import goals as GL                                     # noqa: E402

VP, F, ld, MV = GL.VP, GL.F, GL.ld, GL.MV
INF = MV.INF
CAP = 8                                                # conditions followed at most
N_STATES = 150


def arrival(pl, st, pid):
    """The placement walking reaches first on its least chain to the need's placements."""
    if st[1] in pl.psets[pid]:
        return int(st[1])
    ch = pl.plan(st[0], pid)
    i = ch.ix.get(int(st[1]))
    k = None
    for _ in range(64):
        if ch.Pm[i]:
            break
        j, direct, k = ch.hop(i, k)
        i = j
        if direct:
            break
    return int(ch.S[i])


def soon(pl, st):
    """Predicted steps until pl.goal holds, following the chain's acts in imagination; None if no chain."""
    n, seen = 0, set()
    for _ in range(CAP):
        if pl.hold(pl.goal, st):
            return n
        pl.commit = {}
        res = pl.choose(st)
        act = None if res is None else getattr(res, "act", None)
        if act is None:
            return None
        a, u, j = act
        f = pl.facts[st[0]]
        best = None
        for jj in ([j] if j is not None else pl.showing(f, u)):
            pid = pl.face_pid(st[0], ("face", a, u, jj))
            if not pl.psets[pid]:
                continue
            c = 0 if st[1] in pl.psets[pid] else pl.cost(st, pid)
            if c < INF and (best is None or c < best[0]):
                best = (c, jj, pid)
        if best is None:
            return None
        c, jj, pid = best
        here = (st[0], arrival(pl, st, pid), False)
        s2 = pl.imagine(here, a, u, jj)
        if s2 is None or (s2[0], s2[1]) in seen:
            return None
        seen.add((s2[0], s2[1]))
        n += int(c) + 1
        st = s2
    return None


def job(item):
    k, items = item
    W = VP.WORLD
    out = []
    for lay, s, g, roots in items:
        W.reset()
        fam, col = GL.target(lay, g)
        r = {"goal": g, "true": GL.search(lay, s, fam, col, limit=60)[0], "pred": {}, "seconds": {}}
        for arm, root in roots.items():
            t0 = time.monotonic()
            pl = VP.VPlan(W)
            pl.goal = root
            _, st, _, _ = pl.observe(None, VP.see(lay, s))
            r["pred"][arm] = soon(pl, st)
            r["seconds"][arm] = time.monotonic() - t0
        out.append(r)
    return k, out


def spearman(p, t):
    p, t = np.asarray(p, np.float64), np.asarray(t, np.float64)

    def rank(v):
        u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        return (start + (cnt - 1) / 2.0)[inv]
    if len(p) < 3:
        return None
    rp, rt = rank(p), rank(t)
    if rp.std() == 0 or rt.std() == 0:
        return None
    return round(float(np.corrcoef(rp, rt)[0, 1]), 4)


def soon_act_arm(pool, test, online=True):
    t00 = time.monotonic()
    rule = test[0].rule
    D = GL.CAPTURED["data"][rule]
    rng = np.random.default_rng(59)
    o = GL.OUT.setdefault("worlds", {})[rule] = {}
    w = D["w"] / D["w"].sum()
    rows = rng.choice(len(w), size=min(100_000, len(w)), replace=True, p=w)
    base = GL.feature_matrix(GL.Kd.APP[D["ego0"][rows]])
    GL.VOCAB[id(base)] = int(base.any(0).sum())
    P, sizes = GL.pools(rule, 9000 + "key switch either both".split().index(rule))
    o["goal_frame_pools"] = sizes

    def frames_root(fam, col, seed):
        pf = P.get((fam, col))
        if not pf:
            return ("all", ())
        r2 = np.random.default_rng(seed)
        return GL.as_condition(GL.infer([pf[i] for i in r2.choice(len(pf), size=GL.K, replace=len(pf) < GL.K)], base))

    trs = np.random.default_rng(58)
    states = [(lay, s) for lay in test for _ in range(2) for s in GL.play(lay, 640, trs)]
    states = [states[i] for i in trs.permutation(len(states))[:N_STATES]]
    items = []
    for n, (lay, s) in enumerate(states):
        for gi, g in enumerate(GL.GOALS):
            fam, col = GL.target(lay, g)
            sf, sw = GL.target(lay, GL.SWAP[g])
            items.append((lay, s, g, {"frames": frames_root(fam, col, 100 * n + gi),
                                      "supplied": GL.supplied(fam, col),
                                      "swapped": frames_root(sf, sw, 100 * n + gi + 50)}))
    parts = [c.tolist() for c in np.array_split(np.arange(len(items)), 60) if len(c)]
    recs = [None] * len(items)
    for k, out in pool.imap_unordered(job, [(k, [items[i] for i in c]) for k, c in enumerate(parts)]):
        for i, r in zip(parts[k], out):
            recs[i] = r
    reach = [r for r in recs if r["true"] is not None and r["true"] > 0]
    res = {"states": len(states), "items": len(recs), "reachable_not_yet_met": len(reach)}
    for arm in GL.ARMS:
        have = [r for r in reach if r["pred"][arm] is not None]
        res[arm] = {"rank_correlation": spearman([r["pred"][arm] for r in have], [r["true"] for r in have]),
                    "coverage": round(len(have) / max(len(reach), 1), 4),
                    "mean_abs_error": round(float(np.mean([abs(r["pred"][arm] - r["true"]) for r in have])), 3)
                    if have else None,
                    "mean_ratio": round(float(np.mean([r["pred"][arm] / r["true"] for r in have])), 3) if have else None,
                    "per_goal": {g: spearman([r["pred"][arm] for r in have if r["goal"] == g],
                                             [r["true"] for r in have if r["goal"] == g]) for g in GL.GOALS},
                    "seconds_per_estimate": round(float(np.mean([r["seconds"][arm] for r in recs])), 3)}
    # the swapped control read against the true steps of its own items where both exist
    o["howsoon"] = res
    o["seconds"] = round(time.monotonic() - t00, 1)
    print(json.dumps({"world": rule, **{a: {k: res[a][k] for k in ("rank_correlation", "coverage", "mean_abs_error")}
                                        for a in GL.ARMS}}), flush=True)
    GL.save()
    return {"success": 1.0, "mean_steps_when_successful": 1.0, "moves": 1, "seconds_per_layout_mean": 0.0}, recs


def main():
    if "--k" in sys.argv:                              # card 059's declared revision: ten goal frames
        i = sys.argv.index("--k")
        GL.K = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    GL.goal_act_arm = soon_act_arm
    GL.main()
    GL.OUT["note"] = "Card 059, tools/card059/howsoon.py (card 056's setup)"
    GL.save()


if __name__ == "__main__":
    main()
