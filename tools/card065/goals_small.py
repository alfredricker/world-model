"""Card 065: goals from example frames, seen and pursued through the small view.

Card 056 inferred goals from example frames of the whole map and reached them with version 10. Here everything is
seen through MiniGrid's 7 x 7 occluded view, as version 14 sees it:
  - goal frames: the frame in which the goal became true (a small view shows a goal only where it is), one per
    episode of play like the agent's own (card 056's revision), cropped and occluded;
  - the agent's experience frames (the base rates of the size principle): its stored frames, cropped and occluded;
  - acting: version 14 (card 057's placing and believed views, card 060's exploration, cards 061-062's memory)
    pursuing the inferred goal; the written-in goal as the upper bound.

  bin/prun python tools/card065/goals_small.py runs/054/b_m0.5_399.pt --occlude --frontier --keep-look \
      --one-per-episode --out runs/065/main_399.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for d in ("card062", "card061", "card057", "card056"):
    sys.path.insert(0, str(TOOLS / d))
import believed as BL                                  # noqa: E402  (card 062 -> 061)
import partial as PV                                   # noqa: E402  (card 057, flags from argv)
import goals as GL                                     # noqa: E402  (card 056)

M61 = BL.M61
VP, F, TK, ld = GL.VP, GL.F, GL.TK, GL.ld
NV, CENTRE, FRONT, HELD = GL.NV, GL.CENTRE, GL.FRONT, GL.HELD
MOVES, INTER, LEFT = GL.MOVES, GL.INTER, GL.LEFT
NF = GL.NF
ARMS = ("frames", "supplied")


def small(lay, s):
    return PV.crop(VP.see(lay, s), PV.codes_fam(lay, s))


def features_small(V):
    """The features of a frame seen through the small view: has h, and 256 + u for each u observed."""
    V = np.asarray(V)
    v = V[:NV].astype(np.int64)
    seen = v >= 0
    v = v.copy()
    if seen[CENTRE]:
        v[CENTRE] = GL.undraw_of(v[CENTRE])
    return tuple(sorted({int(V[HELD])} | {256 + int(u) for u in np.unique(v[seen])}))


def base_small(D, rows):
    """(n, NF) bool: the agent's stored frames as the small view showed them."""
    codes = D["ego0"][rows]
    vis = M61.visible(codes)
    Vs = GL.Kd.APP[codes].astype(np.int64)
    n = len(rows)
    v = Vs[:, :NV].copy()
    lut = np.arange(256)
    for c in np.unique(v[:, CENTRE]):
        lut[c] = GL.undraw_of(c)
    v[:, CENTRE] = lut[v[:, CENTRE]]
    B = np.zeros((n, NF), bool)
    B[np.arange(n), Vs[:, HELD]] = True
    r, q = np.nonzero(vis[:, :NV])
    B[r, 256 + v[r, q]] = True
    return B


def pools_small(rule, seed, n_layouts=300, cap=400):
    """Goal frames per (family, colour): one per episode, the frame in which the goal became true, seen small."""
    rng = np.random.default_rng(seed)
    P = {}
    for _ in range(n_layouts * 2):
        lay = ld.make_layout(8, rule, rng)
        first = {}
        for s in GL.play(lay, 640, rng):
            for fam, col in (("hold", lay.door_colour), ("hold", lay.distractor_colour), ("door", lay.door_colour),
                             ("switch", None)):
                if (fam, col) not in first and GL.holds(lay, s, fam, col):
                    first[(fam, col)] = (lay, s)
        for key, ls in first.items():
            P.setdefault(key, []).append(ls)
    out = {}
    for key, L in P.items():
        sel = rng.permutation(len(L))[:cap]
        out[key] = [features_small(small(*L[i])) for i in sel]
    return out, {f"{k[0]} {k[1]}": len(v) for k, v in P.items()}


def episode_small(W, lay, g, root, seed):
    """Card 056's episode with version 14's view, placing, believed tries and exploration."""
    M = F.M
    fam, col = GL.target(lay, g)
    rng = np.random.default_rng(seed)
    pl = VP.VPlan(W)
    pl.goal = root
    s = ld.start_state(lay)
    V = small(lay, s)
    facts, st, _, _ = pl.observe(None, V)
    rec = {"done": False, "steps": VP.BUDGET, "random": 0, "explore": 0, "idle": 0, "pred_wrong": 0,
           "hand_condition": False, "ended": False}
    t0 = time.monotonic()
    for t in range(VP.BUDGET):
        if pl.hold(root, st):
            a = LEFT
            rec["idle"] += 1
        else:
            res = pl.choose(st)
            if res is None:
                a = PV.explore(pl, st)
                if a is None:
                    a = int(rng.integers(6))
                    rec["random"] += 1
                else:
                    rec["explore"] += 1
            else:
                a = res.action
                if any(isinstance(c, tuple) and c and c[0] == "part" and c[1] == VP.HELDP for c in res.trace):
                    rec["hand_condition"] = True
        Vb = pl.view(st)
        pred = pl.step(st, a)
        u, held = pl.front_held(st)
        pcat = M.move_out[a][u] if a in MOVES else W.outcome(a, u, held, pl.ctx_id(st[0], st[1]))[0]
        s2, end = ld.step(lay, s, a)
        V2 = small(lay, s2)
        facts, st2, _, _ = pl.observe(facts, V2, end, prefer=pred[1], cands=PV.moved_to(st, a))
        Vb2 = pl.view(st2)
        if a in MOVES:
            rcat = VP.ENDED if end else (VP.CHANGED if (Vb[:NV] != Vb2[:NV]).any() else VP.UNCHANGED)
        else:
            rcat = int(V[FRONT] != V2[FRONT]) + 2 * int(V[HELD] != V2[HELD])
        if rcat != pcat:
            rec["pred_wrong"] += 1
            if a not in MOVES:
                pl.failed.add((a, M.fidx[st[1]], st[1], held))
                pl.version += 1
        W.learn_try(Vb, a, Vb2, end)
        pl.forget()
        s, st, V = s2, st2, V2
        if GL.holds(lay, s, fam, col):
            rec["done"], rec["steps"] = True, t + 1
            break
        if end:
            rec["ended"] = True
            break
    rec["seconds"] = time.monotonic() - t0
    rec["shortest"] = GL.search(lay, ld.start_state(lay), fam, col)[0]
    return rec


def arm_small(pool, test, online=True):
    t00 = time.monotonic()
    rule = test[0].rule
    D = GL.CAPTURED["data"][rule]
    rng = np.random.default_rng(56)
    o = GL.OUT.setdefault("worlds", {})[rule] = {}
    w = D["w"] / D["w"].sum()
    rows = rng.choice(len(w), size=min(100_000, len(w)), replace=True, p=w)
    base = base_small(D, rows)
    GL.VOCAB[id(base)] = int(base.any(0).sum())
    o["vocabulary"] = GL.VOCAB[id(base)]
    P, sizes = pools_small(rule, 9000 + "key switch either both".split().index(rule))
    o["goal_frame_pools"] = sizes

    def frames_root(lay, g, seed):
        fam, col = GL.target(lay, g)
        pf = P.get((fam, col))
        if not pf:
            return ("all", ())
        r2 = np.random.default_rng(seed)
        return GL.as_condition(GL.infer([pf[i] for i in r2.choice(len(pf), size=GL.K, replace=len(pf) < GL.K)], base))

    items, roots = [], {}
    for i, lay in enumerate(test):
        for gi, g in enumerate(GL.GOALS):
            sd = 10_000 * gi + i
            roots[(i, g)] = {"frames": frames_root(lay, g, sd), "supplied": GL.supplied(*GL.target(lay, g))}
            for arm in ARMS:
                items.append(("episode", lay, g, arm, roots[(i, g)][arm], 1000 + i))
    recs = GL.run_jobs(pool, items)
    acting = {}
    for arm in ARMS:
        rs = [r for r in recs if r["arm"] == arm]
        ok = [r for r in rs if r["done"]]
        acting[arm] = {"success": round(len(ok) / len(rs), 4),
                       "per_goal": {g: round(float(np.mean([r["done"] for r in rs if r["goal"] == g])), 4)
                                    for g in GL.GOALS},
                       "steps_ratio_to_shortest": round(float(np.mean([r["steps"] for r in ok]) /
                                                              np.mean([r["shortest"] for r in ok])), 3) if ok else None,
                       "random_share": round(sum(r["random"] for r in rs) / max(sum(r["steps"] for r in rs), 1), 4),
                       "explore_share": round(sum(r["explore"] for r in rs) / max(sum(r["steps"] for r in rs), 1), 4),
                       "ended_on_goal_square": int(sum(r["ended"] for r in rs)),
                       "seconds_per_episode": round(float(np.mean([r["seconds"] for r in rs])), 3)}
    o["acting"] = acting
    o["goals_inferred"] = Counter(f"{k[1]}: {GL.cname(roots[k]['frames'])}" for k in roots).most_common(12)
    o["frames_equal_supplied"] = round(float(np.mean([roots[k]["frames"] == roots[k]["supplied"] for k in roots])), 4)
    o["seconds"] = round(time.monotonic() - t00, 1)
    print(json.dumps({"world": rule, "acting": {a: acting[a]["success"] for a in ARMS},
                      "per_goal_frames": acting["frames"]["per_goal"], "frames_equal_supplied": o["frames_equal_supplied"]}),
          flush=True)
    GL.save()
    return {"success": acting["frames"]["success"], "mean_steps_when_successful": 1.0, "moves": 1,
            "seconds_per_layout_mean": 0.0}, recs


def main():
    M61.CTX = BL.ctx
    TK._world_init = M61.world_init
    TK.TPlan.observe = PV.observe
    GL.goal_act_arm = arm_small
    GL.episode = episode_small
    GL.main()
    GL.OUT["note"] = "Card 065, tools/card065/goals_small.py (card 056's goals through version 14's small view)"
    GL.save()


if __name__ == "__main__":
    main()
