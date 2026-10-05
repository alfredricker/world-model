"""Card 057: a view smaller than the map.

Version 10's agent on card 054's frozen encoder with identity up to noise, seeing MiniGrid's 7 x 7 view (six rows
ahead, three columns to each side, nothing behind; no occlusion) instead of the 13 x 13 window that always holds the
8 x 8 map. Places outside the view are not observed (-1 in the view).

  observe: the view is placed by the least distance over observed places only; a place whose token was never seen
           is no evidence; only observed places update the tokens.
  tries:   online tries are stored with the believed view (the remembered tokens around the agent, as recall's
           queries already read them), not the cropped one.
  arm B:   when the planner finds no chain, walk to the nearest pose from which a never-seen place would be in view
           (frontier exploration); arm A: random, as version 10.

  bin/prun python tools/card057/partial.py runs/054/b_m0.5_399.pt --arm B --out runs/057/fam_B_399.json [@@ walk2 args]
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch.nn.functional as fn

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card053"))
sys.path.insert(0, str(TOOLS / "card051"))
sys.path.insert(0, str(TOOLS / "card049"))
import recall_probe as RP                              # noqa: E402
import planner_check as PC                             # noqa: E402
import walk2                                           # noqa: E402
import worlds as WD                                    # noqa: E402

NVM = RP.NV
CM = walk2.CM
S7, G = CM.S7, CM.G
MV = S7.MV
VP, F, TK, CR = MV.VP, MV.F, MV.TK, MV.CR
ld = VP.ld
NV, NPL, CENTRE, FRONT, HELD = VP.NV, VP.NPL, VP.CENTRE, VP.FRONT, VP.HELD
MOVES, ABSENT, NT = VP.MOVES, TK.ABSENT, TK.NT
WIN = np.flatnonzero((np.abs(TK.WH[:, 0]) <= 3) & (TK.WH[:, 1] >= 0) & (TK.WH[:, 1] <= 6))     # 49 places
HIDE = np.setdiff1d(np.arange(NV), WIN)
assert len(WIN) == 49 and CENTRE in WIN and FRONT in WIN
ARM = {"arm": "B"}
DEBUG = "--debug" in sys.argv
OCCLUDE = "--occlude" in sys.argv                     # card 060: walls and closed doors hide what lies behind
FRONTIER = "--frontier" in sys.argv                   # card 060, arm B: frontier = unseen next to known walkable
KEEP_LOOK = "--keep-look" in sys.argv                 # card 060's declared revision: the door to look past kept
STATS = Counter()


def crop(V, codes=None):
    V = np.array(V, np.int32)
    V[HIDE] = -1
    if OCCLUDE and codes is not None:
        V[:NV][~visible(codes)] = -1
    return V


R = TK.R
W13 = VP.W13


def blocks(c):
    o = PC.KR.OBJECTS[int(c) // 5]
    return o == "wall" or (isinstance(o, tuple) and o[0] == "door" and o[2] != 2)


def visible(codes):
    """MiniGrid's process_vis on the 7 x 7 view (agent at the bottom centre): walls and closed doors hide what
    lies behind them. codes: the egocentric renderer codes (13 x 13 + the held row)."""
    E = np.asarray(codes)[:NV].reshape(W13, W13)[R - 6:R + 1, R - 3:R + 4]        # [j, i], j = 6 at the agent
    blk = np.vectorize(blocks)(E)
    mask = np.zeros((7, 7), bool)                                                # [i, j]
    mask[3, 6] = True
    for j in reversed(range(7)):
        for i in range(6):
            if mask[i, j] and not blk[j, i]:
                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = mask[i, j - 1] = True
        for i in reversed(range(1, 7)):
            if mask[i, j] and not blk[j, i]:
                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = mask[i, j - 1] = True
    vis = np.zeros(NV, bool)
    jj, ii = np.nonzero(mask.T)
    vis[(R - 6 + jj) * W13 + (R - 3 + ii)] = True
    return vis


def codes_fam(lay, s):
    return VP.Kd.ego(PC.KR.encode_logic_states(lay, [s]), np.array([s[:3]]))[0].reshape(-1)


def codes_wd(M, lay, s):
    if M is WD.CHAIN:
        return WD.C.ego(WD.C.encode(lay, s), s).reshape(-1)
    return WD.ego(WD.encode(lay, s), s).reshape(-1)


# ---------------------------------------------------------------- placing a partial view

def observe(self, world, V, ended=False, prefer=None, cands=None):
    """cands: the placements the action could lead to (its own move, or the same placement); None: every one."""
    M = F.M
    V = np.asarray(V, np.int32)
    vis = V >= 0
    if world is None:
        f = np.full(NT, ABSENT, np.int32)
        f[:NPL] = np.where(vis, V, ABSENT)
        f[CENTRE] = M.undraw[V[CENTRE]]
        pose, miss, ties = 0, 0.0, 1
    else:
        q = np.flatnonzero(vis[:NV])
        q = q[q != CENTRE]
        Aq = M.A[:, q]                                  # placements x observed places: tokens
        known = (Aq >= 0) & (np.asarray(world)[np.maximum(Aq, 0)] != ABSENT)
        Gv = self.shown(world, Aq).astype(np.int64)
        u, inv = np.unique(np.concatenate([Gv.ravel(), V[q]]), return_inverse=True)
        inv = inv.ravel()
        Z = self.W.S.arr[u]
        Dm = np.abs(Z[:, None] - Z[None]).sum(-1)
        d = Dm[inv[:Gv.size].reshape(Gv.shape), inv[Gv.size:][None]] * known
        dist = d.sum(1)
        if cands is not None:
            mask = np.full(len(dist), np.inf)
            mask[list(cands)] = 0.0
            dist = dist + mask
        best = dist.min()
        tied = np.flatnonzero(dist <= best + 1e-9)
        pose = prefer if prefer is not None and (tied == prefer).any() else int(tied[0])
        f = np.array(world, np.int32)
        Ap = M.A[pose]
        sel = (Ap >= 0) & vis
        sel[CENTRE] = False
        f[Ap[sel]] = V[sel]
        f[Ap[CENTRE]] = M.undraw[V[CENTRE]]
        miss, ties = float(best), len(tied)
    return f, (self.intern(f), int(pose), bool(ended)), miss, ties


def moved_to(st, a):
    """The placements action a can lead to from st: the same one, or the move's (forward may be blocked)."""
    c = {int(st[1])}
    if a in MOVES:
        j = F.M.nxt[st[1]][a]
        if j >= 0:
            c.add(int(j))
    return c


# ---------------------------------------------------------------- frontier exploration

def explore(pl, st):
    """The first move toward the nearest pose (standing on a known free token) from which a never-seen place of
    the lattice would be in view; None if there is none."""
    M = F.M
    if not hasattr(M, "win_tokens"):
        M.win_tokens = [A[WIN][A[WIN] >= 0] for A in M.A]
    f = pl.facts[st[0]]
    k = ("explore", st[0])
    pid = pl.canmemo.get(k)
    if pid is None and FRONTIER:
        if not hasattr(M, "nb_tokens"):
            at = TK.ID_AT
            M.nb_tokens = np.full((TK.NT, 4), -1, np.int64)
            for (x, y), i in at.items():
                for q, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    M.nb_tokens[i, q] = at.get((x + dx, y + dy), -1)
        nb = M.nb_tokens
        ok = (nb >= 0) & np.vectorize(lambda h: TK.free_of(int(h)))(np.where(nb >= 0, f[np.maximum(nb, 0)], ABSENT))
        front = (f == ABSENT) & ok.any(1)                 # unseen tokens next to a known walkable one
        P = [p for p in pl.standing(st[0]) if front[M.win_tokens[p]].any()]
        pid = pl.canmemo[k] = pl.intern_p(P)
    if pid is None:
        P = [p for p in pl.standing(st[0]) if (f[M.win_tokens[p]] == ABSENT).any()]
        pid = pl.canmemo[k] = pl.intern_p(P)
    r, _ = pl.reach(st, pid)
    if (r is None or r == G.HERE) and FRONTIER:
        return open_to_see(pl, st)
    return None if r is None or r == G.HERE else int(r)


def open_to_see(pl, st):
    """No unseen place lies next to a known walkable one: a known token that recall says can be made walkable
    (a door) next to unseen places becomes the condition ("walk", j), pursued like any other (its key first);
    opening it lets the agent see beyond. Nearest first."""
    M = F.M
    f = pl.facts[st[0]]
    nb = M.nb_tokens
    op = pl.openable(st[0])
    unseen_next = ((nb >= 0) & (f[np.maximum(nb, 0)] == ABSENT)).any(1)
    js = np.flatnonzero(op[:len(unseen_next)] & unseen_next[:len(op)]).tolist()
    cands = [j for _, j in sorted((pl.closeness(st, M.facing[j]) or (99, 9), j) for j in js)]
    prev = getattr(pl, "look_j", None) if KEEP_LOOK else None
    if prev in cands:                                   # the last step's choice first, while it still gives a plan
        cands = [prev] + [j for j in cands if j != prev]
    for j in cands:
        res = pl.solve(("walk", j), st, 1, [], [], [])
        if res is not None:
            pl.look_j = j
            STATS["open_to_see"] += 1
            if DEBUG:
                print("   open_to_see j", j, "trace", [str(c)[:40] for c in res.trace[:4]], flush=True)
            return int(res.action)
    return None


def fallback(pl, st, rng, rec):
    if DEBUG:
        f = pl.facts[st[0]]
        print("   fallback: unseen tokens", int((f == ABSENT).sum()), "standing", len(pl.standing(st[0])), flush=True)
    if ARM["arm"] == "B":
        a = explore(pl, st)
        if a is not None:
            rec["explore"] += 1
            return a
    rec["random"] += 1
    return int(rng.integers(6))


def disagree(pl, st, truth_view):
    """Believed tiles in the believed 13 x 13 view that differ from the simulator's full view (seen tokens only)."""
    v = pl.shown(pl.facts[st[0]], F.M.A[st[1]][:NV])
    f = pl.facts[st[0]]
    A = F.M.A[st[1]][:NV]
    seen = (A >= 0) & (f[np.maximum(A, 0)] != ABSENT)
    seen[CENTRE] = False
    return int((v[seen] != truth_view[:NV][seen]).sum())


# ---------------------------------------------------------------- the familiar worlds: card 045's loop, cropped

def act_job(job):
    idx, layouts, online = job
    W = VP.WORLD
    M = F.M
    W.reset()
    out = []
    for pos, (i, lay) in enumerate(zip(idx, layouts)):
        t0 = time.monotonic()
        rng = np.random.default_rng(1000 + int(i))
        pl = VP.VPlan(W)
        s = ld.start_state(lay)
        Vt = VP.see(lay, s)
        V = crop(Vt, codes_fam(lay, s))
        facts, st, _, _ = pl.observe(None, V)
        rec = {"position": pos, "done": False, "steps": VP.BUDGET, "random": 0, "explore": 0, "pred_wrong": 0,
               "failed_acts": 0, "max_mismatch": 0.0, "walk": Counter(), "door_steps": 0, "howsoon": [],
               "disagree_end": 0}
        for t in range(VP.BUDGET):
            res = pl.choose(st)
            if res is None:
                a = fallback(pl, st, rng, rec)
            else:
                a = res.action
                rec["walk"]["act" if res.trace[-1][0] == "act" else res.trace[-1][1]] += 1
                rec["door_steps"] += ("walk", "blocked") in res.trace
            Vb = pl.view(st)
            pred = pl.step(st, a)
            u, held = pl.front_held(st)
            pcat = M.move_out[a][u] if a in MOVES else W.outcome(a, u, held, pl.ctx_id(st[0], st[1]))[0]
            s2, end = ld.step(lay, s, a)
            Vt2 = VP.see(lay, s2)
            V2 = crop(Vt2, codes_fam(lay, s2))
            facts, st2, miss, ties = pl.observe(facts, V2, end, prefer=pred[1], cands=moved_to(st, a))
            rec["max_mismatch"] = max(rec["max_mismatch"], miss)
            Vb2 = pl.view(st2)
            if DEBUG and pos == 0:
                print("dbg", t, "a", a, "true", s2[:4], "pose", st2[1], "pred", pred[1], "miss", round(miss, 3),
                      "ties", ties, "wrong", disagree(pl, st2, Vt2), "res", None if res is None else str(res.trace[:3])[:150],
                      flush=True)
            if a in MOVES:
                rcat = VP.ENDED if end else (VP.CHANGED if (Vb[:NV] != Vb2[:NV]).any() else VP.UNCHANGED)
            else:
                rcat = int(V[FRONT] != V2[FRONT]) + 2 * int(V[HELD] != V2[HELD])
            if rcat != pcat:
                rec["pred_wrong"] += 1
                if a not in MOVES:
                    pl.failed.add((a, M.fidx[st[1]], st[1], held))
                    pl.version += 1
                    rec["failed_acts"] += 1
            if online:
                W.learn_try(Vb, a, Vb2, end)
                pl.forget()
            s, st, V, Vt = s2, st2, V2, Vt2
            if end:
                rec["done"], rec["steps"] = True, t + 1
                break
        rec["disagree_end"] = disagree(pl, st, Vt)
        rec["refused"] = pl.n_refused
        rec["computed"] = pl.computed
        rec["seconds"] = time.monotonic() - t0
        rec["walk_moves"], rec["walk_seconds"], rec["plan_seconds"] = pl.walk_moves, pl.walk_seconds, pl.plan_seconds
        out.append(rec)
    return out


_vp_act_arm = VP.act_arm


def act_arm(pool, test, online=True):
    res, recs = MV.act_arm(pool, test, online)
    moves = max(res["moves"], 1)
    res["partial_view"] = {"arm": ARM["arm"], "explore_share": round(sum(r["explore"] for r in recs) / moves, 4),
                           "random_share": round(sum(r["random"] for r in recs) / moves, 4),
                           "believed_tiles_wrong_at_end": int(sum(r["disagree_end"] for r in recs)),
                           "episodes_with_wrong_belief": int(sum(r["disagree_end"] > 0 for r in recs))}
    print({"partial_view": res["partial_view"], "success": res["success"],
           "steps": res["mean_steps_when_successful"]}, flush=True)
    return res, recs


# ---------------------------------------------------------------- the chained rooms and cluttered world: card 049's loop, cropped

def wd_act(W, lay, i, M):
    W.reset()
    pl = S7.Plan047(W)
    rng = np.random.default_rng(1000 + i)
    s = M.start_state(lay)
    Vt = M.see(lay, s)
    V = crop(Vt, codes_wd(M, lay, s))
    facts, st, _, _ = pl.observe(None, V)
    rec = {"random": 0, "explore": 0}
    wrong, mism = 0, 0.0
    end = False
    t0 = time.monotonic()
    for t in range(WD.C.BUDGET):
        res = pl.choose(st)
        a = res.action if res is not None else fallback(pl, st, rng, rec)
        Vb = pl.view(st)
        pred = pl.step(st, a)
        s2, end = M.step(lay, s, a)
        Vt = M.see(lay, s2)
        V2 = crop(Vt, codes_wd(M, lay, s2))
        if DEBUG and i in (0, 3):
            print("dbg", i, t, "a", a, "state", s2[:9], "explore", rec["explore"], "via", rec.get("via"), flush=True)
        facts, st, miss, _ = pl.observe(facts, V2, end, prefer=pred[1], cands=moved_to(st, a))
        wrong += bool(miss)
        mism = max(mism, miss)
        W.learn_try(Vb, a, pl.view(st), end)
        pl.forget()
        s, V = s2, V2
        if end:
            break
    return {"success": bool(end), "steps": t + 1, "random_actions": rec["random"], "explore_steps": rec["explore"],
            "predictions_wrong": wrong, "max_mismatch": mism, "disagree_end": disagree(pl, st, Vt),
            "seconds": round(time.monotonic() - t0, 3)}


# ---------------------------------------------------------------- setup

def main():
    args = sys.argv[1:]
    rest = args[args.index("@@") + 1:] if "@@" in args else None
    args = args[:args.index("@@")] if "@@" in args else args
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    path, out = args[0], get("--out", "runs/057/main.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ARM["arm"] = get("--arm", "B")
    enc = RP.load(path, "transition", False)
    tiles = PC.catalogue()
    z = RP.vectors(enc, list(tiles)).astype(np.float32)
    books = fn.normalize(enc.books.detach(), dim=-1).cpu().numpy().astype(np.float32)
    NVM.encoder = lambda *a: {"z": z, "books": books, "rep": {"encoder": "card 054, frozen", "weights": str(path)}}
    sys.path.insert(0, str(TOOLS / "card054"))
    import identity as IDN                             # noqa: E402
    tau, _ = IDN.noise_scale(enc)
    idn = IDN.Identity(tau).learn(z.reshape(len(z), 4, 8).astype(np.float64))

    def vectors_of(arm, seed, tr_tiles, pairs, groups, dev, log):
        idn.install(z.astype(np.float64))
        return (np.asarray(z, np.float64), [slice(8 * k, 8 * k + 8) for k in range(4)],
                {"encoder": "card 054, identity up to noise", "weights": str(path), "identity": idn.report()})

    RP.CR.vectors_of = vectors_of
    RP.VP.vectors_of = vectors_of
    TK.TPlan.observe = observe
    WD.act = wd_act
    cr_main = CR.main

    def main_partial():                                # after every install of version 10
        VP._act_job = act_job
        VP.act_arm = act_arm
        cr_main()

    CR.main = main_partial
    sys.argv = (["walk2.py"] + rest + ["--out", out]) if rest else \
        ["walk2.py", "--dev", "--arm", "A", "--seeds", "399-399", "--layouts", get("--layouts", "30"),
         "--worlds", get("--worlds", "key,switch,either,both"), "--out", out]
    walk2.main()


if __name__ == "__main__":
    main()
