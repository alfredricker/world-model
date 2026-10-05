"""Card 056: goals from example frames.

Version 10's agent (card 051's walk2 in the familiar worlds) on card 054's frozen encoder with identity up to noise,
with one change: the planner's top condition is a goal inferred from K example frames instead of "the episode ends".

Inference (Bayesian concept learning, the size principle; Tenenbaum and Griffiths 2001). A frame's features are
"the hand holds h" (its held appearance) and "some tile in view shows u" (the appearances in view, the agent's own
tile undrawn), in the agent's own appearance ids. Candidates: the features present in every goal frame. A
hypothesis g (a set of features) scores -|g| log N + K log(1 / p(g)), N the number of candidates and p(g) the share
of the agent's stored experience frames (drawn in proportion to their stored weights) where every feature of g
holds, (n + 1) / (total + 2). Features are added greedily while the score rises.

Planning. ("has", h), ("shows", u) and ("all", g) are conditions; a feature's achievers are version 10's achievers
for a part condition (any pick up, drop or toggle on a tile in view whose predicted effect makes it true, recall's
conditions beneath). When the agent believes its goal holds it idles (turns left).

Arms per test layout and goal: frames (inferred from 5 frames), supplied (the evaluator writes the feature: upper
bound), swapped (another goal's frames; scored on the true goal). Goals: hold the door's key, hold the other key,
the door open, the switch on, in the layout's colours. Also criterion 1 (the inferred condition against the
simulator on random-play states) and criterion 3 (rung 1's interaction-decisive forks).

  bin/prun python tools/card056/goals.py runs/054/b_m0.5_399.pt --out runs/056/main_399.json [--layouts 30]
      [--worlds key,switch] [--codes noise|book]
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch.nn.functional as fn

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card053"))
sys.path.insert(0, str(TOOLS / "card051"))
import recall_probe as RP                              # noqa: E402  (loads card 052's encoder classes)
import planner_check as PC                             # noqa: E402
import walk2                                           # noqa: E402

NVM = RP.NV                                            # card 036's novelty module (the encoder hook)
CM = walk2.CM
TH, S7, G = CM.TH, CM.S7, CM.G
MV = S7.MV
VP, F, TK, CR = MV.VP, MV.F, MV.TK, MV.CR
ld, Kd = VP.ld, VP.Kd
KR = PC.KR
NV, CENTRE, FRONT, HELD = VP.NV, VP.CENTRE, VP.FRONT, VP.HELD
LEFT, PICK, DROP, TOG, MOVES, INTER = VP.LEFT, VP.PICK, VP.DROP, VP.TOG, VP.MOVES, VP.INTER
NF = 512                                               # features: has h (h < 256), shows u (256 + u)
K = 5
GOALS = ("key_match", "key_other", "door", "switch")
SWAP = {"key_match": "key_other", "key_other": "key_match", "door": "switch", "switch": "door"}
ARMS = ("frames", "supplied", "swapped")
OUT = {}
CAPTURED = {}
VOCAB = {}
ONE = "--one-per-episode" in sys.argv     # card 056's declared revision


# ---------------------------------------------------------------- the goals, as the evaluator states them

def target(lay, g):
    """(family, colour) of goal g in layout lay."""
    return {"key_match": ("hold", lay.door_colour), "key_other": ("hold", lay.distractor_colour),
            "door": ("door", lay.door_colour), "switch": ("switch", None)}[g]


def holds(lay, s, fam, col):
    if fam == "hold":
        return (s[3] == 1 and lay.door_colour == col) or (s[3] == 2 and lay.distractor_colour == col)
    if fam == "door":
        return bool(s[6]) and lay.door_colour == col
    return bool(s[7])


def supplied(fam, col):
    """The goal written in by the evaluator, in the agent's appearance ids."""
    if fam == "hold":
        return ("has", int(Kd.APP[KR.code(("key", col))]))
    if fam == "door":
        return ("shows", int(Kd.APP[KR.code(("door", col, 2))]))
    return ("shows", int(Kd.APP[KR.code(("ball", "yellow"))]))


# ---------------------------------------------------------------- features and inference

def undraw_of(c):
    return int(F.M.undraw[int(c)])


def features(V):
    """The frame's features as a sorted tuple of ints (has h, 256 + shown u)."""
    V = np.asarray(V)
    v = V[:NV].astype(np.int64).copy()
    v[CENTRE] = undraw_of(v[CENTRE])
    return tuple(sorted({int(V[HELD])} | {256 + int(u) for u in np.unique(v)}))


def feature_matrix(Vs):
    """(n, NF) bool: the features of each view (rows of appearance ids)."""
    Vs = np.asarray(Vs).astype(np.int64)
    n = len(Vs)
    v = Vs[:, :NV].copy()
    cen = np.unique(v[:, CENTRE])
    lut = np.arange(256)
    for c in cen:
        lut[c] = undraw_of(c)
    v[:, CENTRE] = lut[v[:, CENTRE]]
    B = np.zeros((n, NF), bool)
    B[np.arange(n), Vs[:, HELD]] = True
    B[np.arange(n)[:, None], 256 + v] = True
    return B


def infer(frames, base, contrast=True):
    """The goal's features from the frames' feature tuples; base: (n, NF) bool experience frames. Naming a feature
    costs log N nats, N the features the agent's experience shows (its vocabulary)."""
    common = sorted(set.intersection(*[set(f) for f in frames]))
    if not contrast:
        return tuple(common)
    k, n = len(frames), len(base)
    if not common:
        return ()
    logn = math.log(max(int(VOCAB.get(id(base), base.any(0).sum())), 2))
    chosen, mask, score = [], np.ones(n, bool), k * math.log((n + 2) / (n + 1))
    while True:
        best, bsc = None, score
        for f in common:
            if f in chosen:
                continue
            m = mask & base[:, f]
            p = (m.sum() + 1) / (n + 2)
            sc = -(len(chosen) + 1) * logn + k * math.log(1 / p)
            if sc > bsc + 1e-12:
                best, bsc = f, sc
        if best is None:
            break
        chosen.append(best)
        mask &= base[:, best]
        score = bsc
    return tuple(sorted(chosen))


def as_condition(feats):
    c = [("has", f) if f < 256 else ("shows", f - 256) for f in feats]
    return c[0] if len(c) == 1 else ("all", tuple(c))


def cond_holds_matrix(feats, B):
    return B[:, list(feats)].all(1) if feats else np.ones(len(B), bool)


def cname(c):
    J = VP.WORLD.judge
    if c[0] == "all":
        return " and ".join(cname(x) for x in c[1]) or "(nothing)"
    if c[0] == "has":
        return f"holds {J.name(c[1])}"
    return f"shows {J.name(c[1])}"


# ---------------------------------------------------------------- random play (goal frames, test states)

def play(lay, steps, rng):
    """States of one random-play episode from the ordinary start, as the agent's own experience was collected
    (card 034: uniform random actions, up to 640 steps, no play starts)."""
    s = ld.start_state(lay)
    out = [s]
    for _ in range(steps):
        s2, end = ld.step(lay, s, int(rng.integers(6)))
        if end:
            break
        s = s2
        out.append(s)
    return out


def pools(rule, seed, n_layouts=300, cap=400):
    """Goal frames per (family, colour): feature tuples of states from fresh layouts where the goal holds. Under
    ONE (the declared revision), one frame per episode, so that a draw of K frames comes from K episodes."""
    rng = np.random.default_rng(seed)
    P = {}
    for _ in range(n_layouts * (2 if ONE else 1)):
        lay = ld.make_layout(8, rule, rng)
        ep = {}
        for s in play(lay, 640, rng):
            for fam, col in (("hold", lay.door_colour), ("hold", lay.distractor_colour), ("door", lay.door_colour),
                             ("switch", None)):
                if holds(lay, s, fam, col):
                    ep.setdefault((fam, col), []).append((lay, s))
        for key, L in ep.items():
            P.setdefault(key, []).extend([L[int(rng.integers(len(L)))]] if ONE else L)
    out = {}
    for key, L in P.items():
        sel = rng.permutation(len(L))[:cap]
        out[key] = [features(VP.see(*L[i])) for i in sel]
    return out, {f"{k[0]} {k[1]}": len(v) for k, v in P.items()}


# ---------------------------------------------------------------- the planner's new conditions

class GoalAch:
    __slots__ = ("a", "u", "j")

    def __init__(self):
        self.a, self.u, self.j = None, None, None


def install_planner():
    hold0, ach0 = S7.Plan047.hold, S7.Plan047.achievers

    def hold(self, c, st):
        k = c[0]
        if k == "has":
            return int(self.facts[st[0]][HELD]) == c[1]
        if k == "shows":
            return bool((self.facts[st[0]][TK.LAT] == c[1]).any())
        if k == "all":
            return all(hold(self, x, st) for x in c[1])
        return hold0(self, c, st)

    def achievers(self, c, st):
        k = c[0]
        if k == "all":
            return [(GoalAch(), list(c[1]))]
        if k not in ("has", "shows"):
            return ach0(self, c, st)
        key = (c, st[0], st[1])
        r = self.achmemo.get(key)
        if r is None:
            bit = 2 if k == "has" else 1
            r = []
            for a in INTER:
                for u in self.things(self.facts[st[0]]):
                    for needs in self.conditions(a, u, None, c, st, bit):
                        r.append((VP.Ach(a, u, None), needs + [("face", a, u, None)]))
            self.achmemo[key] = r
        return r

    def choose(self, st):
        root = getattr(self, "goal", ("end",))
        res = None
        for _ in range(12):
            v = self.version
            res = self.solve(root, st, 0, [], [], [])
            if res is not None or self.version == v:
                break
        return res

    S7.Plan047.hold = hold
    S7.Plan047.achievers = achievers
    G.Plan.choose = choose


# ---------------------------------------------------------------- the evaluator's search

def search(lay, s, fam, col, limit=60):
    """(fewest steps to the goal, the first actions of every shortest route) in the simulator; (None, ()) beyond
    limit. Stepping onto the goal square ends the episode and is not a route."""
    if holds(lay, s, fam, col):
        return 0, ()
    first = {s: 0}
    front = [s]
    for d in range(1, limit + 1):
        nxt = {}
        for x in front:
            m = first[x]
            for a in range(6):
                y, end = ld.step(lay, x, a)
                if end or y == x or (y in first and y not in nxt):
                    continue
                nxt[y] = nxt.get(y, 0) | (m if d > 1 else (1 << a))
        hit = 0
        for y, m in nxt.items():
            if holds(lay, y, fam, col):
                hit |= m
        if hit:
            return d, tuple(a for a in range(6) if hit >> a & 1)
        first.update(nxt)
        front = list(nxt)
        if not front:
            break
    return None, ()


# ---------------------------------------------------------------- the workers' jobs

def episode(W, lay, g, root, seed):
    M = F.M
    fam, col = target(lay, g)
    rng = np.random.default_rng(seed)
    pl = VP.VPlan(W)
    pl.goal = root
    s = ld.start_state(lay)
    V = VP.see(lay, s)
    facts, st, _, _ = pl.observe(None, V)
    rec = {"done": False, "steps": VP.BUDGET, "random": 0, "idle": 0, "pred_wrong": 0, "hand_condition": False,
           "ended": False}
    t0 = time.monotonic()
    for t in range(VP.BUDGET):
        if pl.hold(root, st):
            a = LEFT
            rec["idle"] += 1
        else:
            res = pl.choose(st)
            if res is None:
                a = int(rng.integers(6))
                rec["random"] += 1
            else:
                a = res.action
                if any(isinstance(c, tuple) and c and c[0] == "part" and c[1] == VP.HELDP for c in res.trace):
                    rec["hand_condition"] = True
        pred = pl.step(st, a)
        u, held = pl.front_held(st)
        pcat = M.move_out[a][u] if a in MOVES else W.outcome(a, u, held, pl.ctx_id(st[0], st[1]))[0]
        s2, end = ld.step(lay, s, a)
        V2 = VP.see(lay, s2)
        facts, st2, _, _ = pl.observe(facts, V2, end, prefer=pred[1])
        if a in MOVES:
            rcat = VP.ENDED if end else (VP.CHANGED if (V[:NV] != V2[:NV]).any() else VP.UNCHANGED)
        else:
            rcat = int(V[FRONT] != V2[FRONT]) + 2 * int(V[HELD] != V2[HELD])
        if rcat != pcat:
            rec["pred_wrong"] += 1
            if a not in MOVES:
                pl.failed.add((a, M.fidx[st[1]], st[1], held))
                pl.version += 1
        W.learn_try(V, a, V2, end)
        pl.forget()
        s, st, V = s2, st2, V2
        if holds(lay, s, fam, col):
            rec["done"], rec["steps"] = True, t + 1
            break
        if end:
            rec["ended"] = True
            break
    rec["seconds"] = time.monotonic() - t0
    rec["shortest"] = search(lay, ld.start_state(lay), fam, col)[0]
    return rec


def fork(W, lay, s, g, roots):
    """Rung 1's fork: the evaluator's fastest first actions; the agent's action under each root."""
    fam, col = target(lay, g)
    d, best = search(lay, s, fam, col, limit=40)
    out = {"d": d, "best": list(best), "chosen": {}}
    if not d or not best or not set(best) <= set(INTER):
        return out
    for arm, root in roots.items():
        pl = VP.VPlan(W)
        pl.goal = root
        facts, st, _, _ = pl.observe(None, VP.see(lay, s))
        if pl.hold(root, st):
            out["chosen"][arm] = "idle"
        else:
            res = pl.choose(st)
            out["chosen"][arm] = None if res is None else int(res.action)
            if arm == "frames" and out["chosen"][arm] not in best:
                u, h = pl.front_held(st)
                out["miss"] = {"front": W.judge.name(int(u)), "held": W.judge.name(int(h)), "state": list(map(str, s)),
                               "trace": None if res is None else [str(c)[:60] for c in res.trace[:6]]}
    return out


def job(item):
    k, items = item
    W = VP.WORLD
    out = []
    for it in items:
        W.reset()
        if it[0] == "episode":
            _, lay, g, arm, root, seed = it
            r = episode(W, lay, g, root, seed)
            r.update({"goal": g, "arm": arm})
        else:
            _, lay, s, g, roots = it
            r = fork(W, lay, s, g, roots)
            r["goal"] = g
        out.append(r)
    return k, out


def run_jobs(pool, items, chunks=60):
    parts = [c.tolist() for c in np.array_split(np.arange(len(items)), chunks) if len(c)]
    res = [None] * len(items)
    for k, out in pool.imap_unordered(job, [(k, [items[i] for i in c]) for k, c in enumerate(parts)]):
        for i, r in zip(parts[k], out):
            res[i] = r
    return res


# ---------------------------------------------------------------- one world

def goal_act_arm(pool, test, online=True):
    t00 = time.monotonic()
    W = VP.WORLD
    rule = test[0].rule
    D = CAPTURED["data"][rule]
    rng = np.random.default_rng(56)
    o = OUT.setdefault("worlds", {})[rule] = {}
    # the agent's experience: stored frames drawn in proportion to their stored weights
    w = D["w"] / D["w"].sum()
    rows = rng.choice(len(w), size=min(100_000, len(w)), replace=True, p=w)
    base = feature_matrix(Kd.APP[D["ego0"][rows]])
    # goal frames from fresh layouts; test states from random play on the test layouts
    P, sizes = pools(rule, 9000 + "key switch either both".split().index(rule))
    o["goal_frame_pools"] = sizes
    VOCAB[id(base)] = int(base.any(0).sum())
    o["vocabulary"] = VOCAB[id(base)]
    trs = np.random.default_rng(57)
    states = [(lay, s) for lay in test for _ in range(4) for s in play(lay, 640, trs)]

    # criterion 1: the inferred condition against the simulator, per goal instance (up to 500 states where it
    # holds and 500 where it does not, from play on the test layouts)
    c1 = {}
    for (fam, col), pool_f in sorted(P.items(), key=str):
        truth = np.array([holds(lay, s, fam, col) for lay, s in states])
        pos, neg = np.flatnonzero(truth), np.flatnonzero(~truth)
        e = c1.setdefault(f"{fam} {col}", {"positives": int(len(pos)), "pool": len(pool_f)})
        if len(pos) < 20:
            continue
        sel = np.concatenate([trs.permutation(pos)[:500], trs.permutation(neg)[:500]])
        Bt = feature_matrix(np.stack([VP.see(*states[i]) for i in sel]))
        tr_ = truth[sel]
        for kk in (1, 2, 3, 5, 10):
            for contrast in (True, False):
                accs, gs = [], Counter()
                for dr in range(20):
                    fr = [pool_f[i] for i in rng.choice(len(pool_f), size=kk, replace=len(pool_f) < kk)]
                    g = infer(fr, base, contrast)
                    gs[g] += 1
                    pred = cond_holds_matrix(g, Bt)
                    accs.append(0.5 * ((pred & tr_).sum() / tr_.sum() + (~pred & ~tr_).sum() / (~tr_).sum()))
                key = f"K={kk}" + ("" if contrast else ", no base rates")
                e[key] = round(float(np.mean(accs)), 4)
                if kk == K:
                    e[key + ": goals"] = [[cname(as_condition(g)) if g else "(nothing)", n] for g, n in gs.most_common(3)]
    o["criterion_1"] = c1

    # acting: four goals x three arms per test layout
    def frames_root(lay, g, seed):
        fam, col = target(lay, g)
        pf = P.get((fam, col))
        if not pf:                                     # no goal frames for this colour: nothing to infer
            return ("all", ())
        r2 = np.random.default_rng(seed)
        return as_condition(infer([pf[i] for i in r2.choice(len(pf), size=K, replace=len(pf) < K)], base))

    items, roots = [], {}
    for i, lay in enumerate(test):
        for gi, g in enumerate(GOALS):
            sd = 10_000 * gi + i
            roots[(i, g)] = {"frames": frames_root(lay, g, sd), "supplied": supplied(*target(lay, g)),
                             "swapped": frames_root(lay, SWAP[g], sd + 500_000)}
            for arm in ARMS:
                items.append(("episode", lay, g, arm, roots[(i, g)][arm], 1000 + i))
    recs = run_jobs(pool, items)
    acting = {}
    for arm in ARMS:
        rs = [r for r in recs if r["arm"] == arm]
        ok = [r for r in rs if r["done"]]
        acting[arm] = {"success": round(len(ok) / len(rs), 4),
                       "per_goal": {g: round(float(np.mean([r["done"] for r in rs if r["goal"] == g])), 4)
                                    for g in GOALS},
                       "steps_ratio_to_shortest": round(float(np.mean([r["steps"] for r in ok]) /
                                                              np.mean([r["shortest"] for r in ok])), 3) if ok else None,
                       "random_share": round(sum(r["random"] for r in rs) / max(sum(r["steps"] for r in rs), 1), 4),
                       "idle_steps": int(sum(r["idle"] for r in rs)),
                       "ended_on_goal_square": int(sum(r["ended"] for r in rs)),
                       "door_via_hand_condition": round(float(np.mean([r["hand_condition"] for r in rs
                                                                       if r["goal"] == "door"])), 4),
                       "seconds_per_episode": round(float(np.mean([r["seconds"] for r in rs])), 3)}
    o["acting"] = acting
    o["goals_inferred"] = Counter(f"{g}: {cname(roots[k]['frames'])}" for k in roots for g in [k[1]]).most_common(12)
    o["frames_equal_supplied"] = round(float(np.mean([roots[k]["frames"] == roots[k]["supplied"] for k in roots])), 4)

    # criterion 3: rung 1's interaction-decisive forks on random-play states
    cand = [(lay, s) for lay, s in states if any(ld.step(lay, s, a)[0] != s for a in INTER)]
    cand = [cand[i] for i in trs.permutation(len(cand))[:300]]
    fitems = []
    for j, (lay, s) in enumerate(cand):
        i = test.index(lay)
        for g in GOALS:
            fitems.append(("fork", lay, s, g, {a: roots[(i, g)][a] for a in ARMS}))
    frs = run_jobs(pool, fitems)
    dec = [r for r in frs if r["chosen"]]
    c3 = {"candidate_states": len(cand), "decisive_forks": len(dec),
          "per_goal": {g: sum(r["goal"] == g for r in dec) for g in GOALS}}
    for arm in ARMS:
        c3[f"top1_{arm}"] = round(float(np.mean([r["chosen"][arm] in r["best"] for r in dec])), 4) if dec else None
        c3[f"top1_{arm}_per_goal"] = {g: round(float(np.mean([r["chosen"][arm] in r["best"] for r in dec
                                                               if r["goal"] == g])), 4)
                                      for g in GOALS if any(r["goal"] == g for r in dec)}
    c3["misses"] = [dict(r["miss"], goal=r["goal"], best=r["best"], chosen=r["chosen"]["frames"])
                    for r in dec if "miss" in r][:12]
    o["criterion_3"] = c3
    o["seconds"] = round(time.monotonic() - t00, 1)
    fr = acting["frames"]
    print(json.dumps({"world": rule, "acting": {a: acting[a]["success"] for a in ARMS}, "c3": {k: v for k, v in c3.items()
          if k.startswith("top1_") and not k.endswith("goal")}, "c1_min_K5": min((v["K=5"] for v in c1.values() if "K=5" in v), default=None)}),
          flush=True)
    save()
    return {"success": fr["success"], "mean_steps_when_successful": 1.0, "moves": 1, "seconds_per_layout_mean": 0.0}, recs


def save():
    if OUT.get("path"):
        Path(OUT["path"]).write_text(json.dumps(OUT, indent=1, default=str) + "\n")


# ---------------------------------------------------------------- setup

def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    path, out = args[0], get("--out", "runs/056/main.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    OUT.update({"note": "Card 056, tools/card056/goals.py", "weights": path, "path": out, "K": K,
                "one_frame_per_episode": ONE})
    enc = RP.load(path, "transition", False)
    tiles = PC.catalogue()
    z = RP.vectors(enc, list(tiles)).astype(np.float32)
    books = fn.normalize(enc.books.detach(), dim=-1).cpu().numpy().astype(np.float32)
    NVM.encoder = lambda *a: {"z": z, "books": books, "rep": {"encoder": "card 054, frozen", "weights": str(path)}}
    if get("--codes", "noise") == "noise":
        sys.path.insert(0, str(TOOLS / "card054"))
        import identity as IDN                         # noqa: E402
        tau, _ = IDN.noise_scale(enc)
        idn = IDN.Identity(tau).learn(z.reshape(len(z), 4, 8).astype(np.float64))

        def vectors_of(arm, seed, tr_tiles, pairs, groups, dev, log):
            idn.install(z.astype(np.float64))
            return (np.asarray(z, np.float64), [slice(8 * k, 8 * k + 8) for k in range(4)],
                    {"encoder": "card 054, identity up to noise", "weights": str(path), "identity": idn.report()})

        RP.CR.vectors_of = vectors_of
        RP.VP.vectors_of = vectors_of
        OUT["identity"] = idn.report()

    cr_main = CR.main

    def main_with_goals():                             # after every install of version 10
        install_planner()
        rv0 = VP.run_vectors

        def run_vectors(label, z, parts, data, *a, **k):
            CAPTURED["data"] = data
            return rv0(label, z, parts, data, *a, **k)

        VP.run_vectors = run_vectors
        VP.act_arm = goal_act_arm
        cr_main()

    CR.main = main_with_goals
    sys.argv = ["walk2.py", "--dev", "--arm", "A", "--seeds", "399-399", "--layouts", get("--layouts", "30"),
                "--worlds", get("--worlds", "key,switch,either,both"), "--out", out.replace(".json", ".dev.json")]
    t0 = time.monotonic()
    walk2.main()
    OUT["seconds"] = round(time.monotonic() - t0, 1)
    save()


if __name__ == "__main__":
    main()
