"""Card 066: CHARTER's three MiniGrid tiers (the user, 2026-10-05), and an agent run on them.

  tier 1  MiniGrid-DoorKey-8x8-v0            8 x 8    key -> door -> goal square              >= 99% success
  tier 2  MiniGrid-BlockedUnlockPickup-v0   11 x 6    move the ball -> key -> door -> the box  against the best version
  tier 3  MiniGrid-ObstructedMaze-Full-v1   16 x 16   keys in boxes, balls at doors; 3,600 steps  against the best version

The adapter (evaluator side, C1). MiniGrid's own environment steps. The agent sees MiniGrid's 7 x 7 view with
occlusion (card 057's crop of the 13 x 13 window, card 060's process_vis), drawn with MiniGrid's tiles, and the
tile it holds; its six actions are MiniGrid's left, right, forward, pick up, drop and toggle ("done" is not used).
The goal: tier 1, the episode's end (stepping onto the goal square); tiers 2 and 3, holding the mission's object,
written in as that object's tile (card 056's "has"). Success is MiniGrid's: terminated with a reward.

Memory, per tier: random play in the environment (uniform actions, about 3.2 million steps, as card 034's), stored
with card 034's sampling (every step that changed a tile or ended the episode, and one in eight of the rest,
weighted 8); each stored pick up, toggle and drop records what was in view in the agent's belief (card 062).

The tile catalogue gains MiniGrid's keys, doors, balls and boxes in its six colours, appended after card 031's
purple ones so that every earlier code keeps its number; card 054's encoder was trained on these objects in
these colours (card 052's generator). The generator also appends six hues to MiniGrid's shared colour list; the
list is restored here, so the environments generate as published.

The agent: version 14 (card 062's runner: cards 057, 060-062 on card 051's planner, card 054's encoder), or a
walking swapped in by a later card (--walking).

  bin/prun python tools/card066/tiers.py --check                              adapter against MiniGrid's own view
  bin/prun python tools/card066/tiers.py --collect --tier 1                   writes runs/066/memory_tier1.npz
  bin/prun python tools/card066/tiers.py --tier 1 --n 200 --out runs/066/v14_tier1.json [--lattice 12] [--walking v14]
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


def _arg(k, d=None):
    a = sys.argv
    return next((a[i + 1] for i in range(len(a) - 1) if a[i] == k), d)


TIER = int(_arg("--tier", "1"))
LATTICE = {1: 12, 2: 14, 3: 20}                       # declared prior: the token lattice covers the map from any start
os.environ.setdefault("WM_LATTICE", _arg("--lattice", str(LATTICE[TIER])))

import numpy as np                                     # noqa: E402

from worldmodel.envs import keydoor_render as KR      # noqa: E402

HUES = ("red", "green", "blue", "purple", "yellow", "grey")
for _o in ([("key", "purple")] + [("door", "purple", st) for st in range(3)]          # card 031's, in its order
           + [("key", c) for c in HUES] + [("door", c, st) for c in HUES for st in range(3)]
           + [("ball", c) for c in HUES] + [("box", c) for c in HUES]):
    if _o not in KR.OBJ_INDEX:
        KR.OBJ_INDEX[_o] = len(KR.OBJECTS)
        KR.OBJECTS.append(_o)
KR.N_CODES = len(KR.OBJECTS) * 5
assert KR.OBJ_INDEX[("key", "purple")] == 18 and KR.N_CODES <= 255

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for _d in ("card062", "card057", "card056"):
    sys.path.insert(0, str(TOOLS / _d))
import believed as BL                                  # noqa: E402  (card 062 -> 061 -> ... -> 038)
import partial as PV                                   # noqa: E402  (card 057; flags set below)
import goals as GL                                     # noqa: E402  (card 056's "has" goals)

import gymnasium as gym                                # noqa: E402
import minigrid                                        # noqa: E402, F401  (registers the environments)
from minigrid.core import constants as K               # noqa: E402

EXTRA_HUES = K.COLOR_NAMES[len(HUES):]                 # card 052's generator appended hues to the shared list:
del K.COLOR_NAMES[len(HUES):]                          # restored except while the generator runs (card 054's noise)
assert tuple(sorted(K.COLOR_NAMES)) == tuple(sorted(HUES))

M61, walk2 = BL.M61, PV.walk2


def _name_of_code(c):
    """The evaluator's name of a tile in MiniGrid's words (reports and traces only)."""
    o = KR.OBJECTS[int(c) // 5]
    if o is None:
        return "floor"
    if isinstance(o, str):
        return o
    if o[0] == "door":
        return f"door {o[1]} {('locked', 'closed', 'open')[o[2]]}"
    return f"{o[0]} {o[1]}"


BL.C.name_of_code = _name_of_code                      # card 031's names (a ball was "switch", a box "vase")
CM, S7, MV = PV.CM, PV.S7, PV.MV
VP, F, TK, CR, CS, SP = PV.VP, PV.F, PV.TK, PV.CR, MV.CS, MV.SP
Kd = VP.Kd
NV, NPL, CENTRE, FRONT, HELD, R = PV.NV, PV.NPL, PV.CENTRE, PV.FRONT, PV.HELD, TK.R
INTER, MOVES = VP.INTER, VP.MOVES
ENVS = {1: "MiniGrid-DoorKey-8x8-v0", 2: "MiniGrid-BlockedUnlockPickup-v0", 3: "MiniGrid-ObstructedMaze-Full-v1"}
OUT = ROOT / "runs" / "066"
MEMORY_STEPS = 3_200_000                              # random play per tier (card 034: 5,000 episodes of 640 steps)
P_UNIFORM = 1 / 8
SEED_MEMORY, SEED_TEST = 66_000, 1_000_000


# ---------------------------------------------------------------- MiniGrid's grid as tile codes

def _code_table():
    O, C = K.OBJECT_TO_IDX, K.COLOR_TO_IDX
    L = np.full((max(O.values()) + 1, max(C.values()) + 1, 3), -1, np.int64)
    L[O["empty"]] = KR.code(None)
    L[O["wall"]] = KR.code("wall")
    L[O["goal"]] = KR.code("goal")
    for c in HUES:
        for kind in ("key", "ball", "box"):
            L[O[kind], C[c]] = KR.code((kind, c))
        for st in range(3):                             # MiniGrid: 0 open, 1 closed, 2 locked; the renderer: 2 open, 0 locked
            L[O["door"], C[c], st] = KR.code(("door", c, 2 - st))
    return L


CODE = _code_table()


def obj_code(o):
    if o is None:
        return KR.code(None)
    return int(CODE[K.OBJECT_TO_IDX[o.type], K.COLOR_TO_IDX[o.color], 0])


def grid_codes(env):
    """MiniGrid's grid -> top-down tile codes (height + 1, width), the held tile in the last row; no agent drawn."""
    u = env.unwrapped
    e = u.grid.encode()                                 # (width, height, 3)
    g = CODE[e[..., 0], e[..., 1], e[..., 2]].T
    assert (g >= 0).all(), "a tile outside the catalogue"
    held = np.full((1, g.shape[1]), KR.code(None), np.int64)
    held[0, 0] = obj_code(u.carrying)
    return np.concatenate([g, held], 0)


def ego(G, xs, ys, ds):
    """Top-down codes (n, H + 1, W) and the agent's places and headings -> egocentric codes (n, NPL): 13 x 13 around
    the agent, facing up, wall beyond the grid (card 016's), then the held row."""
    G, xs, ys, ds = np.asarray(G), np.asarray(xs), np.asarray(ys), np.asarray(ds)
    n, H, Wd = len(G), G.shape[1] - 1, G.shape[2]
    X, Y = xs[:, None, None] + Kd._DX[ds], ys[:, None, None] + Kd._DY[ds]
    ok = (X >= 0) & (X < Wd) & (Y >= 0) & (Y < H)
    b = np.arange(n)[:, None, None]
    e = np.where(ok, G[b, Y.clip(0, H - 1), X.clip(0, Wd - 1)], Kd.WALL).astype(np.int64)
    e[:, R, R] = (e[:, R, R] // 5) * 5 + Kd.UP + 1
    held = np.full((n, 1, 2 * R + 1), KR.code(None), np.int64)
    held[:, 0, 0] = G[:, H, 0]
    return np.concatenate([e, held], 1).reshape(n, NPL)


def make(tier):
    return gym.make(ENVS[tier]).unwrapped


def now_codes(env):
    u = env.unwrapped
    x, y = u.agent_pos
    return ego(grid_codes(u)[None], [x], [y], [u.agent_dir])[0]


def goal_of(env, tier):
    """The evaluator's goal, in the agent's terms: tier 1 the episode's end; tiers 2 and 3 holding the mission's
    object, written in as that object's tile."""
    if tier == 1:
        return None
    o = env.unwrapped.obj
    return ("has", int(Kd.APP[KR.code((o.type, o.color))]))


# ---------------------------------------------------------------- the adapter's check

def check(n=300):
    """Our 7 x 7 view (codes and what is visible) against MiniGrid's own observation, in random states of every
    tier (the agent's own place aside: MiniGrid shows the carried thing there)."""
    rep = {}
    rev = {}
    for o in KR.OBJECTS:
        rev[KR.code(o)] = o
    for tier in ENVS:
        env = make(tier)
        rng = np.random.default_rng(tier)
        bad_tile = bad_vis = states = 0
        colours = set()
        for ep in range(20):
            env.reset(seed=int(rng.integers(1 << 30)))
            for t in range(n // 20):
                codes = now_codes(env)
                vis = PV.visible(codes)
                obs = env.gen_obs()["image"]               # (7, 7, 3): [i, j], the agent at (3, 6) facing up
                E = codes[:NV].reshape(13, 13)[R - 6:R + 1, R - 3:R + 4]
                Vm = vis.reshape(13, 13)[R - 6:R + 1, R - 3:R + 4]
                for i in range(7):
                    for j in range(7):
                        if (i, j) == (3, 6):
                            continue
                        seen = obs[i, j, 0] != K.OBJECT_TO_IDX["unseen"]
                        bad_vis += seen != Vm[j, i]
                        if seen:
                            bad_tile += int(CODE[obs[i, j, 0], obs[i, j, 1], obs[i, j, 2]]) != int(E[j, i]) // 5 * 5
                states += 1
                g = env.grid.encode()
                colours |= {K.IDX_TO_COLOR[c] for c in np.unique(g[..., 1][g[..., 0] > 2])}
                env.step(int(rng.integers(6)))
        rep[ENVS[tier]] = {"states": states, "tiles_differing": int(bad_tile), "visibility_differing": int(bad_vis),
                           "colours": sorted(colours), "size": [env.width, env.height], "max_steps": env.max_steps}
        print(ENVS[tier], rep[ENVS[tier]], flush=True)
    return rep


# ---------------------------------------------------------------- memory: random play

def _play(job):
    tier, seeds, max_steps = job
    rng = np.random.default_rng(seeds[0] + 7)
    env = make(tier)
    out = {k: [] for k in ("ego0", "ego1", "act", "w", "term1", "pres", "stale")}
    stats = Counter()
    for sd in seeds:
        env.reset(seed=int(sd))
        Gs, xs, ys, ds, A, term = [grid_codes(env)], [env.agent_pos[0]], [env.agent_pos[1]], [env.agent_dir], [], []
        for t in range(max_steps):
            a = int(rng.integers(6))
            _, rew, te, tr, _ = env.step(a)
            A.append(a)
            term.append(bool(te))
            Gs.append(grid_codes(env))
            xs.append(env.agent_pos[0]), ys.append(env.agent_pos[1]), ds.append(env.agent_dir)
            if te or tr:
                break
        G = np.stack(Gs)
        E = ego(G, xs, ys, ds)
        A, term = np.array(A, np.int64), np.array(term)
        Tn = len(A)
        changed = (G[1:] // 5 != G[:-1] // 5).reshape(Tn, -1).any(1)
        forced = changed | term
        i = np.flatnonzero(forced | (rng.random(Tn) < P_UNIFORM))
        r = i[np.isin(A[i], INTER)]
        pres, stale = BL.believe(E, A, r)
        out["ego0"].append(E[i].astype(np.uint8)), out["ego1"].append(E[i + 1].astype(np.uint8))
        out["act"].append(A[i]), out["term1"].append(term[i])
        out["w"].append(np.where(forced[i], 1.0, 1 / P_UNIFORM))
        out["pres"].append(pres), out["stale"].append(stale)
        stats["episodes"] += 1
        stats["steps"] += Tn
        stats["successes"] += int(term.any())
        for t in np.flatnonzero(changed):
            a = int(A[t])
            stats[f"changed_by_{VP.KNAME[a] if a in VP.KNAME else a}"] += 1
        held0, held1 = G[:-1, -1, 0], G[1:, -1, 0]
        stats["pickups"] += int(((held0 == KR.code(None)) & (held1 != KR.code(None))).sum())
        for t in np.flatnonzero(changed & (A == VP.TOG)):
            d = (G[t + 1] != G[t]) & (G[t + 1] // 5 != G[t] // 5)
            for c in G[t + 1][d].tolist():
                o = KR.OBJECTS[c // 5]
                if isinstance(o, tuple) and o[0] == "door" and o[2] == 2:
                    stats["doors_opened"] += 1
                if isinstance(o, tuple) and o[0] == "key":
                    stats["keys_out_of_boxes"] += 1
    return {k: np.concatenate(v) for k, v in out.items()}, stats


def collect(tier, pool):
    env = make(tier)
    max_steps = env.max_steps
    n_ep = MEMORY_STEPS // max_steps
    seeds = SEED_MEMORY + 100_000 * tier + np.arange(n_ep)
    jobs = [(tier, s.tolist(), max_steps) for s in np.array_split(seeds, 80)]
    parts = pool.map(_play, jobs)
    D = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    stats = sum((p[1] for p in parts), Counter())
    return D, dict(stats)


def memory(tier, pool=None):
    path = OUT / f"memory_tier{tier}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}, json.loads((OUT / f"memory_tier{tier}.json").read_text())
    BL.moves_from_card061()
    own = pool is None
    pool = pool or F.pool20()
    t0 = time.monotonic()
    D, stats = collect(tier, pool)
    if own:
        pool.close()
    stats.update({"rows": int(len(D["act"])), "tries": int(np.isin(D["act"], INTER).sum()),
                  "tries_with_stale_belief": int((D["stale"] > 0).sum()), "seconds": round(time.monotonic() - t0, 1),
                  "env": ENVS[tier]})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **D)
    (OUT / f"memory_tier{tier}.json").write_text(json.dumps(stats, indent=1) + "\n")
    return D, stats


# ---------------------------------------------------------------- version 14, installed as card 062's runner installs it

def install_v14():
    PV.OCCLUDE = PV.FRONTIER = PV.KEEP_LOOK = True     # card 060's flags and its declared revision
    PV.ARM["arm"] = "B"
    TK.TPlan.observe = PV.observe                       # card 057's placing
    walk2.install()                                     # card 051: index, threats, commit, two tokens
    CS.install()                                        # card 048's setup, as card 049's runner calls it
    MV.install("045")
    SP.install()
    VP.Kind = CR.kind
    S7.install()
    VP.T.configure()
    GL.install_planner()                                # card 056: goals over the held tile
    M61.CTX = lambda D, sel: D["pres"]                  # card 062: what was in view as the agent believed it
    TK._world_init = M61.world_init                     # card 061: memory learned from the small view


def encoder(path=ROOT / "runs" / "054" / "b_m0.5_399.pt"):
    """Card 054's encoder (seed 399), identity up to noise; the vectors of every tile the catalogue can show."""
    import torch.nn.functional as fn
    enc = PV.RP.load(str(path), "transition", False)
    tiles = PV.PC.catalogue()
    z = PV.RP.vectors(enc, list(tiles)).astype(np.float32)
    sys.path.insert(0, str(TOOLS / "card054"))
    import identity as IDN                             # noqa: E402
    K.COLOR_NAMES.extend(EXTRA_HUES)
    try:
        tau, _ = IDN.noise_scale(enc)
    finally:
        del K.COLOR_NAMES[len(HUES):]
    idn = IDN.Identity(tau).learn(z.reshape(len(z), 4, 8).astype(np.float64))
    idn.install(z.astype(np.float64))
    del fn
    return np.asarray(z, np.float64), [slice(8 * k, 8 * k + 8) for k in range(4)], idn.report()


def setup(tier, log):
    install_v14()
    for f in INSTALL:                                   # a later card's component
        f()
    z, parts, idrep = encoder()
    S = VP.Store(z)
    lut = np.zeros(256, np.uint8)
    lut[:len(S.hof)] = S.hof
    Kd.APP = lut
    D, stats = memory(tier)
    log(f"memory tier {tier}: {stats}")
    W = VP.World(S, parts, D, "cuda", log)
    T61 = W.report["partial_memory"]["moves"]
    BL.moves_from_card061()
    same = all(np.array_equal(np.array(T61[VP.KNAME[a]]["M"]), BL.T[a][0])
               and np.array_equal(np.array(T61[VP.KNAME[a]]["b"]), BL.T[a][1]) for a in MOVES)
    assert same, "the believed views used other transformations than this memory's"
    VP.WORLD, F.M = W, W.M
    import torch
    torch.cuda.empty_cache()                           # the workers use none of it; other runs may share the GPU
    return W, {"memory": stats, "identity": idrep, "model_seconds": W.seconds, "placements": int(len(W.M.A)),
               "token_ids": int(TK.NT), "lattice": int(TK.LR), "moves_same_as_card_061": same}


TRACE = "--trace" in sys.argv


def name(c):
    """A condition in the evaluator's words (trace only)."""
    J = VP.WORLD.judge
    nm = lambda h: J.name(int(h)) if h is not None else "?"
    try:
        if c[0] == "has":
            return f"holds {nm(c[1])}"
        if c[0] == "face":
            return f"face {VP.KNAME.get(c[1], c[1])} {nm(c[2]) if c[3] is None else c[3]}"
        if c[0] == "walk":
            return f"walk {c[1]}"
        if c[0] == "part":
            return f"part {'held' if c[1] == VP.HELDP else 'view'} for {VP.KNAME.get(c[2], c[2])} {nm(c[3])}"
        return str(c)[:40]
    except Exception:
        return str(c)[:40]


INSTALL = []                                            # later cards append their install functions
COUNTS = []                                             # and dicts of counts to report per episode
PLANNER = {"cls": None}


# ---------------------------------------------------------------- one episode

def episode(job):
    tier, seed = job
    W = VP.WORLD
    M = F.M
    W.reset()
    pl = (PLANNER["cls"] or S7.Plan047)(W)
    env = make(tier)
    env.reset(seed=int(seed))
    goal = goal_of(env, tier)
    if goal is not None:
        pl.goal = goal
    rng = np.random.default_rng(seed)
    x0, y0 = env.agent_pos
    codes = now_codes(env)
    V = PV.crop(Kd.APP[codes], codes)
    facts, st, _, _ = pl.observe(None, V)
    rec = {"seed": int(seed), "random": 0, "explore": 0, "beyond_lattice": 0, "pred_wrong": 0}
    te = tr = False
    rew = 0.0
    c0 = [dict(c) for c in COUNTS]
    t0 = time.monotonic()
    for t in range(env.max_steps):
        res = pl.choose(st)
        a = res.action if res is not None else PV.fallback(pl, st, rng, rec)
        if TRACE:
            fr = env.grid.get(*env.front_pos)
            print(f"t {t:3d} a {VP.KNAME.get(a, a) if hasattr(VP.KNAME, 'get') else a} at {tuple(env.agent_pos)} d "
                  f"{env.agent_dir} holds {getattr(env.carrying, 'type', None)} {getattr(env.carrying, 'color', '')} "
                  f"front {getattr(fr, 'type', None)} {getattr(fr, 'color', '')} | "
                  + ("fallback" if res is None else " <- ".join(name(c) for c in res.trace[:6])), flush=True)
        Vb = pl.view(st)
        pred = pl.step(st, a)
        _, rew, te, tr, _ = env.step(a)
        codes = now_codes(env)
        V2 = PV.crop(Kd.APP[codes], codes)
        facts, st2, miss, _ = pl.observe(facts, V2, te, prefer=pred[1], cands=PV.moved_to(st, a))
        rec["pred_wrong"] += bool(miss)
        x, y = env.agent_pos
        rec["beyond_lattice"] += max(abs(x - x0), abs(y - y0)) > TK.LR - R
        W.learn_try(Vb, a, pl.view(st2), te)
        pl.forget()
        st, V = st2, V2
        if te or tr:
            break
    rec.update({"success": bool(te and rew > 0), "steps": t + 1, "seconds": round(time.monotonic() - t0, 2),
                "walk_seconds": round(getattr(pl, "walk_seconds", 0.0), 2),
                "plan_seconds": round(getattr(pl, "plan_seconds", 0.0), 2)})
    for c, b in zip(COUNTS, c0):
        rec.update({k: round(v - b.get(k, 0), 4) for k, v in c.items()})
    return rec


def run(tier, n, out, log, workers=20):
    W, info = setup(tier, log)
    seeds = SEED_TEST + 1000 * tier + np.arange(n)
    pool = F.pool20() if workers == 20 else __import__("multiprocessing").get_context("fork").Pool(workers)
    t0 = time.monotonic()
    recs = []
    for r in pool.imap_unordered(episode, [(tier, int(s)) for s in seeds]):
        recs.append(r)
        log(f"tier {tier} seed {r['seed']}: success {r['success']} steps {r['steps']} random {r['random']} "
            f"explore {r['explore']} {r['seconds']}s ({len(recs)}/{n})")
    pool.close()
    recs.sort(key=lambda r: r["seed"])
    ok = [r for r in recs if r["success"]]
    steps = sum(r["steps"] for r in recs)
    res = {"note": "Card 066, tools/card066/tiers.py", "tier": tier, "env": ENVS[tier], "episodes": n,
           "success": round(len(ok) / n, 4), "successes": len(ok),
           "mean_steps_when_successful": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else None,
           "random_share": round(sum(r["random"] for r in recs) / steps, 4),
           "explore_share": round(sum(r["explore"] for r in recs) / steps, 4),
           "episodes_beyond_lattice": int(sum(r["beyond_lattice"] > 0 for r in recs)),
           "seconds_per_step": round(sum(r["seconds"] for r in recs) / steps, 4),
           "counts": {k: round(sum(r.get(k, 0) for r in recs), 4) for c in COUNTS for k in c},
           "wall_seconds": round(time.monotonic() - t0, 1), "setup": info, "per_episode": recs}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(json.dumps({k: v for k, v in res.items() if k not in ("per_episode", "setup")}))
    return res


def main():
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    if "--check" in sys.argv:
        rep = check()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "adapter_check.json").write_text(json.dumps(rep, indent=1) + "\n")
        return
    if "--collect" in sys.argv:
        D, stats = memory(TIER)
        log(json.dumps(stats))
        return
    walking = _arg("--walking", "v14")
    if TRACE:
        if walking != "v14":
            sys.path.insert(0, str(TOOLS / f"card{walking}"))
            mod = __import__(_arg("--module", "propagate"))
            INSTALL.append(mod.install)
        setup(TIER, log)
        r = episode((TIER, int(_arg("--trace"))))
        log(json.dumps(r))
        return
    if walking != "v14":
        sys.path.insert(0, str(TOOLS / f"card{walking}"))
        mod = __import__(_arg("--module", "propagate"))
        INSTALL.append(mod.install)
        if hasattr(mod, "STATS"):
            COUNTS.append(mod.STATS)
    run(TIER, int(_arg("--n", "20")), _arg("--out", f"runs/066/{walking}_tier{TIER}.json"), log,
        int(_arg("--workers", "20")))


if __name__ == "__main__":
    main()
