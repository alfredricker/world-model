"""Card 048: architecture version 8, unchanged, in three chained rooms.

The world (evaluator side, C1): a 13 x 8 room split by walls at x = 4 and x = 8 into three rooms of 3 x 6, each wall
with a locked door of its own colour. The key world's rule: a door opens when toggled holding the key of its colour,
and closes when toggled open. The agent starts in the middle room with the switch and the vase (as in the key
world, where neither does anything); the goal is in an outer room (left or right at random). Every place is within
6 tiles of the middle room, so it stays inside version 8's placements; the outer rooms leave the view as the agent
walks to the other side.

  one door:  both keys in the middle room; the goal room's door needs its key.
  two doors: the goal room's key lies in the other outer room, behind the other door, whose key is in the middle
             room. Pick up key A, open door A, put key A down, pick up key B, open door B, reach the goal.

The agent is card 047's planner with the key world's memory, as for the familiar key world; memory is reset for
every layout. Nothing in the agent changes.

  bin/prun python tools/card048/chain.py --layouts --n 100                       layouts and shortest routes
  bin/prun python tools/card048/chain.py --random                                 random actions (baseline)
  bin/prun python tools/card048/chain.py --act --seed 399 --n 10 --out runs/048_shakedown_399.json
  bin/prun python tools/card048/chain.py --act --seed 400 --n 100 --n2 30 --out runs/048_seed400.json   (seeds 400-404)
"""
import json
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card047"))
import situations as S7                                # noqa: E402

MV = S7.MV
VP, F, CR, CS, SP = MV.VP, MV.F, MV.CR, MV.CS, MV.SP
Kd = VP.Kd
from worldmodel.envs.keydoor import COLOURS, DIR_VEC   # noqa: E402
from worldmodel.envs.keydoor_render import code        # noqa: E402

LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = range(6)
WD, HT = 13, 8
WALLS = (4, 8)
BUDGET = 200
ROOT = Path(__file__).resolve().parents[2]
LAYOUTS = ROOT / "runs" / "048_layouts.pkl"
VARIANTS = ("one_door", "two_doors")


@dataclass(frozen=True)
class Chain:
    doors: tuple            # ((x, y) of door 0, door 1)
    colours: tuple          # door i's colour, and its key's
    keys: tuple             # where key i lies at the start
    switch: tuple
    vase: tuple
    goal: tuple
    start: tuple            # (x, y, dir)
    variant: str


def start_state(lay):
    """(x, y, dir, carry, key 0, key 1, door 0 open, door 1 open, switch on, vase broken); carry 0 none, i + 1 key i."""
    return (*lay.start, 0, lay.keys[0], lay.keys[1], 0, 0, 0, 0)


def front(s):
    dx, dy = DIR_VEC[s[2]]
    return (s[0] + dx, s[1] + dy)


def cell(lay, s, p):
    x, y = p
    if x <= 0 or y <= 0 or x >= WD - 1 or y >= HT - 1:
        return "wall"
    for i, dp in enumerate(lay.doors):
        if p == dp:
            return f"door{i}"
    if x in WALLS:
        return "wall"
    for i in (0, 1):
        if p == s[4 + i]:
            return f"key{i}"
    if p == lay.switch:
        return "switch"
    if p == lay.vase and not s[9]:
        return "vase"
    if p == lay.goal:
        return "goal"
    return "empty"


def step(lay, s, a):
    x, y, d = s[:3]
    if a == LEFT:
        return (x, y, (d - 1) % 4, *s[3:]), False
    if a == RIGHT:
        return (x, y, (d + 1) % 4, *s[3:]), False
    f = front(s)
    here = cell(lay, s, f)
    t = list(s)
    if a == FORWARD:
        if here in ("empty", "goal") or (here.startswith("door") and s[6 + int(here[4])]):
            return (f[0], f[1], d, *s[3:]), here == "goal"
        return s, False
    if a == PICKUP:
        if s[3] == 0 and here.startswith("key"):
            i = int(here[3])
            t[3], t[4 + i] = i + 1, None
            return tuple(t), False
        return s, False
    if a == DROP:
        if s[3] and here == "empty" and f != lay.goal:
            t[4 + s[3] - 1], t[3] = f, 0
            return tuple(t), False
        return s, False
    if a == TOGGLE:
        if here.startswith("door"):
            i = int(here[4])
            if s[6 + i]:
                t[6 + i] = 0
            elif s[3] == i + 1:
                t[6 + i] = 1
            else:
                return s, False
            return tuple(t), False
        if here == "switch":
            t[8] = 1 - s[8]
            return tuple(t), False
        if here == "vase":
            t[9] = 1
            return tuple(t), False
    return s, False


def encode(lay, s):
    """State -> top-down tile codes (HT + 1, WD), the held thing in the last row, as card 010's logic-door states."""
    g = np.full((HT + 1, WD), code(None), np.int64)
    for x in range(WD):
        for y in range(HT):
            if x in (0, WD - 1) or y in (0, HT - 1) or (x in WALLS and (x, y) not in lay.doors):
                g[y, x] = code("wall")
    g[lay.goal[1], lay.goal[0]] = code("goal")
    for i, (dx, dy) in enumerate(lay.doors):
        g[dy, dx] = code(("door", lay.colours[i], 2 if s[6 + i] else 0))
    g[lay.switch[1], lay.switch[0]] = code(("ball", "yellow" if s[8] else "grey"))
    if not s[9]:
        g[lay.vase[1], lay.vase[0]] = code(("box", "purple"))
    for i in (0, 1):
        if s[4 + i] is not None:
            g[s[4 + i][1], s[4 + i][0]] = code(("key", lay.colours[i]))
    g[HT, 0] = code(None if s[3] == 0 else ("key", lay.colours[s[3] - 1]))
    g[s[1], s[0]] += s[2] + 1
    return g


def ego(g, s):
    """Card 016's egocentric view of any grid: 13 x 13 around the agent, facing up, wall beyond the grid; the held row."""
    x, y, d = s[:3]
    X, Y = x + Kd._DX[d], y + Kd._DY[d]
    ok = (X >= 0) & (X < WD) & (Y >= 0) & (Y < HT)
    e = np.where(ok, g[Y.clip(0, HT - 1), X.clip(0, WD - 1)], Kd.WALL).astype(np.int64)
    e[Kd.R, Kd.R] = (e[Kd.R, Kd.R] // 5) * 5 + Kd.UP + 1
    held = np.zeros((1, 2 * Kd.R + 1), np.int64)
    held[0, 0] = g[HT, 0]
    return np.concatenate([e, held], 0)


def see(lay, s):
    return Kd.APP[ego(encode(lay, s), s)].reshape(MV.NPL).astype(np.int32)


def shortest(lay, cap=200):
    """The evaluator's fewest steps to the goal by breadth-first search over full states (the switch and vase are
    left alone: they open nothing in the key world's rule)."""
    s0 = start_state(lay)
    seen, q = {s0: 0}, deque([s0])
    while q:
        s = q.popleft()
        n = seen[s]
        if n >= cap:
            return None
        for a in range(6):
            f = front(s)
            if a == TOGGLE and cell(lay, s, f) in ("switch", "vase"):
                continue
            s2, end = step(lay, s, a)
            if end:
                return n + 1
            if s2 not in seen:
                seen[s2] = n + 1
                q.append(s2)
    return None


def mirror(lay):
    fx = lambda p: (WD - 1 - p[0], p[1])
    x, y, d = lay.start
    return replace(lay, doors=tuple(fx(p) for p in lay.doors), keys=tuple(fx(p) for p in lay.keys),
                   switch=fx(lay.switch), vase=fx(lay.vase), goal=fx(lay.goal),
                   start=(WD - 1 - x, y, {0: 2, 2: 0}.get(d, d)))


def make_layout(variant, rng):
    """Door 0 at x = 4 leads to the left room, door 1 at x = 8 to the goal's (right) room; mirrored half the time."""
    while True:
        y0, y1 = (int(v) for v in rng.integers(1, HT - 1, 2))
        c = rng.permutation(len(COLOURS))
        mid = [(x, y) for x in range(5, 8) for y in range(1, HT - 1) if (x, y) not in ((5, y0), (7, y1))]
        left = [(x, y) for x in range(1, 4) for y in range(1, HT - 1) if (x, y) != (3, y0)]
        right = [(x, y) for x in range(9, 12) for y in range(1, HT - 1) if (x, y) != (9, y1)]
        pick = rng.permutation(len(mid))
        if variant == "one_door":
            k0, k1, sw, vase, agent = [mid[i] for i in pick[:5]]
        else:
            k0, sw, vase, agent = [mid[i] for i in pick[:4]]
            k1 = left[int(rng.integers(len(left)))]
        goal = right[int(rng.integers(len(right)))]
        lay = Chain(((4, y0), (8, y1)), (COLOURS[c[0]], COLOURS[c[1]]), (k0, k1), sw, vase, goal,
                    (*agent, int(rng.integers(4))), variant)
        if rng.random() < 0.5:
            lay = mirror(lay)
        if shortest(lay) is not None:
            return lay


def make_layouts(n):
    out = {}
    for vi, v in enumerate(VARIANTS):
        rng = np.random.default_rng(48000 + vi)
        lays = [make_layout(v, rng) for _ in range(n)]
        sh = [shortest(lay) for lay in lays]
        out[v] = {"test": [vars(lay) for lay in lays], "shortest_each": sh}
        print(v, "shortest mean", round(float(np.mean(sh)), 2), "min", min(sh), "max", max(sh), flush=True)
    LAYOUTS.write_bytes(pickle.dumps(out))


def load_layouts():
    L = pickle.loads(LAYOUTS.read_bytes())
    for v in L.values():
        v["test"] = [lay if isinstance(lay, Chain) else Chain(**lay) for lay in v["test"]]
    return L


def random_baseline(rollouts=20):
    L = load_layouts()
    res = {}
    for v in VARIANTS:
        hits = []
        for i, lay in enumerate(L[v]["test"]):
            rng = np.random.default_rng(7000 + i)
            for _ in range(rollouts):
                s = start_state(lay)
                for _ in range(BUDGET):
                    s, end = step(lay, s, int(rng.integers(6)))
                    if end:
                        break
                hits.append(end)
        res[v] = float(np.mean(hits))
    print("random actions, goal reached within", BUDGET, "steps:", res)
    return res


def setup(seed):
    """Card 047's agent with the key world's memory, encoder seed `seed` of arm A (as trace runs set it up)."""
    CS.install()
    MV.install("045")
    SP.install()
    VP.Kind, VP.vectors_of = CR.kind, CR.vectors_of
    S7.install()
    VP.T.configure()
    cache = pickle.loads(VP.T.MEMORY.read_bytes())
    groups = VP.T.use_groups(cache["mem"], "cuda")
    z, parts, _ = VP.vectors_of("A", seed, cache["tiles"], cache["pairs"], groups, "cuda", lambda *a: None)
    data = pickle.loads(VP.T.DATA.read_bytes())["data"]
    S = VP.Store(z)
    lut = np.zeros(256, np.uint8)
    lut[:len(S.hof)] = S.hof
    Kd.APP = lut
    W = VP.World(S, parts, data["key"], "cuda", lambda *a: None)
    VP.WORLD, F.M = W, W.M
    return W


def act(W, lay, i):
    """One episode from the same memory; the evaluator's counts."""
    W.reset()
    pl = S7.Plan047(W)
    rng = np.random.default_rng(1000 + i)
    s = start_state(lay)
    V = see(lay, s)
    facts, st, _, _ = pl.observe(None, V)
    rand = wrong = 0
    opened = []
    end = False
    t0 = time.monotonic()
    for t in range(BUDGET):
        res = pl.choose(st)
        a = res.action if res is not None else int(rng.integers(6))
        rand += res is None
        pred = pl.step(st, a)
        s2, end = step(lay, s, a)
        for k in (0, 1):
            if s2[6 + k] and not s[6 + k]:
                opened.append(k)
        V2 = see(lay, s2)
        facts, st, miss, _ = pl.observe(facts, V2, end, prefer=pred[1])
        wrong += bool(miss)
        W.learn_try(V, a, V2, end)
        pl.forget()
        s, V = s2, V2
        if end:
            break
    return {"success": bool(end), "steps": t + 1, "random_actions": rand, "predictions_wrong": wrong,
            "doors_opened": opened, "seconds": round(time.monotonic() - t0, 3)}


def run(seed, n, out, n2=None):
    L = load_layouts()
    W = setup(seed)
    res = {"note": "Card 048, tools/card048/chain.py: version 8 (card 047's agent, key world memory) in chained rooms",
           "seed": seed, "variants": {}}
    for v in VARIANTS:
        k = n2 if v == "two_doors" and n2 else n
        lays, sh = L[v]["test"][:k], L[v]["shortest_each"][:k]
        eps = []
        for i, lay in enumerate(lays):
            e = act(W, lay, i)
            e["shortest"] = sh[i]
            eps.append(e)
            print(seed, v, i, e, flush=True)
        ok = [e for e in eps if e["success"]]
        res["variants"][v] = {
            "layouts": len(eps), "success": round(len(ok) / len(eps), 4),
            "steps_ratio_to_shortest": round(float(np.mean([e["steps"] / e["shortest"] for e in ok])), 4) if ok else None,
            "random_share": round(sum(e["random_actions"] for e in eps) / sum(e["steps"] for e in eps), 4),
            "predictions_wrong": sum(e["predictions_wrong"] for e in eps),
            "seconds_per_layout": round(float(np.mean([e["seconds"] for e in eps])), 3),
            "episodes": eps}
        print(seed, v, {k: x for k, x in res["variants"][v].items() if k != "episodes"}, flush=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    if "--convert" in args:                              # layouts first pickled as __main__.Chain
        L = pickle.loads(LAYOUTS.read_bytes())
        for v in L.values():
            v["test"] = [vars(lay) for lay in v["test"]]
        LAYOUTS.write_bytes(pickle.dumps(L))
    elif "--layouts" in args:
        make_layouts(int(get("--n", "100")))
    elif "--random" in args:
        random_baseline()
    elif "--act" in args:
        seed = int(get("--seed", "399"))
        run(seed, int(get("--n", "10")), get("--out", f"runs/048_seed{seed}.json"), int(get("--n2", "0")) or None)


if __name__ == "__main__":
    main()
