"""Card 049's acting tests outside card 045's harness: card 048's chained rooms and the cluttered key world.

The cluttered key world (evaluator side, C1) is the key world (one 8 x 8 room split by a wall with a locked
door; the door's key, a second key, the switch and the vase, which open nothing) with 2-4 extra objects: keys of
colours the door does not take (the third colour, yellow and purple; yellow and purple never appear in the key
world), switches and vases. Every view is a combination memory never held. The extra keys can be picked up and
dropped like any key; switches toggle, vases break; none of it opens the door. Layouts are kept only when the
evaluator's shortest route needs no extra object moved. Memory is the plain key world's, reset per layout.

  bin/prun python tools/card049/conditions.py --recall v8 --clutter --seed 399 --n 30 --out runs/049_clutter_v8_399.json
"""
import json
import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import conditions as CD
from worldmodel.envs import logicdoor as LD

C, S7, MV = CD.C, CD.S7, CD.MV
Kd, KR = C.Kd, CD.KR
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = range(6)
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
N = 8
ROOT = CD.ROOT
CLUTTER = ROOT / "runs" / "049_clutter_layouts.pkl"
EXTRA_KEY_COLOURS = ("red", "green", "blue", "yellow", "purple")


@dataclass(frozen=True)
class Clutter:
    wall_x: int
    door_y: int
    door_colour: str
    keys: tuple             # ((colour, (x, y)), ...): the door's key first, then the second key, then extra keys
    switches: tuple         # positions; the first is the key world's switch
    vases: tuple            # positions; the first is the key world's vase
    goal: tuple
    start: tuple


def start_state(lay):
    """(x, y, dir, carry, key positions, door open, switches on, vases broken); carry 0 none, i + 1 key i."""
    return (*lay.start, 0, tuple(p for _, p in lay.keys), 0, (0,) * len(lay.switches), (0,) * len(lay.vases))


def cell(lay, s, p):
    x, y = p
    if x <= 0 or y <= 0 or x >= N - 1 or y >= N - 1:
        return ("wall",)
    if x == lay.wall_x:
        return ("door",) if y == lay.door_y else ("wall",)
    for i, q in enumerate(s[4]):
        if q == p:
            return ("key", i)
    for i, q in enumerate(lay.switches):
        if q == p:
            return ("switch", i)
    for i, q in enumerate(lay.vases):
        if q == p and not s[7][i]:
            return ("vase", i)
    if p == lay.goal:
        return ("goal",)
    return ("empty",)


def front(s):
    dx, dy = DIRS[s[2]]
    return (s[0] + dx, s[1] + dy)


def step(lay, s, a):
    x, y, d, carry, keys, door, sw, va = s
    if a == LEFT:
        return (x, y, (d - 1) % 4, *s[3:]), False
    if a == RIGHT:
        return (x, y, (d + 1) % 4, *s[3:]), False
    f = front(s)
    here = cell(lay, s, f)
    if a == FORWARD:
        if here[0] in ("empty", "goal") or (here[0] == "door" and door):
            return (f[0], f[1], d, *s[3:]), here[0] == "goal"
        return s, False
    if a == PICKUP:
        if carry == 0 and here[0] == "key":
            i = here[1]
            k = list(keys)
            k[i] = None
            return (x, y, d, i + 1, tuple(k), door, sw, va), False
        return s, False
    if a == DROP:
        if carry and here[0] == "empty" and f != lay.goal:
            k = list(keys)
            k[carry - 1] = f
            return (x, y, d, 0, tuple(k), door, sw, va), False
        return s, False
    if a == TOGGLE:
        if here[0] == "door":
            if door:
                return (x, y, d, carry, keys, 0, sw, va), False
            if carry and lay.keys[carry - 1][0] == lay.door_colour:
                return (x, y, d, carry, keys, 1, sw, va), False
            return s, False
        if here[0] == "switch":
            t = list(sw)
            t[here[1]] = 1 - t[here[1]]
            return (x, y, d, carry, keys, door, tuple(t), va), False
        if here[0] == "vase":
            t = list(va)
            t[here[1]] = 1
            return (x, y, d, carry, keys, door, sw, tuple(t)), False
    return s, False


def encode(lay, s):
    g = np.full((N + 1, N), KR.code(None), np.int64)
    for x in range(N):
        for y in range(N):
            if x in (0, N - 1) or y in (0, N - 1) or (x == lay.wall_x and y != lay.door_y):
                g[y, x] = KR.code("wall")
    g[lay.goal[1], lay.goal[0]] = KR.code("goal")
    g[lay.door_y, lay.wall_x] = KR.code(("door", lay.door_colour, 2 if s[5] else 0))
    for i, p in enumerate(lay.switches):
        g[p[1], p[0]] = KR.code(("ball", "yellow" if s[6][i] else "grey"))
    for i, p in enumerate(lay.vases):
        if not s[7][i]:
            g[p[1], p[0]] = KR.code(("box", "purple"))
    for (c, _), p in zip(lay.keys, s[4]):
        if p is not None:
            g[p[1], p[0]] = KR.code(("key", c))
    g[N, 0] = KR.code(None if s[3] == 0 else ("key", lay.keys[s[3] - 1][0]))
    g[s[1], s[0]] += s[2] + 1
    return g


def ego(g, s):
    x, y, d = s[:3]
    X, Y = x + Kd._DX[d], y + Kd._DY[d]
    ok = (X >= 0) & (X < N) & (Y >= 0) & (Y < N)
    e = np.where(ok, g[Y.clip(0, N - 1), X.clip(0, N - 1)], Kd.WALL).astype(np.int64)
    e[Kd.R, Kd.R] = (e[Kd.R, Kd.R] // 5) * 5 + Kd.UP + 1
    held = np.zeros((1, 2 * Kd.R + 1), np.int64)
    held[0, 0] = g[N, 0]
    return np.concatenate([e, held], 0)


def see(lay, s):
    return Kd.APP[ego(encode(lay, s), s)].reshape(MV.NPL).astype(np.int32)


def shortest(lay, cap=200):
    """Fewest steps to the goal with every object but the door's key left alone (no other key picked up, no
    switch or vase toggled), so a layout that needs an object moved out of the way has no route."""
    s0 = start_state(lay)
    seen, q = {s0: 0}, deque([s0])
    while q:
        s = q.popleft()
        n = seen[s]
        if n >= cap:
            return None
        for a in range(6):
            here = cell(lay, s, front(s))
            if a == PICKUP and here[0] == "key" and here[1] >= 1:
                continue
            if a == TOGGLE and here[0] in ("switch", "vase"):
                continue
            s2, end = step(lay, s, a)
            if end:
                return n + 1
            if s2 not in seen:
                seen[s2] = n + 1
                q.append(s2)
    return None


def make_layout(rng):
    while True:
        base = LD.make_layout(N, "key", rng)
        used = {base.key, base.distractor, base.switch, base.vase, base.goal, base.start[:2],
                (base.wall_x - 1, base.door_y), (base.wall_x + 1, base.door_y)}
        free = [(x, y) for x in range(1, N - 1) for y in range(1, N - 1)
                if x != base.wall_x and (x, y) not in used]
        k = int(rng.integers(2, 5))
        if len(free) < k:
            continue
        spots = [free[i] for i in rng.permutation(len(free))[:k]]
        third = next(c for c in ("red", "green", "blue") if c not in (base.door_colour, base.distractor_colour))
        keys = [(base.door_colour, base.key), (base.distractor_colour, base.distractor)]
        sw, va = [base.switch], [base.vase]
        for p in spots:
            kind = rng.choice(["key", "key", "switch", "vase"])
            if kind == "key":
                keys.append((str(rng.choice([third, "yellow", "purple"])), p))
            elif kind == "switch":
                sw.append(p)
            else:
                va.append(p)
        lay = Clutter(base.wall_x, base.door_y, base.door_colour, tuple(keys), tuple(sw), tuple(va), base.goal,
                      base.start)
        if shortest(lay) is not None:
            return lay


def clutter_layouts(n=100):
    if CLUTTER.exists():
        d = pickle.loads(CLUTTER.read_bytes())
        if len(d["test"]) >= n:
            return [Clutter(**x) for x in d["test"]], d["shortest_each"]
    rng = np.random.default_rng(49000)
    lays = [make_layout(rng) for _ in range(n)]
    sh = [shortest(lay) for lay in lays]
    CLUTTER.write_bytes(pickle.dumps({"test": [vars(x) for x in lays], "shortest_each": sh}))
    print("clutter layouts:", n, "shortest mean", round(float(np.mean(sh)), 2), flush=True)
    return lays, sh


# ---------------------------------------------------------------- acting

def act(W, lay, i, M):
    """One episode from the same memory (card 048's loop, with the world's own step and view)."""
    W.reset()
    pl = S7.Plan047(W)
    rng = np.random.default_rng(1000 + i)
    s = M.start_state(lay)
    V = M.see(lay, s)
    facts, st, _, _ = pl.observe(None, V)
    rand = wrong = 0
    end = False
    t0 = time.monotonic()
    for t in range(C.BUDGET):
        res = pl.choose(st)
        a = res.action if res is not None else int(rng.integers(6))
        rand += res is None
        pred = pl.step(st, a)
        s2, end = M.step(lay, s, a)
        V2 = M.see(lay, s2)
        facts, st, miss, _ = pl.observe(facts, V2, end, prefer=pred[1])
        wrong += bool(miss)
        W.learn_try(V, a, V2, end)
        pl.forget()
        s, V = s2, V2
        if end:
            break
    return {"success": bool(end), "steps": t + 1, "random_actions": rand, "predictions_wrong": wrong,
            "seconds": round(time.monotonic() - t0, 3)}


class _Mod:
    def __init__(self, **k):
        self.__dict__.update(k)


CHAIN = _Mod(start_state=C.start_state, step=C.step, see=C.see)
CLUT = _Mod(start_state=start_state, step=step, see=see)


def run(args, recall, view):
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    seed, n = int(get("--seed", "399")), int(get("--n", "30"))
    out = Path(get("--out", f"runs/049_{recall}_{seed}.json"))
    t0 = time.monotonic()
    if "--chain" in args:
        variant = get("--chain", "one_door")
        L = C.load_layouts()[variant]
        lays, sh, M, name = L["test"][:n], L["shortest_each"][:n], CHAIN, f"chain_{variant}"
    else:
        lays, sh = clutter_layouts(max(n, 100))
        lays, sh, M, name = lays[:n], sh[:n], CLUT, "clutter"
    W = C.setup(seed)
    setup_s = round(time.monotonic() - t0, 1)
    rep = {a: k.report.get("conditions") for a, k in W.kinds.items() if k.report.get("conditions")}
    eps = []
    for i, lay in enumerate(lays):
        e = act(W, lay, i, M)
        e["shortest"] = sh[i]
        eps.append(e)
        print(seed, name, i, e, flush=True)
    ok = [e for e in eps if e["success"]]
    res = {"note": "Card 049, tools/card049/worlds.py", "recall": recall, "view": view, "seed": seed, "test": name,
           "layouts": len(eps), "success": round(len(ok) / len(eps), 4),
           "steps_ratio_to_shortest": round(float(np.mean([e["steps"] / e["shortest"] for e in ok])), 4) if ok else None,
           "random_share": round(sum(e["random_actions"] for e in eps) / sum(e["steps"] for e in eps), 4),
           "seconds_per_layout": round(float(np.mean([e["seconds"] for e in eps])), 3), "setup_seconds": setup_s,
           "conditions": {str(k): v for k, v in rep.items()}, "episodes": eps}
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print({k: v for k, v in res.items() if k not in ("episodes", "conditions")}, flush=True)
