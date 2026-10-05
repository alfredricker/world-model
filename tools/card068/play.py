"""Card 068: play starts in the random play that builds memory.

Card 066's memory is random play from the environment's own starts. In tier 2 it walked onto an open door 0 to 8
times per door colour among the stored rows, too few for recall to know that an open door of every colour can be
walked through; a locked door whose open state is not known to be walkable is not "openable", and the agent never
makes it a condition. CHARTER's declared curriculum for the chained rooms (2026-09-24) did this job there: half of
the collected episodes are play starts, in any room, some doors open, half holding a locked door's key. The same here:

  play start (half the episodes, at random):  every door open with probability 1/2 (unlocked); with probability
             1/2 the agent holds the key of a door still locked, taken from where it lies (the floor or a box);
             the agent stands on an empty place anywhere in the map, facing any way.

Then uniform random actions, as card 066. Everything else is card 066's (sampling, believed views, the agent).

  bin/prun python tools/card068/play.py --collect --tier 2                    writes runs/068/memory_tier2.npz
  bin/prun python tools/card068/play.py --tier 2 --n 100 --cap 300 --walking 067 --out runs/068/tier2.json
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card066"))
import tiers as T                                      # noqa: E402

PLAY = 0.5
T.OUT = T.ROOT / "runs" / "068"


def play_start(env, rng):
    u = env.unwrapped
    g = u.grid
    cells = [(i, j) for j in range(g.height) for i in range(g.width)]
    doors = [o for i, j in cells if (o := g.get(i, j)) is not None and o.type == "door"]
    for d in doors:
        if rng.random() < 0.5:
            d.is_locked, d.is_open = False, True
    held = None
    locked = [d for d in doors if d.is_locked]
    if locked and rng.random() < 0.5:
        d = locked[int(rng.integers(len(locked)))]
        for i, j in cells:
            o = g.get(i, j)
            if o is not None and o.type == "key" and o.color == d.color:
                g.set(i, j, None)
                held = o
                break
            if o is not None and o.type == "box" and o.contains is not None and o.contains.type == "key" \
                    and o.contains.color == d.color:
                held, o.contains = o.contains, None
                break
        if held is not None:
            held.cur_pos = np.array([-1, -1])
            u.carrying = held
    empty = [(i, j) for i, j in cells if g.get(i, j) is None]
    i, j = empty[int(rng.integers(len(empty)))]
    u.agent_pos = np.array([i, j])
    u.agent_dir = int(rng.integers(4))
    return held is not None


def _play(job):
    """Card 066's random play, with play starts; also counts forward steps onto an open door, per colour."""
    tier, seeds, max_steps = job
    rng = np.random.default_rng(seeds[0] + 7)
    env = T.make(tier)
    out = {k: [] for k in ("ego0", "ego1", "act", "w", "term1", "pres", "stale")}
    stats = Counter()
    open_door = {T.KR.code(("door", c, 2)): c for c in T.HUES}
    for sd in seeds:
        env.reset(seed=int(sd))
        if rng.random() < PLAY:
            stats["play_starts"] += 1
            stats["play_starts_holding_a_key"] += play_start(env, rng)
        Gs, xs, ys, ds, A, term = [T.grid_codes(env)], [env.agent_pos[0]], [env.agent_pos[1]], [env.agent_dir], [], []
        for t in range(max_steps):
            a = int(rng.integers(6))
            _, rew, te, tr, _ = env.step(a)
            A.append(a)
            term.append(bool(te))
            Gs.append(T.grid_codes(env))
            xs.append(env.agent_pos[0]), ys.append(env.agent_pos[1]), ds.append(env.agent_dir)
            if te or tr:
                break
        G = np.stack(Gs)
        E = T.ego(G, xs, ys, ds)
        A, term = np.array(A, np.int64), np.array(term)
        Tn = len(A)
        changed = (G[1:] // 5 != G[:-1] // 5).reshape(Tn, -1).any(1)
        forced = changed | term
        i = np.flatnonzero(forced | (rng.random(Tn) < T.P_UNIFORM))
        r = i[np.isin(A[i], T.INTER)]
        pres, stale = T.BL.believe(E, A, r)
        out["ego0"].append(E[i].astype(np.uint8)), out["ego1"].append(E[i + 1].astype(np.uint8))
        out["act"].append(A[i]), out["term1"].append(term[i])
        out["w"].append(np.where(forced[i], 1.0, 1 / T.P_UNIFORM))
        out["pres"].append(pres), out["stale"].append(stale)
        stats["episodes"] += 1
        stats["steps"] += Tn
        stats["successes"] += int(term.any())
        moved = (np.diff(np.array(xs)) != 0) | (np.diff(np.array(ys)) != 0)
        for t in np.flatnonzero((A == T.VP.FWD) & moved):
            c = open_door.get(int(E[t, T.FRONT]) // 5 * 5)
            if c is not None:
                stats[f"walked_onto_open_door_{c}"] += 1
                stats[f"stored_onto_open_door_{c}"] += int(t in set(i.tolist()))
        for t in np.flatnonzero(changed & (A == T.VP.TOG)):
            d = (G[t + 1] != G[t]) & (G[t + 1] // 5 != G[t] // 5)
            for c in G[t + 1][d].tolist():
                o = T.KR.OBJECTS[c // 5]
                if isinstance(o, tuple) and o[0] == "door" and o[2] == 2:
                    stats["doors_opened"] += 1
    return {k: np.concatenate(v) for k, v in out.items()}, stats


T._play = _play

if __name__ == "__main__":
    T.main()
