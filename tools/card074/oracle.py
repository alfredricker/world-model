"""Card 074's feasibility gate for tier 2: a breadth-first oracle on the simulator's state of MiniGrid's
BlockedUnlockPickup layouts (the tier's 100 test seeds), as card 073's oracle did for the decoy world.

State: position, direction, held object, every pickable object's place, the door (locked, closed, open). Rules as
MiniGrid's: forward onto empty floor or an open door; pick up a key, ball or box in front with an empty hand (picking
up the mission's box ends the episode); drop onto an empty tile in front; toggle a locked door with its own key held
(it opens), a closed door (it opens), an open door (it closes). Reported: layouts solvable within the step limit, and
the shortest lengths.

  bin/prun python tools/card074/oracle.py --out runs/074/oracle_tier2.json
"""
import json
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid                                        # noqa: F401
import numpy as np

SEED_TEST = 1_000_000
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
PICKABLE = ("key", "ball", "box")


def layout(env):
    u = env.unwrapped
    walls, objs, door = set(), [], None
    for x in range(u.width):
        for y in range(u.height):
            o = u.grid.get(x, y)
            if o is None:
                continue
            if o.type == "wall":
                walls.add((x, y))
            elif o.type in PICKABLE:
                objs.append(((x, y), o.type, o.color))
            elif o.type == "door":
                door = ((x, y), o.color, 2 if o.is_open else (0 if o.is_locked else 1))
    target = next(i for i, (_, t, c) in enumerate(objs) if t == u.obj.type and c == u.obj.color)
    return walls, objs, door, target, (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir)), u.max_steps


def shortest(walls, objs, door, target, start, limit):
    dpos, dcol, ds0 = door
    s0 = (start[0], start[1], start[2], -1, tuple(p for p, _, _ in objs), ds0)
    seen = {s0: 0}
    q = deque([s0])
    while q:
        s = q.popleft()
        n = seen[s]
        if n >= limit:
            continue
        x, y, d, c, pos, ds = s
        front = (x + DIRS[d][0], y + DIRS[d][1])
        at = next((i for i, p in enumerate(pos) if p == front), None)
        nxt = [(x, y, (d - 1) % 4, c, pos, ds), (x, y, (d + 1) % 4, c, pos, ds)]
        if front not in walls and at is None and (front != dpos or ds == 2):
            nxt.append((front[0], front[1], d, c, pos, ds))
        if at is not None and c == -1:                 # pick up
            if at == target:
                return n + 1
            p2 = list(pos)
            p2[at] = None
            nxt.append((x, y, d, at, tuple(p2), ds))
        if c >= 0 and front not in walls and at is None and front != dpos:   # drop
            p2 = list(pos)
            p2[c] = front
            nxt.append((x, y, d, -1, tuple(p2), ds))
        if front == dpos:                              # toggle
            if ds == 0 and c >= 0 and objs[c][1] == "key" and objs[c][2] == dcol:
                nxt.append((x, y, d, c, pos, 2))
            elif ds == 1:
                nxt.append((x, y, d, c, pos, 2))
            elif ds == 2:
                nxt.append((x, y, d, c, pos, 1))
        for t in nxt:
            if t not in seen:
                seen[t] = n + 1
                q.append(t)
    return None


def main():
    out = Path(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == "--out"),
                    "runs/074/oracle_tier2.json"))
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    lens, unsolved = [], []
    for seed in SEED_TEST + 1000 * 2 + np.arange(100):
        env.reset(seed=int(seed))
        n = shortest(*layout(env))
        (unsolved.append(int(seed)) if n is None else lens.append(n))
    res = {"note": "Card 074 gate: breadth-first oracle on tier 2's test layouts (BlockedUnlockPickup)",
           "solvable": len(lens), "unsolvable_seeds": unsolved, "shortest_mean": round(float(np.mean(lens)), 2),
           "shortest_max": int(max(lens))}
    print(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
