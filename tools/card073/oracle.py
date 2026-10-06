"""Card 073's feasibility gate: a breadth-first oracle on the simulator's state for the decoy world's test layouts.

For each fold (the door's hue) and each of card 072's 100 test seeds, the layout is generated as the runs generate
it (tools/card069/relation.py's DoorKeyDecoy, reset with the seed), then searched over (position, direction, held
key, both keys' places, door locked/closed/open) with MiniGrid's rules: forward onto empty floor, the goal or an
open door; pick up a key in front with an empty hand; drop onto an empty tile in front; toggle a locked door with
its own key held (it opens), a closed door (it opens), an open door (it closes). Reported: layouts solvable within
the episode's step limit, and the shortest lengths.

  bin/prun python tools/card073/oracle.py --out runs/073/oracle.json
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import gymnasium as gym                                # noqa: E402
import minigrid                                        # noqa: E402,F401
import relation as R                                   # noqa: E402

HUES = ("red", "green", "blue", "purple", "yellow", "grey")
SEED_TEST = 1_000_000
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def layout(env):
    u = env.unwrapped
    walls, keys, door, goal = set(), [], None, None
    for x in range(u.width):
        for y in range(u.height):
            o = u.grid.get(x, y)
            if o is None:
                continue
            if o.type == "wall":
                walls.add((x, y))
            elif o.type == "key":
                keys.append(((x, y), o.color))
            elif o.type == "door":
                door = ((x, y), o.color)
            elif o.type == "goal":
                goal = (x, y)
    return walls, keys, door, goal, (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir)), u.max_steps


def shortest(walls, keys, door, goal, start, limit):
    kcol = [c for _, c in keys]
    dpos, dcol = door
    s0 = (start[0], start[1], start[2], -1, keys[0][0], keys[1][0], 0)
    seen = {s0: 0}
    q = deque([s0])
    while q:
        s = q.popleft()
        n = seen[s]
        if n >= limit:
            continue
        x, y, d, c, k0, k1, ds = s
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        front = (fx, fy)
        kp = [k0, k1]
        key_at = next((i for i in (0, 1) if kp[i] == front), None)
        nxt = [(x, y, (d - 1) % 4, c, k0, k1, ds), (x, y, (d + 1) % 4, c, k0, k1, ds)]
        if front == goal:
            return n + 1
        free = front not in walls and key_at is None and (front != dpos or ds == 2)
        if free:
            nxt.append((fx, fy, d, c, k0, k1, ds))
        if key_at is not None and c == -1:             # pick up
            kp2 = list(kp)
            kp2[key_at] = None
            nxt.append((x, y, d, key_at, kp2[0], kp2[1], ds))
        if c >= 0 and front not in walls and key_at is None and front != dpos and front != goal:   # drop
            kp2 = list(kp)
            kp2[c] = front
            nxt.append((x, y, d, -1, kp2[0], kp2[1], ds))
        if front == dpos:                              # toggle
            if ds == 0 and c >= 0 and kcol[c] == dcol:
                nxt.append((x, y, d, c, k0, k1, 2))
            elif ds == 1:
                nxt.append((x, y, d, c, k0, k1, 2))
            elif ds == 2:
                nxt.append((x, y, d, c, k0, k1, 1))
        for t in nxt:
            if t not in seen:
                seen[t] = n + 1
                q.append(t)
    return None


def main():
    out = Path(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == "--out"), "runs/073/oracle.json"))
    env_id = R.register()
    res = {"note": "Card 073 gate: breadth-first oracle on the decoy world's test layouts", "folds": {}}
    for hue in HUES:
        R.DECOY.update(door=[hue], decoy=[c for c in HUES if c != hue])
        env = gym.make(env_id)
        lens, unsolved = [], []
        for seed in SEED_TEST + 1000 + np.arange(100):
            env.reset(seed=int(seed))
            walls, keys, door, goal, start, limit = layout(env)
            assert door[1] == hue and len(keys) == 2
            n = shortest(walls, keys, door, goal, start, limit)
            (unsolved.append(int(seed)) if n is None else lens.append(n))
        res["folds"][hue] = {"solvable": len(lens), "unsolvable_seeds": unsolved,
                             "shortest_mean": round(float(np.mean(lens)), 2), "shortest_max": int(max(lens))}
        print(hue, res["folds"][hue], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
