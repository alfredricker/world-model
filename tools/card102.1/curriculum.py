"""Card 102.1's crafting curriculum (Appendix B, a declared data policy, C3). It sets only where episodes start; the
agent still receives pixels and its own actions, and the evaluator's state is kept for scoring only.

Per episode, on card 098.5's world seeds (2000 + w; worlds 0-7 train, 8-9 test): a new world (Crafter's reset), by
day; wood, stone, coal, iron and sapling each drawn independently 0-4; with probability 0.8 a table and independently
0.5 a furnace on a free grass, sand or path place 1-3 steps from the agent (steps as the 3 x 3 square counts them),
uniformly over the places; then 64 uniformly random actions.

  bin/prun python tools/card102.1/curriculum.py      → runs/102.1/curr_w{0..9}.npz
"""
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ['health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone', 'coal', 'iron', 'diamond', 'wood_pickaxe',
         'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']   # as card 098.5's collect.py


def local_semantic(sem, pos):
    """The semantic class at each of the view's 7 x 9 world places (row r, column c), -1 outside the world (card
    098.5's collect.py; copied, since that folder's crafter.py would shadow the package)."""
    out = np.full((7, 9), -1, np.int16)
    for r in range(7):
        for c in range(9):
            x, y = pos[0] - 4 + c, pos[1] - 3 + r
            if 0 <= x < sem.shape[0] and 0 <= y < sem.shape[1]:
                out[r, c] = sem[x, y]
    return out

OUT = ROOT / "runs" / "102.1"
EPISODES, STEPS, SEED = 200, 64, 2000
MATERIALS = ["wood", "stone", "coal", "iron", "sapling"]


def place_near(env, rng, name, taken):
    p = np.array(env._player.pos)
    opts = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if max(abs(dx), abs(dy)) == 0:
                continue
            q = (int(p[0] + dx), int(p[1] + dy))
            mat, obj = env._world[q]
            if mat in ("grass", "sand", "path") and obj is None and q not in taken:
                opts.append(q)
    if not opts:
        return None
    q = opts[rng.integers(len(opts))]
    env._world[q] = name
    taken.add(q)
    return (q[0] - int(p[0]), q[1] - int(p[1]))


def job(w):
    import crafter
    t0 = time.time()
    env = crafter.Env(seed=SEED + w)
    rng = np.random.default_rng(10000 + w)
    n = EPISODES * STEPS
    O = np.zeros((n, 64, 64, 3), np.uint8)
    A = np.zeros(n, np.int8)
    S = np.zeros((n, 7, 9), np.int16)
    INV = np.zeros((n, 16), np.int8)
    L = np.zeros(n, np.float32)
    FACE, POS = np.zeros((n, 2), np.int8), np.zeros((n, 2), np.int16)
    SLEEP, DONE = np.zeros(n, bool), np.zeros(n, bool)
    EP = np.zeros(n, np.int32)
    TABLE, FURN = np.full((n, 2), 99, np.int8), np.full((n, 2), 99, np.int8)
    t = 0
    for e in range(EPISODES):
        env.reset()
        p = env._player
        for m in MATERIALS:
            p.inventory[m] = int(rng.integers(0, 5))
        taken = set()
        tab = place_near(env, rng, "table", taken) if rng.random() < 0.8 else None
        fur = place_near(env, rng, "furnace", taken) if rng.random() < 0.5 else None
        obs = env.render()
        for s in range(STEPS):
            O[t], L[t], FACE[t], POS[t], SLEEP[t] = obs, env._world.daylight, p.facing, p.pos, p.sleeping
            S[t] = local_semantic(env._sem_view(), p.pos)
            INV[t] = [p.inventory[k] for k in ITEMS]
            EP[t] = w * 100000 + e
            if tab is not None:
                TABLE[t] = tab
            if fur is not None:
                FURN[t] = fur
            a = int(rng.integers(17))
            A[t] = a
            obs, _, done, _ = env.step(a)
            DONE[t] = done or s == STEPS - 1
            t += 1
            if done:
                break
    O, A, S, INV, L, FACE, POS, SLEEP, DONE, EP, TABLE, FURN = (x[:t] for x in
                                                               (O, A, S, INV, L, FACE, POS, SLEEP, DONE, EP, TABLE, FURN))
    np.savez_compressed(OUT / f"curr_w{w}.npz", obs=O, act=A, sem=S, inv=INV, light=L, face=FACE, pos=POS,
                        sleep=SLEEP, done=DONE, ep=EP, table_start=TABLE, furnace_start=FURN)
    return w, t, round(time.time() - t0, 1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with get_context("fork").Pool(10) as pool:
        for r in pool.imap_unordered(job, range(10)):
            print("worker, steps, seconds", r, flush=True)
