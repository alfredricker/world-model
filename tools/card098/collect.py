"""Card 098: random play in Crafter (uniform over the 17 actions), the image per step and, for the evaluator only,
the semantic map around the player, the inventory and the daylight.

  bin/prun python tools/card098/collect.py --steps 100000      → runs/098/play.npz
"""
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "098"
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
STEPS = int(arg("--steps", "100000"))
WORKERS = 10
ITEMS = ['health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone', 'coal', 'iron', 'diamond', 'wood_pickaxe',
         'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']


def local_semantic(sem, pos):
    """The semantic class at each of the view's 7 x 9 world tiles (row r, column c), -1 outside the world."""
    out = np.full((7, 9), -1, np.int16)
    for r in range(7):
        for c in range(9):
            x, y = pos[0] - 4 + c, pos[1] - 3 + r
            if 0 <= x < sem.shape[0] and 0 <= y < sem.shape[1]:
                out[r, c] = sem[x, y]
    return out


def job(w):
    import crafter
    n = STEPS // WORKERS
    env = crafter.Env(seed=1000 + w)
    rng = np.random.default_rng(w)
    obs = env.reset()
    O = np.zeros((n, 64, 64, 3), np.uint8)
    A = np.zeros(n, np.int8)
    S = np.zeros((n, 7, 9), np.int16)
    INV = np.zeros((n, 16), np.int8)
    L = np.zeros(n, np.float32)
    FACE = np.zeros((n, 2), np.int8)
    EP = np.zeros(n, np.int32)
    DONE = np.zeros(n, bool)
    ep = 0
    info = None
    for t in range(n):
        O[t] = obs                                      # the frame the action is taken in
        L[t] = env._world.daylight
        FACE[t] = env._player.facing
        if info is not None:
            S[t] = local_semantic(info["semantic"], info["player_pos"])
            INV[t] = [info["inventory"][k] for k in ITEMS]
        else:
            S[t] = local_semantic(env._sem_view(), env._player.pos)
            INV[t] = [env._player.inventory[k] for k in ITEMS]
        EP[t] = w * 100000 + ep
        a = int(rng.integers(17))
        A[t] = a
        obs, _, done, info = env.step(a)
        DONE[t] = done
        if done:
            obs, info, ep = env.reset(), None, ep + 1
    return O, A, S, INV, L, FACE, EP, DONE


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with get_context("fork").Pool(WORKERS) as pool:
        parts = pool.map(job, range(WORKERS))
    O, A, S, INV, L, FACE, EP, DONE = (np.concatenate(x) for x in zip(*parts))
    np.savez(OUT / "play.npz", obs=O, act=A, sem=S, inv=INV, light=L, face=FACE, ep=EP, done=DONE)
    print("steps", len(A), "episodes", len(np.unique(EP)), "night share (light < 0.3)", round(float((L < 0.3).mean()), 3),
          "classes", np.unique(S).tolist(), flush=True)
