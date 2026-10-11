"""Card 098.5 (and 098.4's labels): random play in Crafter, uniform over the 17 actions, on seeds 2000-2009 (worlds
card 098.3's encoder never saw). Kept per step: the frame and the action; for the evaluator only, the semantic view,
the inventory, the light, the agent's facing, sleeping flag and world position, and every object in view with its
persistent identity (its slot in the world's object list, never reused within an episode), kind, offset from the
agent and facing (arrows).

Card 098.4's labels: before each step, a copy of the simulator steps with noop instead; its semantic view, inventory,
sleeping flag and objects in view are kept. Every object, the player included, draws from its own random stream
(seeded by episode and its slot), so an object's draws change only when what it does changes (a zombie following the
player's new position), not because another object drew more or fewer numbers; this changes no rule of the game.
Without it, 40% of move steps diverged (which creatures update depends on the player's position). The world's own
stream is then used only for spawning; DIVERGE marks steps where it differs between the copies after the step.

  bin/prun python tools/card098.5/collect.py [--steps 30000]      → runs/098.5/play_w{0..9}.npz
"""
import copy
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "098.5"
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
STEPS = int(arg("--steps", "30000"))
WORKERS = 10
SEED = 2000
ITEMS = ['health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone', 'coal', 'iron', 'diamond', 'wood_pickaxe',
         'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
KINDS = ["Player", "Cow", "Zombie", "Skeleton", "Arrow", "Plant"]   # semantic ids 13..18


def local_semantic(sem, pos):
    """The semantic class at each of the view's 7 x 9 world places (row r, column c), -1 outside the world."""
    out = np.full((7, 9), -1, np.int16)
    for r in range(7):
        for c in range(9):
            x, y = pos[0] - 4 + c, pos[1] - 3 + r
            if 0 <= x < sem.shape[0] and 0 <= y < sem.shape[1]:
                out[r, c] = sem[x, y]
    return out


def in_view(env, t):
    """Rows (t, identity, kind, dx, dy, fx, fy) for every object in the 9 x 7 view but the player."""
    px, py = env._player.pos
    rows = []
    for i, obj in enumerate(env._world._objects):
        if obj is None or obj is env._player:
            continue
        dx, dy = int(obj.pos[0] - px), int(obj.pos[1] - py)
        if abs(dx) <= 4 and abs(dy) <= 3:
            f = getattr(obj, "facing", (0, 0))
            rows.append((t, i, KINDS.index(type(obj).__name__), dx, dy, int(f[0]), int(f[1])))
    return rows


def own_streams():
    """Every object added to a world gets its own random stream, seeded by the episode and its slot."""
    import crafter.engine as E
    add0 = E.World.add

    def add(self, obj):
        add0(self, obj)
        obj.random = np.random.RandomState((getattr(self, "_episode_base", 0) * 1000003 + len(self._objects)) % 2 ** 32)

    E.World.add = add


def job(w):
    import crafter
    own_streams()
    t0 = time.time()
    env = crafter.Env(seed=SEED + w)
    rng = np.random.default_rng(SEED + w)
    env._world._episode_base = 1000 * w + 1
    obs = env.reset()
    n = STEPS
    O = np.zeros((n, 64, 64, 3), np.uint8)
    A = np.zeros(n, np.int8)
    S, CS = np.zeros((n, 7, 9), np.int16), np.zeros((n, 7, 9), np.int16)
    INV, CINV = np.zeros((n, 16), np.int8), np.zeros((n, 16), np.int8)
    L = np.zeros(n, np.float32)
    FACE, POS = np.zeros((n, 2), np.int8), np.zeros((n, 2), np.int16)
    SLEEP, CSLEEP = np.zeros(n, bool), np.zeros(n, bool)
    EP = np.zeros(n, np.int32)
    DONE, DIVERGE = np.zeros(n, bool), np.zeros(n, bool)
    OBJ, COBJ = [], []
    ep = 0
    for t in range(n):
        p = env._player
        O[t], L[t], FACE[t], POS[t], SLEEP[t] = obs, env._world.daylight, p.facing, p.pos, p.sleeping
        S[t] = local_semantic(env._sem_view(), p.pos)
        INV[t] = [p.inventory[k] for k in ITEMS]
        EP[t] = w * 100000 + ep
        OBJ += in_view(env, t)
        a = int(rng.integers(17))
        A[t] = a
        c = copy.deepcopy(env)                          # the same world, doing nothing instead
        c.step(0)
        CS[t] = local_semantic(c._sem_view(), c._player.pos)
        CINV[t] = [c._player.inventory[k] for k in ITEMS]
        CSLEEP[t] = c._player.sleeping
        COBJ += in_view(c, t)
        obs, _, done, _ = env.step(a)
        s0, s1 = env._world.random.get_state(), c._world.random.get_state()
        DIVERGE[t] = not (s0[2] == s1[2] and np.array_equal(s0[1], s1[1]))
        DONE[t] = done
        if done:
            ep += 1
            env._world._episode_base = 1000 * w + ep + 1
            obs = env.reset()
    OBJ, COBJ = np.array(OBJ, np.int32).reshape(-1, 7), np.array(COBJ, np.int32).reshape(-1, 7)
    np.savez(OUT / f"play_w{w}.npz", obs=O, act=A, sem=S, inv=INV, light=L, face=FACE, pos=POS, sleep=SLEEP, ep=EP,
             done=DONE, obj=OBJ, cf_sem=CS, cf_inv=CINV, cf_sleep=CSLEEP, cf_obj=COBJ, diverge=DIVERGE)
    return w, n, ep + 1, int(DIVERGE.sum()), round(time.time() - t0, 1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with get_context("fork").Pool(WORKERS) as pool:
        for r in pool.imap_unordered(job, range(WORKERS)):
            print("worker, steps, episodes, diverged, seconds", r, flush=True)
