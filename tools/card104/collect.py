"""Card 104's play: random play in Crafter on seeds 2010-2012 (workers 10-12), rendered at 64 px (Crafter's default,
7 px per tile) and at 144 px (16 px per tile, the sprites' own resolution) in one process. Crafter is not reproducible
across processes (despawning iterates a set ordered by memory address), so both renders must come from the same run.

The 64 px play is stored in card 098.5's format beside its own (runs/098.5/play_w<w>.npz, without 098.4's
counterfactual copies), so that 103.4's placing and 103.5's files read it unchanged. The 16 px view (112 x 144, the
9 x 7 world rows only) goes to runs/104/hi_w<w>.npy. Every object draws from its own random stream, as in 098.5. The 16 px
render's night noise draws from its own stream, so the world's stream is consumed exactly as by the 64 px observation.

  bin/prun python tools/card104/collect.py [--steps 30000] [--workers 10,11,12]
"""
import importlib.util
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
STEPS = int(arg("--steps", "30000"))
WORKERS = [int(x) for x in arg("--workers", "10,11,12").split(",")]
spec = importlib.util.spec_from_file_location("collect0985", ROOT / "tools" / "card098.5" / "collect.py")
C = importlib.util.module_from_spec(spec)
sys.argv, argv = [sys.argv[0]], sys.argv                # 098.5's module parses its own --steps at import
spec.loader.exec_module(C)
sys.argv = argv


def job(w):
    import crafter
    C.own_streams()
    t0 = time.time()
    env = crafter.Env(seed=C.SEED + w)
    rng = np.random.default_rng(C.SEED + w)
    hi_rng = np.random.RandomState(C.SEED + w + 7919)
    env._world._episode_base = 1000 * w + 1
    obs = env.reset()
    n = STEPS
    O = np.zeros((n, 64, 64, 3), np.uint8)
    HI = np.lib.format.open_memmap(ROOT / "runs" / "104" / f"hi_w{w}.npy", "w+", np.uint8, (n, 112, 144, 3))
    A = np.zeros(n, np.int8)
    S = np.zeros((n, 7, 9), np.int16)
    INV = np.zeros((n, 16), np.int8)
    L = np.zeros(n, np.float32)
    FACE, POS = np.zeros((n, 2), np.int8), np.zeros((n, 2), np.int16)
    SLEEP, DONE = np.zeros(n, bool), np.zeros(n, bool)
    EP = np.zeros(n, np.int32)
    OBJ = []
    ep = 0
    for t in range(n):
        p = env._player
        O[t], L[t], FACE[t], POS[t], SLEEP[t] = obs, env._world.daylight, p.facing, p.pos, p.sleeping
        wr, env._world.random = env._world.random, hi_rng
        HI[t] = env.render((144, 144))[:112, :144]
        env._world.random = wr
        S[t] = C.local_semantic(env._sem_view(), p.pos)
        INV[t] = [p.inventory[k] for k in C.ITEMS]
        EP[t] = w * 100000 + ep
        OBJ += C.in_view(env, t)
        a = int(rng.integers(17))
        A[t] = a
        obs, _, done, _ = env.step(a)
        DONE[t] = done
        if done:
            ep += 1
            env._world._episode_base = 1000 * w + ep + 1
            obs = env.reset()
    HI.flush()
    del HI
    OBJ = np.array(OBJ, np.int32).reshape(-1, 7)
    np.savez(ROOT / "runs" / "098.5" / f"play_w{w}.npz", obs=O, act=A, sem=S, inv=INV, light=L, face=FACE, pos=POS,
             sleep=SLEEP, ep=EP, done=DONE, obj=OBJ)
    return w, n, ep + 1, round(time.time() - t0, 1)


if __name__ == "__main__":
    (ROOT / "runs" / "104").mkdir(parents=True, exist_ok=True)
    with get_context("fork").Pool(len(WORKERS)) as pool:
        for r in pool.imap_unordered(job, WORKERS):
            print("worker, steps, episodes, seconds", r, flush=True)
