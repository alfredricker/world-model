"""Card 095.1: the state before and after every stored pick up, toggle and drop, per world.

Replays memory 066's play exactly (card 094.1's pkeys.py: the same seeds and sampling, the replay checked against
the stored memory) and keeps, per interaction try: the action, its weight, its episode, the believed codes per place
(card 062's belief, places relative to the agent), and the true window's codes before and after (169 places + the
hand). Also the world's token vectors (S.arr), the code → token map (APP) and, per token, whether card 094 attends to
it, so the arms need no agent setup.

  <version 20's flags> bin/prun python tools/card095.1/states.py tier2     → runs/095.1/states_tier2.npz
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card094.1"))
import pkeys as PK                                     # noqa: E402  (sets up the harness for sys.argv[1]'s world)

T, BL, FILES = PK.T, PK.BL, PK.FILES
OUT = ROOT / "runs" / "095.1"
EP = []                                                # per believe() call (one per episode): interaction tries
_bp = PK.believe_pos


def believe_ep(ego, act, rows):
    EP.append(len(rows))
    return _bp(ego, act, rows)


def job(j):
    PK.SIDE_ROWS.clear()
    EP.clear()
    D, stats = T._play(j)
    return D, np.concatenate(PK.SIDE_ROWS), np.array(EP, np.int64), stats


def replay(tier):
    BL.moves_from_card061()
    BL.believe = believe_ep
    env = T.make(tier)
    n_ep = T.MEMORY_STEPS // env.max_steps
    seeds = T.SEED_MEMORY + 100_000 * tier + np.arange(n_ep)
    jobs = [(tier, s.tolist(), env.max_steps) for s in np.array_split(seeds, 80)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(20) as pool:
        parts = pool.map(job, jobs)
    D = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    POS = np.concatenate([p[1] for p in parts])
    per_ep = np.concatenate([p[2] for p in parts])
    return D, POS, np.repeat(np.arange(len(per_ep)), per_ep)


if __name__ == "__main__":
    t0 = time.monotonic()
    W, info = T.setup(PK.TIER, lambda m: None)
    z = np.load(T.OUT / f"memory_tier{PK.TIER}.npz")
    D, POS, ep = replay(PK.TIER)
    same = {k: bool(np.array_equal(D[k], z[k])) for k in ("act", "ego0", "pres")}
    print("replay equals stored memory:", same, flush=True)
    assert all(same.values()), same
    inter = np.flatnonzero(np.isin(D["act"], T.INTER))
    assert len(inter) == len(POS) == len(ep)
    arr = W.S.arr.astype(np.float32).copy()
    APP = np.asarray(T.Kd.APP, np.int64)
    att = np.array([bool(FILES.static(W, h)) for h in range(len(arr))])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"states_{PK.WORLD}.npz", act=D["act"][inter], w=D["w"][inter], ep=ep, pos=POS,
                        e0=D["ego0"][inter][:, :170], e1=D["ego1"][inter][:, :170], pres=D["pres"],
                        arr=arr, app=APP, attended=att, wh=BL.WH, front=T.FRONT, held=T.HELD,
                        centre=BL.CENTRE)
    print(PK.WORLD, "tries", len(inter), "episodes", int(ep.max()) + 1, "tokens", len(arr), "attended", int(att.sum()),
          "seconds", round(time.monotonic() - t0, 1), flush=True)
