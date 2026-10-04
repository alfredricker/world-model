"""Card 051: a step-by-step trace of the planner's chain (evaluator names), for diagnosis.

  bin/prun python tools/card051/trace.py --world two_doors --seed 399 --layouts 0,1 --steps 60
  bin/prun python tools/card051/trace.py --world clutter --seed 403 --layouts 66 --steps 40
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1 as S1                                     # noqa: E402

C, S7, VP = S1.C, S1.S7, S1.VP
ANAME = ("left", "right", "forward", "pickup", "drop", "toggle")


def fmt(W, x):
    nm = lambda h: W.judge.name(int(h))
    if not isinstance(x, tuple) or not x:
        return str(x)
    k = x[0]
    if k == "act":
        return f"act {ANAME[x[1]]}"
    if k == "face":
        return f"face({ANAME[x[1]]} {nm(x[2])} @{x[3]})"
    if k == "part":
        return f"part({'held' if x[1] == VP.HELDP else 'view'} {ANAME[x[2]]} {nm(x[3])} @{x[4]} c{x[5]})"
    if k == "walk":
        return f"walk({x[1]})"
    if k == "tile":
        return f"tile(@{x[1]} -> {nm(x[2])})"
    return str(x)


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    world, seed = get("--world", "two_doors"), int(get("--seed", "399"))
    idx = [int(x) for x in get("--layouts", "0").split(",")]
    steps = int(get("--steps", "60"))
    W = S1.setup(get("--recall", "index"), seed)
    if "--debug" in args:
        sys.modules["threats"].DEBUG = True
    if world == "clutter":
        lays, _ = S1.WD.clutter_layouts(100)
        M = S1.WD.CLUT
    else:
        lays = C.load_layouts()[world]["test"]
        M = S1.WD.CHAIN
    for i in idx:
        lay = lays[i]
        print("layout", i, lay, flush=True)
        W.reset()
        pl = S7.Plan047(W)
        rng = np.random.default_rng(1000 + i)
        s = M.start_state(lay)
        V = M.see(lay, s)
        facts, st, _, _ = pl.observe(None, V)
        for t in range(steps):
            res = pl.choose(st)
            a = res.action if res is not None else int(rng.integers(6))
            u, h = pl.front_held(st)
            chain = " <- ".join(fmt(W, x) for x in res.trace) if res is not None else "RANDOM"
            print(f"t{t:3d} s={s[:4]} front={W.judge.name(int(u))} held={W.judge.name(int(h))} "
                  f"a={ANAME[a]} | {chain}", flush=True)
            pred = pl.step(st, a)
            s2, end = M.step(lay, s, a)
            V2 = M.see(lay, s2)
            facts, st, miss, _ = pl.observe(facts, V2, end, prefer=pred[1])
            W.learn_try(V, a, V2, end)
            pl.forget()
            s, V = s2, V2
            if end:
                print("reached the goal at step", t + 1)
                break


if __name__ == "__main__":
    main()
