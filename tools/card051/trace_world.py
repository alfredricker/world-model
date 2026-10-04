"""Card 051: a step-by-step trace in the familiar worlds (key, switch, either, both), for diagnosis.

  bin/prun python tools/card051/trace_world.py --recall walk2 --world both --seed 403 --layouts 0-29 --steps 200
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1 as S1                                     # noqa: E402
import trace as TR                                     # noqa: E402

C, S7, VP = S1.C, S1.S7, S1.VP


def setup(recall, seed, world):
    S1.setup(recall, seed)                             # installs the agent (and builds the key world once)
    VP.T.configure()
    cache = pickle.loads(VP.T.MEMORY.read_bytes())
    groups = VP.T.use_groups(cache["mem"], "cuda")
    z, parts, _ = VP.vectors_of("A", seed, cache["tiles"], cache["pairs"], groups, "cuda", lambda *a: None)
    d = pickle.loads(VP.T.DATA.read_bytes())["data"][world]
    S = VP.Store(z)
    lut = np.zeros(256, np.uint8)
    lut[:len(S.hof)] = S.hof
    C.Kd.APP = lut
    W = VP.World(S, parts, d, "cuda", lambda *a: None)
    VP.WORLD, C.F.M = W, W.M
    return W, d["test"]


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    a, b = get("--layouts", "0-29").split("-")
    steps = int(get("--steps", "200"))
    W, test = setup(get("--recall", "walk2"), int(get("--seed", "403")), get("--world", "both"))
    ld = VP.ld
    for i in range(int(a), int(b) + 1):
        lay = test[i]
        W.reset()
        pl = S7.Plan047(W)
        rng = np.random.default_rng(1000 + i)
        s = ld.start_state(lay)
        V = VP.see(lay, s)
        facts, st, _, _ = pl.observe(None, V)
        lines, end = [], False
        for t in range(steps):
            res = pl.choose(st)
            act = res.action if res is not None else int(rng.integers(6))
            u, h = pl.front_held(st)
            chain = " <- ".join(TR.fmt(W, x) for x in res.trace) if res is not None else "RANDOM"
            lines.append(f"t{t:3d} s={s[:4]} front={W.judge.name(int(u))} held={W.judge.name(int(h))} "
                         f"a={TR.ANAME[act]} | {chain}")
            pred = pl.step(st, act)
            s2, end = ld.step(lay, s, act)
            V2 = VP.see(lay, s2)
            facts, st, _, _ = pl.observe(facts, V2, end, prefer=pred[1])
            W.learn_try(V, act, V2, end)
            pl.forget()
            s, V = s2, V2
            if end:
                break
        print(f"layout {i}: {'goal at step ' + str(t + 1) if end else 'FAILED'}", flush=True)
        if not end and "--show" in args:
            print(lay)
            print("\n".join(lines[:int(get("--show", "40"))]))


if __name__ == "__main__":
    main()
