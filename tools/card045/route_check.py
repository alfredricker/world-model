"""Card 045's gate: does the route predict whether the approach gets through?

First, Q's held-out check: Q trained with 5% and with 20% of the target placements left out of the loss, then its V on
those against the evaluator's open-space steps.

Then, for each world (seed 399's encoder, arm A), at the start of the first --layouts test layouts, pairs of placements
(c, p) the agent could stand at are drawn. The planner predicts the approach from c to p clear when its route
(the tokens forward steps onto) is all walkable. The evaluator then sets the simulator at c and takes the
approach's moves (System 1's first move, then the next, by the learned transformations): it got through when the
simulator stands at p after V(c, p) moves. Also: whether the moves taken were V(c, p).

  bin/prun python tools/card045/route_check.py --layouts 30 --pairs 3000 --out runs/045_route_check.json
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import movement as MV                                  # noqa: E402

VP, F, CS, ld = MV.VP, MV.F, MV.CS, MV.ld


def world_model(world, seed, data):
    import slot_planner as SP_
    SP_.install()
    VP.Kind, VP.vectors_of = MV.CR.kind, MV.CR.vectors_of
    cache = pickle.loads(VP.T.MEMORY.read_bytes())
    groups = VP.T.use_groups(cache["mem"], "cuda")
    z, parts, _ = VP.vectors_of("A", seed, cache["tiles"], cache["pairs"], groups, "cuda", lambda *a: None)
    S = VP.Store(z)
    lut = np.zeros(256, np.uint8)
    lut[:len(S.hof)] = S.hof
    VP.Kd.APP = lut
    W = VP.World(S, parts, data[world], "cuda", lambda *a: None)
    VP.WORLD, F.M = W, W.M
    return W


def where(lay, M, p):
    """The simulator's (x, y, heading) of placement p."""
    c, a = MV.cell_of(lay, M.cidx[p]), MV.cell_of(lay, M.fidx[p])
    return (int(c[0]), int(c[1]), ld.DIR_VEC.index((int(a[0] - c[0]), int(a[1] - c[1]))))


def check(W, lay, n_pairs, rng):
    M = F.M
    pl = VP.VPlan(W)
    s0 = ld.start_state(lay)
    _, st, _, _ = pl.observe(None, VP.see(lay, s0))
    w = pl.walkable(st[0])
    S = np.flatnonzero(w[M.cidx_arr] & (M.fidx_arr >= 0))
    c_, p_ = rng.choice(S, n_pairs), rng.choice(S, n_pairs)
    out, moves = Counter(), Counter()
    for c, p in zip(c_.tolist(), p_.tolist()):
        if c == p:
            continue
        R_ = M.routes[c, p]
        clear = bool(M.route_ok[c, p]) and bool(w[R_[R_ >= 0]].all())
        s = (*where(lay, M, c), *s0[3:])
        cur, n, through = c, 0, False
        for _ in range(int(M.V[c, p]) + 1):
            a = int(M.first[cur, p])
            if a < 0:
                break
            s2, end = ld.step(lay, s, a)
            n += 1
            if end or s2[:3] == s[:3] and a == MV.FWD:
                break
            cur, s = int(M.nxt_arr[cur, a]), s2
            if cur == p:
                through = s[:3] == where(lay, M, p)
                break
        out[("clear" if clear else "blocked", "through" if through else "stopped")] += 1
        if through:
            moves["equal_V" if n == M.V[c, p] else "differ"] += 1
    return out, moves


def q_heldout(M, shares=(0.05, 0.2)):
    """Q trained with a share of the target placements left out of the loss; V on them and on the rest against
    the evaluator's open-space steps."""
    import torch
    T = {a: (np.array(v["M"], np.int64), np.array(v["b"], np.int64))
         for a, v in zip(MV.MOVES, (M.tokens_report["moves"][VP.KNAME[a]] for a in MV.MOVES))}
    X = MV.targets(2 * MV.R)
    Ve = MV.exact_open(T, X)
    far = ~(X == np.array([0, 0, 0, 1])).all(1)
    out = {}
    for share in shares:
        held = (np.random.default_rng(4545).random(len(X)) < share) & far
        net, loss = MV.train_q(T, "cuda" if torch.cuda.is_available() else "cpu", held=held)
        Q = MV.q_of(net, X)
        V, _ = MV.read_q(Q, X)
        e = np.abs(Q.min(1) - Ve)
        out[f"held_{share}"] = {
            "held_out_placements": int(held.sum()), "of": int(len(X)), "final_loss": round(loss, 6),
            "V_exact_share_held_out": round(float((V == Ve)[held].mean()), 4),
            "V_within_one_held_out": round(float((np.abs(V - Ve) <= 1)[held].mean()), 4),
            "mean_error_held_out": round(float(e[held].mean()), 4),
            "V_exact_share_trained": round(float((V == Ve)[~held & far].mean()), 4),
            "mean_error_trained": round(float(e[~held & far].mean()), 4)}
    return out


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    n_lay, n_pairs = int(get("--layouts", "30")), int(get("--pairs", "3000"))
    CS.install()
    MV.install("045")
    VP.T.configure()
    data = pickle.loads(VP.T.DATA.read_bytes())["data"]
    rng = np.random.default_rng(45)
    res = {"note": "Card 045, tools/card045/route_check.py, seed 399, arm A", "worlds": {}}
    for world in VP.WORLDS:
        W = world_model(world, 399, data)
        if "q_heldout" not in res:
            res["q_heldout"] = q_heldout(W.M)
            print("q held out", json.dumps(res["q_heldout"]), flush=True)
        tot, mv = Counter(), Counter()
        for lay in data[world]["test"][:n_lay]:
            W.reset()
            o, m = check(W, lay, n_pairs, rng)
            tot.update(o)
            mv.update(m)
        n = sum(tot.values())
        agree = tot[("clear", "through")] + tot[("blocked", "stopped")]
        r = res["worlds"][world] = {
            "pairs": n, "agreement": round(agree / n, 6),
            "clear_through": tot[("clear", "through")], "clear_stopped": tot[("clear", "stopped")],
            "blocked_through": tot[("blocked", "through")], "blocked_stopped": tot[("blocked", "stopped")],
            "through_in_V_moves": round(mv["equal_V"] / max(sum(mv.values()), 1), 6),
            "walking": W.M.tokens_report["walking"]}
        print(world, json.dumps({k: v for k, v in r.items() if k != "walking"}), flush=True)
    Path(get("--out", "runs/045_route_check.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
