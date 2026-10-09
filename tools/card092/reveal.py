"""Card 092 in the agent: when the goal's tile is in no place of the believed map, the need "a tile like it comes into
view" is pursued before the fallback's guesses (card 072) and random actions:

  1. exploration toward never-seen places next to known walkable ones (card 057), as before;
  2. else a known tile that recall can make walkable, with never-seen places beyond it (card 060's open_to_see, its key
     and blockers chained as any ("walk", j) need), ranked by recall's view effect:

     P(u comes into view | toggle j) = (Σ_k w_k n_k(u) + α p0(u)) / (Σ_k w_k n_k + α)

     over the stored toggles k of this memory, by their front tile: w_k = 1 for the same tile as j, else
     exp(−λ·|z_j − z_fk|) with recall's own fitted metric for toggle's front part; n_k(u) the tries after which a tile
     like u came into view away from the front (like: the same tile, else exp(−λ·|z_u − z_r|) with pick up's front
     metric); p0(u) the share over every toggle; α = 1. Then by closeness, as card 060.

  WM_REVEAL=recall | oracle (gate: 1 for a locked door, else 0; written by hand) ...   tools/card069/run.py hooks it
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATS = {"reveal_unseen_goal_steps": 0, "reveal_acts": 0, "reveal_explore": 0, "reveal_open": 0, "reveal_none": 0}
STATE = {}
ALPHA = 1.0
DEBUG = os.environ.get("WM_REVEAL_DEBUG") == "1"


def _name(W, h):
    inv = STATE.setdefault("inv", {int(hh): t for t, hh in reversed(list(enumerate(W.S.hof.tolist())))})
    t = inv.get(int(h))
    return STATE["VP"].CO.name_of_code(t) if t is not None else "?"


def effects(T, W, tier):
    """Per stored toggle: its front tile's handle and the handles that came into view away from the front."""
    sys.path.insert(0, str(ROOT / "tools" / "card061"))
    import moves as M61                                # noqa: E402
    z = np.load(T.OUT / f"memory_tier{tier}.npz")
    act, ego0, ego1, pres = z["act"], z["ego0"], z["ego1"], z["pres"]
    inter = np.flatnonzero(np.isin(act, T.INTER))
    rows = np.flatnonzero(act[inter] == T.VP.TOG)
    APP = T.Kd.APP
    vis = M61.visible(ego1[inter[rows]].astype(np.int64))
    skip = {int(APP[T.KR.code(None)])}
    fronts, revealed = [], []
    for i, r in enumerate(rows):
        before = {int(APP[c]) for c in np.flatnonzero(pres[r])}
        m = vis[i, :T.NV].copy()
        m[T.FRONT] = m[T.CENTRE] = False           # the front's own change; the agent's tile
        after = {int(APP[c]) for c in ego1[inter[r], :T.NV][m]}
        fronts.append(int(APP[ego0[inter[r], T.FRONT]]))
        revealed.append(frozenset(after - before - skip))
    fr = np.array(fronts)
    uf = np.unique(fr)
    n = np.array([(fr == f).sum() for f in uf], np.float64)
    rev = [[s for s, f in zip(revealed, fronts) if f == g and s] for g in uf]
    return {"fronts": uf, "n": n, "revealed": rev, "tries": len(rows),
            "with_reveal": int(sum(1 for s in revealed if s))}


def p_reveal(W, u, j_tile):
    E = STATE["E"]
    A = W.S.arr
    kt, kp = W.kinds[STATE["VP"].TOG], W.kinds[STATE["VP"].PICK]
    D = A.shape[1]
    lt = np.asarray(kt.lam2[:D], np.float64)
    lp = np.asarray(kp.lam2[:D], np.float64)
    key = (u, j_tile)
    r = STATE["P"].get(key)
    if r is not None:
        return r
    w = np.where(E["fronts"] == j_tile, 1.0, np.exp(-(np.abs(A[E["fronts"]] - A[j_tile]) @ lt)))
    like = lambda s: max((1.0 if int(x) == u else float(np.exp(-(np.abs(A[int(x)] - A[u]) @ lp))) for x in s), default=0.0)
    nu = np.array([sum(like(s) for s in rv) for rv in E["revealed"]])
    p0 = (nu.sum() + 1e-3) / (E["n"].sum() + 1e-3)
    r = STATE["P"][key] = float((w @ nu + ALPHA * p0) / (w @ E["n"] + ALPHA))
    return r


def ranked_open_to_see(pl, st):
    """Card 060's open_to_see with its candidates ranked by P(the goal's tile comes into view) first."""
    PV, F = STATE["PV"], STATE["F"]
    u = STATE.get("u")
    if u is None:
        return STATE["open0"](pl, st)
    M = F.M
    f = pl.facts[st[0]]
    nb = M.nb_tokens
    op = pl.openable(st[0])
    unseen_next = ((nb >= 0) & (f[np.maximum(nb, 0)] == PV.ABSENT)).any(1)
    js = np.flatnonzero(op[:len(unseen_next)] & unseen_next[:len(op)]).tolist()
    W = pl.W
    mode = STATE["mode"]
    if mode == "oracle":
        pr = lambda j: 1.0 if _name(W, f[j]).startswith("door") and "locked" in _name(W, f[j]) else 0.0
    else:
        pr = lambda j: p_reveal(W, u, int(f[j]))
    if DEBUG:
        doors = [j for j in range(len(op)) if _name(W, f[j]).startswith("door")]
        print(f"   reveal: openable {int(op.sum())} unseen-next {int(unseen_next.sum())} candidates {len(js)}; doors "
              f"{[(j, _name(W, f[j]), bool(op[j]), bool(unseen_next[j])) for j in doors]}", flush=True)
    keyed = [(-round(pr(j), 6), pl.closeness(st, M.facing[j]) or (99, 9), j) for j in js]
    cands = [j for *_, j in sorted(keyed)]
    for j in cands:
        t0 = time.monotonic()
        res = pl.solve(("walk", j), st, 1, [], [], [])
        if DEBUG:
            print(f"   reveal: j {j} {_name(W, f[j])} p {pr(j):.3f} plan {res is not None} {time.monotonic() - t0:.2f}s",
                  flush=True)
        if res is not None:
            pl.look_j = j
            STATS["reveal_open"] += 1
            return int(res.action)
    return None


def install(T):
    PV = T.PV
    STATE.update(VP=T.VP, PV=PV, F=PV.F, G=None, P={}, mode=os.environ.get("WM_REVEAL", "recall"))
    STATE["open0"] = PV.open_to_see
    PV.open_to_see = ranked_open_to_see
    base = PV.fallback
    explore0_ref = [PV.explore]

    def fallback(pl, st, rng, rec):
        g = getattr(pl, "goal", None)
        if g is not None and g[0] == "has" and not (pl.facts[st[0]][T.TK.LAT] == g[1]).any():
            STATS["reveal_unseen_goal_steps"] += 1
            STATE["u"] = int(g[1])
            t0 = time.monotonic()
            try:
                a = explore0_ref[0](pl, st)
            finally:
                STATE["u"] = None
            if DEBUG:
                print(f"   reveal: explore -> {a} {time.monotonic() - t0:.2f}s", flush=True)
            if a is not None:
                TRY = sys.modules.get("trying")
                if TRY is not None and TRY.STATE.get("hyp") is not None:  # the need to see comes before a guess
                    TRY.STATE["hyp"] = None
                    TRY._clear(pl.W, pl)
                STATS["reveal_acts"] += 1
                rec["explore"] += 1
                return a
            STATS["reveal_none"] += 1
        return base(pl, st, rng, rec)

    PV.fallback = fallback


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        STATE["E"] = effects(T, W, tier)
        info["reveal"] = {"mode": STATE.get("mode", os.environ.get("WM_REVEAL", "recall")), "toggles": STATE["E"]["tries"],
                          "toggles_with_a_reveal": STATE["E"]["with_reveal"], "fronts": len(STATE["E"]["fronts"])}
        log(f"card 092 reveal: {info['reveal']}")
        return W, info

    T.setup = setup
    T.INSTALL.append(lambda *a, **k: install(T))           # after card 072's (run.py hooks this later)
