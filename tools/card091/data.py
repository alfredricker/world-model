"""Card 091: stored tries as token situations, for the router.

A try of forward, pick up, drop or toggle becomes:
  - tokens: every tile the agent sees (MiniGrid's 7 x 7 view with occlusion, card 061's `visible`), its code and its
    where (x right, y ahead of the agent); the held tile (kind "hand"); for pick up, drop and toggle the tiles the agent
    believed present but does not see (card 062's believed view, which memory keeps without positions: kind
    "remembered"). At most L tokens, padded.
  - outcome: forward 0 moved, 1 blocked, 2 ended (as World.learn_try); pick up, drop, toggle: front changed + 2 x hand
    changed (codes up to the agent's state, as memory's play counts changes).
  - weight: memory's sampling weight (forced rows 1, uniform rows 8).
The codes are read through the agent's encoder (vectors per code), in the router.

  bin/prun python tools/card091/data.py            writes runs/091/tokens_<world>.npz for the worlds below
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card066"))
if "--tier" not in sys.argv:
    sys.argv += ["--tier", "1"]
import tiers as T                                      # noqa: E402

sys.path.insert(0, str(ROOT / "tools" / "card061"))
import moves as M61                                    # noqa: E402

OUT = ROOT / "runs" / "091"
L = 64
ACTS = (2, 3, 4, 5)                                    # forward, pick up, drop, toggle
WORLDS = {"tier1": ROOT / "runs" / "066" / "memory_tier1.npz", "tier2": ROOT / "runs" / "066" / "memory_tier2.npz",
          "tier3": ROOT / "runs" / "066" / "memory_tier3.npz",
          "decoy": ROOT / "runs" / "070" / "decoy_memory066" / "memory_tier1.npz"}
WH = np.array([T.TK.where_of(i) for i in range(T.NV)], np.int64)
VIS, HAND, REM = 1, 2, 3


def tokens(D):
    keep = np.isin(D["act"], ACTS)
    inter = np.isin(D["act"], T.INTER)
    pres_row = np.full(len(D["act"]), -1)
    pres_row[np.flatnonzero(inter)] = np.arange(int(inter.sum()))
    idx = np.flatnonzero(keep)
    e0, e1, a = D["ego0"][idx].astype(np.int64), D["ego1"][idx].astype(np.int64), D["act"][idx]
    n = len(idx)
    vis = M61.visible(e0)
    code = np.zeros((n, L), np.int16)
    wx = np.zeros((n, L), np.int8)
    wy = np.zeros((n, L), np.int8)
    kind = np.zeros((n, L), np.int8)
    for r in range(n):
        q = np.flatnonzero(vis[r, :T.NV])
        q = q[q != T.CENTRE][:L - 1]
        k = len(q)
        code[r, :k], wx[r, :k], wy[r, :k], kind[r, :k] = e0[r, q], WH[q, 0], WH[q, 1], VIS
        code[r, k], kind[r, k] = e0[r, T.HELD], HAND
        k += 1
        pr = pres_row[idx[r]]
        if pr >= 0:
            seen = set((e0[r, q] // 5 * 5).tolist())
            rem = [c for c in np.flatnonzero(D["pres"][pr]) if c // 5 * 5 not in seen][:L - k]
            code[r, k:k + len(rem)], kind[r, k:k + len(rem)] = rem, REM
    fwd = a == 2
    moved = (e0[:, :T.NV] != e1[:, :T.NV]).any(1)
    out = np.where(D["term1"][idx], 2, np.where(moved, 0, 1))
    fc = (e0[:, T.FRONT] // 5) != (e1[:, T.FRONT] // 5)
    hc = (e0[:, T.HELD] // 5) != (e1[:, T.HELD] // 5)
    out = np.where(fwd, out, fc.astype(np.int64) + 2 * hc.astype(np.int64))
    return {"code": code, "wx": wx, "wy": wy, "kind": kind, "act": a, "out": out, "w": D["w"][idx],
            "front": e0[:, T.FRONT] // 5 * 5, "held": e0[:, T.HELD] // 5 * 5}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in [x for x in sys.argv[1:] if x in WORLDS] or list(WORLDS):
        z = np.load(WORLDS[name])
        D = {k: z[k] for k in z.files}
        tk = tokens(D)
        np.savez_compressed(OUT / f"tokens_{name}.npz", **tk)
        print(name, {a: int((tk["act"] == a).sum()) for a in ACTS},
              "outcomes", {a: np.bincount(tk["out"][tk["act"] == a], minlength=4).tolist() for a in ACTS}, flush=True)


if __name__ == "__main__":
    main()
