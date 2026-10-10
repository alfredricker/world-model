"""Card 095.1: the state of each stored interaction try as a set of tokens, and what happened to each.

State: the token at every place within 2 steps of the agent (whatever it is: a drop makes a token appear on the
floor, so the floor where it lands must be in the state), the attended tokens beyond (card 094: memory or card 087
says some action changes them, or they end the episode), nearest first, at most VMAX, and the hand (place 169).
Places are relative to the agent (card 062's believed view). A token changed when its place's appearance changed
between the true windows before and after; what it became is the token the window shows after.

Held out: every tenth episode (whole episodes).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "095.1"
DMAX, VMAX = 2, 6
HAND = 169


def load(world):
    z = np.load(RUNS / f"states_{world}.npz")
    return {k: z[k] for k in z.files}


def build(z):
    """Token arrays for every try: H (n, L) token handles (-1 pad), P (n, L) places, C (n, L) changed,
    A (n, L) handle after (-1 where unchanged or pad)."""
    WH, centre = z["wh"][:169], int(z["centre"])
    dist = np.abs(WH).sum(1)
    order = [p for p in np.lexsort((np.arange(169), dist)).tolist() if p != centre]
    near = [p for p in order if dist[p] <= DMAX]
    far = [p for p in order if dist[p] > DMAX]
    app, att = z["app"], z["attended"]
    pos, e0, e1 = z["pos"].astype(np.int64), z["e0"].astype(np.int64), z["e1"].astype(np.int64)
    n = len(pos)
    L = len(near) + VMAX + 1
    H = np.full((n, L), -1, np.int64)
    P = np.zeros((n, L), np.int64)
    H[:, :len(near)] = np.where(pos[:, near] >= 0, app[np.maximum(pos[:, near], 0)], -1)
    P[:, :len(near)] = near
    fp = pos[:, far]
    fh = np.where(fp >= 0, app[np.maximum(fp, 0)], -1)
    ok = (fh >= 0) & att[np.maximum(fh, 0)]
    rank = np.cumsum(ok, 1) - 1
    take = ok & (rank < VMAX)
    r, c = np.nonzero(take)
    H[r, len(near) + rank[r, c]] = fh[r, c]
    P[r, len(near) + rank[r, c]] = np.asarray(far)[c]
    H[:, -1] = app[e0[:, HAND]]
    P[:, -1] = HAND
    pl = np.where(H >= 0, P, 0)
    C = (e0[np.arange(n)[:, None], pl] // 5 != e1[np.arange(n)[:, None], pl] // 5) & (H >= 0)
    A = np.where(C, app[e1[np.arange(n)[:, None], pl]], -1)
    return H, P, C, A


def split(z):
    """Train and held-out tries: every tenth episode held out."""
    test = z["ep"] % 10 == 0
    return ~test, test


def missed(z, C):
    """Changes in the true window at places the state does not hold (report only)."""
    e0, e1 = z["e0"].astype(np.int64), z["e1"].astype(np.int64)
    allc = (e0 // 5 != e1 // 5).sum(1)
    return int((allc - C.sum(1)).clip(0).sum()), int(allc.sum())
