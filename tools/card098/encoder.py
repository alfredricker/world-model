"""Card 098: an encoder of Crafter's 7 x 7 tiles learned from continuity, and codes from its embeddings.

Pairs (t, t+1) in which the agent did not move (no move action, no sleep). Every tile of the 9 x 9 grid (world rows
0-6, inventory rows 7-8) is an anchor:
  positives: the same place at t+1 (the same thing, a little more or less light), and the places of frame t+1 whose
             tile at t looked the same as the anchor (pixel distance below NEAR, within one frame: the same light);
  negatives: the places of frame t+1 whose tile at t looked clearly different (pixel distance above FAR).
Tiles of different frames are never negatives: they may show the same thing under other light. Loss: supervised
contrastive (Khosla et al. 2020) at temperature TAU. Codes: leader clustering of the embeddings with a radius from
the positive pairs' distances (their 99th percentile).

  bin/prun python tools/card098/encoder.py      → runs/098/encoder.pt, runs/098/codes.npz, runs/098/results.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "098"
dev = "cuda" if torch.cuda.is_available() else "cpu"
MOVES = (1, 2, 3, 4)
SLEEP = 6
NEAR = float(os.environ.get("WM98_NEAR", "0.02"))     # mean squared pixel difference (pixels in 0..1)
FAR = float(os.environ.get("WM98_FAR", "0.08"))
FNC = os.environ.get("WM98_FNC") == "1"                 # revision C: other places negative unless the encoder already
                                                        # finds them as near as true pairs (Huynh et al. 2022)
EXACT = os.environ.get("WM98_EXACT") == "1"             # revision E: where a place drew identically at t and t+1 (no
                                                        # noise there), two such places are the same thing iff their
                                                        # pixels are identical; other pairs as revision C
CHANGED = os.environ.get("WM98_CHANGED") == "1"         # revision G: in a frame pair drawn without noise (most places
                                                        # identical), a place that changed shows a different thing
PAIR = os.environ.get("WM98_PAIR") == "1"               # revision H: a changed place's two tiles pushed apart by a
                                                        # margin (Hadsell, Chopra and LeCun 2006), since InfoNCE drops
                                                        # an anchor that has no positive
NORM = os.environ.get("WM98_NORM") == "1"               # revision D: each frame's contrast normalised (per channel)
TAG = os.environ.get("WM98_TAG", "")
BOUNDARY = os.environ.get("WM98_RADIUS") in ("boundary", "exact")   # the code radius from training's own same/different
FOUND_EXACT = os.environ.get("WM98_RADIUS") == "exact"  # pairs; "exact": only noise-free pairs set it, and only tiles
                                                        # whose place drew identically when redrawn found codes
TAU = 0.1
STEPS, BATCH = 4000, 48
log = lambda m: print(m, flush=True)


def tiles(obs):
    """(n, 64, 64, 3) frames → (n, 81, 3, 7, 7) tiles in 0..1, row-major over the 9 x 9 grid."""
    t = obs[:, :63, :63].reshape(len(obs), 9, 7, 9, 7, 3).transpose(0, 1, 3, 5, 2, 4)
    return t.reshape(len(obs), 81, 3, 7, 7)


class Enc(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(),
                               nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 9, 128), nn.ReLU(),
                               nn.Linear(128, d))

    def forward(self, x):
        return fn.normalize(self.c(x), dim=-1)


def pairs(z):
    act, ep, done = z["act"], z["ep"], z["done"]
    t = np.arange(len(act) - 1)
    ok = (ep[t] == ep[t + 1]) & ~done[t] & ~np.isin(act[t], MOVES) & (act[t] != SLEEP)
    return t[ok]


def norm(X):
    """Revision D: each frame's pixels standardised per channel over its 81 tiles, then mapped back to the day
    frames' typical spread (DAY_STD) so that NEAR and FAR keep their meaning."""
    m = X.mean((1, 3, 4), keepdim=True)
    sd = X.std((1, 3, 4), keepdim=True).clamp(min=1e-3)
    return ((X - m) / sd) * DAY_STD + 0.5


DAY_STD = 0.25


def masks(enc, X0, X1):
    """Training's rule: which (place at t, place at t+1) pairs are the same thing (pos) and which different (neg)."""
    B = len(X0)
    if NORM:
        X0, X1 = norm(X0), norm(X1)
    e0 = enc(X0.reshape(-1, 3, 7, 7)).reshape(B, 81, -1)
    e1 = enc(X1.reshape(-1, 3, 7, 7)).reshape(B, 81, -1)
    d = ((X0[:, :, None] - X0[:, None]) ** 2).mean((-1, -2, -3))      # (B, 81, 81) pixel difference within frame t
    pos = d < NEAR                                                      # includes the same place (d = 0)
    neg = d > FAR
    if FNC or EXACT:                                   # every other place a negative, unless as near as a true pair
        with torch.no_grad():
            s = torch.einsum("bid,bjd->bij", e0, e1)
            same = torch.diagonal(s, dim1=1, dim2=2)   # the same place at t+1
            near = s >= torch.quantile(same.flatten().float(), 0.05)
        neg = ~pos & ~near
    if EXACT:
        exact = ((X0 - X1) ** 2).mean((-1, -2, -3)) < NEAR                 # (B, 81): this place drew identically
        both = exact[:, :, None] & exact[:, None, :]
        eye = torch.eye(81, dtype=torch.bool, device=X0.device)[None]
        pos = torch.where(both, d < NEAR, eye)                               # elsewhere only the same place
        neg = torch.where(both, d >= NEAR, ~eye & ~near)
        if CHANGED:
            quiet = exact.float().mean(1) > 0.5                              # (B,): most places drew identically
            moved = (~exact & quiet[:, None])[:, :, None] & eye              # a real change at this place
            pos, neg = pos & ~moved, neg | moved
    return pos, neg, e0, e1


def changed(X0, X1):
    """(B, 81): places that changed in a frame pair drawn without noise (most places identical)."""
    exact = ((X0 - X1) ** 2).mean((-1, -2, -3)) < NEAR
    return ~exact & (exact.float().mean(1) > 0.5)[:, None]


def batch_loss(enc, X0, X1):
    """X0, X1: (B, 81, 3, 7, 7) float tensors of frames t and t+1."""
    pos, neg, e0, e1 = masks(enc, X0, X1)
    s = torch.einsum("bid,bjd->bij", e0, e1) / TAU
    s = s.masked_fill(~(pos | neg), -1e9)
    lp = s - torch.logsumexp(s, -1, keepdim=True)
    has = pos.any(-1)                                  # anchors with a positive (revision G can remove the only one)
    loss = -((lp * pos).sum(-1)[has] / pos.sum(-1)[has]).mean()
    if PAIR:
        ch = changed(X0, X1)
        if ch.any():
            dd = (e0 - e1).norm(dim=-1)[ch]
            loss = loss + (fn.relu(1.0 - dd) ** 2).mean()
    return loss


def leader(E, radius, n=30000, seed=0, founders=None):
    """Codes: prototypes by leader clustering of a sample (each embedding joins the nearest prototype within radius,
    else starts one); then every embedding takes its nearest prototype. Also returns the share beyond the radius.
    founders: the embeddings allowed to start a prototype (default all)."""
    rng = np.random.default_rng(seed)
    F = E if founders is None else founders
    S = F[rng.choice(len(F), min(n, len(F)), replace=False)]
    P = np.zeros((0, E.shape[1]), np.float32)
    for e in S:
        if len(P):
            d = np.linalg.norm(P - e, axis=1)
            if d.min() < radius:
                continue
        P = np.vstack([P, e[None]])
    Pt, code, far = torch.as_tensor(P, device=dev), np.zeros(len(E), np.int64), 0
    for s in range(0, len(E), 65536):
        d = torch.cdist(torch.as_tensor(E[s:s + 65536], device=dev), Pt)
        m, j = d.min(1)
        code[s:s + 65536] = j.cpu().numpy()
        far += int((m >= radius).sum())
    return code, P, far / len(E)


@torch.no_grad()
def embed(enc, T):
    """T: (n, 81, 3, 7, 7) whole frames' tiles (frames are normalised as a whole under revision D)."""
    out = []
    for s in range(0, len(T), 2000):
        X = torch.as_tensor(T[s:s + 2000], dtype=torch.float32, device=dev) / 255.0
        if NORM:
            X = norm(X)
        out.append(enc(X.reshape(-1, 3, 7, 7)).reshape(len(X), 81, -1).cpu().numpy())
    return np.concatenate(out)


def purity(code, label, mask):
    """Share of tiles whose code's majority label (over the masked tiles) is their own label."""
    c, l = code[mask], label[mask]
    maj = {}
    for k in np.unique(c):
        v, cnt = np.unique(l[c == k], return_counts=True)
        maj[int(k)] = v[cnt.argmax()]
    return float(np.mean([maj[int(k)] == x for k, x in zip(c, l)]))


@torch.no_grad()
def boundary(enc, Tt, P, rng, n=200):
    """The code radius: the embedding distance that best separates (balanced accuracy) the pairs training called the
    same thing from those it called different, over n batches' pairs under training's own rule."""
    D, Y = [], []
    for _ in range(n // BATCH + 1):
        b = rng.choice(P, BATCH, replace=False)
        X0, X1 = Tt[b].to(dev).float() / 255.0, Tt[b + 1].to(dev).float() / 255.0
        pos, neg, e0, e1 = masks(enc, X0, X1)
        if FOUND_EXACT:
            exact = ((X0 - X1) ** 2).mean((-1, -2, -3)) < NEAR
            both = exact[:, :, None] & exact[:, None, :]
            pos, neg = pos & both, neg & both
        dist = torch.cdist(e0, e1)
        for m, y in ((pos, 1), (neg, 0)):
            x = dist[m]
            x = x[torch.randperm(len(x), device=dev)[:20000]]
            D.append(x.cpu().numpy())
            Y.append(np.full(len(x), y))
    D, Y = np.concatenate(D), np.concatenate(Y)
    grid = np.quantile(D, np.linspace(0.001, 0.999, 999))
    acc = [((D[Y == 1] < r).mean() + (D[Y == 0] >= r).mean()) / 2 for r in grid]
    return float(grid[int(np.argmax(acc))])


def majority_codes(c, l):
    """Per class: the codes whose majority class it is."""
    out = {}
    for k in np.unique(c):
        v, cnt = np.unique(l[c == k], return_counts=True)
        out[int(v[cnt.argmax()])] = out.get(int(v[cnt.argmax()]), 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    t0 = time.monotonic()
    z = np.load(OUT / "play.npz")
    obs, light = z["obs"], z["light"]
    T = tiles(obs)                                       # (n, 81, 3, 7, 7) uint8
    P = pairs(z)
    log(f"frames {len(obs)}, pairs without a move {len(P)} ({round(time.monotonic() - t0, 1)} s)")
    rng = np.random.default_rng(98)
    torch.manual_seed(98)
    enc = Enc().to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Tt = torch.as_tensor(T)                               # CPU, uint8
    if "--codes-only" in sys.argv:                        # an encoder already trained: only the codes again
        enc.load_state_dict(torch.load(OUT / f"encoder{TAG}.pt"))
    for step in range(0 if "--codes-only" in sys.argv else STEPS):
        b = rng.choice(P, BATCH, replace=False)
        X0 = Tt[b].to(dev).float() / 255.0
        X1 = Tt[b + 1].to(dev).float() / 255.0
        loss = batch_loss(enc, X0, X1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log(f"step {step} loss {float(loss.detach()):.4f} ({round(time.monotonic() - t0, 1)} s)")
    torch.save(enc.state_dict(), OUT / f"encoder{TAG}.pt")
    enc.eval()
    if BOUNDARY:                                       # the distance that best separates training's positives from negatives
        radius = boundary(enc, Tt, P, rng)
    else:                                              # positive pairs' distances → the code radius
        b = rng.choice(P, 2000, replace=False)
        e0 = embed(enc, T[b]).reshape(-1, 32)
        e1 = embed(enc, T[b + 1]).reshape(-1, 32)
        radius = float(np.quantile(np.linalg.norm(e0 - e1, axis=1), 0.99))
    # codes on a sample of frames (every 5th), world tiles labelled by the semantic map, inventory by (item, count)
    idx = np.arange(0, len(obs), 5)
    E = embed(enc, T[idx])
    founders = None
    if FOUND_EXACT:                                    # tiles whose place drew identically in the next frame
        nxt = np.minimum(idx + 1, len(obs) - 1)
        same_ep = (z["ep"][nxt] == z["ep"][idx]) & ~z["done"][idx]
        exact = np.zeros((len(idx), 81), bool)
        for s0 in range(0, len(idx), 2000):
            a = T[idx[s0:s0 + 2000]].astype(np.float32) / 255.0
            b2 = T[nxt[s0:s0 + 2000]].astype(np.float32) / 255.0
            exact[s0:s0 + 2000] = ((a - b2) ** 2).mean((-1, -2, -3)) < NEAR
        exact &= same_ep[:, None]
        founders = E.reshape(-1, E.shape[-1])[exact.ravel()]
        rep_exact = round(float(exact.mean()), 4)
    code, protos, far = leader(E.reshape(-1, E.shape[-1]), radius, founders=founders)
    code = code.reshape(len(idx), 81)
    sem = z["sem"][idx].reshape(len(idx), 63)
    world = code[:, :63]
    L = np.repeat(light[idx], 63).reshape(len(idx), 63)
    ok = sem >= 0
    inv = z["inv"][idx]                                  # (n, 16) counts; slot k of 18 shows item k
    inv_code = code[:, 63:79]
    inv_label = np.where(inv > 0, np.arange(16)[None] * 100 + inv, -1)
    rep = {"frames": int(len(obs)), "pairs": int(len(P)), "radius": round(radius, 4), "codes": int(len(protos)),
           "semantic classes seen": int(len(np.unique(sem[ok]))),
           "purity, world tiles": round(purity(world.ravel(), sem.ravel(), ok.ravel()), 4),
           "purity, night (light < 0.3)": round(purity(world.ravel(), sem.ravel(), (ok & (L < 0.3)).ravel()), 4),
           "purity, day (light > 0.7)": round(purity(world.ravel(), sem.ravel(), (ok & (L > 0.7)).ravel()), 4),
           "inventory: code determines (item, count)": round(purity(inv_code.ravel(), inv_label.ravel(),
                                                                    np.ones(inv_label.size, bool)), 4),
           "share beyond the radius of every prototype": round(far, 4),
           "codes per class": {int(k): int(len(np.unique(world[ok][sem[ok] == k]))) for k in np.unique(sem[ok])},
           "codes per class, by majority": majority_codes(world[ok], sem[ok]),
           "seconds": round(time.monotonic() - t0, 1)}
    if FOUND_EXACT:
        rep["share of tiles drawn identically when redrawn (founders)"] = rep_exact
    rep["variant"] = {"NEAR": NEAR, "FAR": FAR, "FNC": FNC, "EXACT": EXACT, "CHANGED": CHANGED, "PAIR": PAIR, "NORM": NORM, "radius": os.environ.get("WM98_RADIUS") or "positives' 99th percentile"}
    np.savez_compressed(OUT / f"codes{TAG}{'_' + os.environ['WM98_RADIUS'][0] if BOUNDARY else ''}.npz", idx=idx, code=code, protos=protos, radius=radius)
    (OUT / f"results{TAG}{'_' + os.environ['WM98_RADIUS'][0] if BOUNDARY else ''}.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps(rep))
