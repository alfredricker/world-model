"""Card 098.2: Crafter tokens across light. Card 098's encoder (variant H: still pairs judged only between places
drawn identically when redrawn; a changed place as a pair loss) plus pairs across light found through the agent's
own moves.

Moves: per action, the view shift (one tile or none) that best matches the next frame most often; per step, the
action's shift happened if the shifted frame matches better than the unshifted one. Offsets: the shifts summed within
an episode. Pairs: an anchor frame drawn without noise (its next still frame drew most places identically) and another
frame of its episode, light differing by at least 0.2, view overlapping in at least 20 world places. The anchor
labels the other: a place corresponding to a tile identical to the anchor is the same thing, to a different tile a
different thing, outside the overlap unknown.

  bin/prun python tools/card098.2/across.py      → runs/098.2/results.json, runs/098.2/encoder.pt
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

os.environ.update(WM98_NEAR="0.0005", WM98_EXACT="1", WM98_CHANGED="1", WM98_PAIR="1", WM98_RADIUS="exact")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card098"))
import encoder as EN                                   # noqa: E402

OUT = ROOT / "runs" / ("098.3" if "--blocked" in sys.argv else "098.2")
dev, log = EN.dev, EN.log
SH = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]        # (dx, dy): the world's shift in the view
MIN_OVERLAP, MIN_DLIGHT = 20, 0.2
JOINT = "--joint" in sys.argv                          # revision 1: both frames' tiles as candidates in one softmax
BLOCKED = "--blocked" in sys.argv                      # card 098.3: still pairs by the view, not by the action
TAG = ("_joint" if JOINT else "") + ("_blocked" if BLOCKED else "")


def wdist(a, b, dx, dy):
    """Mean squared pixel difference between frame b's world tiles and frame a's shifted by (dx, dy), on the overlap
    (a, b: (7, 9, 3, 7, 7) uint8)."""
    r0, r1, c0, c1 = max(0, dy), 7 + min(0, dy), max(0, dx), 9 + min(0, dx)
    return float(((a[r0 - dy:r1 - dy, c0 - dx:c1 - dx].astype(np.float32)
                   - b[r0:r1, c0:c1].astype(np.float32)) ** 2).mean())


def moves(T, act, ep, done, rng):
    """The learned shift per action, and per step the shift that happened (index into SH)."""
    W = T[:, :63].reshape(len(T), 7, 9, 3, 7, 7)
    ok = np.flatnonzero((ep[:-1] == ep[1:]) & ~done[:-1])
    s = rng.choice(ok, 20000, replace=False)
    best = np.array([np.argmin([wdist(W[i], W[i + 1], *sh) for sh in SH]) for i in s])
    shift = {a: int(np.bincount(best[act[s] == a], minlength=5).argmax()) for a in np.unique(act[s])}
    step = np.zeros(len(T), np.int64)
    for i in ok:
        k = shift.get(int(act[i]), 0)
        if k and wdist(W[i], W[i + 1], *SH[k]) < wdist(W[i], W[i + 1], 0, 0):
            step[i] = k
    return shift, step


def offsets(step, ep):
    """Each frame's offset: the shifts summed from its episode's start."""
    O = np.zeros((len(step), 2), np.int64)
    for i in range(1, len(step)):
        O[i] = 0 if ep[i] != ep[i - 1] else O[i - 1] + SH[step[i - 1]]
    return O


def anchors(T, P):
    """Still-pair frames whose next frame drew most places identically (drawn without noise)."""
    out = []
    for s in range(0, len(P), 5000):
        b = P[s:s + 5000]
        d = ((T[b].astype(np.float32) - T[b + 1].astype(np.float32)) / 255.0) ** 2
        out.append(b[(d.mean((-1, -2, -3)) < EN.NEAR).mean(1) > 0.5])
    return np.concatenate(out)


def partner(t1, O, light, by_ep, ep, rng, tries=30):
    fr = by_ep[ep[t1]]
    for t2 in rng.choice(fr, min(tries, len(fr)), replace=False):
        d = O[t2] - O[t1]
        ov = max(0, 9 - abs(d[0])) * max(0, 7 - abs(d[1]))
        if abs(light[t2] - light[t1]) >= MIN_DLIGHT and ov >= MIN_OVERLAP:
            return int(t2)
    return None


CENTRE = 3 * 9 + 4                                    # the agent's own tile, at the centre of every view


def corr(d):
    """For a later frame offset by d = (DX, DY): each of its 63 world places' place in the anchor frame, or -1. The
    agent's own tile is not a world place: the centre corresponds to the centre (the agent itself), and a world place
    the agent covered in either frame is unknown."""
    out = np.full(63, -1, np.int64)
    for r in range(7):
        for c in range(9):
            r1, c1 = r - d[1], c - d[0]
            if 0 <= r1 < 7 and 0 <= c1 < 9:
                out[r * 9 + c] = r1 * 9 + c1
    out[out == CENTRE] = -1
    out[CENTRE] = CENTRE
    return out


def cross_loss(enc, X1, X2, C):
    """X1, X2: (B, 63, 3, 7, 7) anchor and later frames' world tiles; C: (B, 63) correspondences."""
    B = len(X1)
    e1 = enc(X1.reshape(-1, 3, 7, 7)).reshape(B, 63, -1)
    e2 = enc(X2.reshape(-1, 3, 7, 7)).reshape(B, 63, -1)
    d1 = ((X1[:, :, None] - X1[:, None]) ** 2).mean((-1, -2, -3))          # (B, 63, 63) within the anchor frame
    has = C >= 0
    dc = torch.gather(d1, 2, C.clamp(min=0)[:, None, :].expand(B, 63, 63))  # anchor i vs the place j corresponds to
    pos = has[:, None, :] & (dc < EN.NEAR)
    neg = has[:, None, :] & (dc >= EN.NEAR)
    if JOINT:                                          # revision: the anchor frame's own tiles join the candidates, so a
        eye = torch.eye(63, dtype=torch.bool, device=X1.device)[None]      # shift shared by the later frame's tiles
        pos = torch.cat([(d1 < EN.NEAR) & ~eye, pos], 2)                     # no longer cancels in the softmax
        neg = torch.cat([d1 >= EN.NEAR, neg], 2)
        e2 = torch.cat([e1, e2], 1)
    s = torch.einsum("bid,bjd->bij", e1, e2) / EN.TAU
    s = s.masked_fill(~(pos | neg), -1e9)
    lp = s - torch.logsumexp(s, -1, keepdim=True)
    a = pos.any(-1)
    return -((lp * pos).sum(-1)[a] / pos.sum(-1)[a]).mean()


def day_codes_at_night(code, sem, L):
    """Share of night world tiles whose code holds at least 1% of their class's day tiles."""
    ok_n, hit = 0, 0
    for k in np.unique(sem[sem >= 0]):
        day = code[(sem == k) & (L > 0.7)]
        night = code[(sem == k) & (L < 0.3)]
        if not len(night):
            continue
        if len(day):
            v, n = np.unique(day, return_counts=True)
            keep = set(v[n >= 0.01 * len(day)].tolist())
            hit += int(np.isin(night, list(keep)).sum())
        ok_n += len(night)
    return hit / max(ok_n, 1)


def evaluate(enc, z, T, Tt, P, rng, rep):
    radius = EN.boundary(enc, Tt, P, rng)
    obs, light = z["obs"], z["light"]
    idx = np.arange(0, len(obs), 5)
    E = EN.embed(enc, T[idx])
    nxt = np.minimum(idx + 1, len(obs) - 1)
    same_ep = (z["ep"][nxt] == z["ep"][idx]) & ~z["done"][idx]
    exact = np.zeros((len(idx), 81), bool)
    for s0 in range(0, len(idx), 2000):
        a = T[idx[s0:s0 + 2000]].astype(np.float32) / 255.0
        b = T[nxt[s0:s0 + 2000]].astype(np.float32) / 255.0
        exact[s0:s0 + 2000] = ((a - b) ** 2).mean((-1, -2, -3)) < EN.NEAR
    exact &= same_ep[:, None]
    code, protos, far = EN.leader(E.reshape(-1, E.shape[-1]), radius, founders=E.reshape(-1, E.shape[-1])[exact.ravel()])
    code = code.reshape(len(idx), 81)
    sem = z["sem"][idx].reshape(len(idx), 63)
    world = code[:, :63]
    L = np.repeat(light[idx], 63).reshape(len(idx), 63)
    ok = sem >= 0
    inv = z["inv"][idx]
    inv_label = np.where(inv > 0, np.arange(16)[None] * 100 + inv, -1)
    rep.update({"radius": round(radius, 4), "codes": int(len(protos)),
                "night tiles in a code of their class's day tiles": round(day_codes_at_night(world, sem, L), 4),
                "purity, world tiles": round(EN.purity(world.ravel(), sem.ravel(), ok.ravel()), 4),
                "purity, night (light < 0.3)": round(EN.purity(world.ravel(), sem.ravel(), (ok & (L < 0.3)).ravel()), 4),
                "purity, day (light > 0.7)": round(EN.purity(world.ravel(), sem.ravel(), (ok & (L > 0.7)).ravel()), 4),
                "inventory: code determines (item, count)": round(EN.purity(code[:, 63:79].ravel(), inv_label.ravel(),
                                                                         np.ones(inv_label.size, bool)), 4),
                "codes per class, by majority": EN.majority_codes(world[ok], sem[ok])})
    np.savez_compressed(OUT / f"codes{TAG}.npz", idx=idx, code=code, protos=protos, radius=radius)
    return rep


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    z = np.load(EN.OUT / "play.npz")
    T = EN.tiles(z["obs"])
    act, ep, done, light = z["act"], z["ep"], z["done"], z["light"]
    rng = np.random.default_rng(982)
    torch.manual_seed(982)
    shift, step = moves(T, act, ep, done, rng)
    O = offsets(step, ep)
    P = EN.pairs(z)
    if BLOCKED:                                        # card 098.3: still = the action's shift did not happen (blocked
        t = np.arange(len(act) - 1)                    # moves included); sleep stays out
        P = t[(ep[t] == ep[t + 1]) & ~done[t] & (step[t] == 0) & (act[t] != EN.SLEEP)]
        rep0 = {"still pairs by action / by the view": [int(len(EN.pairs(z))), int(len(P))],
                "blocked moves among them": int(np.isin(act[P], [a for a, k in shift.items() if k]).sum())}
    A = anchors(T, P)
    by_ep = {e: np.flatnonzero(ep == e) for e in np.unique(ep)}
    # data check: pairs by the darker frame's light
    chk = [partner(int(t), O, light, by_ep, ep, rng) for t in rng.choice(A, 3000, replace=False)]
    dark = [min(light[t], light[c]) for t, c in zip(A, chk) if c is not None]
    rep = {"learned shifts (action: dx, dy)": {int(a): SH[k] for a, k in shift.items() if k},
           "steps whose move shifted the view": int((step > 0).sum()),
           "move steps": int(np.isin(act, [a for a, k in shift.items() if k]).sum()),
           "anchor frames": int(len(A)),
           "anchors with a partner (of 3000 tried)": int(sum(c is not None for c in chk)),
           "partners' darker light: below 0.3 / 0.3-0.5 / above 0.5": [int(np.sum(np.array(dark) < 0.3)),
                                                                     int(np.sum((np.array(dark) >= 0.3) & (np.array(dark) < 0.5))),
                                                                     int(np.sum(np.array(dark) >= 0.5))]}
    if BLOCKED:
        rep.update(rep0)
    log(json.dumps(rep))
    enc = EN.Enc().to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Tt = torch.as_tensor(T)
    W = torch.as_tensor(T[:, :63])
    for it in range(EN.STEPS):
        b = rng.choice(P, EN.BATCH, replace=False)
        loss = EN.batch_loss(enc, Tt[b].to(dev).float() / 255.0, Tt[b + 1].to(dev).float() / 255.0)
        t1s, t2s, Cs = [], [], []
        for t in rng.choice(A, EN.BATCH * 2, replace=False):
            c = partner(int(t), O, light, by_ep, ep, rng, tries=10)
            if c is not None:
                t1s.append(int(t)); t2s.append(c); Cs.append(corr(O[c] - O[int(t)]))
            if len(t1s) == EN.BATCH:
                break
        if t1s:
            loss = loss + cross_loss(enc, W[t1s].to(dev).float() / 255.0, W[t2s].to(dev).float() / 255.0,
                                     torch.as_tensor(np.stack(Cs), device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0:
            log(f"step {it} loss {float(loss.detach()):.4f} pairs {len(t1s)} ({round(time.monotonic() - t0)} s)")
    torch.save(enc.state_dict(), OUT / f"encoder{TAG}.pt")
    enc.eval()
    evaluate(enc, z, T, Tt, P, rng, rep)
    # the baseline: card 098's variant H with the same code rule, scored the same way
    h = np.load(EN.OUT / "codes_H_e.npz")
    sem = z["sem"][h["idx"]].reshape(len(h["idx"]), 63)
    Lh = np.repeat(light[h["idx"]], 63).reshape(len(h["idx"]), 63)
    rep["variant H (098), night tiles in a code of their class's day tiles"] = round(
        day_codes_at_night(h["code"][:, :63], sem, Lh), 4)
    hb = np.load(EN.OUT / "codes_H_b.npz")
    rep["variant H boundary (098), the same"] = round(day_codes_at_night(hb["code"][:, :63], sem, Lh), 4)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"results{TAG}.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps(rep))
