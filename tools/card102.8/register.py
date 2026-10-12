"""Card 102.8: the view's motion in pixels, then patches fixed in the world.

Placing in pixels (card 102.5's rule, at pixel resolution): each step the view's shift s (pixels, -9..9 each way) is
the most probable given belief and the action,
    score(s) = log p(s | a) - SSD(s) / (2 sigma^2) - 3/2 x (pixels not yet in belief),
SSD over the pixels of the frame's world region already in belief (a pixel canvas of the episode, last value seen),
sigma^2 the matching noise (mean squared residual per pixel at the chosen shifts, re-estimated once), and p(s | a)
learned from the agent's own placements (counts, Dirichlet 1). The sums come from convolutions, all candidates at once.

Patches fixed in the world: once each frame's origin is known, P x P patches are cut on a lattice fixed in the
world frame (the cut's phase in the frame is minus the origin, modulo P), resampled to 7 x 7, encoded by card 102.7's
simplified recipe, and coded. Stability: the share of world patches seen at two consecutive steps whose code is
unchanged, on steps where the view moved against steps where it did not.

  bin/prun python tools/card102.8/register.py [--P 8]      → runs/102.8/register_P<P>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
for p in ("card098", "card102.7"):
    sys.path.insert(0, str(ROOT / "tools" / p))
import encoder as EN                                   # noqa: E402
import patches as PT                                   # noqa: E402

OUT, dev = ROOT / "runs" / "102.8", "cuda"
HW, WW, K = 49, 63, 9                                  # the frame's world region (rows, columns); shifts -K..K pixels
NS = 2 * K + 1


def frames(ws):
    zs = [np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz") for w in ws]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("act", "done", "pos", "sleep", "light")}
    d["ep"] = np.concatenate([z["ep"] + 1000000 * w for z, w in zip(zs, ws)])
    d["obs"] = np.concatenate([z["obs"][:, :HW, :WW] for z in zs])
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    d["seg"] = np.cumsum(brk)
    d["seg"][d["sleep"]] = -1
    return d


@torch.no_grad()
def place(d, prior, s2, learn, half=700):
    """Each frame's origin (x, y) in pixels in its segment's world frame."""
    seg, act, obs = d["seg"], d["act"].astype(int), d["obs"]
    n = len(seg)
    org = np.zeros((n, 2), np.int64)
    sh = np.full((n, 2), -99)
    canvas = torch.zeros(3, 2 * half, 2 * half, device=dev)
    known = torch.zeros(1, 2 * half, 2 * half, device=dev)
    ones = torch.ones(1, 1, HW, WW, device=dev)
    res, cnt = 0.0, 0
    lo = (0, 0)
    for t in range(n):
        if seg[t] < 0:
            continue
        f = torch.as_tensor(obs[t], device=dev).permute(2, 0, 1).float() / 255
        if t == 0 or seg[t - 1] != seg[t]:
            canvas.zero_()
            known.zero_()
            org[t] = (0, 0)
        else:
            a = int(act[t - 1])
            lp = torch.as_tensor(np.log(prior[a] / prior[a].sum()), device=dev).reshape(NS, NS)
            x0, y0 = int(org[t - 1, 0]) + half - K, int(org[t - 1, 1]) + half - K
            C = canvas[:, y0:y0 + HW + 2 * K, x0:x0 + WW + 2 * K]
            M = known[:, y0:y0 + HW + 2 * K, x0:x0 + WW + 2 * K]
            nk = fn.conv2d(M[None], ones)[0, 0]                                           # (NS, NS)
            w2 = fn.conv2d((M * C * C).sum(0, keepdim=True)[None], ones)[0, 0]
            cr = fn.conv2d((M * C)[None], f[None])[0, 0]
            v2 = fn.conv2d(M[None], (f * f).sum(0, keepdim=True)[None])[0, 0]
            ssd = (v2 - 2 * cr + w2).clamp(min=0)
            # the full likelihood: a known pixel by its residual, an unseen one at a matched pixel's expected cost
            # (3 channels x 1/2), so shifting into the unseen is neither favoured nor punished and the prior counts
            score = torch.where(nk > 0, lp - ssd / (2 * s2) - 1.5 * (HW * WW - nk), lp - 1e9)
            k = int(score.argmax())
            dy, dx = divmod(k, NS)
            org[t] = org[t - 1] + (dx - K, dy - K)
            sh[t - 1] = (dx - K, dy - K)
            if learn:
                prior[a, k] += 1
            if nk[dy, dx] > 0:
                res += float(ssd[dy, dx])
                cnt += int(nk[dy, dx]) * 3
        x, y = int(org[t, 0]) + half, int(org[t, 1]) + half
        if 0 <= x <= 2 * half - WW and 0 <= y <= 2 * half - HW:
            canvas[:, y:y + HW, x:x + WW] = f
            known[:, y:y + HW, x:x + WW] = 1
    return org, sh, (res / cnt if cnt else s2)


def cut_world(obs, org, P):
    """Patches on a lattice fixed in the world: per frame, rows x cols patches of P pixels whose world corners are
    multiples of P. Returns (n, rows * cols, 3, 7, 7) and each patch's world lattice index (n, rows * cols, 2)."""
    rows, cols = (HW - P + 1) // P, (WW - P + 1) // P
    n = len(obs)
    T = np.zeros((n, rows * cols, 3, 7, 7), np.uint8)
    I = np.zeros((n, rows * cols, 2), np.int64)
    for t in range(n):
        ox, oy = (-org[t]) % P
        x = torch.as_tensor(obs[t, oy:oy + rows * P, ox:ox + cols * P]).permute(2, 0, 1).float()[None]
        p = x.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, P, P)
        if P != 7:
            p = fn.interpolate(p, size=(7, 7), mode="area")
        T[t] = p.round().clamp(0, 255).byte().numpy()
        gx = (org[t, 0] + ox) // P + np.arange(cols)
        gy = (org[t, 1] + oy) // P + np.arange(rows)
        I[t] = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    return T, I


def main():
    P = int(sys.argv[sys.argv.index("--P") + 1]) if "--P" in sys.argv else 8
    t0 = time.monotonic()
    tr, te = frames((0, 1, 2)), frames((8,))
    prior, s2 = np.ones((17, NS * NS)), 0.01
    for it in range(2):
        pr = prior.copy()
        org_tr, _, s2n = place(tr, pr, s2, learn=True)
        s2 = max(s2n, 1e-5)
    org_te, sh, _ = place(te, pr.copy(), s2, learn=False)
    seg = te["seg"]
    st = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:]))
    true = 7 * (te["pos"][st + 1] - te["pos"][st])
    right = (sh[st] == true).all(1)
    moved = (true != 0).any(1)
    day = te["light"][st] >= 0.3
    rep = {"pixel patch size P": P, "matching noise sigma^2": round(s2, 6), "test steps": int(len(st)),
           "view's motion read right (pixels)": round(float(right.mean()), 4),
           "when it moved, when it did not": [round(float(right[moved].mean()), 4), round(float(right[~moved].mean()), 4)],
           "by day, by night": [round(float(right[day].mean()), 4), round(float(right[~day].mean()), 4)],
           "misreads, most common (true dx, dy, read dx, dy): count": {
               str(tuple(q.tolist())): int(c) for q, c in sorted(zip(*np.unique(np.concatenate([true[~right], sh[st][~right]], 1),
                                                                     axis=0, return_counts=True)), key=lambda x: -x[1])[:8]},
           "learned prior, likeliest shift per arrow": {a: [int(v) for v in np.array(divmod(int(pr[a].argmax()), NS))[::-1] - K]
                                                         for a in (1, 2, 3, 4)}}
    print(json.dumps(rep, indent=1), f"({time.monotonic() - t0:.0f} s)", flush=True)
    if "--place-only" in sys.argv:
        return
    # patches fixed in the world, encoded and coded by card 102.7's simplified recipe
    Ttr, Itr = cut_world(tr["obs"], org_tr, P)
    Tte, Ite = cut_world(te["obs"], org_te, P)
    N = Ttr.shape[1]
    Pr = np.flatnonzero((tr["seg"][:-1] >= 0) & (tr["seg"][:-1] == tr["seg"][1:]) & ~np.isin(tr["act"][:-1], (1, 2, 3, 4)))
    rng = np.random.default_rng(1028)
    torch.manual_seed(1028)
    enc = EN.Enc().to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Tt = torch.as_tensor(Ttr)
    for step in range(4000):
        b = rng.choice(Pr, 48, replace=False)
        l, _, _, _ = PT.loss(enc, Tt[b].to(dev).float() / 255, Tt[b + 1].to(dev).float() / 255)
        opt.zero_grad()
        l.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        Dp, Dn = [], []
        for _ in range(10):
            b = rng.choice(Pr, 48, replace=False)
            _, e0, e1, pos = PT.loss(enc, Tt[b].to(dev).float() / 255, Tt[b + 1].to(dev).float() / 255)
            dist = torch.cdist(e0, e1)
            Dp.append(dist[pos].cpu().numpy())
            Dn.append(dist[~pos].cpu().numpy())
        Dp, Dn = np.concatenate(Dp), np.concatenate(Dn)
        cand = np.quantile(np.concatenate([Dp, Dn]), np.linspace(0.01, 0.99, 99))
        radius = float(cand[np.argmax([(Dp < c).mean() + (Dn >= c).mean() for c in cand])])
        E = []
        for s in range(0, len(Tte), 2000):
            X = torch.as_tensor(Tte[s:s + 2000], device=dev).float() / 255
            E.append(enc(X.reshape(-1, 3, 7, 7)).reshape(len(X), N, -1).cpu().numpy())
        E = np.concatenate(E)
        Etr = []
        for s in range(0, len(Ttr), 20 * 2000):
            X = torch.as_tensor(Ttr[s:s + 40000:20], device=dev).float() / 255
            Etr.append(enc(X.reshape(-1, 3, 7, 7)).cpu().numpy())
    _, protos, _ = EN.leader(np.concatenate(Etr), radius)
    Pt = torch.as_tensor(protos, device=dev)
    F = E.reshape(-1, 32)
    code = np.concatenate([torch.cdist(torch.as_tensor(F[s:s + 65536], device=dev), Pt).argmin(1).cpu().numpy()
                           for s in range(0, len(F), 65536)]).reshape(len(E), N)
    same_m, same_s = [], []
    for j, t in enumerate(st):
        a = {tuple(i): c for i, c in zip(Ite[t].tolist(), code[t])}
        b = [(a[tuple(i)], c) for i, c in zip(Ite[t + 1].tolist(), code[t + 1]) if tuple(i) in a]
        if b:
            v = np.mean([x == y for x, y in b])
            (same_m if moved[j] else same_s).append(v)
    rep["codes"] = int(len(protos))
    rep["world patches keeping their code: across a move, across a still step"] = [round(float(np.mean(same_m)), 4),
                                                                                   round(float(np.mean(same_s)), 4)]
    rep["seconds"] = round(time.monotonic() - t0, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"register_P{P}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
