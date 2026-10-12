"""Card 103.1: the world layer and the camera layer. Each pixel of the next frame is explained either by the world
(a pixel canvas of the episode seen through a window that shifts) or by content carried with the camera (the same
view position in the last frame). Which pixels the camera carries is learned, not declared.

  score(s) = log p(s | a) + sum_{q in S} log( pi_q P_cam(q) + (1 - pi_q) P_world(q | s) )
  P_world(q | s): a robust mixture, (1 - eps) N(f_q - canvas(q + o + s); sigma_t^2) + eps (outliers uniform), so that a
                  few changed pixels (a placed block, a passing cow) cannot outvote thousands of matching ones
  P_cam(q):       the same mixture against the last frame at the same view position q
  each comparison first removes a global brightness offset per colour channel (the median difference over the
  support), as direct visual odometry does for exposure: Crafter's renderer truncates, so every pixel drops one level
  whenever daylight dips below 1
  S:              the pixels the canvas knows under every candidate shift, so that every hypothesis is judged on the
                  same evidence; candidates are the shifts the learned prior gives at least 1e-3 (all on pass 0)
  sigma_t^2:      this frame's noise: the median residual at the shift chosen with the training value, then chosen again
  pi_q:           the share of view position q carried by the camera: EM on histograms of log P_cam - log P_world at
                  the chosen shift, over steps where the view moved, counted only where the canvas holds world content
                  confirmed from two view positions (content that stayed put in the world while the view moved);
                  Beta(1, 9) prior; estimated once, on pass 0 (every pixel the world's), then fixed
  p(s | a):       the agent's own placements, counted (Dirichlet 1), learned on training, fixed on test
Every pixel with pi_q < 1/2 is written into the canvas each step (the latest view of the world; writing only where
the world explained a pixel left stale pixels that the camera then always explained better).
Shifts are every pixel shift up to K each way; the step length is not declared. Arms: --region world (the frame's
49 x 63 world region, as card 102.8) or full (all 64 x 64: the inventory bar must be found as camera content too);
--pi none (every pixel the world's).

  bin/prun python tools/card103.1/place.py crafter|world2|world3|world4 [--region world|full] [--pi learned|none]
      → runs/103.1/place_<world>_<region>_<pi>.json, pi_<...>.npy, org_<...>_test.npy
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT, dev = ROOT / "runs" / "103.1", "cuda"
K = 9
NS = 2 * K + 1
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
WORLD = sys.argv[1]
REGION, PI = arg("--region", "world"), arg("--pi", "learned")
HW, WW = (49, 63) if REGION == "world" else (64, 64)
HOOK = None                                            # a diagnostic callback, set by a caller
TAU = float(arg("--tau", 1))                           # evidence tempered: pixels per independent observation (card 103.4)
NB, BLO = 121, -30.0                                   # histogram of log P_cam - log P_world: 121 bins over [-30, 30]


def load(split):
    if WORLD == "crafter":
        ws = (0, 1) if split == "train" else (8,)
        zs = [np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz") for w in ws]
        d = {k: np.concatenate([z[k] for z in zs]) for k in ("act", "done", "pos", "sleep", "light")}
        d["ep"] = np.concatenate([z["ep"] + 1000000 * w for z, w in zip(zs, ws)])
        d["obs"] = np.concatenate([z["obs"][:, :HW, :WW] for z in zs])
    else:
        z = np.load(ROOT / "runs" / "card102" / f"{WORLD}_{split}.npz")
        n = len(z["act"])
        d = {"act": z["act"], "done": z["done"], "ep": z["ep"], "obs": z["obs"][:, :HW, :WW],
             "light": z["light"], "sleep": z["sleep"] if "sleep" in z.files else np.zeros(n, bool),
             "pos": z["pos"] if "pos" in z.files else np.zeros((n, 2), np.int16)}
    if arg("--n", None):
        d = {k: v[:int(arg("--n", 0))] for k, v in d.items()}
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    d["seg"] = np.cumsum(brk)
    d["seg"][d["sleep"]] = -1
    return d


def mix(r2, s2, eps):
    """log of the robust mixture for squared residuals r2 summed over 3 channels (outliers uniform on the cube)."""
    g = -r2 / (2 * s2) - 1.5 * np.log(2 * np.pi * s2)
    return torch.logaddexp(g + np.log(1 - eps), torch.full_like(g, np.log(eps)))


@torch.no_grad()
def place(d, prior, P, learn, cand=None, half=900):
    """P: sigma^2 (floor and start for each frame), eps, pi (HW, WW). Returns origins, shifts, M-step statistics."""
    seg, act, obs = d["seg"], d["act"].astype(int), d["obs"]
    n = len(seg)
    org = np.zeros((n, 2), np.int64)
    sh = np.full((n, 2), -99)
    canvas = torch.zeros(3, 2 * half, 2 * half, device=dev)
    known = torch.zeros(2 * half, 2 * half, dtype=torch.bool, device=dev)
    conf = torch.zeros(2 * half, 2 * half, dtype=torch.bool, device=dev)       # confirmed from two view positions
    first = torch.zeros(2, 2 * half, 2 * half, dtype=torch.long, device=dev)   # origin when the content was first seen
    pi = torch.as_tensor(P["pi"], device=dev).float().clamp(1e-4, 1 - 1e-4)
    lpi, l1pi = pi.log(), (1 - pi).log()
    write0 = pi < 0.5
    s2, eps = P["s2"], P["eps"]
    hist = torch.zeros(HW * WW, NB, device=dev)
    st = {k: torch.zeros((), device=dev, dtype=torch.float64) for k in ("in_r", "in_r2", "n", "re_n", "re_same")}
    sup_n = []
    prev = None
    for t in range(n):
        if seg[t] < 0:
            prev = None
            continue
        f = torch.as_tensor(obs[t], device=dev).permute(2, 0, 1).float() / 255
        fresh = t == 0 or seg[t - 1] != seg[t] or prev is None
        s2t = s2
        if fresh:
            canvas.zero_()
            known.zero_()
            conf.zero_()
            org[t] = (0, 0)
        else:
            a = int(act[t - 1])
            lp = torch.as_tensor(np.log(prior[a] / prior[a].sum()), device=dev).float().reshape(NS, NS)
            x0, y0 = int(org[t - 1, 0]) + half - K, int(org[t - 1, 1]) + half - K
            C = canvas[:, y0:y0 + HW + 2 * K, x0:x0 + WW + 2 * K]
            M = known[y0:y0 + HW + 2 * K, x0:x0 + WW + 2 * K]
            Cu = C.unfold(1, HW, 1).unfold(2, WW, 1)                              # 3, NS, NS, HW, WW
            Mu = M.unfold(0, HW, 1).unfold(1, WW, 1)                              # NS, NS, HW, WW
            cm = torch.ones(NS, NS, dtype=torch.bool, device=dev) if cand is None else cand[a]
            sup = Mu[cm].all(0)                                                   # known under every candidate
            diff = Cu - f[:, None, None]
            if sup.sum() > 50:                       # a global brightness offset per channel and shift (exposure)
                diff = diff - diff[..., sup].median(-1).values[..., None, None]
            r2 = (diff ** 2).sum(0)
            dc = prev - f
            rc = ((dc - dc.flatten(1).median(1).values[:, None, None]) ** 2).sum(0)
            sup_n.append(int(sup.sum()))
            for rep in range(2):
                lw = mix(r2, s2t, eps)                                            # log P_world(q | s)
                lc = mix(rc, s2t, eps)                                            # log P_cam(q)
                lq = torch.logaddexp(lpi + lc, l1pi + lw)
                score = torch.where(cm, lp + (lq * sup).sum((-1, -2)) / TAU, torch.full_like(lp, -1e30))
                k = int(score.argmax())
                dy, dx = divmod(k, NS)
                if rep == 0:                                                      # this frame's noise, at its shift
                    m = r2[dy, dx][sup & (pi < 0.5)]
                    if m.numel() < 50:
                        break
                    s2t = max(float(m.median()) / 2.366, s2)
            org[t] = org[t - 1] + (dx - K, dy - K)
            sh[t - 1] = (dx - K, dy - K)
            if HOOK is not None:
                HOOK(t, a, dx - K, dy - K, lq, sup, lp, lw, lc, s2t)
            mk = Mu[dy, dx]
            rw = (l1pi + lw[dy, dx] - lq[dy, dx]).exp()                           # the world's share of each pixel
            if learn:
                prior[a, k] += 1
                g = -r2[dy, dx] / (2 * s2t) - 1.5 * np.log(2 * np.pi * s2t) + np.log(1 - eps)
                rin = (g - lw[dy, dx]).exp() * rw * mk                            # inlier and the world's
                st["in_r"] += rin.sum()
                st["in_r2"] += (rin * r2[dy, dx]).sum() * (s2 / s2t)              # in units of the floor's frames
                st["n"] += (rw * mk).sum()
                if (dx, dy) != (K, K):                                            # the view moved: who carried q
                    cw = conf[y0 + dy:y0 + dy + HW, x0 + dx:x0 + dx + WW] & mk
                    lam = (lc - lw[dy, dx]).clamp(BLO, -BLO)
                    b = ((lam - BLO) / (-2 * BLO) * (NB - 1)).round().long().flatten()
                    sel = cw.flatten()
                    hist[torch.arange(HW * WW, device=dev)[sel], b[sel]] += 1
        write = write0                     # the canvas takes the latest view of every pixel the world carries
        x, y = int(org[t, 0]) + half, int(org[t, 1]) + half
        if 0 <= x <= 2 * half - WW and 0 <= y <= 2 * half - HW:
            Cw = canvas[:, y:y + HW, x:x + WW]
            same = known[y:y + HW, x:x + WW] & (((Cw - f) ** 2).sum(0) < 9 * s2t)
            o = torch.as_tensor(org[t], device=dev)[:, None, None]
            moved_since = (first[:, y:y + HW, x:x + WW] != o).any(0)
            reob = write & known[y:y + HW, x:x + WW] & moved_since & conf[y:y + HW, x:x + WW]
            st["re_n"] += reob.sum()                      # confirmed world content seen again from a new place
            st["re_same"] += (reob & same).sum()          # ... and found as written: the placements agree
            new = write & ~same
            conf[y:y + HW, x:x + WW] = torch.where(write, same & (moved_since | conf[y:y + HW, x:x + WW]),
                                                   conf[y:y + HW, x:x + WW])
            first[:, y:y + HW, x:x + WW] = torch.where(new[None], o, first[:, y:y + HW, x:x + WW])
            canvas[:, y:y + HW, x:x + WW] = torch.where(write, f, Cw)
            known[y:y + HW, x:x + WW] |= write
        prev = f
    return org, sh, {k: float(v) for k, v in st.items()}, hist.cpu().numpy(), sup_n


def em_pi(hist, iters=300, a=1.0, b=9.0):
    """pi per view position from histograms of log P_cam - log P_world (two-component mixture, Beta(a, b) prior)."""
    lam = np.linspace(BLO, -BLO, NB)
    pi = np.full(len(hist), a / (a + b))
    for _ in range(iters):
        r = 1 / (1 + np.exp(-(np.log(pi / (1 - pi))[:, None] + lam[None])))
        pi = ((hist * r).sum(1) + a - 1 + 1e-9) / (hist.sum(1) + a + b - 2 + 2e-9)
        pi = pi.clip(1e-4, 1 - 1e-4)
    return pi


def main():
    t0 = time.monotonic()
    tr, te = load("train"), load("test")
    P = {"s2": 1e-3, "eps": 0.05, "pi": np.zeros((HW, WW))}
    log, cand = [], None
    for it in range(3):
        pr = np.ones((17, NS * NS))
        _, _, st, hist, sup_n = place(tr, pr, P, learn=True, cand=cand)
        P["s2"] = float(np.clip(st["in_r2"] / (3 * st["in_r"]), 1e-5, 1e-2))
        P["eps"] = float(np.clip(1 - st["in_r"] / st["n"], 1e-3, 0.5))
        if PI == "learned" and it < 1:
            P["pi"] = em_pi(hist).reshape(HW, WW)
        q = pr / pr.sum(1, keepdims=True)
        cand = torch.as_tensor((q >= 1e-3).reshape(17, NS, NS), device=dev)
        log.append({"pass": it, "sigma^2 (floor)": round(P["s2"], 6), "eps": round(P["eps"], 4),
                    "pixels carried by the camera (pi > 1/2)": int((P["pi"] > 0.5).sum()),
                    "support, median pixels": int(np.median(sup_n)), "seconds": round(time.monotonic() - t0)})
        print(log[-1], flush=True)
    org, sh, _, _, sup_n = place(te, pr.copy(), P, learn=False, cand=cand)
    seg = te["seg"]
    st = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:]))
    true = 7 * (te["pos"][st + 1] - te["pos"][st]).astype(np.int64)
    right = (sh[st] == true).all(1)
    moved = (true != 0).any(1)
    day = te["light"][st] >= 0.3
    pi = P["pi"]
    rep = {"world": WORLD, "region": REGION, "pi": PI, "passes": log, "test steps": int(len(st)),
           "view's motion read right": round(float(right.mean()), 4),
           "when it moved, when it did not": [round(float(right[moved].mean()), 4) if moved.any() else None,
                                              round(float(right[~moved].mean()), 4)],
           "by day, by night": [round(float(right[day].mean()), 4),
                                round(float(right[~day].mean()), 4) if (~day).any() else None],
           "misreads, most common (true dx, dy, read dx, dy): count": {
               str(tuple(q.tolist())): int(c) for q, c in sorted(zip(*np.unique(
                   np.concatenate([true[~right], sh[st][~right]], 1), axis=0, return_counts=True)),
                   key=lambda x: -x[1])[:8]},
           "candidate shifts per action (prior >= 1e-3)": {a: int(cand[a].sum()) for a in range(17)},
           "learned prior, likeliest shift per action": {
               a: [int(v) for v in np.array(divmod(int(pr[a].argmax()), NS))[::-1] - K] for a in range(17)}}
    car = pi > 0.5
    if WORLD in ("crafter", "world3"):
        tile = np.zeros_like(car)
        tile[21:28, 28:35] = True                         # the agent's tile (evaluator's knowledge, for scoring only)
        rep["camera pixels: in the agent's tile, elsewhere in the world region, in the inventory bar"] = [
            int((car & tile).sum()), int((car & ~tile)[:49].sum()), int(car[49:].sum())]
    else:
        rep["camera pixels (any)"] = int(car.sum())
    rep["seconds"] = round(time.monotonic() - t0, 1)
    OUT.mkdir(exist_ok=True, parents=True)
    tag = f"{WORLD}_{REGION}_{PI}" + (f"_tau{arg('--tau', '1')}" if TAU != 1 else "") + ("_n" + arg("--n", "") if arg("--n", None) else "")
    (OUT / f"place_{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
    np.save(OUT / f"pi_{tag}.npy", pi.astype(np.float32))
    np.save(OUT / f"org_{tag}_test.npy", org)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("passes", "candidate shifts per action (prior >= 1e-3)",
                                                                  "learned prior, likeliest shift per action")}, indent=1))


if __name__ == "__main__":
    main()
