"""Card 104.1: the view's motion from a pretrained motion model. 103.4's placing scores each candidate shift s of the view
by its prior p(s | a) and a tempered pixel match against the episode's canvas. Here the evidence is the motion a frozen,
pretrained model reads between frames t and t+1, scored per point q by the robust mixture

  log[ (1 - eps) N2(F_q + s; sigma_t^2) + eps / A ]      (content moves by -s when the view shifts by s; A = 361 px^2)

summed over the points the world carries (103.1's camera pixels, pi >= 1/2, left out) and divided by tau_M. sigma_t^2:
this pair's median squared residual at its best shift over the median of chi-square with 2 degrees of freedom, floored
at sigma0^2. Only steps with more than one candidate shift (the move actions: stay, or one tile) need the evidence.

Models: RAFT (torchvision raft_large, C_T_SKHT_V2): dense flow between the two frames. CoTracker3 (torch hub, offline
checkpoint, run causally): a 12 x 16 grid of points queried on the window's first frame and again on frame t, tracked
over the window t-6 .. t+1 (cut at the segment's start), read between t and t+1.
Resolutions: the stored 64 px frames (7 px per tile) enlarged 4x bicubic, or card 104's 16 px render (workers 10-12)
enlarged 2x (RAFT needs at least 128 px).
Every flow is returned in 64 px units on the 49 x 63 world region.

  bin/prun python tools/card104.1/flow.py gate [raft] [cotracker]    → runs/104.1/gate.json (worker 10, M alone)
  bin/prun python tools/card104.1/flow.py flow MODEL curr|play W     → runs/104.1/flow_<MODEL>_<src>_w<W>.npz
  bin/prun python tools/card104.1/flow.py fit MODEL                  → runs/104.1/fit_<MODEL>.json (sigma0, eps, tau_M)
  bin/prun python tools/card104.1/flow.py place MODEL                → runs/104.1/org_<src>_w<W>.npy, placing_<MODEL>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "104.1"
dev = "cuda"
K, NS, UNIT, HW, WW, A = 9, 19, 7, 49, 63, 361.0
PZ = np.load(ROOT / "runs" / "103.5" / "placing_tau300.npz")
PI = PZ["pi"]                                            # 103.1's camera layer, learned (the agent's tile)
CAND = PZ["cand"].reshape(17, -1)
PRIOR = PZ["pr"] / PZ["pr"].sum(1, keepdims=True)        # p(shift | action), 103.4's habit
SHIFTS = np.array([(k % NS - K, k // NS - K) for k in range(NS * NS)])     # (dx, dy) per candidate index
CHI2_MED = 2 * np.log(2)


def source(kind, w, hi=False):
    z = np.load(ROOT / "runs" / ("102.1" if kind == "curr" else "098.5") / (f"curr_w{w}.npz" if kind == "curr" else f"play_w{w}.npz"))
    d = {k: z[k] for k in ("act", "done", "pos", "sleep", "light", "ep")}
    d["obs"] = z["obs"][:, :HW, :WW]
    if hi:
        d["hi"] = np.load(ROOT / "runs" / "104" / f"hi_w{w}.npy", mmap_mode="r")
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    d["seg"] = np.cumsum(brk)
    d["seg"][d["sleep"]] = -1
    a = d["act"].astype(int)
    n = len(a)
    pair = np.r_[(d["seg"][:-1] >= 0) & (d["seg"][:-1] == d["seg"][1:]), False]
    d["steps"] = np.flatnonzero(pair & (CAND[a].sum(1) > 1))          # steps whose shift needs evidence
    d["start"] = np.zeros(n, np.int64)                                 # first frame of each step's segment
    s0 = 0
    for t in range(1, n):
        if d["seg"][t] != d["seg"][t - 1]:
            s0 = t
        d["start"][t] = s0
    return d


def frames(d, idx, res):
    """(B, 3, H, W) float 0..255 at the model's resolution."""
    if res == "hi":                                                    # 112 x 144 → 224 x 288 (RAFT needs >= 128)
        x = torch.as_tensor(np.stack([d["hi"][i] for i in idx]), device=dev).permute(0, 3, 1, 2).float()
        x = Fn.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 255)
    else:
        x = torch.as_tensor(d["obs"][idx], device=dev).permute(0, 3, 1, 2).float()
        x = Fn.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 255)
        x = Fn.pad(x, (0, 4, 0, 4), mode="replicate")                  # 196 x 252 → 200 x 256 (multiples of 8)
    return x


def to64(F, res):
    """A flow field (B, 2, H, W) at the model's resolution → (B, 2, 49, 63) in 64 px units."""
    if res == "hi":
        return Fn.interpolate(F, size=(HW, WW), mode="area") * (UNIT / 32)
    return Fn.avg_pool2d(F[:, :, :4 * HW, :4 * WW], 4) / 4


class Raft:
    def __init__(self):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.net = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
        self.mask = torch.as_tensor(PI < 0.5, device=dev).flatten()

    @torch.no_grad()
    def __call__(self, d, steps, res, bs=32):
        Fs, Ms = [], []
        for b in range(0, len(steps), bs):
            t = steps[b:b + bs]
            x0, x1 = frames(d, t, res) / 127.5 - 1, frames(d, t + 1, res) / 127.5 - 1
            F = to64(self.net(x0, x1, num_flow_updates=12)[-1], res)
            Fs.append(F.flatten(2).transpose(1, 2).half().cpu())
            Ms.append(self.mask.expand(len(t), -1).cpu())
        return torch.cat(Fs).numpy(), torch.cat(Ms).numpy()


class CoTracker:
    def __init__(self, gy=12, gx=16, back=6):
        self.net = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True).to(dev).eval()
        self.gy, self.gx, self.back = gy, gx, back

    def grid(self, res):
        H, W = (224, 288) if res == "hi" else (196, 252)
        ys, xs = (np.arange(self.gy) + 0.5) * H / self.gy, (np.arange(self.gx) + 0.5) * W / self.gx
        p = np.array([(x, y) for y in ys for x in xs], np.float32)
        s = (UNIT / 32) if res == "hi" else 0.25
        q64 = np.floor(p * s).astype(int)
        keep = PI[q64[:, 1].clip(0, HW - 1), q64[:, 0].clip(0, WW - 1)] < 0.5
        return p, keep, s

    @torch.no_grad()
    def __call__(self, d, steps, res, bs=1):              # its offline forward supports one clip per batch
        p, keep, s = self.grid(res)
        P = len(p)
        Fs, Ms = np.zeros((len(steps), 2 * P, 2), np.float16), np.zeros((len(steps), 2 * P), bool)
        w0 = np.maximum(d["start"][steps], steps - self.back)
        L = steps + 2 - w0
        for n in np.unique(L):
            ii = np.flatnonzero(L == n)
            for b in range(0, len(ii), bs):
                j = ii[b:b + bs]
                idx = (w0[j][:, None] + np.arange(n)[None]).reshape(-1)
                v = frames(d, idx, res)
                v = v.reshape(len(j), n, *v.shape[1:])
                q = torch.as_tensor(np.concatenate([np.concatenate([np.full((P, 1), f, np.float32), p], 1)
                                                    for f in (0, n - 2)]), device=dev)       # the window's start and t
                tr, vis = self.net(v, queries=q[None].repeat(len(j), 1, 1))
                dF = (tr[:, n - 1] - tr[:, n - 2]) * s
                Fs[j] = dF.half().cpu().numpy()
                Ms[j] = (vis[:, n - 1] & vis[:, n - 2]).cpu().numpy() & np.r_[keep, keep][None]
        return Fs, Ms


def loglik(F, M, cands, s20=0.01, eps=0.05):
    """Sum over points of the robust mixture, per candidate: (n, len(cands)); sigma_t^2 per pair (two passes)."""
    F = torch.as_tensor(F, device=dev).float()
    M = torch.as_tensor(M, device=dev).float()
    S = torch.as_tensor(SHIFTS[cands], device=dev).float()                             # (C, 2)
    r2 = ((F[:, None] + S[None, :, None]) ** 2).sum(-1)                                # (n, C, Q)
    s2 = torch.full((len(F), 1, 1), s20, device=dev)
    for _ in range(2):
        g = -r2 / (2 * s2) - torch.log(2 * np.pi * s2)
        ll = (torch.logaddexp(g + np.log(1 - eps), torch.full_like(g, np.log(eps / A))) * M[:, None]).sum(-1)
        best = ll.argmax(1)
        rb = r2[torch.arange(len(F)), best]
        med = torch.nan_to_num(torch.nanmedian(rb.masked_fill(M == 0, float("nan")), 1).values, nan=1.0)
        s2 = (med / CHI2_MED).clamp(min=s20)[:, None, None]
    return ll.cpu().numpy(), s2.flatten().cpu().numpy()


def decide(d, F, M, prior, tau, s20, eps):
    """Shifts for d's evidence steps: argmax over each action's candidates of log prior + loglik / tau."""
    a = d["act"].astype(int)[d["steps"]]
    sh = np.zeros((len(d["steps"]), 2), int)
    for act in np.unique(a):
        i = np.flatnonzero(a == act)
        c = np.flatnonzero(CAND[act])
        ll = np.concatenate([loglik(F[i[b:b + 2048]], M[i[b:b + 2048]], c, s20, eps)[0] for b in range(0, len(i), 2048)])
        lp = np.log(prior[act, c] / prior[act, c].sum()) if prior is not None else np.zeros(len(c))
        sh[i] = SHIFTS[c[(lp[None] + ll / tau).argmax(1)]]
    return sh


def score(d, sh):
    """Evaluator only: blocked moves and moves that succeeded, by day and night."""
    t = d["steps"]
    true = UNIT * (d["pos"][t + 1] - d["pos"][t]).astype(int)
    right = (sh == true).all(1)
    blocked = (true == 0).all(1)
    night = d["light"][t] < 0.5
    f = lambda m: [round(float(right[m].mean()), 4) if m.any() else None, int(m.sum())]
    return {"blocked moves, by day": f(blocked & ~night), "blocked moves, at night": f(blocked & night),
            "blocked moves": f(blocked), "moves that succeeded, by day": f(~blocked & ~night),
            "moves that succeeded, at night": f(~blocked & night), "moves that succeeded": f(~blocked)}


def gate():
    OUT.mkdir(parents=True, exist_ok=True)
    d = source("play", 10, hi=True)
    rep = json.loads((OUT / "gate.json").read_text()) if (OUT / "gate.json").exists() else {}
    rep.update({"worker": 10, "evidence steps": int(len(d["steps"]))})
    for name, M in (("raft", Raft), ("cotracker", CoTracker)):
        if len(sys.argv) > 2 and name not in sys.argv[2:]:
            continue
        m = M()
        for res in ("x4", "hi"):
            t0 = time.monotonic()
            F, Mk = m(d, d["steps"], res)
            torch.cuda.synchronize()
            batch_s = time.monotonic() - t0
            t1 = time.monotonic()                                          # one step at a time, as in the agent
            for t in d["steps"][:50]:
                Fi, Mi = m(d, np.array([t]), res, bs=1)
                decide(dict(d, steps=np.array([t])), Fi, Mi, None, 1.0, 0.01, 0.05)
            torch.cuda.synchronize()
            fps = 50 / (time.monotonic() - t1)
            sh = decide(d, F, Mk, None, 1.0, 0.01, 0.05)
            rep[f"{name} {res}"] = dict(score(d, sh), **{"pairs per second, batched": round(len(F) / batch_s, 1),
                                                         "steps per second, one at a time": round(fps, 1)})
            print(name, res, json.dumps(rep[f"{name} {res}"]), flush=True)
            (OUT / "gate.json").write_text(json.dumps(rep, indent=1) + "\n")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    {"gate": gate}[sys.argv[1]]()
