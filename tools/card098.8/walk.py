"""Card 098.8: the encoder trained by one rule of continuity for every token, shaped by its own motion and by the
action's change.

A walk over tokens (Jabri, Owens and Efros 2020). Each frame's 63 world tiles are tokens, each with its encoder vector
z and its place in the episode's own coordinates (card 098.5's placing step, place.py). One step, token i at t to
token j at t+1, has one score:
  W:      z_i . z_j / tau                                                   (appearance only)
  +M:     + log m(offset_ij | z_i, a)        m: the motion network; offsets beyond two places get no weight
  +C:     the mixture of two explanations
            stays:    log(1 - pi) + z_i . z_j / tau + log m
            changed:  log pi + c(z_i, a, z_agent, where i is from the agent) . z_j / tau + log m
  +N:     each tile encoded with its neighbours (the 3 x 3 tiles around it, places outside the view marked)
The score normalised over j is the forward step, over i the backward step.

How the motion network learns (--motion; the card's revision is "appearance"):
  appearance  it predicts the offsets of the links appearance alone makes (z_i . z_j / tau), each weighted by how
              surely appearance's one-step cycle comes back; in the walk its prediction is a fixed prior
  walk        trained through the walk (the design as first written; it learned "nothing moves")
  own         it predicts the offsets of the walk's own links, a fixed prior in the walk
Its loss reaches the encoder through its input in every case.

Two losses, both the walk's:
  cycle:  forward through k consecutive frames and back (the palindrome); the walk must come back to its own start
          token (a token drawn identically does not count, or nothing makes it stay on the same token);
  tie:    the start frame's tokens against the start frame's and the far frame's in one softmax (both frames as
          candidates, so a light shift shared by one frame does not cancel); targets are the start frame's tokens
          drawn identically and the far frame's tokens the forward walk reached, weighted by how surely the walk
          came back (a link it cannot come back through counts little).

  bin/prun python tools/card098.8/walk.py --arm W|WM|WMC|WMCN [--motion appearance|walk|own] [--kmax 40]
                                          [--steps N] [--time 200] [--no-vectors]
      → runs/098.8/<arm>[_<motion>]/encoder.pt, codes.npz, train.json, vec_w*.npy (float16, 63 world places)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card098"))
import encoder as EN                                   # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
ARM = arg("--arm", "WMC")
MOTION_FIT = arg("--motion", "appearance")
MOTION, CHANGE, NEIGH = "M" in ARM, "C" in ARM, "N" in ARM
FIXED = MOTION and MOTION_FIT in ("appearance", "own")  # motion's prior held fixed in the walk
STEPS = int(arg("--steps", "12000"))
TIME = int(arg("--time", "0"))                         # time this many steps, then stop
KMIN, KMAX = 2, int(arg("--kmax", "40"))
PLAY = ROOT / "runs" / "098.5"
OUT = ROOT / "runs" / "098.8" / (ARM + ("" if MOTION_FIT == "appearance" else "_" + MOTION_FIT)
                                 + ("" if KMAX == 40 else f"_k{KMAX}"))
TRAIN, TEST = list(range(8)), [8, 9]
dev, log = EN.dev, EN.log
TAU, NEAR = EN.TAU, 0.0005                             # card 098.2's settings
B = 12
DROP = 0.0                                             # no edge dropout: a tile has one right continuation, and dropping
                                                       # it (Jabri drops 10% of links among patches) forces a wrong one
PI0 = 0.05                                             # prior chance of a change
CENTRE = 31
RC = np.array([(k % 9, k // 9) for k in range(63)])   # (column, row) of each world place
OFF = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)]   # the motion network's 25 offsets


def load(w):
    """One worker's play: the world view (49 x 63 pixels), actions, the steps that may follow each other, and each
    frame's offset."""
    z = np.load(PLAY / f"play_w{w}.npz")
    img = np.ascontiguousarray(z["obs"][:, :49, :63])
    act, ep, done = z["act"].astype(np.int64), z["ep"], z["done"]
    O = np.load(ROOT / "runs" / "098.8" / f"place_w{w}.npz")["O"]       # the placing step's own motion (place.py)
    ok = np.zeros(len(act), bool)
    ok[:-1] = (ep[:-1] == ep[1:]) & ~done[:-1]
    return {"img": img, "act": act, "O": O, "ok": ok, "z": z}


def tiles(img):
    """(..., 49, 63, 3) uint8 tensor → (..., 63, 3, 7, 7) uint8, row-major over the 7 x 9 world places."""
    sh = img.shape[:-3]
    t = img.reshape(-1, 7, 7, 9, 7, 3).permute(0, 1, 3, 5, 2, 4)
    return t.reshape(*sh, 63, 3, 7, 7)


def crops(img):
    """(n, 49, 63, 3) uint8 tensor → (n, 63, 4, 21, 21): each tile with the 3 x 3 tiles around it; a fourth channel
    marks the places inside the view."""
    x = img.permute(0, 3, 1, 2).float() / 255.0
    x = torch.cat([x, torch.ones_like(x[:, :1])], 1)
    x = fn.pad(x, (7, 7, 7, 7))
    u = fn.unfold(x, 21, stride=7)                                       # (n, 4 * 441, 63)
    return u.transpose(1, 2).reshape(len(img), 63, 4, 21, 21)


class EncN(nn.Module):
    """Card 098's encoder shape, reading a tile with its neighbours (4 x 21 x 21)."""
    def __init__(self, d=32):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 16, 128),
                               nn.ReLU(), nn.Linear(128, d))

    def forward(self, x):
        return fn.normalize(self.c(x), dim=-1)


def encode(enc, img):
    """img (..., 49, 63, 3) uint8 tensor on the device → (..., 63, d)."""
    sh = img.shape[:-3]
    flat = img.reshape(-1, 49, 63, 3)
    if NEIGH:
        z = enc(crops(flat).reshape(-1, 4, 21, 21))
    else:
        z = enc(tiles(flat).reshape(-1, 3, 7, 7).float() / 255.0)
    return z.reshape(*sh, 63, -1)


@torch.no_grad()
def embed(enc, img, chunk=1000):
    """Every frame of one worker: (n, 63, d) float32 numpy."""
    out = []
    for s in range(0, len(img), chunk):
        out.append(encode(enc, torch.as_tensor(img[s:s + chunk], device=dev)).cpu().numpy())
    return np.concatenate(out)


class Motion(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d + 17, 128), nn.ReLU(), nn.Linear(128, len(OFF)))

    def forward(self, z, a):                           # z (..., d), a (...) action ids
        return fn.log_softmax(self.f(torch.cat([z, fn.one_hot(a, 17).float()], -1)), -1)


class Change(nn.Module):
    """From a token's vector, the action, the agent's vector and where the token is from the agent (one of the nine
    places around it, or farther): the chance it changes, and the vector it becomes."""
    def __init__(self, d=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * d + 17 + 10, 128), nn.ReLU(), nn.Linear(128, d + 1))
        with torch.no_grad():
            self.f[-1].bias[-1] = float(np.log(PI0 / (1 - PI0)))

    def forward(self, z, a, za, rel):
        h = self.f(torch.cat([z, fn.one_hot(a, 17).float(), za, fn.one_hot(rel, 10).float()], -1))
        return fn.normalize(h[..., :-1], dim=-1), fn.logsigmoid(h[..., -1]), fn.logsigmoid(-h[..., -1])


def near_agent():
    """Each world place's index among the nine places around the agent (0-8), 9 if farther."""
    out = np.full(63, 9, np.int64)
    for k, (c, r) in enumerate(RC):
        if abs(c - 4) <= 1 and abs(r - 3) <= 1:
            out[k] = (r - 2) * 3 + (c - 3)
    return out


NEAR_AGENT = torch.as_tensor(near_agent(), device=dev)


def offset_index(dO):
    """dO (n, 2): the world's shift in the view between t and t+1 (O[t+1] - O[t]). Returns (n, 63, 63) indices into
    OFF of token i's world move to token j, -1 beyond two places."""
    rc = torch.as_tensor(RC, device=dev)
    d = rc[None, None, :, :] - rc[None, :, None, :] - torch.as_tensor(dO, device=dev)[:, None, None, :]
    ok = (d.abs() <= 2).all(-1)
    return torch.where(ok, (d[..., 1] + 2) * 5 + (d[..., 0] + 2), torch.full_like(ok, -1, dtype=torch.long))


def scores(Z0, Z1, a, dO, nets):
    """Z0, Z1 (n, 63, d): frames t and t+1; a (n,): the action at t. The step's score (n, 63, 63), and what the
    motion network's fit needs (its log-probabilities, the offsets, appearance's score)."""
    app = torch.einsum("nid,njd->nij", Z0, Z1) / TAU
    s, extra = app, None
    if MOTION:
        lm = nets["motion"](Z0, a[:, None].expand(-1, 63))                 # (n, 63, 25)
        idx = offset_index(dO)
        lmo = torch.gather(lm.detach() if FIXED else lm, 2, idx.clamp(min=0)).masked_fill(idx < 0, -1e4)
        if CHANGE:
            za = Z0[:, CENTRE:CENTRE + 1].expand(-1, 63, -1)
            c, lpi, l1pi = nets["change"](Z0, a[:, None].expand(-1, 63), za, NEAR_AGENT[None].expand(len(Z0), -1))
            sc = torch.einsum("nid,njd->nij", c, Z1) / TAU
            s = torch.logaddexp(l1pi[..., None] + s, lpi[..., None] + sc)
        s = s + lmo
        extra = (lm, idx, app.detach())
    return s, extra


def motion_fit(F, Bk, extra):
    """The motion network predicts the offsets of the links: appearance's alone (the revision) or the walk's own,
    each weighted by how surely its one-step cycle comes back."""
    lm, idx, app = extra
    with torch.no_grad():
        if MOTION_FIT == "appearance":
            Fd, Bd = torch.softmax(app, -1), torch.softmax(app, -2)
        else:
            Fd, Bd = F.reshape(-1, 63, 63), Bk.reshape(-1, 63, 63).transpose(-1, -2)
        r = (Fd * Bd).sum(-1)                                                                # (n, 63)
        q = torch.zeros(Fd.shape[0], 63, len(OFF), device=dev).scatter_add_(2, idx.clamp(min=0), Fd * (idx >= 0))
    return -(r * (q * lm).sum(-1)).sum() / r.sum().clamp(min=1e-6)


def batch(data, rng, k):
    """B walks of k steps: frames (B, k+1) as (worker, t)."""
    out = []
    while len(out) < B:
        w = int(rng.choice(TRAIN))
        d = data[w]
        t0 = int(rng.integers(0, len(d["act"]) - k - 1))
        if d["ok"][t0:t0 + k].all():
            out.append((w, t0))
    return out


def walk_loss(enc, nets, data, rng, k):
    W = batch(data, rng, k)
    X = torch.as_tensor(np.stack([data[w]["img"][t:t + k + 1] for w, t in W]), device=dev)  # (B, k+1, 49, 63, 3)
    A = torch.as_tensor(np.stack([data[w]["act"][t:t + k] for w, t in W]), device=dev)       # (B, k)
    O = np.stack([data[w]["O"][t:t + k + 1] for w, t in W])                                  # (B, k+1, 2)
    Z = encode(enc, X)                                                                       # (B, k+1, 63, d)
    dO = (O[:, 1:] - O[:, :-1]).reshape(-1, 2)
    d = Z.shape[-1]
    S, extra = scores(Z[:, :-1].reshape(-1, 63, d), Z[:, 1:].reshape(-1, 63, d), A.reshape(-1), dO, nets)
    S = S.reshape(B, k, 63, 63)
    F = torch.softmax(S, -1)                                                                 # forward: i → j
    Bk = torch.softmax(S, -2).transpose(-1, -2)                                              # backward: j → i
    fit = motion_fit(F, Bk, extra) if FIXED else torch.zeros((), device=dev)
    p = torch.eye(63, device=dev)[None].expand(B, -1, -1)
    for t in range(k):
        p = p @ F[:, t]
    fwd = p                                                                                  # start → far frame
    for t in reversed(range(k)):
        p = p @ Bk[:, t]
    X0 = tiles(X[:, 0]).float() / 255.0                                                      # (B, 63, 3, 7, 7)
    same = ((X0[:, :, None] - X0[:, None]) ** 2).mean((-1, -2, -3)) < NEAR                   # (B, 63, 63), diagonal in
    ret = torch.diagonal(p, dim1=1, dim2=2).clamp(min=1e-6)                                  # (B, 63): its own start
    cycle = -ret.log().mean()
    # tie: the start frame against the start and the far frame in one softmax
    Z0, Zk = Z[:, 0], Z[:, -1]
    s = torch.einsum("nid,njd->nij", Z0, torch.cat([Z0, Zk], 1)) / TAU                       # (B, 63, 126)
    eye = torch.eye(63, dtype=torch.bool, device=dev)[None]
    s = s.masked_fill(torch.cat([eye, torch.zeros_like(eye)], 2), -1e4)
    ls = torch.log_softmax(s, -1)
    with torch.no_grad():
        q0 = (same & ~eye).float()
        q0 = q0 / q0.sum(-1, keepdim=True).clamp(min=1)
        has0 = (same & ~eye).any(-1, keepdim=True).float()
        q = torch.cat([0.5 * q0, fwd * (1 - 0.5 * has0)], -1)
        wgt = ret.detach()                                                                    # surely it came back
    tie = -((q * ls).sum(-1) * wgt).sum() / wgt.sum().clamp(min=1e-6)
    return cycle, tie, fit, ret.mean().detach()


@torch.no_grad()
def tracking_check(enc, nets, d, n=400, rng=None):
    """The gate: by day (light above 0.7), the forward step's likeliest continuation against the evaluator's. Creature
    tokens: the same object at t+1; scenery: the same world place (truth: the agent's true motion)."""
    z = d["z"]
    light, pos, obj = z["light"], z["pos"].astype(int), z["obj"]
    sem = z["sem"].reshape(len(light), 63)
    at = {}
    for t, i, k, dx, dy, fx, fy in obj:
        at.setdefault(int(t), {})[int(i)] = (int(k), (dy + 3) * 9 + (dx + 4))
    ts = np.flatnonzero(d["ok"] & (light > 0.7))
    ts = rng.choice(ts, min(n, len(ts)), replace=False)
    X = torch.as_tensor(np.stack([d["img"][t:t + 2] for t in ts]), device=dev)
    Z = encode(enc, X)
    a = torch.as_tensor(d["act"][ts], device=dev)
    S, _ = scores(Z[:, 0], Z[:, 1], a, d["O"][ts + 1] - d["O"][ts], nets)
    j = S.argmax(-1).cpu().numpy()                                                           # (n, 63)
    cre, scn = [0, 0], [0, 0]
    for b, t in enumerate(ts):
        o0, o1 = at.get(int(t), {}), at.get(int(t) + 1, {})
        occ1 = {p for _, p in o1.values()}
        occ0 = {p for _, p in o0.values()}
        for i, (k, p0) in o0.items():
            if k in (1, 2, 3, 4) and i in o1:
                cre[0] += int(j[b, p0] == o1[i][1])
                cre[1] += 1
        dp = pos[t + 1] - pos[t]
        for p0 in range(63):
            if p0 == CENTRE or p0 in occ0 or sem[t, p0] < 0:
                continue
            c, r = RC[p0][0] - dp[0], RC[p0][1] - dp[1]
            if 0 <= c < 9 and 0 <= r < 7 and r * 9 + c not in occ1 and r * 9 + c != CENTRE:
                scn[0] += int(j[b, p0] == r * 9 + c)
                scn[1] += 1
    return {"creatures: likeliest continuation is the same object": [round(cre[0] / max(cre[1], 1), 4), cre[1]],
            "scenery: likeliest continuation is the same place": [round(scn[0] / max(scn[1], 1), 4), scn[1]]}


@torch.no_grad()
def radius_and_codes(enc, nets, data, rng):
    """The code radius from training's own matches: links taken with probability above 0.9 both ways are the same;
    tokens of one frame that drew identically when redrawn but differ in pixels are different. Codes: leader
    clustering, founded only by tiles drawn identically when redrawn (card 098's rule)."""
    Ds, Dd = [], []
    for _ in range(40):
        w = int(rng.choice(TRAIN))
        d = data[w]
        ts = rng.choice(np.flatnonzero(d["ok"][:-1] & d["ok"][1:]), 64, replace=False)
        X = torch.as_tensor(np.stack([d["img"][t:t + 3] for t in ts]), device=dev)
        Z = encode(enc, X)
        S, _ = scores(Z[:, 0], Z[:, 1], torch.as_tensor(d["act"][ts], device=dev), d["O"][ts + 1] - d["O"][ts], nets)
        both = (torch.softmax(S, -1) > 0.9) & (torch.softmax(S, -2) > 0.9)
        Ds.append(torch.cdist(Z[:, 0], Z[:, 1])[both].cpu().numpy())
        X1, X2 = tiles(X[:, 1]).float() / 255.0, tiles(X[:, 2]).float() / 255.0
        exact = ((X1 - X2) ** 2).mean((-1, -2, -3)) < NEAR
        pix = ((X1[:, :, None] - X1[:, None]) ** 2).mean((-1, -2, -3))
        diff = exact[:, :, None] & exact[:, None, :] & (pix >= NEAR)
        Dd.append(torch.cdist(Z[:, 1], Z[:, 1])[diff].cpu().numpy())
    Ds, Dd = np.concatenate(Ds), np.concatenate(Dd)
    Dd = Dd[rng.choice(len(Dd), min(len(Dd), 200000), replace=False)]
    grid = np.quantile(np.concatenate([Ds, Dd]), np.linspace(0.001, 0.999, 999))
    acc = [((Ds < r).mean() + (Dd >= r).mean()) / 2 for r in grid]
    radius = float(grid[int(np.argmax(acc))])
    E, F = [], []
    for w in TRAIN:
        d = data[w]
        idx = np.arange(0, len(d["act"]) - 1, 20)
        e = embed(enc, d["img"][idx])
        t1 = tiles(torch.as_tensor(d["img"][idx])).float() / 255.0
        t2 = tiles(torch.as_tensor(d["img"][idx + 1])).float() / 255.0
        ex = ((((t1 - t2) ** 2).mean((-1, -2, -3)) < NEAR).numpy()) & d["ok"][idx][:, None]
        E.append(e.reshape(-1, e.shape[-1]))
        F.append(e[ex])
    code, protos, far = EN.leader(np.concatenate(E), radius, founders=np.concatenate(F))
    return radius, protos, {"same links": int(len(Ds)), "different pairs": int(len(Dd)),
                            "balanced accuracy at the radius": round(float(max(acc)), 4),
                            "codes": int(len(protos)), "share beyond every prototype": round(far, 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rng = np.random.default_rng(988)
    torch.manual_seed(988)
    data = {w: load(w) for w in TRAIN}
    log(f"arm {OUT.name}: loaded workers {TRAIN} ({time.monotonic() - t0:.0f} s)")
    enc = (EncN() if NEIGH else EN.Enc()).to(dev)
    nets = nn.ModuleDict({"motion": Motion(), "change": Change()}).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(nets.parameters()), lr=1e-3)
    rep = {"arm": OUT.name, "motion": MOTION_FIT if MOTION else None, "steps": STEPS, "batch": B, "k": [KMIN, KMAX],
           "log": [], "gate": {}}
    n = TIME or STEPS
    t1 = time.monotonic()
    for it in range(n):
        k = int(rng.integers(KMIN, KMAX + 1))
        cycle, tie, fit, ret = walk_loss(enc, nets, data, rng, k)
        loss = cycle + tie + fit
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == n - 1:
            row = {"step": it, "cycle": round(float(cycle), 4), "tie": round(float(tie), 4),
                   "motion fit": round(float(fit), 4), "came back": round(float(ret), 4),
                   "s": round(time.monotonic() - t1, 1)}
            rep["log"].append(row)
            log(json.dumps(row))
        if not TIME and (it + 1) % (STEPS // 4) == 0 and it + 1 < STEPS:
            enc.eval()
            rep["gate"][f"tracking, worker 7, after {(it + 1) * 4 // STEPS} quarter(s)"] = \
                tracking_check(enc, nets, data[7], rng=rng)
            log(json.dumps(rep["gate"]))
            enc.train()
    if TIME:
        log(f"{TIME} steps in {time.monotonic() - t1:.1f} s: {STEPS} steps ≈ "
            f"{(time.monotonic() - t1) / TIME * STEPS / 60:.1f} min")
        return
    enc.eval()
    rep["gate"]["tracking, worker 7, at the end"] = tracking_check(enc, nets, data[7], rng=rng)
    torch.save({"enc": enc.state_dict(), "nets": nets.state_dict()}, OUT / "encoder.pt")
    radius, protos, info = radius_and_codes(enc, nets, data, rng)
    rep["radius"], rep["codes"] = round(radius, 4), info
    with torch.no_grad():                              # the chance of a change the change network gives, by place
        if CHANGE:
            d = data[7]
            ts = rng.choice(np.flatnonzero(d["ok"]), 200, replace=False)
            Z = encode(enc, torch.as_tensor(d["img"][ts], device=dev))
            a = torch.as_tensor(d["act"][ts], device=dev)
            _, lpi, _ = nets["change"](Z, a[:, None].expand(-1, 63), Z[:, CENTRE:CENTRE + 1].expand(-1, 63, -1),
                                       NEAR_AGENT[None].expand(len(ts), -1))
            pi = lpi.exp().cpu().numpy()
            rep["chance of a change, median: the agent's place, the four beside it, farther"] = [
                round(float(np.median(pi[:, CENTRE])), 4),
                round(float(np.median(pi[:, [22, 30, 32, 40]])), 4),
                round(float(np.median(pi[:, NEAR_AGENT.cpu().numpy() == 9])), 4)]
    np.savez_compressed(OUT / "codes.npz", protos=protos, radius=radius)
    if "--no-vectors" not in sys.argv:
        for w in range(10):
            z = np.load(PLAY / f"play_w{w}.npz")
            np.save(OUT / f"vec_w{w}.npy", embed(enc, np.ascontiguousarray(z["obs"][:, :49, :63])).astype(np.float16))
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / "train.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps({k: v for k, v in rep.items() if k != "log"}))


if __name__ == "__main__":
    main()
