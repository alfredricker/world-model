"""Card 102.4: where content went, read from arm A's learned pointer instead of card 102.2's written rules.

Arm A (card 102.1's kept form: two pair layers, softmax attention over every tile, raw positions, the action) is
trained to predict every place's next content (its encoder vector one step on), at every step, by its four ways:
keep, copy from another token by its pointer, change by a learned shift, set. Draws are prioritised by the last loss
(alpha 0.6, no importance weights; card 102.2's training format).

Reading a step: each place's way is the one with the largest weight (averaged over the four parts of the vector).
  keep or change  the content came from the place itself;
  copy            from the token the pointer picks;
  set             from nothing (new).
A token's identity goes to one place: first where it was kept, then by pointer weight; a second claim is new.
The view's own shift is the offset most places' content moved by. Each tile's fate in the step, against that shift:
still, one of four one-step moves, changed in place (kept, code changed), moved further (not used as evidence), or
ended (no place took it). Groups: card 102.2's evidence, learned prior and components, unchanged. Belief out of
view: card 102.3's, on these tracks, every tile a token (codes never seen to move are kept in the map).

  bin/prun python tools/card102.4/pointer.py world2|world3|crafter [--steps N] [--tag T]
      → runs/102.4/<world><tag>.json, net_<world>.pt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for p in ("card102.1", "card102.2", "card102.3"):
    sys.path.insert(0, str(ROOT / "tools" / p))
import arms as AR                                      # noqa: E402
import layers as LY                                    # noqa: E402
import belief as BL                                    # noqa: E402

RUNS, dev = ROOT / "runs" / "102.4", "cuda"
EGO = "--ego" in sys.argv                              # revision 2: the view placed first; the pointer reads world motion
CM = "--cm" in sys.argv                                # revision 2: a fifth way, move and change (copy, plus a paid change)
NW = 5 if CM else 4
R, C, L = 7, 9, 63
WHERE = AR.WHERE[:L]
YX = np.array([(k // C, k % C) for k in range(L)])
E4 = LY.E4
arg = AR.arg


# ------------------------------------------------------------------------------------------------ data
def load(world, split):
    if world == "world2":
        d = LY.load("world2", split)
        z = np.load(ROOT / "runs" / "card102" / f"world2_{split}.npz")
        d["light"] = np.ones(len(d["seg"]), np.float32)
    else:
        d = BL.load(world, split)
        if world == "world3":
            z = np.load(ROOT / "runs" / "card102" / f"world3_{split}.npz")
            d["act"] = z["act"]
        else:
            ws = range(8) if split == "train" else (8, 9)
            d["act"] = np.concatenate([np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")["act"] for w in ws])
    d["V"] = np.asarray(d["vec"][:, :L]) if d["vec"].shape[1] >= L else None
    d["steps"] = np.flatnonzero((d["seg"][:-1] >= 0) & (d["seg"][:-1] == d["seg"][1:]))
    n = len(d["seg"])
    d["al"] = np.tile(np.arange(L), (n, 1))                # the next frame's place showing place i's world spot
    d["shift"] = np.zeros((n, 2), np.int64)
    if EGO and world != "world2":                         # revision 2: the view's motion read first (card 102.5)
        org = np.load(ROOT / "runs" / "102.5" / f"org_{world}_{split}.npy")
        S = d["steps"]
        sh = org[S + 1] - org[S]                           # (x, y): the view's origin moved by this
        d["shift"][S] = sh
        vx, vy = YX[:, 1][None] - sh[:, :1], YX[:, 0][None] - sh[:, 1:]
        ok = (vx >= 0) & (vx < C) & (vy >= 0) & (vy < R)
        d["al"][S] = np.where(ok, vy * C + vx, -1)
    return d


# ------------------------------------------------------------------------------------------------ arm A
def batch(d, i):
    X = torch.as_tensor(d["V"][i], device=dev).float()
    Y = torch.as_tensor(d["V"][i + 1], device=dev).float()
    a = torch.as_tensor(d["act"][i].astype(np.int64), device=dev)
    W = WHERE.to(dev)[None].expand(len(i), -1, -1)
    return X, Y, a, W


def train(d, steps, seed=1021, bs=128, alpha=0.6):
    """Card 102.1's loss: a change head on every place (does its code change?) and the four ways' content where it
    does. Weighting every place's content alike learns "nothing changes" (the first run: gate 32%, keep 43%)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = AR.PairNet(32).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    S = d["steps"]
    code = d["code"].reshape(len(d["seg"]), L)
    pri = np.ones(len(S))
    t0 = time.monotonic()
    for step in range(steps):
        P = pri ** alpha
        P /= P.sum()
        k = np.sort(rng.choice(len(S), bs, p=P))
        X, Y, a, W = batch(d, S[k])
        logit, after, _, _, _ = net(X, a, None, W)
        own = torch.as_tensor(code[S[k]] != code[S[k] + 1], device=dev)
        lc = fn.binary_cross_entropy_with_logits(logit, own.float(), reduction="none").sum(1)
        la = ((after - Y).abs().sum(-1) * own).sum(1)
        li = lc + 0.1 * la
        pri[k] = li.detach().cpu().numpy() + 1e-3
        loss = li.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print({"step": step + 1, "change": round(lc.mean().item(), 4), "after": round(la.mean().item(), 4),
                   "s": round(time.monotonic() - t0, 1)}, flush=True)
    return net


# ------------------------------------------------------------------------------------------------ revision 1: MDL
class MDLNet(AR.PairNet):
    """Arm A's ways judged by description length (revision 1): each place's next content is coded by one way, and
    the loss is the expected code length in nats, so a way is chosen for what it costs, not blended.
      distortion  naming the next code from the way's prediction: -log softmax over the codes' prototypes of
                  -|prediction - prototype|^2 / tau (tau learned); codes are the model's names, so the content is
                  coded at their resolution, and an exact prediction costs nothing;
      keep        distortion only;
      copy        from a source j: distortion of x_j, plus the source's cost under a learned motion prior
                  p(j | action) over the offset from i to j (any positions, not a grid); bits back: KL(q_src || p);
      change      x + z, set z: z drawn from a Gaussian whose rate (KL to the data's spread) is paid;
      the source  a latent the pointer learns wherever the content changed, by copy's own cost (EM);
      the way     its prior p(way | action); the loss is the mixture's code length -log sum_w p(w|a) e^-C_w, so
                  each way learns from the places whose outcome it explains (its posterior share), and a way
                  head q(way | input) learns that posterior (cross-entropy) for reading before the outcome."""

    def __init__(self, dv, protos, m, sd, na=17, d=64):
        super().__init__(dv, na, d)
        self.way = nn.Linear(d, NW)
        self.way_prior = nn.Embedding(na, NW)
        self.cm, self.cm_s = nn.Linear(d, dv), nn.Linear(d, dv)
        self.mot = nn.Sequential(nn.Linear(2 + d, d), nn.ReLU(), nn.Linear(d, 1))
        self.aemb = nn.Embedding(na, d)
        self.shift_s, self.set_s = nn.Linear(d, dv), nn.Linear(d, dv)
        P = torch.as_tensor(protos).float()
        self.register_buffer("P", P)
        d2 = torch.cdist(P, P) ** 2 + torch.eye(len(P)) * 1e9
        self.lt = nn.Parameter(torch.log(d2.min(1).values.median() / 4).clamp(min=-12))
        self.register_buffer("m", torch.as_tensor(m).float())
        self.register_buffer("sd", torch.as_tensor(sd).float())

    def name(self, pred):
        """log p(code k | prediction), (..., K)."""
        d2 = (pred ** 2).sum(-1, keepdim=True) - 2 * pred @ self.P.T + (self.P ** 2).sum(-1)
        return torch.log_softmax(-d2.clamp(min=0) / self.lt.exp(), -1)

    def parts(self, X, a, W):
        n, L, _ = X.shape
        h = self.what(X) + self.where(W)
        for layer in self.layers:
            h, ptr, _ = layer(h, W, a, None)
        eye = torch.eye(L, dtype=torch.bool, device=X.device)[None]
        lq = torch.log_softmax(ptr.masked_fill(eye, -1e9), -1)                    # q(j | i): the pointer
        off = W[:, :, None] - W[:, None]                                           # i's where minus j's
        ae = self.aemb(a)[:, None, None].expand(-1, L, L, -1)
        lp = torch.log_softmax(self.mot(torch.cat([off, ae], -1))[..., 0].masked_fill(eye, -1e9), -1)
        return h, lq, lp

    def costs(self, X, k, a, W):
        """Per place: the code length of each way (n, L, 4); k is the next code's index (n, L)."""
        h, lq, lp = self.parts(X, a, W)
        n, L = k.shape
        nx = self.name(X)                                                          # (n, L, K)
        d_keep = -nx.gather(2, k[..., None])[..., 0]
        dij = -nx.gather(2, k[:, None, :].expand(-1, L, -1)).transpose(1, 2)       # [i, j]: naming i's code from x_j
        q = lq.exp()
        c_copy = (q * (dij + lq - lp)).sum(-1)
        sd = self.sd.expand_as(X)

        def gauss(mu, ls, base, pm):
            z = mu + ls.exp() * torch.randn_like(mu) if self.training else mu
            kl = (torch.log(sd) - ls + (ls.exp() ** 2 + (mu - pm) ** 2) / (2 * sd ** 2) - 0.5).sum(-1)
            return -self.name(base + z).gather(2, k[..., None])[..., 0] + kl
        c_change = gauss(self.shift(h), self.shift_s(h).clamp(-8, 2), X, 0.0)
        c_set = gauss(self.set(h), self.set_s(h).clamp(-8, 2), 0.0, self.m.expand_as(X))
        ways = [d_keep, c_copy, c_change, c_set]
        if CM:                                            # move and change: x_j + z_i, z's information paid
            mu, ls = self.cm(h), self.cm_s(h).clamp(-8, 2)
            z = mu + ls.exp() * torch.randn_like(mu) if self.training else mu
            kl = (torch.log(sd) - ls + (ls.exp() ** 2 + mu ** 2) / (2 * sd ** 2) - 0.5).sum(-1)
            nm = self.name(X[:, None, :, :] + z[:, :, None, :])                    # (n, i, j, K)
            dcm = -nm.gather(3, k[:, :, None, None].expand(-1, -1, L, 1))[..., 0]   # naming i's code from x_j + z_i
            self._dcm = dcm
            ways.append((q * (dcm + lq - lp)).sum(-1) + kl)
        C = torch.stack(ways, -1)
        return C, torch.log_softmax(self.way(h), -1), torch.log_softmax(self.way_prior(a), -1)[:, None]

    def loss(self, X, k, a, W, moved, seen=None):
        seen = torch.ones_like(moved) if seen is None else seen                     # places whose next content is seen
        C, lw, lpw = self.costs(X, k.clamp(min=0), a, W)
        lpost = lpw - C
        code = -torch.logsumexp(lpost, -1)                                         # nats per place
        post = torch.softmax(lpost, -1).detach()
        # the source is a latent the pointer learns wherever the content changed (EM), whichever way wins there
        return (code * seen).sum(1), (-(post * lw).sum(-1) * seen).sum(1) + (C[..., 1] * moved).sum(1)

    @torch.no_grad()
    def read(self, X, a, W):
        """The chosen way, the pointer's pick and weight, and the chosen way's prediction."""
        h, lq, lp = self.parts(X, a, W)
        w = self.way(h).argmax(-1)
        m, j = lq.exp().max(-1)
        copy = X.gather(1, j[..., None].expand(-1, -1, X.shape[-1]))
        wl = [X, copy, X + self.shift(h), self.set(h)] + ([copy + self.cm(h)] if CM else [])
        ways = torch.stack(wl, -2)
        pred = ways.gather(2, w[..., None, None].expand(-1, -1, 1, X.shape[-1]))[:, :, 0]
        return w, j, m, pred


@torch.no_grad()
def read_post(net, X, k, a, W):
    """Reading an observed step (both frames known): the way and source that describe it most briefly, as card
    102.2's rules read observed steps. Copy is priced by its best source, min_j (naming i's code from x_j - log
    p(j | action)), the shortest description of what happened (training's average over the pointer is a bound on it
    for learning). k: the next codes' indices (-1 unknown)."""
    ok = k >= 0
    kk = k.clamp(min=0)
    C, lw, lpw = net.costs(X, kk, a, W)
    h, lq, lp = net.parts(X, a, W)
    L = X.shape[1]
    dij = -net.name(X).gather(2, kk[:, None, :].expand(-1, L, -1)).transpose(1, 2)
    cj = dij - lp                                                              # each source's description length
    cmin, j = cj.min(-1)
    C = C.clone()
    C[..., 1] = cmin
    if CM:                                                # move and change, by its best source
        sd = net.sd.expand_as(X)
        mu = net.cm(h)
        kl = ((mu ** 2) / (2 * sd ** 2)).sum(-1)
        c4, j4 = (net._dcm - lp).min(-1)
        C[..., 4] = c4 + kl
        use4 = (lpw - C).argmax(-1) == 4
        j = torch.where(use4, j4, j)
    w = (lpw - C).argmax(-1)
    m = torch.softmax(-cj, -1).max(-1).values
    w = torch.where(ok, w, net.way(h).argmax(-1))
    return w, j, m


def scales(d):
    """Every training code's prototype (its vectors' mean), the index of each state's codes, and the data's mean
    and spread."""
    code = d["code"].reshape(len(d["seg"]), L)
    u, inv = np.unique(code, return_inverse=True)
    inv = inv.reshape(code.shape)
    dv = d["V"].shape[-1]
    S, N = np.zeros((len(u), dv)), np.zeros(len(u))
    for s in range(0, len(code), 20000):
        V = np.asarray(d["V"][s:s + 20000], np.float32).reshape(-1, dv)
        np.add.at(S, inv[s:s + 20000].ravel(), V)
        N += np.bincount(inv[s:s + 20000].ravel(), minlength=len(u))
    P = (S / np.maximum(N, 1)[:, None]).astype(np.float32)
    m = (P * N[:, None]).sum(0) / N.sum()
    sd = np.sqrt(((P - m) ** 2 * N[:, None]).sum(0) / N.sum()) + 1e-3
    return P, inv, m.astype(np.float32), sd.astype(np.float32)


def train_mdl(d, steps, seed=1021, bs=128, alpha=0.6):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    P, inv, m, sd = scales(d)
    net = MDLNet(d["V"].shape[-1], P, m, sd).to(dev)
    net.codes = np.unique(d["code"])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    S = d["steps"]
    pri = np.ones(len(S))
    t0 = time.monotonic()
    for step in range(steps):
        Pr = pri ** alpha
        Pr /= Pr.sum()
        k = np.sort(rng.choice(len(S), bs, p=Pr))
        X, _, a, W = batch(d, S[k])
        al = d["al"][S[k]]
        kn = np.where(al >= 0, np.take_along_axis(inv[S[k] + 1], np.maximum(al, 0), 1), -1)
        seen = torch.as_tensor(al >= 0, device=dev).float()
        moved = torch.as_tensor((inv[S[k]] != kn) & (al >= 0), device=dev).float()
        li, lq = net.loss(X, torch.as_tensor(kn, device=dev), a, W, moved, seen)
        pri[k] = li.detach().cpu().numpy() + 1e-3
        loss = li.mean() + float(arg("--wq", "1")) * lq.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print({"step": step + 1, "nats per state": round(li.mean().item(), 3), "way head": round(lq.mean().item(), 4), "tau": float(f"{net.lt.exp().item():.3g}"),
                   "s": round(time.monotonic() - t0, 1)}, flush=True)
    print({"codes": len(P)}, flush=True)
    return net


@torch.no_grad()
def readout_mdl(net, d, bs=256):
    """Form A: the way head's way and the pointer (before the outcome; the prediction, for the gate). Form B: the
    posterior given the observed next step (reading what happened). Returned as (way B, way A, ...)."""
    S = d["steps"]
    code = d["code"].reshape(len(d["seg"]), L)
    u = net.codes
    way, wayP, src, srcP, wgt, wgtP, pred = [], [], [], [], [], [], []
    for s in range(0, len(S), bs):
        i = S[s:s + bs]
        X, _, a, W = batch(d, i)
        w, j, m, p = net.read(X, a, W)
        al = d["al"][i]
        cn = np.take_along_axis(code[i + 1], np.maximum(al, 0), 1)
        k = np.searchsorted(u, cn)
        k = np.where((u[np.minimum(k, len(u) - 1)] == cn) & (al >= 0), k, -1)
        wp, jp, mp = read_post(net, X, torch.as_tensor(k, device=dev), a, W)
        way.append(w.cpu().numpy().astype(np.int8))
        src.append(j.cpu().numpy().astype(np.int16))
        wgt.append(m.cpu().numpy().astype(np.float32))
        wayP.append(wp.cpu().numpy().astype(np.int8))
        srcP.append(jp.cpu().numpy().astype(np.int16))
        wgtP.append(mp.cpu().numpy().astype(np.float32))
        pred.append(p.half().cpu().numpy())
    c = np.concatenate
    return (c(wayP), c(srcP), c(wgtP)), (c(way), c(src), c(wgt)), c(pred)


@torch.no_grad()
def readout(net, d, bs=512):
    """Per step: each place's way (0 keep, 1 copy, 2 change, 3 set) read from the gates at every place (form B) and
    with the change head first (form A: a place it calls unchanged is kept), the pointer's pick and weight, and the
    prediction (the change head's: unchanged places keep their content)."""
    S = d["steps"]
    way, wayA, src, wgt, pred = [], [], [], [], []
    for s in range(0, len(S), bs):
        i = S[s:s + bs]
        X, _, a, W = batch(d, i)
        n = len(i)
        h = net.what(X) + net.where(W)
        for layer in net.layers:
            h, ptr, _ = layer(h, W, a, None)
        ptr = ptr.masked_fill(torch.eye(L, dtype=torch.bool, device=dev)[None], -1e9)
        sp = torch.softmax(ptr, -1)
        g = torch.softmax(net.gate(h).view(n, L, AR.NP_, 4), -1).mean(2)
        logit, after, _, _, _ = net(X, a, None, W)
        wb = g.argmax(-1)
        way.append(wb.cpu().numpy().astype(np.int8))
        wayA.append(torch.where(logit > 0, wb, 0).cpu().numpy().astype(np.int8))
        m, j = sp.max(-1)
        src.append(j.cpu().numpy().astype(np.int16))
        wgt.append((m * g[..., 1]).cpu().numpy().astype(np.float32))
        pred.append(torch.where((logit > 0)[..., None], after, X).half().cpu().numpy())
    return (np.concatenate(way), np.concatenate(wayA), np.concatenate(src), np.concatenate(wgt),
            np.concatenate(pred))


def tracks(d, way, src, wgt):
    """Identities and fates from the readout (the interface of card 102.2's analyse)."""
    n, seg, code = len(d["seg"]), d["seg"], d["code"].reshape(len(d["seg"]), L)
    ident = np.full((n, L), -1, np.int64)
    fate0 = np.full((n, L), -1, np.int64)
    bg = np.zeros((n, 2), np.int64)
    nid = 0
    row = {t: k for k, t in enumerate(d["steps"])}
    for t in range(n):
        if seg[t] < 0:
            continue
        if t == 0 or seg[t - 1] != seg[t]:
            ident[t] = np.arange(nid, nid + L)
            nid += L
        k = row.get(t)
        if k is None:
            continue
        w, j = way[k], src[k]
        source = np.full(L, -1)
        taken = np.zeros(L, bool)
        keep = np.flatnonzero((w == 0) | (w == 2))
        source[keep] = keep
        taken[keep] = True
        for p in np.flatnonzero((w == 1) | (w == 4))[np.argsort(-wgt[k][(w == 1) | (w == 4)])]:   # copies, the surest first
            if not taken[j[p]]:
                source[p] = j[p]
                taken[j[p]] = True
        has = source >= 0
        off = YX[has] - YX[source[has]]                  # (dy, dx) the content moved by
        vals, cnt = np.unique(off, axis=0, return_counts=True)
        dy, dx = vals[cnt.argmax()]
        bg[t] = (dx, dy)                                 # the view's own shift, as card 102.2's (dx, dy)
        for p in range(L):
            q = source[p]
            if q < 0:
                ident[t + 1, p] = nid
                nid += 1
                continue
            ident[t + 1, p] = ident[t, q]
            ey, ex = YX[p] - YX[q] - (dy, dx)
            if (ex, ey) == (0, 0):
                fate0[t, q] = 5 if code[t + 1, p] != code[t, q] else 0
            elif (ex, ey) in E4:
                fate0[t, q] = 1 + E4.index((int(ex), int(ey)))
            else:
                fate0[t, q] = 6                          # moved further than one step: not used as evidence
        fate0[t, ~taken] = 7                             # no place took it: it ended
    f = fate0[np.isin(ident, np.unique(ident[(fate0 >= 1) & (fate0 <= 5)]))]
    f = f[(f >= 0) & (f <= 5)]
    return {"ident": ident.reshape(n, R, C), "fate0": fate0.reshape(n, R, C), "bg": bg, "seg": seg,
            "p_move": float((f >= 1).mean()) if len(f) else 0.1}


def tracks_ego(d, way, src, wgt):
    """Revision 2: identities and fates when the view was placed first. Place p's way and source describe the
    content of p's world spot at the next step (the next frame's place al[p]); offsets are motion in the world."""
    n, seg, code = len(d["seg"]), d["seg"], d["code"].reshape(len(d["seg"]), L)
    ident = np.full((n, L), -1, np.int64)
    fate0 = np.full((n, L), -1, np.int64)
    bg = np.zeros((n, 2), np.int64)
    nid = 0
    row = {t: k for k, t in enumerate(d["steps"])}
    for t in range(n):
        if seg[t] < 0:
            continue
        if t == 0 or seg[t - 1] != seg[t]:
            ident[t] = np.arange(nid, nid + L)
            nid += L
        k = row.get(t)
        if k is None:
            continue
        al = d["al"][t]
        w, j = way[k], src[k]
        source = np.full(L, -1)
        taken = np.zeros(L, bool)
        keep = np.flatnonzero((w == 0) | (w == 2))
        source[keep] = keep
        taken[keep] = True
        for p in np.flatnonzero((w == 1) | (w == 4))[np.argsort(-wgt[k][(w == 1) | (w == 4)])]:
            if not taken[j[p]]:
                source[p] = j[p]
                taken[j[p]] = True
        bg[t] = -d["shift"][t]                           # the content's motion in the view (card 102.2's (dx, dy))
        fate0[t] = 7                                     # no place took it: it ended
        for p in range(L):
            q, r = source[p], al[p]
            if r < 0:                                    # this world spot left the view
                if q >= 0 and q == p:
                    fate0[t, q] = -1
                continue
            if q < 0:
                ident[t + 1, r] = nid
                nid += 1
                continue
            ident[t + 1, r] = ident[t, q]
            ey, ex = YX[p] - YX[q]
            if (ex, ey) == (0, 0):
                fate0[t, q] = 5 if code[t + 1, r] != code[t, q] else 0
            elif (ex, ey) in E4:
                fate0[t, q] = 1 + E4.index((int(ex), int(ey)))
            else:
                fate0[t, q] = 6
        new = ident[t + 1] < 0                           # places entering the view
        ident[t + 1, new] = np.arange(nid, nid + new.sum())
        nid += int(new.sum())
    f = fate0[np.isin(ident, np.unique(ident[(fate0 >= 1) & (fate0 <= 5)]))]
    f = f[(f >= 0) & (f <= 5)]
    return {"ident": ident.reshape(n, R, C), "fate0": fate0.reshape(n, R, C), "bg": bg, "seg": seg,
            "p_move": float((f >= 1).mean()) if len(f) else 0.1}


def step_identity(te, A):
    """World 3 and Crafter: the share of true objects seen at two consecutive steps whose token identity carries,
    by the evaluator's kind (a diagnostic of the tracks)."""
    at = {}
    for t, i, k, dx, dy in te["obj"]:
        at.setdefault(int(t), {})[int(i)] = (int(k), BL.AR_ + int(dy), BL.AC + int(dx))
    out = {}
    for t in range(len(te["seg"]) - 1):
        if te["seg"][t] < 0 or te["seg"][t] != te["seg"][t + 1]:
            continue
        for i, (k, r, c) in at.get(t, {}).items():
            if i in at.get(t + 1, {}):
                _, r1, c1 = at[t + 1][i]
                if 0 <= r < R and 0 <= c < C and 0 <= r1 < R and 0 <= c1 < C:
                    o = out.setdefault(k, [0, 0])
                    o[0] += int(A["ident"][t, r, c] == A["ident"][t + 1, r1, c1])
                    o[1] += 1
    return {k: [round(v[0] / v[1], 4), v[1]] for k, v in sorted(out.items())}


def false_groups(A, G, seg, light):
    ok = seg >= 0
    multi = np.array([(np.unique(G[t].ravel(), return_counts=True)[1] >= 2).any() if ok[t] else False
                      for t in range(len(seg))])
    day = ok & (light >= 0.3)
    return {"states with a group of two or more tiles": round(float(multi[ok].mean()), 4),
            "by day": round(float(multi[day].mean()), 4), "states": int(ok.sum())}


def evidence_groups(A, prior=None):
    """Card 102.2's evidence and groups (LY.evidence, LY.groups), the same sums, streamed: a running pair total per
    segment instead of every pair at every step, which with every tile a participant needs tens of GB.
    Returns the segments' final evidence (for LY.fit_prior) and, given the prior, each position's group."""
    p = A["p_move"]
    a, b = np.log((1 - LY.EPS) / p), np.log(LY.EPS / (2 * p * (1 - p)))
    seg, ident, fate0 = A["seg"], A["ident"], A["fate0"]
    group = ident.copy()
    out = []
    for s in np.unique(seg[seg >= 0]):
        ts = np.flatnonzero(seg == s)
        I, F = ident[ts].reshape(len(ts), -1), fate0[ts].reshape(len(ts), -1)
        ids = np.unique(I[(F >= 1) & (F <= 5)])
        if len(ids) < 2:
            continue
        assert len(ids) <= 12000, f"segment {s}: {len(ids)} participants, over 0.6 GB of pair evidence"
        E = np.zeros((len(ids), len(ids)), np.float32)
        seen = np.zeros((len(ids), len(ids)), bool)
        for t in range(len(ts)):
            if prior is not None and t > 0:                # groups of state t from the evidence up to step t - 1
                k = np.searchsorted(ids, I[t])
                inn = (k < len(ids)) & (ids[np.minimum(k, len(ids) - 1)] == I[t])
                kk = np.unique(k[inn])
                if len(kk) >= 2:
                    pos = E[np.ix_(kk, kk)] + prior > 0
                    np.fill_diagonal(pos, False)
                    if pos.any():
                        comp = np.arange(len(kk))
                        for i, j in np.argwhere(np.triu(pos)):
                            ri, rj = i, j
                            while comp[ri] != ri:
                                ri = comp[ri]
                            while comp[rj] != rj:
                                rj = comp[rj]
                            if ri != rj:
                                comp[max(ri, rj)] = min(ri, rj)
                        for q in range(len(kk)):
                            r = q
                            while comp[r] != r:
                                r = comp[r]
                            comp[q] = r
                        lut = dict(zip(ids[kk], ids[kk][comp]))
                        g = group[ts[t]].reshape(-1)
                        for c, v in enumerate(I[t]):
                            if v in lut:
                                g[c] = lut[v]
            m = (F[t] >= 0) & (F[t] <= 5)
            k = np.searchsorted(ids, I[t][m])
            inn = (k < len(ids)) & (ids[np.minimum(k, len(ids) - 1)] == I[t][m])
            k, f = k[inn], F[t][m][inn]
            if len(k) < 2:
                continue
            ff = np.full(len(ids), -1)                       # one fate per participant at this step, as LY.evidence
            ff[k] = f
            k = np.unique(k)
            f = ff[k]
            co = (f[:, None] == f[None]) & (f[:, None] >= 1)
            split = f[:, None] != f[None]
            E[np.ix_(k, k)] += a * co + b * split
            seen[np.ix_(k, k)] = True
        out.append((ts, ids, E[:, :, None], seen))
    return out, (round(float(a), 3), round(float(b), 3)), group


def group(A_tr, A_te):
    ev_tr, ab, _ = evidence_groups(A_tr)
    prior, share = LY.fit_prior(ev_tr)
    A_te["p_move"] = A_tr["p_move"]
    _, _, G = evidence_groups(A_te, prior)
    return G, {"evidence for, against": ab, "prior log-odds": round(prior, 3),
               "share of pairs one object (training)": round(share, 4)}


def belief(world, tr, te, A_tr, A_te):
    """Card 102.3's learned belief on these tracks; every tile is a token."""
    out = []
    for d, A in ((tr, A_tr), (te, A_te)):
        n = len(d["seg"])
        origin = np.zeros((n, 2), np.int64)
        for t in range(1, n):
            if d["seg"][t] >= 0 and d["seg"][t] == d["seg"][t - 1]:
                origin[t] = origin[t - 1] - A["bg"][t - 1]
        rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
        steps = [None if d["seg"][t] < 0 else
                 {"id": A["ident"][t].ravel(), "code": d["code"][t].ravel(), "x": origin[t, 0] + cc.ravel(),
                  "y": origin[t, 1] + rr.ravel(), "vr": rr.ravel(), "vc": cc.ravel()} for t in range(n)]
        out.append((steps, origin))
    M = BL.learn(out[0][0], tr["seg"], out[0][1])
    o = te["obj"]
    at = {}
    for t, i, k, dx, dy in o:
        at.setdefault(int(t), []).append((int(i), BL.AR_ + int(dy), BL.AC + int(dx)))
    cell = {(t, vr, vc): i for t, v in at.items() for i, vr, vc in v}
    kind_of = {int(i): int(k) for _, i, k, _, _ in o}
    BL.truth_cell = lambda t, vr, vc: cell.get((t, vr, vc))
    steps, origin = out[1]
    ev, sec = BL.run(steps, te["seg"], origin, M, "learned", 1000000, lambda t: at.get(t, []), te["light"],
                     te["movers"], te["static"], kind_of, te["sem"], te["floor"])
    r = BL.summarise(ev, te["movers"], te["static"], world == "crafter")
    r["belief seconds per step"] = round(sec, 6)
    r["codes kept as static"] = sum(m["static"] for m in M.values())
    r["move rate per code (rate, steps), most seen first"] = {int(k): (round(m["move rate"], 4), m["steps"]) for k, m in
                                                              sorted(M.items(), key=lambda x: -x[1]["steps"])[:30]}
    rules = json.loads((ROOT / "runs" / "102.3" / f"belief_{world}.json").read_text())["learned"]
    r["the rules' tracks (card 102.3)"] = {k: rules[k] for k in ("log-likelihood per return (all)",
                                                                   "identity kept (all returns)",
                                                                   "inside the 90% region (returns with a file)")}
    return r


def fates_by_case(te, A):
    """World 2: the pointer's fate of each moving or changing object tile against the evaluator's, by case."""
    TF = LY.true_fates(te)
    f = A["fate0"]
    obj, sem, code = te["obj"].astype(np.int64), te["sem"].astype(np.int64), te["code"]
    out = {}
    for t in np.flatnonzero(te["seg"][:-1] == te["seg"][1:]):
        for o in np.unique(obj[t][TF[t] >= 1]):
            m = obj[t] == o
            if o == 99:
                case = "agent"
            elif o == 1:
                case = "door"
            elif len(np.unique(sem[t][m])) > 1:
                case = "shape of several colours"
            else:
                case = None
            ys, xs = np.nonzero(m)
            for y, x in zip(ys, xs):
                c = case
                if c is None:                            # one colour: a middle keeps its code where it stands
                    c = "one colour, middle" if code[t + 1][y, x] == code[t][y, x] else "one colour, end"
                r = out.setdefault(c, [0, 0])
                r[0] += int(f[t, y, x] == TF[t, y, x])
                r[1] += 1
    return {k: {"right": round(v[0] / v[1], 4), "tiles": v[1]} for k, v in out.items()}


def main():
    world = sys.argv[1]
    steps = int(arg("--steps", "12000"))
    tag = arg("--tag", "")
    t0 = time.monotonic()
    tr, te = load(world, "train"), load(world, "test")
    if "--n" in sys.argv:                                 # a smoke test on the first n steps of each split
        k = int(arg("--n", "0"))
        for d in (tr, te):
            for key in ("code", "vec", "seg", "act", "sem", "obj", "light", "pos", "V"):
                if key in d and d[key] is not None and key != "obj":
                    d[key] = d[key][:k]
            d["obj"] = d["obj"][d["obj"][:, 0] < k] if d["obj"].ndim == 2 and d["obj"].shape[1] == 5 else d["obj"][:k]
            d["steps"] = d["steps"][d["steps"] < k - 1]
    print(world, "steps", len(tr["steps"]), len(te["steps"]), f"({time.monotonic() - t0:.0f} s)", flush=True)
    MDL = "--mdl" in sys.argv
    if "--load" in sys.argv:                              # read a saved network again (no training)
        P_, _, m_, sd_ = scales(tr)
        net = MDLNet(tr["V"].shape[-1], P_, m_, sd_).to(dev)
        net.load_state_dict(torch.load(RUNS / f"net_{world}{arg('--load', '')}.pt"))
        net.codes = np.unique(tr["code"])
    else:
        net = train_mdl(tr, steps, alpha=float(arg("--alpha", "0.6"))) if MDL else train(tr, steps)
    net.eval()
    torch.save(net.state_dict(), RUNS / f"net_{world}{tag}.pt")
    rep = {"world": world, "training steps": steps}
    A = {"A": {}, "B": {}}
    for split, d in (("train", tr), ("test", te)):
        if MDL:
            (way, src, wgt), (wayA, srcA, wgtA), pred = readout_mdl(net, d)
        else:
            way, wayA, src, wgt, pred = readout(net, d)
            srcA, wgtA = src, wgt
        TR = tracks_ego if EGO else tracks
        A["B"][split] = TR(d, way, src, wgt)
        A["A"][split] = TR(d, wayA, srcA, wgtA)
        if split == "test":
            for f, w in (("A", wayA), ("B", way)):
                rep[f"form {f}: ways used (test places)"] = {nm: round(float((w == k).mean()), 4)
                                                             for k, nm in enumerate(("keep", "copy", "change", "set", "move and change"))}
            if world == "world2":                         # the gate: every place's next code right
                protos = np.load(ROOT / "runs" / "102.2" / "world2_codes.npz")["protos"]
                right = []
                for s in range(0, len(pred), 4096):
                    P = pred[s:s + 4096].astype(np.float32)
                    k = np.argmin(((P[:, :, None] - protos[None, None]) ** 2).sum(-1), -1)
                    right.append((k == d["code"].reshape(len(d["seg"]), L)[d["steps"][s:s + 4096] + 1]).all(1))
                rep["gate: every place's next code right"] = round(float(np.concatenate(right).mean()), 5)
        del way, wayA, src, wgt, srcA, wgtA, pred
    print("read out", f"({time.monotonic() - t0:.0f} s)", flush=True)
    for f in ("A", "B"):
        At, Ae = A[f]["train"], A[f]["test"]
        r = rep[f"form {f}"] = {}
        G, r["groups"] = group(At, Ae)
        if world == "world2":
            r["criterion 1: fates by case"] = fates_by_case(te, Ae)
            TF = LY.true_fates(te)
            m = (TF >= 1) & (te["obj"] > 0)
            r["criterion 1: fates of moving or changing object tiles right"] = round(float((Ae["fate0"][m] == TF[m]).mean()), 5)
            r["criterion 1: objects exactly one group"] = LY.objects_exact(te, G)
        else:
            r["criterion 2: false groups"] = false_groups(Ae, G, te["seg"], te["light"])
            r["objects' identity carried one step in view, by kind"] = step_identity(te, Ae)
            r["criterion 3: belief on the pointer's tracks"] = belief(world, tr, te, At, Ae)
        print(f, f"({time.monotonic() - t0:.0f} s)", flush=True)
    if world == "crafter":                               # the rules on the same play, for criterion 2's comparison
        ground, _ = LY.learn_ground(tr)
        R_tr, R_te = LY.analyse(tr, ground), LY.analyse(te, ground, p_move=None)
        Gr, _ = group(R_tr, R_te)
        rep["false groups (the rules, on the same play)"] = false_groups(R_te, Gr, te["seg"], te["light"])
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (RUNS / f"{world}{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
