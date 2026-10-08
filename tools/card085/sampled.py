"""Card 085: recall's weight fits on sampled pairs, so that building memory grows linearly with it.

Version 18's fits of card 039 (lambda and alpha) and card 042 (lambda1, lambda2, beta) compare every pair of stored
keys at every step: n^2 per step, which fails on tier 3 (54,119 keys for one action: a 698 GiB tensor). Here, above
M_EXACT keys per action, each step compares a sample of query keys with a sample of other keys, each sum over others
scaled by its sampling fraction (a minibatch estimate of the same leave-one-out likelihood). At M_EXACT keys or fewer
the fits are version 18's, unchanged. Installed by install() (WM_SAMPLED=1); the objectives, steps and learning rates
are the originals'.

  WM_SAMPLED=1 ...                       before T.setup(); see tools/card085/run.py
  WM_SAMPLE_QUERIES=1                    card 085.2: only the query keys sampled, each compared with every key
"""
import os
import sys

import numpy as np

SP = sys.modules["slot_planner"]
CR = sys.modules["code_recall"]
VP = SP.VP
M_EXACT = int(os.environ.get("WM_SAMPLE_EXACT", "2048"))    # at most this many keys: the exact fits
B = int(os.environ.get("WM_SAMPLE_B", "512"))              # query keys per step
MC = int(os.environ.get("WM_SAMPLE_C", "512"))             # other keys per step, drawn uniformly
K = int(os.environ.get("WM_SAMPLE_K", "8"))                # keys per query from its own code group
BIG = 1e9                                                  # a padded place: never nearest
DECAY = os.environ.get("WM_SAMPLE_DECAY") == "1"             # card 085.1: the rate falls to zero over the last half
QUERIES = os.environ.get("WM_SAMPLE_QUERIES") == "1"      # card 085.2: sample only the query keys, compare with all
BQ = int(os.environ.get("WM_SAMPLE_BQ", "64"))              # card 085.2: query keys per step
NQ = int(os.environ.get("WM_SAMPLE_NQ", "4096"))            # card 085.2: query keys for alpha's grid and the reports
CHUNK = int(os.environ.get("WM_SAMPLE_CHUNK", "1024"))      # card 085.2: stored sets per slice
QS = 64                                                     # card 085.2: query keys compared at once
STATS = {"sampled_fits": [], "exact_fits": 0}


def _schedule(opt, steps):
    """Card 085.1: the learning rate constant for the first half of the steps, then falling linearly to zero, so the
    minibatch fit ends on the optimum, not at a noisy point near it. Without WM_SAMPLE_DECAY=1: constant (card 085)."""
    import torch
    h = steps // 2
    f = (lambda t: 1.0 if t < h else max(0.0, (steps - t) / (steps - h))) if DECAY else (lambda t: 1.0)
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _torch():
    import torch
    return torch


def _sets(S, ulist):
    """Every distinct stored set as a padded (ns, m, Dv) tensor and its mask."""
    torch = _torch()
    m = max(1, max(len(SP.SETS[s]) for s in ulist))
    ns, Dv = len(ulist), S.arr.shape[1]
    E = np.zeros((ns, m, Dv), np.float32)
    mk = np.zeros((ns, m), bool)
    for i, s in enumerate(ulist):
        E[i, :len(SP.SETS[s])] = S.arr[SP.SETS[s]]
        mk[i, :len(SP.SETS[s])] = True
    return torch.as_tensor(E, device="cuda"), torch.as_tensor(mk, device="cuda")


def cross_chamfer(Et, mt, pa, pb, lamv):
    """Chamfer distance (A's things to B's nearest, plus B's to A's) between the sets at positions pa and those at
    pb, under view weights lamv: (|pa|, |pb|). The same numbers as card 039's a2b + a2b.T on those pairs."""
    torch = _torch()
    EA, MA, EB, MB = Et[pa], mt[pa], Et[pb], mt[pb]
    a, m, Dv = EA.shape
    b = EB.shape[0]
    C = SP._wl1()(EA.reshape(a * m, Dv), EB.reshape(b * m, Dv), lamv).reshape(a, m, b, m)
    ab = C.masked_fill(~MB[None, None], BIG).min(-1).values.masked_fill(~MA[:, :, None], 0.0).sum(1)
    ba = C.masked_fill(~MA[:, :, None, None], BIG).min(1).values.masked_fill(~MB[None], 0.0).sum(-1)
    return ab + ba


def paired_chamfer(Et, mt, pa, pb, lamv):
    """Chamfer distance between the set at pa[i] and each set at pb[i, j]: (B, K)."""
    EA, MA, EB, MB = Et[pa], mt[pa], Et[pb], mt[pb]                    # (B, m, Dv), (B, m), (B, K, m, Dv), (B, K, m)
    C = ((EA[:, None, :, None, :] - EB[:, :, None, :, :]).abs() * lamv).sum(-1)    # (B, K, a, b)
    ab = C.masked_fill(~MB[:, :, None, :], BIG).min(-1).values.masked_fill(~MA[:, None, :], 0.0).sum(-1)
    ba = C.masked_fill(~MA[:, None, :, None], BIG).min(2).values.masked_fill(~MB, 0.0).sum(-1)
    return ab + ba


def _isp(x):
    return float(np.log(np.expm1(x)))


# ---------------------------------------------------------------- card 085.2: query keys sampled, every key compared

def match_feats(Et, mt, pa, pb, lamv):
    """For each set at pa and each set at pb: the per-number |differences| summed over the chamfer matching under lamv
    (A's things to B's nearest, and B's to A's), (|pa|, |pb|, Dv). Its product with lamv is cross_chamfer's distance;
    with the matching held fixed, its gradient in lamv is the exact one, since a minimum passes its gradient to the
    nearest thing (card 039's chamfer_a2b). Made a slice of pb at a time, without gradient."""
    torch = _torch()
    with torch.no_grad():
        EA, MA = Et[pa], mt[pa]
        a, m, Dv = EA.shape
        F = torch.zeros(a, len(pb), Dv, device=Et.device)
        ia = torch.arange(a, device=Et.device)
        # the fused kernel indexes (a m) x (b m) x Dv places: kept below 2^30, since past 2^31 it wrote out of
        # bounds (Xid 31 at 1/4 of tier 3's memory)
        ch = max(1, min(CHUNK, (1 << 30) // max(1, a * m * m * Dv)))
        for s0 in range(0, len(pb), ch):
            sl = pb[s0:s0 + ch]
            EB, MB = Et[sl], mt[sl]
            b = EB.shape[0]
            ib = torch.arange(b, device=Et.device)
            C = SP._wl1()(EA.reshape(a * m, Dv), EB.reshape(b * m, Dv), lamv).reshape(a, m, b, m)
            j = C.masked_fill(~MB[None, None], BIG).argmin(-1)                  # (a, m, b): B's thing nearest A's
            d = (EA[:, :, None, :] - EB[ib[None, None, :], j]).abs() * MA[:, :, None, None]
            F[:, s0:s0 + b] += d.sum(1)
            del j, d
            i = C.masked_fill(~MA[:, :, None, None], BIG).argmin(1)             # (a, b, m): A's thing nearest B's
            del C
            d = (EA[ia[:, None, None], i] - EB[None]).abs() * MB[None, :, :, None]
            F[:, s0:s0 + b] += d.sum(2)
            del i, d
    return F


class _Keys:
    """Every stored key of one fit, on the GPU, for comparing sampled query keys with all of them."""

    def __init__(self, S, Xfh, kpos, ulist):
        torch = _torch()
        f32 = dict(dtype=torch.float32, device="cuda")
        self.Xt = torch.as_tensor(Xfh, **f32)
        self.Et, self.mt = _sets(S, ulist)
        assert bool(self.mt.any(1).all()), "an empty view set: match_feats assumes none"
        self.kp = torch.as_tensor(np.asarray(kpos), device="cuda")
        self.n, self.D2 = self.Xt.shape
        self.Dv = self.Et.shape[2]
        self.allsets = torch.arange(self.Et.shape[0], device="cuda")

    def fh(self, q, lam_fh):
        """Front and held part of the distance from each query key to every key, (|q|, n)."""
        return (self.Xt[q][:, None] - self.Xt[None]).abs() @ lam_fh

    def view(self, q, lamv, keys=None):
        """View part of the distance from each query key to every key (or to keys), (|q|, n or |keys|); its gradient
        in lamv is exact."""
        torch = _torch()
        uq, inv = torch.unique(self.kp[q], return_inverse=True)
        if keys is None:
            F = match_feats(self.Et, self.mt, uq, self.allsets, lamv.detach())
            return (F @ lamv)[inv][:, self.kp]
        sb, kinv = torch.unique(self.kp[keys], return_inverse=True)
        F = match_feats(self.Et, self.mt, uq, sb, lamv.detach())
        return (F @ lamv)[inv][:, kinv]


def _eval_sample(n, gen):
    torch = _torch()
    if n <= NQ:
        return torch.arange(n, device="cuda")
    return torch.randperm(n, device="cuda", generator=gen)[:NQ]


def fit_lambda_queries(S, Xfh, kpos, ulist, Rk, group, steps=1500, lr=0.02):
    """card 039's fit_lambda_sets with BQ query keys sampled per step, each compared with every key: an unbiased
    estimate of the same leave-one-out objective and its gradient."""
    import time
    t_fit = time.monotonic()
    torch = _torch()
    f32 = dict(dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(85)
    K_ = _Keys(S, Xfh, kpos, ulist)
    n, D2 = K_.n, K_.D2
    Rt = torch.as_tensor(Rk, **f32)
    L = Rt.shape[1]
    g = torch.as_tensor(np.asarray(group), device="cuda")
    draw = lambda k: (torch.arange(n, device="cuda") if k >= n         # every key: the exact fit (a diagnosis)
                      else torch.randint(n, (k,), device="cuda", generator=gen))
    dist = lambda lam, q: K_.fh(q, lam[:D2]) + K_.view(q, lam[D2:])

    with torch.no_grad():
        d1 = torch.cat([dist(torch.ones(D2 + K_.Dv, **f32), q) for q in draw(min(n, 512)).split(QS)])
        med = float(d1[d1 > 0].median()) if bool((d1 > 0).any()) else 1.0

    def rows(theta, q):
        lam = torch.nn.functional.softplus(theta)
        k = torch.exp(-dist(lam, q)) * (g[q][:, None] != g[None]).float()
        P = (k @ Rt + 1.0 / L) / (k.sum(-1, keepdim=True) + 1.0)
        return (Rt[q] * torch.log(P)).sum(-1)

    qe = _eval_sample(n, gen)
    ll_eval = lambda th: float(torch.cat([rows(th, q) for q in qe.split(QS)]).mean())
    th = torch.nn.Parameter(torch.full((D2 + K_.Dv,), _isp(1.0 / max(med, 1e-6)), **f32))
    with torch.no_grad():
        l0 = ll_eval(th)
    if n > 2 and len(np.unique(np.asarray(group))) > 1:
        opt = torch.optim.Adam([th], lr=lr)
        sch = _schedule(opt, steps)
        for _ in range(steps):
            q = draw(BQ)
            opt.zero_grad()
            for qs in q.split(QS):                            # the step's queries a slice at a time, gradients summed
                (-rows(th, qs).sum() / len(q)).backward()
            opt.step()
            sch.step()
    with torch.no_grad():
        l1 = ll_eval(th)
        lam = torch.nn.functional.softplus(th).double().cpu().numpy()
    STATS["sampled_fits"].append({"fit": "card039 lambda (queries)", "keys": n, "sets": len(ulist),
                                  "loo_start": round(l0, 4), "loo_fitted": round(l1, 4), "eval_keys": len(qe),
                                  "seconds": round(time.monotonic() - t_fit, 1)})
    return lam, l0, l1


def fit_alpha_queries(kd, counts):
    """card 039's fit_alpha_dist with up to NQ query keys (every key at NQ or fewer), each against every key."""
    import time
    t_fit = time.monotonic()
    torch = _torch()
    f32 = dict(dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(851)
    K_ = _Keys(kd.S, kd.X, kd.kpos, kd.ulist)
    lam = torch.as_tensor(kd.lam, **f32)
    share = torch.as_tensor(counts / counts.sum(1, keepdims=True), **f32)
    q = _eval_sample(K_.n, gen)
    allk = torch.arange(K_.n, device="cuda")
    wns = []
    with torch.no_grad():
        for qq in q.split(QS):
            k = torch.exp(-(K_.fh(qq, lam[:K_.D2]) + K_.view(qq, lam[K_.D2:])))
            k = torch.where(k < VP.KMIN, torch.zeros_like(k), k) * (qq[:, None] != allk[None]).float()
            wns.append((k @ share).double().cpu().numpy())
    q = q.cpu().numpy()
    wn = np.concatenate(wns)
    cq = counts[q]
    qv = (wn + VP.PRIOR) / (wn.sum(1, keepdims=True) + 1.0)
    nq = cq.sum(1, keepdims=True)
    msk = cq > 0
    rest = np.maximum(cq - 1.0, 0.0)

    def ll(alpha):
        a = np.broadcast_to(alpha, nq.shape)
        P = (rest + a * qv) / np.maximum(nq - 1.0 + a, 1e-12)
        return float((cq[msk] * np.log(P[msk])).sum() / cq.sum())

    grid = np.exp(np.linspace(-9.0, 9.0, 181))
    vals = [ll(a) for a in grid]
    i = int(np.argmax(vals))
    STATS["sampled_fits"].append({"fit": "card039 alpha (queries)", "keys": K_.n, "query_keys": len(q),
                                  "alpha": float(grid[i]), "seconds": round(time.monotonic() - t_fit, 1)})
    return float(grid[i]), vals[i], ll(wn.sum(1, keepdims=True) + 1.0)


def fit_codes_queries(S, Xfh, kpos, ulist, counts, fid, hid, steps=2000, lr=0.03, sharp=8.0):
    """card 042's fit_codes with BQ query keys sampled per step, each compared with every key (the similar-tile
    level) and with every key of its code group (the same-tile level)."""
    import time
    t_fit = time.monotonic()
    torch = _torch()
    f32 = dict(dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(852)
    K_ = _Keys(S, Xfh, kpos, ulist)
    n, D2, Dv = K_.n, K_.D2, K_.Dv
    Ct = torch.as_tensor(counts, **f32)
    Rt = Ct / Ct.sum(1, keepdim=True)
    L = Ct.shape[1]
    _, code = np.unique(np.stack([fid, hid], 1), axis=0, return_inverse=True)
    code_t = torch.as_tensor(code.ravel(), device="cuda")
    allk = torch.arange(n, device="cuda")
    draw = lambda k: (torch.arange(n, device="cuda") if k >= n         # every key: the exact fit (a diagnosis)
                      else torch.randint(n, (k,), device="cuda", generator=gen))

    with torch.no_grad():
        q0 = draw(min(n, 512))
        dv = torch.cat([K_.view(q, torch.ones(Dv, **f32)) for q in q0.split(QS)])
        d1 = torch.cat([K_.fh(q, torch.ones(D2, **f32)) for q in q0.split(QS)]) + dv
        med = float(d1[d1 > 0].median())
        pv = dv[dv > 0]
        p1 = float(torch.quantile(pv[:2_000_000], 0.01)) if len(pv) else 1.0
    lam2_0, lam1_0 = 1.0 / med, sharp / p1
    th2 = torch.nn.Parameter(torch.full((D2 + Dv,), _isp(lam2_0), **f32))
    dl = torch.nn.Parameter(torch.full((Dv,), _isp(max(lam1_0 - lam2_0, 1e-3)), **f32))
    lb = torch.nn.Parameter(torch.tensor(0.0, **f32))

    def predict(q):
        lam2 = torch.nn.functional.softplus(th2)
        lam1v = lam2[D2:] + torch.nn.functional.softplus(dl)
        sel = torch.nonzero(torch.isin(code_t, code_t[q])).ravel()          # the keys of the queries' code groups
        same = (code_t[q][:, None] == code_t[sel][None]) & (q[:, None] != sel[None])
        k1 = torch.exp(-K_.view(q, lam1v, sel)) * same.float()
        N = k1 @ Ct[sel]
        k2 = torch.exp(-(K_.fh(q, lam2[:D2]) + K_.view(q, lam2[D2:]))) * (q[:, None] != allk[None]).float()
        s = (k2 @ Rt + 1.0 / L) / (k2.sum(-1, keepdim=True) + 1.0)
        b = torch.exp(lb)
        return (N + b * s) / (N.sum(-1, keepdim=True) + b)

    rows = lambda P, q: (Rt[q] * torch.log(P.clamp_min(1e-12))).sum(-1)
    with torch.no_grad():
        qe = _eval_sample(n, gen)
        l0 = float(torch.cat([rows(predict(q), q) for q in qe.split(QS)]).mean())
    opt = torch.optim.Adam([th2, dl, lb], lr=lr)
    sch = _schedule(opt, steps)
    for _ in range(steps):
        q = draw(BQ)
        opt.zero_grad()
        for qs in q.split(QS):                                # the step's queries a slice at a time, gradients summed
            (-rows(predict(qs), qs).sum() / len(q)).backward()
        opt.step()
        sch.step()
    with torch.no_grad():                                       # every key's prediction, for card 042's report
        Pall, rall = [], []
        for q in allk.split(QS):
            P = predict(q)
            Pall.append(P.cpu().numpy())
            rall.append(rows(P, q).cpu().numpy())
        lam2 = torch.nn.functional.softplus(th2).double().cpu().numpy()
        lam1v = lam2[D2:] + torch.nn.functional.softplus(dl).double().cpu().numpy()
    r = np.concatenate(rall)
    STATS["sampled_fits"].append({"fit": "card042 codes (queries)", "keys": n, "groups": int(code.max() + 1),
                                  "loo_start": round(l0, 4), "loo_fitted": round(float(r.mean()), 4),
                                  "seconds": round(time.monotonic() - t_fit, 1)})
    return dict(lam2=lam2, lam1v=lam1v, beta=float(np.exp(float(lb.detach()))), l0=l0, rows=r,
                P=np.concatenate(Pall))


# ---------------------------------------------------------------- card 039's lambda

def fit_lambda_sets(S, Xfh, kpos, ulist, Rk, group, steps=1500, lr=0.02):
    """card 039's fit_lambda_sets on sampled pairs (above M_EXACT keys)."""
    n = len(Xfh)
    if n <= M_EXACT:
        STATS["exact_fits"] += 1
        return _FIT_SETS(S, Xfh, kpos, ulist, Rk, group, steps, lr)
    if QUERIES:
        return fit_lambda_queries(S, Xfh, kpos, ulist, Rk, group, steps, lr)
    torch = _torch()
    f32 = dict(dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(85)
    Xt, Rt = torch.as_tensor(Xfh, **f32), torch.as_tensor(Rk, **f32)
    L, D2 = Rt.shape[1], Xt.shape[1]
    Et, mt = _sets(S, ulist)
    kp = torch.as_tensor(np.asarray(kpos), device="cuda")
    g = torch.as_tensor(np.asarray(group), device="cuda")
    draw = lambda k: torch.randint(n, (k,), device="cuda", generator=gen)
    scale = n / MC

    def dist(lam, q, c):
        return SP._wl1()(Xt[q], Xt[c], lam[:D2]) + cross_chamfer(Et, mt, kp[q], kp[c], lam[D2:])

    with torch.no_grad():
        d1 = dist(torch.ones(D2 + Et.shape[2], **f32), draw(B), draw(MC))
        med = float(d1[d1 > 0].median()) if bool((d1 > 0).any()) else 1.0

    def ll(theta, q, c):
        lam = torch.nn.functional.softplus(theta)
        k = torch.exp(-dist(lam, q, c)) * (g[q][:, None] != g[c][None]).float()
        P = (scale * (k @ Rt[c]) + 1.0 / L) / (scale * k.sum(-1, keepdim=True) + 1.0)
        return (Rt[q] * torch.log(P)).sum(-1).mean()

    th = torch.nn.Parameter(torch.full((D2 + Et.shape[2],), _isp(1.0 / max(med, 1e-6)), **f32))
    qe, ce = draw(B), draw(MC)                                  # a fixed sample to report the likelihood on
    with torch.no_grad():
        l0 = float(ll(th, qe, ce))
    opt = torch.optim.Adam([th], lr=lr)
    sch = _schedule(opt, steps)
    for _ in range(steps):
        loss = -ll(th, draw(B), draw(MC))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        l1 = float(ll(th, qe, ce))
        lam = torch.nn.functional.softplus(th).double().cpu().numpy()
    STATS["sampled_fits"].append({"fit": "card039 lambda", "keys": n, "sets": len(ulist), "loo_start": round(l0, 4),
                                  "loo_fitted": round(l1, 4)})
    return lam, l0, l1


# ---------------------------------------------------------------- card 039's alpha

class LazyDist:
    """Stands for card 039's n x n key distances; fit_alpha below samples them instead."""

    def __init__(self, kind):
        self.kind = kind


def key_dists(self, idx):
    if len(idx) > M_EXACT and len(idx) == len(self.keys):
        return LazyDist(self)
    return _KEY_DISTS(self, idx)


def fit_alpha_dist(Dist, counts, rounds=8):
    """card 039's fit_alpha_dist; given LazyDist, the neighbours' vote of each of rounds x B sampled keys over MC
    sampled others (scaled), and the same grid of alpha."""
    if not isinstance(Dist, LazyDist):
        return _FIT_ALPHA(Dist, counts)
    if QUERIES:
        return fit_alpha_queries(Dist.kind, counts)
    torch = _torch()
    kd = Dist.kind
    n = len(kd.keys)
    gen = torch.Generator(device="cuda").manual_seed(851)
    f32 = dict(dtype=torch.float32, device="cuda")
    Xt = torch.as_tensor(kd.X, **f32)
    Et, mt = _sets(kd.S, kd.ulist)
    kp = torch.as_tensor(np.asarray(kd.kpos), device="cuda")
    lam = torch.as_tensor(kd.lam, **f32)
    D2 = Xt.shape[1]
    share = torch.as_tensor(counts / counts.sum(1, keepdims=True), **f32)
    qs, wns = [], []
    with torch.no_grad():
        for _ in range(rounds):
            q = torch.randint(n, (B,), device="cuda", generator=gen)
            c = torch.randint(n, (MC,), device="cuda", generator=gen)
            d = SP._wl1()(Xt[q], Xt[c], lam[:D2]) + cross_chamfer(Et, mt, kp[q], kp[c], lam[D2:])
            k = torch.exp(-d)
            k = torch.where(k < VP.KMIN, torch.zeros_like(k), k) * (q[:, None] != c[None]).float()
            qs.append(q.cpu().numpy())
            wns.append(((n / MC) * (k @ share[c])).double().cpu().numpy())
    q = np.concatenate(qs)
    wn = np.concatenate(wns)
    cq = counts[q]
    qv = (wn + VP.PRIOR) / (wn.sum(1, keepdims=True) + 1.0)
    nq = cq.sum(1, keepdims=True)
    msk = cq > 0
    rest = np.maximum(cq - 1.0, 0.0)

    def ll(alpha):
        a = np.broadcast_to(alpha, nq.shape)
        P = (rest + a * qv) / np.maximum(nq - 1.0 + a, 1e-12)
        return float((cq[msk] * np.log(P[msk])).sum() / cq.sum())

    grid = np.exp(np.linspace(-9.0, 9.0, 181))
    vals = [ll(a) for a in grid]
    i = int(np.argmax(vals))
    STATS["sampled_fits"].append({"fit": "card039 alpha", "keys": n, "alpha": float(grid[i])})
    return float(grid[i]), vals[i], ll(wn.sum(1, keepdims=True) + 1.0)


# ---------------------------------------------------------------- card 042's two levels

def fit_codes_sampled(S, Xfh, kpos, ulist, counts, fid, hid, steps=2000, lr=0.03, sharp=8.0):
    """card 042's fit_codes on sampled pairs: the similar-tile level over MC keys drawn uniformly, the same-tile level
    over K keys drawn from each query's own code group, each scaled by its sampling fraction."""
    if QUERIES:
        return fit_codes_queries(S, Xfh, kpos, ulist, counts, fid, hid, steps, lr, sharp)
    torch = _torch()
    f32 = dict(dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(852)
    n, L = counts.shape
    Ct = torch.as_tensor(counts, **f32)
    Rt = Ct / Ct.sum(1, keepdim=True)
    Xt = torch.as_tensor(Xfh, **f32)
    D2 = Xt.shape[1]
    Et, mt = _sets(S, ulist)
    Dv = Et.shape[2]
    kp = torch.as_tensor(np.asarray(kpos), device="cuda")
    _, code = np.unique(np.stack([fid, hid], 1), axis=0, return_inverse=True)
    code = code.ravel()
    order = np.argsort(code, kind="stable")
    gsize = np.bincount(code)
    gstart = np.concatenate([[0], np.cumsum(gsize)[:-1]])
    code_t, order_t = torch.as_tensor(code, device="cuda"), torch.as_tensor(order, device="cuda")
    gsize_t, gstart_t = torch.as_tensor(gsize, device="cuda"), torch.as_tensor(gstart, device="cuda")
    draw = lambda k: torch.randint(n, (k,), device="cuda", generator=gen)

    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n) - np.repeat(gstart, gsize)
    pos_t = torch.as_tensor(pos, device="cuda")

    def draw_same(q):
        """K keys per query from its own code group, itself excluded, (B, K); and whether the group has another."""
        gq, sz = code_t[q], gsize_t[code_t[q]]
        r = (torch.rand(len(q), K, device="cuda", generator=gen) * (sz[:, None] - 1).clamp(min=1)).long()
        r = torch.minimum(r, (sz[:, None] - 2).clamp(min=0))
        r = r + (r >= pos_t[q][:, None]).long()                       # skip the query itself
        return order_t[gstart_t[gq][:, None] + r.clamp(max=sz[:, None] - 1)], (sz > 1)

    def parts(lam2, lam1v, q, c2, c1):
        """k1 (same codes, over c1 = (draws (B, K), has another)) and k2 (similar, over c2), with their scales."""
        c1, other = c1
        k2 = torch.exp(-(SP._wl1()(Xt[q], Xt[c2], lam2[:D2]) + cross_chamfer(Et, mt, kp[q], kp[c2], lam2[D2:])))
        k2 = k2 * (q[:, None] != c2[None]).float()
        k1 = torch.exp(-paired_chamfer(Et, mt, kp[q], kp[c1], lam1v)) * other[:, None].float()
        s1 = (gsize_t[code_t[q]][:, None].float() - 1.0) / K           # the group's other keys per draw
        return k1, s1, k2, n / len(c2)

    with torch.no_grad():
        q, c2 = draw(B), draw(MC)
        ones_v = torch.ones(Dv, **f32)
        dv = cross_chamfer(Et, mt, kp[q], kp[c2], ones_v)
        d1 = SP._wl1()(Xt[q], Xt[c2], torch.ones(D2, **f32)) + dv
        med = float(d1[d1 > 0].median())
        pv = dv[dv > 0]
        p1 = float(torch.quantile(pv[:2_000_000], 0.01)) if len(pv) else 1.0
    lam2_0, lam1_0 = 1.0 / med, sharp / p1
    th2 = torch.nn.Parameter(torch.full((D2 + Dv,), _isp(lam2_0), **f32))
    dl = torch.nn.Parameter(torch.full((Dv,), _isp(max(lam1_0 - lam2_0, 1e-3)), **f32))
    lb = torch.nn.Parameter(torch.tensor(0.0, **f32))

    def predict(q, c2, c1):
        lam2 = torch.nn.functional.softplus(th2)
        lam1v = lam2[D2:] + torch.nn.functional.softplus(dl)
        k1, s1, k2, s2 = parts(lam2, lam1v, q, c2, c1)
        N = s1 * (k1[:, :, None] * Ct[c1[0]]).sum(1)
        s = (s2 * (k2 @ Rt[c2]) + 1.0 / L) / (s2 * k2.sum(-1, keepdim=True) + 1.0)
        b = torch.exp(lb)
        return (N + b * s) / (N.sum(-1, keepdim=True) + b)

    rows = lambda P, q: (Rt[q] * torch.log(P.clamp_min(1e-12))).sum(-1)
    with torch.no_grad():
        qe = draw(B)
        ev = (qe, draw(MC), draw_same(qe))
        l0 = float(rows(predict(*ev), qe).mean())
    opt = torch.optim.Adam([th2, dl, lb], lr=lr)
    sch = _schedule(opt, steps)
    for _ in range(steps):
        q = draw(B)
        loss = -rows(predict(q, draw(MC), draw_same(q)), q).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():                                       # every key's prediction, for card 042's report
        Pall, rall = [], []
        for i in range(0, n, B):
            q = torch.arange(i, min(n, i + B), device="cuda")
            P = predict(q, draw(MC), draw_same(q))
            Pall.append(P.cpu().numpy())
            rall.append(rows(P, q).cpu().numpy())
        lam2 = torch.nn.functional.softplus(th2).double().cpu().numpy()
        lam1v = lam2[D2:] + torch.nn.functional.softplus(dl).double().cpu().numpy()
    r = np.concatenate(rall)
    STATS["sampled_fits"].append({"fit": "card042 codes", "keys": n, "groups": int(len(gsize)),
                                  "loo_start": round(l0, 4), "loo_fitted": round(float(r.mean()), 4)})
    return dict(lam2=lam2, lam1v=lam1v, beta=float(np.exp(float(lb.detach()))), l0=l0, rows=r,
                P=np.concatenate(Pall))


# ---------------------------------------------------------------- the gate: a kind's fitted weights, evaluated exactly

def exact_eval(kd, chunk=32):
    """With kd's fitted weights fixed: card 039's and card 042's leave-one-out objectives over every pair of keys (as
    their fits score them), and each key's predicted category at card 042's level. For the gate (a few thousand
    keys); quadratic."""
    torch = _torch()
    f32 = dict(dtype=torch.float32, device="cuda")
    n = len(kd.keys)
    D2 = kd.X.shape[1]
    Xt = torch.as_tensor(kd.X, **f32)
    Et, mt = _sets(kd.S, kd.ulist)
    kp = torch.as_tensor(np.asarray(kd.kpos), device="cuda")
    Rk = torch.as_tensor(kd.counts / kd.counts.sum(1, keepdims=True), **f32)
    Ct = torch.as_tensor(kd.lc, **f32)
    Rt = Ct / Ct.sum(1, keepdim=True)
    L1, L2 = Rk.shape[1], Rt.shape[1]
    g = torch.as_tensor(np.asarray(kd.group), device="cuda")
    _, code = np.unique(np.stack([kd.fid, kd.hid], 1), axis=0, return_inverse=True)
    code_t = torch.as_tensor(code.ravel(), device="cuda")
    lam = torch.as_tensor(kd.lam, **f32)
    lam2 = torch.as_tensor(kd.lam2, **f32)
    lam1v = torch.as_tensor(kd.lam1[D2:], **f32)
    beta = float(kd.beta)
    M = np.zeros((len(kd.lcat), kd.counts.shape[1]))
    M[np.arange(len(kd.lcat)), kd.lcat] = 1
    allk = torch.arange(n, device="cuda")
    l39, l42, cat = [], [], []
    with torch.no_grad():
        for i in range(0, n, chunk):
            q = torch.arange(i, min(n, i + chunk), device="cuda")
            self_ = (q[:, None] != allk[None]).float()
            k = torch.exp(-(SP._wl1()(Xt[q], Xt, lam[:D2]) + cross_chamfer(Et, mt, kp[q], kp, lam[D2:])))
            k = k * (g[q][:, None] != g[None]).float()
            P = (k @ Rk + 1.0 / L1) / (k.sum(-1, keepdim=True) + 1.0)
            l39.append((Rk[q] * torch.log(P)).sum(-1).cpu().numpy())
            k2 = torch.exp(-(SP._wl1()(Xt[q], Xt, lam2[:D2]) + cross_chamfer(Et, mt, kp[q], kp, lam2[D2:]))) * self_
            k1 = torch.exp(-cross_chamfer(Et, mt, kp[q], kp, lam1v)) * self_ * (code_t[q][:, None] == code_t[None]).float()
            N = k1 @ Ct
            s2 = (k2 @ Rt + 1.0 / L2) / (k2.sum(-1, keepdim=True) + 1.0)
            P2 = (N + beta * s2) / (N.sum(-1, keepdim=True) + beta)
            l42.append((Rt[q] * torch.log(P2.clamp_min(1e-12))).sum(-1).cpu().numpy())
            cat.append((P2.double().cpu().numpy() @ M).argmax(1))
    return {"loo_card039": float(np.concatenate(l39).mean()), "loo_card042": float(np.concatenate(l42).mean()),
            "category": np.concatenate(cat), "alpha": float(kd.alpha), "beta": beta}


def install():
    """Replace the three fits (each keeps version 18's path at M_EXACT keys or fewer)."""
    global _FIT_SETS, _FIT_ALPHA, _KEY_DISTS
    if getattr(SP, "_card085", False):
        return
    _FIT_SETS, _FIT_ALPHA, _KEY_DISTS = SP.fit_lambda_sets, SP.fit_alpha_dist, SP.SlotKind.key_dists
    SP.fit_lambda_sets = fit_lambda_sets
    SP.fit_alpha_dist = fit_alpha_dist
    SP.SlotKind.key_dists = key_dists
    CR.SAMPLED_FIT = sys.modules[__name__]
    SP._card085 = True
