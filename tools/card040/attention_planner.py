"""Card 040: relations from attention over the things present, as a part of recall's key.

Card 039's planner (tools/card039/slot_planner.py, imported unchanged) keys recall for pick up, toggle and drop on
the thing in front, the thing held and the set of things in view. Here the key also holds relations between those
things, computed like a transformer's attention scores and learned with recall's other weights:

  tokens     a thing is one token (its encoder vector) or several (its parts: the encoder's first-layer features
             at each pixel, or the oracle's four one-hot pieces)
  score      per head h, between a token p of thing a and a token q of thing b:
               dot   (Wq_h p) . (Wk_h q) / sqrt(r)            (a transformer's score)
               dist  -|W_h p - W_h q|^2 / sqrt(r)             (the same, with one projection and the norms)
  relation   r_h(a, b): over a's tokens, the mean of the log-mean-exp of their scores with b's tokens (attention
             pooled into one number per head), averaged with r_h(b, a)
  key part   per head: front-held; front-in view and held-in view (the best match in view); in view-in view (the
             best pair). Recall's distance adds sum lambda_r |R - R'|.

Wq, Wk (or W) and lambda_r are fitted per action with recall's other weights by card 039's objective (leave out
each (front, held) pair, predict its outcomes from the other keys). Nothing names a dimension, part or attribute,
and nothing is stored per thing: relations are computed for any two things. Only the scores pass on (a relational
bottleneck): no thing's features are mixed into another's.

  python tools/card040/attention_planner.py --check --arm B --tokens parts --score dist --seeds 400-400 \\
      --worlds key,either,both --layouts 10 --out runs/040_check_B_parts_dist.json
"""
import dataclasses
import inspect
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card039"))
import slot_planner as SP                                       # noqa: E402

VP = SP.VP
CFG = {"tokens": "parts", "score": "dist", "heads": 2, "rank": 32, "steps": 1500, "bound": False, "holdout": "pair", "held_identity": True, "roles": "pairs", "queries": 4}
PARTS_DIR = VP.ROOT / "runs" / "040_parts"
NREL = 4                                   # front-held, front-in view, held-in view, in view-in view
REL_NAMES = ("front-held", "front-view", "held-view", "view-view")


# ---------------------------------------------------------------- tokens

class Tokens:
    """Each handle's tokens, (m, d). Vector mode: the handle's vector. Parts mode: arm B's first-layer features at
    the 64 pixels of the tile (a handle no tile has: the decoder's picture of its vector), or the oracle's four
    pieces (each piece's numbers, the rest zero)."""

    def __init__(self, S, mode, arm, parts, net=None):
        self.S, self.mode, self.arm, self.parts, self.net = S, mode, arm, parts, net
        self.code = {}
        for c, h in enumerate(S.hof.tolist()):
            self.code.setdefault(int(h), c)
        self.cache = {}

    def __call__(self, h):
        h = int(h)
        t = self.cache.get(h)
        if t is None:
            v = self.S.arr[h]
            if self.mode == "vector":
                t = v[None].copy()
            elif self.arm == "1":
                t = np.zeros((len(self.parts), len(v)))
                for k, sl in enumerate(self.parts):
                    t[k, sl] = v[sl]
            else:
                t = self.net.parts(self.code.get(h), v)
            self.cache[h] = t
        return t


class PartNet:
    """Arm B's encoder and decoder, retrained exactly as card 038 trained them (the saved encoders kept only the
    vectors; retraining reproduces them to 0.0, checked), kept under runs/040_parts/."""

    def __init__(self, seed, cache, groups, dev):
        import torch
        self.torch, self.dev = torch, dev
        f = PARTS_DIR / f"encB_s{seed}.pt"
        enc, dec = self._nets()
        if f.exists():
            st = torch.load(f, map_location=dev)
            enc.load_state_dict(st["enc"])
            dec.load_state_dict(st["dec"])
        else:
            src = inspect.getsource(VP.train_b).replace("return z_all, theta.detach().cpu().numpy(), rep",
                                                        "return z_all, enc, dec")
            ns = dict(VP.__dict__)
            exec(src, ns)
            z, enc, dec = ns["train_b"](cache["tiles"], cache["pairs"], groups, VP.MU, seed, dev)
            saved = np.load(VP.ENC_B / f"mu{VP.MU}_s{seed}.npz")["z"]
            assert np.abs(z - saved).max() < 1e-6, "retrained encoder differs from the saved one"
            PARTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()}, f)
        self.enc, self.dec = enc.eval(), dec.eval()
        self.x = torch.as_tensor(VP.CO.TILES / 255.0, dtype=torch.float32, device=dev).permute(0, 3, 1, 2)

    def _nets(self):
        nn = self.torch.nn
        enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                            nn.ReLU(), nn.Flatten(), nn.Linear(32 * 16, 32)).to(self.dev)
        dec = nn.Sequential(nn.Linear(32, 32 * 16), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)),
                            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()).to(self.dev)
        return enc, dec

    def parts(self, code, v):
        torch = self.torch
        with torch.no_grad():
            if code is not None:
                x = self.x[code:code + 1]
            else:
                x = self.dec(torch.as_tensor(v[None], dtype=torch.float32, device=self.dev))
            f = self.enc[:2](x)[0]                                           # (32, 8, 8)
        return f.permute(1, 2, 0).reshape(-1, f.shape[0]).double().cpu().numpy()


TOK = None


# ---------------------------------------------------------------- attention scores pooled into relations

def pair_matrix(torch, X, Wq, Wk, score):
    """Relations among the things whose tokens are X (n, m, d): (H, n, n)."""
    H, r, _ = Wq.shape
    n, m, _ = X.shape
    Q = torch.einsum("hrd,nmd->hnmr", Wq, X).reshape(H, n * m, r)
    K = Q if score == "dist" else torch.einsum("hrd,nmd->hnmr", Wk, X).reshape(H, n * m, r)
    G = Q @ K.transpose(1, 2)
    if score == "dist":
        sq = (Q * Q).sum(-1)
        G = -(sq[:, :, None] + sq[:, None, :] - 2 * G).clamp_min(0.0)
    G = (G / r ** 0.5).reshape(H, n, m, n, m)
    lme = torch.logsumexp(G, -1) - float(np.log(m))                          # (H, n, m, n): a's token p vs b
    a2b = lme.mean(2)                                                         # (H, n, n)
    M = (a2b + a2b.transpose(1, 2)) / 2
    if CFG["bound"]:                    # a similarity in (0, 1]: every clear mismatch near 0, whatever the values
        M = torch.exp(M) if score == "dist" else torch.sigmoid(M)
    return M


def key_relations(torch, M, fi, hi, members, msk, kp):
    """Each key's relations (n, NREL * H) from M (H, nu, nu): front-held; front and held against the best match in
    view; the best pair in view. members (ns, mx) index things, msk marks real ones; kp maps keys to sets."""
    H = M.shape[0]
    neg = torch.finfo(M.dtype).min
    fh = M[:, fi, hi]                                                         # (H, n)
    mem, mk = members[kp], msk[kp]                                            # (n, mx)
    fv = torch.gather(M[:, fi], 2, mem[None].expand(H, -1, -1)).masked_fill(~mk[None], neg).amax(-1)
    hv = torch.gather(M[:, hi], 2, mem[None].expand(H, -1, -1)).masked_fill(~mk[None], neg).amax(-1)
    Mv = M[:, members[:, :, None], members[:, None, :]]                       # (H, ns, mx, mx)
    eye = torch.eye(members.shape[1], dtype=torch.bool, device=M.device)
    ok = msk[:, :, None] & msk[:, None, :] & ~eye[None]
    vv = Mv.masked_fill(~ok[None], neg).amax(-1).amax(-1)                     # (H, ns)
    vv = torch.where(ok.any(-1).any(-1)[None], vv, torch.zeros_like(vv))[:, kp]
    fv = torch.where(mk.any(-1)[None], fv, torch.zeros_like(fv))
    hv = torch.where(mk.any(-1)[None], hv, torch.zeros_like(hv))
    return torch.cat([fh, fv, hv, vv], 0).T                                   # (n, NREL * H)


# ---------------------------------------------------------------- roles as tokens

NTAG = 3                                    # where a thing is: ahead, in the hand, elsewhere in view
PAIRTYPES = [(a, b) for a in range(NTAG) for b in range(a, NTAG)]


def token_batch(torch, keys, dev):
    """Each key's tokens: (n, T, D + NTAG) features and (n, T) mask. A token is a thing's vector with its place:
    the thing ahead, the thing in the hand, each distinct thing elsewhere in view."""
    S = TOK.S
    rows = []
    for f, h, s in keys:
        toks = [(int(f), 0), (int(h), 1)] + [(int(j), 2) for j in SP.SETS[int(s)]]
        rows.append(toks)
    T = max(len(r) for r in rows)
    D = S.arr.shape[1]
    X = np.zeros((len(rows), T, D + NTAG))
    M = np.zeros((len(rows), T), bool)
    for i, r in enumerate(rows):
        for t, (hd, tag) in enumerate(r):
            X[i, t, :D] = S.arr[hd]
            X[i, t, D + tag] = 1.0
            M[i, t] = True
    return torch.as_tensor(X, dtype=torch.float64, device=dev), torch.as_tensor(M, device=dev)


def token_relations(torch, X, M, Wq, Wk, A, B, score):
    """Attention over every pair of tokens. Per head h, a bounded score s_h(i, j); per query g, a softmax over the
    pairs of (A_g . the pair's places + B_g s_h(i, j)) picks the pairs that matter, and the query's relation is
    the weighted score. Returns (n, G * H)."""
    H, r, _ = Wq.shape
    n, T, _ = X.shape
    Q = torch.einsum("hrd,ntd->hntr", Wq, X)
    K = Q if score == "dist" else torch.einsum("hrd,ntd->hntr", Wk, X)
    G_ = Q @ K.transpose(-1, -2)                                              # (H, n, T, T)
    if score == "dist":
        sq = (Q * Q).sum(-1)
        Sc = torch.exp(-(sq[..., :, None] + sq[..., None, :] - 2 * G_).clamp_min(0.0) / r ** 0.5)
    else:
        Sc = torch.sigmoid(G_ / r ** 0.5)
    return pool_tokens(torch, Sc, X[..., -NTAG:], M, A, B)


def token_index(keys, ui):
    """Parts mode: each key's tokens as indices into the things ui maps, with places and mask."""
    rows = [[(ui[int(f)], 0), (ui[int(h)], 1)] + [(ui[int(j)], 2) for j in SP.SETS[int(s)]] for f, h, s in keys]
    T = max(len(r) for r in rows)
    idx = np.zeros((len(rows), T), np.int64)
    tag = np.zeros((len(rows), T, NTAG))
    msk = np.zeros((len(rows), T), bool)
    for i, r in enumerate(rows):
        for t, (u, g) in enumerate(r):
            idx[i, t], tag[i, t, g], msk[i, t] = u, 1.0, True
    return idx, tag, msk


def pool_tokens(torch, Sc, tag, M, A, B):
    """Per query g, a softmax over the pairs of tokens of (A_g . the pair's places + B_g s_h(i, j)); the query's
    relation is the weighted score. Sc (H, n, T, T), tag (n, T, NTAG), M (n, T). Returns (n, G * H)."""
    n, T = M.shape
    pt = torch.stack([tag[..., a][:, :, None] * tag[..., b][:, None, :] + (tag[..., b][:, :, None] * tag[..., a][:, None, :] if a != b else 0)
                      for a, b in PAIRTYPES], -1).clamp_max(1.0)              # (n, T, T, P)
    ok = M[:, :, None] & M[:, None, :] & ~torch.eye(T, dtype=torch.bool, device=M.device)[None]
    ok = ok & torch.triu(torch.ones(T, T, dtype=torch.bool, device=M.device), 1)[None]
    logit = torch.einsum("ntup,gp->gntu", pt, A)[:, None] + B[:, None, None, None, None] * Sc[None]   # (G, H, n, T, T)
    logit = logit.masked_fill(~ok[None, None], float("-inf"))
    w = torch.softmax(logit.reshape(*logit.shape[:3], -1), -1).reshape(logit.shape)
    R = (w * Sc[None]).sum((-1, -2))                                          # (G, H, n)
    return R.permute(2, 0, 1).reshape(n, -1)


class Rel:
    """One action's fitted relation weights, and relations for any keys (cached per pair of things)."""

    def __init__(self, Wq, Wk, lamR, score, A=None, B=None):
        import torch
        torch.set_num_threads(1)
        self.torch, self.score = torch, score
        self.Wq, self.Wk, self.lamR, self.A, self.B = Wq, Wk, lamR, A, B
        self.pc, self.kc = {}, {}

    def many(self, keys):
        """Relations for many keys (n, len(lamR)), cached per key."""
        keys = [tuple(int(v) for v in k) for k in keys]
        if self.A is None:
            return np.stack([self.of(k) for k in keys])
        todo = [k for k in dict.fromkeys(keys) if k not in self.kc]
        if todo:
            torch = self.torch
            dev = self.Wq.device
            with torch.no_grad():
                if CFG["tokens"] == "parts":        # each thing's parts; the score per pair of things, then pooled
                    hs = sorted({int(h) for f, hd, s_ in todo for h in [f, hd] + SP.SETS[int(s_)].tolist()})
                    pc = self.pairs(hs)
                    ui = {h: i for i, h in enumerate(hs)}
                    Mh = torch.as_tensor(np.array([[pc[(a, b)] for b in hs] for a in hs]).transpose(2, 0, 1),
                                         dtype=torch.float64, device=dev)                       # (H, nh, nh)
                    idx, tag, msk = token_index(todo, ui)
                    it = torch.as_tensor(idx, device=dev)
                    Sc = Mh[:, it[:, :, None], it[:, None, :]]
                    R = pool_tokens(torch, Sc, torch.as_tensor(tag, dtype=torch.float64, device=dev),
                                    torch.as_tensor(msk, device=dev), self.A, self.B).cpu().numpy()
                else:
                    X, M = token_batch(torch, todo, dev)
                    R = token_relations(torch, X, M, self.Wq, self.Wk, self.A, self.B, self.score).cpu().numpy()
            for k, r in zip(todo, R):
                self.kc[k] = r
        return np.stack([self.kc[k] for k in keys])

    def pairs(self, hs):
        """r_h for every pair among the handles hs (cached): dict (a, b) -> (H,)."""
        torch = self.torch
        hs = sorted({int(h) for h in hs})
        todo = [h for h in hs if any((h, g) not in self.pc for g in hs)]
        if todo:
            X = torch.as_tensor(np.stack([TOK(h) for h in hs]), dtype=torch.float64, device=self.Wq.device)
            with torch.no_grad():
                M = pair_matrix(torch, X, self.Wq, self.Wk, self.score).cpu().numpy()
            for i, a in enumerate(hs):
                for j, b in enumerate(hs):
                    self.pc[(a, b)] = M[:, i, j]
        return self.pc

    def of(self, key):
        if self.A is not None:
            return self.many([key])[0]
        f, h, s = int(key[0]), int(key[1]), int(key[2])
        mem = SP.SETS[s].tolist()
        pc = self.pairs([f, h] + mem)
        H = self.Wq.shape[0]
        fh = pc[(f, h)]
        fv = np.max([pc[(f, j)] for j in mem], 0) if mem else np.zeros(H)
        hv = np.max([pc[(h, j)] for j in mem], 0) if mem else np.zeros(H)
        pr = [pc[(a, b)] for a in mem for b in mem if a != b]
        vv = np.max(pr, 0) if pr else np.zeros(H)
        return np.concatenate([fh, fv, hv, vv])


def fit_with_relations(kd, S, Xfh, kpos, ulist, Rk, group, steps=None, lr=0.02):
    """Card 039's fit (fit_lambda_sets) with the relation part added to the distance; the projections and lambda_r
    are fitted with the other weights. Sets kd.rel; returns card 039's lambda and the objective before and after."""
    import torch
    steps = steps or CFG["steps"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f64 = dict(dtype=torch.float64, device=dev)
    Xt = torch.as_tensor(Xfh, **f64)
    Rt = torch.as_tensor(Rk, **f64)
    n, L = Rt.shape
    Dfh = torch.abs(Xt[:, None] - Xt[None])
    m = max(len(SP.SETS[s]) for s in ulist)
    ns, Dv = len(ulist), S.arr.shape[1]
    E = np.zeros((ns, m, Dv))
    mk = np.zeros((ns, m), bool)
    for i, s in enumerate(ulist):
        E[i, :len(SP.SETS[s])] = S.arr[SP.SETS[s]]
        mk[i, :len(SP.SETS[s])] = True
    Et, mt = torch.as_tensor(E, **f64), torch.as_tensor(mk, device=dev)
    diff = torch.abs(Et[:, None, :, None, :] - Et[None, :, None, :, :])
    kp = torch.as_tensor(np.asarray(kpos), device=dev)

    def setdist(lamv):
        Wd = (diff @ lamv).masked_fill(~mt[None, :, None, :], float("inf"))
        a2b = Wd.min(-1).values.masked_fill(~mt[:, None, :], 0.0).sum(-1)
        return a2b + a2b.T

    # the relation part: every thing among the keys, its tokens, and each key's indices into them
    U = sorted({h for k in kd.keys for h in k[:2]} | {int(h) for s in ulist for h in SP.SETS[s]})
    ui = {h: i for i, h in enumerate(U)}
    X = torch.as_tensor(np.stack([TOK(h) for h in U]), **f64)
    fi = torch.as_tensor([ui[k[0]] for k in kd.keys], device=dev)
    hi = torch.as_tensor([ui[k[1]] for k in kd.keys], device=dev)
    members = np.zeros((ns, m), np.int64)
    for i, s in enumerate(ulist):
        members[i, :len(SP.SETS[s])] = [ui[int(h)] for h in SP.SETS[s]]
    members, mkt = torch.as_tensor(members, device=dev), mt
    H, r, d = CFG["heads"], CFG["rank"], X.shape[-1]
    g = torch.Generator(device="cpu").manual_seed(0)
    Wq = torch.nn.Parameter((torch.randn(H, r, d, generator=g, dtype=torch.float64) / d ** 0.5).to(dev))
    Wk = Wq if CFG["score"] == "dist" else \
        torch.nn.Parameter((torch.randn(H, r, d, generator=g, dtype=torch.float64) / d ** 0.5).to(dev))

    tokens_mode = CFG["roles"] == "tokens"
    parts_tokens = tokens_mode and CFG["tokens"] == "parts"
    if tokens_mode and not parts_tokens:    # every thing a token tagged with its place; queries pick the pairs
        Wq = torch.nn.Parameter((torch.randn(H, r, d + NTAG, generator=g, dtype=torch.float64) / d ** 0.5).to(dev))
        Wk = Wq if CFG["score"] == "dist" else \
            torch.nn.Parameter((torch.randn(H, r, d + NTAG, generator=g, dtype=torch.float64) / d ** 0.5).to(dev))
    if tokens_mode:
        Gq = CFG["queries"]
        A = torch.nn.Parameter((0.1 * torch.randn(Gq, len(PAIRTYPES), generator=g, dtype=torch.float64)).to(dev))
        Bq = torch.nn.Parameter(torch.ones(Gq, dtype=torch.float64, device=dev))
        if parts_tokens:                # things' parts scored as in pairs mode, then every pair of tokens pooled
            idx, tag, msk = token_index(kd.keys, ui)
            Ik = torch.as_tensor(idx, device=dev)
            Tg, Mk_ = torch.as_tensor(tag, **f64), torch.as_tensor(msk, device=dev)
        else:
            Xk, Mk_ = token_batch(torch, kd.keys, dev)

    def relations():
        if parts_tokens:
            M = pair_matrix(torch, X, Wq, Wk, CFG["score"])
            return pool_tokens(torch, M[:, Ik[:, :, None], Ik[:, None, :]], Tg, Mk_, A, Bq)
        if tokens_mode:
            return token_relations(torch, Xk, Mk_, Wq, Wk, A, Bq, CFG["score"])
        return key_relations(torch, pair_matrix(torch, X, Wq, Wk, CFG["score"]), fi, hi, members, mkt, kp)

    D2 = Xfh.shape[1]

    hmask = torch.ones(D2 + Dv, **f64)
    if not CFG["held_identity"]:        # the held thing counts only through its relations (a strict bottleneck)
        hmask[D2 // 2:D2] = 0.0

    def base(lam):
        lam = lam * hmask
        return Dfh @ lam[:D2] + setdist(lam[D2:])[kp][:, kp]

    with torch.no_grad():
        d1 = base(torch.ones(D2 + Dv, **f64))
        med = float(d1[d1 > 0].median()) if bool((d1 > 0).any()) else 1.0
        R0 = relations()
        dr = torch.abs(R0[:, None] - R0[None]).sum(-1)
        medr = float(dr[dr > 0].median()) if bool((dr > 0).any()) else 1.0
    lam0 = 1.0 / max(med, 1e-6)
    g_ = torch.as_tensor(np.asarray(group), device=dev)
    keep = (g_[:, None] != g_[None]).double()
    if CFG["holdout"] == "front":       # leave out every key with the same thing in front: predict a thing never tried
        fr = torch.as_tensor([k[0] for k in kd.keys], device=dev)
        keep = keep * (fr[:, None] != fr[None]).double()
    th = torch.nn.Parameter(torch.full((D2 + Dv,), float(np.log(np.expm1(lam0))), **f64))
    nR = R0.shape[1]
    thr = torch.nn.Parameter(torch.full((nR,), float(np.log(np.expm1(1.0 / max(medr, 1e-6) / 2))), **f64))
    params = [th, thr, Wq] + ([] if Wk is Wq else [Wk]) + ([A, Bq] if tokens_mode else [])

    def ll():
        lam, lr_ = torch.nn.functional.softplus(th), torch.nn.functional.softplus(thr)
        R = relations()
        dist = base(lam) + torch.abs(R[:, None] - R[None]) @ lr_
        k = torch.exp(-dist) * keep
        P = (k @ Rt + 1.0 / L) / (k.sum(-1, keepdim=True) + 1.0)
        return (Rt * torch.log(P)).sum(-1).mean()

    with torch.no_grad():
        l0 = float(ll())
    if n > 2 and bool(keep.any()):
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            loss = -ll()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        l1 = float(ll())
        lam = (torch.nn.functional.softplus(th) * hmask).cpu().numpy()
        lamR = torch.nn.functional.softplus(thr).cpu().numpy()
    cpu = lambda t: t.detach().cpu()    # predictions run in the acting workers, which cannot use the GPU
    kd.rel = Rel(cpu(Wq), cpu(Wk), lamR, CFG["score"], cpu(A) if tokens_mode else None, cpu(Bq) if tokens_mode else None)
    return lam, l0, l1


_fit039 = SP.fit_lambda_sets


class RelKind(SP.SlotKind):
    """Card 039's SlotKind with the relation part in the key."""
    building = None

    def __init__(self, *a, **kw):
        RelKind.building = self
        SP.fit_lambda_sets = lambda *b, **k: fit_with_relations(RelKind.building, *b, **k)
        try:
            super().__init__(*a, **kw)
        finally:
            SP.fit_lambda_sets = _fit039
        self.n0 = len(self.keys)
        rel = self.rel_part()
        if rel.A is None:
            lr_ = rel.lamR.reshape(NREL, -1)
            self.report["relation_lambda"] = {nm: [round(float(x), 3) for x in row] for nm, row in zip(REL_NAMES, lr_)}
        else:                           # per query: its weight per head, and the pair of places it leans to most
            lr_ = rel.lamR.reshape(CFG["queries"], -1)
            names = ("ahead", "hand", "view")
            A_ = rel.A.cpu().numpy()
            self.report["relation_lambda"] = {
                f"q{g} " + "-".join(names[x] for x in PAIRTYPES[int(A_[g].argmax())]): [round(float(x), 3) for x in lr_[g]]
                for g in range(len(lr_))}

    def rel_part(self):
        if not hasattr(self, "rel"):                                         # too few pairs to fit: unfitted
            import torch
            H, r = CFG["heads"], CFG["rank"]
            d = TOK(self.keys[0][0]).shape[-1]
            dev = "cpu"
            g = torch.Generator(device="cpu").manual_seed(0)
            if CFG["roles"] == "tokens":
                W = (torch.randn(H, r, d + NTAG, generator=g, dtype=torch.float64) / d ** 0.5).to(dev)
                A = torch.zeros(CFG["queries"], len(PAIRTYPES), dtype=torch.float64, device=dev)
                B = torch.ones(CFG["queries"], dtype=torch.float64, device=dev)
                self.rel = Rel(W, W, np.zeros(CFG["queries"] * H), CFG["score"], A, B)
            else:
                W = (torch.randn(H, r, d, generator=g, dtype=torch.float64) / d ** 0.5).to(dev)
                self.rel = Rel(W, W, np.zeros(NREL * H), CFG["score"])
        if not hasattr(self, "Rstore") or len(self.Rstore) < len(self.keys):
            have = 0 if not hasattr(self, "Rstore") else len(self.Rstore)
            new = self.rel.many(self.keys[have:])
            self.Rstore = new if have == 0 else np.concatenate([self.Rstore, new])
        return self.rel

    def dists(self, qs):
        out = super().dists(qs)
        rel = self.rel_part()
        Rq = rel.many(qs)
        return out + np.abs(Rq[:, None] - self.Rstore[None, :len(self.keys)]) @ rel.lamR

    def reset(self):
        super().reset()
        if hasattr(self, "Rstore"):
            self.Rstore = self.Rstore[:self.n0]


def kind(name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
    cls = RelKind if nkey == 3 else SP._Kind
    return cls(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)


ARM = {"arm": "B"}


class TokWorld(SP.SlotWorld):
    """Card 039's world, with the tokens set up from its vectors before the kinds are fitted (acting)."""

    def __init__(self, S, parts, *a, **kw):
        global TOK
        TOK = Tokens(S, CFG["tokens"], ARM["arm"], parts, None)
        super().__init__(S, parts, *a, **kw)


def install():
    VP.Kind, VP.World = kind, TokWorld


# ---------------------------------------------------------------- the check before acting

def colour(name):
    return name.split()[-1] if name.split() else ""


def check(arm, seeds, worlds, n, out, log):
    """Toggling the closed door, from memory before acting: as the view is (holding nothing), after each imagined
    pickup of a key in view (holding it), and after turning the switch on; against the world's rule (the
    evaluator's). In the key world also on new layouts whose door and its key have a colour never seen
    (yellow, purple: card 031's world (c))."""
    global TOK
    T, F, ld = VP.T, VP.F, VP.ld
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pickle.loads(T.DATA.read_bytes())
    cache = pickle.loads(T.MEMORY.read_bytes())
    groups = T.use_groups(cache["mem"], dev)
    res = {"note": "Card 040 check, tools/card040/attention_planner.py --check", "arm": arm, "cfg": dict(CFG),
           "seeds": []}
    for seed in seeds:
        z, parts, _ = VP.vectors_of(arm, seed, cache["tiles"], cache["pairs"], groups, dev, log)
        S = VP.Store(z)
        lut = np.zeros(256, np.uint8)
        lut[:len(S.hof)] = S.hof
        VP.Kd.APP = lut
        net = PartNet(seed, cache, groups, dev) if (CFG["tokens"] == "parts" and arm != "1") else None
        TOK = Tokens(S, CFG["tokens"], arm, parts, net)
        rs = {"seed": seed, "worlds": {}}
        for world in worlds:
            t0 = time.monotonic()
            W = SP.SlotWorld(S, parts, d["data"][world], dev, lambda m: None)
            VP.WORLD, F.M = W, W.M
            nm = lambda h: W.judge.name(int(h))
            kd = W.kinds[VP.TOG]
            sets = {"familiar": list(d["data"][world]["test"][:n])}
            if world == "key":
                for col in ("yellow", "purple"):
                    rng = np.random.default_rng(777)
                    sets[f"new colour {col}"] = [dataclasses.replace(VP.CO._MAKE(8, "key", rng), door_colour=col)
                                                 for _ in range(n)]
            wr = {}
            for sname, lays in sets.items():
                tally, wrong = {}, []
                for li, lay in enumerate(lays):
                    W.reset()
                    pl = VP.VPlan(W)
                    f, st, _, _ = pl.observe(None, VP.see(lay, ld.start_state(lay)))
                    things = np.unique(f[:VP.NV]).tolist()
                    door = [u for u in things if nm(u).startswith("closed door")]
                    if not door:
                        continue
                    door = int(door[0])
                    on0 = any(nm(u).startswith("switch on") for u in things)
                    cases = [("holding nothing", st, on0)]
                    for k in things:
                        if nm(k).startswith("key"):
                            fits = colour(nm(k)) == colour(nm(door))
                            cases.append(("holding the matching key" if fits else "holding a wrong key",
                                          pl.imagine(st, VP.PICK, int(k), None), on0))
                    cases += [("switch turned on", pl.imagine(st, VP.TOG, int(u), None), True)
                              for u in things if nm(u).startswith("switch off")]
                    for cname, s2, on in cases:
                        t = tally.setdefault(cname, [0, 0])
                        t[1] += 1
                        if s2 is None:
                            wrong.append((li, cname, "no change predicted"))
                            if cname.startswith("holding") and cname != "holding nothing":
                                # the pickup was not predicted: ask toggle directly, the key put in the hand
                                k = next(int(u) for u in things if nm(u).startswith("key")
                                         and (colour(nm(u)) == colour(nm(door))) == (cname == "holding the matching key"))
                                tk = tally.setdefault(cname + " (put in hand)", [0, 0])
                                tk[1] += 1
                                pred = kd.cat_of([(door, k, pl.ctx_id(st[0], st[1]))])[0]
                                if bool(pred and pred & 1) == SP.RULE[world](cname == "holding the matching key", on):
                                    tk[0] += 1
                            continue
                        held = int(pl.facts[s2[0]][VP.HELD])
                        fits = nm(held).startswith("key") and colour(nm(held)) == colour(nm(door))
                        pred = kd.cat_of([(door, held, pl.ctx_id(s2[0], s2[1]))])[0]
                        truth = SP.RULE[world](fits, on)
                        if bool(pred and pred & 1) == truth:
                            t[0] += 1
                        else:
                            wrong.append((li, cname, f"predicted {pred}, rule {'opens' if truth else 'stays'}"))
                wr[sname] = {c: f"{a}/{b}" for c, (a, b) in tally.items()}
                wr[sname + " wrong (first 6)"] = wrong[:6]
            wr["toggle_kind"] = kd.report
            wr["seconds"] = round(time.monotonic() - t0, 1)
            rs["worlds"][world] = wr
            log(f"  arm {arm} {CFG['tokens']} {CFG['score']} bound={CFG['bound']} holdout={CFG['holdout']} seed {seed} {world} ({wr['seconds']}s): "
                + "; ".join(f"{s}: {wr[s]}" for s in sets)
                + f" | toggle lambda {kd.report['lambda_sums']}, relations {kd.report.get('relation_lambda')}"
                + f", loo {kd.report.get('loo_start')} -> {kd.report.get('loo_fitted')}")
        res["seeds"].append(rs)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def main():
    install()
    args = sys.argv[1:]
    get = lambda k, dflt: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), dflt)
    CFG["tokens"], CFG["score"] = get("--tokens", CFG["tokens"]), get("--score", CFG["score"])
    CFG["heads"], CFG["steps"] = int(get("--heads", CFG["heads"])), int(get("--steps", CFG["steps"]))
    CFG["bound"], CFG["holdout"] = "--bound" in args, get("--holdout", CFG["holdout"])
    CFG["held_identity"] = "--no-held-identity" not in args
    CFG["roles"], CFG["queries"] = get("--roles", CFG["roles"]), int(get("--queries", CFG["queries"]))
    if "--check" in args:
        VP.T.configure()
        t00 = time.monotonic()
        log = lambda msg: print(f"[{time.monotonic() - t00:6.0f}s] {msg}", flush=True)
        a_, b_ = get("--seeds", "400-400").split("-")
        check(get("--arm", "B"), range(int(a_), int(b_) + 1), tuple(get("--worlds", "key,either,both").split(",")),
              int(get("--layouts", "10")), Path(get("--out", "runs/040_check.json")), log)
        return
    # acting (familiar worlds): card 038's runs through card 039's set view and this card's relations. A stored
    # success's held thing still enters the conditions; its relations are recomputed against the thing in front.
    if CFG["tokens"] != "vector":
        sys.exit("acting is built for vector tokens only")
    if "--out" not in args:
        sys.exit("--out is required (card 038's default path would be overwritten)")
    ARM["arm"] = get("--arm", "B")
    VP.main()
    out = Path(get("--out", ""))
    if out.exists() and "--dev" not in args:
        r = json.loads(out.read_text())
        r["note"] = f"Card 040, tools/card040/attention_planner.py, {CFG}"
        out.write_text(json.dumps(r, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
