"""Card 052, step 2b: gate 2 with the first interaction term (effect: card 033's recall term), trained online.

Step 2a's harness (drift.py: stream, buffer, probe set, checkpoints) with one term added to version 8's objective.
For pick up, drop and toggle, each try is stored as the front tile and the held tile before the action (rendered with
the episode's nuisance; an empty hand is the floor tile) and its outcome: nothing, the front tile changes, the hand
changes, both. Per update and per action, 128 tries are drawn from the last 20,000 (half with a change, half
without); an event's key is the two tiles' pieces and their distance in each part (card 033's "rel" keys); the term
is minus recall's leave-one-out log-likelihood of each try's outcome from the other tries in the batch, with
attention weights lambda per action learned alongside, at weight MU.

Also reported at every checkpoint: the term's value with outcomes shuffled across tries (action sensitivity).

  bin/prun python tools/card052/effect.py --seed 399 --checkpoints 40 --mu 0.1 --out runs/052/effect_399.json
"""
import copy
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift as DR                                     # noqa: E402

fn = torch.nn.functional
K, M, DIM = DR.K, DR.M, DR.DIM
ACTS = (3, 4, 5)                                       # pick up, drop, toggle
NK = 4                                                 # nothing, front changes, hand changes, both
D_KEY = 2 * K * DIM + K
MU = 0.1
PER = 128
CHECK = None
INTERACTION = "effect"                                 # step 2g: "diff" replaces the effect term
DIFF_W = 1.0
INV_W = 1.0
GATES = True                                           # step T: the transition model reads its inputs through gates
SP_W = 0.01                                            # step T: L1 weight on the gates; card 053: None = measured
MASK_TAU = 0.5                                         # card 053: the gates are random on/off masks (Gumbel-sigmoid)
VIS_W = 0.0                                            # card 053 (fix 1): weight of the visibility margin; 0 = off
VIS_MARGIN = 1.0                                       # card 053: a visibly changed tile's vector moves at least this
PULL_W = 0.0                                           # card 054 (step B, third arm): pull together batch tiles within pixel noise
PAIR_M = 0.0                                           # card 054 (step B): margin between visibly different batch tiles; 0 = off
VIS_THRESH = None                                      # card 053: pixel change beyond noise (set from untouched cells)
CALIBRATE_AT = 2000                                    # card 053: updates with the gates at 0.5 before the measurement
VAR_W = 10.0                                           # step T: the variance floor's weight
TARGET_RATE = 0.99                                     # step T: the target encoder's moving average
VIEW_N = 64                                            # view pairs per update (step R1b: 256)
DRIFT = 0.0                                            # step R1: the tint's random-walk step, as a share of TINT
VIEWS = False                                          # step 2g: the stream also returns natural views
REL_W = 0.0                                            # step 2e: the relation term's weight (rho)
REL_N = 512
START_SHARE = 0.0                                      # step 2d: share of episodes that are play starts
START_LEN = 10
STARTS = [(ev, k, c) for c in DR.GN.TRAIN for ev, k in
          (("unlock_key", "door"), ("wrong_key", "door"), ("unlock_switch", "door"), ("switch", "switch"),
           ("open", "door"), ("pickup", "key"), ("pickup", "ball"), ("pickup", "box"))]


class Stream(DR.Stream):
    """Step 2a's stream, also returning the tries of pick up, drop and toggle with their outcome."""

    def __init__(self, seed, colours=None):
        super().__init__(seed, colours=DR.GN.TRAIN if colours is None else colours)
        self.seed = seed

    def new_episode(self):
        """Random play, or (step 2d) with probability START_SHARE a play start drawn uniformly from STARTS."""
        self.limit = 100
        if START_SHARE > 0 and self.rng.random() < START_SHARE:
            st = STARTS[int(self.rng.integers(len(STARTS)))]
            if not hasattr(self, "gens"):
                self.gens = {}
            if st not in self.gens:
                self.gens[st] = DR.GN.Generator(int(self.rng.integers(2 ** 31)), start=st)
            self.env = self.gens[st].episode()
            self.limit = START_LEN
        else:
            self.env = self.gen.episode()
        self.tint = self.rng.uniform(0, DR.GN.TINT, 3)
        self.t = 0

    def step(self):
        if self.env is None or self.t >= getattr(self, "limit", 100):
            self.new_episode()
        a = int(self.rng.integers(6))
        f0 = self._front()
        k0 = None if f0 is None else tuple(f0.encode())
        h0 = self.env.carrying
        c0 = None if h0 is None else tuple(h0.encode())
        t0 = self._tile(f0) if a in ACTS else None
        th = self._tile(h0) if a in ACTS else None
        views = []
        if VIEWS:                                      # step 2g: two cells other than the front cell, before the step
            W, H = self.env.grid.width, self.env.grid.height
            fx, fy = (int(v) for v in self.env.front_pos)
            for _ in range(2):
                x, y = int(self.rng.integers(W)), int(self.rng.integers(H))
                if (x, y) != (fx, fy):
                    views.append(((x, y), self._tile(self.env.grid.get(x, y))))
        _, _, term, trunc, _ = self.env.step(a)
        self.t += 1
        if DRIFT > 0:                                  # step R1: the tint drifts between moments (own generator,
            if not hasattr(self, "drift_rng"):         # so the rest of the stream is drawn as before)
                self.drift_rng = np.random.default_rng(self.seed + 99)
            t = self.tint + self.drift_rng.normal(0, DRIFT * DR.GN.TINT, 3)
            t = np.abs(t)                              # reflected at 0 ...
            self.tint = DR.GN.TINT - np.abs(DR.GN.TINT - t)   # ... and at TINT
        self.views = [(v, self._tile(self.env.grid.get(*xy))) for xy, v in views]   # ... and after it
        tiles, pairs, tries = [], [], []
        if a in ACTS:
            f1 = self._front()
            k1 = None if f1 is None else tuple(f1.encode())
            h1 = self.env.carrying
            c1 = None if h1 is None else tuple(h1.encode())
            if k0 != k1:
                pairs.append((t0, self._tile(f1)))
            tries.append((a, t0, th, int(k0 != k1) + 2 * int(c0 != c1), (k0, c0),
                          self._tile(f1), self._tile(h1)))     # step T: the front and held tiles after
        W, H = self.env.grid.width, self.env.grid.height
        self.last_ids = []
        fr = self._front()
        tiles.append(self._tile(fr))
        self.last_ids.append(None if fr is None else tuple(fr.encode()))
        for _ in range(4):
            o = self.env.grid.get(int(self.rng.integers(W)), int(self.rng.integers(H)))
            tiles.append(self._tile(o))
            self.last_ids.append(None if o is None else tuple(o.encode()))
        if term or trunc:
            self.env = None
        self.tries = tries
        return tiles, pairs


def loo(X, R, lam):
    """Per action, the mean over tries of sum_c r_c log P_-i(c): each try predicted from the others' votes and a
    uniform prior (card 033's loo_loglik, without padding). X (A, N, D), R (A, N, NK), lam (A, D). Returns (A,)."""
    d = (torch.abs(X[:, :, None] - X[:, None]) * lam[:, None, None]).sum(-1)
    k = torch.exp(-d) * (1 - torch.eye(X.shape[1], device=X.device))
    P = (k @ R + 1.0 / NK) / (k.sum(-1, keepdim=True) + 1.0)
    return (R * torch.log(P)).sum(-1).mean(1)


class Encoder(DR.Encoder):
    def __init__(self, seed, dev="cuda"):
        super().__init__(seed, dev)
        self.theta = None
        if INTERACTION == "transition":                # step T
            self.target = copy.deepcopy(self.enc)
            for q in self.target.parameters():
                q.requires_grad_(False)
            n_in = 2 * K * DIM + K
            self.trans = nn.ModuleList([nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 2 * K * DIM))
                                        for _ in ACTS]).to(dev)
            self.gates = nn.Parameter(torch.zeros(len(ACTS), 3 * K, device=dev))   # logits; sigmoid 0.5 at start
            self.opt.add_param_group({"params": list(self.trans.parameters()) + ([self.gates] if GATES else [])})
            self.sp_w = SP_W
            if GATES and SP_W is None:                 # card 053: held at 0.5 until the balance is measured
                self.gates.requires_grad_(False)

    # -- step T: transitions read through sparse conditions
    def tpieces(self, x):
        return fn.normalize(self.target(x).reshape(len(x), K, DIM), dim=-1)

    def gate_values(self):
        return torch.sigmoid(self.gates) if GATES else torch.ones_like(self.gates)

    def masks(self, ai, n, sample, force=None):
        """Card 053: each input is on or off. Training draws a hard mask per try (Gumbel-sigmoid, straight-through;
        Lachapelle et al. 2022, CDL), so the next layer cannot rescale a half-closed gate; acting uses p > 0.5.
        force = (i, 0 or 1) fixes input i for the balance measurement."""
        if not GATES:
            return torch.ones(n, 3 * K, device=self.dev)
        p = self.gate_values()[ai].expand(n, -1)
        if sample:
            u = torch.rand_like(p).clamp(1e-6, 1 - 1e-6)
            soft = torch.sigmoid((torch.logit(p.clamp(1e-6, 1 - 1e-6)) + torch.log(u) - torch.log(1 - u)) / MASK_TAU)
            m = (soft > 0.5).float() + soft - soft.detach()
        else:
            m = (p > 0.5).float()
        if force is not None:
            m = m.clone()
            m[:, force[0]] = float(force[1])
        return m

    def predict_after(self, ai, zf, zh, sample=False, force=None):
        """The transition model of action ACTS[ai]: the after-pieces of front and held from the before-pieces, read
        through the masks on the 12 candidate conditions (front parts, held parts, front-held distance per part)."""
        g = self.masks(ai, len(zf), sample, force)
        rel = (zf - zh).norm(dim=-1)
        inp = torch.cat([(zf * g[:, :K, None]).reshape(len(zf), -1), (zh * g[:, K:2 * K, None]).reshape(len(zh), -1),
                         rel * g[:, 2 * K:]], 1)
        d = self.trans[ai](inp).reshape(len(zf), 2, K, DIM)
        return fn.normalize(zf + d[:, 0], dim=-1), fn.normalize(zh + d[:, 1], dim=-1)

    def transition(self, tries):
        total, parts = 0.0, []
        for ai, tr in enumerate(tries):
            zf, zh = self.pieces(self.x([t[0] for t in tr])), self.pieces(self.x([t[1] for t in tr]))
            pf, ph = self.predict_after(ai, zf, zh, sample=True)
            with torch.no_grad():
                tf, th = self.tpieces(self.x([t[3] for t in tr])), self.tpieces(self.x([t[4] for t in tr]))
            l = ((pf - tf) ** 2).sum((-1, -2)).mean() + ((ph - th) ** 2).sum((-1, -2)).mean()
            total = total + l
            parts.append(float(l.detach()))
            if VIS_W > 0:                              # card 053, fix 1: what an action visibly changes moves the vector
                total = total + VIS_W * self.visibility(tr, zf, zh)
        if GATES and self.sp_w is not None:            # card 053: one weight per action, from its own gains
            w = torch.as_tensor(self.sp_w, dtype=torch.float32, device=self.dev).reshape(-1, 1)
            total = total + (w * self.gate_values()).sum()
        self.last_trans = parts
        return total

    def visibility(self, tr, zf, zh):
        """C-SWM's margin (Kipf et al. 2020) on the agent's own observed changes: a front or held tile whose pixels
        changed beyond noise across the action (mean absolute difference above VIS_THRESH, set from untouched
        cells) must move at least VIS_MARGIN (whole-vector distance); unchanged tiles are not pushed."""
        hinge, n = 0.0, 0
        for zb, ia in ((zf, 3), (zh, 4)):
            ib = 0 if ia == 3 else 1
            ch = [i for i, t in enumerate(tr) if pix_change(t[ib], t[ia]) > VIS_THRESH]
            if ch:
                za = self.pieces(self.x([tr[i][ia] for i in ch]))
                d = (za - zb[ch]).reshape(len(ch), -1).norm(dim=1)
                hinge = hinge + torch.relu(VIS_MARGIN - d).sum()
                n += len(ch)
        self.last_vis = (float(hinge) / max(n, 1), n)
        return hinge / max(n, 1)

    @torch.no_grad()
    def balance(self, tries, draws=8):
        """Card 053: per action and input, the transition loss with the input off minus with it on (the others drawn
        at 0.5), averaged over draws: the gain the input brings, against which its penalty is set."""
        gain = torch.zeros(len(ACTS), 3 * K, device=self.dev)
        for ai, tr in enumerate(tries):
            zf, zh = self.pieces(self.x([t[0] for t in tr])), self.pieces(self.x([t[1] for t in tr]))
            tf, th = self.tpieces(self.x([t[3] for t in tr])), self.tpieces(self.x([t[4] for t in tr]))
            for i in range(3 * K):
                for _ in range(draws):
                    for v, sgn in ((0, 1.0), (1, -1.0)):
                        pf, ph = self.predict_after(ai, zf, zh, sample=True, force=(i, v))
                        gain[ai, i] += sgn * (((pf - tf) ** 2).sum((-1, -2)).mean() + ((ph - th) ** 2).sum((-1, -2)).mean())
        return (gain / draws).cpu()

    def anchor_transition(self, z, x=None):
        """Step T's anchor: a variance floor (VICReg's hinge, each number's spread at least 1/sqrt(DIM) over the
        batch) and the identity transition of one cell a step apart, against the target encoder. Card 054 (step B,
        PAIR_M > 0): the floor is replaced by C-SWM's margin between every two batch tiles whose pixels differ beyond
        noise (mean absolute difference above VIS_THRESH): their whole vectors at least PAIR_M apart. Noisy copies
        (pixels within noise) are not pushed."""
        zf = z.reshape(len(z), -1)
        if PAIR_M > 0 and x is not None:
            px = torch.cdist(x.reshape(len(x), -1), x.reshape(len(x), -1), p=1) / x[0].numel()
            diff = torch.triu(px > VIS_THRESH, diagonal=1)
            d = torch.cdist(zf, zf)
            floor = torch.relu(PAIR_M - d[diff]).mean() if diff.any() else 0.0
            if PULL_W > 0:                             # noisy copies of one appearance: one point
                same = torch.triu(px <= VIS_THRESH, diagonal=1)
                if same.any():
                    floor = floor + PULL_W * d[same].pow(2).mean()
            self.last_pair = (float(floor), int(diff.sum()))
        else:
            floor = VAR_W * torch.relu(DIM ** -0.5 - zf.std(0)).mean()
        v = getattr(self, "_inv", None)
        if not v:
            return floor
        a = self.pieces(self.x([p[0] for p in v]))
        with torch.no_grad():
            b = self.tpieces(self.x([p[1] for p in v]))
        return floor + ((a - b) ** 2).sum((-1, -2)).mean()

    def after_step(self):
        if INTERACTION == "transition":
            with torch.no_grad():
                for pt, po in zip(self.target.parameters(), self.enc.parameters()):
                    pt.mul_(TARGET_RATE).add_(po, alpha=1 - TARGET_RATE)

    def keys(self, f, h):
        zf, zh = self.pieces(self.x(f)), self.pieces(self.x(h))
        rel = (zf - zh).norm(dim=-1)
        return torch.cat([zf.reshape(len(f), -1), zh.reshape(len(h), -1), rel], 1)

    def batch(self, tries):
        """tries: per action, a list of (front, held, outcome). Returns X (A, N, D), R (A, N, NK)."""
        X = torch.stack([self.keys([t[0] for t in tr], [t[1] for t in tr]) for tr in tries])
        R = torch.stack([fn.one_hot(torch.as_tensor([t[2] for t in tr], device=self.dev), NK).float() for tr in tries])
        return X, R

    def start_theta(self, X):
        with torch.no_grad():
            th = []
            for a in range(len(X)):
                d = torch.abs(X[a][:, None] - X[a][None]).sum(-1)
                lam = 1.0 / d[d > 0].median().clamp(min=1e-6)
                th.append(torch.full((D_KEY,), float(torch.log(torch.expm1(lam))), device=self.dev))
        self.theta = nn.Parameter(torch.stack(th))
        self.opt.add_param_group({"params": [self.theta]})

    def effect(self, tries):
        X, R = self.batch(tries)
        if self.theta is None:
            self.start_theta(X.detach())
        return -loo(X, R, fn.softplus(self.theta)).sum()

    def invariance(self):
        """Step 2g, VISReg's invariance term on natural views: two renderings of one cell, one step apart."""
        v = getattr(self, "_inv", None)
        if not v:
            return 0.0
        return INV_W * fn.mse_loss(self.pieces(self.x([p[0] for p in v])), self.pieces(self.x([p[1] for p in v])))

    def align_uniform(self):
        """Step R1b, Wang and Isola (2020) as published, on one batch of natural view pairs: alignment, the mean
        squared distance between the two views' normalised whole vectors; uniformity over each view's vectors."""
        v = getattr(self, "_inv", None)
        if not v:
            return 0.0
        a = fn.normalize(self.pieces(self.x([p[0] for p in v])).reshape(len(v), -1), dim=-1)
        b = fn.normalize(self.pieces(self.x([p[1] for p in v])).reshape(len(v), -1), dim=-1)
        align = (a - b).pow(2).sum(-1).mean()
        unif = (DR.uniformity(a) + DR.uniformity(b)) / 2
        self.last_au = (float(align.detach()), float(unif.detach()))
        return INV_W * align + unif

    @torch.no_grad()
    def noise_tau(self):
        """How far apart two views of one cell lie, per part: the 95th percentile (0 when there are no views)."""
        v = getattr(self, "_inv", None)
        if not v:
            return 0.0
        d = (self.pieces(self.x([p[0] for p in v])) - self.pieces(self.x([p[1] for p in v]))).norm(dim=-1)
        return float(torch.quantile(d.flatten(), 0.95))

    def differentiate(self, tries):
        """Step 2g, error-driven differentiation. Recall by vectors predicts each try from the others; for each error
        (true outcome given < 0.5), its most similar try with another outcome is the partner; in the slot (front or
        held tile) whose two tiles differ more, the part where they differ most is pushed apart by a hinge up to the
        median distance between that part's codebook entries, weighted by the error's surprise. Pairs whose tiles
        differ no more than two views of one cell are skipped. Recall's attention weights are fitted on stopped
        vectors. Nothing pulls tiles together."""
        X, R = self.batch(tries)
        if self.theta is None:
            self.start_theta(X.detach())
        fit = -loo(X.detach(), R, fn.softplus(self.theta)).sum()
        tau = max(self.noise_tau(), 1e-6)
        books = fn.normalize(self.books.detach(), dim=-1)
        margin = torch.stack([torch.pdist(books[k]).median() for k in range(K)])
        total, n_err, n_push, n_held = 0.0, 0, 0, 0
        for a in range(len(tries)):
            x, r = X[a], R[a]
            with torch.no_grad():
                d = (torch.abs(x[:, None] - x[None]) * fn.softplus(self.theta[a])).sum(-1)
                k = torch.exp(-d) * (1 - torch.eye(len(x), device=self.dev))
                P = (k @ r + 1.0 / NK) / (k.sum(-1, keepdim=True) + 1.0)
                pt = (P * r).sum(-1)
                other = (r @ r.T) == 0
                kk = torch.where(other, k, torch.full_like(k, -1.0))
                j = kk.argmax(1)
                err = pt < 0.5
                ok = err & (kk.amax(1) >= 0)
            n_err += int(err.sum())
            if not ok.any():
                continue
            i = torch.nonzero(ok).flatten()
            jj = j[i]
            zf = x[:, :K * DIM].reshape(len(x), K, DIM)
            zh = x[:, K * DIM:2 * K * DIM].reshape(len(x), K, DIM)
            df = (zf[i] - zf[jj]).norm(dim=-1)
            dh = (zh[i] - zh[jj]).norm(dim=-1)
            use_h = dh.detach().amax(1) > df.detach().amax(1)
            dsel = torch.where(use_h[:, None], dh, df)
            part = dsel.detach().argmax(1)
            dmax = dsel.gather(1, part[:, None]).squeeze(1)
            real = (dmax.detach() > tau).float()
            total = total + (-torch.log(pt[i]) * real * torch.relu(margin[part] - dmax)).sum() / len(x)
            n_push += int(real.sum())
            n_held += int((use_h.float() * real).sum())
        self.diff_stats = (n_err, n_push, n_held)
        self.last_effect = float(fit.detach())
        return DIFF_W * total + fit

    def extra(self):
        tries = getattr(self, "_tries", None)
        if not tries:
            return 0.0
        if INTERACTION == "diff":                      # step 2g: replaces the effect term
            return self.differentiate(tries)
        if INTERACTION == "transition":                # step T: replaces it too; recall's weights on stopped vectors
            X, R = self.batch(tries)
            if self.theta is None:
                self.start_theta(X.detach())
            fit = -loo(X.detach(), R, fn.softplus(self.theta)).sum()
            self.last_effect = float(fit.detach())
            return self.transition(tries) + fit
        e = self.effect(tries)
        self.last_effect = float(e.detach())
        out = MU * e
        if REL_W > 0 and getattr(self, "_rel", None):
            r = self.relation(self._rel)
            self.last_relation = None if r is None else float(r.detach())
            if r is not None:
                out = out + REL_W * r
        return out

    def relation(self, tries):
        """Step 2e: toggle tries grouped by the front tile's code tuple; in groups with both outcomes, P(same) =
        sigmoid(a - b * min over parts of |front - held|), trained against "the front tile changed"."""
        if not hasattr(self, "rel_ab"):
            self.rel_ab = nn.Parameter(torch.tensor([0.0, 0.5413], device=self.dev))   # a = 0, b = softplus(.) = 1
            self.opt.add_param_group({"params": [self.rel_ab]})
        zf, zh = self.pieces(self.x([t[0] for t in tries])), self.pieces(self.x([t[1] for t in tries]))
        with torch.no_grad():
            _, idx = self.quant(zf)
        y = np.array([t[2] & 1 for t in tries])
        groups = {}
        for i, g in enumerate(map(tuple, idx.cpu().numpy().tolist())):
            groups.setdefault(g, []).append(i)
        sel = [i for g in groups.values() if len(set(y[g])) > 1 for i in g]
        self.rel_used = len(sel)
        if not sel:
            return None
        sel = torch.as_tensor(sel, device=self.dev)
        d = (zf[sel] - zh[sel]).norm(dim=-1).amin(1)
        logit = self.rel_ab[0] - fn.softplus(self.rel_ab[1]) * d
        return fn.binary_cross_entropy_with_logits(logit, torch.as_tensor(y, dtype=torch.float32, device=self.dev)[sel])

    @torch.no_grad()
    def sensitivity(self, tries, rng):
        X, R = self.batch(tries)
        lam = fn.softplus(self.theta)
        ll = loo(X, R, lam)
        Rs = torch.stack([r[torch.as_tensor(rng.permutation(len(r)), device=self.dev)] for r in R])
        return ll.cpu().numpy().tolist(), loo(X, Rs, lam).cpu().numpy().tolist()


def pix_change(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).mean() / 255)


@torch.no_grad()
def book_health(enc, probe):
    """Card 053: per part, pairs of used entries whose regions overlap (card 035's radius: the farthest of the
    entry's probe pieces, at least the median distance of all probe pieces to their entries), and the share of
    probe pieces within 0.05 of the boundary to their second-nearest entry."""
    z = enc.pieces(enc.x(probe))
    e = fn.normalize(enc.books, dim=-1)
    d = torch.cdist(z.transpose(0, 1), e)
    s, o = d.sort(-1)
    dup = []
    for k in range(K):
        a, dist = o[k, :, 0], s[k, :, 0]
        used = sorted(set(a.tolist()))
        med = float(dist.median())
        r = {c: max(float(dist[a == c].max()), med) for c in used}
        dup.append(sum(float(torch.dist(e[k, i], e[k, j])) < r[i] + r[j] for x, i in enumerate(used) for j in used[x + 1:]))
    return {"overlapping_entry_pairs_per_part": dup,
            "boundary_share": round(float(((s[..., 1] - s[..., 0]) < 0.05).float().mean()), 4)}


def draw(buf, rng):
    """Per action: PER distinct tries, half with a change and half without (fewer changed ones: the rest unchanged)."""
    out = []
    for a in ACTS:
        ch, no = buf[a][1], buf[a][0]
        nc = min(PER // 2, len(ch))
        pick = [ch[i] for i in rng.choice(len(ch), size=nc, replace=False)] if nc else []
        pick += [no[i] for i in rng.choice(len(no), size=PER - nc, replace=False)]
        out.append(pick)
    return out


def run(seed, checkpoints, buffer, every, upd_per_ck, out):
    t0 = time.monotonic()
    stream = Stream(seed)
    probe, pid = DR.probe_set(seed)
    ident = sorted(set(pid), key=str)
    enc = Encoder(seed)
    tiles, pairs = deque(maxlen=buffer), deque(maxlen=max(buffer // 10, 100))
    buf = {a: (deque(maxlen=buffer // 2), deque(maxlen=buffer // 2)) for a in ACTS}
    views = deque(maxlen=max(buffer // 10, 100))       # step 2g: natural views
    rng = np.random.default_rng(seed + 2)
    prev, rows, steps = None, [], 0
    calib = None

    def play():
        nonlocal steps
        t, p = stream.step()
        tiles.extend(t)
        pairs.extend(p)
        for t in stream.tries:
            a, f, h, o = t[:4]
            buf[a][int(o > 0)].append((f, h, o) + tuple(t[5:7]))
        views.extend(getattr(stream, "views", []))
        steps += 1

    while len(tiles) < 2000 or min(len(buf[a][0]) for a in ACTS) < PER or ((VIS_W > 0 or PAIR_M > 0) and len(views) < 1000):
        play()
    global VIS_THRESH
    if VIS_W > 0 or PAIR_M > 0:                        # card 053: the noise ceiling, from untouched cells a step apart
        VIS_THRESH = 1.25 * max(pix_change(a, b) for a, b in list(views)[:1000])
    for ck in range(checkpoints):
        for g in enc.opt.param_groups:
            g["lr"] = 1e-3 * DR.DECAY ** ck
        effs, rels, diffs = [], [], []
        for _ in range(upd_per_ck):
            for _ in range(every):
                play()
            bi = rng.integers(len(tiles), size=256)
            pi = rng.integers(len(pairs), size=min(64, len(pairs))) if pairs else []
            enc._tries = draw(buf, rng)
            if views:
                enc._inv = [views[i] for i in rng.integers(len(views), size=min(VIEW_N, len(views)))]
            if REL_W > 0:                              # step 2e: toggle tries for the relation term, half changed
                ch, no = buf[5][1], buf[5][0]
                nc = min(REL_N // 2, len(ch))
                enc._rel = [ch[i] for i in rng.choice(len(ch), size=nc, replace=False)] + \
                           [no[i] for i in rng.choice(len(no), size=min(REL_N - nc, len(no)), replace=False)]
            loss = enc.update([tiles[i] for i in bi], [pairs[i] for i in pi])
            if INTERACTION == "transition" and GATES and SP_W is None and enc.n == CALIBRATE_AT:
                gain = enc.balance(draw(buf, rng))     # card 053: the penalty from the measured balance
                enc.sp_w = [0.5 * float(g.clamp_min(0).median()) for g in gain]
                enc.gates.requires_grad_(True)
                calib = {"gain_per_action_input": [[round(float(v), 5) for v in r] for r in gain], "sp_w": enc.sp_w}
                print({"calibration": calib}, flush=True)
            effs.append(enc.last_effect)
            if INTERACTION == "diff":
                diffs.append(enc.diff_stats)
            if REL_W > 0:
                rels.append((getattr(enc, "last_relation", None), getattr(enc, "rel_used", 0)))
        sens = [enc.sensitivity(draw(buf, rng), rng) for _ in range(10)]
        ll = np.mean([s[0] for s in sens], 0)
        ll_sh = np.mean([s[1] for s in sens], 0)
        c, err = enc.codes(probe)
        tup = [tuple(r) for r in c.tolist()]
        flip = None if prev is None else float(np.mean([a != b for a, b in zip(tup, prev)]))
        if prev is not None:
            P0, P1 = np.array(prev), np.array(tup)
            pair_change = [round(float(((P0[:, k][:, None] == P0[:, k][None]) != (P1[:, k][:, None] == P1[:, k][None])).mean()), 4)
                           for k in range(K)]
            per_part = [round(float(np.mean(P0[:, k] != P1[:, k])), 4) for k in range(K)]
        else:
            pair_change = per_part = None
        prev = tup
        row = {"checkpoint": ck + 1, "updates": (ck + 1) * upd_per_ck, "play_steps": steps, "loss": round(loss, 5),
               "effect": round(float(np.mean(effs)), 4),
               "diff_errors_pushes_held_per_update": None if not diffs else [round(float(v), 2) for v in np.mean(diffs, 0)],
               "align_uniform_last": [round(v, 4) for v in getattr(enc, "last_au", ())] or None,
               "transition_loss_per_action": [round(v, 4) for v in getattr(enc, "last_trans", ())] or None,
               "visibility_last": None if VIS_W == 0 else [round(v, 4) for v in getattr(enc, "last_vis", (0, 0))],
               "pair_margin_last": None if PAIR_M == 0 else [round(v, 4) for v in getattr(enc, "last_pair", (0, 0))],
               "gates": None if INTERACTION != "transition" else [[round(float(v), 3) for v in r] for r in enc.gate_values()],
               "relation": None if not rels else round(float(np.mean([r for r, _ in rels if r is not None] or [np.nan])), 4),
               "relation_tries_used": None if not rels else round(float(np.mean([u for _, u in rels])), 1),
               "loo_loglik_per_action": [round(v, 4) for v in ll.tolist()],
               "loo_loglik_shuffled": [round(v, 4) for v in ll_sh.tolist()],
               "probe_rebuild_mse": round(err, 6), "code_flip_rate": None if flip is None else round(flip, 4),
               "pair_change_per_part": pair_change, "flip_per_part": per_part,
               "distinct_tuples": len(set(tup)), "probe_identities": len(ident),
               "identities_split": sum(len({t for t, q in zip(tup, pid) if q == i}) > 1 for i in ident),
               "tuples_shared_by_identities": sum(len({q for t2, q in zip(tup, pid) if t2 == t}) > 1 for t in set(tup)),
               "codes_used_per_part": [len(set(c[:, k].tolist())) for k in range(K)],
               "tries_changed_per_action": [len(buf[a][1]) for a in ACTS],
               "restarts": enc.restarts, **book_health(enc, probe), "seconds": round(time.monotonic() - t0, 1)}
        if CHECK is not None:                           # further measures at each checkpoint (step 2c)
            row.update(CHECK(enc))
        rows.append(row)
        print(row, flush=True)
        if ck == checkpoints - 1:                      # from step R1b: the encoder is kept for diagnostics
            torch.save({"enc": enc.enc.state_dict(), "dec": enc.dec.state_dict(), "books": enc.books.detach().cpu(),
                        "theta": None if enc.theta is None else enc.theta.detach().cpu(),
                        "trans": enc.trans.state_dict() if hasattr(enc, "trans") else None,
                        "gates": enc.gates.detach().cpu() if hasattr(enc, "gates") else None},
                       str(out).replace(".json", ".pt"))
        Path(out).write_text(json.dumps({"note": "Card 052 step 2b, tools/card052/effect.py", "seed": seed, "mu": MU,
                                         "ema": DR.Encoder.EMA, "start_share": START_SHARE, "relation_weight": REL_W,
                                         "anchor": DR.Encoder.ANCHOR, "codebook": DR.Encoder.CODEBOOK, "interaction": INTERACTION, "diff_w": DIFF_W, "inv_w": INV_W, "drift": DRIFT, "gates_on": GATES, "sp_w": SP_W, "vis_w": VIS_W, "pair_m": PAIR_M, "pull_w": PULL_W, "vis_margin": VIS_MARGIN, "vis_thresh": VIS_THRESH, "calibration": calib, "mask_tau": MASK_TAU, "var_w": VAR_W, "buffer": buffer, "update_every_steps": every,
                                         "updates_per_checkpoint": upd_per_ck, "rows": rows}, indent=1) + "\n")


def main():
    global MU, START_SHARE, REL_W, INTERACTION, DIFF_W, INV_W, VIEWS, DRIFT, VIEW_N, GATES, SP_W, VIS_W, PAIR_M, PULL_W
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    Path(get("--out", "runs/052/effect.json")).parent.mkdir(parents=True, exist_ok=True)
    MU = float(get("--mu", "0.1"))
    DR.DECAY = float(get("--decay", "1"))
    DR.Encoder.EMA = float(get("--ema", "0"))
    DR.Encoder.RESTART_EVERY = int(get("--restart", "250"))
    START_SHARE = float(get("--starts", "0"))
    REL_W = float(get("--rel", "0"))
    DR.Encoder.ANCHOR = get("--anchor", "pixels")
    DR.Encoder.CODEBOOK = get("--codebook", "learned")
    INTERACTION = get("--interaction", "effect")
    DIFF_W = float(get("--diff-w", "1"))
    INV_W = float(get("--inv-w", "1"))
    VIEWS = DR.Encoder.ANCHOR != "pixels" or INTERACTION == "diff"
    DRIFT = float(get("--drift", "0"))
    GATES = get("--gates", "1") == "1"
    VIS_W = float(get("--vis-w", "0"))
    PAIR_M = float(get("--pair-m", "0"))
    PULL_W = float(get("--pull-w", "0"))
    SP_W = None if get("--sp-w", "0.01") == "auto" else float(get("--sp-w", "0.01"))
    VIEW_N = 256 if DR.Encoder.ANCHOR in ("align_uniform", "transition") else 64
    run(int(get("--seed", "399")), int(get("--checkpoints", "40")), int(get("--buffer", "20000")),
        int(get("--every", "10")), int(get("--updates", "2000")), get("--out", "runs/052/effect.json"))


if __name__ == "__main__":
    main()
