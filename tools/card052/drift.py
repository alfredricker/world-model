"""Card 052, step 2a: the encoder learned online from the generator's tiles, and its drift.

Version 8's encoder objective (card 031's sliced VQ: rebuilding the tile, codebook and commitment terms, and the
adaptive pair term on tiles an action changed in place; 4 parts of 8 numbers, 8 codes each), trained online:
random play over step 1's generator streams tries into a buffer of the last N; the encoder takes one update of
256 tiles from the buffer every 10 steps of play. Without card 033's recall term (it needs memories keyed by the
codes it trains, which is what the hand-off is for). Card 031 re-seeded codes by which tiles are the same thing
(evaluator knowledge); here a code unused for 250 updates is restarted at the pieces of a random tile from the
batch, drawn by rebuild error (dead-code restart, as in Jukebox and SoundStream; no labels). The first run without
any re-seeding collapsed to 6-9 code tuples on 2,000 probe tiles.

Drift: a fixed probe set of 2,000 tiles (rendered once, nuisance included, from separate episodes in training
colours); at every checkpoint (2,000 updates) each probe tile's code tuple is read; the code flip rate is the share
of probe tiles whose tuple differs from the previous checkpoint's. Also reported: how many distinct tuples the
probe set uses, and the probe tiles' rebuild error.

  bin/prun python tools/card052/drift.py --seed 399 --checkpoints 40 --buffer 20000 --out runs/052/drift_399.json
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generator as GN                                 # noqa: E402

fn = torch.nn.functional
K, M, DIM = 4, 8, 8
DECAY = 1.0


class Stream:
    """Random play over the generator; per step: the front tile before and after (a pair when it changed) and
    4 random cells of the grid, each rendered at 8 pixels with the episode's tint and fresh noise."""

    def __init__(self, seed, colours=GN.TRAIN):
        self.gen = GN.Generator(seed, colours=colours)
        self.rng = np.random.default_rng(seed + 1)
        self.env, self.t = None, 0

    def _tile(self, obj):
        return (GN.tile(obj, self.tint, self.rng) * 255).astype(np.uint8)

    def _front(self):
        x, y = self.env.front_pos
        return self.env.grid.get(int(x), int(y))

    def step(self):
        if self.env is None or self.t >= 100:
            self.env = self.gen.episode()
            self.tint = self.rng.uniform(0, GN.TINT, 3)
            self.t = 0
        a = int(self.rng.integers(6))
        f0 = self._front()
        k0 = None if f0 is None else tuple(f0.encode())
        t0 = self._tile(f0) if a in (3, 4, 5) else None    # rendered before the step: doors change in place
        _, _, term, trunc, _ = self.env.step(a)
        self.t += 1
        tiles, pairs = [], []
        if a in (3, 4, 5):
            f1 = self._front()
            k1 = None if f1 is None else tuple(f1.encode())
            if k0 != k1:                                # an action changed the tile in place
                pairs.append((t0, self._tile(f1)))
        W, H = self.env.grid.width, self.env.grid.height
        self.last_ids = []
        fr = self._front()                             # the front tile, and 4 random cells
        tiles.append(self._tile(fr))
        self.last_ids.append(None if fr is None else tuple(fr.encode()))
        for _ in range(4):
            o = self.env.grid.get(int(self.rng.integers(W)), int(self.rng.integers(H)))
            tiles.append(self._tile(o))
            self.last_ids.append(None if o is None else tuple(o.encode()))
        if term or trunc:
            self.env = None
        return tiles, pairs


def probe_set(seed, n=2000):
    """Fixed probe tiles, rendered once (with nuisance), from separate episodes; with their evaluator identities
    (object encoding), for the report only."""
    s = Stream(seed + 777)
    per = {}
    for _ in range(60000):                            # stratified by identity: up to 40 tiles each
        t, _ = s.step()
        for x, q in zip(t, s.last_ids):
            if len(per.setdefault(q, [])) < 40:
                per[q].append(x)
    out, ids = [], []
    for q in sorted(per, key=str):
        out.extend(per[q])
        ids.extend([q] * len(per[q]))
    return np.stack(out), ids


def visreg(z, slices=64, target=1.0):
    """VISReg's regularizer (Wu, Balestriero and Levine 2026, Algorithm 1): centring, scale (target - std per
    dimension) and shape (sliced Wasserstein distance of the standardised numbers, scale stopped, to an isotropic
    Gaussian). Step 2g applies it to the normalised parts, where an even spread has std 1/sqrt(DIM)."""
    mu = z.mean(0)
    zc = z - mu
    std = zc.std(0, unbiased=False)
    p = (zc / (std.detach() + 1e-6)) @ fn.normalize(torch.randn(z.shape[1], slices, device=z.device), dim=0)
    n = len(z)
    q = torch.distributions.Normal(0.0, 1.0).icdf(torch.arange(1, n + 1, device=z.device) / (n + 1))
    return mu.pow(2).mean() + (target - std).pow(2).mean() + (torch.sort(p, 0).values - q[:, None]).pow(2).mean()


def uniformity(z, t=2.0):
    """Wang and Isola (2020): log of the mean Gaussian potential over all pairs of the batch's whole vectors (the
    four parts together, normalised); every pair of different vectors is pushed apart, however often each occurs."""
    v = fn.normalize(z.reshape(len(z), -1), dim=-1)
    return torch.pdist(v).pow(2).mul(-t).exp().mean().log()


class Encoder:
    ANCHOR = "pixels"                                  # step 2g: "uniformity" replaces the decoder ("visreg" dropped)
    CODEBOOK = "learned"                               # step 2g: "fixed" entries
    EMA = 0.0                                          # > 0: codebooks as moving averages (VQ-VAE-2), restarts rarer
    RESTART_EVERY = 250

    def __init__(self, seed, dev="cuda"):
        torch.manual_seed(seed)
        self.dev = dev
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                 nn.ReLU(), nn.Flatten(), nn.Linear(32 * 16, K * DIM)).to(dev)
        self.dec = nn.Sequential(nn.Linear(K * DIM, 32 * 16), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)),
                                 nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()).to(dev)
        self.books = nn.Parameter(torch.randn(K, M, DIM, device=dev))
        if self.CODEBOOK == "fixed":                   # step 2g: fixed entries, the part's axes (as FSQ: no codebook loss)
            with torch.no_grad():
                self.books.copy_(torch.eye(M, DIM, device=dev).expand(K, M, DIM))
            self.books.requires_grad_(False)
        self.opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()) + [self.books], lr=1e-3)
        self.used = torch.zeros(K, M, device=dev)
        self.ema_n = torch.ones(K, M, device=dev)
        self.ema_w = fn.normalize(self.books.detach().clone(), dim=-1)
        self.n = 0
        self.restarts = 0

    def x(self, tiles):
        return torch.as_tensor(np.asarray(tiles) / 255.0, dtype=torch.float32, device=self.dev).permute(0, 3, 1, 2)

    def pieces(self, x):
        return fn.normalize(self.enc(x).reshape(len(x), K, DIM), dim=-1)

    def quant(self, z):
        e = fn.normalize(self.books, dim=-1)
        d = torch.cdist(z.transpose(0, 1), e)
        idx = d.argmin(2).T
        return e[torch.arange(K, device=self.dev)[None], idx], idx

    def update(self, tiles, pairs, pair_w=0.1):
        x = self.x(tiles)
        raw = self.enc(x)
        z = fn.normalize(raw.reshape(len(x), K, DIM), dim=-1)
        zq, _ = self.quant(z)
        if self.ANCHOR == "pixels":                    # version 8: rebuild the tile through the codes
            st = z + (zq - z).detach()
            loss = fn.mse_loss(self.dec(st.reshape(len(x), -1)), x)
        elif self.ANCHOR == "visreg":                  # step 2g, dropped: its shape term collapsed the network here
            loss = visreg(z.reshape(len(x), -1), target=DIM ** -0.5) + self.invariance()
        elif self.ANCHOR == "align_uniform":           # step R1b: alignment and uniformity on one batch of view pairs
            loss = self.align_uniform()
        else:                                          # step 2g: uniformity (Wang and Isola 2020), no decoder
            loss = uniformity(z) + self.invariance()
        book = 0.0 if (self.EMA or self.CODEBOOK == "fixed") else fn.mse_loss(zq, z.detach())
        loss = loss + book + 0.25 * fn.mse_loss(z, zq.detach())
        if pairs:
            za = self.pieces(self.x([p[0] for p in pairs]))
            zb = self.pieces(self.x([p[1] for p in pairs]))
            d = (za - zb).norm(dim=-1)
            mid = (d.amax(1, keepdim=True) + d.amin(1, keepdim=True)).detach() / 2
            loss = loss + pair_w * (d * (d < mid)).sum(1).mean()
        loss = loss + self.extra()                    # further terms (step 2b on); none in step 2a
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            _, idx = self.quant(z.detach())
            for k in range(K):
                self.used[k].index_add_(0, idx[:, k], torch.ones(len(idx), device=self.dev))
            if self.EMA:
                for k in range(K):
                    oh = fn.one_hot(idx[:, k], M).float()
                    self.ema_n[k] = self.EMA * self.ema_n[k] + (1 - self.EMA) * oh.sum(0)
                    self.ema_w[k] = self.EMA * self.ema_w[k] + (1 - self.EMA) * (oh.T @ z[:, k].detach())
                    self.books[k] = self.ema_w[k] / self.ema_n[k].clamp_min(1e-5)[:, None]
            self.n += 1
            if self.n % self.RESTART_EVERY == 0:      # dead-code restart
                err = ((self.dec(zq.reshape(len(x), -1)) - x) ** 2).mean((1, 2, 3))
                if self.ANCHOR != "pixels":            # no decoder: a uniformly drawn tile
                    err = torch.ones_like(err)
                p = err.cpu().numpy().astype(np.float64)
                p /= p.sum()
                for k in range(K):
                    for c in torch.nonzero(self.used[k] == 0).flatten().tolist():
                        i = int(np.random.default_rng(self.n * 31 + k * 7 + c).choice(len(x), p=p))
                        self.books[k, c] = z[i, k].detach()
                        self.ema_w[k, c] = z[i, k].detach()
                        self.ema_n[k, c] = 1.0
                        self.restarts += 1
                self.used.zero_()
        return float(loss)

    def extra(self):
        return 0.0

    def invariance(self):
        return 0.0

    def align_uniform(self):
        return 0.0

    @torch.no_grad()
    def codes(self, tiles):
        out, err = [], []
        for i in range(0, len(tiles), 1000):
            x = self.x(tiles[i:i + 1000])
            z = self.pieces(x)
            zq, idx = self.quant(z)
            out.append(idx.cpu().numpy())
            err.append(((self.dec(zq.reshape(len(x), -1)) - x) ** 2).mean((1, 2, 3)).cpu().numpy())
        return np.concatenate(out), float(np.concatenate(err).mean())


def run(seed, checkpoints, buffer, every, upd_per_ck, out):
    t0 = time.monotonic()
    stream = Stream(seed)
    probe, pid = probe_set(seed)
    ident = sorted(set(pid), key=str)
    enc = Encoder(seed)
    tiles, pairs = deque(maxlen=buffer), deque(maxlen=max(buffer // 10, 100))
    rng = np.random.default_rng(seed + 2)
    prev, rows, steps = None, [], 0
    while len(tiles) < 2000:                           # a first fill
        t, p = stream.step()
        tiles.extend(t)
        pairs.extend(p)
        steps += 1
    for ck in range(checkpoints):
        for g in enc.opt.param_groups:                 # the schedule: plasticity decays (DECAY = 1: constant)
            g["lr"] = 1e-3 * DECAY ** ck
        for _ in range(upd_per_ck):
            for _ in range(every):
                t, p = stream.step()
                tiles.extend(t)
                pairs.extend(p)
                steps += 1
            bi = rng.integers(len(tiles), size=256)
            pi = rng.integers(len(pairs), size=min(64, len(pairs))) if pairs else []
            loss = enc.update([tiles[i] for i in bi], [pairs[i] for i in pi])
        c, err = enc.codes(probe)
        tup = [tuple(r) for r in c.tolist()]
        flip = None if prev is None else float(np.mean([a != b for a, b in zip(tup, prev)]))
        # label-free: per part, the share of probe-tile pairs whose same-code / different-code status changed
        if prev is not None:
            P0, P1 = np.array(prev), np.array(tup)
            pr = []
            for k in range(K):
                s0 = P0[:, k][:, None] == P0[:, k][None]
                s1 = P1[:, k][:, None] == P1[:, k][None]
                pr.append(float((s0 != s1).mean()))
            pair_change = [round(v, 4) for v in pr]
        else:
            pair_change = None
        per_part = None if prev is None else [float(np.mean([a[k] != b[k] for a, b in zip(tup, prev)])) for k in range(K)]
        prev = tup
        row = {"checkpoint": ck + 1, "updates": (ck + 1) * upd_per_ck, "play_steps": steps, "loss": round(loss, 5),
               "probe_rebuild_mse": round(err, 6), "code_flip_rate": None if flip is None else round(flip, 4), "pair_change_per_part": pair_change,
               "flip_per_part": None if per_part is None else [round(v, 4) for v in per_part],
               "distinct_tuples": len(set(tup)), "probe_identities": len(ident),
               "identities_split": sum(len({t for t, q in zip(tup, pid) if q == i}) > 1 for i in ident),
               "tuples_shared_by_identities": sum(len({q for t2, q in zip(tup, pid) if t2 == t}) > 1 for t in set(tup)), "codes_used_per_part": [len(set(c[:, k].tolist())) for k in range(K)],
               "restarts": enc.restarts, "seconds": round(time.monotonic() - t0, 1)}
        rows.append(row)
        print(row, flush=True)
        Path(out).write_text(json.dumps({"note": "Card 052 step 2a, tools/card052/drift.py", "seed": seed,
                                         "buffer": buffer, "update_every_steps": every,
                                         "updates_per_checkpoint": upd_per_ck, "rows": rows}, indent=1) + "\n")


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    Path(get("--out", "runs/052/drift.json")).parent.mkdir(parents=True, exist_ok=True)
    global DECAY
    DECAY = float(get("--decay", "1"))
    GN.TINT = float(get("--tint", str(GN.TINT)))      # diagnostics: nuisance switched off
    GN.NOISE = float(get("--noise", str(GN.NOISE)))
    Encoder.EMA = float(get("--ema", "0"))
    Encoder.RESTART_EVERY = int(get("--restart", "250"))
    run(int(get("--seed", "399")), int(get("--checkpoints", "40")), int(get("--buffer", "20000")),
        int(get("--every", "10")), int(get("--updates", "2000")), get("--out", "runs/052/drift.json"))


if __name__ == "__main__":
    main()
