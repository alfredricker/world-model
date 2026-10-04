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
REL_W = 0.0                                            # step 2e: the relation term's weight (rho)
REL_N = 512
START_SHARE = 0.0                                      # step 2d: share of episodes that are play starts
START_LEN = 10
STARTS = [(ev, k, c) for c in DR.GN.TRAIN for ev, k in
          (("unlock_key", "door"), ("wrong_key", "door"), ("unlock_switch", "door"), ("switch", "switch"),
           ("open", "door"), ("pickup", "key"), ("pickup", "ball"), ("pickup", "box"))]


class Stream(DR.Stream):
    """Step 2a's stream, also returning the tries of pick up, drop and toggle with their outcome."""

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
        _, _, term, trunc, _ = self.env.step(a)
        self.t += 1
        tiles, pairs, tries = [], [], []
        if a in ACTS:
            f1 = self._front()
            k1 = None if f1 is None else tuple(f1.encode())
            h1 = self.env.carrying
            c1 = None if h1 is None else tuple(h1.encode())
            if k0 != k1:
                pairs.append((t0, self._tile(f1)))
            tries.append((a, t0, th, int(k0 != k1) + 2 * int(c0 != c1), (k0, c0)))
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

    def extra(self):
        tries = getattr(self, "_tries", None)
        if not tries:
            return 0.0
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
    rng = np.random.default_rng(seed + 2)
    prev, rows, steps = None, [], 0

    def play():
        nonlocal steps
        t, p = stream.step()
        tiles.extend(t)
        pairs.extend(p)
        for a, f, h, o, _ in stream.tries:
            buf[a][int(o > 0)].append((f, h, o))
        steps += 1

    while len(tiles) < 2000 or min(len(buf[a][0]) for a in ACTS) < PER:
        play()
    for ck in range(checkpoints):
        for g in enc.opt.param_groups:
            g["lr"] = 1e-3 * DR.DECAY ** ck
        effs, rels = [], []
        for _ in range(upd_per_ck):
            for _ in range(every):
                play()
            bi = rng.integers(len(tiles), size=256)
            pi = rng.integers(len(pairs), size=min(64, len(pairs))) if pairs else []
            enc._tries = draw(buf, rng)
            if REL_W > 0:                              # step 2e: toggle tries for the relation term, half changed
                ch, no = buf[5][1], buf[5][0]
                nc = min(REL_N // 2, len(ch))
                enc._rel = [ch[i] for i in rng.choice(len(ch), size=nc, replace=False)] + \
                           [no[i] for i in rng.choice(len(no), size=REL_N - nc, replace=False)]
            loss = enc.update([tiles[i] for i in bi], [pairs[i] for i in pi])
            effs.append(enc.last_effect)
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
               "restarts": enc.restarts, "seconds": round(time.monotonic() - t0, 1)}
        if CHECK is not None:                           # further measures at each checkpoint (step 2c)
            row.update(CHECK(enc))
        rows.append(row)
        print(row, flush=True)
        Path(out).write_text(json.dumps({"note": "Card 052 step 2b, tools/card052/effect.py", "seed": seed, "mu": MU,
                                         "ema": DR.Encoder.EMA, "start_share": START_SHARE, "relation_weight": REL_W, "buffer": buffer, "update_every_steps": every,
                                         "updates_per_checkpoint": upd_per_ck, "rows": rows}, indent=1) + "\n")


def main():
    global MU, START_SHARE, REL_W
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    Path(get("--out", "runs/052/effect.json")).parent.mkdir(parents=True, exist_ok=True)
    MU = float(get("--mu", "0.1"))
    DR.DECAY = float(get("--decay", "1"))
    DR.Encoder.EMA = float(get("--ema", "0"))
    DR.Encoder.RESTART_EVERY = int(get("--restart", "250"))
    START_SHARE = float(get("--starts", "0"))
    REL_W = float(get("--rel", "0"))
    run(int(get("--seed", "399")), int(get("--checkpoints", "40")), int(get("--buffer", "20000")),
        int(get("--every", "10")), int(get("--updates", "2000")), get("--out", "runs/052/effect.json"))


if __name__ == "__main__":
    main()
