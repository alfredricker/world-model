"""Card 091: the router, trained by a soft nearest-neighbour vote over stored tries (neighbourhood components analysis,
Goldberger et al. 2004: recall's leave-one-out fit, with a network as the metric).

  e = f(tokens, action):  each token (its code's encoder vector, its where, its kind) → a shared layer, plus the
                          action → two attention layers over the tokens → mean → 32 numbers
  P(outcome | query) = Σ_c softmax_c(−‖e_q − e_c‖₁ / τ + log w_c) · onehot(outcome_c)  over candidate stored tries c
                       of the same action

Training: every step one action; 256 queries (half drawn by sampling weight, half uniform over (front, held)
combinations, card 071's unit), 2,048 candidates drawn by sampling weight from the training worlds; the query itself is
never a candidate, and for half the queries no candidate shares its (front, held) combination (card 082.1: both kinds
of hidden query).

  bin/prun python tools/card091/router.py --train tier1,tier2,decoy --steps 4000 --out runs/091/router.pt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "091"
ACTS = (2, 3, 4, 5)
NOUT = {2: 3, 3: 4, 4: 4, 5: 4}
fn = torch.nn.functional


def code_vectors():
    """The agent's encoder (card 070's) on every catalogue code: (codes, 32)."""
    p = OUT / "code_vectors.npy"
    if p.exists():
        return np.load(p)
    sys.path.insert(0, str(ROOT / "tools" / "card066"))
    sys.argv += ["--tier", "1"]
    import tiers as T                                  # noqa: E402
    z, _, _ = T.encoder(ROOT / "runs" / "070" / "encoder.pt")
    np.save(p, z.astype(np.float32))
    return z.astype(np.float32)


class Router(nn.Module):
    def __init__(self, d=64, out=32):
        super().__init__()
        self.inp = nn.Linear(32 + 2 + 4, d)
        self.act = nn.Embedding(6, d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True)
        self.att = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, out)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def forward(self, Z, wx, wy, kind, act):
        x = torch.cat([Z, wx[..., None] / 6.0, wy[..., None] / 6.0, fn.one_hot(kind.long(), 4).float()], -1)
        h = self.inp(x) + self.act(act)[:, None]
        pad = kind == 0
        h = self.att(h, src_key_padding_mask=pad)
        keep = (~pad).float()[..., None]
        return self.out((h * keep).sum(1) / keep.sum(1).clamp_min(1))


class Data:
    def __init__(self, worlds, zc, dev="cuda"):
        parts = [dict(np.load(OUT / f"tokens_{w}.npz")) for w in worlds]
        self.d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.world = np.concatenate([np.full(len(p["act"]), i) for i, p in enumerate(parts)])
        self.zc = torch.as_tensor(zc, device=dev)
        self.dev = dev
        self.t = {k: torch.as_tensor(self.d[k].astype(np.int64), device=dev) for k in ("code", "wx", "wy", "kind", "act", "out")}
        self.w = torch.as_tensor(self.d["w"], dtype=torch.float32, device=dev)
        combo = self.d["front"].astype(np.int64) * 1000 + self.d["held"].astype(np.int64)
        self.combo = torch.as_tensor(combo, device=dev)
        self.rows = {a: np.flatnonzero(self.d["act"] == a) for a in ACTS}
        self.prior = {a: torch.as_tensor((np.bincount(self.d["out"][self.rows[a]], weights=self.d["w"][self.rows[a]],
                                                      minlength=NOUT[a]) + 1.0) / (self.d["w"][self.rows[a]].sum() + NOUT[a]),
                                         dtype=torch.float32, device=dev) for a in ACTS}
        self.combos = {}
        for a in ACTS:
            r = self.rows[a]
            u, inv = np.unique(combo[r], return_inverse=True)
            self.combos[a] = (r, inv, len(u))

    def batch(self, idx):
        i = torch.as_tensor(idx, device=self.dev)
        return (self.zc[self.t["code"][i]], self.t["wx"][i].float(), self.t["wy"][i].float(), self.t["kind"][i],
                self.t["act"][i])


def vote(eq, ec, wc, oc, nout, tau, prior, alpha, exclude=None):
    """Kernel-weighted counts with the action's outcome frequencies as a prior of weight alpha (recall's own form,
    P = (N + alpha prior) / (|N| + alpha)): far neighbours give the prior, not a confident guess. (q, nout)."""
    d = torch.cdist(eq, ec, p=1) / tau
    k = torch.exp(-d) * wc[None]
    if exclude is not None:
        k = k.masked_fill(exclude, 0.0)
    N = k @ fn.one_hot(oc, nout).float()
    return (N + alpha * prior[None]) / (N.sum(1, keepdim=True) + alpha)


def train(worlds, steps, out, seed=91, Q=256, C=2048, lr=1e-3, log_every=250):
    zc = code_vectors()
    D = Data(worlds, zc)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Router().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    hist = []
    for step in range(steps):
        a = ACTS[step % len(ACTS)]
        r, inv, nc = D.combos[a]
        wr = D.d["w"][r] / D.d["w"][r].sum()
        q1 = rng.choice(r, Q // 2, p=wr)
        pick = rng.integers(nc, size=Q // 2)                    # uniform over combinations
        order = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[order], np.arange(nc))
        sizes = np.bincount(inv, minlength=nc)
        q2 = r[order[starts[pick] + (rng.random(Q // 2) * sizes[pick]).astype(np.int64)]]
        qi = np.concatenate([q1, q2])
        ci = rng.choice(r, C, p=wr)
        eq, ec = net(*D.batch(qi)), net(*D.batch(ci))
        qt, ct = torch.as_tensor(qi, device="cuda"), torch.as_tensor(ci, device="cuda")
        excl = qt[:, None] == ct[None]
        hide = torch.zeros(Q, dtype=torch.bool, device="cuda")
        hide[Q // 2:] = True
        hide = hide[torch.randperm(Q, device="cuda")]
        excl |= hide[:, None] & (D.combo[qt][:, None] == D.combo[ct][None])
        P = vote(eq, ec, D.w[ct], D.t["out"][ct], NOUT[a], net.log_tau.exp(), D.prior[a], net.log_alpha.exp(), excl)
        loss = -torch.log(P[torch.arange(Q), D.t["out"][qt]]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0:
            hist.append({"step": step + 1, "loss": round(float(loss), 4), "tau": round(float(net.log_tau.exp()), 4),
                         "alpha": round(float(net.log_alpha.exp()), 4), "seconds": round(time.monotonic() - t0, 1)})
            print(hist[-1], flush=True)
    torch.save({"net": net.state_dict(), "worlds": worlds, "steps": steps, "hist": hist}, out)
    return net, hist


@torch.no_grad()
def embed(net, D, rows, bs=4096):
    out = []
    for i in range(0, len(rows), bs):
        out.append(net(*D.batch(rows[i:i + bs])))
    return torch.cat(out)


@torch.no_grad()
def evaluate(net, world, n_query=4000, K=256, seed=0, hide_combo=False):
    """Held-out queries from one world, the rest of that world's memory as candidates (top K by distance), per
    action: mean log-likelihood per try (by sampling weight), against two baselines: the action's outcome frequencies,
    and the outcome frequencies of the query's own (front, held) combination (identity, as recall's own level) with
    the action's frequencies as fallback."""
    zc = code_vectors()
    D = Data([world], zc)
    rng = np.random.default_rng(seed)
    rep = {}
    for a in ACTS:
        r = D.rows[a]
        q = rng.choice(r, min(n_query, len(r)), replace=False)
        rest = np.setdiff1d(r, q)
        eq, ec = embed(net, D, q), embed(net, D, rest)
        oc, wc = D.t["out"][torch.as_tensor(rest, device="cuda")], D.w[torch.as_tensor(rest, device="cuda")]
        cq = D.combo[torch.as_tensor(q, device="cuda")]
        cc = D.combo[torch.as_tensor(rest, device="cuda")]
        tau = net.log_tau.exp()
        lls = []
        for i in range(0, len(q), 256):
            d = torch.cdist(eq[i:i + 256], ec, p=1)
            if hide_combo:
                d = d.masked_fill(cq[i:i + 256][:, None] == cc[None], float("inf"))
            dk, ik = d.topk(K, largest=False)
            kk = torch.exp(-dk / tau) * wc[ik]
            N = (kk[..., None] * fn.one_hot(oc[ik], NOUT[a]).float()).sum(1)
            al = net.log_alpha.exp()
            P = (N + al * D.prior[a][None]) / (N.sum(1, keepdim=True) + al)
            yt = D.t["out"][torch.as_tensor(q[i:i + 256], device="cuda")]
            lls.append(torch.log(P[torch.arange(len(yt)), yt]))
        ll = torch.cat(lls).cpu().numpy()
        wq = D.d["w"][q]
        y = D.d["out"][q]
        f = np.bincount(D.d["out"][rest], weights=D.d["w"][rest], minlength=NOUT[a]) + 1e-4
        f = f / f.sum()
        base = np.log(f[y])
        combo_r = D.d["front"][rest].astype(np.int64) * 1000 + D.d["held"][rest]
        combo_q = D.d["front"][q].astype(np.int64) * 1000 + D.d["held"][q]
        idb = []
        for cqv, yv in zip(combo_q, y):
            m = combo_r == cqv
            if hide_combo or not m.any():
                idb.append(np.log(f[yv]))
            else:
                g = np.bincount(D.d["out"][rest][m], weights=D.d["w"][rest][m], minlength=NOUT[a]) + 1e-2 * f
                idb.append(np.log(g[yv] / g.sum()))
        avg = lambda v: round(float((np.asarray(v) * wq).sum() / wq.sum()), 4)
        rare = y != np.bincount(y, weights=wq).argmax()
        rep[str(a)] = {"queries": len(q), "router": avg(ll), "action frequencies": avg(base),
                       "own (front, held) combination": avg(idb),
                       "router on the rarer outcomes": round(float(ll[rare].mean()), 4) if rare.any() else None,
                       "combination on the rarer outcomes": round(float(np.asarray(idb)[rare].mean()), 4) if rare.any() else None}
    return rep


if __name__ == "__main__":
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    out = Path(get("--out", str(OUT / "router.pt")))
    if "--train" in args:
        net, hist = train(get("--train", "tier1,tier2,decoy").split(","), int(get("--steps", "4000")), out)
    else:
        ck = torch.load(out)
        net = Router().cuda()
        net.load_state_dict(ck["net"])
    net.eval()
    rep = {"router": str(out)}
    for w in get("--eval", "tier1,tier2,tier3").split(","):
        rep[w] = {"rows hidden": evaluate(net, w), "combinations hidden": evaluate(net, w, hide_combo=True)}
        print(w, json.dumps(rep[w]), flush=True)
    (OUT / (out.stem + "_eval.json")).write_text(json.dumps(rep, indent=1) + "\n")
