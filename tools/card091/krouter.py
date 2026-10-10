"""Card 091: the router in recall's key form (what the agent queries): the front tile, the held tile and the believed
view's tiles, each the agent's 32-number vector with a role (front, hand, view); attention over them; an embedding.
The prior for a key is a kernel vote over the same world's other stored keys, with the action's outcome frequencies
as a prior of learned weight alpha:

    P(c | q) = (Σ_k exp(−‖e_q − e_k‖₁ / τ) C_k,c + α f_c) / (Σ_k exp(−‖e_q − e_k‖₁ / τ) |C_k| + α)

Trained on tiers 1, 2 and the decoy world's keys (memory 066), never tier 3. Each step: one world (by its keys) and
one action; queries half by tries, half uniform over (front, held) combinations; the query key itself is never a
candidate, and for half the queries no key of its combination is (card 082.1).

  bin/prun python tools/card091/krouter.py --train tier1,tier2,decoy --steps 4000 --out runs/091/krouter.pt
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("WM_KR_DIR", ROOT / "runs" / "091"))   # card 094: object-file keys in runs/094
ACTS = (2, 3, 4, 5)
NOUT = {2: 3, 3: 4, 4: 4, 5: 4}
fn = torch.nn.functional


class KRouter(nn.Module):
    def __init__(self, d=64, out=32):
        super().__init__()
        self.inp = nn.Linear(32, d)
        self.role = nn.Embedding(3, d)                 # front, hand, view
        self.act = nn.Embedding(6, d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True)
        self.att = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(d, out)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def forward(self, Z, role, mask, act):
        """Z (n, L, 32) token vectors; role (L,) or (n, L); mask (n, L) True where present; act (n,)."""
        h = self.inp(Z) + self.role(role.expand(Z.shape[0], -1) if role.dim() == 1 else role) + self.act(act)[:, None]
        h = self.att(h, src_key_padding_mask=~mask)
        m = mask.float()[..., None]
        return self.out((h * m).sum(1) / m.sum(1).clamp_min(1))


def tokens(K, a, dev):
    """A world's keys of action a as (Z, role, mask) tensors."""
    if a == 2:
        Z = K["2_F"][:, None]
        mask = np.ones((len(Z), 1), bool)
        role = np.array([0])
    else:
        Z = np.concatenate([K[f"{a}_F"][:, None], K[f"{a}_H"][:, None], K[f"{a}_V"]], 1)
        mask = np.concatenate([np.ones((len(Z), 2), bool), K[f"{a}_vm"]], 1)
        role = np.array([0, 1] + [2] * K[f"{a}_V"].shape[1])
    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=dev)
    return t(Z, torch.float32), t(role, torch.long), t(mask, torch.bool)


class World:
    def __init__(self, name, dev="cuda"):
        K = dict(np.load(OUT / f"keys_{name}.npz"))
        self.name, self.K = name, K
        self.tok, self.C, self.combo, self.fid = {}, {}, {}, {}
        for a in ACTS:
            self.tok[a] = tokens(K, a, dev)
            self.C[a] = torch.as_tensor(K[f"{a}_C"], dtype=torch.float32, device=dev)
            fid = K[f"{a}_fid"].astype(np.int64)
            hid = K[f"{a}_hid"].astype(np.int64) if a != 2 else np.zeros_like(fid)
            self.combo[a] = torch.as_tensor((fid - fid.min()) * 10 ** 6 + (hid - hid.min()), device=dev)
            self.fid[a] = torch.as_tensor(fid, device=dev)


def embed(net, tok, a, idx=None, bs=8192):
    Z, role, mask = tok
    if idx is not None:
        Z, mask = Z[idx], mask[idx]
    out = []
    for i in range(0, len(Z), bs):
        act = torch.full((len(Z[i:i + bs]),), a, dtype=torch.long, device=Z.device)
        out.append(net(Z[i:i + bs], role, mask[i:i + bs], act))
    return torch.cat(out)


def vote(eq, ec, Cc, prior, tau, alpha, exclude=None):
    k = torch.exp(-torch.cdist(eq, ec, p=1) / tau)
    if exclude is not None:
        k = k.masked_fill(exclude, 0.0)
    N = k @ Cc
    return (N + alpha * prior[None]) / (N.sum(1, keepdim=True) + alpha)


def prior_of(C):
    f = C.sum(0) + 1.0
    return f / f.sum()


def train(names, steps, out, seed=91, Q=128, lr=1e-3, log_every=250):
    worlds = [World(n) for n in names]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = KRouter().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    hist = []
    size = np.array([sum(len(w.C[a]) for a in ACTS) for w in worlds], np.float64)
    for step in range(steps):
        a = ACTS[step % len(ACTS)]
        w = worlds[rng.choice(len(worlds), p=size / size.sum())]
        C = w.C[a]
        n = len(C)
        if n < 3:
            continue
        cw = C.sum(1).cpu().numpy()
        q1 = rng.choice(n, Q // 2, p=cw / cw.sum())
        combo = w.combo[a].cpu().numpy()
        u, inv = np.unique(combo, return_inverse=True)
        pick = rng.integers(len(u), size=Q // 2)
        q2 = np.array([rng.choice(np.flatnonzero(inv == p)) for p in pick])
        qi = torch.as_tensor(np.concatenate([q1, q2]), device="cuda")
        e = embed(net, w.tok[a], a)
        excl = qi[:, None] == torch.arange(n, device="cuda")[None]
        hide = torch.rand(len(qi), device="cuda") < 0.5
        excl |= hide[:, None] & (w.combo[a][qi][:, None] == w.combo[a][None])
        P = vote(e[qi], e, C, prior_of(C), net.log_tau.exp(), net.log_alpha.exp(), excl)
        tq = C[qi] / C[qi].sum(1, keepdim=True)
        loss = -(tq * torch.log(P.clamp_min(1e-12))).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0:
            hist.append({"step": step + 1, "loss": round(float(loss), 4), "tau": round(float(net.log_tau.exp()), 4),
                         "alpha": round(float(net.log_alpha.exp()), 4), "seconds": round(time.monotonic() - t0, 1)})
            print(hist[-1], flush=True)
    torch.save({"net": net.state_dict(), "worlds": names, "steps": steps, "hist": hist}, out)
    return net


@torch.no_grad()
def evaluate(net, name, hide, n_query=4000, seed=0):
    """Mean log-likelihood per stored try of held-out keys (sampled by tries), the query's combination ("combination")
    or every key with its front tile ("front") hidden; against the action's outcome frequencies."""
    w = World(name)
    rng = np.random.default_rng(seed)
    rep = {}
    for a in ACTS:
        C = w.C[a]
        n = len(C)
        cw = C.sum(1).cpu().numpy()
        q = torch.as_tensor(rng.choice(n, min(n_query, n), p=cw / cw.sum()), device="cuda")
        e = embed(net, w.tok[a], a)
        f = prior_of(C)
        lls = []
        key = w.combo[a] if hide == "combination" else w.fid[a]
        for i in range(0, len(q), 512):
            qi = q[i:i + 512]
            P = vote(e[qi], e, C, f, net.log_tau.exp(), net.log_alpha.exp(), key[qi][:, None] == key[None])
            tq = C[qi] / C[qi].sum(1, keepdim=True)
            lls.append((tq * torch.log(P.clamp_min(1e-12))).sum(1))
        ll = torch.cat(lls)
        tq = C[q] / C[q].sum(1, keepdim=True)
        rep[str(a)] = {"router": round(float(ll.mean()), 4),
                       "action frequencies": round(float((tq * torch.log(f)[None]).sum(1).mean()), 4)}
    return rep


def load(path, dev="cuda"):
    ck = torch.load(path, map_location=dev)
    net = KRouter().to(dev)
    net.load_state_dict(ck["net"])
    return net.eval()


if __name__ == "__main__":
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    out = Path(get("--out", str(OUT / "krouter.pt")))
    net = train(get("--train", "tier1,tier2,decoy").split(","), int(get("--steps", "4000")), out) if "--train" in args \
        else load(out)
    net.eval()
    rep = {"router": str(out)}
    for name in get("--eval", "tier1,tier2,tier3").split(","):
        rep[name] = {h: evaluate(net, name, h) for h in ("combination", "front")}
        print(name, json.dumps(rep[name]), flush=True)
    (OUT / (out.stem + "_eval.json")).write_text(json.dumps(rep, indent=1) + "\n")
