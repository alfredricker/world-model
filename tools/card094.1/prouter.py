"""Card 094.1: card 091's router with each token's place relative to the agent as an input.

Tokens: the front tile (at the front's place), the held tile (the hand, a place of its own), and up to 6 attended
tokens of the believed view at their places (pkeys.py); each token's input is its vector + its role + a learned
embedding of its place (one per place of the 13 x 13 view, Shaw et al. 2018's per-offset representation, here the
offset from the agent). The vote, τ, α and the training (tiers 1, 2 and the decoy world; queries half by tries, half
by (front, held) combination; half the queries with their whole combination hidden) are card 091's. With tens of
thousands of keys per action, each step votes over the queries' own combinations and a random 16,384 others.

  bin/prun python tools/card094.1/prouter.py --train tier1,tier2,decoy --steps 4000 --out runs/094.1/prouter.pt
  bin/prun python tools/card094.1/prouter.py --out runs/094.1/prouter.pt --eval tier1,tier2      (evaluation)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card091"))
import krouter as KR                                   # noqa: E402

OUT = ROOT / "runs" / "094.1"
ACTS = KR.ACTS
NV, FRONT_PLACE, HAND_PLACE = 169, None, 169           # FRONT_PLACE set from the keys (the front's place index)
NCAND = 16384


class PRouter(KR.KRouter):
    def __init__(self, d=64, out=32):
        super().__init__(d, out)
        self.place = nn.Embedding(NV + 1, d)

    def forward(self, Z, role, mask, act, place):
        r = role.expand(Z.shape[0], -1) if role.dim() == 1 else role
        h = self.inp(Z) + self.role(r) + self.act(act)[:, None] + self.place(place)
        h = self.att(h, src_key_padding_mask=~mask)
        m = mask.float()[..., None]
        return self.out((h * m).sum(1) / m.sum(1).clamp_min(1))


def tokens(K, a, dev, front_place):
    if a == 2:
        Z = K["2_F"][:, None]
        mask = np.ones((len(Z), 1), bool)
        role = np.array([0])
        place = np.full((len(Z), 1), front_place)
    else:
        Z = np.concatenate([K[f"{a}_F"][:, None], K[f"{a}_H"][:, None], K[f"{a}_V"]], 1)
        mask = np.concatenate([np.ones((len(Z), 2), bool), K[f"{a}_vm"]], 1)
        role = np.array([0, 1] + [2] * K[f"{a}_V"].shape[1])
        place = np.concatenate([np.full((len(Z), 1), front_place), np.full((len(Z), 1), HAND_PLACE),
                                np.where(K[f"{a}_vm"], K[f"{a}_P"], 0)], 1)
    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=dev)
    return t(Z, torch.float32), t(role, torch.long), t(mask, torch.bool), t(place, torch.long)


class World(KR.World):
    def __init__(self, name, front_place, dev="cuda", folder=OUT, prefix="pkeys"):
        K = dict(np.load(folder / f"{prefix}_{name}.npz"))
        self.name, self.K = name, K
        self.tok, self.C, self.combo, self.fid = {}, {}, {}, {}
        for a in ACTS:
            self.tok[a] = tokens(K, a, dev, front_place)
            self.C[a] = torch.as_tensor(K[f"{a}_C"], dtype=torch.float32, device=dev)
            fid = K[f"{a}_fid"].astype(np.int64)
            hid = K[f"{a}_hid"].astype(np.int64) if a != 2 else np.zeros_like(fid)
            self.combo[a] = torch.as_tensor((fid - fid.min()) * 10 ** 6 + (hid - hid.min()), device=dev)
            self.fid[a] = torch.as_tensor(fid, device=dev)


def embed(net, tok, a, idx=None, bs=8192):
    Z, role, mask, place = tok
    if idx is not None:
        Z, mask, place = Z[idx], mask[idx], place[idx]
    out = []
    for i in range(0, len(Z), bs):
        act = torch.full((len(Z[i:i + bs]),), a, dtype=torch.long, device=Z.device)
        out.append(net(Z[i:i + bs], role, mask[i:i + bs], act, place[i:i + bs]))
    return torch.cat(out)


def train(names, steps, out, front_place, seed=91, Q=128, lr=1e-3, log_every=250):
    worlds = [World(n, front_place) for n in names]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = PRouter().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    hist = []
    size = np.array([sum(float(w.C[a].sum()) for a in ACTS) for w in worlds])
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
        qs = np.concatenate([q1, q2])
        own = np.flatnonzero(np.isin(combo, combo[qs]))
        if len(own) > NCAND:
            own = rng.choice(own, NCAND, replace=False)
        cand = np.unique(np.concatenate([qs, own, rng.choice(n, min(n, NCAND), replace=False)]))
        ci = torch.as_tensor(cand, device="cuda")
        pos = {int(k): i for i, k in enumerate(cand)}
        qi = torch.as_tensor([pos[int(k)] for k in qs], device="cuda")
        e = embed(net, w.tok[a], a, ci)
        Cc = C[ci]
        cmb = w.combo[a][ci]
        excl = qi[:, None] == torch.arange(len(ci), device="cuda")[None]
        hide = torch.rand(len(qi), device="cuda") < 0.5
        excl |= hide[:, None] & (cmb[qi][:, None] == cmb[None])
        P = KR.vote(e[qi], e, Cc, KR.prior_of(C), net.log_tau.exp(), net.log_alpha.exp(), excl)
        tq = Cc[qi] / Cc[qi].sum(1, keepdim=True)
        loss = -(tq * torch.log(P.clamp_min(1e-12))).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0:
            hist.append({"step": step + 1, "loss": round(float(loss), 4), "tau": round(float(net.log_tau.exp()), 4),
                         "alpha": round(float(net.log_alpha.exp()), 4), "seconds": round(time.monotonic() - t0, 1)})
            print(hist[-1], flush=True)
    torch.save({"net": net.state_dict(), "worlds": names, "steps": steps, "hist": hist, "front_place": front_place}, out)
    return net


@torch.no_grad()
def evaluate(net, w, a_embed, hide, n_query=4000, seed=0):
    """Mean log-likelihood per stored try of held-out keys, sampled by tries.
    hide: "combination" / "front" (card 091's: the query's combination, or every key with its front, hidden; the vote
    over every other key); "identity" (the run-time vote, card 091.1: over the keys of the query's own combination,
    the query's own try taken out of its key's counts)."""
    rng = np.random.default_rng(seed)
    rep = {}
    for a in ACTS:
        C = w.C[a]
        n = len(C)
        cw = C.sum(1).cpu().numpy()
        q = torch.as_tensor(rng.choice(n, n_query, p=cw / cw.sum()), device="cuda")    # tries, with replacement
        e = a_embed(w, a)
        f = KR.prior_of(C)
        tau, alpha = net.log_tau.exp(), net.log_alpha.exp()
        lls = []
        for i in range(0, len(q), 256):
            qi = q[i:i + 256]
            tq = C[qi] / C[qi].sum(1, keepdim=True)
            if hide == "identity":
                same = w.combo[a][qi][:, None] == w.combo[a][None]
                k = torch.exp(-torch.cdist(e[qi], e, p=1) / tau) * same
                N = k @ C
                N = N - k[torch.arange(len(qi)), qi][:, None] * tq          # the query's own try left out
                P = (N.clamp_min(0) + alpha * f[None]) / (N.clamp_min(0).sum(1, keepdim=True) + alpha)
            else:
                key = w.combo[a] if hide == "combination" else w.fid[a]
                P = KR.vote(e[qi], e, C, f, tau, alpha, key[qi][:, None] == key[None])
            lls.append((tq * torch.log(P.clamp_min(1e-12))).sum(1))
        ll = torch.cat(lls)
        tq = C[q] / C[q].sum(1, keepdim=True)
        rep[str(a)] = {"router": round(float(ll.mean()), 4),
                       "action frequencies": round(float((tq * torch.log(f)[None]).sum(1).mean()), 4)}
    return rep


def load(path, dev="cuda"):
    ck = torch.load(path, map_location=dev)
    net = PRouter().to(dev)
    net.load_state_dict(ck["net"])
    return net.eval(), ck["front_place"]


if __name__ == "__main__":
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    out = Path(get("--out", str(OUT / "prouter.pt")))
    front_place = int(get("--front", "-1"))
    if "--train" in args:
        net = train(get("--train", "tier1,tier2,decoy").split(","), int(get("--steps", "4000")), out, front_place)
    else:
        net, front_place = load(out)
    net.eval()
    base = KR.load(str(ROOT / "runs" / "091" / "krouter_a.pt"))
    rep = {"router": str(out)}
    for name in get("--eval", "tier1,tier2").split(","):
        wp = World(name, front_place)
        wb = KR.World(name)
        ep = lambda w, a: embed(net, w.tok[a], a)
        eb = lambda w, a: KR.embed(base, w.tok[a], a)
        rep[name] = {}
        for h in ("combination", "front"):
            rep[name][h] = {"positions": evaluate(net, wp, ep, h), "card 091 (set form)": evaluate(base, wb, eb, h)}
            print(name, h, json.dumps(rep[name][h]), flush=True)
    (OUT / (out.stem + "_eval.json")).write_text(json.dumps(rep, indent=1) + "\n")
