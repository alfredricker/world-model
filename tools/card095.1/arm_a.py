"""Card 095.1, arm A: one network for every token of the state.

Each token: its vector (the world's encoder vector), a learned embedding of its place (169 places relative to the
agent and the hand, 170 in all) and the action; attention over the state's tokens (2 layers, as card 091's router).
Per token: the probability it changes, and what it becomes, per part of its vector (4 parts of 8) by card 038's four
ways, mixed by learned gates: keep it, copy it from another token (attention over the state's tokens), shift it by a
learned change, or set it. The result is snapped to the nearest token the encoder knows (L1).

Trained on the training episodes of tiers 1, 2 and the decoy world (tok.split), never tier 3.

  bin/prun python tools/card095.1/arm_a.py --steps 6000          → runs/095.1/arm_a.pt, runs/095.1/a_<tier>.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
import tok                                             # noqa: E402

NP, PW = 4, 8
KEEP, COPY, SHIFT, SET = range(4)
dev = "cuda"


class StateNet(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.inp = nn.Linear(32, d)
        self.place = nn.Embedding(170, d)
        self.act = nn.Embedding(6, d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True)
        self.att = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.change = nn.Linear(d, 1)
        self.gate = nn.Linear(d, NP * 4)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.shift, self.set = nn.Linear(d, 32), nn.Linear(d, 32)

    def forward(self, X, P, mask, a):
        h = self.inp(X) + self.place(P) + self.act(a)[:, None]
        h = self.att(h, src_key_padding_mask=~mask)
        logit = self.change(h)[..., 0]
        s = (self.q(h) @ self.k(h).transpose(1, 2)) / h.shape[-1] ** 0.5
        eye = torch.eye(h.shape[1], dtype=torch.bool, device=h.device)[None]
        s = s.masked_fill(~mask[:, None, :] | eye, -1e9)
        ptr = torch.softmax(s, -1)                     # (n, L, L): which token each token copies from
        copy = ptr @ X
        g = torch.softmax(self.gate(h).view(*h.shape[:2], NP, 4), -1)
        ways = torch.stack([X, copy, X + self.shift(h), self.set(h)], -1).view(*h.shape[:2], NP, PW, 4)
        after = (ways * g[..., None, :]).sum(-1).reshape(*h.shape[:2], 32)
        return logit, after, ptr, g


class Data:
    def __init__(self, world, which):
        z = tok.load(world)
        H, P, C, A = tok.build(z)
        tr, te = tok.split(z)
        s = tr if which == "train" else te
        self.idx = np.flatnonzero(s)
        self.arr = torch.as_tensor(z["arr"], device=dev)
        known = np.unique(z["app"])
        self.known = torch.as_tensor(known, device=dev)
        t = lambda x, dt=torch.long: torch.as_tensor(x[s], dtype=dt, device=dev)
        self.H, self.P, self.C, self.A = t(H), t(P), t(C, torch.bool), t(A)
        self.act, self.w = t(z["act"]), t(z["w"], torch.float32)
        self.n = len(self.idx)

    def batch(self, i):
        H = self.H[i]
        mask = H >= 0
        X = self.arr[H.clamp_min(0)] * mask[..., None]
        return X, self.P[i], mask, self.act[i], self.C[i], self.A[i], self.w[i]


def loss_of(net, d, i):
    X, P, mask, a, C, A, w = d.batch(i)
    logit, after, _, _ = net(X, P, mask, a)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, C.float(), reduction="none")
    lc = ((bce * mask).sum(1) * w).sum() / w.sum()
    ch = C & mask
    tgt = d.arr[A.clamp_min(0)]
    l1 = (after - tgt).abs().sum(-1)
    la = ((l1 * ch).sum(1) * w).sum() / w.sum()
    return lc + 0.1 * la, float(lc), float(la)


def train(steps, out, seed=95, bs=512, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [Data(w, "train") for w in ("tier1", "tier2", "decoy")]
    size = np.array([d.n for d in data], np.float64)
    net = StateNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    for step in range(steps):
        d = data[rng.choice(len(data), p=size / size.sum())]
        i = torch.as_tensor(rng.integers(d.n, size=bs), device=dev)
        loss, lc, la = loss_of(net, d, i)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print({"step": step + 1, "change": round(lc, 5), "after_l1": round(la, 4),
                   "seconds": round(time.monotonic() - t0, 1)}, flush=True)
    torch.save({"net": net.state_dict(), "steps": steps}, out)
    return net


@torch.no_grad()
def predict(net, world, bs=4096):
    d = Data(world, "test")
    K = d.arr[d.known]
    out = {k: [] for k in ("p", "after", "ptr_front", "gate")}
    for s in range(0, d.n, bs):
        i = torch.arange(s, min(s + bs, d.n), device=dev)
        X, P, mask, a, C, A, w = d.batch(i)
        logit, after, ptr, g = net(X, P, mask, a)
        snap = d.known[torch.cdist(after.reshape(-1, 32), K, p=1).argmin(1)].view(after.shape[:2])
        fslot = (P == 71) & mask                       # the front's slot (report only: the net is not told)
        pf = (ptr * fslot[:, None, :]).sum(-1)
        out["p"].append(torch.sigmoid(logit).masked_fill(~mask, 0).cpu())
        out["after"].append(torch.where(mask, snap, -1).cpu())
        out["ptr_front"].append(pf.cpu())
        out["gate"].append(g.argmax(-1).cpu())
    r = {k: torch.cat(v).numpy() for k, v in out.items()}
    r["idx"] = d.idx
    return r


if __name__ == "__main__":
    args = sys.argv[1:]
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 6000
    out = tok.RUNS / "arm_a.pt"
    if "--eval-only" in args:
        net = StateNet().to(dev)
        net.load_state_dict(torch.load(out)["net"])
    else:
        net = train(steps, out)
    net.eval()
    for w in ("tier1", "tier2", "tier3"):
        r = predict(net, w)
        np.savez_compressed(tok.RUNS / f"a_{w}.npz", **r)
        print(w, "predicted", len(r["idx"]), flush=True)
