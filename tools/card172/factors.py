"""Card 172: a scene state whose coordinates are found from how frames change over time.

  factors.py g                                   fit the pose term alone, freeze it -> runs/172/g.pt (+ its accuracy)
  factors.py fit --K 32 --seed 0 --minutes 5 --out short      label-free fit -> runs/172/<out>.pt
  factors.py fit ... --steps 5000                the same for a fixed number of steps
  factors.py fit ... --bound 1                   the gate's upper bound: pairs chosen by the true scene state

A frame's 1,024 frozen patch vectors (64 numbers each, runs/132/pca.pt), each with its grid position, are pooled by K
learned queries into K numbers in [-1, 1]. Training: for a moment t of an episode, pick the true near moment (within 5
steps) out of 17: itself, the 12 moments 6-20 steps away whose pose is nearest t's, and 4 moments of other episodes.
score(a, b) = -sum_k |s_a[k] - s_b[k]| / tau + g(pose_a, pose_b), g frozen. No contact value, no truth, no decoder.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/172"
DEV = "cuda"
T = 201
WIN, RING_LO, RING_HI = 5, 6, 20          # steps of 5 frames
N_RING, N_OTHER, POOL_OTHER = 12, 4, 64
LAMBDA = 0.01
BUF, SWAP, SWAP_EVERY, BATCH = 96, 8, 40, 96
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


class Factors(nn.Module):
    def __init__(self, K=32, d_in=64, d_hid=128):
        super().__init__()
        self.K = K
        g = torch.linspace(-1.0, 1.0, 32)
        self.register_buffer("grid", torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(1024, 2))
        self.patch_trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hid), nn.GELU())
        self.pos_proj = nn.Linear(2, d_hid, bias=False)
        self.trunk2 = nn.Sequential(nn.LayerNorm(d_hid), nn.Linear(d_hid, d_hid), nn.GELU())
        self.query = nn.Linear(d_hid, K)                               # one learned query per coordinate
        self.read = nn.Sequential(nn.LayerNorm(d_hid + 2), nn.Linear(d_hid + 2, d_hid), nn.GELU())   # shared by all
        self.out_w = nn.Parameter(torch.randn(K, d_hid) * 0.05)
        self.out_b = nn.Parameter(torch.zeros(K))
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        """z [N, 1024, 64] -> s [N, K] in [-1, 1]."""
        p = self.trunk2(self.patch_trunk(z) + self.pos_proj(self.grid)[None])
        w = torch.softmax(self.query(p), dim=1)                        # [N, 1024, K]
        pooled = torch.einsum("npk,nph->nkh", w, p)
        ctr = torch.einsum("npk,pd->nkd", w, self.grid)
        u = self.read(torch.cat([pooled, ctr], -1))                    # [N, K, H]
        return torch.tanh((u * self.out_w[None]).sum(-1) + self.out_b[None])


class PoseTerm(nn.Module):
    """g(pose_a, pose_b): what the two poses alone say about the moments being near in time."""
    def __init__(self, d=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, pa, pb):
        return self.net(torch.cat([pa, pb, pa - pb, (pa - pb).abs()], -1)).squeeze(-1)


def load_meta(name):
    m = np.load(OUT / f"meta_{name}.npz")
    return {k: m[k] for k in m.files}


def pose_stats():
    p = load_meta("train")["pose"].reshape(-1, 12)
    return p.mean(0).astype(np.float32), (p.std(0) + 1e-6).astype(np.float32)


WIN, RING_LO, RING_HI = int(arg("--win", WIN)), int(arg("--ring_lo", RING_LO)), int(arg("--ring_hi", RING_HI))
RING = torch.tensor([o for o in range(-RING_HI, RING_HI + 1) if abs(o) >= RING_LO])      # 30 offsets by default
G_NAME = arg("--g", "g")


def sample(pose, n, gen):
    """pose [E, T, 12] standardised, on the device. -> episode [n], t [n], candidates' (episode, t) [n, 17]; column 0 is true."""
    E = pose.shape[0]
    dev = pose.device
    e = torch.randint(0, E, (n,), device=dev, generator=gen)
    t = torch.randint(RING_HI, T - RING_HI, (n,), device=dev, generator=gen)
    d = torch.randint(1, WIN + 1, (n,), device=dev, generator=gen) * (torch.randint(0, 2, (n,), device=dev, generator=gen) * 2 - 1)
    ring_t = t[:, None] + RING.to(dev)[None]                                               # [n, 30]
    dp = (pose[e[:, None], ring_t] - pose[e, t][:, None]).abs().sum(-1)
    near = ring_t.gather(1, dp.topk(N_RING, dim=1, largest=False).indices)                 # the nearest in pose
    e2 = (e[:, None] + torch.randint(1, E, (n, POOL_OTHER), device=dev, generator=gen)) % E
    t2 = torch.randint(0, T, (n, POOL_OTHER), device=dev, generator=gen)
    dp2 = (pose[e2, t2] - pose[e, t][:, None]).abs().sum(-1)
    j = dp2.topk(N_OTHER, dim=1, largest=False).indices
    ce = torch.cat([e[:, None].expand(-1, 1 + N_RING), e2.gather(1, j)], 1)
    ct = torch.cat([(t + d)[:, None], near, t2.gather(1, j)], 1)
    return e, t, ce, ct


def sample_by_truth(pose, truth, n, gen):
    """The gate's upper bound: the true one shares the scene's state with t, 12 false ones differ in exactly one variable."""
    E = pose.shape[0]
    dev = pose.device
    e = torch.randint(0, E, (n,), device=dev, generator=gen)
    t = torch.randint(0, T, (n,), device=dev, generator=gen)
    cube, dr, wi, lk = truth
    D = torch.stack([(cube[e] - cube[e, t][:, None]).norm(dim=-1) > 0.01, (dr[e] - dr[e, t][:, None]).abs() > 0.01,
                     (wi[e] - wi[e, t][:, None]).abs() > 0.01, lk[e][..., 0] != lk[e, t][:, None, 0], lk[e][..., 1] != lk[e, t][:, None, 1]], -1)
    nd = D.sum(-1)                                                                          # [n, T]
    r = torch.rand(n, T, device=dev, generator=gen)
    same = (nd == 0); same[torch.arange(n, device=dev), t] = False
    has = same.any(1)
    tp = torch.where(has, (r + same.float() * 2).argmax(1), (t + 1).clamp(max=T - 1))
    one = (r + (nd == 1).float() * 2 + (nd >= 1).float()).topk(N_RING, dim=1).indices      # one-variable moments first
    e2 = (e[:, None] + torch.randint(1, E, (n, N_OTHER), device=dev, generator=gen)) % E
    t2 = torch.randint(0, T, (n, N_OTHER), device=dev, generator=gen)
    ce = torch.cat([e[:, None].expand(-1, 1 + N_RING), e2], 1)
    ct = torch.cat([tp[:, None], one, t2], 1)
    return e, t, ce, ct


def fit_g():
    torch.manual_seed(0)
    mu, sd = pose_stats()
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    P = {n: torch.from_numpy((load_meta(n)["pose"] - mu) / sd).to(DEV) for n in ("train", "val")}
    g = PoseTerm().to(DEV)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-4)
    for st in range(1, 4001):
        e, t, ce, ct = sample(P["train"], 1024, gen)
        logit = g(P["train"][e, t][:, None].expand(-1, ce.shape[1], -1), P["train"][ce, ct])
        loss = F.cross_entropy(logit, torch.zeros(len(e), dtype=torch.long, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 1000 == 0:
            print(f"g step {st} loss {loss.item():.3f}", flush=True)
    g.eval()
    acc = {}
    with torch.no_grad():
        for n in ("train", "val"):
            e, t, ce, ct = sample(P[n], 40000, gen)
            logit = g(P[n][e, t][:, None].expand(-1, ce.shape[1], -1), P[n][ce, ct])
            dp = (P[n][ce, ct] - P[n][e, t][:, None]).abs().sum(-1)
            acc[n] = dict(g_picks_true_pct=round(100 * float((logit.argmax(1) == 0).float().mean()), 1),
                          nearest_pose_picks_true_pct=round(100 * float((dp.argmin(1) == 0).float().mean()), 1),
                          g_loss=round(float(F.cross_entropy(logit, torch.zeros(len(e), dtype=torch.long, device=DEV))), 3))
    acc["chance_pct"] = round(100.0 / ce.shape[1], 1); acc["chance_loss"] = round(float(np.log(ce.shape[1])), 3)
    acc["win"], acc["ring"] = WIN, [RING_LO, RING_HI]
    torch.save(dict(state_dict=g.state_dict(), mu=mu, sd=sd, acc=acc, win=WIN, ring=[RING_LO, RING_HI]), OUT / f"{G_NAME}.pt")
    import json
    json.dump(acc, open(OUT / f"{G_NAME}.json", "w"), indent=1)
    print(json.dumps(acc), flush=True)


def fit():
    K, seed, out = int(arg("--K", 32)), int(arg("--seed", 0)), arg("--out", "fit")
    minutes, steps, bound = float(arg("--minutes", 0)), int(arg("--steps", 0)), bool(int(arg("--bound", 0)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    ck = torch.load(OUT / f"{G_NAME}.pt", map_location=DEV, weights_only=False)
    assert ck.get("win", 5) == WIN and ck.get("ring", [6, 20]) == [RING_LO, RING_HI], "pose term fitted for other moments"
    g = PoseTerm().to(DEV); g.load_state_dict(ck["state_dict"]); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    meta = load_meta("train")
    X = np.load(OUT / "feat_train.npy", mmap_mode="r")
    NE = X.shape[0]
    pose_all = ((meta["pose"] - ck["mu"]) / ck["sd"]).astype(np.float32)
    ids = rng.choice(NE, BUF, replace=False)
    feat = torch.from_numpy(np.stack([np.asarray(X[i]) for i in ids])).to(DEV)             # fp16 [BUF, T, 1024, 64]
    pose = torch.from_numpy(pose_all[ids]).to(DEV)
    tr_keys = ("cube", "drawer", "window", "locks")
    truth = [torch.from_numpy(np.asarray(meta[k][ids], np.float32 if k != "locks" else np.int64)).to(DEV) for k in tr_keys] if bound else None

    net = Factors(K=K).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    t0, st, acc_run, last = time.time(), 0, 0.0, time.time()
    zero = torch.zeros(BATCH, dtype=torch.long, device=DEV)
    while True:
        st += 1
        if st % SWAP_EVERY == 0:                                                           # short sequential reads from disk
            slots, new = rng.choice(BUF, SWAP, replace=False), rng.choice(NE, SWAP, replace=False)
            feat[torch.from_numpy(slots).to(DEV)] = torch.from_numpy(np.stack([np.asarray(X[i]) for i in new])).to(DEV)
            pose[torch.from_numpy(slots).to(DEV)] = torch.from_numpy(pose_all[new]).to(DEV)
            if bound:
                for k_i, k in enumerate(tr_keys):
                    truth[k_i][torch.from_numpy(slots).to(DEV)] = torch.from_numpy(np.asarray(meta[k][new], np.float32 if k != "locks" else np.int64)).to(DEV)
        e, t, ce, ct = sample_by_truth(pose, truth, BATCH, gen) if bound else sample(pose, BATCH, gen)
        C = ce.shape[1]
        frames = torch.cat([feat[e, t], feat[ce.reshape(-1), ct.reshape(-1)]], 0).float()
        s = net(frames)
        sa, sc = s[:BATCH], s[BATCH:].reshape(BATCH, C, K)
        tau = net.log_tau.exp().clamp(0.02, 20.0)
        logit = -(sa[:, None] - sc).abs().sum(-1) / tau + g(pose[e, t][:, None].expand(-1, C, -1), pose[ce, ct])
        nce = F.cross_entropy(logit, zero)
        spread = s.std(0).sum()
        loss = nce + LAMBDA * spread
        lr = 1e-3 * min(1.0, st / 200.0)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        acc_run = 0.98 * acc_run + 0.02 * float((logit.argmax(1) == 0).float().mean())
        if time.time() - last > 30:
            last = time.time()
            print(f"step {st} {time.time()-t0:.0f}s nce {nce.item():.3f} picks true {100*acc_run:.1f}% spread {spread.item():.2f} tau {tau.item():.3f} "
                  f"used {(s.std(0) > 0.02).sum().item()}/{K}", flush=True)
        if (steps and st >= steps) or (minutes and time.time() - t0 > minutes * 60):
            break
    torch.save(dict(state_dict=net.state_dict(), K=K, seed=seed, steps=st, sec=round(time.time() - t0, 1), bound=bound,
                    picks_true_pct=round(100 * acc_run, 1)), OUT / f"{out}.pt")
    print(f"saved {out}.pt steps {st} in {time.time()-t0:.0f}s picks true {100*acc_run:.1f}%", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fit_g() if sys.argv[1] == "g" else fit()
