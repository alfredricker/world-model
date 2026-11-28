"""Card 173: card 172's encoder and frozen pose term; the temperature fixed, a cost on change between consecutive steps.

  factors173.py --K 8 --seed 0 --steps 4000 --out k8_s0        label-free fit -> runs/173/<out>.pt
  factors173.py --K 8 --seed 0 --steps 4000 --bound 1 --out bound8   the gate's upper bound: pairs chosen by the true
                                                                scene state, and no pose term in the score

score(a, b) = -sum_k |s_a[k] - s_b[k]| / 0.5 + g(pose_a, pose_b)        (g frozen, card 172's; left out for the bound)
loss        = pick the true near moment out of 17 (InfoNCE) + 1.0 * mean over the batch of sum_k |s_{t+1}[k] - s_t[k]|
No spread penalty, no learned temperature, no contact value, no decoder. Truth only when --bound 1.
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import factors as FA

OUT = FA.ROOT / "runs/173"
TAU, LAM_SLOW = 0.5, 1.0
DEV, T, BATCH, BUF, SWAP, SWAP_EVERY = FA.DEV, FA.T, FA.BATCH, FA.BUF, FA.SWAP, FA.SWAP_EVERY
arg = FA.arg


def fit():
    K, seed, out = int(arg("--K", 8)), int(arg("--seed", 0)), arg("--out", "fit")
    steps, bound = int(arg("--steps", 4000)), bool(int(arg("--bound", 0)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    ck = torch.load(FA.OUT / "g.pt", map_location=DEV, weights_only=False)
    assert ck.get("win", 5) == FA.WIN and ck.get("ring", [6, 20]) == [FA.RING_LO, FA.RING_HI]
    g = FA.PoseTerm().to(DEV); g.load_state_dict(ck["state_dict"]); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    meta = FA.load_meta("train")
    X = np.load(FA.OUT / "feat_train.npy", mmap_mode="r")
    NE = X.shape[0]
    pose_all = ((meta["pose"] - ck["mu"]) / ck["sd"]).astype(np.float32)
    ids = rng.choice(NE, BUF, replace=False)
    feat = torch.from_numpy(np.stack([np.asarray(X[i]) for i in ids])).to(DEV)
    pose = torch.from_numpy(pose_all[ids]).to(DEV)
    tr_keys = ("cube", "drawer", "window", "locks")
    cast = lambda k, a: np.asarray(a, np.float32 if k != "locks" else np.int64)
    truth = [torch.from_numpy(cast(k, meta[k][ids])).to(DEV) for k in tr_keys] if bound else None

    net = FA.Factors(K=K).to(DEV)
    net.log_tau.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    t0, acc_run, last = time.time(), 0.0, time.time()
    zero = torch.zeros(BATCH, dtype=torch.long, device=DEV)
    for st in range(1, steps + 1):
        if st % SWAP_EVERY == 0:
            slots, new = rng.choice(BUF, SWAP, replace=False), rng.choice(NE, SWAP, replace=False)
            sl = torch.from_numpy(slots).to(DEV)
            feat[sl] = torch.from_numpy(np.stack([np.asarray(X[i]) for i in new])).to(DEV)
            pose[sl] = torch.from_numpy(pose_all[new]).to(DEV)
            if bound:
                for k_i, k in enumerate(tr_keys):
                    truth[k_i][sl] = torch.from_numpy(cast(k, meta[k][new])).to(DEV)
        e, t, ce, ct = FA.sample_by_truth(pose, truth, BATCH, gen) if bound else FA.sample(pose, BATCH, gen)
        C = ce.shape[1]
        t1 = torch.where(t < T - 1, t + 1, t - 1)                                         # the consecutive step
        frames = torch.cat([feat[e, t], feat[e, t1], feat[ce.reshape(-1), ct.reshape(-1)]], 0).float()
        s = net(frames)
        sa, s1, sc = s[:BATCH], s[BATCH:2 * BATCH], s[2 * BATCH:].reshape(BATCH, C, K)
        logit = -(sa[:, None] - sc).abs().sum(-1) / TAU
        if not bound:
            logit = logit + g(pose[e, t][:, None].expand(-1, C, -1), pose[ce, ct])
        nce = F.cross_entropy(logit, zero)
        slow = (sa - s1).abs().sum(-1).mean()
        loss = nce + LAM_SLOW * slow
        for pg in opt.param_groups:
            pg["lr"] = 1e-3 * min(1.0, st / 200.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        acc_run = 0.98 * acc_run + 0.02 * float((logit.argmax(1) == 0).float().mean())
        if time.time() - last > 30 or st == steps:
            last = time.time()
            print(f"step {st} {time.time()-t0:.0f}s nce {nce.item():.3f} slow {slow.item():.3f} picks true {100*acc_run:.1f}% "
                  f"std {s.std(0).mean().item():.3f} used {(s.std(0) > 0.02).sum().item()}/{K}", flush=True)
    torch.save(dict(state_dict=net.state_dict(), K=K, seed=seed, steps=steps, sec=round(time.time() - t0, 1), bound=bound,
                    picks_true_pct=round(100 * acc_run, 1), tau=TAU, lam_slow=LAM_SLOW), OUT / f"{out}.pt")
    print(f"saved {out}.pt steps {steps} in {time.time()-t0:.0f}s picks true {100*acc_run:.1f}%", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fit()
