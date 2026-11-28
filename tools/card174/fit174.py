"""Card 174: card 173's label-free fit, unchanged in form, with the GPU's memory capped and reported.

  fit174.py --K 8 --seed 1 --steps 4000 --out k8_s1 [--amp 1]     -> runs/174/<out>.pt
--amp 1 runs the encoder in half precision (the loss in full), to keep a fit under half the card's memory.
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, resource
import factors as FA

OUT = FA.ROOT / "runs/174"
AMP = bool(int(FA.arg("--amp", 0)))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", 0.5)))
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
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
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
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            s = net(frames)
        s = s.float()
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
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        scaler.step(opt); scaler.update()
        acc_run = 0.98 * acc_run + 0.02 * float((logit.argmax(1) == 0).float().mean())
        if time.time() - last > 30 or st == steps:
            last = time.time()
            print(f"step {st} {time.time()-t0:.0f}s nce {nce.item():.3f} slow {slow.item():.3f} picks true {100*acc_run:.1f}% "
                  f"std {s.std(0).mean().item():.3f} used {(s.std(0) > 0.02).sum().item()}/{K}", flush=True)
    torch.save(dict(state_dict=net.state_dict(), K=K, seed=seed, steps=steps, sec=round(time.time() - t0, 1), bound=bound,
                    picks_true_pct=round(100 * acc_run, 1), tau=TAU, lam_slow=LAM_SLOW, amp=AMP), OUT / f"{out}.pt")
    print(f"saved {out}.pt steps {steps} in {time.time()-t0:.0f}s picks true {100*acc_run:.1f}% | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fit()
