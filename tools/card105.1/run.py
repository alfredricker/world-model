"""Card 105.1: one transition model in breaking world 1. Every token's next vector from the 81 tiles of the fixed view
and the action (noop or a move); no part for moves, no outcome categories.

Forms: tokens (card 102.1's world 1 encoder vectors) or truth (the evaluator's kinds as one-hot vectors: the gate's
upper bound). Arms: R (relative positions only) or A (raw positions as inputs). Scored on the test worlds (lamps in
columns 5-7, never seen in training) by decoding each predicted vector to its nearest code (tokens) or kind (truth),
each code read as its most common evaluator kind in training (scoring only).

  bin/prun python tools/card105.1/run.py tokens|truth R|A [--seed 1021] [--steps 15000]  → runs/105.1/<form>_<arm>_s<seed>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card105"))
from model import TransitionModel, weighted_loss          # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
FORM, ARM = sys.argv[1], sys.argv[2]
SEED, STEPS, BS = int(arg("--seed", 1021)), int(arg("--steps", 15000)), int(arg("--bs", 128))
OUT = ROOT / "runs" / ("105.6" if ARM == "Rb" else "105.1")
dev = "cuda"
NT = 81
P_TILE = torch.tensor([(k % 9, k // 9) for k in range(NT)], dtype=torch.float32)
F_TILE = torch.tensor([0 if k < 63 else 1 for k in range(NT)])
MOVE = {1: -1, 2: 1, 3: -9, 4: 9}                      # evaluator only: a move's target cell


def load(split):
    z = np.load(ROOT / "runs" / "card102" / f"world1_{split}.npz")
    act, ep, done = z["act"].astype(np.int64), z["ep"], z["done"]
    n = len(act)
    t = np.flatnonzero(np.r_[ep[1:] == ep[:-1], False] & ~done)
    lab = np.full((n, NT), 7, np.int64)
    lab[:, :63] = z["sem"].reshape(n, 63)
    if FORM == "truth":
        V = np.eye(8, dtype=np.float16)[lab]
        code = lab
    else:
        V = np.load(ROOT / "runs" / "102.1" / f"world1_vec_{split}.npy")
        code = np.load(ROOT / "runs" / "102.1" / f"world1_code_{split}.npy").astype(np.int64)
    return {"V": torch.as_tensor(V), "t": t, "act": act, "lab": lab, "code": code, "push": z["push"]}


def protos():
    if FORM == "truth":
        return torch.eye(8)
    return torch.as_tensor(np.load(ROOT / "runs" / "102.1" / "world1_codes.npz")["protos"]).float()


def batch(d, idx):
    V0 = d["V"][idx].to(dev).float()
    V1 = d["V"][idx + 1].to(dev).float()
    B = len(idx)
    P = P_TILE.to(dev).expand(B, NT, 2)
    F = F_TILE.to(dev).expand(B, NT)
    M = torch.ones(B, NT, dtype=torch.bool, device=dev)
    a = torch.as_tensor(d["act"][idx], device=dev)
    return V0, V1, P, F, M, a


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.monotonic()
    tr, te = load("train"), load("test")
    net = TransitionModel(tr["V"].shape[-1], 5, arm=ARM).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    for step in range(STEPS):
        idx = rng.choice(tr["t"], BS, replace=False)
        V0, V1, P, F, M, a = batch(tr, idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, pres = net(V0, P, F, M, a)
        loss = weighted_loss(nxt.float(), pres.float(), V1, M, V0, M)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} loss {float(loss):.5f} ({time.monotonic() - t0:.0f} s)", flush=True)
    net.eval()
    # each code's evaluator kind (scoring only), from training pairs
    c, l = tr["code"].reshape(-1), tr["lab"].reshape(-1)
    kind_of = {int(k): int(np.bincount(l[c == k]).argmax()) for k in np.unique(c)}
    Pr = protos().to(dev)
    pred = np.zeros((len(te["t"]), NT), np.int64)
    with torch.no_grad():
        for s in range(0, len(te["t"]), 1024):
            idx = te["t"][s:s + 1024]
            V0, V1, P, F, M, a = batch(te, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nxt, _ = net(V0, P, F, M, a)
            code = torch.cdist(nxt.float().reshape(-1, Pr.shape[1]), Pr).argmin(1).reshape(len(idx), NT).cpu().numpy()
            pred[s:s + len(idx)] = np.vectorize(lambda k: kind_of.get(int(k), -1))(code)
    t = te["t"]
    L0, L1, act, push = te["lab"][t], te["lab"][t + 1], te["act"][t], te["push"][t]
    right = pred == L1
    lamp = np.isin(L0, (4, 5))
    ch = L0 != L1
    agent = np.argmax(L0[:, :63] == 2, 1)
    mv = np.isin(act, list(MOVE))
    target = agent + np.array([MOVE.get(int(x), 0) for x in act])
    blocked = mv & (L1[np.arange(len(t)), agent] == 2)
    both = right[np.arange(len(t)), agent] & right[np.arange(len(t)), np.clip(target, 0, 62)]
    col = np.arange(NT) % 9
    rep = {"form": FORM, "arm": ARM, "seed": SEED, "steps": STEPS, "test steps": int(len(t)),
           "1 lamps at pushes, changed: right": round(float(right[lamp & ch & push[:, None]].mean()), 4),
           "1 lamps at pushes, unchanged: right": round(float(right[lamp & ~ch & push[:, None]].mean()), 4),
           "1 lamps at pushes, changed, by column": {int(k): round(float(right[lamp & ch & push[:, None] & (col == k)].mean()), 4)
                                                     for k in (5, 6, 7) if (lamp & ch & push[:, None] & (col == k)).any()},
           "lamps at pushes (changed, unchanged)": [int((lamp & ch & push[:, None]).sum()), int((lamp & ~ch & push[:, None]).sum())],
           "2 move steps: agent's place and target right, moved": round(float(both[mv & ~blocked].mean()), 4),
           "2 move steps: agent's place and target right, blocked": round(float(both[blocked].mean()), 4),
           "move steps (moved, blocked)": [int((mv & ~blocked).sum()), int(blocked.sum())],
           "3 whole next state right (63 world tiles)": round(float(right[:, :63].all(1).mean()), 4),
           "nothing changes: whole next state right": round(float((L0[:, :63] == L1[:, :63]).all(1).mean()), 4),
           "tokens that changed: right": round(float(right[ch].mean()), 4),
           "seconds": round(time.monotonic() - t0, 1)}
    (OUT / f"{FORM}_{ARM}_s{SEED}.json").write_text(json.dumps(rep, indent=1) + "\n")
    torch.save(net.state_dict(), OUT / f"{FORM}_{ARM}_s{SEED}_net.pt")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
