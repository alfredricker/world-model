"""Card 102.6: actions with a where. Arm A (card 102.1's pair function) reads an action as its kind (wait, click) and,
when it has one, its where: every token's offset from the click enters the pair function (102.1's locus input), so
what a click changes is a learned relation between the action's where and the tokens. Trained by card 102.1's loss (a
change head on every place, content where the code changes), draws prioritised by the last loss (alpha 0.6).

Scored on breaking world 5's test play (held-out shapes; column 6, never clicked in training): the whole next state
right (every place's next code), by target kind, column and shape edge or inside. Baselines: "nothing changes", and
the same network with the click's where withheld (--nowhere).

  bin/prun python tools/card102.6/clicks.py [--nowhere] [--steps N]      → runs/102.6/clicks[_nowhere].json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card102.1"))
import arms as AR                                      # noqa: E402

RUNS, dev = ROOT / "runs" / "102.6", "cuda"
R, C, L = 7, 9, 63
WHERE = AR.WHERE[:L]
NOWHERE = "--nowhere" in sys.argv


def load(split):
    z = np.load(ROOT / "runs" / "card102" / f"world5_{split}.npz")
    n = len(z["act"])
    d = {"V": np.load(RUNS / f"world5_vec_{split}.npy")[:, :L].astype(np.float32),
         "code": np.load(RUNS / f"world5_code_{split}.npy")[:, :L].astype(np.int64),
         "act": z["act"].astype(np.int64), "ep": z["ep"], "done": z["done"], "event": z["event"],
         "sem": z["sem"].reshape(n, L), "obj": z["obj"].reshape(n, L)}
    d["steps"] = np.flatnonzero((d["ep"][:-1] == d["ep"][1:]) & ~d["done"][:-1])
    return d


def batch(d, i):
    X = torch.as_tensor(d["V"][i], device=dev)
    a = d["act"][i]
    kind = torch.as_tensor((a > 0).astype(np.int64), device=dev)                # 0 wait, 1 click
    cell = np.maximum(a - 1, 0)
    loc = WHERE[torch.as_tensor(cell)].to(dev)                                   # the click's where (raw position)
    if NOWHERE:
        loc = torch.zeros_like(loc)
    loc = torch.where(kind[:, None] > 0, loc, torch.zeros_like(loc))
    W = WHERE.to(dev)[None].expand(len(i), -1, -1)
    return X, kind, W, loc


def train(d, steps, seed=1026, bs=128, alpha=0.6):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = AR.PairNet(32, na=2, rel=False, gates=False, locus=True).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    S, pri, t0 = d["steps"], np.ones(len(d["steps"])), time.monotonic()
    for step in range(steps):
        P = pri ** alpha
        P /= P.sum()
        k = np.sort(rng.choice(len(S), bs, p=P))
        i = S[k]
        X, kind, W, loc = batch(d, i)
        logit, after, _, _, _ = net(X, kind, None, W, loc)
        own = torch.as_tensor(d["code"][i] != d["code"][i + 1], device=dev)
        lc = fn.binary_cross_entropy_with_logits(logit, own.float(), reduction="none").sum(1)
        la = ((after - torch.as_tensor(d["V"][i + 1], device=dev)).abs().sum(-1) * own).sum(1)
        li = lc + 0.1 * la
        pri[k] = li.detach().cpu().numpy() + 1e-3
        opt.zero_grad()
        li.mean().backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print({"step": step + 1, "change": round(lc.mean().item(), 4), "after": round(la.mean().item(), 4),
                   "s": round(time.monotonic() - t0, 1)}, flush=True)
    return net


@torch.no_grad()
def predict(net, d, protos, bs=1024):
    S, out = d["steps"], []
    for s in range(0, len(S), bs):
        i = S[s:s + bs]
        X, kind, W, loc = batch(d, i)
        logit, after, _, _, _ = net(X, kind, None, W, loc)
        P = torch.where((logit > 0)[..., None], after, X)
        k = torch.cdist(P.reshape(-1, 32), protos).argmin(-1).reshape(len(i), L)
        out.append(k.cpu().numpy())
    return np.concatenate(out)


def main():
    steps = int(AR.arg("--steps", "12000"))
    tr, te = load("train"), load("test")
    protos = torch.as_tensor(np.load(RUNS / "world5_codes.npz")["protos"], device=dev).float()
    t0 = time.monotonic()
    net = train(tr, steps)
    net.eval()
    S = te["steps"]
    pred = predict(net, te, protos)
    nxt = te["code"][S + 1]
    whole = (pred == nxt).all(1)
    keep = (te["code"][S] == nxt).all(1)
    a = te["act"][S]
    cell = np.maximum(a - 1, 0)
    sem_at = te["sem"][S, cell]
    kind = np.where(a == 0, "wait", np.where(sem_at >= 8, "shape", np.where(np.isin(sem_at, (4, 5)), "lamp",
                                                                              np.where(sem_at == 1, "wall", "floor"))))
    obj = te["obj"][S]
    edge = np.zeros(len(S), bool)                          # a click on a shape tile with a 4-neighbour outside it
    for j in np.flatnonzero(kind == "shape"):
        y, x = divmod(int(cell[j]), C)
        o = obj[j, cell[j]]
        edge[j] = any(not (0 <= y + dy < R and 0 <= x + dx < C) or obj[j, (y + dy) * C + x + dx] != o
                      for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    held = (a > 0) & (cell % C == 6)
    rep = {"training steps": steps, "where withheld": NOWHERE, "test steps": int(len(S)),
           "whole next state right (all steps)": round(float(whole.mean()), 4),
           "nothing changes (baseline)": round(float(keep.mean()), 4),
           "clicks that change something": int(((kind == "shape") | (kind == "lamp")).sum())}
    for k in ("shape", "lamp", "floor", "wall", "wait"):
        m = kind == k
        rep[f"whole next state right, {k}"] = [round(float(whole[m].mean()), 4), int(m.sum())]
    ch = (kind == "shape") | (kind == "lamp")
    rep["changing clicks, column 6 (never clicked in training)"] = [round(float(whole[ch & held].mean()), 4), int((ch & held).sum())]
    rep["changing clicks, other columns"] = [round(float(whole[ch & ~held].mean()), 4), int((ch & ~held).sum())]
    rep["shape clicks on an edge tile, inside tile"] = [round(float(whole[(kind == "shape") & edge].mean()), 4),
                                                        round(float(whole[(kind == "shape") & ~edge].mean()), 4)
                                                        if ((kind == "shape") & ~edge).any() else None]
    rep["seconds"] = round(time.monotonic() - t0, 1)
    RUNS.mkdir(exist_ok=True)
    tag = "_nowhere" if NOWHERE else ""
    (RUNS / f"clicks{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
    torch.save(net.state_dict(), RUNS / f"net{tag}.pt")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
