"""Card 108: card 105.10's transition model, retrained with one more kind of query: "what is here k steps later?".

Unchanged from 105.10 (tools/card105.2/run.py, --loss code --origins pose): the tokens (the 81 tiles, 098.3's 32-d
vectors), the view placed by the agent's own pose, arm Rb, the next code's likelihood, the data (the curriculum's
worlds 0-7 and random play workers 0-1), 30,000 steps of 128 one-step transitions with the agent's action.

Added, in every step, 128 queries:
  k        1 with probability 0.3 (with the agent's action), else log-uniform over 2..50, given as a "k steps pass"
           token in the action token's place (action ids 17 + k - 2); the actions in between are not given
  inputs   the view at t, with places masked out (vector zero, absent): with probability 0.5 the part of the view
           outside a shifted copy of itself (a patch whose edge was never seen), each place with probability
           p ~ U(0, 0.3), and the agent's own tile with probability 0.3
  targets  each place in view at both t and t + k (no identities: place contents); carried tiles only for k = 1
Both parts are scored by 105.8's code likelihood, each averaged over its own targets, and summed.

  bin/prun python tools/card108/train.py count                    → runs/108/data_check.json (section 4)
  bin/prun python tools/card108/train.py train --seed S [--steps N] → runs/108/model_s<S>.json, _net.pt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
CMD, SEED, STEPS = sys.argv[1], int(arg("--seed", 1021)), int(arg("--steps", 30000))
sys.argv = ["run.py", "--loss", "code", "--origins", "pose"]
sys.path.insert(0, str(ROOT / "tools" / "card105.2"))
sys.path.insert(0, str(ROOT / "tools" / "card105"))
import run as R                                          # noqa: E402
from model import TransitionModel, code_loss             # noqa: E402

OUT = ROOT / "runs" / "108"
dev = R.dev
KMAX, NA = 50, 17 + 49                                   # actions 0..16; "k steps pass" for k = 2..50
AGENT = 31                                               # the agent's tile (the view is centred on the pose)


def reach(D):
    """For each stored step t, the largest k such that t -> t + 1 -> ... -> t + k are all awake transitions of one
    episode (105.2's rule for one step, chained)."""
    ok = np.zeros(len(D["act"]) + 1, bool)
    ok[D["t"]] = True
    r = np.zeros(len(ok), np.int64)
    for t in range(len(ok) - 2, -1, -1):
        r[t] = r[t + 1] + 1 if ok[t] else 0
    return r[:-1]


def kmap(D, t, k):
    """For each view token at t, its index in the view at t + k (same world place), else -1; carried tiles map to
    themselves for k = 1 and have no target otherwise."""
    sh = D["pos"][t + k, 0] - D["pos"][t, 0]             # the view's origin moves with the pose
    c1, r1 = R.COLS[None, :63] - sh[:, :1], R.ROWS[None, :63] - sh[:, 1:]
    km = np.full((len(t), R.NT), -1, np.int64)
    km[:, :63] = np.where((c1 >= 0) & (c1 < 9) & (r1 >= 0) & (r1 < 7), r1 * 9 + c1, -1).astype(np.int64)
    km[:, 63:] = np.where((k == 1)[:, None], np.arange(63, R.NT)[None], -1)
    return km


def sample_queries(D, order, sreach, rng, n):
    """n (t, k) pairs: k = 1 with probability 0.3, else log-uniform over 2..50; t uniform among steps that reach k."""
    k = np.where(rng.random(n) < 0.3, 1, np.rint(np.exp(rng.uniform(np.log(1.5), np.log(KMAX + 0.5), n)))).astype(int)
    k = k.clip(1, KMAX)
    lo = np.searchsorted(sreach, k)                      # order sorted by reach: the eligible steps are a suffix
    t = order[lo + (rng.random(n) * (len(order) - lo)).astype(int)]
    return t, k


def masks(rng, n):
    """Present (True) per view place for n queries: a shifted copy of the view, scattered places, the agent's tile."""
    M = np.ones((n, R.NT), bool)
    shift = rng.random(n) < 0.5
    ox, oy = rng.integers(-4, 5, n) * shift, rng.integers(-3, 4, n) * shift
    c, r = R.COLS[None, :63] - ox[:, None], R.ROWS[None, :63] - oy[:, None]
    M[:, :63] = (c >= 0) & (c < 9) & (r >= 0) & (r < 7)
    M[:, :63] &= rng.random((n, 63)) >= rng.uniform(0, 0.3, n)[:, None]
    M[rng.random(n) < 0.3, AGENT] = False
    return M


def query_batch(D, t, k, M):
    """Inputs (masked places zero and absent), targets at t + k, validity, positions, frames, action ids."""
    km = kmap(D, t, k)
    V0 = D["V"][t].to(dev).float()
    Mt = torch.as_tensor(M, device=dev)
    V0 = V0 * Mt[..., None]
    V1 = D["V"][t + k].to(dev).float()
    kmt = torch.as_tensor(km, device=dev)
    Vn = torch.gather(V1, 1, kmt.clamp(min=0)[..., None].expand(-1, -1, V1.shape[-1]))
    P = torch.as_tensor(D["pos"][t], device=dev)
    F = R.FRAME.to(dev).expand(len(t), R.NT)
    a = np.where(k == 1, D["act"][t], 17 + k - 2)
    return V0, Vn, kmt >= 0, P, F, Mt, torch.as_tensor(a, device=dev)


def count():
    """Section 4: k-step training pairs by k (places in view at both t and t + k), and pairs with a creature in view
    at t whose place is still in view at t + k (scoring labels, counted only here)."""
    tr = R.load(R.TRAIN, "tokens")
    rch = reach(tr)
    out = {"one-step training transitions": int(len(tr["t"]))}
    rows = {}
    for k in (1, 2, 5, 10, 20, 50):
        t = np.flatnonzero(rch >= k)
        km = kmap(tr, t, np.full(len(t), k))
        both = (km[:, :63] >= 0).sum(1)
        lab0 = tr["lab"][t, :63]
        cre = np.isin(lab0, (14, 15, 16)) & (km[:, :63] >= 0)
        rows[k] = {"pairs": int(len(t)), "mean places in view at both": round(float(both.mean()), 1),
                   "creature places at t still in view at t + k": int(cre.sum())}
    out["by k"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def train():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.monotonic()
    tr, te = R.load(R.TRAIN, "tokens"), R.load(R.TEST, "tokens")
    rch = reach(tr)
    order = np.argsort(rch, kind="stable")
    sreach = rch[order]
    print(f"loaded: {len(tr['t'])} training and {len(te['t'])} test transitions ({time.monotonic() - t0:.0f} s)",
          flush=True)
    net = TransitionModel(tr["V"].shape[-1], NA, arm="Rb").to(dev)
    log_tau = torch.nn.Parameter(torch.tensor(-2.3, device=dev))
    Pt, plab = R.decoder(tr, "tokens")
    opt = torch.optim.AdamW([*net.parameters(), log_tau], lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    BS = 128
    for step in range(STEPS):
        i = rng.choice(len(tr["t"]), BS, replace=False)            # 105.10's one-step batch, unchanged
        V0, Vn, valid, P, F, M, a = R.batch(tr, i)
        tq, kq = sample_queries(tr, order, sreach, rng, BS)
        V0q, Vnq, validq, Pq, Fq, Mq, aq = query_batch(tr, tq, kq, masks(rng, BS))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, pres = net(torch.cat([V0, V0q]), torch.cat([P, Pq]), torch.cat([F, Fq]), torch.cat([M, Mq]),
                            torch.cat([a, aq]))
        nxt, pres = nxt.float(), pres.float()
        one = torch.ones_like(M)
        l1 = code_loss(nxt[:BS], pres[:BS], Vn, one, valid, Pt, log_tau)
        lq = code_loss(nxt[BS:], pres[BS:], Vnq, one, validq, Pt, log_tau)
        loss = l1 + lq
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} loss one-step {float(l1):.5f} queries {float(lq):.5f} ({time.monotonic() - t0:.0f} s)",
                  flush=True)
    net.eval()
    tau = float(log_tau.exp())
    rep = {"seed": SEED, "steps": STEPS, "tau": round(tau, 5), "training transitions": int(len(tr["t"])),
           "origins": "the agent's own pose (105.10)", "queries": "k = 1 (p 0.3) or log-uniform 2..50, masked",
           **R.evaluate(net, te, Pt, plab, tau), "seconds": round(time.monotonic() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"model_s{SEED}.json").write_text(json.dumps(rep, indent=1) + "\n")
    torch.save({"net": net.state_dict(), "tau": tau}, OUT / f"model_s{SEED}_net.pt")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"count": count, "train": train}[CMD]()
