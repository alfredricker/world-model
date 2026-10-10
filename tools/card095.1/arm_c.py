"""Card 095.1, arm C: arm B's vote as the evidence, arm A's prediction as its prior (the user, 2026-10-09: "similar
(B) -> A", two steps):

    P(token changes) = (B's kernel-weighted changed count + alpha * A's P) / (B's kernel-weighted count + alpha)

alpha per action and world, chosen on a grid by the leave-one-try-out log-likelihood of a sample of training
instances (arm_b.py's b_loo_<world>.npz). What a changed token becomes: arm B's relation where B has changed
neighbours, else arm A's result.

  bin/prun python tools/card095.1/arm_c.py      → runs/095.1/c_<tier>.npz, runs/095.1/c_alpha.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
import tok                                             # noqa: E402
import arm_a as AA                                     # noqa: E402

GRID = np.logspace(-2, 4, 25)
EPS = 1e-6


def combine(nc, n, pa, alpha):
    return np.clip((nc + alpha * pa) / (n + alpha), EPS, 1 - EPS)


@torch.no_grad()
def pa_of(net, z, H, P, rows, slots, bs=4096):
    arr = torch.as_tensor(z["arr"], device=AA.dev)
    out = np.zeros(len(rows))
    for s in range(0, len(rows), bs):
        rr = rows[s:s + bs]
        Ht = torch.as_tensor(H[rr], device=AA.dev)
        mask = Ht >= 0
        X = arr[Ht.clamp_min(0)] * mask[..., None]
        logit, _, _, _ = net(X, torch.as_tensor(P[rr], device=AA.dev), mask,
                             torch.as_tensor(z["act"][rr], device=AA.dev))
        out[s:s + bs] = torch.sigmoid(logit).cpu().numpy()[np.arange(len(rr)), slots[s:s + bs]]
    return out


if __name__ == "__main__":
    net = AA.StateNet().to(AA.dev)
    net.load_state_dict(torch.load(tok.RUNS / "arm_a.pt")["net"])
    net.eval()
    alphas = {}
    for world in ("tier1", "tier2", "tier3"):
        z = tok.load(world)
        H, P, C, A = tok.build(z)
        tr, te = tok.split(z)
        te_idx = np.flatnonzero(te)
        b = np.load(tok.RUNS / f"b_{world}.npz")
        lo = np.load(tok.RUNS / f"b_loo_{world}.npz")
        a = np.load(tok.RUNS / f"a_{world}.npz")
        pa_lo = pa_of(net, z, H, P, lo["r"], lo["c"])
        row = np.searchsorted(te_idx, b["r"])
        pa_te = a["p"][row, b["c"]]
        act_te = z["act"][b["r"]]
        p = np.zeros(len(b["r"]))
        alphas[world] = {}
        for x in (3, 4, 5):
            m = lo["act"] == x
            ll = [float((lo["w"][m] * np.where(lo["ch"][m], np.log(combine(lo["nc"][m], lo["n"][m], pa_lo[m], al)),
                                               np.log(1 - combine(lo["nc"][m], lo["n"][m], pa_lo[m], al)))).sum())
                  for al in GRID]
            al = float(GRID[int(np.argmax(ll))])
            alphas[world][str(x)] = al
            q = act_te == x
            p[q] = combine(b["nc"][q], b["n"][q], pa_te[q], al)
        after = np.where(b["after"] >= 0, b["after"], a["after"][row, b["c"]])
        np.savez_compressed(tok.RUNS / f"c_{world}.npz", r=b["r"], c=b["c"], p=p, after=after)
        print(world, "alpha per action", alphas[world], flush=True)
    (tok.RUNS / "c_alpha.json").write_text(json.dumps(alphas, indent=1) + "\n")
