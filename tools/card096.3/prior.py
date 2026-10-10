"""Card 096.3: arm A trained on the state the agent asks it about (the token in front and the hand), not on whole
states. Offline.

Each stored interaction try becomes (action, front token, hand token) -> (front changed, hand changed, what each
became), weighted by its try weight: card 095.1's StateNet and loss on two tokens. Worlds as version 24's arm A
(tiers 1, 2, decoy; never tier 3); 20% of each world's events held out (card 096.2's draw); three seeds.
Scored on the held-out events of tiers 2 and 3, on events that change something and events that change nothing
separately: arm A alone and arm C (version 24's alpha) at n = 0, 1, 3 (card 096.1's exact computation); how often
the three seeds' copies agree; and at full counts on every event. Baseline: card 096.2's "try" copy (whole states,
the same held-out events) and version 24's own copy (which saw every tier 2 event).

  bin/prun python tools/card096.3/prior.py               → runs/096.3/prior.json, runs/096.3/arm_a2_s<seed>.pt
  bin/prun python tools/card096.3/prior.py --deploy      → runs/096.3/arm_a2_deploy.pt (nothing held out, seed 95)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card096.2"))
import sleep as SL                                     # noqa: E402

AA, FS, tok = SL.AA, SL.FS, SL.tok
OUT = ROOT / "runs" / "096.3"
dev = AA.dev
FRONT, HAND = 71, 169
log = lambda m: print(m, flush=True)


class Pairs:
    """Every interaction try of a world as (front, hand), without the held-out events."""

    def __init__(self, world, held):
        z = tok.load(world)
        H, P, C, A = tok.build(z)
        r = np.arange(len(H))
        fi = np.argmax(P == FRONT, 1)
        ok = (P[r, fi] == FRONT) & (H[:, -1] >= 0)
        u, h, act = H[r, fi], H[:, -1], z["act"]
        keep = ok & np.array([(int(a), int(x), int(y)) not in held for a, x, y in zip(act, u, h)])
        t = lambda x, dt=torch.long: torch.as_tensor(x[keep], dtype=dt, device=dev)
        self.X = t(np.stack([u, h], 1))
        self.C = t(np.stack([C[r, fi], C[:, -1]], 1), torch.bool)
        self.A = t(np.stack([A[r, fi], A[:, -1]], 1))
        self.act, self.w = t(act), t(z["w"], torch.float32)
        self.arr = torch.as_tensor(z["arr"], device=dev)
        self.n = int(keep.sum())

    def batch(self, i):
        X = self.arr[self.X[i]]
        P = torch.tensor([[FRONT, HAND]], device=dev).expand(len(i), 2)
        mask = torch.ones(len(i), 2, dtype=torch.bool, device=dev)
        return X, P, mask, self.act[i], self.C[i], self.A[i], self.w[i]


def train(name, worlds, held, seed, steps=6000, bs=512, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [Pairs(w, held[w]) for w in worlds]
    size = np.array([d.n for d in data], np.float64)
    net = AA.StateNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    for step in range(steps):
        d = data[rng.choice(len(data), p=size / size.sum())]
        i = torch.as_tensor(rng.integers(d.n, size=bs), device=dev)
        loss, lc, la = AA.loss_of(net, d, i)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save({"net": net.state_dict(), "steps": steps, "form": "front and hand"}, OUT / f"arm_a2_{name}.pt")
    log(f"arm A on two tokens, {name}: {round(time.monotonic() - t0, 1)} s, tries {[d.n for d in data]}")
    return net.cpu().eval()


def load(path):
    net = AA.StateNet()
    net.load_state_dict(torch.load(path, map_location="cpu")["net"])
    return net.eval()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    worlds = ("tier1", "tier2", "decoy")
    if "--deploy" in sys.argv:
        train("deploy", worlds, {w: set() for w in worlds}, 95)
        sys.exit()
    rng = np.random.default_rng(96)
    held = {w: SL.held_out(w, rng) for w in ("tier1", "tier2", "decoy", "tier3")}   # card 096.2's draw
    nets = {f"s{s}": train(f"s{s}", worlds, held, s) for s in (95, 96, 97)}
    nets["whole states (096.2 try)"] = load(ROOT / "runs" / "096.2" / "arm_a_try.pt")
    nets["version 24"] = load(ROOT / "runs" / "095.1" / "arm_a.pt")
    rep = {}
    for tier in (2, 3):
        evs, known, arr, alpha0 = SL.base(tier)
        for a, an in FS.ACTS.items():
            E = [e for e in evs if e["key"][0] == a]
            Eh = [e for e in E if e["key"] in held[f"tier{tier}"]]
            r = {}
            pas = {nm: [FS.arm_a(net, arr, known, a, *e["key"][1:]) for e in Eh] for nm, net in nets.items()}
            for grp in ("change", "no change"):
                idx = [i for i, e in enumerate(Eh) if (e["q"] > 0.5).any() == (grp == "change")]
                g = {"events": len(idx)}
                if idx:
                    for nm in nets:
                        g[f"{nm}: A"] = round(float(np.mean([SL.judge(Eh[i], *pas[nm][i], "A", 0)[0] for i in idx])), 3)
                        for n in SL.NS:
                            g[f"{nm}: C n={n}"] = round(float(np.mean(
                                [SL.judge(Eh[i], *pas[nm][i], alpha0[str(a)], n)[0] for i in idx])), 3)
                    for n in SL.NS:
                        g[f"B n={n}"] = round(float(np.mean([SL.judge(Eh[i], np.zeros(2), [-1, -1], None, n)[0]
                                                             for i in idx])), 3)
                    sides = np.array([[(pas[f"s{s}"][i][0] > 0.5).tolist() for i in idx] for s in (95, 96, 97)])
                    g["seeds agree (A's side, both tokens)"] = round(float((sides == sides[0]).all((0, 2)).mean()), 3)
                r[grp] = g
            full = {}
            for nm in ("s95", "version 24"):
                pall = [FS.arm_a(nets[nm], arr, known, a, *e["key"][1:]) for e in E]
                full[nm] = round(float(np.mean([SL.judge(e, p, s, alpha0[str(a)], "full")[0]
                                                for e, (p, s) in zip(E, pall)])), 4)
            r["full counts, every event (C)"] = full
            rep[f"tier{tier} {an}"] = r
            log(f"tier {tier} {an}: {json.dumps(r)}")
    (OUT / "prior.json").write_text(json.dumps(rep, indent=1) + "\n")
