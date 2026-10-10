"""Card 096.2: a sleep that replays stored tries into arm A by event, with the prior's weight fitted where the prior is
used (events memory lacks). Offline.

Events: (action, token in front, token in the hand), per world (card 096.1). 20% of each world's events per action
are held out of every arm A trained here (all their tries). Three arm A's, 6,000 steps each (card 095.1's recipe):
  try:     tries sampled as card 095.1 does (weighted by try), worlds tier 1, tier 2, decoy     (version 24's recipe)
  event:   each event equally (a random event, then a random try of it), the same worlds
  event3:  as event, with tier 3's own tries added (the agent's world consolidated)
alpha per (tier, action, arm): version 24's (card 095.1's, fitted per try at full data), and "cross": fitted on half
of the held-out events by their zero-occurrence likelihood and scored on the other half (and the reverse).
Scored on the held-out events of tiers 2 and 3 at n = 0, 1, 3 (card 096.1's exact computation), and at full counts on
every event (the prior must not hurt where memory has the event).

  bin/prun python tools/card096.2/sleep.py      → runs/096.2/sleep.json, runs/096.2/arm_a_<arm>.pt
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for c in ("card095.1", "card095.2", "card095.3", "card096.1"):
    sys.path.insert(0, str(ROOT / "tools" / c))
import tok                                             # noqa: E402
import arm_a as AA                                     # noqa: E402
import fewshot as FS                                   # noqa: E402

OUT = ROOT / "runs" / "096.2"
dev = AA.dev
FRONT = 71
HELD_SHARE = 0.2
NS = (0, 1, 3)
GRID = np.logspace(-3, 4, 29)
EPS = 1e-6
log = lambda m: print(m, flush=True)


def event_keys(H, P, act):
    r = np.arange(len(H))
    fi = np.argmax(P == FRONT, 1)
    u = np.where(P[r, fi] == FRONT, H[r, fi], -1)
    return list(zip(act.tolist(), u.tolist(), H[:, -1].tolist()))


def held_out(world, rng):
    z = tok.load(world)
    H, P, C, A = tok.build(z)
    keys = sorted(set(event_keys(H, P, z["act"])))
    out = set()
    for a in (3, 4, 5):
        ks = [k for k in keys if k[0] == a]
        pick = rng.choice(len(ks), int(round(HELD_SHARE * len(ks))), replace=False) if ks else []
        out.update(ks[i] for i in pick)
    return out


class EData(AA.Data):
    """Card 095.1's training data without the held-out events; by event: sampled per event, every try weight 1."""

    def __init__(self, world, held, by_event):
        super().__init__(world, "train")
        H, P, act = self.H.cpu().numpy(), self.P.cpu().numpy(), self.act.cpu().numpy()
        keys = event_keys(H, P, act)
        keep = np.array([k not in held for k in keys])
        ki = torch.as_tensor(np.flatnonzero(keep), device=dev)
        for name in ("H", "P", "C", "A", "act", "w"):
            setattr(self, name, getattr(self, name)[ki])
        self.idx, self.n = self.idx[keep], int(keep.sum())
        ids = {}
        ev = np.array([ids.setdefault(k, len(ids)) for k, m in zip(keys, keep) if m], np.int64)
        order = np.argsort(ev, kind="stable")
        self.rows = np.split(order, np.cumsum(np.bincount(ev, minlength=len(ids)))[:-1])
        self.n_ev = len(ids)
        self.by_event = by_event
        if by_event:
            self.w = torch.ones_like(self.w)

    def sample(self, rng, bs):
        if not self.by_event:
            return rng.integers(self.n, size=bs)
        e = rng.integers(self.n_ev, size=bs)
        return np.array([self.rows[k][rng.integers(len(self.rows[k]))] for k in e.tolist()])


def train(name, worlds, held, by_event, steps=6000, seed=95, bs=512, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [EData(w, held[w], by_event) for w in worlds]
    size = np.array([d.n_ev if by_event else d.n for d in data], np.float64)
    net = AA.StateNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.monotonic()
    for step in range(steps):
        d = data[rng.choice(len(data), p=size / size.sum())]
        i = torch.as_tensor(d.sample(rng, bs), device=dev)
        loss, lc, la = AA.loss_of(net, d, i)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save({"net": net.state_dict(), "steps": steps}, OUT / f"arm_a_{name}.pt")
    log(f"arm A {name}: {round(time.monotonic() - t0, 1)} s, worlds {worlds}, events {[d.n_ev for d in data]}")
    return net.cpu().eval()


def base(tier):
    """Per event of the tier: the vote's parts without the event, its outcome, and B's results at each n and at full."""
    path = os.environ.get("WM_STORE", str(ROOT / "runs" / "095.3" / "store_tier{tier}.pkl")).format(tier=tier)
    data = pickle.loads(Path(path).read_bytes())
    ev, known, arr = FS.events(tier)
    arr = arr.astype(np.float32)
    V = np.concatenate([arr, np.zeros((1, arr.shape[1]), np.float32)])
    for st in data["stores"].values():
        st.arm.V = V
    out = []
    for (a, u, h), (W, cnt, wf, wh, outs) in ev.items():
        st = data["stores"][a]
        q = np.array([wf / W, wh / W])
        mo = max(outs, key=outs.get) if outs else (-1, -1)
        f2 = FS.fields(u, h)
        parts, resB = [], {}
        wbar = W / cnt
        for i in range(2):
            Nc, N, gi, wgt = st.vote(f2[i])
            ex = {int(g) for g, wg in zip(gi.tolist(), wgt.tolist()) if wg >= 1 - 1e-9}
            Gn = sum(float(st.arm.Gn[g]) for g in ex)
            Gnc = sum(float(st.arm.Gnc[g]) for g in ex)
            parts.append((Nc - Gnc, N - Gn, Gnc, Nc, N))
            for n in NS + ("full",):
                add_c = (n * wbar * q[i]) if n != "full" else Gnc
                scale = (add_c / Gnc) if Gnc > 0 else 0.0
                x = FS.result_scaled(st, f2[i], gi, wgt, ex, scale, known, arr)
                resB.setdefault(n, []).append(int(x))
        out.append({"key": (a, u, h), "q": q, "mo": mo, "parts": parts, "wbar": wbar, "W": W, "resB": resB})
    return out, known, arr, data["alpha"]


def judge(e, pa, snap, alpha, n):
    q, mo = e["q"], e["mo"]
    P, after = np.zeros(2), [-1, -1]
    for i in range(2):
        Nc0, N0, Gnc, Nc, N = e["parts"][i]
        if n == "full":                                # the vote as stored
            Nc0, N0, add_n, add_c = Nc, N, 0.0, 0.0
        else:
            add_n, add_c = n * e["wbar"], n * e["wbar"] * q[i]
        if alpha is None:                              # B alone
            P[i] = (Nc0 + add_c) / (N0 + add_n) if N0 + add_n > 0 else 0.5
        elif alpha == "A":
            P[i] = pa[i]
        else:
            P[i] = (Nc0 + add_c + alpha * pa[i]) / (N0 + add_n + alpha)
        if P[i] > 0.5:
            x = e["resB"][n][i] if alpha != "A" else -1
            after[i] = x if x >= 0 else int(snap[i])
    maj = q > 0.5
    side = ((P > 0.5) == maj).all() and not (np.abs(P - 0.5) < 1e-12).any()
    ok = all(not maj[i] or after[i] == mo[i] for i in range(2))
    ll = float(np.sum(q * np.log(np.clip(P, EPS, 1)) + (1 - q) * np.log(np.clip(1 - P, EPS, 1))))
    return bool(side and ok), ll


def fit_alpha(evs, pas):
    best = None
    for g in GRID:
        ll = sum(judge(e, pa, sn, g, 0)[1] for e, (pa, sn) in zip(evs, pas))
        if best is None or ll > best[1]:
            best = (float(g), ll)
    return best[0]


def sign_p(won, lost):
    from math import comb
    n, k = won + lost, min(won, lost)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(96)
    worlds = ("tier1", "tier2", "decoy", "tier3")
    held = {w: held_out(w, rng) for w in worlds}
    log(f"held-out events: { {w: len(held[w]) for w in worlds} }")
    nets = {"try": train("try", worlds[:3], held, False), "event": train("event", worlds[:3], held, True),
            "event3": train("event3", worlds, held, True)}
    rep = {}
    for tier in (2, 3):
        evs, known, arr, alpha0 = base(tier)
        hk = held[f"tier{tier}"]
        rep[f"tier{tier}"] = {}
        for a, an in FS.ACTS.items():
            E = [e for e in evs if e["key"][0] == a]
            Eh = [e for e in E if e["key"] in hk]
            fold = np.random.default_rng(a).permutation(len(Eh)) % 2
            r = {"events": len(E), "held": len(Eh)}
            right = {}
            for name, net in nets.items():
                pas = [FS.arm_a(net, arr, known, a, *e["key"][1:]) for e in Eh]
                pall = [FS.arm_a(net, arr, known, a, *e["key"][1:]) for e in E]
                al = [fit_alpha([e for e, f in zip(Eh, fold) if f == k], [p for p, f in zip(pas, fold) if f == k])
                      if (fold == k).any() else alpha0[str(a)] for k in (0, 1)]
                al_e = [al[1 - f] for f in fold]                # each event scored with the other half's alpha
                r[f"alpha_cross_{name}"] = al
                for n in NS:
                    right[(name, "C0", n)] = [judge(e, p, s, alpha0[str(a)], n)[0] for e, (p, s) in zip(Eh, pas)]
                    right[(name, "Cx", n)] = [judge(e, p, s, ax, n)[0] for e, (p, s), ax in zip(Eh, pas, al_e)]
                    right[(name, "A", n)] = [judge(e, p, s, "A", n)[0] for e, (p, s) in zip(Eh, pas)]
                alf = float(np.exp(np.mean(np.log(al))))
                right[(name, "Cx", "full")] = [judge(e, p, s, alf, "full")[0] for e, (p, s) in zip(E, pall)]
                right[(name, "C0", "full")] = [judge(e, p, s, alpha0[str(a)], "full")[0] for e, (p, s) in zip(E, pall)]
            for n in NS:
                right[("-", "B", n)] = [judge(e, np.zeros(2), [-1, -1], None, n)[0] for e in Eh]
            r["right"] = {f"{k[0]} {k[1]} n={k[2]}": round(float(np.mean(v)), 4) if v else None for k, v in right.items()}
            base_v = right[("try", "C0", 0)]
            new_v = right[("event3", "Cx", 0)]
            won = sum(b and not o for b, o in zip(new_v, base_v))
            lost = sum(o and not b for b, o in zip(new_v, base_v))
            r["event3_cross_vs_v24_at_n0"] = {"won": won, "lost": lost, "p": round(sign_p(won, lost), 4)}
            rep[f"tier{tier}"][an] = r
            log(f"tier {tier} {an}: {json.dumps(r)}")
    (OUT / "sleep.json").write_text(json.dumps(rep, indent=1) + "\n")
