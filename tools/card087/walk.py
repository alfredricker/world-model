"""Card 087 in the agent: a tile's forward outcome (walkable, blocked, ends) from its own tries first, with the
property ensemble as the prior in place of recall's neighbours (card 038):

    P(h) = (N_own(h) + beta f(z_h)) / (|N_own(h)| + beta)

f is the ensemble's mean (runs/087/ensemble.pt, tools/card087/props.py); beta is fitted by leave-one-try-out
likelihood over memory's stored forward tries, as card 050's alpha. Everything that reads walkability goes through
World.move_cat (M.free, M.move_out, openable), so only that changes. Pick up and toggle are not read by version 18;
they are card 086's roles when it runs.

  WM_PROPS=1 ...        tools/card069/run.py hooks this after setup
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card087"))
import props as PR                                     # noqa: E402

STATS = {"props_prior_tiles": 0}                 # counters (the runner diffs them per episode)
INFO = {}
STATE = {}
GRID = np.logspace(-3, 3, 25)


def nets():
    """The ensemble's weights as numpy arrays: the episode workers are forked and must not touch the GPU."""
    out = []
    for sd in torch.load(PR.OUT / "ensemble.pt", map_location="cpu"):
        out.append([(sd[f"f.{i}.weight"].double().numpy(), sd[f"f.{i}.bias"].double().numpy()) for i in (0, 2, 4)])
    return out


def forward_probs(z):
    """The ensemble's mean P(forward outcome) for one vector z (numpy, the same network as props.Net)."""
    ps = []
    for layers in STATE["nets"]:
        x = np.asarray(z, np.float64)
        for j, (Wt, b) in enumerate(layers):
            x = x @ Wt.T + b
            if j < 2:
                x = np.maximum(x, 0.0)
        e = np.exp(x[:3] - x[:3].max())
        ps.append(e / e.sum())
    return np.mean(ps, 0)


def prior(W, h):
    """f(z_h) over the forward categories (CHANGED 0 = moved, UNCHANGED 1 = blocked, ENDED 2 = ended)."""
    c = STATE["prior"]
    r = c.get(h)
    if r is None:
        r = c[h] = forward_probs(W.S.arr[h])
        STATS["props_prior_tiles"] += 1
    return r


def fit_beta(W, VP):
    """beta by leave-one-try-out likelihood of the stored forward tries (each try predicted from its tile's other
    tries and the prior)."""
    kd = W.kinds[VP.FWD]
    N = np.asarray(kd.counts, np.float64)[:, :3]
    F = np.stack([prior(W, int(k[0])) for k in kd.keys])
    n = N.sum(1, keepdims=True)
    best = None
    for b in GRID:
        P = (N - 1 + b * F) / (n - 1 + b)
        m = N > 0
        ll = float((N[m] * np.log(np.clip(P[m], 1e-12, 1))).sum() / N.sum())
        if best is None or ll > best[1]:
            best = (float(b), ll)
    return best


def install(W, T):
    VP = T.VP
    STATE.update(nets=nets(), prior={})
    beta, ll = fit_beta(W, VP)
    STATE["beta"] = beta
    cls = type(W)
    if not getattr(cls, "_card087", False):
        base = cls.move_cat

        def move_cat(self, a, h):
            if a != VP.FWD or "nets" not in STATE:
                return base(self, a, h)
            h = int(h)
            kd = self.kinds[VP.FWD]
            i = kd.index.get((h,))
            own = np.zeros(3) if i is None else np.asarray(kd.counts[i], np.float64)[:3]
            P = (own + STATE["beta"] * prior(self, h)) / (own.sum() + STATE["beta"])
            return int(P.argmax())

        cls.move_cat = move_cat
        cls._card087 = True
    W.clear_lazy()
    INFO.update(beta=round(beta, 4), loo_per_try=round(ll, 5), stored_forward_keys=len(W.kinds[VP.FWD].keys))
    return dict(INFO)


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        info["props"] = install(W, T)
        log(f"card 087 properties: {info['props']}")
        return W, info

    T.setup = setup
