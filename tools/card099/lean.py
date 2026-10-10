"""Card 099: version 24 without the parts it no longer reads (WM_LEAN=1; run without WM_ROUTER, WM_ROUTER_ID,
WM_SAMPLED*, WM_STORED_FITS and WM_SPLIT). WM_LEAN_GATE=1: fit arm C's own beta, write runs/099/beta_tier<N>.json
and stop before the episodes (the card's gate).

Pick up, drop and toggle are predicted by arm C alone (card 095.3), so their kinds keep only what arm C, the planner
and card 072 call: the category from arm C's predict (cat_of, cached until the kind's next try, as card 042's), the
result (arm C's, set per kind by 095.3), and the weight of this episode's own tries (beta), which arm C now fits on
its own stored tries instead of borrowing card 050's. Card 039's weights, card 042's two levels, card 049's
admission, card 050's alpha, card 051's groups and card 047's situations are not built: any other attribute read
raises, so a run that completes shows that nothing removed was read. Card 074.2's pair-distance cache is taken off
card 039's kinds, which no longer exist.

Beta (card 050's likelihood on arm C's store): each stored try of action a with front u and hand h is predicted from
the other stored tries with the same (a, u, h), with arm C's categories as the prior,
  P = (N_own - 1 + beta P_C) / (N_own_total - 1 + beta),
summed over tries (their weights), on card 050's grid; the first maximum is taken, as card 050 did.

  WM_LEAN=1 WM_LEAN_GATE=1 <version 25's flags> bin/prun python tools/card069/run.py tiers --tier 3 --n 1 ...
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "099"
GATE = os.environ.get("WM_LEAN_GATE") == "1"
STATS = {}                                              # nothing per episode; setup's numbers go in info["lean"]
SK = sys.modules["slot_planner"].SlotKind
PD0 = SK.pair_distance                                  # before card 074.2's install wraps it
BORROWED = {1: {3: 0.000123, 4: 0.000123, 5: 0.000123},  # version 24's (card 050's alpha); 3 pick up, 4 drop, 5 toggle
            2: {3: 0.000123, 4: 0.000123, 5: 0.000123},
            3: {3: 0.000123, 4: 0.000123, 5: 12.182494}}


class Removed(AttributeError):
    pass


class LeanKind:
    """Pick up, toggle or drop: arm C answers; nothing of version 20's recall is built."""

    def __init__(self, name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
        self.name, self.S, self.parts, self.nkey, self.ncat = name, S, parts, nkey, ncat
        self.D = S.arr.shape[1]
        self.pair, self.moves = True, False
        self.beta = None                               # set by arm C's fit (hook), before any episode
        self.ccache = {}
        self.report = {"lean": "arm C answers", "stored_tries": int(len(w))}

    def __getattr__(self, k):                           # only called for attributes not set above
        raise Removed(f"card 099: {self.name}.{k} belongs to a removed part")

    def predict(self, qs):                              # replaced by arm C's (card 095.3) at setup
        raise Removed(f"card 099: {self.name}.predict before arm C")

    def result(self, q, c):                             # replaced per kind by arm C's (card 095.3) at setup
        raise Removed(f"card 099: {self.name}.result before arm C")

    def templates(self, u):
        raise Removed(f"card 099: {self.name}.templates (version 20's situations)")

    def opened(self, free):
        raise Removed(f"card 099: {self.name}.opened (version 20's situations)")

    def cat_of(self, qs):                               # card 042's
        todo = [q for q in dict.fromkeys(qs) if q not in self.ccache]
        if todo:
            for q, p in zip(todo, self.predict(todo)):
                self.ccache[q] = int(p.argmax()) if p.max() >= 0.5 else None
        return [self.ccache[q] for q in qs]

    def forget(self):
        self.ccache = {}

    def reset(self):
        self.forget()

    def add(self, key, cat, after=None, w=1.0):        # the try itself is arm C's (095.3's own tries)
        self.forget()


def install():
    T = sys.modules["tiers"]
    VP = T.VP
    kind0 = VP.Kind

    def kind(name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
        if nkey == 3:
            return LeanKind(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)
        return kind0(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)

    VP.Kind = kind
    SK.pair_distance = PD0
    R = sys.modules.get("relation")
    if R is not None:                                   # card 072's clear also drops the lean kinds' caches
        clear0 = R._clear

        def _clear(W):
            clear0(W)
            for kd in W.kinds.values():
                if isinstance(kd, LeanKind):
                    kd.forget()

        R._clear = _clear


def own_tries(tier, a):
    """(front, hand, category, weight) of every stored try of action a: the tries arm C's store is built from."""
    z = np.load(ROOT / "runs" / "095.1" / f"states_tier{tier}.npz")
    s = np.flatnonzero(z["act"] == a)
    e0, e1, app = z["e0"][s].astype(np.int64), z["e1"][s].astype(np.int64), z["app"]
    fp, hp = int(z["front"]), int(z["held"])
    u, h = app[e0[:, fp]], app[e0[:, hp]]
    cat = (e0[:, fp] // 5 != e1[:, fp] // 5).astype(np.int64) + 2 * (e0[:, hp] // 5 != e1[:, hp] // 5)
    return u, h, cat, z["w"][s]


def fit_beta(W, tier, a, SC, ALPHAS):
    u, h, cat, w = own_tries(tier, a)
    pairs, inv = np.unique(np.stack([u, h], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    G = np.zeros((len(pairs), 4))
    np.add.at(G, (inv, cat), w)
    Pc = np.stack([SC.categories(SC.arm_c(W, a, int(p[0]), int(p[1]))[0]) for p in pairs])
    Gt = G.sum(1, keepdims=True)
    msk = G > 0

    def ll(beta):
        P = (G - 1.0 + beta * Pc) / (Gt - 1.0 + beta)
        return float((G[msk] * np.log(np.maximum(P[msk], 1e-300))).sum())

    vals = [ll(b) for b in ALPHAS]
    j = int(np.argmax(vals))
    return float(ALPHAS[j]), {"beta": round(float(ALPHAS[j]), 6), "ll": round(vals[j], 2),
                              "ll_store_only": round(ll(1e9), 2), "pairs": int(len(pairs)),
                              "pairs_with_more_than_one_try": int((Gt[:, 0] > 1).sum()),
                              "tries": int(len(u)), "weight": float(w.sum())}


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        VP = T.VP
        SC = sys.modules["statechange"]
        OWN = sys.modules["own"]
        assert SC.MODE == "1", "card 099 needs arm C (WM_STATECHANGE=1)"
        t0 = time.monotonic()
        rep = {}
        for a in VP.INTER:
            kd = W.kinds[a]
            assert isinstance(kd, LeanKind)
            kd.beta, rep[VP.KNAME[a]] = fit_beta(W, tier, a, SC, OWN.ALPHAS)
            rep[VP.KNAME[a]]["borrowed"] = BORROWED[tier][a]
            rep[VP.KNAME[a]]["same_grid_value"] = abs(kd.beta - BORROWED[tier][a]) <= 1e-6 * max(1.0, kd.beta)
        SC.STATE["cache"].clear()                       # the fit's queries: not this run's (counters stay comparable)
        rep["seconds"] = round(time.monotonic() - t0, 1)
        info["lean"] = rep
        log(f"card 099: arm C's own beta {json.dumps(rep)}")
        if GATE:
            OUT.mkdir(parents=True, exist_ok=True)
            out = {"tier": tier, "betas": rep, "model_seconds": info.get("model_seconds")}
            (OUT / f"beta_tier{tier}.json").write_text(json.dumps(out, indent=1) + "\n")
            log(f"card 099 gate written: runs/099/beta_tier{tier}.json")
            sys.exit(0)
        return W, info

    T.setup = setup
