"""Card 095.7: when there is no plan and nothing unseen to walk to, act on the nearest token whose result arm C is
unsure of (WM_INFORM=1; needs WM_STATECHANGE=1).

Arm C's result is decided per part of the vector by the way (keep, copy, shift, set) with the most neighbour weight
(card 095.1). For shift and set, the winning value's share of that way's weight says how sure the result is; keep
and copy are sure. Certainty = the smallest share over parts. A candidate: arm C predicts the token changes
(P >= 0.5) and certainty < SURE. The fallback becomes: explore (unseen places; doors next to them) → the nearest
candidate (walk to face it, then act) → card 072's guesses → random.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.3"))
import statechange as SC                               # noqa: E402

SURE = float(os.environ.get("WM_INFORM_SURE", "0.8"))
STATS = {"inform_actions": 0, "inform_targets": 0}
STATE = {"cert": {}}
DEBUG = os.environ.get("WM_SC_DEBUG") == "1"


def certainty(W, a, u, h):
    """The front token's result certainty under action a with h held (1: sure), cached for the run."""
    k = (a, u, h)
    r = STATE["cert"].get(k)
    if r is None:
        p, nb, _ = SC.arm_c(W, a, u, h)
        st = SC.STATE["stores"][a]
        gi, wgt = nb[0]
        idx, wk = [], []
        for g, wg in zip(gi.tolist(), wgt.tolist()):
            for i in st.by_group.get(g, ()):
                idx.append(i)
                wk.append(wg * st.wch[i])
        r = 1.0
        if idx:
            idx, wk = np.array(idx), np.array(wk)
            top = np.argsort(-wk)[:64]
            top = top[wk[top] > 0]
            V, arr = st.arm.V, SC.STATE["arr"]
            Fi, after, w = st.Fch[idx[top]], arr[st.ach[idx[top]]], wk[top]
            q = SC.fields(u, h)[0]
            for k_ in range(4):
                sl = slice(k_ * 8, (k_ + 1) * 8)
                a_, own = after[:, sl], V[Fi[:, 0]][:, sl]
                score = {"keep": w[np.abs(a_ - own).sum(1) < 1e-5].sum()}
                for rr in range(Fi.shape[1] - 2):
                    cr = V[Fi[:, 2 + rr]][:, sl]
                    score[("copy", rr)] = w[(np.abs(a_ - cr).sum(1) < 1e-5) & (Fi[:, 2 + rr] >= 0)].sum()
                share = {}
                for vals, kind in ((a_ - own, "shift"), (a_, "set")):
                    _, inv = np.unique(np.round(vals, 5), axis=0, return_inverse=True)
                    sw = np.bincount(inv.ravel(), weights=w)
                    score[kind] = sw.max()
                    share[kind] = sw.max() / sw.sum()
                way = max(score, key=score.get)
                if way in share:
                    r = min(r, float(share[way]))
        r = STATE["cert"][k] = (float(p[0]), r)
    return r


def informative(pl, st):
    """The first move toward (or the action on) the nearest token with an uncertain result; None if none."""
    W, VP = pl.W, SC.STATE["VP"]
    M = VP.F.M
    f = pl.facts[st[0]]
    held = int(f[VP.HELD])
    known = SC.STATE["known_set"]
    cands = []
    for u in np.unique(f[:VP.NV]).tolist():
        if u < 0 or u not in known or M.free[u]:
            continue
        for a in (VP.TOG, VP.PICK):
            p, cert = certainty(W, a, int(u), held)
            if p < 0.5 or cert >= SURE:
                continue
            for j in np.flatnonzero(f[:VP.NV] == u).tolist():
                cands.append((pl.closeness(st, M.facing[j]) or (99, 9), a, int(u), j))
    STATS["inform_targets"] += len(cands)
    for _, a, u, j in sorted(cands):
        n = ("face", a, u, j)
        if pl.hold(n, st):
            STATS["inform_actions"] += 1
            return a
        res = pl.walk(n, st, 0, [], [], [])
        if res is not None:
            STATS["inform_actions"] += 1
            if DEBUG:
                print(f"   inform: {VP.KNAME.get(a, a)} {W.judge.name(u)} at {j}", flush=True)
            return int(res.action)
    return None


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        PV = sys.modules["partial"]
        TRY = sys.modules["trying"]
        SC.STATE.update(VP=T.VP)
        fallback0 = PV.fallback

        def fallback(pl, st, rng, rec):
            if TRY.STATE["hyp"] is None:
                a = PV.explore(pl, st)
                if a is not None:
                    rec["explore"] += 1
                    return a
                a = informative(pl, st)
                if a is not None:
                    return a
            return fallback0(pl, st, rng, rec)

        PV.fallback = fallback
        info["inform"] = {"sure": SURE}
        log(f"card 095.7: informative step, certainty below {SURE}")
        return W, info

    T.setup = setup
