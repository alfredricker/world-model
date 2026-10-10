"""Card 094's attention: which tokens get an object file.

A token is attended when
  - it is the goal's token (pl.goal = ("has", h)), or the goal is "the episode ends" and forward onto it ends it;
  - it is named in what version 20 keeps between steps (the planner's kept choices, card 060's door, card 072's
    guess); or
  - some action can change it: memory holds a pick up or toggle with a token of its code in front after which the
    front changed (recall's own evidence, same code), or card 087's property network says pick up or toggle changes
    it (P >= 0.5). Failed tries alone do not rule a token out (a failed try is situational, card 072: a locked door
    of a colour never opened in memory may open with its key); only memory and the properties together do.
Floor, walls and other tokens no action changes get no file; they are the layout. Cached per handle.
"""
import sys

import numpy as np

STATE = {}


def _code(W, h):
    return sys.modules["code_recall"].code_id(W.S, int(h))


def _memory(W, VP):
    """Per action (pick up, toggle): codes tried in front, and codes after whose try the front changed."""
    r = STATE.get("mem")
    if r is None:
        tried, changed = set(), set()
        for a in (VP.PICK, VP.TOG):
            kd = W.kinds[a]
            n = len(kd.keys)
            aw = np.asarray(kd.aw)[:n]
            front_changed = aw[:, [c for c in range(aw.shape[1]) if c & 1]].sum(1) > 0
            for k, ch in zip(kd.keys[:n], front_changed):
                c = _code(W, k[0])
                tried.add(c)
                if ch:
                    changed.add(c)
        r = STATE["mem"] = (tried, changed)
    return r


def _props(W, h):
    """Card 087's ensemble: P(pick up changes it), P(toggle changes it)."""
    walk = sys.modules.get("walk")
    if walk is None or "nets" not in walk.STATE:
        return 0.0, 0.0
    z = np.asarray(W.S.arr[int(h)], np.float64)
    ps, ts = [], []
    for layers in walk.STATE["nets"]:
        x = z
        for j, (Wt, b) in enumerate(layers):
            x = x @ Wt.T + b
            if j < 2:
                x = np.maximum(x, 0.0)
        ps.append(1 / (1 + np.exp(-x[3])))
        ts.append(1 / (1 + np.exp(-x[4])))
    return float(np.mean(ps)), float(np.mean(ts))


def changeable(W, VP, h):
    """Some action can change token h (by its code): memory saw it change, or card 087's properties say it can."""
    c = STATE.setdefault("ch", {})
    r = c.get(int(h))
    if r is None:
        tried, changed = _memory(W, VP)
        r = _code(W, h) in changed
        if not r:
            p, t = _props(W, h)
            r = p >= 0.5 or t >= 0.5
        c[int(h)] = r
    return r


def ends(W, VP, h):
    c = STATE.setdefault("end", {})
    r = c.get(int(h))
    if r is None:
        r = c[int(h)] = W.move_cat(VP.FWD, int(h)) == VP.ENDED
    return r


def kept_tokens(pl, st):
    """Tokens named in what version 20 keeps between steps: kept choices, card 060's door, card 072's guess."""
    f = pl.facts[st[0]]
    out = set()
    for a, u, j in (getattr(pl, "commit", None) or {}).values():   # card 051's opkey: (action, token, place)
        if u is not None:
            out.add(int(u))
        if j is not None and 0 <= j < len(f) and f[j] >= 0:
            out.add(int(f[j]))
    j = getattr(pl, "look_j", None)
    if j is not None and 0 <= j < len(f) and f[j] >= 0:
        out.add(int(f[j]))
    TRY = sys.modules.get("trying")
    hyp = TRY.STATE.get("hyp") if TRY is not None else None
    if hyp is not None:
        out.add(int(hyp["u"]))
    return out


def attended(W, VP, pl, st, h):
    goal = getattr(pl, "goal", ("end",))
    if goal[0] == "has" and int(goal[1]) == int(h):
        return True
    if goal[0] == "end" and ends(W, VP, h):
        return True
    return changeable(W, VP, h) or int(h) in kept_tokens(pl, st)
