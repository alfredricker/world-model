"""Card 095.3: arm C in the agent for pick up, drop and toggle (WM_STATECHANGE=1); with WM_STATECHANGE=time, version
20's recall unchanged, timed and counted the same way (the data check and the comparison).

A query (a, u in front, h in the hand, view set) becomes two arm C instances, u at the front's place and h at the
hand's, with only the front and the hand known (card 095.3's gate: the other context places marginalised):

  P(token changes) = (B's kernel-weighted changed count + α · A's P) / (B's kernel-weighted count + α)
  B: card 095.2's radius index over the pruned store (memory 066's interaction tries; 095.1's admitted conditions
     on the known columns, 095.2's forgetting); A: card 095.1's network on the two tokens.

The four categories (front changed, hand changed) are the product of the two; this episode's own tries count first,
P = (N_own + β P_C) / (N_own + β), with version 20's β. What a changed token becomes: this episode's own try at the
same tokens, else B's relation, else A's.

  WM_STATECHANGE=1 <version 20's flags> bin/prun python tools/card069/run.py tiers --tier 2 ...
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
for c in ("card095.1", "card095.2", "card095.3"):
    sys.path.insert(0, str(ROOT / "tools" / c))
import tok                                             # noqa: E402
import arm_a as AA                                     # noqa: E402
import arm_b as AB                                     # noqa: E402
import scale as SC                                     # noqa: E402
import gate as GT                                      # noqa: E402

MODE = os.environ.get("WM_STATECHANGE", "1")
OUT = ROOT / "runs" / "095.3"
FRONT, HAND = 71, 169
STATS = {"sc_queries": 0, "sc_new": 0, "sc_seconds": 0.0, "sc_result_seconds": 0.0, "sc_tries": 0,
         "sc_tries_right": 0, "pick_hand_predicted": 0, "pick_hand_not_front": 0}
STATE = {"own": {}, "own_after": {}, "cache": {}, "res": {}}


# ---------------------------------------------------------------- the store (built once per tier, in the parent)

def build_store(tier, log):
    path = OUT / f"store_tier{tier}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    world = f"tier{tier}"
    z = tok.load(world)
    H, P, C, A = tok.build(z)
    near_n = int((np.abs(z["wh"][:169]).sum(1) <= tok.DMAX).sum()) - 1
    F, r, c = AB.fields(H, P, near_n)
    ch, af, w = C[r, c], A[r, c], z["w"][r].astype(np.float64)
    act = z["act"][r]
    arr = z["arr"]
    b_rep = json.loads((tok.RUNS / f"b_{world}.json").read_text())
    stores, info = {}, {}
    for a in (3, 4, 5):
        s = np.flatnonzero(act == a)                   # every stored try (memory 066), not only 095.1's training split
        arm = GT.marginal(SC.rebuild_arm(arr, F[s], w[s], ch[s], b_rep[str(a)]["admitted"]))
        drop, restored, maxmove = SC.forget(arm)
        kept = SC.subset(arm, ~drop)
        sc = s[ch[s]]
        u, inv = np.unique(np.concatenate([GT.mask_fields(F[sc]), af[sc][:, None]], 1), axis=0, return_inverse=True)
        wm = np.bincount(inv.ravel(), weights=w[sc], minlength=len(u))
        kept.Vt, kept._gc = None, None
        stores[a] = SC.Store(kept, u[:, :-1], u[:, -1], wm)
        info[str(a)] = {"admitted": [str(x) for x in kept.adm], "groups": int(len(arm.G)), "kept": int((~drop).sum()),
                        "restored": restored, "changed instances merged": int(len(u))}
        log(f"card 095.3 store, action {a}: {info[str(a)]}")
    out = {"stores": stores, "alpha": json.loads((tok.RUNS / "c_alpha.json").read_text())[world],
           "arr": arr, "info": info}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


# ---------------------------------------------------------------- arm C for one (action, front, hand)

def _vectors(W):
    """B's token vectors, grown with the world's store (row -1: absent)."""
    n = len(W.S.arr)
    if STATE.get("nV") != n:
        V = np.concatenate([W.S.arr.astype(np.float32), np.zeros((1, W.S.arr.shape[1]), np.float32)])
        for st in STATE["stores"].values():
            st.arm.V = V
        STATE["nV"] = n
        STATE["arr"] = W.S.arr.astype(np.float32)
    return STATE["arr"]


@torch.no_grad()
def _arm_a(W, a, u, h):
    arr = _vectors(W)
    X = torch.as_tensor(arr[[u, h]])[None]
    Pt = torch.tensor([[FRONT, HAND]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    logit, after, _, _ = STATE["net"](X, Pt, mask, torch.tensor([a]))
    known = STATE["known"]
    snap = known[np.abs(arr[known][None] - after[0].numpy()[:, None]).sum(-1).argmin(1)]
    return torch.sigmoid(logit[0]).numpy().astype(np.float64), snap


def fields(u, h):
    f = np.full((2, 15), -1, np.int64)
    f[:, 2], f[:, 14] = u, h
    f[0, :2] = (u, FRONT)
    f[1, :2] = (h, HAND)
    return f


def arm_c(W, a, u, h):
    """(P(front changes), P(hand changes)) and what B needs for the results, cached for the run (the store is
    fixed; this episode's own tries are applied on top)."""
    k = (a, u, h)
    r = STATE["cache"].get(k)
    if r is None:
        STATS["sc_new"] += 1
        pa, snap = _arm_a(W, a, u, h)
        st, al = STATE["stores"][a], STATE["alpha"][str(a)]
        p, nb = np.zeros(2), []
        for i, f in enumerate(fields(u, h)):
            Nc, N, gi, wg = st.vote(f)
            p[i] = (Nc + al * pa[i]) / (N + al)
            nb.append((gi, wg))
        r = STATE["cache"][k] = (p, nb, snap)
    return r


def results(W, a, u, h):
    """What the front's and the hand's tokens become, if they change: B's relation, else A's."""
    k = (a, u, h)
    r = STATE["res"].get(k)
    if r is None:
        p, nb, snap = arm_c(W, a, u, h)
        st = STATE["stores"][a]
        _vectors(W)
        out = []
        for i, f in enumerate(fields(u, h)):
            x = st.result(f, *nb[i], STATE["known"], STATE["arr"])
            out.append(int(x) if x >= 0 else int(snap[i]))
        r = STATE["res"][k] = tuple(out)
    return r


def categories(p):
    pf, ph = p
    return np.array([(1 - pf) * (1 - ph), pf * (1 - ph), (1 - pf) * ph, pf * ph])


# ---------------------------------------------------------------- hooks

def _install_c(W, T):
    VP = T.VP
    torch.set_num_threads(1)
    data = build_store(STATE["tier"], STATE["log"])
    arr0 = data["arr"]
    assert np.allclose(arr0, W.S.arr[:len(arr0)]), "095.1's token vectors are not this world's"
    net = AA.StateNet()
    net.load_state_dict(torch.load(tok.RUNS / "arm_a.pt", map_location="cpu")["net"])
    net.eval()
    STATE.update(stores=data["stores"], alpha=data["alpha"], net=net, known=np.unique(np.asarray(T.Kd.APP, np.int64)),
                 acts={id(W.kinds[a]): a for a in VP.INTER}, W=W)
    cls = type(W.kinds[VP.PICK])
    assert all(type(W.kinds[a]) is cls for a in VP.INTER)
    assert all(W.kinds[a].ncat == 4 for a in VP.INTER)
    base_predict = cls.predict

    def predict(self, qs):
        a = STATE["acts"].get(id(self))
        if a is None:
            return base_predict(self, qs)
        t0 = time.perf_counter()
        out = np.zeros((len(qs), 4))
        for i, q in enumerate(qs):
            u, h = int(q[0]), int(q[1])
            Pc = categories(arm_c(STATE["W"], a, u, h)[0])
            N = STATE["own"].get((a, u, h))
            out[i] = Pc if N is None else (N + self.beta * Pc) / (N.sum() + self.beta)
        STATS["sc_queries"] += len(qs)
        STATS["sc_seconds"] += time.perf_counter() - t0
        return out

    cls.predict = predict
    for a in VP.INTER:
        def result(q, c, a=a):
            t0 = time.perf_counter()
            u, h = int(q[0]), int(q[1])
            r = STATE["own_after"].get((a, u, h, int(c)))
            if r is None:
                fa, ha = results(STATE["W"], a, u, h)
                r = (fa if c & 1 else u, ha if c & 2 else h)
            STATS["sc_result_seconds"] += time.perf_counter() - t0
            return r

        W.kinds[a].result = result
    W.clear_lazy()
    return {"mode": "arm C", "stores": data["info"], "alpha": data["alpha"]}


def _install_time(W, T):
    """Version 20's recall for the three actions, timed (no change)."""
    VP = T.VP
    acts = {id(W.kinds[a]) for a in VP.INTER}
    cls = type(W.kinds[VP.PICK])
    base_predict = cls.predict

    def predict(self, qs):
        if id(self) not in acts:
            return base_predict(self, qs)
        t0 = time.perf_counter()
        out = base_predict(self, qs)
        STATS["sc_queries"] += len(qs)
        STATS["sc_seconds"] += time.perf_counter() - t0
        return out

    cls.predict = predict
    for a in VP.INTER:
        kd = W.kinds[a]
        plain = kd.result

        def result(q, c, plain=plain):
            t0 = time.perf_counter()
            r = plain(q, c)
            STATS["sc_result_seconds"] += time.perf_counter() - t0
            return r

        kd.result = result
    return {"mode": "version 20, timed"}


def _install_common(W, T):
    """Both modes: this episode's own tries (arm C reads them), the check of each real try against the prediction,
    and criterion 3's count of pick ups predicted to put another token in the hand than the one in front."""
    VP = T.VP
    World = type(W)
    learn0, reset0, outcome0 = World.learn_try, World.reset, World.outcome

    def learn_try(self, V, a, V2, end):
        if a in VP.INTER:
            f0, h0, f1, h1 = int(V[VP.FRONT]), int(V[VP.HELD]), int(V2[VP.FRONT]), int(V2[VP.HELD])
            c_pred, after_pred = outcome0(self, a, f0, h0, self.ctx_of_view(V))
            cat = int(f1 != f0) + 2 * int(h1 != h0)
            STATS["sc_tries"] += 1
            STATS["sc_tries_right"] += int((c_pred or 0) == cat and (cat == 0 or tuple(after_pred) == (f1, h1)))
        out = learn0(self, V, a, V2, end)
        if a in VP.INTER:
            k = (a, f0, h0)
            N = STATE["own"].setdefault(k, np.zeros(4))
            N[cat] += 1.0
            STATE["own_after"][(a, f0, h0, cat)] = (f1, h1)
        return out

    def reset(self):
        STATE["own"].clear()
        STATE["own_after"].clear()
        return reset0(self)

    def outcome(self, a, u, h, cid):
        c, after = outcome0(self, a, u, h, cid)
        if a == VP.PICK and c and c & 2:
            STATS["pick_hand_predicted"] += 1
            STATS["pick_hand_not_front"] += int(after[1] != int(u))
        return c, after

    World.learn_try, World.reset, World.outcome = learn_try, reset, outcome


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        STATE.update(tier=tier, log=log)
        info["statechange"] = _install_c(W, T) if MODE == "1" else _install_time(W, T)
        _install_common(W, T)
        log(f"card 095.3: {json.dumps(info['statechange'])}")
        return W, info

    T.setup = setup
