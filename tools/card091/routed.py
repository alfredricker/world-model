"""Card 091 in the agent: the key-form router (tools/card091/krouter.py) as recall's prior, in place of version 18's
neighbour vote; own tries first is unchanged.

  pick up, drop, toggle (card 051's IndexKind.predict):  P = (N_own + β P_route) / (|N_own| + β), β version 18's
  forward (card 038's Kind.cat_of on the forward kind):  P = (N_own + α P_route) / (|N_own| + α), α version 18's;
                                                         no own tries: P_route over the other stored tiles
  P_route(q) = (Σ_k exp(−‖e_q − e_k‖₁ / τ) C_k + α_r f) / (Σ_k exp(−‖e_q − e_k‖₁ / τ) |C_k| + α_r)

over the same action's stored keys k in this world's memory (their counts C_k by category, as they grow online). The
stored keys' embeddings are computed once at setup; keys added during an episode and the queries are embedded as
they come (on the CPU, one thread: the episode workers are forked).

  WM_ROUTER=runs/091/krouter_a.pt ...        tools/card069/run.py hooks this after setup (tools/card091/routed.py)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card091"))
import krouter as KR                                   # noqa: E402

STATS = {"router_queries": 0, "router_keys_embedded": 0}    # counters (the runner diffs them per episode)
INFO = {}
STATE = {}


@torch.no_grad()
def _embed(a, Z, mask):
    net = STATE["net"]
    role = STATE["role"][a]
    act = torch.full((len(Z),), a, dtype=torch.long)
    return net(torch.as_tensor(Z), role, torch.as_tensor(mask), act).numpy().astype(np.float64)


def key_tokens(W, a, keys):
    A = W.S.arr.astype(np.float32)
    if a == STATE["fwd"]:
        return A[[int(k[0]) for k in keys]][:, None], np.ones((len(keys), 1), bool)
    SP = sys.modules["slot_planner"]
    V = 6                                              # version 18's believed sets hold at most 6 tiles
    Z = np.zeros((len(keys), 2 + V, A.shape[1]), np.float32)
    m = np.zeros((len(keys), 2 + V), bool)
    for i, k in enumerate(keys):
        s = SP.SETS[int(k[2])][:V]
        Z[i, 0], Z[i, 1] = A[int(k[0])], A[int(k[1])]
        Z[i, 2:2 + len(s)] = A[s]
        m[i, :2 + len(s)] = True
    return Z, m


def stored(W, a, kd):
    """The embeddings of every stored key of action a: the keys stored at setup embedded once, keys added later (an
    episode's own tries; reset() drops them again) looked up by key."""
    n = len(kd.keys)
    base = STATE["E"][id(kd)]
    n0 = len(base)
    if n <= n0:
        return base[:n]
    tail = tuple(kd.keys[n0:n])
    c = STATE["tail"].get(id(kd))
    if c is None or c[0] != tail:
        c = STATE["tail"][id(kd)] = (tail, np.vstack([base] + [query(W, a, k)[None] for k in tail]))
    return c[1]


def query(W, a, q):
    c = STATE["Q"]
    r = c.get((a, q))
    if r is None:
        r = c[(a, q)] = _embed(a, *key_tokens(W, a, [q]))[0]
    return r


def route(W, a, kd, qs, C, exclude=None):
    """P_route for queries qs over the stored keys (counts C, by category)."""
    E = stored(W, a, kd)
    eq = np.stack([query(W, a, q) for q in qs])
    d = np.abs(eq[:, None] - E[None]).sum(-1)
    k = np.exp(-d / STATE["tau"])
    if exclude is not None:
        r = np.flatnonzero(exclude >= 0)
        k[r, exclude[r]] = 0.0
    N = k @ C
    f = C.sum(0) + 1.0
    f = f / f.sum()
    STATS["router_queries"] += len(qs)
    return (N + STATE["alpha"] * f) / (N.sum(1, keepdims=True) + STATE["alpha"])


def install(W, T):
    VP = T.VP
    torch.set_num_threads(1)
    net = KR.load(os.environ["WM_ROUTER"], dev="cpu")
    STATE.update(net=net, tau=float(net.log_tau.exp()), alpha=float(net.log_alpha.exp()), E={}, Q={}, tail={},
                 fwd=VP.FWD, role={VP.FWD: torch.tensor([0]), **{a: torch.tensor([0, 1] + [2] * 6) for a in VP.INTER}})
    acts = {id(W.kinds[a]): a for a in VP.INTER}
    STATE["acts"] = acts
    cls = type(W.kinds[VP.PICK])
    assert all(type(W.kinds[a]) is cls for a in VP.INTER), [type(W.kinds[a]).__name__ for a in VP.INTER]
    if not getattr(cls, "_card091", False):
        def predict(self, qs):
            a = STATE["acts"].get(id(self))
            if a is None:
                return predict.base(self, qs)
            n = len(self.keys)
            L = self.lc.shape[1]
            M = np.zeros((L, self.ncat))
            M[np.arange(L), np.asarray(self.lcat)[:L]] = 1.0
            C = self.lc[:n] @ M
            P = route(STATE["W"], a, self, qs, C)
            N = np.stack([self.own_counts(q) for q in qs]) @ M
            return (N + self.beta * P) / (N.sum(1, keepdims=True) + self.beta)

        predict.base = cls.predict
        cls.predict = predict
        cls._card091 = True
    STATE["W"] = W
    kd = W.kinds[VP.FWD]

    def cat_of(qs, kd=kd):
        todo = [q for q in dict.fromkeys(qs) if q not in kd.ccache]
        if todo:
            C = np.asarray(kd.counts, np.float64)[:, :3]
            own = np.array([kd.index.get(q, -1) for q in todo])
            P = route(W, VP.FWD, kd, todo, C, exclude=own)
            for i, q in enumerate(todo):
                p = P[i]
                if own[i] >= 0:
                    p = (C[own[i]] + kd.alpha * p) / (C[own[i]].sum() + kd.alpha)
                kd.ccache[q] = int(p.argmax()) if p.max() >= 0.5 else None
        return [kd.ccache[q] for q in qs]

    kd.cat_of = cat_of
    for a, k in [(a, W.kinds[a]) for a in VP.INTER] + [(VP.FWD, kd)]:
        STATE["E"][id(k)] = _embed(a, *key_tokens(W, a, k.keys))
        STATS["router_keys_embedded"] += len(k.keys)
    W.clear_lazy()
    INFO.update(router=os.environ["WM_ROUTER"], tau=round(STATE["tau"], 4), alpha=round(STATE["alpha"], 4),
                stored_keys={str(a): len(W.kinds[a].keys) for a in (VP.FWD,) + tuple(VP.INTER)})
    return dict(INFO)


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        info["router"] = install(W, T)
        log(f"card 091 router: {info['router']}")
        return W, info

    T.setup = setup
