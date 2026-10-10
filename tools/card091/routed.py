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
sys.path.insert(0, str(ROOT / "tools" / "card094.2"))
import admitted as ADM                                 # noqa: E402

STATS = {"router_queries": 0, "router_keys_embedded": 0, "router_identity": 0}    # counters (the runner diffs them per episode)
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
        s = ADM.view_of(W, a, SP.SETS[int(k[2])])[:V]   # card 094.2: WM_ROUTER_VIEW cuts the view (unset: all)
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


def _et(kd, E):
    """E as a float32 tensor, kept while the stored keys do not change."""
    c = STATE["ET"].get(id(kd))
    if c is None or c[0] is not E:
        c = STATE["ET"][id(kd)] = (E, torch.as_tensor(E, dtype=torch.float32))
    return c[1]


def ident(W, a, kd):
    """Per stored key its identity: the front and held codes (card 042's code_id); forward: the front code."""
    n = len(kd.keys)
    c = STATE["ID"].get(id(kd))
    if c is None or len(c) < n:
        CR = sys.modules["code_recall"]
        part = 1 if a == STATE["fwd"] else 2
        c = STATE["ID"][id(kd)] = np.array([[CR.code_id(W.S, int(h)) for h in k[:part]] for k in kd.keys],
                                           np.int64).reshape(n, part)
    return c[:n]


def groups(W, a, kd):
    """Stored keys by identity (front and held codes; forward: front code) → their indices; rebuilt as keys grow."""
    n = len(kd.keys)
    c = STATE["GR"].get(id(kd))
    if c is None or c[0] != n:
        ids = ident(W, a, kd)
        g = {}
        for i, row in enumerate(map(tuple, ids.tolist())):
            g.setdefault(row, []).append(i)
        c = STATE["GR"][id(kd)] = (n, {k: np.asarray(v, np.int64) for k, v in g.items()})
    return c[1]


def _vote(eq, Et, C, exclude_cols=None):
    d = torch.cdist(torch.as_tensor(eq, dtype=torch.float32), Et, p=1).double().numpy()
    k = np.exp(-d / STATE["tau"])                      # (queries, keys): no (queries, keys, 32) temporary
    if exclude_cols is not None:
        r = np.flatnonzero(exclude_cols >= 0)
        k[r, exclude_cols[r]] = 0.0
    return k @ C


DEBUG = os.environ.get("WM_ROUTER_DEBUG") is not None
DEBUG_ACT = int(os.environ.get("WM_ROUTER_DEBUG", "-1"))


def _nm(W, h):
    inv = STATE.setdefault("inv", {int(hh): t for t, hh in reversed(list(enumerate(W.S.hof.tolist())))})
    t = inv.get(int(h))
    return STATE["VP"].CO.name_of_code(t) if t is not None else int(h)


def route(W, a, kd, qs, C, exclude=None):
    """P_route for queries qs over the stored keys (counts C, by category). Card 091.1 (WM_ROUTER_ID=1): only over
    the stored keys with the query's front and held codes when there are any (computed over that group alone)."""
    E = stored(W, a, kd)
    eq = np.stack([query(W, a, q) for q in qs])
    f = C.sum(0) + 1.0
    f = f / f.sum()
    N = np.zeros((len(qs), C.shape[1]))
    rest = np.arange(len(qs))
    if STATE.get("identity"):
        CR = sys.modules["code_recall"]
        part = 1 if a == STATE["fwd"] else 2
        G = groups(W, a, kd)
        todo = []
        for i, q in enumerate(qs):
            idx = G.get(tuple(CR.code_id(W.S, int(h)) for h in q[:part]))
            if idx is not None and exclude is not None and exclude[i] >= 0:
                idx = idx[idx != exclude[i]]
            if idx is not None and len(idx):
                N[i] = _vote(eq[i:i + 1], torch.as_tensor(E[idx], dtype=torch.float32), C[idx])[0]
                STATS["router_identity"] += 1
                if DEBUG and a == DEBUG_ACT:
                    k = np.exp(-torch.cdist(torch.as_tensor(eq[i:i + 1], dtype=torch.float32),
                                            torch.as_tensor(E[idx], dtype=torch.float32), p=1).double().numpy()[0] / STATE["tau"])
                    top = np.argsort(-k)[:5]
                    print(f"   route: act {a} q {[_nm(W, h) for h in q]} group {len(idx)} counts {C[idx].sum(0).round(1).tolist()} "
                          f"vote {N[i].round(3).tolist()} top {[(round(float(k[t]), 3), C[idx][t].round(1).tolist(), [_nm(W, h) for h in kd.keys[idx[t]][2:]]) for t in top]}",
                          flush=True)
            else:
                todo.append(i)
        rest = np.asarray(todo, np.int64)
    if len(rest):
        ex = None if exclude is None else np.asarray(exclude)[rest]
        N[rest] = _vote(eq[rest], STATE["Et"](kd, E), C, ex)
    STATS["router_queries"] += len(qs)
    P = (N + STATE["alpha"] * f) / (N.sum(1, keepdims=True) + STATE["alpha"])
    if DEBUG and a == DEBUG_ACT:
        for i, q in enumerate(qs):
            if str(_nm(W, q[0])).startswith("box"):
                print(f"   routeP: q {[_nm(W, h) for h in q]} prior {f.round(3).tolist()} P {P[i].round(3).tolist()}", flush=True)
    return P


def install(W, T):
    VP = T.VP
    torch.set_num_threads(1)
    net = KR.load(os.environ["WM_ROUTER"], dev="cpu")
    STATE["VP"] = VP
    STATE.update(net=net, tau=float(net.log_tau.exp().detach()), alpha=float(net.log_alpha.exp().detach()), E={}, Q={}, tail={}, ET={}, Et=_et, ID={}, GR={}, identity=os.environ.get("WM_ROUTER_ID") == "1",
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
            out = (N + self.beta * P) / (N.sum(1, keepdims=True) + self.beta)
            if DEBUG and a == DEBUG_ACT:
                for i, q in enumerate(qs):
                    if str(_nm(STATE["W"], q[0])).startswith("box"):
                        print(f"   predict: q {[_nm(STATE['W'], h) for h in q[:2]]} ctx {q[2]} own {N[i].round(2).tolist()} "
                              f"beta {round(float(self.beta), 3)} route {P[i].round(3).tolist()} -> {out[i].round(3).tolist()}", flush=True)
            return out

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
    INFO.update(router=os.environ["WM_ROUTER"], identity_first=STATE["identity"], tau=round(STATE["tau"], 4), alpha=round(STATE["alpha"], 4),
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
