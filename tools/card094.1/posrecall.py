"""Card 094.1 in the agent: recall's prior (card 091's router, card 091.1's identity groups) over object files with
their places relative to the agent.

- Every view recall is asked about (the planner's ctx_id, a real try's ctx_of_view) gets a context id that keeps the
  attended tokens' places: the same token set as version 20's id (so own tries first, admitted conditions and the
  situation index read exactly what they read now), plus, for the router, up to 6 attended tokens with their place,
  nearest first (pkeys.py's order).
- Queries about stored situations (card 047's, set form, no places) keep version 20's vote.
- For pick up, toggle and drop the router votes over this world's positioned stored keys (runs/094.1/pkeys_<tier>.npz,
  embedded once at setup) and the episode's own tries (positioned as they happen), within the query's (front, held)
  codes where any exist (card 091.1), else over all; τ and α are the positional router's.
- Forward is unchanged (card 087's properties, version 20).

  WM_PROUTER=runs/094.1/prouter.pt   (with version 20's flags; tools/card069/run.py hooks it after the router)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card094.1"))
sys.path.insert(0, str(ROOT / "tools" / "card094"))
import prouter as PR                                   # noqa: E402
import files as FILES                                  # noqa: E402

STATS = {"prouter_queries": 0, "prouter_identity": 0, "prouter_unpositioned": 0, "prouter_own_keys": 0}
STATE = {"ptok": {}, "pid": {}}
VMAX = 6


def _order(T):
    o = STATE.get("order")
    if o is None:
        WH = T.BL.WH
        o = np.lexsort((np.arange(len(WH)), np.abs(WH).sum(1)))
        o = STATE["order"] = [p for p in o.tolist() if p not in (T.BL.CENTRE, T.FRONT)]
    return o


def positioned(W, T, v, sid):
    """The context id for view v (handles per place) whose token set is sid: a copy of sid that keeps the places."""
    toks = []
    for p in _order(T):
        h = int(v[p])
        if h >= 0 and FILES.static(W, h):
            toks.append((h, p))
            if len(toks) == VMAX:
                break
    key = (int(sid), tuple(toks))
    r = STATE["pid"].get(key)
    if r is None:
        SP = sys.modules["slot_planner"]
        r = STATE["pid"][key] = len(SP.SETS)
        SP.SETS.append(SP.SETS[int(sid)])
        STATE["ptok"][r] = tuple(toks)
    return r


def _tokens(W, a, keys):
    """(Z, role, mask, place) for keys (front, held, positioned ctx id)."""
    A = W.S.arr.astype(np.float32)
    n = len(keys)
    Z = np.zeros((n, 2 + VMAX, A.shape[1]), np.float32)
    m = np.zeros((n, 2 + VMAX), bool)
    pl = np.zeros((n, 2 + VMAX), np.int64)
    for i, k in enumerate(keys):
        Z[i, 0], Z[i, 1] = A[int(k[0])], A[int(k[1])]
        m[i, :2] = True
        pl[i, 0], pl[i, 1] = STATE["front_place"], PR.HAND_PLACE
        toks = STATE["ptok"].get(int(k[2]), ())
        for j, (h, p) in enumerate(toks):
            Z[i, 2 + j], m[i, 2 + j], pl[i, 2 + j] = A[h], True, p
    role = torch.tensor([0, 1] + [2] * VMAX)
    return torch.as_tensor(Z), role, torch.as_tensor(m), torch.as_tensor(pl)


@torch.no_grad()
def _embed(a, Z, role, m, pl, bs=8192):
    net = STATE["net"]
    out = []
    for i in range(0, len(Z), bs):
        act = torch.full((len(Z[i:i + bs]),), a, dtype=torch.long)
        out.append(net(Z[i:i + bs], role, m[i:i + bs], act, pl[i:i + bs]))
    return torch.cat(out).numpy().astype(np.float32) if out else np.zeros((0, 32), np.float32)


def install(W, T, R):
    VP = T.VP
    torch.set_num_threads(1)
    net, fp = PR.load(os.environ["WM_PROUTER"], dev="cpu")
    STATE.update(net=net, front_place=fp, tau=float(net.log_tau.exp()), alpha=float(net.log_alpha.exp()), own={})
    K = np.load(ROOT / "runs" / "094.1" / f"pkeys_tier{STATE['tier']}.npz")
    CR = sys.modules["code_recall"]
    store = {}
    for a in (VP.PICK, VP.DROP, VP.TOG):
        tok = PR.tokens(K, a, "cpu", fp)
        E = _embed(a, *tok)
        g = {}
        for i, key in enumerate(zip(K[f"{a}_fid"].tolist(), K[f"{a}_hid"].tolist())):
            g.setdefault(key, []).append(i)
        store[a] = {"E": E, "C": K[f"{a}_C"].astype(np.float64), "G": {k: np.asarray(v) for k, v in g.items()}}
    STATE["store"] = store
    STATE["kd_n0"] = {a: len(W.kinds[a].keys) for a in store}
    route0 = R.route

    def own_keys(W, a, kd):
        """The episode's own tries of action a (kd.keys beyond setup's), embedded once each, with their counts."""
        n0, n = STATE["kd_n0"][a], len(kd.keys)
        c = STATE["own"].get(a)
        if c is None or c[0] != tuple(kd.keys[n0:n]):
            tail = list(kd.keys[n0:n])
            if tail:
                L = kd.lc.shape[1]
                M = np.zeros((L, kd.ncat))
                M[np.arange(L), np.asarray(kd.lcat)[:L]] = 1.0
                E = _embed(a, *_tokens(W, a, tail))
                C = kd.lc[n0:n] @ M
                ids = [(CR.code_id(W.S, int(k[0])), CR.code_id(W.S, int(k[1]))) for k in tail]
            else:
                E, C, ids = np.zeros((0, 32), np.float32), np.zeros((0, kd.ncat)), []
            c = STATE["own"][a] = (tuple(tail), E, C, ids)
            STATS["prouter_own_keys"] = sum(len(x[0]) for x in STATE["own"].values())
        return c[1], c[2], c[3]

    def route(W_, a, kd, qs, C, exclude=None):
        if a not in store:
            return route0(W_, a, kd, qs, C, exclude)
        pos = [i for i, q in enumerate(qs) if int(q[2]) in STATE["ptok"]]
        if len(pos) < len(qs):                         # stored situations (set form, card 047): version 20's vote
            STATS["prouter_unpositioned"] += len(qs) - len(pos)
            P0 = route0(W_, a, kd, qs, C, exclude)
            if not pos:
                return P0
            P0[pos] = route(W_, a, kd, [qs[i] for i in pos], C)
            return P0
        s = store[a]
        Eo, Co, ido = own_keys(W_, a, kd)
        eq = _embed(a, *_tokens(W_, a, qs))
        f = s["C"].sum(0) + 1.0
        f = f / f.sum()
        P = np.zeros((len(qs), s["C"].shape[1]))
        for i, q in enumerate(qs):
            key = (CR.code_id(W_.S, int(q[0])), CR.code_id(W_.S, int(q[1])))
            idx = s["G"].get(key)
            own = [j for j, k in enumerate(ido) if k == key]
            if idx is not None or own:
                STATS["prouter_identity"] += 1
                Es = s["E"][idx] if idx is not None else np.zeros((0, 32), np.float32)
                Cs = s["C"][idx] if idx is not None else np.zeros((0, s["C"].shape[1]))
                if own:
                    Es, Cs = np.vstack([Es, Eo[own]]), np.vstack([Cs, Co[own][:, :Cs.shape[1]]])
            else:
                Es, Cs = (np.vstack([s["E"], Eo]), np.vstack([s["C"], Co[:, :s["C"].shape[1]]])) if len(Eo) \
                    else (s["E"], s["C"])
            d = torch.cdist(torch.as_tensor(eq[i:i + 1]), torch.as_tensor(Es), p=1).double().numpy()[0]
            N = np.exp(-d / STATE["tau"]) @ Cs
            P[i] = (N + STATE["alpha"] * f) / (N.sum() + STATE["alpha"])
        STATS["prouter_queries"] += len(qs)
        return P

    R.route = route
    TK = T.TK

    def ctx_id(self, fid, pose):
        k = ("p", fid, pose)
        r = self.ctxmemo.get(k)
        if r is None:
            v = self.shown(self.facts[fid], T.F.M.A[pose][:T.BL.NV])
            sid = self.W.ctx_of(np.unique(v[self.W.keep]))
            r = self.ctxmemo[k] = positioned(self.W, T, v, sid)
        return r

    TK.TPlan.ctx_id = ctx_id
    cov0 = type(W).ctx_of_view

    def ctx_of_view(self, V):
        sid = cov0(self, V)
        return positioned(self, T, np.asarray(V)[:T.BL.NV], sid)

    type(W).ctx_of_view = ctx_of_view
    return {"prouter": os.environ["WM_PROUTER"], "tau": round(STATE["tau"], 4), "alpha": round(STATE["alpha"], 4),
            "stored_keys": {str(a): len(s["C"]) for a, s in store.items()}}


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        STATE["tier"] = tier
        FILES.STATE["VP"] = T.VP
        W, info = setup0(tier, log)
        info["prouter"] = install(W, T, sys.modules["routed"])
        log(f"card 094.1 positional router: {info['prouter']}")
        return W, info

    T.setup = setup
