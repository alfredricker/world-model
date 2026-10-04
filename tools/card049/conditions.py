"""Card 049: recall through the conditions that matter.

Card 047's agent, with recall for pick up, toggle and drop changed. A stored key i is weighted for a query q by
the conditions admitted for that action only:

  w_iq = exp(-sum over admitted conditions c of lambda_c d_c(i, q))
  P(class | q) = (sum_i w_iq n_i + a / L) / (sum_i w_iq |n_i| + a)

Candidate conditions, read from the encoder's vectors and codes (4 parts of 8):
  ("front", p)    L1 distance between the front tiles' part p
  ("held", p)     the same for the held tiles
  ("rel", p)      the front-held distance within part p, compared between i and q (a relation; no query-key product)
  ("view", t)     whether a token with code tuple t (all four pieces: the same appearance) lies in view (0 or 1),
                  compared between i and q. A single piece is not enough: pieces are shared across tiles (the
                  switch on and the wall share part 1's code), so a piece's presence loses which token it is on
                  (the shakedown, 2026-10-03)

Admission: the front tile's four parts are in from the start. Then the candidate whose best lambda most raises the
leave-one-key-out log likelihood of the stored outcomes (summed over tries) is added, while the gain exceeds
log(number of candidates). Then every admitted lambda and a are refitted together. A token present in every
stored try gives no gain and stays out. Moves, draw and undraw keep card 042's recall; the planner, tokens and
walking are card 047's.

Gate modes (section 5 of the card) filter the view before recall sees it, for version 8's recall:
  --view switch   only the floor and the switch's tiles (the upper bound: what the evaluator knows matters)
  --view none     only the floor (no view at all: must fail the switch world)

  bin/prun python tools/card049/conditions.py --recall v8 --view switch --chain one_door --seed 401 --n 100 --out runs/049_gate_upper_401.json
  bin/prun python tools/card049/conditions.py --recall cond --dev --arm A --seeds 399-399 --layouts 30 --out runs/049_shake_familiar.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card048"))
import chain as C                                      # noqa: E402  (card 047's agent and card 048's world)

S7, MV, VP, CR, SP = C.S7, C.MV, C.VP, C.CR, C.SP
KR = sys.modules["worldmodel.envs.keydoor_render"]
NP_ = 4                                                # parts
PW = 8                                                 # numbers per part
MAX_ADMIT = 12
GRID = np.exp(np.linspace(math.log(1e-2), math.log(1e2), 25))
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- codes of a handle

def tuple_of(S, h):
    """The 4 code pieces of the tile with handle h (-1 for a "new" piece)."""
    h = int(h)
    inv = CR.CC.setdefault("inv", {})
    if not inv:
        for t, hh in enumerate(S.hof.tolist()):
            inv.setdefault(int(hh), t)
    t = inv.get(h)
    tup = tuple(int(x) for x in CR.CC["ca"][t]) if t is not None else CR.code_of_vec(S.arr[h])
    return tuple(-1 if x == CR.NV.NEW else x for x in tup)


# ---------------------------------------------------------------- the kind

class CondKind(CR.CodeKind):
    """Card 042's kind (kept for the planner's situations, which read its codes and lambda2), predicting through
    admitted conditions."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tcodes = {}
        self.vcache = {}
        cand = ([("front", p) for p in range(NP_)] + [("held", p) for p in range(NP_)]
                + [("rel", p) for p in range(NP_)])
        seen = sorted({self.codes(h) for key in self.keys for h in SP.SETS[key[2]] if min(self.codes(h)) >= 0})
        cand += [("view", t) for t in seen]
        self.cand = cand
        self.thandle = {self.codes(h): int(h) for key in self.keys for h in SP.SETS[key[2]]}
        self.admit()

    # -- features
    def codes(self, h):
        r = self.tcodes.get(int(h))
        if r is None:
            r = self.tcodes[int(h)] = tuple_of(self.S, h)
        return r

    def view_feats(self, sid):
        r = self.vcache.get(sid)
        if r is None:
            present = {self.codes(h) for h in SP.SETS[sid]}
            r = self.vcache[sid] = np.array([1.0 if c[1] in present else 0.0 for c in self.cand
                                             if c[0] == "view"])
        return r

    def feats(self, keys):
        """Per key: front parts (4 x 8), held parts (4 x 8), relation per part (4), view presence (V)."""
        A = self.S.arr
        f = np.stack([A[int(k[0])] for k in keys]).reshape(len(keys), NP_, PW)
        h = np.stack([A[int(k[1])] for k in keys]).reshape(len(keys), NP_, PW)
        rel = np.abs(f - h).sum(-1)
        v = np.stack([self.view_feats(int(k[2])) for k in keys])
        return f, h, rel, v

    def cand_dist(self, ci, Fa, Fb):
        """Distance matrix (|a|, |b|) of candidate ci between two feature sets."""
        c = self.cand[ci]
        if c[0] == "front":
            return np.abs(Fa[0][:, None, c[1]] - Fb[0][None, :, c[1]]).sum(-1)
        if c[0] == "held":
            return np.abs(Fa[1][:, None, c[1]] - Fb[1][None, :, c[1]]).sum(-1)
        if c[0] == "rel":
            return np.abs(Fa[2][:, None, c[1]] - Fb[2][None, :, c[1]])
        j = self.vidx[ci]
        return np.abs(Fa[3][:, None, j] - Fb[3][None, :, j])

    # -- admission
    def admit(self):
        t0 = time.monotonic()
        self.vidx = {}
        j = 0
        for ci, c in enumerate(self.cand):
            if c[0] == "view":
                self.vidx[ci] = j
                j += 1
        n = len(self.keys)
        Fk = self.feats(self.keys)
        dev = "cuda"
        f32 = dict(dtype=torch.float64, device=dev)
        Dm, scale = [], []
        for ci in range(len(self.cand)):
            d = self.cand_dist(ci, Fk, Fk)
            pos = d[d > 0]
            s = float(np.median(pos)) if len(pos) else 1.0
            scale.append(s)
            Dm.append(torch.as_tensor(d / s, **f32))
        self.scale = np.array(scale)
        Ct = torch.as_tensor(self.lc, **f32)
        L = Ct.shape[1]
        tot = Ct.sum(1)
        eye = torch.eye(n, **f32).bool()

        def ll(dist, a):
            w = torch.exp(-dist).masked_fill(eye, 0.0)
            P = (w @ Ct + a / L) / ((w @ tot)[:, None] + a)
            return float((Ct * torch.log(P.clamp_min(1e-300))).sum())

        adm = [ci for ci, c in enumerate(self.cand) if c[0] == "front"]
        lam = {}
        base = torch.zeros((n, n), **f32)
        for ci in adm:                                   # the front tile: one shared lambda over its parts first
            base = base + Dm[ci]
        best = max(GRID, key=lambda g: ll(g * base, 1.0))
        for ci in adm:
            lam[ci] = float(best)
        dist = best * base
        cur = ll(dist, 1.0)
        start = cur
        cost = math.log(len(self.cand))
        trace = []
        while len(adm) < 4 + MAX_ADMIT:
            top = None
            for ci in range(len(self.cand)):
                if ci in adm:
                    continue
                for g in GRID:
                    v = ll(dist + g * Dm[ci], 1.0)
                    if top is None or v > top[0]:
                        top = (v, ci, float(g))
            if top is None or top[0] - cur <= cost:
                break
            v, ci, g = top
            trace.append({"cond": self.cname(ci), "gain": round(v - cur, 2)})
            adm.append(ci)
            lam[ci] = g
            dist = dist + g * Dm[ci]
            cur = v
        # joint refit of every admitted lambda and a
        th = torch.nn.Parameter(torch.tensor([math.log(lam[ci]) for ci in adm], **f32))
        la = torch.nn.Parameter(torch.tensor(0.0, **f32))
        Ds = torch.stack([Dm[ci] for ci in adm])
        opt = torch.optim.Adam([th, la], lr=0.05)
        for _ in range(300):
            w = torch.exp(-(torch.exp(th)[:, None, None] * Ds).sum(0)).masked_fill(eye, 0.0)
            a = torch.exp(la)
            P = (w @ Ct + a / L) / ((w @ tot)[:, None] + a)
            loss = -(Ct * torch.log(P.clamp_min(1e-300))).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.adm = adm
        self.lamc = torch.exp(th).detach().cpu().numpy() / self.scale[adm]
        self.beta = float(torch.exp(la).detach())
        self.base3 = (self.lc.copy(),)
        self.report["conditions"] = {
            "admitted": [self.cname(ci) for ci in adm], "trace": trace, "candidates": len(self.cand),
            "cost_nats": round(cost, 2), "ll_front_only": round(start, 2), "ll_greedy": round(cur, 2),
            "ll_refit": round(-float(loss.detach()), 2), "a": round(self.beta, 4), "seconds": round(time.monotonic() - t0, 1)}
        print(self.name, "admitted", self.report["conditions"]["admitted"], "trace", trace, flush=True)

    def cname(self, ci):
        c = self.cand[ci]
        if c[0] != "view":
            return f"{c[0]}:{c[1]}"
        h = self.thandle.get(c[1])
        t = CR.CC.get("inv", {}).get(h) if h is not None else None
        return f"view:{KR.OBJECTS[t // 5] if t is not None else c[1]}"

    # -- prediction
    def key_feats(self):
        tag = (len(self.keys), self.keys[-1])
        if getattr(self, "_fk", None) is None or self._fk[0] != tag:
            self._fk = (tag, self.feats(self.keys))
        return self._fk[1]

    def cond_weights(self, qs):
        Fk, Fq = self.key_feats(), self.feats(qs)
        d = np.zeros((len(qs), len(self.keys)))
        for ci, l in zip(self.adm, self.lamc):
            d += l * self.cand_dist(ci, Fq, Fk)
        return np.exp(-d)

    def weights(self, qs):
        w = self.cond_weights(qs)
        n = len(self.keys)
        lc = self.lc[:n]
        share = lc / lc.sum(1, keepdims=True)
        N = w @ lc
        L = lc.shape[1]
        s = np.full((len(qs), L), 1.0 / L)
        return w, np.zeros_like(w), np.ones((len(qs), 1)), share, N, s


# ---------------------------------------------------------------- the view filter (gate)

VIEW = {"mode": None}


def filtered_ctx(self, hs):
    hs = [int(h) for h in hs]
    keep = VIEW.get(id(self.S))
    if keep is None:
        floor = int(C.Kd.APP[KR.code(None)])
        sw = {int(C.Kd.APP[KR.code(o)]) for o in (("ball", "grey"), ("ball", "yellow"))}
        keep = VIEW[id(self.S)] = {floor} | (sw if VIEW["mode"] == "switch" else set())
    return SP.set_of(sorted(set(hs) & keep) or sorted(keep)[:1])


# ---------------------------------------------------------------- install

_kind = CR.kind


def cond_kind(name, S, parts, nkey, ncat, keys, cats, w, after=None, other=None, lam=None):
    if nkey == 3:
        return CondKind(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)
    return _kind(name, S, parts, nkey, ncat, keys, cats, w, after, other, lam)


def install(recall, view):
    if recall == "cond":
        CR.kind = cond_kind
    if view:
        VIEW["mode"] = view
        SP.SlotWorld.ctx_of = filtered_ctx


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    recall, view = get("--recall", "cond"), get("--view", None)
    for k in ("--recall", "--view"):
        if k in sys.argv:
            i = sys.argv.index(k)
            del sys.argv[i:i + 2]
    install(recall, view)
    if "--chain" in args or "--clutter" in args:
        import worlds as WD                             # noqa: E402  (card 048's chained rooms, the cluttered key world)
        WD.run(args, recall, view)
        return
    S7.install()
    MV.main()


if __name__ == "__main__":
    main()
