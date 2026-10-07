"""Card 080: card 079's router with a monotone readout of the relation as the vote's prior (report only).

Arms per fit: "vote" (card 079: the outcome rates as prior), "readout" (the readout alone), "both" (the vote with the
readout as its prior, weight beta). The readout: per category, logit = u . roles + (v . roles + v0) * r + b, with r =
|P (z_front - z_held)|_1 and roles the two tiles' roles under the other actions: linear, so monotone, in r. Seeds 79
(judged), 80, 81.

Card 079's description follows.

  WM_REL_ENCODER=runs/070/encoder.pt bin/prun python tools/card079/effects.py red      # one fold
  ... effects.py none                                                                  # the upper bound: nothing held out

Per fold: version 18's memory for the decoy world with the fold hue's door openings removed (card 070), toggle recall's
stored tries (front, held, view set; outcome counts), and a table of queries from the decoy world's start view: the six
locked doors, a wall and the floor in front, each with the six keys or nothing held. The evaluator's truth: only a
locked door with its own key opens. Version 18's recall (W.outcome) and the router answer every cell.

The router: per stored try, the front and held tiles' roles under the other actions (forward moves onto it; a pick up
changes it, empty-handed: the agent's own recall), |P (z_front - z_held)|, and per tile in view |P (z_v - z_front)|,
|P (z_v - z_held)| and its role, summed; no tile's vector is read directly. A stored try counts by
exp(-|w * (e_q - e_j)|_1); category probabilities are the vote of the stored counts, with a prior. Fitted by the
likelihood of the stored outcomes, each try predicted from tries of other (front, held) combinations (card 071).
"Own tries first" stays: a query identical to a stored try takes that try's counts.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
HUE = sys.argv[1]
sys.argv = ["run.py", "decoy", "--fold", HUE if HUE != "none" else "red", "--hold", "opening", "--online", "0",
            "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

T, R = RUN.T, RUN.R
VP, TK, SP = T.VP, T.TK, sys.modules["slot_planner"]
DEV = "cuda"
torch.manual_seed(79)
t00 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)

T.ENVS[1] = R.register()
T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
_memory = T.memory


def memory(tier, pool=None):
    D, stats = _memory(tier, pool)
    if HUE == "none":
        return D, stats
    f = D["ego0"][:, T.FRONT].astype(np.int64) // 5 * 5
    h = D["ego0"][:, T.HELD].astype(np.int64) // 5 * 5
    drop = (D["act"] == VP.TOG) & (f == T.KR.code(("door", HUE, 0))) & (h == T.KR.code(("key", HUE)))
    inter = np.isin(D["act"], T.INTER)
    keep_t = ~drop[inter]
    D = {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}
    return D, {**stats, "opening_tries_removed": int(drop.sum()), "rows": int(len(D["act"]))}


T.memory = memory
W, info = T.setup(1, log)
kd = W.kinds[VP.TOG]
app = lambda o: int(T.Kd.APP[T.KR.code(o)])
pl = T.S7.Plan047(W)
env = T.make(1)
env.reset(seed=T.SEED_TEST + 999)
codes = T.now_codes(env)
_, st, _, _ = pl.observe(None, T.PV.crop(T.Kd.APP[codes], codes))
ctx, empty = pl.ctx_id(st[0], st[1]), int(pl.facts[st[0]][VP.HELD])
P = torch.load(os.environ.get("WM_REL_ENCODER", "runs/070/encoder.pt"))["P"].double().numpy()
Z = W.S.arr

# ---------------------------------------------------------------- the table and version 18's answers
doors = {c: app(("door", c, 0)) for c in T.HUES}
keys = {c: app(("key", c)) for c in T.HUES}
NAME = {}
for h in range(len(Z)):                                # the evaluator's names, to build the table only
    try:
        NAME.setdefault(W.judge.name(h), h)
    except Exception:                                  # noqa: BLE001
        pass
fronts = [("door " + c, doors[c]) for c in T.HUES] + [(nm, NAME[nm]) for nm in ("wall", "floor") if nm in NAME]
helds = [("key " + c, keys[c]) for c in T.HUES] + [("nothing", empty)]
cells = [(fn, fu, hn, hh, fn.startswith("door") and hn.startswith("key") and fn[5:] == hn[4:])
         for fn, fu in fronts for hn, hh in helds]
v18 = [int(W.outcome(VP.TOG, fu, hh, ctx)[0]) for _, fu, _, hh, _ in cells]

# ---------------------------------------------------------------- roles and inputs (card 079)
ROLE = {}


def role(h):
    h = int(h)
    if h not in ROLE:
        ROLE[h] = [float(TK.free_of(h)), float(W.outcome(VP.PICK, h, empty, ctx)[0] != 0)]
    return ROLE[h]


def feats(key):
    f, h, sid = key
    pair = role(f) + role(h) + np.abs(P @ (Z[f] - Z[h])).tolist()
    view = [np.abs(P @ (Z[v] - Z[f])).tolist() + np.abs(P @ (Z[v] - Z[h])).tolist() + role(v)
            for v in SP.SETS[sid]] or [[0.0] * 18]
    return pair, view


def tensors(ks):
    fs = [feats(k) for k in ks]
    m = max(len(v) for _, v in fs)
    pair = torch.tensor([p for p, _ in fs], dtype=torch.float32, device=DEV)
    view = torch.zeros(len(ks), m, 18, device=DEV)
    mask = torch.zeros(len(ks), m, device=DEV)
    for i, (_, v) in enumerate(fs):
        view[i, :len(v)] = torch.tensor(v, dtype=torch.float32)
        mask[i, :len(v)] = 1
    return pair, view, mask


class Model(nn.Module):
    def __init__(self, ncat, h=64, de=16):
        super().__init__()
        self.pv = nn.Sequential(nn.Linear(18, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(12 + h, h), nn.ReLU(), nn.Linear(h, de))
        self.w = nn.Parameter(torch.zeros(de))
        self.la = nn.Parameter(torch.tensor(0.0))
        self.U = nn.Linear(4, ncat)                       # the readout: logit = U roles + (V roles + v0) r
        self.V = nn.Linear(4, ncat)

    def embed(self, pair, view, mask):
        return self.rho(torch.cat([pair, (self.pv(view) * mask[..., None]).sum(1)], -1))

    def kernel(self, a, b):
        return torch.exp(-(torch.abs(a[:, None] - b[None]) * nn.functional.softplus(self.w)).sum(-1))

    def readout(self, pair):
        roles, r = pair[:, :4], pair[:, 4:].sum(1, keepdim=True)
        return torch.softmax(self.U(roles) + self.V(roles) * r, -1)

    def predict(self, arm, pq, eq, e, C, prior, mask=None):
        if arm == "readout":
            return self.readout(pq)
        K = self.kernel(eq, e)
        if mask is not None:
            K = K * mask
        a = nn.functional.softplus(self.la)
        pri = prior[None] if arm == "vote" else self.readout(pq)
        return (K @ C + a * pri) / (K @ C.sum(1, keepdim=True) + a)


C = torch.tensor(kd.counts, dtype=torch.float32, device=DEV)
prior = C.sum(0) / C.sum()
combo = {}
G = torch.tensor([combo.setdefault(k[:2], len(combo)) for k in kd.keys], device=DEV)
Xp, Xv, Xm = tensors(kd.keys)
qp, qv, qm = tensors([(fu, hh, ctx) for _, fu, _, hh, _ in cells])
n = len(kd.keys)
fold = [i for i, c in enumerate(cells) if c[0] == f"door {HUE}" and c[2] == f"key {HUE}"]
right_of = lambda cats: [bool((c != 0) == truth) for c, (_, _, _, _, truth) in zip(cats, cells)]
v18r = right_of(v18)
runs = []
for seed in (79, 80, 81):
    for arm in ("vote", "readout", "both"):
        torch.manual_seed(seed)
        model = Model(C.shape[1]).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3 if arm != "readout" else 1e-2)
        for step in range(2000):
            qi = torch.randint(n, (min(512, n),), device=DEV)
            e = model.embed(Xp, Xv, Xm) if arm != "readout" else None
            Pc = model.predict(arm, Xp[qi], None if e is None else e[qi], e, C, prior,
                               mask=(G[qi][:, None] != G[None]))
            loss = -(C[qi] * torch.log(Pc.clamp(min=1e-6))).sum() / C[qi].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            e = model.embed(Xp, Xv, Xm)
            Pq = model.predict(arm, qp, model.embed(qp, qv, qm), e, C, prior).cpu().numpy()
            dkey = model.V.weight @ torch.tensor([0., 0., 0., 1.], device=DEV) + model.V.bias   # door [0,0], key [0,1]
            beta = float(nn.functional.softplus(model.la))
        cats = []
        for (fn, fu, hn, hh, _), p in zip(cells, Pq):
            own = kd.index.get((fu, hh, ctx))
            if own is not None:                            # own tries first (card 050)
                cnt = kd.counts[own]
                p = cnt / cnt.sum()
            cats.append(int(p.argmax()) if p.max() >= 0.5 else 0)
        rr = right_of(cats)
        runs.append({"seed": seed, "arm": arm, "right": sum(rr), "of": len(cells),
                     "fold_pair": [cats[i] for i in fold], "fold_pair_p": [[round(float(x), 3) for x in Pq[i]] for i in fold],
                     "new_errors": [(cells[i][0], cells[i][2]) for i in range(len(cells)) if not rr[i] and i not in fold],
                     "weight_of_prior": round(beta, 4), "door_key_slope_on_r": [round(float(x), 3) for x in dkey]})
        log(json.dumps(runs[-1]))
res = {"note": "Card 080, tools/card080/ordered.py", "fold": HUE, "toggle_keys": n,
       "v18_right": sum(v18r), "v18_fold_pair": [v18[i] for i in fold],
       "fold_relation": [round(float(np.abs(P @ (Z[cells[i][1]] - Z[cells[i][3]])).sum()), 3) for i in fold],
       "runs": runs}
out = ROOT / "runs" / "080" / f"fold_{HUE}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1, default=str) + "\n")
log("wrote " + str(out))
