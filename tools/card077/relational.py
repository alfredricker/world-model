"""Card 077: card 076's router reading what tokens do and how they relate, not how they look.

Per token: its role (runs/077/affordances_<world>.json: walk onto, ends, pick up changes it, toggle changes it), its
marks and offset (card 076), and its relations to the named tokens and the hand: |P (z_i - z_j)| with card 070's P
(frozen), and the offsets between them. No token's vector is read directly. Arms: "relational" (roles and
relations), "roles" (no relations), "relations" (no roles), "appearance" (card 076's router).

Splits: "transfer" trains on tier 2 only and scores the decoy world (key known and trying); "within" is card 076's
protocol (all training, all test). Gate: identical inputs as the relational router reads them (relations rounded to
0.1) give the same outcome on at least 99% of training weighings.

  bin/prun python tools/card077/relational.py --out runs/077/relational.json
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card076"))
import router as R                                     # noqa: E402

DEV = R.DEV
OUT = Path(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == "--out"),
                "runs/077/relational.json"))
P = torch.load("runs/070/encoder.pt")["P"].float().to(DEV)          # (8, 32), card 070's relation, frozen
ROLES = {}
for wname in ("tier1", "tier2", "decoy"):
    a = json.load(open(f"runs/077/affordances_{wname}.json"))["affordances"]
    ROLES[wname] = {int(h): v for h, v in a.items()}


def world(src):
    return "decoy" if src.startswith("decoy") else src


class RFeat(R.Feat):
    """Card 076's tensors plus each token's role in its world."""

    def build(self, ws):
        b = super().build(ws)
        m = b["idx"].shape[1]
        role = np.zeros((len(ws), m, 4), np.float32)
        for i, w in enumerate(ws):
            tab = ROLES[world(w["source"])]
            for k, t in enumerate(w["tokens"]):
                role[i, k] = tab.get(int(t[0]), (0, 0, 0, 0))
        b["role"] = torch.tensor(role, device=DEV)
        return b


class RelRouter(R.Router):
    def __init__(self, nq, arm, h=64, de=32):
        nn.Module.__init__(self)
        self.arm = arm
        if arm == "appearance":
            din = 32 + 3 + 4 + 2 * (32 + 2) + nq
        else:
            din = 3 + 4 + 2 * 2 + nq + (4 if arm in ("relational", "roles") else 0) + \
                (3 * 8 if arm in ("relational", "relations") else 0)
        self.phi = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h + nq, h), nn.ReLU(), nn.Linear(h, de))
        self.w = nn.Parameter(torch.zeros(de))
        self.la = nn.Parameter(torch.tensor(0.0))

    def embed(self, V, b):
        if self.arm == "appearance":
            return R.Router.embed(self, V, b)
        z = V[b["idx"]]
        xy = b["xy"] / 6.0
        mk, mask, q = b["mk"], b["mask"], b["q"]
        parts = [xy, xy.abs().sum(-1, keepdim=True), mk]
        rel = []
        for bit in (0, 1, 2):                                   # the hand, the tokens A names, the tokens n names
            sel = mk[..., bit:bit + 1] * mask[..., None]
            cnt = sel.sum(1).clamp(min=1)
            zc = (z * sel).sum(1) / cnt
            if bit:
                parts.append(xy - ((xy * sel).sum(1) / cnt)[:, None])
            rel.append(torch.abs((z - zc[:, None]) @ P.T))
        if self.arm in ("relational", "roles"):
            parts.append(b["role"])
        if self.arm in ("relational", "relations"):
            parts += rel
        x = torch.cat(parts + [q[:, None].expand(-1, z.shape[1], -1)], -1)
        s = (self.phi(x) * mask[..., None]).sum(1)
        return self.rho(torch.cat([s, q], -1))


def fit(train, F, arm, steps=R.STEPS):
    """Card 076's fit (leave whole layouts out; real orders weighted by their rarity), for one arm."""
    b = F.build(train)
    y = torch.tensor([float(w["outcome"] == "n first") for w in train], device=DEV)
    lay = {l: i for i, l in enumerate(sorted({w["layout"] for w in train}))}
    L = torch.tensor([lay[w["layout"]] for w in train], device=DEV)
    p0 = float(y.mean())
    wpos = (1 - p0) / max(p0, 1e-6)
    model = RelRouter(len(F.qv), arm).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=R.LR)
    n = len(train)
    for _ in range(steps):
        qi = torch.randint(n, (min(R.BATCH, n),), device=DEV)
        bi = torch.randint(n, (min(R.BANK, n),), device=DEV)
        K = model.kernel(model.embed(F.V, R.sub(b, qi)), model.embed(F.V, R.sub(b, bi))) * (L[qi][:, None] != L[bi][None])
        Pv = model.vote(K, y[bi], p0).clamp(1e-6, 1 - 1e-6)
        yq = y[qi]
        loss = -(wpos * yq * torch.log(Pv) + (1 - yq) * torch.log(1 - Pv)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, b, y, p0


def gate_key(w, F, Vn):
    """The relational router's inputs, relations rounded to 0.1."""
    tab = ROLES[world(w["source"])]
    toks = w["tokens"]
    z = {i: Vn[F.hid[t[0]]] for i, t in enumerate(toks)}
    ctr = []
    for bit in (1, 2, 4):
        sel = [z[i] for i, t in enumerate(toks) if t[3] & bit]
        ctr.append(np.mean(sel, 0) if sel else np.zeros(32))
    Pn = P.cpu().numpy()
    key = []
    for i, t in enumerate(toks):
        rels = tuple(np.round(np.abs(Pn @ (z[i] - c)), 1).tolist() for c in ctr)
        key.append((tuple(tab.get(int(t[0]), (0, 0, 0, 0))), t[1], t[2], t[3], tuple(tuple(r) for r in rels)))
    return R.query_key(w) + tuple(sorted(key, key=str))


def run_split(train, test, F, arm, log):
    t0 = time.monotonic()
    model, b, y, p0 = fit(train, F, arm)
    pred = R.run_recall(R.OwnRouter(model, F, train, b, y, p0), test)
    log(f"  {arm}: fit and score {time.monotonic() - t0:.0f}s")
    return pred


def main():
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    train, H1 = R.load(R.TRAIN)
    test, H2 = R.load(R.TEST)
    H = {**H1, **H2}
    qkeys = sorted({(k, x) for w in train + test for k, x in enumerate(R.query_key(w))}, key=str)
    F = RFeat(H, {k: i for i, k in enumerate(qkeys)})
    Vn = F.V.cpu().numpy()
    res = {"note": "Card 077, tools/card077/relational.py"}
    grp = defaultdict(list)
    for w in train:
        grp[gate_key(w, F, Vn)].append(w["outcome"] == "n first")
    res["gate_inputs_consistent"] = round(sum(max(sum(v), len(v) - sum(v)) for v in grp.values()) / len(train), 4)
    res["gate_distinct_inputs"] = len(grp)
    log(f"gate: identical relational inputs consistent on {res['gate_inputs_consistent']} ({len(grp)} distinct)")
    tr2 = [w for w in train if w["source"] == "tier2"]
    decoy = [w for w in test if w["source"].startswith("decoy")]
    res["transfer"] = {"train": len(tr2), "train_orders": int(sum(w["outcome"] == "n first" for w in tr2)),
                       "test": len(decoy), "test_orders": int(sum(w["outcome"] == "n first" for w in decoy))}
    arms = ("relational", "appearance", "roles", "relations")
    preds = {}
    for arm in arms:
        preds[arm] = run_split(tr2, decoy, F, arm, log)
    preds["always none"] = [0.0] * len(decoy)
    res["transfer"]["report"] = R.report(decoy, preds, {})
    yd = np.array([w["outcome"] == "n first" for w in decoy])
    rel_ok = (np.array(preds["relational"]) >= 0.5) == yd
    app_ok = (np.array(preds["appearance"]) >= 0.5) == yd
    b_, c_ = int((rel_ok & ~app_ok).sum()), int((~rel_ok & app_ok).sum())
    res["transfer"]["relational_vs_appearance"] = {"b": b_, "c": c_, "p": round(R.mcnemar(b_, c_), 5)}
    for arm in arms:
        log(f"transfer {arm}: {json.dumps(res['transfer']['report'][arm]['all'])}")
    preds = {}
    for arm in ("relational", "appearance"):
        preds[arm] = run_split(train, test, F, arm, log)
    preds["always none"] = [0.0] * len(test)
    res["within"] = {"report": R.report(test, preds, {})}
    t2 = [i for i, w in enumerate(test) if w["source"] == "tier2"]
    y2 = np.array([test[i]["outcome"] == "n first" for i in t2])
    ok = {a: (np.array([preds[a][i] for i in t2]) >= 0.5) == y2 for a in ("relational", "appearance")}
    b_, c_ = int((ok["relational"] & ~ok["appearance"]).sum()), int((~ok["relational"] & ok["appearance"]).sum())
    res["within"]["tier2_relational_vs_appearance"] = {"b": b_, "c": c_, "p": round(R.mcnemar(b_, c_), 5)}
    for arm in ("relational", "appearance"):
        for s, v in res["within"]["report"][arm].items():
            log(f"within {arm} {s}: {json.dumps(v)}")
    m, b, y, p0 = fit(train, F, "relational", steps=10)
    rr = R.OwnRouter(m, F, train, b, y, p0)
    t0 = time.monotonic()
    R.run_recall(rr, test[:2000])
    res["seconds_per_prediction"] = round((time.monotonic() - t0) / 2000, 6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
