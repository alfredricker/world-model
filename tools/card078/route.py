"""Card 078: "on the way" per token, added to card 077's relational router; the transfer split and card 076's
protocol.

On the way: making the token walkable shortens, or opens, the agent's way to a tile next to the achiever's target.
Computed on the weighing's own tokens (the agent's belief): walkable as recall predicts (card 077's roles); the agent
stands where the hand token's offset says; breadth-first over the four neighbours, headings ignored. Two searches per
weighing, from the agent (dA) and from the walkable tiles next to the target (dG); a token i not walkable is on the
way when min over its neighbours a, b of dA[a] + 2 + dG[b] (or dA[a] + 1 when i is itself next to the target) is
below the current distance.

  bin/prun python tools/card078/route.py --out runs/078/route.json
"""
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card077"))
import relational as RL                                # noqa: E402

R, DEV = RL.R, RL.DEV
OUT = Path(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == "--out"), "runs/078/route.json"))
NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
INF = 10 ** 6


def bfs(starts, ok):
    d = {s: 0 for s in starts}
    q = deque(starts)
    while q:
        x, y = q.popleft()
        for dx, dy in NB:
            p = (x + dx, y + dy)
            if p not in d and ok(p):
                d[p] = d[(x, y)] + 1
                q.append(p)
    return d


def on_way(w):
    """Per token (in w["tokens"] order): 1 if it is on the way from the agent to a tile next to the target."""
    tab = RL.ROLES[RL.world(w["source"])]
    toks = w["tokens"]
    hand = next(t for t in toks if t[3] & 1)
    agent = (hand[1], hand[2])
    at = {(t[1], t[2]): t for t in toks if not t[3] & 1}
    walk = lambda p: p == agent or (p in at and tab.get(int(at[p][0]), (0, 0, 0, 0))[0] == 1)
    goals = [p for p in NB if walk(p)]
    dA = bfs([agent], walk)
    dG = bfs(goals, walk) if goals else {}
    d0 = min((dA[g] for g in goals if g in dA), default=INF)
    out = []
    for t in toks:
        p = (t[1], t[2])
        if t[3] & 1 or walk(p):
            out.append(0)
            continue
        best = INF
        nbs = [(p[0] + dx, p[1] + dy) for dx, dy in NB]
        for a in nbs:
            if a in dA:
                if p in NB:
                    best = min(best, dA[a] + 1)
                for b in nbs:
                    if b in dG:
                        best = min(best, dA[a] + 2 + dG[b])
        out.append(int(best < d0))
    return out


class WFeat(RL.RFeat):
    def build(self, ws):
        b = super().build(ws)
        m = b["idx"].shape[1]
        way = np.zeros((len(ws), m, 1), np.float32)
        for i, w in enumerate(ws):
            ow = w.get("_way")
            if ow is None:
                ow = w["_way"] = on_way(w)
            way[i, :len(ow), 0] = ow
        b["way"] = torch.tensor(way, device=DEV)
        return b


class WayRouter(RL.RelRouter):
    """Card 077's relational router plus on the way ("route"), or without card 070's relations ("route only")."""

    def __init__(self, nq, arm, h=64, de=32):
        nn.Module.__init__(self)
        self.arm = arm
        din = 3 + 4 + 2 * 2 + nq + 4 + 1 + (3 * 8 if arm == "route" else 0)
        self.phi = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h + nq, h), nn.ReLU(), nn.Linear(h, de))
        self.w = nn.Parameter(torch.zeros(de))
        self.la = nn.Parameter(torch.tensor(0.0))

    def embed(self, V, b):
        z = V[b["idx"]]
        xy = b["xy"] / 6.0
        mk, mask, q = b["mk"], b["mask"], b["q"]
        parts = [xy, xy.abs().sum(-1, keepdim=True), mk, b["role"], b["way"]]
        rel = []
        for bit in (0, 1, 2):
            sel = mk[..., bit:bit + 1] * mask[..., None]
            cnt = sel.sum(1).clamp(min=1)
            zc = (z * sel).sum(1) / cnt
            if bit:
                parts.append(xy - ((xy * sel).sum(1) / cnt)[:, None])
            rel.append(torch.abs((z - zc[:, None]) @ RL.P.T))
        if self.arm == "route":
            parts += rel
        x = torch.cat(parts + [q[:, None].expand(-1, z.shape[1], -1)], -1)
        s = (self.phi(x) * mask[..., None]).sum(1)
        return self.rho(torch.cat([s, q], -1))


def fit(train, F, arm):
    if arm not in ("route", "route only"):
        return RL.fit(train, F, arm)
    b = F.build(train)
    y = torch.tensor([float(w["outcome"] == "n first") for w in train], device=DEV)
    lay = {l: i for i, l in enumerate(sorted({w["layout"] for w in train}))}
    L = torch.tensor([lay[w["layout"]] for w in train], device=DEV)
    p0 = float(y.mean())
    wpos = (1 - p0) / max(p0, 1e-6)
    model = WayRouter(len(F.qv), arm).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=R.LR)
    n = len(train)
    for _ in range(R.STEPS):
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


def main():
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    train, H1 = R.load(R.TRAIN)
    test, H2 = R.load(R.TEST)
    qkeys = sorted({(k, x) for w in train + test for k, x in enumerate(R.query_key(w))}, key=str)
    F = WFeat({**H1, **H2}, {k: i for i, k in enumerate(qkeys)})
    Vn = F.V.cpu().numpy()
    for w in train + test:
        w["_way"] = on_way(w)
    log("on the way computed")
    res = {"note": "Card 078, tools/card078/route.py"}
    # data check: is some token on the way, in orders and non-orders, per world
    dc = {}
    for name, ws in (("tier2 train", [w for w in train if w["source"] == "tier2"]),
                     ("decoy test", [w for w in test if w["source"].startswith("decoy")])):
        for o in ("n first", "none"):
            sel = [w for w in ws if w["outcome"] == o]
            dc[f"{name}, {o}"] = [len(sel), round(float(np.mean([any(w["_way"]) for w in sel])), 3) if sel else None]
    res["data_check_some_token_on_the_way"] = dc
    log(f"data check: {dc}")
    grp = defaultdict(list)
    for w in train:
        grp[RL.gate_key(w, F, Vn) + (tuple(w["_way"]),)].append(w["outcome"] == "n first")
    res["gate_inputs_consistent"] = round(sum(max(sum(v), len(v) - sum(v)) for v in grp.values()) / len(train), 4)
    log(f"gate {res['gate_inputs_consistent']}")
    tr2 = [w for w in train if w["source"] == "tier2"]
    decoy = [w for w in test if w["source"].startswith("decoy")]
    preds = {}
    for arm in ("route", "route only", "relational"):
        m, b, y, p0 = fit(tr2, F, arm)
        preds[arm] = R.run_recall(R.OwnRouter(m, F, tr2, b, y, p0), decoy)
        log(f"transfer {arm}: {json.dumps(R.score(decoy, preds[arm]))}")
    preds["always none"] = [0.0] * len(decoy)
    res["transfer"] = R.report(decoy, preds, {})
    preds = {}
    for arm in ("route", "relational"):
        m, b, y, p0 = fit(train, F, arm)
        preds[arm] = R.run_recall(R.OwnRouter(m, F, train, b, y, p0), test)
    preds["always none"] = [0.0] * len(test)
    res["within"] = R.report(test, preds, {})
    for arm in ("route", "relational"):
        for s, v in res["within"][arm].items():
            log(f"within {arm} {s}: {json.dumps(v)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
