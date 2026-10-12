"""Card 102.2, criterion 2: effects on whole objects, with tokens of one tile, fate groups or the true objects.

Arm A of card 102.1's kept form (two pair layers, softmax attention over every tile, raw positions) on world 2's
63 tiles. A token is a group of tiles: its members' states after the pair layers are averaged, and the group's fate
(still, one of four shifts, changed in place) is predicted once for all its members. What a tile changed in place
becomes is predicted per member (its next vector, snapped to the nearest known code). Forms differ only in the groups:
  tiles  every tile its own group (version 27's tokens);
  fate   card 102.2's groups from common fate (layers.py), from the evidence before each state;
  truth  the evaluator's objects (the upper bound).
Labels are the agent's own: each tile's fate from layers.py. A first-sight head gives every two tiles a score of
belonging to one object (a dot product of their states); it is trained on hindsight (the groups at the end of each
episode, for pairs with a tile that ever moved or changed) and read only on objects that have not yet moved.

Revision 1 (--within): a third pair layer of the same form runs within each group (each member reads its co-members'
states and where they stand), and the group's state is pooled by attention (one learned score per member, softmax
within the group) in place of the mean.

  bin/prun python tools/card102.2/predict.py tiles|fate|truth [--steps N] [--alpha A] [--beta B] [--within] [--tag T]
      → runs/102.2/predict_<form><tag>.json   (--alpha: prioritised replay's exponent, 0.6; 0 draws uniformly)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card102.1"))
sys.path.insert(0, str(ROOT / "tools" / "card102.2"))
import arms as AR                                      # noqa: E402
import layers as LY                                    # noqa: E402

RUNS, dev = ROOT / "runs" / "102.2", "cuda"
L = 63
WHERE = AR.WHERE[:L]


def data(split, form):
    z = np.load(ROOT / "runs" / "card102" / f"world2_{split}.npz")
    g = np.load(RUNS / f"groups_world2_{split}.npz")
    V = np.load(RUNS / f"world2_vec_{split}.npy")[:, :L]
    code = np.load(RUNS / f"world2_code_{split}.npy")[:, :L].astype(np.int64)
    seg = z["ep"].astype(np.int64)
    t = np.flatnonzero(np.r_[seg[1:] == seg[:-1], False])
    obj = z["obj"].reshape(len(seg), L).astype(np.int64)
    if form == "tiles":
        G = np.tile(np.arange(L), (len(seg), 1))
    elif form == "fate":
        G = g["group"].reshape(len(seg), L)
    else:                                               # the evaluator's objects; ground and walls one tile each
        G = np.where(obj > 0, 1000 + obj, np.arange(L)[None])
    # hindsight for the first-sight head: each position's group at the end of its episode, and whether it ever moved
    ident, fate0 = g["ident"].reshape(len(seg), L), g["fate0"].reshape(len(seg), L)
    FG, PART = np.zeros_like(ident), np.zeros(ident.shape, bool)
    grp = g["group"].reshape(len(seg), L)
    for s in np.unique(seg):
        ts = np.flatnonzero(seg == s)
        o = np.argsort(ident[ts[-1]])
        keys, vals = ident[ts[-1]][o], grp[ts[-1]][o]
        I = ident[ts]
        j = np.searchsorted(keys, I).clip(0, L - 1)
        FG[ts] = np.where(keys[j] == I, vals[j], -1 - np.arange(L)[None])
        PART[ts] = np.isin(I, I[(fate0[ts] >= 1) & (fate0[ts] <= 5)])
    lab = fate0.copy()
    return {"V": V[t], "Va": V[t + 1], "code": code[t], "codea": code[t + 1], "act": z["act"][t].astype(np.int64),
            "G": G[t], "lab": lab[t], "FG": FG[t], "PART": PART[t], "obj": obj[t], "obja": obj[t + 1],
            "sem": z["sem"].reshape(len(seg), L)[t], "agent": np.argmax(z["sem"].reshape(len(seg), L)[t] == 2, 1),
            "seg": seg[t], "t": t}


class Net(nn.Module):
    def __init__(self, dv=32, na=17, d=64):
        super().__init__()
        self.what = nn.Sequential(nn.Linear(dv, d), nn.ReLU(), nn.Linear(d, d))
        self.where = nn.Linear(2, d, bias=False)
        self.layers = nn.ModuleList([AR.PairLayer(d, na) for _ in range(2)])
        self.fate = nn.Linear(d, 6)
        self.after = nn.Linear(2 * d, dv)
        self.same = nn.Linear(d, 32)
        if WITHIN:                                      # revision 1: a pair layer within each group, attention pooling
            self.inner = AR.PairLayer(d, na)
            self.pool = nn.Linear(d, 1)

    def forward(self, X, a, G):
        n = len(X)
        W = WHERE.to(X.device)[None].expand(n, -1, -1)
        h = self.what(X) + self.where(W)
        for layer in self.layers:
            h, _, _ = layer(h, W, a, None)
        u = self.same(h)
        if WITHIN:
            Mb = G[:, :, None] == G[:, None]            # same group (each tile with itself)
            h, _, _ = self.inner(h, W, a, Mb)            # each member reads its co-members and where they stand
            att = torch.softmax(self.pool(h)[:, None, :, 0].expand(-1, L, -1).masked_fill(~Mb, -1e9), -1)
            hg = att @ h                                  # the group's state, pooled by attention within the group
        else:
            M = (G[:, :, None] == G[:, None]).float()
            hg = (M @ h) / M.sum(2, keepdim=True)         # each member reads its group's mean state
        return self.fate(hg), X + self.after(torch.cat([h, hg], -1)), (u @ u.transpose(1, 2)) / 32 ** 0.5


def batch(d, i):
    T = lambda k, dt=None: torch.as_tensor(d[k][i], device=dev) if dt is None else torch.as_tensor(d[k][i], device=dev).to(dt)
    return T("V", torch.float32), T("act"), T("G")


def train(d, steps, seed=1021, bs=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0 = time.monotonic()
    # prioritised replay (Schaul et al. 2016): steps drawn in proportion to their last loss to the power ALPHA,
    # weighted by (N P(i))^-beta, beta rising from 0.4 to 1, so the expected gradient is that of uniform draws
    N = len(d["act"])
    pri = np.ones(N)
    for step in range(steps):
        P = pri ** ALPHA
        P /= P.sum()
        i = np.sort(rng.choice(N, bs, p=P))
        beta = BETA0 + (1 - BETA0) * step / max(1, steps - 1) if BETA0 > 0 else 0.0
        w = torch.as_tensor((N * P[i]) ** -beta, device=dev, dtype=torch.float32)
        w = w / w.max()
        fl, after, same = net(*batch(d, i))
        lab = torch.as_tensor(d["lab"][i], device=dev)
        ok = (lab >= 0) & (lab <= 5)
        lf_i = (fn.cross_entropy(fl.reshape(-1, 6), lab.clamp(0, 5).reshape(-1), reduction="none").reshape(lab.shape)
                * ok).sum(1)
        tgt = torch.as_tensor(d["Va"][i], device=dev).float()
        la_i = ((after - tgt).abs().sum(-1) * (lab == 5)).sum(1)
        pri[i] = (lf_i + 0.1 * la_i).detach().cpu().numpy() + 1e-3
        lf, la = (w * lf_i).mean(), (w * la_i).mean()
        FG, PART = torch.as_tensor(d["FG"][i], device=dev), torch.as_tensor(d["PART"][i], device=dev)
        y = (FG[:, :, None] == FG[:, None]).float()
        m = (PART[:, :, None] | PART[:, None]) & ~torch.eye(L, dtype=torch.bool, device=dev)[None]
        ls = (fn.binary_cross_entropy_with_logits(same, y, reduction="none") * m).sum() / m.sum().clamp(min=1)
        loss = lf + 0.1 * la + ls
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print({"step": step + 1, "fate": round(lf.item(), 4), "after": round(la.item(), 4),
                   "same": round(ls.item(), 4), "s": round(time.monotonic() - t0, 1)}, flush=True)
    return net


@torch.no_grad()
def run(net, d, bs=512):
    F, A, S = [], [], []
    for s in range(0, len(d["act"]), bs):
        i = np.arange(s, min(s + bs, len(d["act"])))
        fl, after, same = net(*batch(d, i))
        F.append(fl.argmax(-1).cpu().numpy())
        A.append(after.cpu().numpy())
        S.append(torch.sigmoid(same).cpu().numpy() > 0.5)
    return np.concatenate(F), np.concatenate(A), np.concatenate(S)


def score(d, F, A, protos):
    """Steps where the agent walks into a shape or a door tile: is every tile of that object right?"""
    TF = d["truth fates"]
    res = {}
    started, cur = {}, None
    snap = lambda v: int(np.argmin(((v[None] - protos) ** 2).sum(-1)))
    for k in range(len(d["act"])):
        if d["seg"][k] != cur:
            cur, started = d["seg"][k], {}
        a = int(d["act"][k])
        ag = int(d["agent"][k])
        if a in LY_MOVES:
            dx, dy = LY_MOVES[a]
            tx, ty = ag % 9 + dx, ag // 9 + dy
            o = d["obj"][k][ty * 9 + tx] if 0 <= tx < 9 and 0 <= ty < 7 else 0
            if 0 < o < 99:
                m = d["obj"][k] == o
                tf = TF[k][m]
                right = (F[k][m] == tf).all() and all(snap(A[k][j]) == d["codea"][k][j]
                                                      for j in np.flatnonzero(m) if TF[k][j] == 5)
                kind = "door" if o == 1 else ("push" if (tf >= 1).any() else "blocked push")
                if kind != "door":
                    kind += ", one colour" if len(np.unique(d["sem"][k][m])) == 1 else ", several colours"
                when = "object moved or changed before" if started.get(o) else "first time"
                for key in (when, f"{when}: {kind}"):
                    res.setdefault(key, [0, 0])
                    res[key][0] += bool(right)
                    res[key][1] += 1
        for o in np.unique(d["obj"][k][TF[k] >= 1]):
            started[o] = True
    return {k: {"every tile right": round(v[0] / v[1], 4), "steps": v[1]} for k, v in sorted(res.items())}


def first_sight(d, S, ground_codes, every=5):
    """Objects of two or more tiles not yet moved: tiles exactly one group by the first-sight head alone
    (components of pairs scored above one half, among tiles that are not ground)."""
    TF = d["truth fates"]
    ok = n = 0
    started, cur = {}, None
    for k in range(len(d["act"])):
        if d["seg"][k] != cur:
            cur, started = d["seg"][k], {}
        if k % every == 0:
            ng = ~np.isin(d["code"][k], ground_codes)
            P = S[k] & ng[:, None] & ng[None]
            comp = np.arange(L)
            for i, j in np.argwhere(np.triu(P, 1)):
                ri, rj = i, j
                while comp[ri] != ri:
                    ri = comp[ri]
                while comp[rj] != rj:
                    rj = comp[rj]
                comp[max(ri, rj)] = min(ri, rj)
            for q in range(L):
                r = q
                while comp[r] != r:
                    r = comp[r]
                comp[q] = r
            for o in np.unique(d["obj"][k]):
                if o == 0 or o == 99 or started.get(o) or (d["obj"][k] == o).sum() < 2:
                    continue
                m = d["obj"][k] == o
                g = comp[m]
                ok += bool((g == g[0]).all() and (comp == g[0]).sum() == m.sum())
                n += 1
        for o in np.unique(d["obj"][k][TF[k] >= 1]):
            started[o] = True
    return {"objects not yet moved, exactly one group": round(ok / max(n, 1), 4), "of": n}


LY_MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
WITHIN = "--within" in sys.argv                        # revision 1
ALPHA = float(AR.arg("--alpha", "0.6"))                 # prioritised replay's exponent; 0 is uniform draws
BETA0 = float(AR.arg("--beta", "0.4"))                  # importance weights' exponent at the start (rising to 1); 0: none


def main():
    form = sys.argv[1]
    steps = int(AR.arg("--steps", "8000"))
    t0 = time.monotonic()
    tr, te = data("train", form), data("test", form)
    zt = np.load(ROOT / "runs" / "card102" / "world2_test.npz")
    seg = zt["ep"].astype(np.int64)
    TFall = LY.true_fates({"obj": zt["obj"], "sem": zt["sem"], "seg": seg}).reshape(len(seg), L)
    te["truth fates"] = TFall[te["t"]]
    print("data", len(tr["act"]), len(te["act"]), f"({time.monotonic() - t0:.0f} s)", flush=True)
    net = train(tr, steps)
    net.eval()
    F, A, S = run(net, te)
    protos = np.load(RUNS / "world2_codes.npz")["protos"]
    rep = {"form": form, "steps": steps, "within-group layer (revision 1)": WITHIN, "replay": f"prioritised, alpha {ALPHA}, beta from {BETA0}", "criterion 2": score(te, F, A, protos)}
    lab = te["lab"]
    okl = (lab >= 0) & (lab <= 5)
    rep["every tile's fate right (own labels), all test steps"] = round(float((F == lab)[okl].mean()), 5)
    ground = json.loads((RUNS / "layers_world2.json").read_text())["ground codes"]
    rep["first sight"] = first_sight(te, S, ground)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (RUNS / f"predict_{form}{AR.arg('--tag', '')}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
