"""Card 102.7: tokens not cut on the world's grid. The adapter cuts Crafter's frames into square patches at a stride
equal to their size, from an offset, and resamples each to 7 x 7 pixels for the encoder:
  A  7 px from (0, 0)  Crafter's own tiles (as now): 9 x 7 world patches
  B  7 px from (3, 3)  the right size, the wrong phase: 8 x 6
  C  8 px from (0, 0)  the wrong size (the view moves 7 px, 7/8 of a patch): 8 x 6
For each: card 098.3's encoder recipe, simplified alike for all (positives: places drawn identically within the frame,
and the same place at the next frame when the agent did not move; negatives: the rest), codes by leader clustering at
the radius that best separates those pairs, then card 102.5's placing and contingency on the patch tokens.

  bin/prun python tools/card102.7/patches.py A|B|C      → runs/102.7/patches_<v>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
for p in ("card098", "card102.5"):
    sys.path.insert(0, str(ROOT / "tools" / p))
import encoder as EN                                   # noqa: E402
import self as SF                                      # noqa: E402

OUT, dev = ROOT / "runs" / "102.7", "cuda"
FORMS = {"A": (7, 0, 9, 7), "B": (7, 3, 8, 6), "C": (8, 0, 8, 6)}     # size, offset, columns, rows
NEAR, TAU = 0.0005, 0.1
TRAIN_W, TEST_W = (0, 1, 2), (8,)


def cut(obs, size, off, cols, rows):
    """(n, 64, 64, 3) → (n, rows * cols, 3, 7, 7) uint8."""
    out = np.zeros((len(obs), rows * cols, 3, 7, 7), np.uint8)
    for s in range(0, len(obs), 4000):
        x = torch.as_tensor(obs[s:s + 4000, off:off + rows * size, off:off + cols * size]).permute(0, 3, 1, 2).float()
        p = x.unfold(2, size, size).unfold(3, size, size)                          # n, 3, rows, cols, size, size
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, size, size)
        if size != 7:
            p = fn.interpolate(p, size=(7, 7), mode="area")
        out[s:s + 4000] = p.round().clamp(0, 255).byte().reshape(-1, rows * cols, 3, 7, 7).numpy()
    return out


def load(ws, form):
    size, off, cols, rows = FORMS[form]
    zs = [np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz") for w in ws]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("act", "ep", "done", "pos", "sleep", "light")}
    d["ep"] = np.concatenate([z["ep"] + 1000000 * w for z, w in zip(zs, ws)])
    d["T"] = np.concatenate([cut(z["obs"], size, off, cols, rows) for z in zs])
    brk = np.r_[True, (d["ep"][1:] != d["ep"][:-1]) | d["sleep"][1:] | d["sleep"][:-1] | d["done"][:-1]]
    d["seg"] = np.cumsum(brk)
    d["seg"][d["sleep"]] = -1
    return d


def loss(enc, X0, X1):
    B, N = X0.shape[:2]
    e0 = enc(X0.reshape(-1, 3, 7, 7)).reshape(B, N, -1)
    e1 = enc(X1.reshape(-1, 3, 7, 7)).reshape(B, N, -1)
    d = ((X0[:, :, None] - X0[:, None]) ** 2).mean((-1, -2, -3))
    pos = d < NEAR
    s = torch.einsum("bid,bjd->bij", e0, e1) / TAU
    lp = s - torch.logsumexp(s, -1, keepdim=True)
    return -((lp * pos).sum(-1) / pos.sum(-1)).mean(), e0, e1, pos


def main():
    form = sys.argv[1]
    size, off, cols, rows = FORMS[form]
    t0 = time.monotonic()
    tr, te = load(TRAIN_W, form), load(TEST_W, form)
    N = rows * cols
    P = np.flatnonzero((tr["seg"][:-1] >= 0) & (tr["seg"][:-1] == tr["seg"][1:]) & ~np.isin(tr["act"][:-1], (1, 2, 3, 4)))
    rng = np.random.default_rng(1027)
    torch.manual_seed(1027)
    enc = EN.Enc().to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Tt = torch.as_tensor(tr["T"])
    for step in range(4000):
        b = rng.choice(P, 48, replace=False)
        l, _, _, _ = loss(enc, Tt[b].to(dev).float() / 255, Tt[b + 1].to(dev).float() / 255)
        opt.zero_grad()
        l.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():                                 # the radius: best balanced split of training's own pairs
        Dp, Dn = [], []
        for _ in range(10):
            b = rng.choice(P, 48, replace=False)
            _, e0, e1, pos = loss(enc, Tt[b].to(dev).float() / 255, Tt[b + 1].to(dev).float() / 255)
            dist = torch.cdist(e0, e1)
            Dp.append(dist[pos].cpu().numpy())
            Dn.append(dist[~pos].cpu().numpy())
        Dp, Dn = np.concatenate(Dp), np.concatenate(Dn)
        cand = np.quantile(np.concatenate([Dp, Dn]), np.linspace(0.01, 0.99, 99))
        radius = float(cand[np.argmax([(Dp < c).mean() + (Dn >= c).mean() for c in cand])])

    def embed(T):
        out = []
        with torch.no_grad():
            for s in range(0, len(T), 2000):
                X = torch.as_tensor(T[s:s + 2000], device=dev).float() / 255
                out.append(enc(X.reshape(-1, 3, 7, 7)).reshape(len(X), N, -1).cpu().numpy())
        return np.concatenate(out)
    Etr, Ete = embed(tr["T"]), embed(te["T"])
    _, protos, far = EN.leader(Etr[::20].reshape(-1, 32), radius)
    Pt = torch.as_tensor(protos, device=dev)

    def nearest(E):
        F = E.reshape(-1, 32)
        out = np.concatenate([torch.cdist(torch.as_tensor(F[s:s + 65536], device=dev), Pt).argmin(1).cpu().numpy()
                              for s in range(0, len(F), 65536)])
        return out.reshape(len(E), rows, cols)
    rep = {"form": form, "patch size, offset, columns, rows": [size, off, cols, rows], "radius": round(radius, 4),
           "codes": int(len(protos))}
    SF.R, SF.C, SF.L = rows, cols, N                       # card 102.5's placing on these tokens
    SF.VY, SF.VX = np.divmod(np.arange(N), cols)
    cx, cy = (31 - off) // size, (24 - off) // size        # the patch holding the agent's tile's centre (31, 24)
    dd = {}
    for name, d, E in (("train", tr, Etr), ("test", te, Ete)):
        dd[name] = {"seg": d["seg"], "act": d["act"], "pos": d["pos"].astype(np.int64), "V": E,
                    "code": nearest(E).astype(np.int64), "light": d["light"]}
    del tr["T"], te["T"]
    prior = np.ones((17, len(SF.SH)))
    b, still = 1.0, None
    for it in range(2):
        pr = prior.copy()
        org, _, bn = SF.place(dd["train"], pr, b, learn=True, still=still)
        b = max(bn, 1e-3)
        M = SF.motions(dd["train"], org)
        still = {c: (sum(1 for r in rows_ if r[1:] == (0, 0)) + 1) / (len(rows_) + 2) for c, rows_ in M.items()}
    _, sh, _ = SF.place(dd["test"], pr.copy(), b, learn=False, still=still)
    seg = dd["test"]["seg"]
    st = np.flatnonzero((seg[:-1] >= 0) & (seg[:-1] == seg[1:]))
    got = np.array([SF.SH[k] for k in sh[st]])
    true = dd["test"]["pos"][st + 1] - dd["test"]["pos"][st]
    right = (got == true).all(1)
    moved = (true != 0).any(1)
    day = dd["test"]["light"][st] >= 0.3
    rep["view's motion read right"] = round(float(right.mean()), 4)
    rep["when it moved, when it did not"] = [round(float(right[moved].mean()), 4), round(float(right[~moved].mean()), 4)]
    rep["by day, by night"] = [round(float(right[day].mean()), 4), round(float(right[~day].mean()), 4)]
    U = {c: SF.uncertainty(r) for c, r in M.items() if len(r) >= 50}
    code = dd["train"]["code"].reshape(len(seg := dd["train"]["seg"]), -1)
    centre = cy * cols + cx
    truth = {c for c in U if (code[:, centre] == c).sum() > 0.5 * (code == c).sum()}
    ctrl = {c for c, (u, _) in U.items() if u >= 0.5}
    rep["controlled codes: found, the evaluator's, both"] = [len(ctrl), len(truth), len(ctrl & truth)]
    rep["U: highest of the evaluator's, highest other"] = [round(max([U[c][0] for c in truth] or [0]), 3),
                                                          round(max([U[c][0] for c in U if c not in truth] or [0]), 3)]
    rep["seconds"] = round(time.monotonic() - t0, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"patches_{form}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
