"""Card 069's feasibility gate: can one weighting over the 32 numbers of card 054's tile vectors tell a key and a
locked door of the same colour from a pair of different colours, for colours left out of the fit and for hues the
encoder never saw in training (card 052's held-out pink, brown and teal)?

Feature of a pair: |z_door - z_key| (32 numbers). A relation is a non-negative weighting w; the pair is "the same"
when w . |z_door - z_key| < t. Fitted by logistic regression with w >= 0 on the six MiniGrid colours with one colour
left out at a time; scored on the left-out colour's 11 pairs, and on the three unseen hues' 9 pairs (3 same, 6 not)
with a weighting fitted on all six. The per-part form (card 049's rel:p, one quarter of the vector) is scored alike.

  bin/prun python tools/card069/gate.py                                              card 054's encoder
  bin/prun python tools/card069/gate.py --encoder runs/069/encoder.pt --name gate_trained   card 069's relation too
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card066"))
sys.argv += ["--tier", "1"]
import tiers as T                                      # noqa: E402  (the catalogue, card 054's encoder loader)

from minigrid.core import constants as K               # noqa: E402
from minigrid.core.world_object import Door, Key       # noqa: E402

SIX = T.HUES
UNSEEN = ("pink", "brown", "teal")


def draw(obj):
    """As the agent's catalogue draws a tile (card 053's planner_check.catalogue: no grid lines)."""
    img = np.zeros((24, 24, 3), np.uint8)
    obj.render(img)
    return T.PV.PC.downsample(img, 3)


def tiles(colours):
    K.COLOR_NAMES.extend([c for c in T.EXTRA_HUES if c not in K.COLOR_NAMES])
    out = []
    for c in colours:
        out.append(draw(Door(c, is_open=False, is_locked=True)))
        out.append(draw(Key(c)))
    del K.COLOR_NAMES[len(SIX):]
    return np.stack(out)


def fit(X, y, steps=3000, lr=0.05, mask=None):
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    u = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    m = torch.ones(X.shape[1], dtype=torch.float64) if mask is None else torch.tensor(mask, dtype=torch.float64)
    opt = torch.optim.Adam([u, b], lr=lr)
    wpos = (yt.numel() - yt.sum()) / yt.sum()                 # same-colour pairs are 1 in 6: balance the classes
    for _ in range(steps):
        w = torch.nn.functional.softplus(u) * m
        s = b - Xt @ w                                        # "the same" when the weighted difference is small
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s, yt, pos_weight=wpos) + 1e-4 * w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = (torch.nn.functional.softplus(u) * m).detach().numpy()
    return w, float(b.detach())


def pairs(Z, colours):
    """Z: (2 * len(colours), 32), door then key per colour. Every (door c, key c') pair."""
    X, y, who = [], [], []
    for i, c in enumerate(colours):
        for j, c2 in enumerate(colours):
            X.append(np.abs(Z[2 * i] - Z[2 * j + 1]))
            y.append(float(c == c2))
            who.append((c, c2))
    return np.array(X), np.array(y), who


def learned(Z6, Z9, P):
    """Card 069's relation, ||P z_door - P z_key||_1 with P as trained: only the threshold is fitted (the midpoint
    between the largest same-colour and the smallest different-colour relation among the fitted pairs)."""
    def rel(Z, colours):
        X, y, who = pairs(Z, colours)
        return np.abs((Z[0::2][:, None] - Z[1::2][None]) @ P.T).sum(-1).reshape(-1), y, who
    r6, y6, who6 = rel(Z6, SIX)
    right = total = 0
    for c in SIX:
        tr = np.array([c not in p for p in who6])
        t = 0.5 * (r6[tr & (y6 > 0)].max() + r6[tr & (y6 == 0)].min())
        right += int(((r6[~tr] < t) == (y6[~tr] > 0)).sum())
        total += int((~tr).sum())
    t = 0.5 * (r6[y6 > 0].max() + r6[y6 == 0].min())
    r9, y9, who9 = rel(Z9, SIX + UNSEEN)
    sel = np.array([p[0] in UNSEEN and p[1] in UNSEEN for p in who9])
    allu = np.array([p[0] in UNSEEN or p[1] in UNSEEN for p in who9])
    return {"left_out_colour_pairs_right": f"{right}/{total}",
            "unseen_hue_pairs_right": f"{int(((r9[sel] < t) == (y9[sel] > 0)).sum())}/{int(sel.sum())}",
            "pairs_with_an_unseen_hue_right": f"{int(((r9[allu] < t) == (y9[allu] > 0)).sum())}/{int(allu.sum())}",
            "same_colour_relation_max_six": round(float(r6[y6 > 0].max()), 3),
            "different_colour_relation_min_six": round(float(r6[y6 == 0].min()), 3)}


def main():
    args = sys.argv[1:]
    path = args[args.index("--encoder") + 1] if "--encoder" in args else str(ROOT / "runs" / "054" / "b_m0.5_399.pt")
    name = args[args.index("--name") + 1] if "--name" in args else "gate"
    enc = T.PV.RP.load(path, "transition", False)
    Z6 = T.PV.RP.vectors(enc, list(tiles(SIX))).astype(np.float64)
    Z9 = T.PV.RP.vectors(enc, list(tiles(SIX + UNSEEN))).astype(np.float64)
    X, y, who = pairs(Z6, SIX)
    rep = {"encoder": path}
    ck = torch.load(path)
    if ck.get("P") is not None:
        rep["learned relation (card 069)"] = learned(Z6, Z9, ck["P"].double().numpy())
    for form, masks in (("whole vector", [None]), ("one quarter (card 049)", [np.repeat(np.eye(4)[p], 8) for p in range(4)])):
        best = None
        for mask in masks:
            right, total = 0, 0
            for c in SIX:
                tr = np.array([c not in p for p in who])
                w, b = fit(X[tr], y[tr], mask=mask)
                pred = (b - X[~tr] @ w) > 0
                right += int((pred == (y[~tr] > 0)).sum())
                total += int((~tr).sum())
            w, b = fit(X, y, mask=mask)
            Xu, yu, whou = pairs(Z9, SIX + UNSEEN)
            sel = np.array([p[0] in UNSEEN and p[1] in UNSEEN for p in whou])
            pu = (b - Xu[sel] @ w) > 0
            r = {"left_out_colour_pairs_right": f"{right}/{total}",
                 "unseen_hue_pairs_right": f"{int((pu == (yu[sel] > 0)).sum())}/{int(sel.sum())}",
                 "weights_carrying_half_the_mass": int((np.cumsum(np.sort(w)[::-1]) < w.sum() / 2).sum() + 1)}
            if best is None or right > int(best["left_out_colour_pairs_right"].split("/")[0]):
                best = r
        rep[form] = best
    print(json.dumps(rep, indent=1))
    out = ROOT / "runs" / "069"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
