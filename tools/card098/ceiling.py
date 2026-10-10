"""Card 098, feasibility gate: the most any tile encoder could reach. A classifier with the encoder's form (one 7 x 7
tile in, card 098's network and a linear head) trained on the evaluator's semantic labels (never available to the
agent), on frames of workers 0-7, judged on workers 8-9; accuracy by daylight and, at night, by ring (the tile's
distance from the player's tile, 0-4). Crafter adds noise only below daylight 0.5, growing toward the view's edges.

  bin/prun python tools/card098/ceiling.py      → runs/098/ceiling.json
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

import encoder as EN

OUT, dev = EN.OUT, EN.dev
RING = np.array([max(abs(c - 4), abs(r - 3)) for r in range(7) for c in range(9)])
log = EN.log


def bins(light):
    return np.digitize(light, [0.1, 0.3, 0.5, 0.7])     # 0: below 0.1 ... 4: above 0.7


if __name__ == "__main__":
    t0 = time.monotonic()
    z = np.load(OUT / "play.npz")
    T = EN.tiles(z["obs"])[:, :63]                      # world tiles only
    sem = z["sem"].reshape(len(T), 63)
    light, ep = z["light"], z["ep"]
    test = ep // 100000 >= 8
    classes = np.unique(sem[sem >= 0])
    lab = np.searchsorted(classes, np.maximum(sem, 0))
    ok = sem >= 0
    tr = np.argwhere(~test[:, None] & ok)               # (frame, tile)
    te = np.argwhere(test[:, None] & ok)
    torch.manual_seed(98)
    rng = np.random.default_rng(98)
    body = EN.Enc().to(dev)
    head = nn.Linear(32, len(classes)).to(dev)
    opt = torch.optim.Adam(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    Tt = torch.as_tensor(T)
    for step in range(6000):
        i = tr[rng.integers(len(tr), size=1024)]
        x = Tt[i[:, 0], i[:, 1]].to(dev).float() / 255.0
        y = torch.as_tensor(lab[i[:, 0], i[:, 1]], device=dev)
        loss = fn.cross_entropy(head(body(x) * 10), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            log(f"step {step} loss {float(loss.detach()):.4f} ({round(time.monotonic() - t0)} s)")
    body.eval()
    pred = np.zeros(len(te), np.int64)
    with torch.no_grad():
        for s in range(0, len(te), 50000):
            i = te[s:s + 50000]
            x = Tt[i[:, 0], i[:, 1]].to(dev).float() / 255.0
            pred[s:s + 50000] = head(body(x) * 10).argmax(1).cpu().numpy()
    right = pred == lab[te[:, 0], te[:, 1]]
    lb, rg = bins(light[te[:, 0]]), RING[te[:, 1]]
    names = ["below 0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "above 0.7"]
    rep = {"test tiles": int(len(te)), "overall": round(float(right.mean()), 4),
           "by daylight": {names[b]: [round(float(right[lb == b].mean()), 4), int((lb == b).sum())] for b in range(5)},
           "night (below 0.3) by ring": {int(r): round(float(right[(lb <= 1) & (rg == r)].mean()), 4) for r in range(5)},
           "night (below 0.3)": round(float(right[lb <= 1].mean()), 4),
           "day (above 0.7)": round(float(right[lb == 4].mean()), 4),
           "seconds": round(time.monotonic() - t0, 1)}
    (OUT / "ceiling.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps(rep))
