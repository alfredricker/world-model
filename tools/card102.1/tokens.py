"""Card 102.1's tokens. Every frame's 81 tiles (the 9 x 7 world above the two carried rows) become vectors, by
Crafter's adapter (7 x 7-pixel tiles); identity codes are the nearest prototype.

- Crafter's curriculum (runs/102.1/curr_w*.npz): card 098.3's encoder and prototypes, the best kept, as they are.
- Breaking world 1 (runs/card102/world1_*.npz): card 098.3's recipe (still pairs judged between places drawn
  identically; a changed place as a pair loss; the radius from training's own pairs) trained on the world's own
  training play. The world's view never shifts, so every consecutive pair is still by the view (card 098.3's rule).

  bin/prun python tools/card102.1/tokens.py crafter | world1
      → runs/102.1/{curr,world1}_vec_*.npy (float16, n x 81 x 32), *_code_*.npy, world1_codes.npz
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.update(WM98_NEAR="0.0005", WM98_EXACT="1", WM98_CHANGED="1", WM98_PAIR="1", WM98_RADIUS="exact")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card098"))
import encoder as EN                                   # noqa: E402

OUT = ROOT / "runs" / "102.1"
ENC098_3 = ROOT / "runs" / "098.3" / "encoder_joint_blocked.pt"
CODES098_3 = ROOT / "runs" / "098.3" / "codes_joint_blocked.npz"


def nearest(E, protos):
    P = torch.as_tensor(protos, device=EN.dev, dtype=torch.float32)
    out = []
    for s in range(0, len(E), 200000):
        out.append(torch.cdist(torch.as_tensor(E[s:s + 200000], device=EN.dev), P).argmin(1).cpu())
    return torch.cat(out).numpy()


def crafter():
    enc = EN.Enc().to(EN.dev)
    enc.load_state_dict(torch.load(ENC098_3))
    enc.eval()
    protos = np.load(CODES098_3)["protos"]
    for w in range(10):
        z = np.load(OUT / f"curr_w{w}.npz")
        E = EN.embed(enc, EN.tiles(z["obs"]))
        np.save(OUT / f"curr_vec_w{w}.npy", E.astype(np.float16))
        np.save(OUT / f"curr_code_w{w}.npy", nearest(E.reshape(-1, 32), protos).reshape(len(E), 81).astype(np.int16))
        print("crafter worker", w, E.shape, flush=True)


def world1():
    z = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    T = EN.tiles(z["obs"])
    ep, done = z["ep"], z["done"]
    P = np.flatnonzero((ep[:-1] == ep[1:]) & ~done[:-1])
    rng = np.random.default_rng(1021)
    torch.manual_seed(1021)
    enc = EN.Enc().to(EN.dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Tt = torch.as_tensor(T)
    for step in range(EN.STEPS):
        b = rng.choice(P, EN.BATCH, replace=False)
        loss = EN.batch_loss(enc, Tt[b].to(EN.dev).float() / 255.0, Tt[b + 1].to(EN.dev).float() / 255.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print("world 1 encoder step", step, round(float(loss), 4), flush=True)
    enc.eval()
    torch.save(enc.state_dict(), OUT / "world1_encoder.pt")
    radius = EN.boundary(enc, Tt, P, rng)
    E = EN.embed(enc, T[::10]).reshape(-1, 32)
    _, protos, far = EN.leader(E, radius)
    np.savez(OUT / "world1_codes.npz", protos=protos, radius=radius)
    print("world 1: radius", round(radius, 4), "codes", len(protos), "beyond", round(far, 4), flush=True)
    for split in ("train", "test"):
        zs = np.load(ROOT / "runs" / "card102" / f"world1_{split}.npz")
        E = EN.embed(enc, EN.tiles(zs["obs"]))
        np.save(OUT / f"world1_vec_{split}.npy", E.astype(np.float16))
        np.save(OUT / f"world1_code_{split}.npy", nearest(E.reshape(-1, 32), protos).reshape(len(E), 81).astype(np.int16))
        # each code's class (evaluator only): purity of codes over the world's kinds
        sem = zs["sem"].reshape(len(E), 63)
        c = np.load(OUT / f"world1_code_{split}.npy")[:, :63]
        pur = np.mean([np.bincount(sem[c == k]).max() / (c == k).sum() for k in np.unique(c)])
        print("world 1", split, "codes used", len(np.unique(c)), "mean purity per code", round(float(pur), 4), flush=True)


if __name__ == "__main__":
    {"crafter": crafter, "world1": world1}[sys.argv[1]]()
