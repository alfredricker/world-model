"""Card 102.2's tokens for breaking world 2: card 098.3's encoder recipe (as card 102.1's world 1) trained on the
world's own training play; identity codes are the nearest prototype. Crafter's tokens are card 102.1's.

  bin/prun python tools/card102.2/tokens.py      → runs/102.2/world2_{vec,code}_{train,test}.npy, world2_codes.npz
  bin/prun python tools/card102.2/tokens.py world3      → runs/102.3/world3_* (card 102.3)
  bin/prun python tools/card102.2/tokens.py world4      → runs/102.5/world4_* (card 102.5)
  bin/prun python tools/card102.2/tokens.py world5      → runs/102.6/world5_* (card 102.6)
  bin/prun python tools/card102.2/tokens.py world6      → runs/109/world6_* (card 109)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card102.1"))
import tokens as TK                                    # noqa: E402  (sets card 098.3's recipe)

EN, OUT = TK.EN, ROOT / "runs" / "102.2"


def main(world="world2", out=OUT, seed=1031, still_only=False):
    z = np.load(ROOT / "runs" / "card102" / f"{world}_train.npz")
    T = EN.tiles(z["obs"])
    ep, done = z["ep"], z["done"]
    P = np.flatnonzero((ep[:-1] == ep[1:]) & ~done[:-1])
    if still_only:                                      # a view that follows the agent is still only on its noops
        P = P[z["act"][P] == 0]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
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
            print(world, "encoder step", step, round(float(loss), 4), flush=True)
    enc.eval()
    torch.save(enc.state_dict(), out / f"{world}_encoder.pt")
    radius = EN.boundary(enc, Tt, P, rng)
    E = EN.embed(enc, T[::10]).reshape(-1, 32)
    _, protos, far = EN.leader(E, radius)
    np.savez(out / f"{world}_codes.npz", protos=protos, radius=radius)
    print(world, "radius", round(radius, 4), "codes", len(protos), "beyond", round(far, 4), flush=True)
    for split in ("train", "test"):
        zs = np.load(ROOT / "runs" / "card102" / f"{world}_{split}.npz")
        E = EN.embed(enc, EN.tiles(zs["obs"]))
        np.save(out / f"{world}_vec_{split}.npy", E.astype(np.float16))
        c = TK.nearest(E.reshape(-1, 32), protos).reshape(len(E), 81).astype(np.int16)
        np.save(out / f"{world}_code_{split}.npy", c)
        sem = zs["sem"].reshape(len(E), 63)
        c = c[:, :63]
        pur = np.mean([np.bincount(sem[c == k]).max() / (c == k).sum() for k in np.unique(c)])
        print(world, split, "codes used", len(np.unique(c)), "mean purity per code", round(float(pur), 4), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["world6"]:                     # card 109's breaking world 6 (searched caves)
        (ROOT / "runs" / "109").mkdir(exist_ok=True)
        main("world6", ROOT / "runs" / "109", 1091, still_only=True)
    elif sys.argv[1:2] == ["world3"]:                   # card 102.3's breaking world 3
        main("world3", ROOT / "runs" / "102.3", 1041, still_only=True)
    elif sys.argv[1:2] == ["world5"]:                   # card 102.6's breaking world 5 (clicks)
        (ROOT / "runs" / "102.6").mkdir(exist_ok=True)
        main("world5", ROOT / "runs" / "102.6", 1061)
    elif sys.argv[1:2] == ["world4"]:                   # card 102.5's breaking world 4b (no body)
        (ROOT / "runs" / "102.5").mkdir(exist_ok=True)
        main("world4", ROOT / "runs" / "102.5", 1051)
    else:
        main()
