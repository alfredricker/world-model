"""Card 054 step A: on a frozen encoder, per part, the noise scale (the largest part-distance between an untouched
cell's two renders a step apart, over 1,000 pairs; tau = 1.25 x that) against the smallest part-distance between
visibly different tiles (probe identities whose clean renders differ in pixels beyond noise). Also the whole-tile
version, and which identity pairs fall within tau in every part (they would share a code tuple).

  bin/prun python tools/card054/gap.py runs/053/s1v_V0_399.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card053"))
import recall_probe as RP                              # noqa: E402

EF, GN, DR = RP.EF, RP.GN, RP.DR


def main():
    path = sys.argv[1]
    GN.TINT = 0.0
    EF.START_SHARE = 0.5
    EF.VIEWS = True                                    # untouched cells are recorded only with views on
    enc = RP.load(path, "transition", False)
    s = EF.Stream(399)
    pairs = []
    while len(pairs) < 1000:
        s.step()
        pairs += list(s.views)
    a = RP.vectors(enc, [p[0] for p in pairs[:1000]]).reshape(1000, 4, 8)
    b = RP.vectors(enc, [p[1] for p in pairs[:1000]]).reshape(1000, 4, 8)
    noise = np.linalg.norm(a - b, axis=-1)                  # (1000, 4)
    tau = 1.25 * noise.max(0)
    tiles, ids = DR.probe_set(399)
    z = RP.vectors(enc, list(tiles)).reshape(len(tiles), 4, 8)
    ident = sorted(set(ids), key=str)
    cent = np.stack([z[[i for i, q in enumerate(ids) if q == u]].mean(0) for u in ident])
    within = [np.linalg.norm(z[[i for i, q in enumerate(ids) if q == u]] - cent[j], axis=-1).max(0)
              for j, u in enumerate(ident)]
    d = np.linalg.norm(cent[:, None] - cent[None], axis=-1)     # (I, I, 4)
    iu = np.triu_indices(len(ident), 1)
    print(f"untouched pairs: part noise max {np.round(noise.max(0), 4)}, tau {np.round(tau, 4)}")
    print(f"probe copies: largest distance from their identity's mean, per part {np.round(np.max(within, 0), 4)}")
    for k in range(4):
        dk = d[:, :, k][iu]
        print(f"part {k}: identity pairs within tau {int((dk < tau[k]).sum())} of {len(dk)}; "
              f"smallest nonzero gap {np.round(np.sort(dk[dk > tau[k]])[:3], 4)}")
    same_all = np.all(d[iu] < tau, axis=-1)
    print(f"identity pairs within tau in every part (one code tuple): {int(same_all.sum())}")
    for i, j in zip(iu[0][same_all][:12], iu[1][same_all][:12]):
        print("   ", ident[i], ident[j])


if __name__ == "__main__":
    main()
