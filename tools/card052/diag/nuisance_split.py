"""Card 052: which nuisance splits a saved encoder's codes? Probe tiles (step 2a's probe set, seed 399) rendered
clean, with noise only, with tint only, and with both; per condition: identities split over several code tuples,
tuples shared by identities, and the mean distance within an identity against the nearest other identity.

  bin/prun python tools/card052/diag/nuisance_split.py runs/052/r1b_diag_A_399.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import effect as EF                                    # noqa: E402

DR, GN = EF.DR, EF.DR.GN


def main():
    ck = torch.load(sys.argv[1])
    DR.Encoder.CODEBOOK = "fixed"
    enc = EF.Encoder(399)
    enc.enc.load_state_dict(ck["enc"])
    with torch.no_grad():
        enc.books.copy_(ck["books"].to(enc.dev))
    tint0, noise0 = GN.TINT, GN.NOISE
    for name, t, n in (("clean", 0.0, 0.0), ("noise only", 0.0, noise0), ("tint only", tint0, 0.0), ("both", tint0, noise0)):
        GN.TINT, GN.NOISE = t, n
        tiles, ids = DR.probe_set(399)
        c, _ = enc.codes(tiles)
        tup = [tuple(r) for r in c.tolist()]
        ident = sorted(set(ids), key=str)
        split = sum(len({u for u, q in zip(tup, ids) if q == i}) > 1 for i in ident)
        shared = sum(len({q for u2, q in zip(tup, ids) if u2 == u}) > 1 for u in set(tup))
        with torch.no_grad():
            z = enc.pieces(enc.x(tiles)).reshape(len(tiles), -1).cpu().numpy()
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        cent = {i: z[[k for k, q in enumerate(ids) if q == i]] for i in ident}
        within = np.mean([np.linalg.norm(v - v.mean(0), axis=1).mean() for v in cent.values()])
        mu = np.stack([v.mean(0) for v in cent.values()])
        dd = np.linalg.norm(mu[:, None] - mu[None], axis=-1) + np.eye(len(mu)) * 9
        print(f"{name:10s}: identities split {split:2d} of {len(ident)}, tuples shared {shared:2d}, tuples {len(set(tup)):3d}; "
              f"spread within an identity {within:.3f}, nearest other identity {dd.min(1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
