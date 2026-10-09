"""Card 091: recall's stored keys per world, in the form the agent queries them, for the key-form router.

Per action, every stored key: the front tile's vector, the held tile's vector, the believed view's tiles (version 18's
context set, at most 6) as vectors, and the key's outcome counts by category (bit 0 the front tile changed, bit 1 the
hand changed; forward: moved, blocked, ended); plus the front and held code ids, for hiding a combination. Vectors are
the agent's own (S.arr, card 070's encoder).

  <version 18's flags> bin/prun python tools/card091/keys.py tier2        → runs/091/keys_tier2.npz
  (decoy: the decoy world's memory, card 070, under tier 1)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORLD = sys.argv[1]
TIER = int(WORLD[-1]) if WORLD.startswith("tier") else 1
sys.argv = ["run.py", "tiers", "--tier", str(TIER), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

T = RUN.T
VP = T.VP
if WORLD == "decoy":
    T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
VMAX = 6


def dump(W):
    SP = sys.modules["slot_planner"]
    CR = sys.modules["code_recall"]
    A = W.S.arr.astype(np.float32)
    out = {}
    kd = W.kinds[VP.FWD]
    h = np.array([int(k[0]) for k in kd.keys])
    out["2_F"] = A[h]
    out["2_C"] = np.asarray(kd.counts, np.float64)[:, :3]
    out["2_fid"] = np.array([CR.code_id(W.S, x) for x in h])
    for a in (VP.PICK, VP.DROP, VP.TOG):
        kd = W.kinds[a]
        n = len(kd.keys)
        L = kd.lc.shape[1]
        M = np.zeros((L, kd.ncat))
        M[np.arange(L), np.asarray(kd.lcat)[:L]] = 1.0
        V = np.zeros((n, VMAX, A.shape[1]), np.float32)
        vm = np.zeros((n, VMAX), bool)
        for i, k in enumerate(kd.keys):
            s = SP.SETS[int(k[2])][:VMAX]
            V[i, :len(s)] = A[s]
            vm[i, :len(s)] = True
        out[f"{a}_F"] = A[[int(k[0]) for k in kd.keys]]
        out[f"{a}_H"] = A[[int(k[1]) for k in kd.keys]]
        out[f"{a}_V"], out[f"{a}_vm"] = V, vm
        out[f"{a}_C"] = kd.lc[:n] @ M
        out[f"{a}_fid"], out[f"{a}_hid"] = np.asarray(kd.fid[:n]), np.asarray(kd.hid[:n])
    return out


if __name__ == "__main__":
    W, info = T.setup(TIER, lambda m: None)
    out = dump(W)
    p = ROOT / "runs" / "091" / f"keys_{WORLD}.npz"
    np.savez_compressed(p, **out)
    print(WORLD, {k: v.shape for k, v in out.items() if k.endswith("_C")}, flush=True)
