"""Card 035: recall picks few parts.

Card 034's recall term trains card 031's encoder, with a group-sparsity penalty on recall's weights:
beta x the sum, over (world, action) and over the eight parts of an event's key (four of the tile in
front, four of the held tile), of the Euclidean norm of that part's eight weights. Each action's outcome
is then predicted from as few parts as possible, so the recall term pulls tiles together only there,
and the other parts stay free to keep tiles distinct (Lachapelle et al. 2022, mechanism sparsity).

Steps (drafting probe):
  bin/prun python tools/card035/sparse.py --probe --mu MU --beta B --seeds 100-102 --out runs/035_probe_MU_B.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card034"))
import teach as T  # noqa: E402  (card 034: memory with forward, priors, recall reports)

R, C, NEW = T.R, T.C, T.NEW
ROOT = Path(__file__).resolve().parents[2]
PARTS = [f"front {k}" for k in range(4)] + [f"held {k}" for k in range(4)]


def group_sparsity(beta):
    """beta x sum over groups and parts of the norm of the part's weights; lam: (G, 64)."""
    def pen(lam):
        return beta * lam.reshape(lam.shape[0], 8, 8).norm(dim=-1).sum()
    return pen


def part_shares(theta, names):
    """Per (world, action): each part's share of the summed part norms of recall's weights."""
    out = {}
    for th, (w, a) in zip(theta, names):
        lam = np.log1p(np.exp(np.asarray(th)))
        n = np.linalg.norm(lam.reshape(8, 8), axis=-1)
        sh = n / n.sum()
        out[f"{w} {T.ANAME[a]}"] = {"norm_sum": round(float(n.sum()), 3), "shares": [round(float(v), 3) for v in sh],
                                    "parts_above_10pc": [PARTS[i] for i in range(8) if sh[i] >= 0.1]}
    return out


def main():
    args = sys.argv[1:]
    flag = lambda f: f in args
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    T.configure()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    cache = pickle.loads(T.MEMORY.read_bytes())
    mem, tiles, pairs = cache["mem"], cache["tiles"], cache["pairs"]
    groups = T.use_groups(mem, dev)

    if flag("--probe"):
        mu, beta, M = float(get("--mu", "0.1")), float(get("--beta", "0")), int(get("--M", "8"))
        a, b = get("--seeds", "100-102").split("-")
        upd = int(get("--updates", "5000"))
        out = Path(get("--out", f"runs/035_probe_{mu}_{beta}.json"))
        res = {"note": "Card 035 drafting probe", "mu": mu, "beta": beta, "M": M, "seeds": []}
        for seed in range(int(a), int(b) + 1):
            ca, rep, z, th = R.train_encoder(tiles, pairs, groups, mu, mode="own", seed=seed, dev=dev, M=M,
                                             updates=upd, lam_penalty=group_sparsity(beta) if beta > 0 else None)
            o = {"seed": seed, "encoder": rep, "tiles": R.tiles_ok(ca, tiles),
                 "new_training_tiles": [f"{C.name_of_code(c)} (codebook {k})" for c in tiles
                                        for k in range(ca.shape[1]) if ca[c, k] == NEW],
                 "parts": part_shares(th, groups.names) if th else {},
                 "new_tiles": T.new_tiles_report(ca, tiles), "kinds_check": T.kinds_check(z, tiles)}
            if th:
                o["recall"] = T.recall_reports(z, th, mem)
            res["seeds"].append(o)
            out.write_text(json.dumps(res, indent=1, default=str) + "\n")
            log(f"mu {mu} beta {beta} seed {seed}: {rep}; {o['tiles']}; new {o['new_training_tiles']}")
        return


if __name__ == "__main__":
    main()
