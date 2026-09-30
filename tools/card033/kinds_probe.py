"""Probe after card 033 (2026-09-29): does card 031's encoder have any part of its output in which the
members of a kind (the three keys, the three closed doors, ...) agree with each other and differ from
every other familiar tile, and in which a key or door of a new colour falls in with its kind?

Kinds here are the evaluator's labels; this asks what the encoder offers, not what the agent learns.
Parts: each of the encoder's four pieces (8 numbers), all 32 numbers, the four codebooks' codes, and
the raw pixels as a baseline. Card 031's encoder is card 033's with no replay (mu = 0).

    bin/prun python tools/card033/kinds_probe.py --seeds 100-104 --out runs/033_kinds_probe_a.json

Second probe (option 1, "let the kinds teach the encoder", best case): --kind-w W adds W x a supervised
contrastive term (`2004_11362`, temperature 0.1) on piece 0 of the 20 familiar tiles, with the
evaluator's kinds as labels (each other tile its own kind). The yellow and purple tiles are never
trained on; the question is whether they fall in with their kind in piece 0.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recall as R  # noqa: E402

C, KR = R.C, R.KR
COLOURS = ("red", "green", "blue", "yellow", "purple")
NEWC = ("yellow", "purple")


def kind_of(name):
    w = name.split()
    return " ".join(w[:-1]) if w[-1] in COLOURS else name


def new_tiles():
    out = {}
    for col in NEWC:
        door = KR.code(("door", col, 2))
        out.update({f"key {col}": KR.code(("key", col)), f"closed door {col}": KR.code(("door", col, 0)),
                    f"open door {col}": door, f"agent in doorway {col}": door + C.UP + 1})
    return out


def check(D, fam, kinds, new):
    """D: distances between all tile codes in one part. For each kind with two or more familiar
    members: coherent (every member's nearest other familiar tile is a member), separated (the
    widest spread inside the kind is smaller than the nearest distance to a non-member), and for
    each new tile of that kind: nearest (its nearest familiar tile is a member) and with_kind (it is
    nearer to every member than to any non-member)."""
    out = {}
    for k, mem in kinds.items():
        non = [t for t in fam if t not in mem]
        coherent = all(min([t for t in fam if t != m], key=lambda t: D[m, t]) in mem for m in mem)
        spread = max(D[a, b] for a in mem for b in mem if a != b)
        gap = min(D[a, b] for a in mem for b in non)
        r = {"coherent": bool(coherent), "separated": bool(spread < gap),
             "spread": round(float(spread), 3), "gap": round(float(gap), 3)}
        for nm, c in new.items():
            if kind_of(nm) != k:
                continue
            near = min(fam, key=lambda t: D[c, t])
            r[nm] = {"nearest": C.name_of_code(near),
                     "with_kind": bool(max(D[c, m] for m in mem) < min(D[c, t] for t in non))}
        out[k] = r
    return out


def kind_term(tiles, kind_ids, w, dev, piece=0, tau=0.1):
    """w x the supervised contrastive loss on one piece: each tile with other members of its kind is
    pulled toward them against all other familiar tiles."""
    import torch
    idx = torch.as_tensor(tiles, device=dev)
    lab = torch.as_tensor(kind_ids, device=dev)
    same = (lab[:, None] == lab[None]) & ~torch.eye(len(tiles), dtype=torch.bool, device=dev)
    has = same.any(1)

    def f(z_all):
        z = z_all[idx, piece]
        sim = z @ z.T / tau
        sim = sim.masked_fill(torch.eye(len(tiles), dtype=torch.bool, device=dev), -1e9)
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        per = -(logp * same).sum(1) / same.sum(1).clamp(min=1)
        return w * per[has].mean()
    return f


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    a, b = get("--seeds", "100-109").split("-")
    out = Path(get("--out", "runs/033_kinds_probe.json"))
    kind_w = float(get("--kind-w", "0"))
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = pickle.loads(R.CACHE.read_bytes())
    tiles, pairs = [int(t) for t in cache["tiles"]], cache["pairs"]
    train_extra = get("--train-extra", "")      # diagnostic: a new colour's tiles joined to training, with their kinds
    new_all = new_tiles()
    if train_extra:
        tiles += [c for nm, c in new_all.items() if nm.endswith(train_extra)]
    groups = R.Groups(cache["mem"], dev)
    names = {c: C.name_of_code(c) for c in tiles}
    kinds = {}
    for c in tiles:
        kinds.setdefault(kind_of(names[c]), []).append(c)
    kinds = {k: v for k, v in kinds.items() if len(v) > 1}
    new = {nm: c for nm, c in new_all.items() if not (train_extra and nm.endswith(train_extra))}
    allc = tiles + list(new.values())
    dist = lambda X: np.linalg.norm(X[:, None] - X[None], axis=-1)

    px = R.TILES.reshape(len(R.TILES), -1) / 255.0
    res = {"note": __doc__.strip().splitlines()[0], "kind_w": kind_w, "train_extra": train_extra, "kinds": {k: [names[c] for c in v] for k, v in kinds.items()},
           "pixels": check(dist(px), tiles, kinds, new), "seeds": []}
    for seed in range(int(a), int(b) + 1):
        extra = None
        if kind_w > 0:
            kid = {}
            ids = [kid.setdefault(kind_of(names[c]), len(kid)) for c in tiles]
            extra = kind_term(tiles, ids, kind_w, dev)
        ca, rep, z, _ = R.train_encoder(tiles, pairs, groups, 0.0, seed=seed, dev=dev, extra=extra)
        parts = {f"piece {k}": z[:, 8 * k:8 * k + 8] for k in range(4)}
        parts["all 32"] = z
        s = {"seed": seed, "encoder": rep, "tiles": R.tiles_ok(ca, tiles),
             "new_tiles": [f"{names[c]} (codebook {k})" for c in tiles for k in range(ca.shape[1]) if ca[c, k] == C.NEW],
             "collisions": C.codes_report(ca, tiles)["collisions"]}
        s.update({p: check(dist(X), tiles, kinds, new) for p, X in parts.items()})
        s["codes"] = {}
        for k in range(ca.shape[1]):
            per = {}
            for kd, mem in kinds.items():
                vals = {int(ca[m, k]) for m in mem}
                shared = len(vals) == 1 and C.NEW not in vals
                v = vals.pop() if shared else None
                per[kd] = {"members_share_a_code": shared,
                           "others_with_it": [names[t] for t in tiles if shared and t not in mem and ca[t, k] == v],
                           **{nm: ("new" if ca[c, k] == C.NEW else "same" if shared and ca[c, k] == v else "other")
                              for nm, c in new.items() if kind_of(nm) == kd}}
            s["codes"][f"codebook {k}"] = per
        res["seeds"].append(s)
        out.write_text(json.dumps(res, indent=1) + "\n")
        print(f"seed {seed} done: {rep}", flush=True)


if __name__ == "__main__":
    main()
