"""Card 036: fresh codes for new things.

Card 035's reading marks a piece "new" when it lies beyond alpha x its nearest code's radius, and every
such piece shows the same marker, so new tiles collide. Here the new pieces of each codebook are grouped
by complete linkage, cut at alpha x m_k (m_k: the codebook's median distance of training pieces to their
codes, card 035's radius floor), and each group gets a fresh code of that codebook. Encoders: card 035's
saved mu = 0.01 encoders; alpha = 6. Nothing is trained.

Run: bin/prun python tools/card036/fresh.py [--enc runs/035_enc] [--mu 0.01] [--seeds 300-309] [--out ...]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card035"))
import novelty as N  # noqa: E402  (card 035: read_codes, code_radii; card 034 and card 031 below it)

C, KP, NEW, K = N.C, N.KP, N.NEW, N.K
ROOT = Path(__file__).resolve().parents[2]
ALPHA = 6.0
M = 8
VIEW = [int(c) for c in C.VIEW_CODES]          # every tile the egocentric view can show


def complete_linkage(D, cut):
    """Groups of points in which every pair lies within cut (agglomerative, complete linkage)."""
    groups = [[i] for i in range(len(D))]
    while len(groups) > 1:
        best, pair = None, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = max(D[i, j] for i in groups[a] for j in groups[b])
                if d <= cut and (best is None or d < best):
                    best, pair = d, (a, b)
        if pair is None:
            break
        a, b = pair
        groups[a] = groups[a] + groups[b]
        del groups[b]
    return groups


def fresh_codes(z_all, books, tiles, alpha=ALPHA, universe=VIEW):
    """Card 035's reading, then fresh codes for the new pieces of the tiles in universe. Returns the codes
    (every tile; tiles outside universe keep "new"), the fresh codes' vectors {(k, code): mean piece}, and a
    report of the cut and the distances."""
    ca = N.read_codes(z_all, books, tiles, alpha)
    z = z_all.reshape(len(z_all), K, -1)
    vec, rep = {}, {}
    for k in range(K):
        zt = z[np.asarray(tiles), k]
        d = N.cdist(zt, books[k])
        m_k = float(np.median(d.min(1)))
        cut = alpha * m_k
        idx = [c for c in universe if ca[c, k] == NEW]
        if not idx:
            rep[k] = {"cut": round(cut, 4), "new_pieces": 0, "fresh_codes": 0}
            continue
        D = N.cdist(z[idx, k], z[idx, k])
        groups = complete_linkage(D, cut)
        for j, g in enumerate(sorted(groups, key=lambda g: min(idx[i] for i in g))):
            code = M + j
            for i in g:
                ca[idx[i], k] = code
            vec[(k, code)] = z[[idx[i] for i in g], k].mean(0)
        within = [D[i, j] for g in groups for i in g for j in g if i < j]
        between = [D[i, j] for a in range(len(groups)) for b in range(a + 1, len(groups))
                   for i in groups[a] for j in groups[b]]
        rep[k] = {"cut": round(cut, 4), "new_pieces": len(idx), "fresh_codes": len(groups),
                  "largest_within_over_cut": round(max(within) / cut, 3) if within else None,
                  "smallest_between_over_cut": round(min(between) / cut, 3) if between else None}
    return ca, vec, rep


def named_apart(ca, tiles):
    new = list(KP.new_tiles().values())
    t = [tuple(int(v) for v in ca[c]) for c in list(tiles) + new]
    names = [C.name_of_code(c) for c in list(tiles) + new]
    same = sorted({f"{names[i]} = {names[j]}" for i in range(len(t)) for j in range(i + 1, len(t)) if t[i] == t[j]})
    return {"apart": len(set(t)) == len(t), "same": same}


def colour_codes(ca):
    """For each new colour: codebooks in which its four new tiles share one fresh code that no tile of the
    other colour has."""
    nt = KP.new_tiles()
    out = {}
    for col, other in (("yellow", "purple"), ("purple", "yellow")):
        mine = [c for nm, c in nt.items() if nm.endswith(col)]
        theirs = [c for nm, c in nt.items() if nm.endswith(other)]
        hits = []
        for k in range(K):
            v = {int(ca[c, k]) for c in mine}
            if len(v) == 1 and next(iter(v)) >= M and next(iter(v)) not in {int(ca[c, k]) for c in theirs}:
                hits.append(k)
        out[col] = hits
    return out


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    enc, mu = Path(get("--enc", "runs/035_enc")), get("--mu", "0.01")
    a, b = get("--seeds", "300-309").split("-")
    out = Path(get("--out", "runs/036_fresh.json"))
    save_dir = Path(get("--save", "runs/036_codes"))
    save_dir.mkdir(parents=True, exist_ok=True)
    cache = pickle.loads(N.T.MEMORY.read_bytes())
    tiles = cache["tiles"]
    res = {"alpha": ALPHA, "arms": {"1_labels": {}, "2_fresh": [], "3_one_marker": []}}
    lab = C.oracle_codes()
    res["arms"]["1_labels"] = named_apart(lab, tiles)
    for seed in range(int(a), int(b) + 1):
        d = np.load(enc / f"mu{mu}_s{seed}.npz", allow_pickle=True)
        z, books = d["z"], d["books"]
        ca, vec, rep = fresh_codes(z, books, tiles)
        old = N.read_codes(z, books, tiles, ALPHA)
        assert (ca[tiles] == old[tiles]).all()          # familiar tiles keep their codes
        nt = KP.new_tiles()
        res["arms"]["2_fresh"].append({"seed": seed, **named_apart(ca, tiles), "colour_codes": colour_codes(ca),
                                       "codebooks": rep,
                                       "new_tiles": {nm: [int(v) for v in ca[c]] for nm, c in nt.items()}})
        res["arms"]["3_one_marker"].append({"seed": seed, **named_apart(old, tiles)})
        np.savez(save_dir / f"mu{mu}_s{seed}.npz", codes=ca,
                 fresh_keys=np.array([list(k) for k in vec], np.int64).reshape(-1, 2),
                 fresh_vecs=np.array(list(vec.values())).reshape(len(vec), -1))
        print(f"seed {seed}: apart {res['arms']['2_fresh'][-1]['apart']} (one marker: "
              f"{res['arms']['3_one_marker'][-1]['apart']}); colour codes {colour_codes(ca)}; "
              f"{ {k: (v['new_pieces'], v['fresh_codes']) for k, v in rep.items()} }", flush=True)
    res["seeds_apart"] = {"2_fresh": sum(s["apart"] for s in res["arms"]["2_fresh"]),
                          "3_one_marker": sum(s["apart"] for s in res["arms"]["3_one_marker"])}
    res["criterion_1"] = res["seeds_apart"]["2_fresh"] >= 8
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"labels_apart": res["arms"]["1_labels"]["apart"], **res["seeds_apart"],
                      "criterion_1": res["criterion_1"]}))


if __name__ == "__main__":
    main()
