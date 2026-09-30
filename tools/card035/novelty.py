"""Card 035: novelty judged by a code's own tiles.

Card 034's recall term trains card 031's encoder (8 codes per codebook). What changes is how the codes are
read. Card 031 called a piece "new" when it lay farther from its nearest code than half that code's distance
to the next code. Here every piece takes its nearest used code, and is "new" only when it lies farther from
it than alpha times the code's radius: the farthest of the code's own training tiles, and at least the
median distance of the codebook's training tiles to their codes. alpha is chosen on familiar tiles by
leaving each one out in turn: where others share its code it should get that code, and where none do it
should be "new".

Steps:
  bin/prun python tools/card035/novelty.py --gate MU                       10 encoders at weight MU, saved
  bin/prun python tools/card035/novelty.py --calibrate MU                  leave-one-out choice of alpha
  bin/prun python tools/card035/novelty.py --main --arm 2 --mu MU --alpha A --seeds 300-302 --out runs/035_arm2_a.json
  bin/prun python tools/card035/novelty.py --labels --out runs/035_arm1.json
Add --small for a smoke test (card 034's small data, one seed, short training).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card034"))
import teach as T  # noqa: E402  (card 034: memory with forward, priors, test worlds, reports)

R, C, KP, NEW = T.R, T.C, T.KP, T.NEW
ROOT = Path(__file__).resolve().parents[2]
SMALL = T.SMALL
SUF = T.SUF
SEEDS = (300,) if SMALL else tuple(range(300, 310))
ENC = ROOT / "runs" / f"035_enc{SUF}"
ALPHAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
K = 4


# ---------------------------------------------------------------- encoders

def encoder(mu, seed, tiles, pairs, groups, dev, log=print):
    """Train (or load) one encoder: pieces of every tile, codebook vectors, recall's weights, and card 031's
    reading of the codes (half the gap)."""
    f = ENC / f"mu{mu}_s{seed}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files} | {"rep": json.loads(str(d["rep"]))}
    ENC.mkdir(parents=True, exist_ok=True)
    ca, rep, z, th = R.train_encoder(tiles, pairs, groups, mu, mode="own", seed=seed, dev=dev, M=8,
                                     **({"updates": 500} if SMALL else {}))
    out = {"z": z, "books": R.LAST["books"], "theta": np.asarray(th), "halfgap": ca}
    np.savez(f, rep=json.dumps(rep), **out)
    return out | {"rep": rep}


# ---------------------------------------------------------------- reading the codes

def cdist(a, b):
    return np.linalg.norm(a[:, None] - b[None], axis=-1)


def code_radii(zt, e):
    """Used codes of one codebook (nearest to at least one training piece) and each one's radius: the
    farthest of its own pieces, at least the median distance of all training pieces to their codes."""
    d = cdist(zt, e)
    a = d.argmin(1)
    dist = d[np.arange(len(zt)), a]
    used = sorted(set(a.tolist()))
    m = float(np.median(dist))
    return used, np.array([max(float(dist[a == c].max()), m) for c in used]) + 1e-6


def read_codes(z_all, books, tiles, alpha):
    """Every tile's nearest used code in each codebook, or "new" beyond alpha x that code's radius."""
    z = z_all.reshape(len(z_all), K, -1)
    tiles = np.asarray(tiles)
    out = np.full((len(z), K), NEW, np.int64)
    for k in range(K):
        used, rad = code_radii(z[tiles, k], books[k])
        d = cdist(z[:, k], books[k][used])
        j = d.argmin(1)
        ok = d[np.arange(len(d)), j] <= alpha * rad[j]
        out[ok, k] = np.asarray(used)[j[ok]]
    return out


def loo(z_all, books, tiles, alphas=ALPHAS):
    """Each training piece left out in turn (radii from the others): right if it gets its own code where
    another training piece shares that code, and "new" where none does. Also the two trivial readings."""
    z = z_all.reshape(len(z_all), K, -1)
    tiles = np.asarray(tiles)
    right = {a: 0 for a in alphas} | {"never new": 0, "always new": 0}
    n = shared_n = 0
    for k in range(K):
        zt, e = z[tiles, k], books[k]
        own = cdist(zt, e).argmin(1)
        for i in range(len(tiles)):
            rest = np.delete(np.arange(len(tiles)), i)
            used, rad = code_radii(zt[rest], e)
            dd = np.linalg.norm(e[used] - zt[i], axis=1)
            j = int(dd.argmin())
            shared = bool((own[rest] == own[i]).any())
            n += 1
            shared_n += int(shared)
            for a in alphas:
                fam = dd[j] <= a * rad[j]
                right[a] += int((fam and used[j] == own[i]) if shared else (not fam))
            right["never new"] += int(shared and used[j] == own[i])
            right["always new"] += int(not shared)
    return right, n, shared_n


def told_apart(ca, tiles):
    """Evaluator: for each new tile, whether its codes differ from every familiar tile's."""
    fam = {tuple(int(v) for v in ca[c]): C.name_of_code(c) for c in tiles}
    out = {}
    for nm, c in KP.new_tiles().items():
        t = tuple(int(v) for v in ca[c])
        out[nm] = {"apart": t not in fam, "same_as": fam.get(t)}
    return out


def verdicts(o, apart):
    v = T.verdicts(o)
    for col in T.COLOURS:
        ok = all(r["apart"] for nm, r in apart.items() if nm.endswith(col))
        v["per_colour"][col]["told_apart"] = ok
        v["per_colour"][col]["criterion_2"] = v["per_colour"][col]["criterion_2"] and ok
    v["criterion_2"] = all(v["per_colour"][col]["criterion_2"] for col in T.COLOURS)
    return v


# ---------------------------------------------------------------- main

def main():
    import pickle
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
    seeds = SEEDS
    if get("--seeds", None):
        a, b = get("--seeds", "").split("-")
        seeds = tuple(range(int(a), int(b) + 1))

    if flag("--gate"):
        mu = float(get("--gate", "0"))
        out = ROOT / "runs" / f"035_gate_{mu}{SUF}.json"
        res = {"mu": mu, "seeds": []}
        for seed in seeds:
            e = encoder(mu, seed, tiles, pairs, groups, dev, log)
            ca = read_codes(e["z"], e["books"], tiles, 1.0)
            s = {"seed": seed, "encoder": e["rep"], **R.tiles_ok(ca, tiles),
                 "halfgap": R.tiles_ok(e["halfgap"], tiles),
                 "collided": [C.name_of_code(c) for c in tiles
                              if sum(tuple(ca[c]) == tuple(ca[t]) for t in tiles) > 1]}
            res["seeds"].append(s)
            log(f"gate mu {mu} seed {seed}: {s}")
            out.write_text(json.dumps(res, indent=1) + "\n")
        res["seeds_distinct"] = sum(s["distinct"] for s in res["seeds"])
        res["qualifies"] = res["seeds_distinct"] == len(seeds)
        out.write_text(json.dumps(res, indent=1) + "\n")
        log(f"mu {mu}: distinct in {res['seeds_distinct']} of {len(seeds)}")
        return

    if flag("--calibrate"):
        mu = float(get("--calibrate", "0"))
        out = ROOT / "runs" / f"035_alpha{SUF}.json"
        tot, n, sh, per = None, 0, 0, []
        for seed in seeds:
            e = encoder(mu, seed, tiles, pairs, groups, dev, log)
            r, ni, si = loo(e["z"], e["books"], tiles)
            per.append({"seed": seed, "right": {str(k): v for k, v in r.items()}, "cases": ni, "shared": si})
            tot = r if tot is None else {k: tot[k] + r[k] for k in r}
            n, sh = n + ni, sh + si
        share = {str(k): round(v / n, 4) for k, v in tot.items()}
        best = max(ALPHAS, key=lambda a: (tot[a], -a))
        res = {"mu": mu, "cases": n, "shared": sh, "share_right": share, "alpha": best, "per_seed": per,
               "passes": bool(share[str(best)] >= 0.9 and tot[best] > max(tot["never new"], tot["always new"]))}
        out.write_text(json.dumps(res, indent=1) + "\n")
        log(f"alpha: {res}")
        return

    d = pickle.loads(T.DATA.read_bytes())
    data, tests = d["data"], d["tests"]
    ref = json.loads(T.REF.read_text())
    out = Path(get("--out", "runs/035_main.json"))
    res = {"note": "Card 035, tools/card035/novelty.py", "arms": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")

    if flag("--labels"):
        ca = C.oracle_codes()
        o = C.run_codes("arm 1", ca, True, data, tests, dev, log, ref)
        ap = told_apart(ca, tiles)
        o["new_tiles"], o["told_apart"] = T.new_tiles_report(ca, tiles), ap
        o["verdicts"] = verdicts(o, ap)
        res["arms"]["1_labels"] = o
        log(f"arm 1 verdicts {o['verdicts']}")
        save()
        return

    arm = get("--arm", "2")
    mu = 0.0 if arm == "3" else float(get("--mu", "0"))
    alpha = float(get("--alpha", "1"))
    runs = res["arms"][{"2": "2_main", "3": "3_no_recall_term", "4": "4_half_gap_rule"}[arm]] = \
        {"mu": mu, "alpha": alpha if arm != "4" else None, "seeds": []}
    for seed in seeds:
        e = encoder(mu, seed, tiles, pairs, groups, dev, log)
        ca = e["halfgap"] if arm == "4" else read_codes(e["z"], e["books"], tiles, alpha)
        ap = told_apart(ca, tiles)
        o = {"seed": seed, "encoder": e["rep"], "tiles": R.tiles_ok(ca, tiles), "codes": C.codes_report(ca, tiles),
             "new_tiles": T.new_tiles_report(ca, tiles), "told_apart": ap, "kinds_check": T.kinds_check(e["z"], tiles)}
        if mu > 0 and arm == "2":
            o["recall"] = T.recall_reports(e["z"], list(e["theta"]), mem)
        oc = C.run_codes(f"arm {arm} seed {seed}", ca, True, data, tests, dev, log, ref)
        o["worlds"] = oc
        o["verdicts"] = verdicts(oc, ap)
        runs["seeds"].append(o)
        log(f"arm {arm} seed {seed}: tiles {o['tiles']}; apart {[k for k, v in ap.items() if not v['apart']]} not; "
            f"verdicts {o['verdicts']}")
        save()
    runs["seeds_passing"] = {c: sum(s["verdicts"][c] for s in runs["seeds"]) for c in ("criterion_1", "criterion_2", "criterion_3")}
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done: {runs['seeds_passing']} -> {out}")


if __name__ == "__main__":
    main()
