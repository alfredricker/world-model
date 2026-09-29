"""Card 032: fewest codes.

Card 031's encoder with one added term: the code length, the sum over codebooks of the entropy of each
codebook's code use over the training tiles (train_codes' dl_w). Everything else is card 031 as built
(tools/card031/codes.py): the model over codes keyed per group, the tests and the arms.

  sweep: the declared rule for the weight, on familiar tiles only: the largest of 0.003, 0.01, 0.03,
         0.1, 0.3 that keeps all training tiles on distinct tuples in all ten seeds.
  main:  arms 3 (exact tile names), 1 (labels), 2 (main: card 031's encoder plus the term), 4 (the term
         without the pairs); ten seeds for 2 and 4.

Run: bin/prun python tools/card032/fewest.py --sweep 0.01 [--out runs/032_sweep_0.01.json]
     bin/prun python tools/card032/fewest.py --w W [--arms 3,1,2,4] [--out runs/032_fewest.json]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card031"))
import codes as C  # noqa: E402

WEIGHTS = (0.003, 0.01, 0.03, 0.1, 0.3)


def tiles_pairs(log):
    """The encoder's data, as card 031 (cached: it depends only on the fixed seeds)."""
    cache = Path("runs/032_tiles_pairs.json")
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["tiles"], [tuple(p) for p in d["pairs"]]
    pool = C.F.pool20()
    worlds = []
    for world in C.WORLDS:
        _, tr, _, _ = C.dl.collect(pool, world, 5000, 11, 0.0, seq_episodes=4)
        e0, e1 = C.ego_codes(tr["c0"], tr["s0"]), C.ego_codes(tr["c1"], tr["s1"])
        worlds.append({"ego0": e0, "ego1": e1, "act": tr["act"].astype("int64"),
                       "view_changed": (C.APP0[e0][:, :C.NV] != C.APP0[e1][:, :C.NV]).any(1),
                       "term1": tr["term1"].astype(bool)})
    pool.close()
    tiles, pairs = C.tiles_and_pairs(worlds)
    cache.write_text(json.dumps({"tiles": [int(t) for t in tiles], "pairs": [[int(a), int(b)] for a, b in pairs]}))
    return tiles, pairs


def sweep(w, out, log):
    tiles, pairs = tiles_pairs(log)
    rows = []
    for seed in range(10):
        ca, rep = C.train_codes(tiles, pairs, seed=seed, dl_w=w)
        by = Counter(tuple(int(v) for v in ca[c]) for c in tiles)
        rep["collisions"] = [[C.name_of_code(c) for c in tiles if tuple(int(v) for v in ca[c]) == t]
                             for t, n in by.items() if n > 1]
        rep["seen_tiles_with_a_new_code"] = [C.name_of_code(c) for c in tiles if (ca[c] == C.NEW).any()]
        rows.append(rep)
        log(f"w {w} seed {seed}: {rep}")
    res = {"weight": w, "seeds": rows, "distinct_in_all_seeds": all(not r["collisions"] for r in rows)}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    if "--sweep" in args:
        w = float(args["--sweep"])
        sweep(w, args.get("--out", f"runs/032_sweep_{w}.json"), log)
        return
    w = float(args["--w"])
    out = Path(args.get("--out", "runs/032_fewest.json"))
    arms = args.get("--arms", "3,1,2,4").split(",")
    C.closer.MEASURE = "step"
    C.dl.MAX_GOALS = 10 ** 9
    C.dl.START_SHARE = 2.0
    C.dl.Exact = C.closer.Approach
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"note": "Card 032, tools/card032/fewest.py; card 031 as built with a code-length term in the encoder.",
           "weight": w, "episodes": 5000, "layouts": 500, "heldout_episodes": 1000, "test_episodes": 1000,
           "seeds": 10, "arms": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    data, tests, tiles, pairs = C.setup(5000, 1000, 500, 1000, res, save, log)
    ref = None
    for arm in arms:
        if arm == "3":
            o = C.run_codes("arm 3", C.APP0[:, None].astype("int64"), False, data, tests, dev, log)
            ref = {x: o[x]["acting"]["mean_steps_when_successful"] for x in C.WORLDS}
            res["arms"]["3_exact_tile_names"] = o
            save()
            continue
        if arm == "1":
            ca = C.oracle_codes()
            o = C.run_codes("arm 1", ca, True, data, tests, dev, log, ref)
            o["codes"] = C.codes_report(ca, tiles)
            o["verdicts"] = C.verdicts(o)
            res["arms"]["1_labels"] = o
            log(f"arm 1 verdicts {o['verdicts']}")
            save()
            continue
        setting = {"2": dict(K=4, M=8, dim=8, pair_w=0.1, dl_w=w), "4": dict(K=4, M=8, dim=8, pair_w=0.0, dl_w=w)}[arm]
        name = {"2": "2_main", "4": "4_no_pairs"}[arm]
        runs = res["arms"][name] = {"setting": setting, "seeds": []}
        for seed in range(10):
            ca, crep = C.train_codes(tiles, pairs, seed=seed, dev=dev, **setting)
            o = C.run_codes(f"arm {arm} seed {seed}", ca, True, data, tests, dev, log, ref)
            o["encoder"] = crep
            o["codes"] = C.codes_report(ca, tiles)
            o["verdicts"] = C.verdicts(o)
            runs["seeds"].append(o)
            log(f"arm {arm} seed {seed}: encoder {crep}; collisions {o['codes']['collisions']}; "
                f"purple {[(k, v['codes']) for k, v in o['codes']['purple'].items()]}; verdicts {o['verdicts']}")
            save()
        runs["seeds_passing"] = {c: sum(s["verdicts"][c] for s in runs["seeds"])
                                 for c in ("criterion_1", "criterion_2", "criterion_3")}
        log(f"arm {arm}: seeds passing {runs['seeds_passing']}")
        save()
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
