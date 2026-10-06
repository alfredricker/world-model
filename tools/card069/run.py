"""Card 069's runs: version 15 (card 067's walking) with card 069's relation (relation.py), on CHARTER's tiers and on
the decoy world's folds.

  tiers:  bin/prun python tools/card069/run.py tiers --tier 1 --online 1 --n 200 --memory 066 --out runs/069/tier1.json
  decoy:  bin/prun python tools/card069/run.py decoy --fold red --online 1 --n 100 --memory 066 --out runs/069/decoy_red_on.json
          (card 070: --hold opening keeps every hue in memory and drops only the fold hue's opening tries)

--memory 066 uses card 066's random play for memory, 068 card 068's play starts (whichever the best version uses).
A decoy fold leaves one hue out of memory: memory's doors and keys (the decoy too) take the other five hues; then
recall is asked about every door hue and key hue (criterion 2), and the test episodes' door and key take the
left-out hue, the decoy one of the other five (criterion 3).
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
arg = lambda k, d=None: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
if MODE == "decoy":
    sys.argv += ["--tier", "1"]
    os.environ["WM_LATTICE"] = "12"                    # tier 1's map size
for c in ("card066", "card067", "card068", "card069"):
    sys.path.insert(0, str(ROOT / "tools" / c))
if arg("--memory", "066") == "068":
    import play as PL                                  # noqa: E402  (card 068's play starts; sets T.OUT to runs/068)
    T = PL.T
else:
    import tiers as T                                  # noqa: E402
import numpy as np                                     # noqa: E402
import propagate as WALK                               # noqa: E402  (card 067)
import relation as R                                   # noqa: E402

T.INSTALL += [WALK.install, R.install]
sys.modules["index"].HOLD["by"] = os.environ.get("WM_HOLD_BY", "try")   # card 071 sets "combination"
T.COUNTS += [WALK.STATS, R.STATS]
R.ONLINE["on"] = arg("--online", "1") == "1"
TRY = None
if os.environ.get("WM_TRYING") == "1":                 # card 072: try the likeliest untried way
    sys.path.insert(0, str(ROOT / "tools" / "card072"))
    import trying as TRY                               # noqa: E402
    T.INSTALL.append(TRY.install)
    T.COUNTS.append(TRY.STATS)
LAST = {}
_make = T.make


def make(tier):
    env = _make(tier)
    LAST["env"] = env
    return env


def relation_gap(W):
    """Card 072: on MiniGrid's six hues, the smallest relation between a locked door and a key of another colour
    less the largest between a locked door and its own key, under the current P (positive: every own key closer)."""
    h = lambda o: int(T.Kd.APP[T.KR.code(o)])
    D = np.stack([W.S.arr[h(("door", c, 0))] for c in T.HUES])
    K = np.stack([W.S.arr[h(("key", c))] for c in T.HUES])
    Rm = np.abs((D[:, None] - K[None]).reshape(len(T.HUES), len(T.HUES), -1) @ R.REL["P"].T).sum(-1)
    eye = np.eye(len(T.HUES), dtype=bool)
    return round(float(Rm[~eye].min() - Rm[eye].max()), 4)


def rel_weights(W):
    return {kd.name: round(float(kd.lamc[kd.adm.index(kd.cand.index(R.CAND))]), 4)
            for kd in R._kinds(W) if R.CAND in kd.cand and kd.cand.index(R.CAND) in kd.adm}


def episode(job):
    r = T.episode(job)
    r["decoy_tries"] = int(getattr(LAST["env"].unwrapped, "decoy_tries", 0))
    if TRY is not None:
        W = T.VP.WORLD
        r["tried"] = list(TRY.STATE["log"])
        r["relation_gap_end"] = relation_gap(W)
        r["P_moved"] = round(float(np.abs(R.REL["P"] - R.REL["P0"]).sum()), 4)
        r["rel_weight_end"] = rel_weights(W)
    return r


T.make = make


def recall_table(W, log):
    """Criterion 2: for every door hue and key hue, does recall predict that toggling the locked door with the key
    opens it (category 1) exactly when the hues match? The context is a decoy-world start's view."""
    VP, KR, Kd = T.VP, T.KR, T.Kd
    h = lambda o: int(Kd.APP[KR.code(o)])
    pl = T.S7.Plan047(W)
    env = _make(1)
    env.reset(seed=T.SEED_TEST + 999)
    codes = T.now_codes(env)
    facts, st, _, _ = pl.observe(None, T.PV.crop(Kd.APP[codes], codes))
    ctx = pl.ctx_id(st[0], st[1])
    rows = []
    for d in T.HUES:
        for k in T.HUES:
            cat = int(W.outcome(VP.TOG, h(("door", d, 0)), h(("key", k)), ctx)[0])
            rows.append({"door": d, "key": k, "category": cat, "right": (cat == 1) == (d == k)})
    kd = W.kinds[VP.TOG]
    combos = len(set(zip(np.asarray(kd.fid).tolist(), np.asarray(kd.hid).tolist())))
    fold = arg("--fold")
    pair_keys = 0 if fold is None else sum(1 for k in kd.keys if int(k[0]) == h(("door", fold, 0))
                                           and int(k[1]) == h(("key", fold)))
    lam = {kd.cname(c): round(float(l), 3) for c, l in zip(kd.adm, kd.lamc)}
    log(f"toggle: {combos} (front, held) combinations stored; the fold's opening pair has {pair_keys} stored keys; "
        f"weights per unit of distance {lam}")
    adm = W.kinds[VP.TOG].report.get("conditions", {}).get("admitted")
    log(f"toggle admitted {adm}; recall right on {sum(r['right'] for r in rows)} of 36 door and key hue pairs")
    return {"toggle_combinations": combos, "fold_pair_keys": pair_keys, "toggle_weights": lam, "toggle_admitted": adm, "pick_up_admitted": W.kinds[VP.PICK].report.get("conditions", {}).get("admitted"),
            "right": sum(r["right"] for r in rows), "pairs": rows}


def main():
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    out = Path(arg("--out"))
    n = int(arg("--n", "100"))
    if MODE == "tiers":
        T.run(T.TIER, n, out, log)
        return
    hue = arg("--fold")
    others = [c for c in T.HUES if c != hue]
    T.ENVS[1] = R.register()
    if arg("--hold", "hue") == "hue":                  # card 069: the hue left out of memory
        T.OUT = ROOT / "runs" / "069" / f"decoy_{hue}_memory{arg('--memory', '066')}"   # memory only (codes, no vectors)
        R.DECOY.update(door=others, decoy=others)
    else:                                              # card 070 (the user, 2026-10-05): every hue in memory, less
        T.OUT = ROOT / "runs" / "070" / f"decoy_memory{arg('--memory', '066')}"        # the hue's opening tries
        R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
        _memory = T.memory

        def memory(tier, pool=None):
            D, stats = _memory(tier, pool)
            f = D["ego0"][:, T.FRONT].astype(np.int64) // 5 * 5
            h = D["ego0"][:, T.HELD].astype(np.int64) // 5 * 5
            drop = (D["act"] == T.VP.TOG) & (f == T.KR.code(("door", hue, 0))) & (h == T.KR.code(("key", hue)))
            inter = np.isin(D["act"], T.INTER)
            keep_t = ~drop[inter]
            D = {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}
            return D, {**stats, "opening_tries_removed": int(drop.sum()), "rows": int(len(D["act"]))}

        T.memory = memory
    W, info = T.setup(1, log)
    table = recall_table(W, log)
    if TRY is not None:
        table["relation_gap_setup"] = relation_gap(W)
        table["rel_weight_setup"] = rel_weights(W)
        log(f"relation gap at setup {table['relation_gap_setup']}; weight on rel:P {table['rel_weight_setup']}")
    R.DECOY.update(door=[hue], decoy=others)
    seeds = T.SEED_TEST + 1000 + np.arange(n)
    pool = T.F.pool20()
    t0 = time.monotonic()
    recs = []
    for r in pool.imap_unordered(episode, [(1, int(s)) for s in seeds]):
        recs.append(r)
        log(f"fold {hue} seed {r['seed']}: success {r['success']} steps {r['steps']} decoy tries {r['decoy_tries']} "
            f"refits {r.get('refits', 0)} ({len(recs)}/{n})")
    pool.close()
    recs.sort(key=lambda r: r["seed"])
    ok = [r for r in recs if r["success"]]
    res = {"note": "Card 069, tools/card069/run.py decoy", "fold": hue, "hold": arg("--hold", "hue"),
           "encoder": str(R.ENCODER), "online": R.ONLINE["on"],
           "memory": arg("--memory", "066"), "episodes": n, "success": round(len(ok) / n, 4),
           "decoy_tries_per_episode": round(float(np.mean([r["decoy_tries"] for r in recs])), 3),
           "mean_steps_when_successful": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else None,
           "wrong_predictions_pick_toggle": int(sum(r.get("wrong_predictions_pick_toggle", 0) for r in recs)),
           "refits": int(sum(r.get("refits", 0) for r in recs)),
           "episodes_out_of_time": int(sum(r["timed_out"] for r in recs)),
           "wall_seconds": round(time.monotonic() - t0, 1), "recall": table, "setup": info, "per_episode": recs}
    if TRY is not None:                                # card 072's reports
        steps = sum(r["steps"] for r in recs)
        res.update({k: round(sum(r.get(k, 0) for r in recs) / n, 3) for k in TRY.STATS})
        res.update({"random_share": round(sum(r["random"] for r in recs) / steps, 4),
                    "explore_share": round(sum(r["explore"] for r in recs) / steps, 4),
                    "relation_gap_end_mean": round(float(np.mean([r["relation_gap_end"] for r in recs])), 4),
                    "P_moved_mean": round(float(np.mean([r["P_moved"] for r in recs])), 4)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(json.dumps({k: v for k, v in res.items() if k not in ("per_episode", "setup", "recall")}
                   | {"recall_right": table["right"], "toggle_admitted": table["toggle_admitted"]}))


if __name__ == "__main__":
    main()
