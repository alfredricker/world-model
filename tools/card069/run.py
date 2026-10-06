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
T.COUNTS += [WALK.STATS, R.STATS]
R.ONLINE["on"] = arg("--online", "1") == "1"
LAST = {}
_make = T.make


def make(tier):
    env = _make(tier)
    LAST["env"] = env
    return env


def episode(job):
    r = T.episode(job)
    r["decoy_tries"] = int(getattr(LAST["env"].unwrapped, "decoy_tries", 0))
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
    adm = W.kinds[VP.TOG].report.get("conditions", {}).get("admitted")
    log(f"toggle admitted {adm}; recall right on {sum(r['right'] for r in rows)} of 36 door and key hue pairs")
    return {"toggle_admitted": adm, "pick_up_admitted": W.kinds[VP.PICK].report.get("conditions", {}).get("admitted"),
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(json.dumps({k: v for k, v in res.items() if k not in ("per_episode", "setup", "recall")}
                   | {"recall_right": table["right"], "toggle_admitted": table["toggle_admitted"]}))


if __name__ == "__main__":
    main()
