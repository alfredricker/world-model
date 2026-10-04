"""Card 051, step 1: the checks.

  same decisions:  the cluttered world and card 048's chained rooms (one door), every action recorded, under
                   card 050's recall and under step 1's index
  flat cost:       memory grown by real tries from random play in cluttered layouts (layouts 100 onward, so the
                   test layouts' own rooms are not in memory), then time per step on the test layouts

  bin/prun python tools/card051/step1.py same --recall own|index --seed 399 --n 30 --out runs/051/s1_same_own_399.json
  bin/prun python tools/card051/step1.py grow --recall index --seed 399 --keys 0,5000,20000,50000 --n 10 --out runs/051/s1_grow_index_399.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index as IX                                     # noqa: E402

CD, CR, S7 = IX.CD, IX.CR, IX.S7
C = CD.C
WD = None
PICK, DROP, TOG = 3, 4, 5


def setup(recall, seed):
    global WD
    if recall == "index":
        IX.install()
    elif recall == "threats":
        import threats                                  # noqa: E402
        threats.install()
    else:
        CR.kind = IX.OWN.own_kind
    import worlds as wd                                 # noqa: E402  (card 049's worlds)
    WD = wd
    return C.setup(seed)


def act(W, lay, i, M, budget=None):
    """card 049's act(), recording every action and the time of each step."""
    W.reset()
    pl = S7.Plan047(W)
    rng = np.random.default_rng(1000 + i)
    s = M.start_state(lay)
    V = M.see(lay, s)
    facts, st, _, _ = pl.observe(None, V)
    acts, times = [], []
    end = False
    for t in range(budget or C.BUDGET):
        t0 = time.monotonic()
        res = pl.choose(st)
        a = res.action if res is not None else int(rng.integers(6))
        pred = pl.step(st, a)
        s2, end = M.step(lay, s, a)
        V2 = M.see(lay, s2)
        facts, st, miss, _ = pl.observe(facts, V2, end, prefer=pred[1])
        W.learn_try(V, a, V2, end)
        pl.forget()
        times.append(time.monotonic() - t0)
        acts.append(int(a) if res is not None else -1 - int(a))
        s, V = s2, V2
        if end:
            break
    return {"success": bool(end), "steps": len(acts), "actions": acts, "seconds": round(sum(times), 3),
            "step_ms_median": round(1000 * float(np.median(times)), 2)}


# ---------------------------------------------------------------- memory grown by random play

def random_tries(W, want, seed):
    """(action, key, category, after) from random play in cluttered layouts 100 onward, until the stored pick
    up, toggle and drop keys would number `want` more."""
    lays, _ = WD.clutter_layouts(400)
    lays = lays[100:]
    rng = np.random.default_rng(seed)
    tries, seen = [], {a: set(W.kinds[a].keys) for a in (PICK, DROP, TOG)}
    new = 0
    li = 0
    while new < want:
        lay = lays[li % len(lays)]
        li += 1
        s = WD.start_state(lay)
        V = WD.see(lay, s)
        for _ in range(200):
            a = int(rng.choice(6, p=[0.15, 0.15, 0.2, 0.2, 0.15, 0.15]))
            s2, end = WD.step(lay, s, a)
            V2 = WD.see(lay, s2)
            if a in (PICK, DROP, TOG):
                f0, h0, f1, h1 = int(V[VP.FRONT]), int(V[VP.HELD]), int(V2[VP.FRONT]), int(V2[VP.HELD])
                key = (f0, h0, W.ctx_of_view(V))
                cat = int(f1 != f0) + 2 * int(h1 != h0)
                tries.append((a, key, cat, (f1, h1)))
                if key not in seen[a]:
                    seen[a].add(key)
                    new += 1
            s, V = s2, V2
            if end:
                break
    return tries


def bulk_add(kd, tries):
    """Every try added as add() would, the arrays grown once."""
    S = kd.S
    newk = list(dict.fromkeys(k for k, _, _ in tries if k not in kd.index))
    if newk:
        for k in newk:
            kd.index[k] = len(kd.keys)
            kd.keys.append(k)
            kd.group.append(kd.gid.setdefault(k[:2], len(kd.gid)))
        kd.kpos = np.concatenate([kd.kpos, np.array([kd._pos(k[2]) for k in newk], np.int64)])
        m = len(newk)
        kd.X = np.vstack([kd.X, np.stack([kd.key_vec(k) for k in newk])])
        kd.counts = np.vstack([kd.counts, np.zeros((m, kd.ncat))])
        kd.sums = np.concatenate([kd.sums, np.zeros((m,) + kd.sums.shape[1:])])
        kd.aw = np.vstack([kd.aw, np.zeros((m, kd.ncat))])
        kd.fid = np.append(kd.fid, [CR.code_id(S, k[0]) for k in newk])
        kd.hid = np.append(kd.hid, [CR.code_id(S, k[1]) for k in newk])
        kd.lc = np.vstack([kd.lc, np.zeros((m, kd.lc.shape[1]))])
    for k, cat, after in tries:
        i = kd.index[k]
        kd.counts[i, cat] += 1.0
        kd._acc(i, cat, after, 1.0)
        j = kd._label(cat, np.asarray(after, np.int64))
        kd.lc[i, j] += 1.0
    kd.forget()
    if isinstance(kd, IX.IndexKind):
        kd.build_index()
    # the grown memory is what reset() returns to
    kd.base = (list(kd.keys), dict(kd.gid), list(kd.group), kd.X.copy(), kd.counts.copy(), dict(kd.aft),
               (kd.sums.copy(), kd.aw.copy()), kd.kpos.copy(), list(kd.ulist))
    kd.base2 = (kd.lc.copy(), dict(kd.lab), list(kd.lcat), list(kd.lafter), kd.fid.copy(), kd.hid.copy())
    if isinstance(kd, IX.IndexKind):
        kd.base_ix = kd._copy_ix()


VP = CD.VP


def main():
    args = sys.argv[1:]
    mode = args[0]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    recall, seed, n = get("--recall", "index"), int(get("--seed", "399")), int(get("--n", "30"))
    out = Path(get("--out", f"runs/051/s1_{mode}_{recall}_{seed}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    W = setup(recall, seed)
    res = {"note": "Card 051 step 1, tools/card051/step1.py", "mode": mode, "recall": recall, "seed": seed,
           "setup_seconds": round(time.monotonic() - t0, 1),
           "conditions": {str(a): k.report.get("conditions") for a, k in W.kinds.items() if k.report.get("conditions")}}
    print({k: v for k, v in res.items() if k != "conditions"}, flush=True)
    if mode == "same":
        cl, _ = WD.clutter_layouts(100)
        ch = C.load_layouts()["one_door"]["test"]
        for name, lays, M in (("clutter", cl[:n], WD.CLUT), ("chain_one_door", ch[:n], WD.CHAIN)):
            eps = []
            for i, lay in enumerate(lays):
                e = act(W, lay, i, M)
                eps.append(e)
                print(name, i, {k: v for k, v in e.items() if k != "actions"}, flush=True)
            res[name] = {"success": round(np.mean([e["success"] for e in eps]), 4),
                         "seconds_per_layout": round(float(np.mean([e["seconds"] for e in eps])), 3),
                         "episodes": eps}
            print(name, {k: v for k, v in res[name].items() if k != "episodes"}, flush=True)
    else:
        sizes = [int(x) for x in get("--keys", "0,5000,20000,50000").split(",")]
        cl, _ = WD.clutter_layouts(100)
        res["sizes"] = []
        have = 0
        limit = float(get("--limit", "1800"))
        for size in sizes:
            if size > have:
                tg = time.monotonic()
                tries = random_tries(W, size - have, seed + size)
                for a in (PICK, DROP, TOG):
                    bulk_add(W.kinds[a], [(k, c, af) for aa, k, c, af in tries if aa == a])
                have = size
                grow_s = round(time.monotonic() - tg, 1)
            else:
                grow_s = 0.0
            readmit = None
            if recall == "index" and size > 0:
                ta = time.monotonic()
                for a in (PICK, DROP, TOG):
                    W.kinds[a].admit()                  # admission again on the grown memory, over groups
                readmit = {"seconds": round(time.monotonic() - ta, 1),
                           "admitted": {VP.KNAME[a]: W.kinds[a].report["conditions"]["admitted"][4:]
                                        for a in (PICK, DROP, TOG)},
                           "groups": {VP.KNAME[a]: W.kinds[a].report["conditions"]["groups_while_admitting"]
                                      for a in (PICK, DROP, TOG)}}
            keys = {VP.KNAME[a]: len(W.kinds[a].keys) for a in (PICK, DROP, TOG)}
            groups = {VP.KNAME[a]: len(W.kinds[a].ix["rep"]) for a in (PICK, DROP, TOG)
                      if hasattr(W.kinds[a], "ix")}
            eps, ts = [], time.monotonic()
            for i, lay in enumerate(cl[:n]):
                eps.append(act(W, lay, i, WD.CLUT))
                if time.monotonic() - ts > limit:
                    break
            steps = sum(e["steps"] for e in eps)
            row = {"added_keys": size, "keys": keys, "groups": groups, "grow_seconds": grow_s, "readmission": readmit,
                   "layouts": len(eps), "success": round(float(np.mean([e["success"] for e in eps])), 4),
                   "ms_per_step": round(1000 * sum(e["seconds"] for e in eps) / steps, 2),
                   "median_step_ms": round(float(np.median([e["step_ms_median"] for e in eps])), 2),
                   "stopped_at_limit": len(eps) < n}
            res["sizes"].append(row)
            print(row, flush=True)
            out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
