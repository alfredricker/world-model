"""Card 111 in breaking world 1: conditions of 105.4's events by contrast over memory (tools/card111/contrast.py).

An event (step t, token i): action a, i's code c0 now and c1 next. Its situation: every remembered training step with
action a, and every tile i' holding c0 there (anchored at i'); success when i' holds c1 next. Facts: the 17 x 13 window
of values around the anchor (the whole grid at any anchor). No pose in world 1, so the agent's token is not a second
anchor (declared in the result); facts about the agent are cells of the window like any other.
Values: perception's codes (form "codes"), or the evaluator's kinds (form "labels": the gate's upper bound).
The truth is 105.4's (the world's rule under one replacement), scored on the same events (106's draw).

  bin/prun python tools/card111/world1.py codes|labels [--per 300] [--smoke] → runs/111/world1_<form>.json
"""
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card111"))
import contrast as CT                                      # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
FORM = sys.argv[1]
PER = int(arg("--per", 300))
SMOKE = "--smoke" in sys.argv
spec = importlib.util.spec_from_file_location("c1054w1", ROOT / "tools" / "card105.4" / "world1.py")
W1 = importlib.util.module_from_spec(spec)
sys.argv = ["world1.py"]
spec.loader.exec_module(W1)
OUT = ROOT / "runs" / "111"
NT, W, H = W1.NT, W1.W, W1.H
FLOOR, WALL, AGENT, BOX, LAMP, LIT = range(6)
DX, DY = W - 1, H - 1                                          # window half-widths: the grid from any anchor
WW, WH = 2 * DX + 1, 2 * DY + 1
MAX_ITEMS = 200_000
P19 = (10, 30, 100, 300)


def windows(X, t, i, K):
    """Values (n, WH * WW) around tiles i of steps t; K outside the grid."""
    y0, x0 = i // W, i % W
    oy, ox = np.meshgrid(np.arange(-DY, DY + 1), np.arange(-DX, DX + 1), indexing="ij")
    yy, xx = y0[:, None] + oy.reshape(-1)[None], x0[:, None] + ox.reshape(-1)[None]
    inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    v = X[t[:, None], np.clip(yy, 0, H - 1) * W + np.clip(xx, 0, W - 1)]
    return np.where(inside, v, K)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1021)
    z = np.load(ROOT / "runs" / "card102" / "world1_test.npz")
    W1.PER = PER
    ev = W1.events(z, rng)                                       # 106's draw: the first use of rng(1021)
    rng2 = np.random.default_rng(111)
    ztr = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    sem, sem_tr = z["sem"].reshape(-1, 63).astype(np.int64), ztr["sem"].reshape(-1, 63).astype(np.int64)
    act, act_tr = z["act"].astype(np.int64), ztr["act"].astype(np.int64)
    if FORM == "labels":
        X, Xtr, K = sem, sem_tr, 6
    else:
        X = np.load(ROOT / "runs" / "102.1" / "world1_code_test.npy")[:, :63].astype(np.int64)
        Xtr = np.load(ROOT / "runs" / "102.1" / "world1_code_train.npy")[:, :63].astype(np.int64)
        K = 7
    T = np.flatnonzero(np.r_[ztr["ep"][1:] == ztr["ep"][:-1], False] & ~ztr["done"])
    W1.PER = 10 ** 9
    occ = {k: len(v) for k, v in W1.events(ztr, np.random.default_rng(0)).items()}
    rep = {"form": FORM, "training occurrences": occ, "smoke": SMOKE}
    cache = {}

    def situation(a, c0, c1):
        if (a, c0) not in cache:
            cache.clear()
            ts = T[act_tr[T] == a]
            tt, ii = np.nonzero(Xtr[ts] == c0)
            tt = ts[tt]
            if len(tt) > MAX_ITEMS:
                s = np.sort(rng2.choice(len(tt), MAX_ITEMS, replace=False))
                tt, ii = tt[s], ii[s]
            cache[(a, c0)] = (tt, ii, windows(Xtr, tt, ii, K), Xtr[tt + 1, ii])
        tt, ii, vals, nxt = cache[(a, c0)]
        key = (a, c0, c1)
        if cache.get("key") != key:
            cache["key"], cache["sit"] = key, CT.Situation(vals, nxt == c1, K)
        return cache["sit"], tt

    res = {}
    for kind, lst in ev.items():
        if SMOKE:
            lst = lst[:30]
        items = []
        for t, i in lst:
            a = int(act[t])
            g = sem[t].reshape(H, W)
            base_true = W1.true_next(g, a).reshape(-1)[i]
            present = [k for k in (FLOOR, WALL, BOX, LAMP, LIT) if (sem[t] == k).any()]
            tc = np.zeros(63, bool)
            for j in range(63):
                if j == i:
                    continue
                for k in ([FLOOR] if sem[t, j] == AGENT else [k for k in present if k != sem[t, j]]):
                    g2 = g.copy().reshape(-1)
                    g2[j] = k
                    tc[j] |= W1.true_next(g2.reshape(H, W), a).reshape(-1)[i] != base_true
            near = np.zeros(63, bool)                            # 105.4's baseline
            ag = int(np.argmax(sem[t] == AGENT))
            for j in range(63):
                if j != i and (max(abs(j % 9 - i % 9), abs(j // 9 - i // 9)) <= 1
                               or max(abs(j % 9 - ag % 9), abs(j // 9 - ag // 9)) <= 1):
                    near[j] = True
            items.append(dict(t=t, i=i, a=a, c0=int(X[t, i]), c1=int(X[t + 1, i]), tc=tc, near=near))
        items.sort(key=lambda e: (e["a"], e["c0"], e["c1"]))

        keep_cache = {}
        why, extra = Counter(), Counter()

        def run(keep_frac=None, seed=0):
            tp = fp = fn = exact = 0
            facts, ms = [], 0.0
            for e in items:
                sit, tt = situation(e["a"], e["c0"], e["c1"])
                keep = None
                if keep_frac is not None:
                    kk = (e["a"], e["c0"], seed)
                    if kk not in keep_cache:
                        kept_t = T[np.random.default_rng(seed).random(len(T)) < keep_frac]
                        keep_cache.clear()
                        keep_cache[kk] = np.isin(tt, kept_t)
                    keep = keep_cache[kk]
                ev_vals = windows(X, np.array([e["t"]]), np.array([e["i"]]), K)[0]
                s = time.monotonic()
                fin = []
                C = CT.greedy(sit, ev_vals, keep, fin)
                ms += time.monotonic() - s
                mc = np.zeros(63, bool)
                y0, x0 = e["i"] // W, e["i"] % W
                for cell, val, is_eq, *_ in C:
                    y, x = y0 + cell // WW - DY, x0 + cell % WW - DX
                    if 0 <= y < H and 0 <= x < W and y * W + x != e["i"]:
                        mc[y * W + x] = True
                tp += int((mc & e["tc"]).sum()); fp += int((mc & ~e["tc"]).sum()); fn += int((~mc & e["tc"]).sum())
                exact += int((mc == e["tc"]).all())
                if keep_frac is None:
                    for j in np.flatnonzero(~mc & e["tc"]):
                        cl = (j // W - y0 + DY) * WW + (j % W - x0 + DX)
                        why[CT.miss_reason(sit, fin[0], cl, int(ev_vals[cl]))] += 1
                    for j in np.flatnonzero(mc & ~e["tc"]):
                        extra[f"{sem[e['t'], j]} at ({j // W - y0}, {j % W - x0})"] += 1
                facts.append([(cell // WW - DY, cell % WW - DX, val, "=" if q else "≠", pw) for cell, val, q, pw, *_ in C])
            pr = lambda x, y: round(x / max(x + y, 1), 4)
            return {"precision": pr(tp, fp), "recall": pr(tp, fn), "events exactly right": round(exact / max(len(items), 1), 4),
                    "seconds per event": round(ms / max(len(items), 1), 4)}, facts

        r, facts = run()
        btp = sum(int((e["near"] & e["tc"]).sum()) for e in items)
        bfp = sum(int((e["near"] & ~e["tc"]).sum()) for e in items)
        bfn = sum(int((~e["near"] & e["tc"]).sum()) for e in items)
        r["events"] = len(items)
        r["baseline 'within one step of i or the agent': precision, recall"] = [round(btp / max(btp + bfp, 1), 4),
                                                                                round(btp / max(btp + bfn, 1), 4)]
        r["missed true conditions, by reason (final subset)"] = dict(why)
        r["false conditions: evaluator kind at (dy, dx) from the event's tile, top 8"] = dict(extra.most_common(8))
        r["true conditions per event (mean)"] = round(float(np.mean([e["tc"].sum() for e in items])), 2)
        r["facts of the first 5 events (dy, dx, value, = or ≠, power)"] = facts[:5]
        print(kind, json.dumps({k: r[k] for k in ("precision", "recall", "events exactly right", "seconds per event")}),
              f"({time.monotonic() - t0:.0f} s)", flush=True)
        p19 = {}
        for n in P19:
            if n < occ[kind]:
                o, _ = run(n / occ[kind], seed=n)
                p19[str(n)] = [o["precision"], o["recall"]]
        r["P19: precision, recall with memory holding about n occurrences"] = p19
        print("  P19", p19, flush=True)
        res[kind] = r
    rep["conditions"] = res
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"world1_{FORM}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
