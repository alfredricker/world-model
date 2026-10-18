"""Card 111.1 in breaking world 1: conditions of 105.4's events by contrast over memory (tools/card111.1/contrast.py).

An event (step t, token i): action a, i's code c0 now and c1 next. Its situation: every remembered training step with
action a, and every tile i' holding c0 there (anchored at i'); success when i' holds c1 next. Facts: the 17 x 13 window
of values around the anchor (the whole grid at any anchor). No pose in world 1, so the agent's token is not a second
anchor (declared in the result); facts about the agent are cells of the window like any other.
Values: perception's codes (form "codes"), or the evaluator's kinds (form "labels": the gate's upper bound).
The truth is 105.4's (the world's rule under one replacement), read only where the replaced state keeps the world's
invariant (every lamp lit exactly when a box is beside it); 105.4's own truth is reported beside it. Same events
(106's draw). Arms: (a) log space only, and (a)-(d).

  bin/prun python tools/card111.1/world1.py codes|labels [--per 300] [--smoke] [--nop19] → runs/111.1/world1_<form>.json
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
sys.path.insert(0, str(ROOT / "tools" / "card111.1"))
import contrast as CT                                      # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
FORM = sys.argv[1]
PER = int(arg("--per", 300))
SMOKE = "--smoke" in sys.argv
NOP19 = "--nop19" in sys.argv
spec = importlib.util.spec_from_file_location("c1054w1", ROOT / "tools" / "card105.4" / "world1.py")
W1 = importlib.util.module_from_spec(spec)
sys.argv = ["world1.py"]
spec.loader.exec_module(W1)
OUT = ROOT / "runs" / "111.1"
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

    def producible(g):
        """The world's own invariant: every lamp is lit exactly when a box stands beside it."""
        for y, x in zip(*np.nonzero((g == LAMP) | (g == LIT))):
            near = any(0 <= y + dy < H and 0 <= x + dx < W and g[y + dy, x + dx] == BOX
                       for dx, dy in W1.MOVES.values())
            if near != (g[y, x] == LIT):
                return False
        return True

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
            tc = np.zeros(63, bool)                              # producible states only (this card)
            judged = np.zeros(63, bool)
            tc105 = np.zeros(63, bool)                           # 105.4's truth
            for j in range(63):
                if j == i:
                    continue
                for k in ([FLOOR] if sem[t, j] == AGENT else [k for k in present if k != sem[t, j]]):
                    g2 = g.copy().reshape(-1)
                    g2[j] = k
                    g2 = g2.reshape(H, W)
                    ch = W1.true_next(g2, a).reshape(-1)[i] != base_true
                    tc105[j] |= ch
                    if producible(g2):
                        judged[j] = True
                        tc[j] |= ch
            near = np.zeros(63, bool)                            # 105.4's baseline
            ag = int(np.argmax(sem[t] == AGENT))
            for j in range(63):
                if j != i and (max(abs(j % 9 - i % 9), abs(j // 9 - i // 9)) <= 1
                               or max(abs(j % 9 - ag % 9), abs(j // 9 - ag // 9)) <= 1):
                    near[j] = True
            items.append(dict(t=t, i=i, a=a, c0=int(X[t, i]), c1=int(X[t + 1, i]), tc=tc, judged=judged, tc105=tc105,
                              near=near))
        items.sort(key=lambda e: (e["a"], e["c0"], e["c1"]))

        keep_cache = {}

        def run(tarone, ties, keep_frac=None, seed=0, diag=False):
            c = Counter()
            why, extra, facts = Counter(), Counter(), []
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
                y0, x0 = e["i"] // W, e["i"] % W

                def tok_of(cell):
                    y, x = y0 + cell // WW - DY, x0 + cell % WW - DX
                    return y * W + x if 0 <= y < H and 0 <= x < W and y * W + x != e["i"] else None

                cell_of = lambda j: int((j // W - y0 + DY) * WW + (j % W - x0 + DX))
                s = time.monotonic()
                fin = []
                C = CT.greedy(sit, ev_vals, keep, fin, tarone=tarone)
                c["sec"] += time.monotonic() - s
                ans = fin[2]
                right, wrong, found = CT.score_sets(C, tok_of, e["tc"], e["judged"], ties)
                r5, w5, f5 = CT.score_sets(C, tok_of, e["tc105"], np.ones(63, bool), ties)
                true = e["tc"] & e["judged"]
                missed = true & ~found
                c["right"] += right; c["wrong"] += wrong; c["true"] += int(true.sum()); c["found"] += int(found.sum())
                c["right105"] += r5; c["wrong105"] += w5; c["true105"] += int(e["tc105"].sum()); c["found105"] += int(f5.sum())
                tj = np.flatnonzero(true)
                ok = CT.contrastable(sit, [cell_of(j) for j in tj], [int(ev_vals[cell_of(j)]) for j in tj], keep)
                for j, o in zip(tj, ok):
                    if o:
                        c["ctrue"] += 1; c["cfound"] += int(found[j])
                c["closed"] += int(ans["closed"])
                c["closed complete"] += int(ans["closed"] and not missed.any())
                if not ans["closed"] and missed.any():
                    nm = {tok_of(cl) for cl in ans["near misses"]}
                    c["missed in open"] += int(missed.sum()); c["missed in near misses"] += sum(int(j in nm) for j in np.flatnonzero(missed))
                for f in C:
                    toks = {tok_of(x) for x in f[7]} - {None}
                    if len(toks) > 1:
                        c["tie sets"] += 1; c["tie members"] += len(toks)
                c["stated"] += len(C)
                if diag:
                    for j in np.flatnonzero(missed):
                        cl = cell_of(j)
                        r_ = CT.miss_reason(sit, fin[0], cl, int(ev_vals[cl]))
                        why[r_ if r_ != "enough cases each side" else "enough: " + CT.why_enough(sit, fin[0], cl, int(ev_vals[cl]))] += 1
                    for f in C:
                        toks = {tok_of(x) for x in (f[7] if ties else [f[0]])} - {None}
                        if toks and not any(true[list(toks)]) and any(e["judged"][list(toks)]):
                            j = min(toks)
                            extra[f"{sem[e['t'], j]} at ({j // W - y0}, {j % W - x0})"] += 1
                    facts.append([(f[0] // WW - DY, f[0] % WW - DX, f[1], "=" if f[2] else "≠", f[3],
                                   len({tok_of(x) for x in f[7]} - {None})) for f in C])
            pr = lambda x, y: round(x / y, 4) if y else None
            n = max(len(items), 1)
            out = {"precision": pr(c["right"], c["right"] + c["wrong"]), "recall": pr(c["found"], c["true"]),
                   "contrastable share": pr(c["ctrue"], c["true"]), "recall on contrastable": pr(c["cfound"], c["ctrue"]),
                   "closed share": round(c["closed"] / n, 4),
                   "closed and complete, of closed": pr(c["closed complete"], c["closed"]),
                   "105.4's truth: precision, recall": [pr(c["right105"], c["right105"] + c["wrong105"]),
                                                       pr(c["found105"], c["true105"])],
                   "missed conditions in open answers": c["missed in open"],
                   "of those, among the near misses (top 10)": pr(c["missed in near misses"], c["missed in open"]),
                   "stated conditions per event": round(c["stated"] / n, 2),
                   "tie sets (share of stated), mean size": [pr(c["tie sets"], c["stated"]), pr(c["tie members"], c["tie sets"])],
                   "seconds per event": round(c["sec"] / n, 4)}
            if diag:
                out["missed true conditions, by reason (final subset)"] = dict(why)
                out["false conditions: evaluator kind at (dy, dx) from the event's tile, top 8"] = dict(extra.most_common(8))
                out["facts of the first 5 events (dy, dx, value, = or ≠, power, tie size)"] = facts[:5]
            return out

        r = {"events": len(items),
             "judged cells per event (mean)": round(float(np.mean([e["judged"].sum() for e in items])), 2),
             "true conditions per event (mean)": round(float(np.mean([(e["tc"] & e["judged"]).sum() for e in items])), 2)}
        tr_ = [e["tc"] & e["judged"] for e in items]
        btp = sum(int((e["near"] & t_).sum()) for e, t_ in zip(items, tr_))
        bfp = sum(int((e["near"] & e["judged"] & ~t_).sum()) for e, t_ in zip(items, tr_))
        bfn = sum(int((~e["near"] & t_).sum()) for e, t_ in zip(items, tr_))
        r["baseline 'within one step of i or the agent': precision, recall"] = [round(btp / max(btp + bfp, 1), 4),
                                                                                round(btp / max(btp + bfn, 1), 4)]
        r["(a) log space only"] = run(False, False)
        r["(a)-(d)"] = run(True, True, diag=True)
        print(kind, json.dumps({k: r["(a)-(d)"][k] for k in ("precision", "recall", "recall on contrastable",
                                                              "closed share", "closed and complete, of closed")}),
              "| (a) only", r["(a) log space only"]["precision"], r["(a) log space only"]["recall"],
              f"({time.monotonic() - t0:.0f} s)", flush=True)
        p19 = {}
        for n in (() if NOP19 else P19):
            if n < occ[kind]:
                o = run(True, True, n / occ[kind], seed=n)
                p19[str(n)] = {k: o[k] for k in ("precision", "recall", "recall on contrastable", "closed and complete, of closed")}
        r["P19 ((a)-(d)), memory holding about n occurrences"] = p19
        print("  P19", json.dumps(p19), flush=True)
        res[kind] = r
    rep["conditions"] = res
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"world1_{FORM}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
