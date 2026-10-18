"""Card 111.1 in Crafter: conditions of 106's events by contrast over memory (tools/card111.1/contrast.py).

Memory: card 105.10's training transitions, every token's value (perception's code, form "codes"; or the evaluator's
label, form "labels": the gate's upper bound). The view follows the agent, so a token's index is its offset from the
agent's own token; carried tokens sit at their fixed places. Situations:
- make (a tool made): steps with the same action whose tool slot holds e's code; success when it holds e's next code;
- moves (blocked, succeeded): steps with the same action; success when the agent's own pose changed as in e (moved or
  not; proprioception, GOAL C1).
The truth is 106's (Crafter's rule, data.yaml, under one replacement). The agent's own token (31) and the tool's slot
are not named.

Arms: (a) log space only, and (a)-(d).

  bin/prun python tools/card111.1/crafter.py codes|labels [--per 300] [--smoke] [--nop19] → runs/111.1/crafter_<form>.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
FORM = sys.argv[1]
PER = int(arg("--per", 300))
SMOKE = "--smoke" in sys.argv
NOP19 = "--nop19" in sys.argv
sys.argv = ["run.py", "--origins", "pose"]
sys.path.insert(0, str(ROOT / "tools" / "card105.2"))
sys.path.insert(0, str(ROOT / "tools" / "card111.1"))
import run as R                                           # noqa: E402
import contrast as CT                                     # noqa: E402

dev = "cuda"
OUT = ROOT / "runs" / "111.1"
GRASS, SAND, PATH, LAVA, TABLE, FURNACE = 2, 5, 4, 7, 11, 12
WALK = (GRASS, SAND, PATH, LAVA)
NEAR9 = [r * 9 + c for r in (2, 3, 4) for c in (3, 4, 5)]
SLOT = {"wood": 68, "stone": 69, "coal": 70, "iron": 71}
NEEDS = {11: ({"wood": 1}, (TABLE,)), 12: ({"wood": 1, "stone": 1}, (TABLE,)),
         13: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE)), 14: ({"wood": 1}, (TABLE,)),
         15: ({"wood": 1, "stone": 1}, (TABLE,)), 16: ({"wood": 1, "coal": 1, "iron": 1}, (TABLE, FURNACE))}
STEP = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
P19 = (10, 30, 100, 300)


def made_by_rule(view, counts, a):
    uses, near = NEEDS[a]
    return all(k in view[NEAR9] for k in near) and all(counts[SLOT[m]] >= n for m, n in uses.items())


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1021)                                     # 106's draw of events
    tr, te = R.load(R.TRAIN, "tokens"), R.load(R.TEST, "tokens")
    if FORM == "labels":
        u = np.unique(np.concatenate([tr["lab"].reshape(-1), te["lab"].reshape(-1)]))
        Xtr, Xte, K = np.searchsorted(u, tr["lab"]), np.searchsorted(u, te["lab"]), len(u)
    else:
        Pt = torch.as_tensor(np.load(ROOT / "runs" / "098.3" / "codes_joint_blocked.npz")["protos"], device=dev).float()
        code = lambda V: np.concatenate([torch.cdist(V[s:s + 4096].to(dev).float().reshape(-1, Pt.shape[1]), Pt)
                                         .argmin(1).view(-1, R.NT).cpu().numpy() for s in range(0, len(V), 4096)])
        Xtr, Xte, K = code(tr["V"]), code(te["V"]), len(Pt)
    rep = {"form": FORM, "values": int(K), "smoke": SMOKE}

    # ---- 106's events and truth
    T, act = te["t"], te["act"][te["t"]]
    lab0 = te["lab"][T]
    inv = (lab0[:, 63:79] - 100) % 10
    counts = np.zeros((len(T), R.NT), np.int64)
    counts[:, 63:79] = inv
    made = np.zeros(len(T), bool)
    for s in range(0, len(T), 2048):
        i = np.arange(s, min(s + 2048, len(T)))
        sl = np.array([R.MAKE_SLOT.get(int(a), 73) for a in act[i]])
        lab1 = te["lab"][T[i] + 1]
        made[i] = np.isin(act[i], list(R.MAKE_SLOT)) & (lab1[np.arange(len(i)), sl] != lab0[i, sl])
    d = te["pos_true"][T + 1] - te["pos_true"][T]
    mvm = np.isin(act, R.MOVES)
    blocked = mvm & (d == 0).all(1)
    pick = lambda m: sorted(rng.choice(np.flatnonzero(m), min(PER, int(m.sum())), replace=False))
    evs = {"make (a tool made)": pick(made & (te["src"][T] < 2)), "move, blocked": pick(blocked),
           "move, succeeded": pick(mvm & ~blocked)}

    Ttr, atr = tr["t"], tr["act"][tr["t"]]
    moved_tr = (tr["pos_true"][Ttr + 1] != tr["pos_true"][Ttr]).any(1)
    l0, l1 = tr["lab"][Ttr], tr["lab"][Ttr + 1]
    sltr = np.array([R.MAKE_SLOT.get(int(a), 73) for a in atr])
    occ = {"make (a tool made)": int((np.isin(atr, list(R.MAKE_SLOT)) & (l1[np.arange(len(Ttr)), sltr] != l0[np.arange(len(Ttr)), sltr])).sum()),
           "move, blocked": int((np.isin(atr, R.MOVES) & ~moved_tr).sum()), "move, succeeded": int((np.isin(atr, R.MOVES) & moved_tr).sum())}
    rep["training occurrences"] = occ
    cache = {}

    def situation(key, rows, y):
        if cache.get("key") != key:
            cache["key"], cache["sit"], cache["rows"] = key, CT.Situation(Xtr[Ttr[rows]], y, K), rows
        return cache["sit"], cache["rows"]

    res = {}
    for name, lst in evs.items():
        if SMOKE:
            lst = lst[:30]
        kind = "make" if name.startswith("make") else "move"
        items = []
        for i in lst:
            a = int(act[i])
            view = lab0[i, :63]
            if kind == "make":
                truth0 = made_by_rule(view, counts[i], a)
                tok = R.MAKE_SLOT[a]
            else:
                dx, dy = STEP[a]
                tok = 31 + dx + 9 * dy
                truth0 = tok if view[tok] in WALK else 31
            reps = {int(view[q]) for q in range(63) if q != 31 and view[q] >= 0}
            tc = np.zeros(R.NT, bool)
            for j in range(63):
                if j == 31 or view[j] < 0:
                    continue
                for k in reps:
                    if k == view[j]:
                        continue
                    v2 = view.copy()
                    v2[j] = k
                    tc[j] |= (made_by_rule(v2, counts[i], a) != truth0) if kind == "make" else \
                        ((tok if v2[tok] in WALK else 31) != truth0)
            if kind == "make":
                for j in range(67, 79):
                    if j != tok and inv[i, j - 63] > 0:
                        c2 = counts[i].copy()
                        c2[j] = 0
                        tc[j] |= made_by_rule(view, c2, a) != truth0
            near = np.zeros(R.NT, bool)                                     # 105.4's baseline: the 3 x 3, carried
            near[NEAR9] = True
            near[63:] = True
            near[31] = False
            if kind == "make":
                near[tok] = False
            if kind == "make":
                key = ("make", a, int(Xte[T[i], tok]), int(Xte[T[i] + 1, tok]))
            else:
                key = ("move", a, bool((d[i] != 0).any()))
            items.append(dict(i=i, a=a, tok=tok, kind=kind, key=key, tc=tc, near=near))
        items.sort(key=lambda e: e["key"])

        def rows_y(e):
            rows = np.flatnonzero(atr == e["a"])
            if e["kind"] == "make":
                _, _, c0, c1 = e["key"]
                rows = rows[Xtr[Ttr[rows], e["tok"]] == c0]
                return rows, Xtr[Ttr[rows] + 1, e["tok"]] == c1
            return rows, moved_tr[rows] == e["key"][2]

        def run(tarone, ties, keep_frac=None, seed=0, diag=False):
            c = Counter()
            why, extra, facts = Counter(), Counter(), []
            kept_rows = None if keep_frac is None else np.random.default_rng(seed).random(len(Ttr)) < keep_frac
            judged = np.ones(R.NT, bool)
            for e in items:
                rows, y = rows_y(e)
                sit, rows = situation(e["key"], rows, y)
                keep = None if kept_rows is None else kept_rows[rows]
                ev = Xte[T[e["i"]]]
                tok_of = lambda cell: None if cell == 31 or (e["kind"] == "make" and cell == e["tok"]) else int(cell)
                s = time.monotonic()
                fin = []
                C = CT.greedy(sit, ev, keep, fin, tarone=tarone)
                c["sec"] += time.monotonic() - s
                ans = fin[2]
                right, wrong, found = CT.score_sets(C, tok_of, e["tc"], judged, ties)
                true = e["tc"]
                missed = true & ~found
                c["right"] += right; c["wrong"] += wrong; c["true"] += int(true.sum()); c["found"] += int(found.sum())
                tj = [int(j) for j in np.flatnonzero(true)]
                ok = CT.contrastable(sit, tj, [int(ev[j]) for j in tj], keep)
                for j, o in zip(tj, ok):
                    if o:
                        c["ctrue"] += 1; c["cfound"] += int(found[j])
                c["closed"] += int(ans["closed"])
                c["closed complete"] += int(ans["closed"] and not missed.any())
                if not ans["closed"] and missed.any():
                    nm = set(ans["near misses"])
                    c["missed in open"] += int(missed.sum()); c["missed in near misses"] += sum(int(j in nm) for j in np.flatnonzero(missed))
                for f in C:
                    toks = {tok_of(x) for x in f[7]} - {None}
                    if len(toks) > 1:
                        c["tie sets"] += 1; c["tie members"] += len(toks)
                c["stated"] += len(C)
                if diag:
                    for j in np.flatnonzero(missed):
                        r_ = CT.miss_reason(sit, fin[0], int(j), int(ev[j]))
                        why[r_ if r_ != "enough cases each side" else "enough: " + CT.why_enough(sit, fin[0], int(j), int(ev[j]))] += 1
                    for f in C:
                        toks = {tok_of(x) for x in (f[7] if ties else [f[0]])} - {None}
                        if toks and not any(true[list(toks)]):
                            j = min(toks)
                            extra[f"label {lab0[e['i'], j]} at token {j}"] += 1
                    facts.append([(f[0], f[1], "=" if f[2] else "≠", f[3], len({tok_of(x) for x in f[7]} - {None})) for f in C])
            pr = lambda x, y: round(x / y, 4) if y else None
            n = max(len(items), 1)
            out = {"precision": pr(c["right"], c["right"] + c["wrong"]), "recall": pr(c["found"], c["true"]),
                   "contrastable share": pr(c["ctrue"], c["true"]), "recall on contrastable": pr(c["cfound"], c["ctrue"]),
                   "closed share": round(c["closed"] / n, 4),
                   "closed and complete, of closed": pr(c["closed complete"], c["closed"]),
                   "missed conditions in open answers": c["missed in open"],
                   "of those, among the near misses (top 10)": pr(c["missed in near misses"], c["missed in open"]),
                   "stated conditions per event": round(c["stated"] / n, 2),
                   "tie sets (share of stated), mean size": [pr(c["tie sets"], c["stated"]), pr(c["tie members"], c["tie sets"])],
                   "seconds per event": round(c["sec"] / n, 4)}
            if diag:
                out["missed true conditions, by reason (final subset)"] = dict(why)
                out["false conditions: evaluator label at token, top 8"] = dict(extra.most_common(8))
                out["facts of the first 5 events (token, value, = or ≠, power, tie size)"] = facts[:5]
            return out

        r = {"events": len(items), "true conditions per event (mean)": round(float(np.mean([e["tc"].sum() for e in items])), 2)}
        btp = sum(int((e["near"] & e["tc"]).sum()) for e in items)
        bfp = sum(int((e["near"] & ~e["tc"]).sum()) for e in items)
        bfn = sum(int((~e["near"] & e["tc"]).sum()) for e in items)
        r["baseline 'the 3 x 3 and every carried slot': precision, recall"] = [round(btp / max(btp + bfp, 1), 4),
                                                                               round(btp / max(btp + bfn, 1), 4)]
        r["(a) log space only"] = run(False, False)
        r["(a)-(d)"] = run(True, True, diag=True)
        print(name, json.dumps({k: r["(a)-(d)"][k] for k in ("precision", "recall", "recall on contrastable",
                                                              "closed share", "closed and complete, of closed")}),
              "| (a) only", r["(a) log space only"]["precision"], r["(a) log space only"]["recall"],
              f"({time.monotonic() - t0:.0f} s)", flush=True)
        p19 = {}
        for n in (() if NOP19 else P19):
            if n < occ[name]:
                o = run(True, True, n / occ[name], seed=n)
                p19[str(n)] = {k: o[k] for k in ("precision", "recall", "recall on contrastable", "closed and complete, of closed")}
        r["P19 ((a)-(d)), memory holding about n occurrences"] = p19
        print("  P19", json.dumps(p19), flush=True)
        res[name] = r
    rep["conditions"] = res
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"crafter_{FORM}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
