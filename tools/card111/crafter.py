"""Card 111 in Crafter: conditions of 106's events by contrast over memory (tools/card111/contrast.py).

Memory: card 105.10's training transitions, every token's value (perception's code, form "codes"; or the evaluator's
label, form "labels": the gate's upper bound). The view follows the agent, so a token's index is its offset from the
agent's own token; carried tokens sit at their fixed places. Situations:
- make (a tool made): steps with the same action whose tool slot holds e's code; success when it holds e's next code;
- moves (blocked, succeeded): steps with the same action; success when the agent's own pose changed as in e (moved or
  not; proprioception, GOAL C1).
The truth is 106's (Crafter's rule, data.yaml, under one replacement). The agent's own token (31) and the tool's slot
are not named.

  bin/prun python tools/card111/crafter.py codes|labels [--per 300] [--smoke] → runs/111/crafter_<form>.json
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
sys.argv = ["run.py", "--origins", "pose"]
sys.path.insert(0, str(ROOT / "tools" / "card105.2"))
sys.path.insert(0, str(ROOT / "tools" / "card111"))
import run as R                                           # noqa: E402
import contrast as CT                                     # noqa: E402

dev = "cuda"
OUT = ROOT / "runs" / "111"
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

        why, extra = Counter(), Counter()

        def run(keep_frac=None, seed=0):
            tp = fp = fn = exact = 0
            facts, sec = [], 0.0
            kept_rows = None if keep_frac is None else np.random.default_rng(seed).random(len(Ttr)) < keep_frac
            for e in items:
                rows, y = rows_y(e)
                sit, rows = situation(e["key"], rows, y)
                keep = None if kept_rows is None else kept_rows[rows]
                s = time.monotonic()
                fin = []
                C = CT.greedy(sit, Xte[T[e["i"]]], keep, fin)
                sec += time.monotonic() - s
                mc = np.zeros(R.NT, bool)
                for cell, *_ in C:
                    if cell != 31 and not (e["kind"] == "make" and cell == e["tok"]):
                        mc[cell] = True
                tp += int((mc & e["tc"]).sum()); fp += int((mc & ~e["tc"]).sum()); fn += int((~mc & e["tc"]).sum())
                exact += int((mc == e["tc"]).all())
                if kept_rows is None:
                    for j in np.flatnonzero(~mc & e["tc"]):
                        why[CT.miss_reason(sit, fin[0], int(j), int(Xte[T[e["i"]], j]))] += 1
                    for j in np.flatnonzero(mc & ~e["tc"]):
                        extra[f"label {lab0[e['i'], j]} at token {j}"] += 1
                facts.append([(cell, val, "=" if q else "≠", pw) for cell, val, q, pw, *_ in C])
            pr = lambda x, y: round(x / max(x + y, 1), 4)
            return {"precision": pr(tp, fp), "recall": pr(tp, fn), "events exactly right": round(exact / max(len(items), 1), 4),
                    "seconds per event": round(sec / max(len(items), 1), 4)}, facts

        r, facts = run()
        btp = sum(int((e["near"] & e["tc"]).sum()) for e in items)
        bfp = sum(int((e["near"] & ~e["tc"]).sum()) for e in items)
        bfn = sum(int((~e["near"] & e["tc"]).sum()) for e in items)
        r["events"] = len(items)
        r["baseline 'the 3 x 3 and every carried slot': precision, recall"] = [round(btp / max(btp + bfp, 1), 4),
                                                                               round(btp / max(btp + bfn, 1), 4)]
        r["missed true conditions, by reason (final subset)"] = dict(why)
        r["false conditions: evaluator label at token, top 8"] = dict(extra.most_common(8))
        r["true conditions per event (mean)"] = round(float(np.mean([e["tc"].sum() for e in items])), 2)
        r["facts of the first 5 events (token, value, = or ≠, power)"] = facts[:5]
        print(name, json.dumps({k: r[k] for k in ("precision", "recall", "events exactly right", "seconds per event")}),
              f"({time.monotonic() - t0:.0f} s)", flush=True)
        p19 = {}
        for n in P19:
            if n < occ[name]:
                o, _ = run(n / occ[name], seed=n)
                p19[str(n)] = [o["precision"], o["recall"]]
        r["P19: precision, recall with memory holding about n occurrences"] = p19
        print("  P19", p19, flush=True)
        res[name] = r
    rep["conditions"] = res
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"crafter_{FORM}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
