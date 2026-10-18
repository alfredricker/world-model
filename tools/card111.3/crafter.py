"""Card 111.3 in Crafter: closing by co-occurring changes, on 111.2's events (seed 1112) and memory.

For each event e: 111.2's answer. Its co-changes are the other tokens whose code changes in the same step, compared
in the world's frame (the next view shifted by the agent's own pose displacement; carried slots stay). Each is asked
of memory as its own event (same action, that token's code; success: its next code). A fact a co-change needs (its
kept facts, and its own code) that varies in e's memory set but never among e's final cases is doubted; closed =
111.2's closed and no doubt. The agent's own token (31) and the made tool's slot are not named.

  bin/prun python tools/card111.3/crafter.py codes|labels [--smoke] → runs/111.3/crafter_<form>.json
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
sys.path.insert(0, str(ROOT / "tools" / "card111.2"))  # 111.2's contrast, unchanged
import run as R                                           # noqa: E402
import contrast as CT                                     # noqa: E402

dev = "cuda"
OUT = ROOT / "runs" / "111.3"
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
    rng = np.random.default_rng(1112)                 # a new draw (111.2); makes are all 236 either way
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

    def aligned_next(X, Ts, disp):
        """Codes at t+1 placed in t's frame by the pose's displacement (dx, dy); -1 where not visible. The view is
        9 x 7 around the agent; carried slots (63-80) do not move."""
        out = np.full((len(Ts), R.NT), -1, np.int64)
        out[:, 63:] = X[Ts + 1, 63:]
        r, cc = np.divmod(np.arange(63), 9)
        for dx, dy in {tuple(x) for x in disp.tolist()}:
            k = (disp[:, 0] == dx) & (disp[:, 1] == dy)
            r2, c2 = r - dy, cc - dx
            ok = (r2 >= 0) & (r2 < 7) & (c2 >= 0) & (c2 < 9)
            src = np.where(ok, r2 * 9 + c2, 0)
            v = X[Ts[k] + 1][:, src]
            out[np.ix_(np.flatnonzero(k), np.arange(63))] = np.where(ok[None], v, -1)
        return out

    NEXT_TR = aligned_next(Xtr, Ttr, tr["pos_true"][Ttr + 1] - tr["pos_true"][Ttr])
    NEXT_TE = aligned_next(Xte, T, d)
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

        c = Counter()
        examples = []
        tok_of_e = lambda e: (lambda cell: None if cell == 31 or (e["kind"] == "make" and cell == e["tok"]) else int(cell))
        recs = []
        for e in items:                                        # pass 1: each event's own answer
            rows, y = rows_y(e)
            sit, rows = situation(e["key"], rows, y)
            fin = []
            C = CT.greedy(sit, Xte[T[e["i"]]], None, fin)
            right, wrong, found = CT.score_sets(C, tok_of_e(e), e["tc"], np.ones(R.NT, bool), True)
            true = e["tc"]
            c["right"] += right; c["wrong"] += wrong; c["true"] += int(true.sum()); c["found"] += int(found.sum())
            recs.append(dict(e=e, C=C, m=fin[0].bool().cpu().numpy(), ans=fin[2], found=found, true=true,
                             missed=true & ~found, kept={x for f in C for x in f[7]}, need={}))
        queries = []                                           # pass 2: co-changes, in the world's frame (pose)
        for r_ in recs:
            e = r_["e"]
            st = T[e["i"]]
            nxt = NEXT_TE[e["i"]]
            for p in np.flatnonzero((nxt >= 0) & (nxt != Xte[st])):
                if p != 31 and p != e["tok"]:
                    queries.append(((e["a"], int(p), int(Xte[st, p]), int(nxt[p])), r_))
        queries.sort(key=lambda q: q[0])
        co_cache = {}
        for (a, p, c0p, c1p), r_ in queries:
            key = (a, p, c0p, c1p)
            if co_cache.get("key") != key:
                rows = np.flatnonzero((atr == a) & (Xtr[Ttr, p] == c0p) & (NEXT_TR[:, p] >= 0))
                co_cache["key"] = key
                co_cache["sit"] = CT.Situation(Xtr[Ttr[rows]], NEXT_TR[rows, p] == c1p, K) if len(rows) >= 2 * CT.N_MIN else None
            s2 = co_cache["sit"]
            facts = [(p, c0p, True)]                           # its own code
            if s2 is not None:
                facts += [(f[0], f[1], f[2]) for f in CT.greedy(s2, Xte[T[r_["e"]["i"]]], None, [])]
            for cell, val, is_eq in facts:
                if cell in r_["kept"] or tok_of_e(r_["e"])(cell) is None:
                    continue
                r_["need"][(int(cell), int(val), bool(is_eq))] = True
            c["co-changes asked"] += 1
        for r_ in recs:                                        # pass 3: doubts (varies in e's memory set, never in its final cases)
            e, ans, missed = r_["e"], r_["ans"], r_["missed"]
            rows, y = rows_y(e)
            sit, rows = situation(e["key"], rows, y)
            m = torch.as_tensor(r_["m"], device=dev)
            doubts = []
            for cell, val, is_eq in r_["need"]:
                col = sit.M[:, cell * sit.K + val].bool()
                holds = col if is_eq else ~col
                if 0 < int(holds.sum()) < sit.n and not (m & ~holds).any():
                    doubts.append((cell, val, is_eq))
            dt = {d_[0] for d_ in doubts}
            closed_old = ans["closed"]
            closed_new = closed_old and not doubts
            c["closed (111.2)"] += int(closed_old); c["closed (111.2) complete"] += int(closed_old and not missed.any())
            c["closed"] += int(closed_new); c["closed complete"] += int(closed_new and not missed.any())
            if closed_old and not closed_new:
                c["reopened"] += 1; c["doubts in reopened"] += len(doubts)
                c["missed in reopened"] += int(missed.sum())
                c["missed in reopened, doubted"] += sum(int(j in dt) for j in np.flatnonzero(missed))
                if len(examples) < 4:
                    examples.append({"event index": int(e["i"]), "missed tokens": [int(j) for j in np.flatnonzero(missed)],
                                     "doubted (token, label at t, code, =)": [[int(cl), int(lab0[e["i"], cl]), int(v), q] for cl, v, q in doubts][:6]})
            if closed_old and not missed.any():
                c["complete (111.2)"] += 1; c["complete (111.2), still closed"] += int(closed_new)
            if closed_old:
                vals = torch.as_tensor(Xte[T[e["i"]]], device=dev)
                cols = torch.arange(R.NT, device=dev) * sit.K + vals
                unvaried = ~((m[:, None] & (sit.M[:, cols] == 0)).any(0))
                for x in r_["kept"] | {31, e["tok"]}:
                    unvaried[x] = False
                c["closed under 'doubt every unvaried fact'"] += int(not bool(unvaried.any()))
            add = np.zeros(R.NT, bool)
            for cl, _, _ in r_["need"]:
                add[cl] = True
            c["stated + co-change facts: found"] += int(((r_["found"] | add) & r_["true"]).sum())
            c["stated + co-change facts: false added"] += int((add & ~r_["true"] & ~r_["found"]).sum())
        pr = lambda x, y: round(x / y, 4) if y else None
        n = len(items)
        r = {"events": n, "precision": pr(c["right"], c["right"] + c["wrong"]), "recall": pr(c["found"], c["true"]),
             "closed share (111.2)": pr(c["closed (111.2)"], n),
             "closed and complete, of closed (111.2)": pr(c["closed (111.2) complete"], c["closed (111.2)"]),
             "closed share": pr(c["closed"], n), "closed events": c["closed"],
             "closed and complete, of closed (1)": pr(c["closed complete"], c["closed"]),
             "reopened": c["reopened"], "missed in reopened": c["missed in reopened"],
             "missed in reopened, doubted (2)": pr(c["missed in reopened, doubted"], c["missed in reopened"]),
             "doubted facts per reopened answer (2)": pr(c["doubts in reopened"], c["reopened"]),
             "complete under 111.2": c["complete (111.2)"],
             "complete under 111.2, still closed (3)": pr(c["complete (111.2), still closed"], c["complete (111.2)"]),
             "closed share under 'doubt every unvaried fact'": pr(c["closed under 'doubt every unvaried fact'"], n),
             "co-changes asked per event": pr(c["co-changes asked"], n),
             "diagnostic: co-change facts stated: recall, false added": [pr(c["stated + co-change facts: found"], c["true"]),
                                                                         c["stated + co-change facts: false added"]],
             "reopened examples": examples}
        print(name, {k: r[k] for k in ("closed and complete, of closed (111.2)", "closed and complete, of closed (1)", "closed events",
                                       "missed in reopened, doubted (2)", "doubted facts per reopened answer (2)",
                                       "complete under 111.2, still closed (3)", "co-changes asked per event")},
              f"({time.monotonic() - t0:.0f} s)", flush=True)
        res[name] = r
    rep["conditions"] = res
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"crafter_{FORM}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
