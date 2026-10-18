"""Card 111.3 in breaking world 1: closing by co-occurring changes.

Memory: card 102's training set ("train"), or that plus 111.2's evaluation rooms ("evalrooms", card 112's gate).
Events and truth: card 112's (111.2's events; the lamp rule and the room's border). For each event e: 111.2's
answer (tools/card111.2/contrast.py). The co-changes of e (other tokens that change in the same step) are asked
of memory as their own events; a fact a co-change needs (its kept facts, and its own code) that varies somewhere in e's memory set but
never among e's final cases is doubted. Closed = 111.2's closed and no doubt.

  bin/prun python tools/card111.3/world1.py train|evalrooms [--smoke] → runs/111.3/world1_<memory>.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MEMORY = sys.argv[1]
SMOKE = "--smoke" in sys.argv
ONLY = next((sys.argv[k + 1] for k in range(len(sys.argv) - 1) if sys.argv[k] == "--kind"), None)
sys.argv = ["explore.py"] + (["--smoke"] if SMOKE else [])
sys.path.insert(0, str(ROOT / "tools" / "card112"))
import explore as X                                        # noqa: E402  (card 112: memory, events, truth)

CT = X.CT
OUT = ROOT / "runs" / "111.3"
W, H, WW, DY, DX = X.W, X.H, X.WW, X.DY, X.DX
WH = 2 * DY + 1


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    ztr = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    valid = np.r_[ztr["ep"][1:] == ztr["ep"][:-1], False] & ~ztr["done"]
    mem = X.Memory(ztr["sem"].reshape(-1, 63).astype(np.int64), ztr["act"].astype(np.int64), valid, 111)
    z = np.load(ROOT / "runs" / "111.2" / "world1_test_fresh.npz")
    sem = z["sem"].reshape(-1, 63).astype(np.int64)
    if MEMORY == "evalrooms":
        mem.new_sem.append(sem); mem.new_act.append(z["act"].astype(np.int64))
        mem.new_valid.append(np.r_[z["ep"][1:] == z["ep"][:-1], False] & ~z["done"])
        mem.refresh()
    ITEMS = X.eval_items()
    res = {}
    for kind, items in ITEMS.items():
        if ONLY and kind != ONLY:
            continue
        c = Counter()
        examples = []
        # pass 1: each event's own answer
        recs = []
        for e in items:
            t, i = e["t"], e["i"]
            y0, x0 = i // W, i % W
            sit = mem.situation(e["a"], e["c0"], e["c1"])
            tok_of = (lambda y0, x0, i: lambda cell: (lambda y, x: y * W + x if 0 <= y < H and 0 <= x < W and y * W + x != i
                                                       else None)(y0 + cell // WW - DY, x0 + cell % WW - DX))(y0, x0, i)
            fin = []
            C = CT.greedy(sit, e["vals"], None, fin)
            right, wrong, found = CT.score_sets(C, tok_of, e["tc"], e["judged"], True)
            true = e["tc"] & e["judged"]
            c["right"] += right; c["wrong"] += wrong; c["true"] += int(true.sum()); c["found"] += int(found.sum())
            recs.append(dict(e=e, y0=y0, x0=x0, tok_of=tok_of, C=C, m=fin[0].bool().cpu().numpy(), ans=fin[2],
                             found=found, true=true, missed=true & ~found, kept={x for f in C for x in f[7]}, need={}))
        # pass 2: the co-changes, grouped by their kind so each memory set is built once
        queries = []
        for r_ in recs:
            t, i = r_["e"]["t"], r_["e"]["i"]
            for j in np.flatnonzero(sem[t + 1] != sem[t]):
                if j != i:
                    queries.append(((r_["e"]["a"], int(sem[t, j]), int(sem[t + 1, j])), r_, int(j)))
        queries.sort(key=lambda q: q[0])
        for (a, c0j, c1j), r_, j in queries:
            t = r_["e"]["t"]
            s2 = mem.situation(a, c0j, c1j)
            v2 = X.windows(sem, np.array([t]), np.array([j]))[0]
            C2 = CT.greedy(s2, v2, None, [])
            yj, xj = j // W, j % W
            for cell2, val, is_eq in [(f[0], f[1], f[2]) for f in C2] + [(DY * WW + DX, c0j, True)]:     # + its own code
                yy, xx = yj + cell2 // WW - DY, xj + cell2 % WW - DX                                    # place in the room
                dy, dx = yy - r_["y0"], xx - r_["x0"]                                                     # offset from e's token
                if abs(dy) > DY or abs(dx) > DX:
                    continue
                cl = int((dy + DY) * WW + (dx + DX))
                if cl in r_["kept"] or cl == DY * WW + DX:
                    continue
                r_["need"][(cl, int(val), bool(is_eq))] = True
            c["co-changes asked"] += 1
        # pass 3: doubts. A needed fact is doubted if it varies somewhere in e's memory set (it can be otherwise) but
        # never among e's final cases (changed after the smoke test: facts constant in all of memory are background)
        for r_ in recs:
            e, m, ans, missed = r_["e"], torch.as_tensor(r_["m"], device=CT.dev), r_["ans"], r_["missed"]
            sit = mem.situation(e["a"], e["c0"], e["c1"])
            doubts = []
            for cl, val, is_eq in r_["need"]:
                col = sit.M[:, cl * sit.K + val].bool()
                holds = col if is_eq else ~col
                n_hold = int(holds.sum())
                if 0 < n_hold < sit.n and not (m & ~holds).any():
                    doubts.append((cl, val, is_eq))
            doubt_toks = {r_["tok_of"](d[0]) for d in doubts} - {None}
            closed_old = ans["closed"]
            closed_new = closed_old and not doubts
            c["closed (111.2)"] += int(closed_old); c["closed (111.2) complete"] += int(closed_old and not missed.any())
            c["closed"] += int(closed_new); c["closed complete"] += int(closed_new and not missed.any())
            y0, x0 = r_["y0"], r_["x0"]
            if closed_old and not closed_new:
                c["reopened"] += 1; c["doubts in reopened"] += len(doubts)
                c["missed in reopened"] += int(missed.sum())
                c["missed in reopened, doubted"] += sum(int(j in doubt_toks) for j in np.flatnonzero(missed))
                if len(examples) < 4:
                    examples.append({"event": [int(y0), int(x0), e["a"]],
                                     "missed": [[int(j // W - y0), int(j % W - x0)] for j in np.flatnonzero(missed)],
                                     "doubted facts (dy, dx, value, =)": [[int(cl // WW - DY), int(cl % WW - DX), v, q] for cl, v, q in doubts]})
            if closed_old and not missed.any():
                c["complete (111.2)"] += 1; c["complete (111.2), still closed"] += int(closed_new)
            if closed_old:                       # baseline: doubt every fact of e never varied among its final cases
                vals = torch.as_tensor(e["vals"], device=CT.dev)
                cols = torch.arange(len(e["vals"]), device=CT.dev) * sit.K + vals
                unvaried = ~((m[:, None] & (sit.M[:, cols] == 0)).any(0))
                for x in r_["kept"] | {DY * WW + DX}:
                    unvaried[x] = False
                inside = torch.as_tensor([r_["tok_of"](x) is not None for x in range(len(e["vals"]))], device=CT.dev)
                c["closed under 'doubt every unvaried fact'"] += int(not bool((unvaried & inside).any()))
            add = np.zeros(63, bool)             # diagnostic: the co-changes' needed facts stated as conditions
            for cl, _, _ in r_["need"]:
                tk = r_["tok_of"](cl)
                if tk is not None:
                    add[tk] = True
            c["stated + co-change facts: found"] += int(((r_["found"] | add) & r_["true"]).sum())
            c["stated + co-change facts: false added"] += int((add & ~r_["true"] & e["judged"] & ~r_["found"]).sum())
        pr = lambda x, y: round(x / y, 4) if y else None
        n = len(items)
        res[kind] = {"events": n, "precision": pr(c["right"], c["right"] + c["wrong"]), "recall": pr(c["found"], c["true"]),
                     "closed share (111.2)": pr(c["closed (111.2)"], n),
                     "closed and complete, of closed (111.2)": pr(c["closed (111.2) complete"], c["closed (111.2)"]),
                     "closed share": pr(c["closed"], n), "closed events": c["closed"],
                     "closed and complete, of closed (1)": pr(c["closed complete"], c["closed"]),
                     "reopened": c["reopened"],
                     "missed in reopened, doubted (2)": pr(c["missed in reopened, doubted"], c["missed in reopened"]),
                     "missed in reopened": c["missed in reopened"],
                     "doubted facts per reopened answer (2)": pr(c["doubts in reopened"], c["reopened"]),
                     "complete under 111.2": c["complete (111.2)"],
                     "complete under 111.2, still closed (3)": pr(c["complete (111.2), still closed"], c["complete (111.2)"]),
                     "closed share under 'doubt every unvaried fact'": pr(c["closed under 'doubt every unvaried fact'"], n),
                     "co-changes asked per event": pr(c["co-changes asked"], n),
                     "diagnostic: co-change facts stated: recall, false added": [pr(c["stated + co-change facts: found"], c["true"]),
                                                                                 c["stated + co-change facts: false added"]],
                     "reopened examples (lamp row, col, action; offsets)": examples}
        r = res[kind]
        print(kind, {k: r[k] for k in ("closed and complete, of closed (111.2)", "closed and complete, of closed (1)", "closed events",
                                       "missed in reopened, doubted (2)", "doubted facts per reopened answer (2)",
                                       "complete under 111.2, still closed (3)")}, f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep = {"memory": MEMORY, "smoke": SMOKE, "conditions": res, "seconds": round(time.monotonic() - t0, 1)}
    (OUT / f"world1_{MEMORY}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
