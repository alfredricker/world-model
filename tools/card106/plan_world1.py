"""Card 106, criterion 3: card 105.5's planner in breaking world 1, its statements read in sleep from the remembered
training events, with or without filter K (a replaced state counts only where an independently trained copy predicts
the same next code; the copy is 105.6's seed 7). Scored on the 200 test worlds of 105.5 and on the solvable ones: a
world is solvable when breadth-first search on the world's own rule lights the lamp within 30 steps (scoring only).

  bin/prun python tools/card106/plan_world1.py none|K      → runs/106/plan_world1_<arm>.json (and solvable_world1.json)
"""
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FILTER = sys.argv[1]
sys.argv = ["world1.py", "model", "--episodes", "200"]
spec = importlib.util.spec_from_file_location("c1055w1", ROOT / "tools" / "card105.5" / "world1.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
OUT = ROOT / "runs" / "106"
NET1 = P.NET
NET2 = P.TransitionModel(P.VTAB.shape[1], 5, arm="Rb").to(P.dev)
NET2.load_state_dict(torch.load(ROOT / "runs" / "105.6" / "tokens_Rb_s7_net.pt"))
NET2.eval()


def predict2(S, U):
    P.NET = NET2
    try:
        return P.predict(S, U)
    finally:
        P.NET = NET1


def learn_statements(per=150):
    """105.5's statements, with filter K on the events and on every replaced state."""
    z = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    sem, act, ep, done = z["sem"].reshape(-1, 63), z["act"].astype(int), z["ep"], z["done"]
    codes = np.vectorize(P.CODE_OF_KIND.get)(sem)
    T = np.flatnonzero(np.r_[ep[1:] == ep[:-1], False] & ~done & np.isin(act, list(P.MOVES)))
    rng = np.random.default_rng(1021)
    rng.shuffle(T)
    T = T[:6000]
    pred = P.predict(codes[T], act[T])
    pred2 = predict2(codes[T], act[T]) if FILTER == "K" else pred
    by_type = defaultdict(list)
    for n, t in enumerate(T):
        for i in np.flatnonzero(pred[n] != codes[t]):
            if pred2[n, i] != pred[n, i]:
                continue                                       # the copies disagree on the event: not known
            key = (int(codes[t, i]), int(pred[n, i]), int(act[t]))
            if len(by_type[key]) < per:
                by_type[key].append((t, int(i), int(pred[n, i])))
    st = defaultdict(Counter)
    for (a, b, u), evs in by_type.items():
        for t, i, b_ in evs:
            s = codes[t]
            present = sorted(set(s.tolist()) - {P.AGENT})
            var, js = [], []
            for j in range(63):
                if j == i:
                    continue
                for k in ([P.CODE_OF_KIND[P.WD.FLOOR]] if s[j] == P.AGENT else [k for k in present if k != s[j]]):
                    v = s.copy()
                    v[j] = k
                    var.append(v)
                    js.append(j)
            p = P.predict(np.array(var), np.full(len(var), u))[:, i]
            keep = predict2(np.array(var), np.full(len(var), u))[:, i] == p if FILTER == "K" else np.ones(len(p), bool)
            cond = sorted({j for j, x, kp in zip(js, p, keep) if x != b_ and kp})
            ix, iy = i % 9, i // 9
            st[(a, b)][(u, tuple((j % 9 - ix, j // 9 - iy, int(s[j])) for j in cond))] += 1
    return {k: [(u, conds, n) for (u, conds), n in c.most_common() if n >= 2] for k, c in st.items()}


def solvable():
    path = OUT / "solvable_world1.json"
    if path.exists():
        return json.loads(path.read_text())
    res = []
    for e in range(200):
        rng = np.random.default_rng(10_000 + e)
        while True:
            env = P.WD.World1(rng, test=True)
            lamps = np.flatnonzero(env.grid().reshape(-1) == P.WD.LAMP)
            if len(lamps):
                break
        L, g0 = int(lamps[0]), env.grid()
        seen, dq, ok = {g0.tobytes()}, deque([(g0, 0)]), False
        while dq:
            g, d = dq.popleft()
            if g.reshape(-1)[L] == P.WD.LIT:
                ok = True
                break
            if d >= 30:
                continue
            for u in (1, 2, 3, 4):
                g2 = P.true_next(g, u)
                if g2.tobytes() not in seen:
                    seen.add(g2.tobytes())
                    dq.append((g2, d + 1))
        res.append(ok)
    path.write_text(json.dumps(res) + "\n")
    return res


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    sol = np.array(solvable())
    t1 = time.monotonic()
    P.STATEMENTS = learn_statements()
    t_learn = time.monotonic() - t1
    res = [P.episode(e) for e in range(200)]
    ok = np.array([r[0] for r in res])
    rep = {"filter": FILTER, "worlds": 200, "solvable (breadth-first search, 30 steps)": int(sol.sum()),
           "lamp lit within 50 steps, all worlds": round(float(ok.mean()), 4),
           "lamp lit, solvable worlds": f"{int(ok[sol].sum())} of {int(sol.sum())} ({ok[sol].mean():.3f})",
           "statements": sum(len(v) for v in P.STATEMENTS.values()),
           "lamp-lighting statements": [(u, list(c), n) for u, c, n in
                                        P.STATEMENTS.get((P.CODE_OF_KIND[P.WD.LAMP], P.CODE_OF_KIND[P.WD.LIT]), [])],
           "seconds to read statements": round(t_learn, 1), "seconds": round(time.monotonic() - t0, 1)}
    (OUT / f"plan_world1_{FILTER}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "lamp-lighting statements"}, indent=1))


if __name__ == "__main__":
    main()
