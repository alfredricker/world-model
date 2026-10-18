"""Card 112: experiment on open answers in breaking world 1.

The agent plays new test rooms (card 102's generator, lamps in columns 5-7). Arms:
- none:      no exploration (memory = card 102's training set);
- random:    each step an action at random (card 102's policy);
- directed:  whenever a token changes code (a, c0 -> c1), memory is asked for the conditions (card 111.2's contrast);
             an open answer becomes a question (its kept facts, anchored on the token). Each step, if a token with code
             c0 satisfies a question's kept facts, take the question's action (fewest remembered cases first);
             otherwise act at random. A question is not tried again in a state it was tried in (a repeated try
             adds no new case; changed after the smoke test, 2026-10-18). Every REBUILD steps memory takes in the new steps, every question is asked
             again, and closed ones are dropped. A kind of change whose last 20 answers were all closed is asked
             1 time in 20 (the cost rule);
- evalrooms: the gate's upper bound: memory also holds every step of 111.2's evaluation rooms (scoring only).
After exploration (and at checkpoints), 111.2's events are scored against memory, with 111.2's truth and the room's
border added to the check for impossible states.

  bin/prun python tools/card112/explore.py --arm directed --seed 0 --steps 10000 [--smoke] → runs/112/<arm>_s<seed>_<steps>.json
"""
import importlib.util
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
ARM = arg("--arm", "random")
SEED = int(arg("--seed", 0))
STEPS = int(arg("--steps", 10000))
SMOKE = "--smoke" in sys.argv
sys.path.insert(0, str(ROOT / "tools" / "card111.2"))
sys.path.insert(0, str(ROOT / "tools" / "card102"))
import contrast as CT                                      # noqa: E402
import worlds as WD                                        # noqa: E402

spec = importlib.util.spec_from_file_location("c1054w1", ROOT / "tools" / "card105.4" / "world1.py")
W1 = importlib.util.module_from_spec(spec)
sys.argv = ["world1.py"]
spec.loader.exec_module(W1)

OUT = ROOT / "runs" / "112"
NT, W, H = 81, 9, 7
FLOOR, WALL, AGENT, BOX, LAMP, LIT = range(6)
K = 6                                                       # kinds; K is PAD outside the room
DX, DY = W - 1, H - 1
WW = 2 * DX + 1
MAX_ITEMS = 200_000
REBUILD = 500
EP_LEN = 100
CHECKS = (2500, 5000, 10000, 50000)
LAMP_KINDS = ("lamp lights", "lamp goes out")


def windows(X, t, i):
    """Values (n, 13 * 17) around tiles i of steps t; K outside the room."""
    y0, x0 = i // W, i % W
    oy, ox = np.meshgrid(np.arange(-DY, DY + 1), np.arange(-DX, DX + 1), indexing="ij")
    yy, xx = y0[:, None] + oy.reshape(-1)[None], x0[:, None] + ox.reshape(-1)[None]
    inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    v = X[t[:, None], np.clip(yy, 0, H - 1) * W + np.clip(xx, 0, W - 1)]
    return np.where(inside, v, K)


class Memory:
    """Stored steps (kinds, actions) and the transitions between them; situations built on demand, a few cached."""

    def __init__(self, sem, act, valid, seed):
        self.sem, self.act, self.valid = sem, act, valid
        self.new_sem, self.new_act, self.new_valid = [], [], []
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.refresh()

    def add_episode(self, grids, acts):
        n = len(acts)
        self.new_sem.append(np.asarray(grids, np.int64).reshape(n, 63))
        self.new_act.append(np.asarray(acts, np.int64))
        v = np.ones(n, bool)
        v[-1] = False                                       # the last step's outcome is not stored (card 102)
        self.new_valid.append(v)

    def refresh(self):
        if self.new_sem:
            self.sem = np.concatenate([self.sem] + self.new_sem)
            self.act = np.concatenate([self.act] + self.new_act)
            self.valid = np.concatenate([self.valid] + self.new_valid)
            self.new_sem, self.new_act, self.new_valid = [], [], []
        self.T = np.flatnonzero(self.valid)
        self.cache = {}

    def situation(self, a, c0, c1):
        key = (a, c0)
        if key not in self.cache:
            if len(self.cache) >= 3:
                self.cache.pop(next(iter(self.cache)))
            ts = self.T[self.act[self.T] == a]
            tt, ii = np.nonzero(self.sem[ts] == c0)
            tt = ts[tt]
            if len(tt) > MAX_ITEMS:
                s = np.sort(self.rng.choice(len(tt), MAX_ITEMS, replace=False))
                tt, ii = tt[s], ii[s]
            self.cache[key] = {"vals": windows(self.sem, tt, ii), "next": self.sem[tt + 1, ii]}
        c = self.cache[key]
        if c.get("c1") != c1:
            c["c1"], c["sit"] = c1, CT.Situation(c["vals"], c["next"] == c1, K)
        return c["sit"]


def producible(g):
    """The world's invariants: every lamp lit exactly when a box stands beside it, and the room's border is wall."""
    if not ((g[0] == WALL).all() and (g[-1] == WALL).all() and (g[:, 0] == WALL).all() and (g[:, -1] == WALL).all()):
        return False
    for y, x in zip(*np.nonzero((g == LAMP) | (g == LIT))):
        near = any(0 <= y + dy < H and 0 <= x + dx < W and g[y + dy, x + dx] == BOX for dx, dy in W1.MOVES.values())
        if near != (g[y, x] == LIT):
            return False
    return True


def eval_items():
    """111.2's events (its fresh rooms, 106's draw) and their truth, with the border added to the check."""
    z = np.load(ROOT / "runs" / "111.2" / "world1_test_fresh.npz")
    W1.PER = 300
    ev = W1.events(z, np.random.default_rng(1021))
    sem, act = z["sem"].reshape(-1, 63).astype(np.int64), z["act"].astype(np.int64)
    out = {}
    for kind, lst in ev.items():
        if SMOKE:
            lst = lst[:30]
        items = []
        for t, i in lst:
            a = int(act[t])
            g = sem[t].reshape(H, W)
            base = W1.true_next(g, a).reshape(-1)[i]
            present = [k for k in (FLOOR, WALL, BOX, LAMP, LIT) if (sem[t] == k).any()]
            tc, judged = np.zeros(63, bool), np.zeros(63, bool)
            for j in range(63):
                if j == i:
                    continue
                for k in ([FLOOR] if sem[t, j] == AGENT else [k for k in present if k != sem[t, j]]):
                    g2 = g.copy().reshape(-1)
                    g2[j] = k
                    g2 = g2.reshape(H, W)
                    if producible(g2):
                        judged[j] = True
                        tc[j] |= W1.true_next(g2, a).reshape(-1)[i] != base
            for j in range(63):
                if j != i and not judged[j] and sem[t + 1, j] != sem[t, j]:
                    judged[j] = tc[j] = True
            items.append(dict(t=t, i=i, a=a, c0=int(sem[t, i]), c1=int(sem[t + 1, i]), tc=tc, judged=judged,
                              vals=windows(sem, np.array([t]), np.array([i]))[0]))
        items.sort(key=lambda e: (e["a"], e["c0"], e["c1"]))
        out[kind] = items
    return out


def evaluate(mem, ITEMS, kinds):
    res = {}
    for kind in kinds:
        c = Counter()
        for e in ITEMS[kind]:
            sit = mem.situation(e["a"], e["c0"], e["c1"])
            y0, x0 = e["i"] // W, e["i"] % W

            def tok_of(cell):
                y, x = y0 + cell // WW - DY, x0 + cell % WW - DX
                return y * W + x if 0 <= y < H and 0 <= x < W and y * W + x != e["i"] else None

            cell_of = lambda j: int((j // W - y0 + DY) * WW + (j % W - x0 + DX))
            fin = []
            C = CT.greedy(sit, e["vals"], None, fin)
            ans = fin[2]
            right, wrong, found = CT.score_sets(C, tok_of, e["tc"], e["judged"], True)
            true = e["tc"] & e["judged"]
            missed = true & ~found
            c["right"] += right; c["wrong"] += wrong; c["true"] += int(true.sum()); c["found"] += int(found.sum())
            tj = np.flatnonzero(true)
            ok = CT.contrastable_sig(sit, [cell_of(j) for j in tj], [int(e["vals"][cell_of(j)]) for j in tj])
            for j, o in zip(tj, ok):
                if o:
                    c["ctrue"] += 1; c["cfound"] += int(found[j])
            c["closed"] += int(ans["closed"]); c["closed complete"] += int(ans["closed"] and not missed.any())
        pr = lambda x, y: round(x / y, 4) if y else None
        n = len(ITEMS[kind])
        res[kind] = {"precision": pr(c["right"], c["right"] + c["wrong"]), "recall": pr(c["found"], c["true"]),
                     "true conditions": c["true"], "found": c["found"],
                     "recall on contrastable": pr(c["cfound"], c["ctrue"]), "closed share": pr(c["closed"], n),
                     "closed events": c["closed"], "closed and complete, of closed": pr(c["closed complete"], c["closed"])}
    lamps = [res[k] for k in LAMP_KINDS if k in res]
    if len(lamps) == 2:
        res["lamps (lighting and going out)"] = {"recall": round(sum(r["found"] for r in lamps) / max(sum(r["true conditions"] for r in lamps), 1), 4)}
    return res


@torch.no_grad()
def ask(mem, g, i, a, c1):
    """Memory's answer for token i of grid g (63,) under action a with outcome c1: kept facts and the answer."""
    sit = mem.situation(a, int(g[i]), c1)
    vals = windows(g[None], np.array([0]), np.array([i]))[0]
    fin = []
    C = CT.greedy(sit, vals, None, fin)
    return C, fin[2], vals


def matches(q, g, wc):
    """Whether a token of grid g (63,) with the question's code satisfies its kept facts in a state this question has
    not been tried in (a repeated try adds no new case). wc caches windows per code."""
    if (g.tobytes(), q["a"]) in q["tried"]:
        return False
    if q["c0"] not in wc:
        toks = np.flatnonzero(g == q["c0"])
        wc[q["c0"]] = windows(g[None].repeat(len(toks), 0), np.arange(len(toks)), toks) if len(toks) else None
    w = wc[q["c0"]]
    if w is None:
        return False
    ok = np.ones(len(w), bool)
    for cell, val, is_eq in q["facts"]:
        ok &= (w[:, cell] == val) if is_eq else (w[:, cell] != val)
    return bool(ok.any())


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    ztr = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    valid = np.r_[ztr["ep"][1:] == ztr["ep"][:-1], False] & ~ztr["done"]
    mem = Memory(ztr["sem"].reshape(-1, 63).astype(np.int64), ztr["act"].astype(np.int64), valid, 111 + SEED)
    ITEMS = eval_items()
    rep = {"arm": ARM, "seed": SEED, "steps": STEPS, "smoke": SMOKE, "events": {k: len(v) for k, v in ITEMS.items()}}
    if ARM == "evalrooms":
        z = np.load(ROOT / "runs" / "111.2" / "world1_test_fresh.npz")
        mem.new_sem.append(z["sem"].reshape(-1, 63).astype(np.int64)); mem.new_act.append(z["act"].astype(np.int64))
        mem.new_valid.append(np.r_[z["ep"][1:] == z["ep"][:-1], False] & ~z["done"])
        mem.refresh()
    checks = {}
    if ARM in ("none", "evalrooms"):
        checks[str(0)] = evaluate(mem, ITEMS, list(ITEMS))
        print(ARM, json.dumps({k: (v.get("precision"), v["recall"]) for k, v in checks["0"].items()}), flush=True)
    else:
        room_rng = np.random.default_rng(1113 + 1000 * SEED)            # rooms shared by the arms of one seed
        act_rng = np.random.default_rng(7 + SEED)
        Q, last = {}, {}
        stats = Counter()
        w = None
        grids, acts = [], []
        for step in range(STEPS):
            if step % EP_LEN == 0:
                if grids:
                    mem.add_episode(grids, acts)
                grids, acts = [], []
                w = WD.World1(room_rng, True)
            g = w.grid().reshape(-1).astype(np.int64)
            a = None
            if ARM == "directed" and Q:
                wc = {}
                hit = [q for q in Q.values() if matches(q, g, wc)]
                if hit:
                    q = min(hit, key=lambda q: q["n"])
                    a = q["a"]
                    q["tried"].add((g.tobytes(), a))
                    stats["tries matching a question"] += 1
            if a is None:
                a = int(act_rng.integers(0, 5))
            grids.append(g.copy()); acts.append(a)
            w.step(a)
            g1 = w.grid().reshape(-1).astype(np.int64)
            ch = np.flatnonzero(g1 != g)
            stats["lamp events seen"] += int((((g == LAMP) & (g1 == LIT)) | ((g == LIT) & (g1 == LAMP))).sum())
            if ARM == "directed":
                for i in ch:
                    kind = (a, int(g[i]), int(g1[i]))
                    dq = last.setdefault(kind, deque(maxlen=20))
                    if len(dq) == 20 and all(dq) and act_rng.random() >= 0.05:
                        continue
                    C, ans, vals = ask(mem, g, int(i), a, int(g1[i]))
                    stats["questions asked"] += 1
                    dq.append(ans["closed"])
                    if not ans["closed"]:
                        facts = tuple(sorted((f[0], f[1], f[2]) for f in C))
                        key = kind + (facts,)
                        if key not in Q:
                            stats["questions raised"] += 1
                            Q[key] = {"a": a, "c0": int(g[i]), "c1": int(g1[i]), "facts": facts, "n": ans["items left"],
                                      "g": g.copy(), "i": int(i), "tried": {(g.tobytes(), a)}}
            if (step + 1) % REBUILD == 0:
                mem.refresh()
                if ARM == "directed":
                    for key, q in list(Q.items()):
                        C, ans, _ = ask(mem, q["g"], q["i"], q["a"], q["c1"])
                        if ans["closed"]:
                            del Q[key]
                            stats["questions closed"] += 1
                        else:
                            q["facts"] = tuple(sorted((f[0], f[1], f[2]) for f in C))
                            q["n"] = ans["items left"]
            if (step + 1) in CHECKS or step + 1 == STEPS:
                if grids:
                    mem.add_episode(grids, acts)
                    grids, acts = [], []
                    w = None
                mem.refresh()
                kinds = list(ITEMS) if step + 1 == STEPS else list(LAMP_KINDS)
                checks[str(step + 1)] = evaluate(mem, ITEMS, kinds)
                print(ARM, SEED, step + 1, json.dumps({k: (v.get("precision"), v["recall"]) for k, v in checks[str(step + 1)].items()}),
                      dict(stats), f"open {len(Q)}", f"({time.monotonic() - t0:.0f} s)", flush=True)
        stats["open questions at the end"] = len(Q)
        names = ["floor", "wall", "agent", "box", "lamp", "lit lamp"]
        rep["open questions at the end (action, code, next code, kept facts, cases)"] = [
            (q["a"], names[q["c0"]], names[q["c1"]], len(q["facts"]), q["n"]) for q in Q.values()][:40]
        rep["exploration"] = dict(stats)
    rep["checks"] = checks
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"{ARM}_s{SEED}_{STEPS}{'_smoke' if SMOKE else ''}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("done", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
