"""Card 105.5 in breaking world 1: planning both ways over conditions read from the transition model.

Everything is in perception's codes (world 1's encoder: one code per kind of tile). Arm "model" uses card 105.6's
transition model (seed 1021); arm "truth" (the upper bound) puts the world's own step function in its place; arm
"random" acts at random.

1. Statements from experience. On training events where the transition predicts a token i changing from code a to b
   under action u, every other world tile j is replaced by each other code seen in the same state (card 105.4); j is
   a condition if some replacement changes i's predicted next code. A statement is the set of (offset of j from i,
   j's code) with its action; statements seen at least twice are kept, by (a, b).
2. Backward. To make place p show code c: if it does, done. If c is the agent's code, walk (3). Otherwise try each
   statement for (code at p -> c), most often seen and fewest unmet conditions first: achieve its conditions one at a
   time in imagination (the agent's own condition last: every action moves the agent), refuse the chain if a
   condition no longer holds when all are done, then imagine the statement's action and keep it only if p becomes c.
   Depth at most 6 (version 27's), a budget of expansions per call, and no state and goal tried twice.
3. Walking. Closeness is propagated (card 067) over the moves the model predicts in the imagined state: the agent put
   at each place by replacement, each move imagined, and a move kept if it only moves the agent.
4. Forward. Every action of a chain is imagined one effect at a time from the present; the plan is carried out in
   the world and made again only when what is seen differs from what was imagined.

Scored on test worlds (lamps at columns 5-7): an unlit lamp lit within 50 steps.

  bin/prun python tools/card105.5/world1.py model|truth|random [--episodes 200]  → runs/105.5/world1_<arm>.json
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
sys.path.insert(0, str(ROOT / "tools" / "card105"))
from model import TransitionModel                          # noqa: E402

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
ARM = sys.argv[1]
EPISODES, MAXSTEPS, DEPTH, BUDGET = int(arg("--episodes", 200)), 50, 6, int(arg("--budget", 400))
OUT = ROOT / "runs" / "105.5"
dev = "cuda"
spec = importlib.util.spec_from_file_location("worlds102", ROOT / "tools" / "card102" / "worlds.py")
WD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(WD)
W, H, NT = 9, 7, 81
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
# perception: each kind of tile draws one vector (world 1's encoder), named by its code; kinds are scoring only
ztest = np.load(ROOT / "runs" / "card102" / "world1_test.npz")
V_ALL = np.load(ROOT / "runs" / "102.1" / "world1_vec_test.npy")[:2000].astype(np.float32)
C_ALL = np.load(ROOT / "runs" / "102.1" / "world1_code_test.npy")[:2000]
VEC = {int(c): torch.as_tensor(V_ALL.reshape(-1, V_ALL.shape[-1])[np.flatnonzero(C_ALL.reshape(-1) == c)[0]])
       for c in np.unique(C_ALL)}
KIND_SEM = ztest["sem"].reshape(-1, 63)[:2000]
CODE_OF_KIND = {int(k): int(np.bincount(C_ALL[:, :63][KIND_SEM == k]).argmax()) for k in np.unique(KIND_SEM)}
KIND_OF_CODE = {c: k for k, c in CODE_OF_KIND.items()}
BOTTOM = int(np.bincount(C_ALL[:, 63:].reshape(-1)).argmax())
AGENT = CODE_OF_KIND[WD.AGENT]                           # the agent's code: what the agent's own actions move (declared)
PROTOS = torch.as_tensor(np.load(ROOT / "runs" / "102.1" / "world1_codes.npz")["protos"], device=dev).float()
P_TILE = torch.tensor([(k % 9, k // 9) for k in range(NT)], dtype=torch.float32, device=dev)
F_TILE = torch.tensor([0 if k < 63 else 1 for k in range(NT)], device=dev)
VTAB = torch.stack([VEC.get(c, torch.zeros_like(VEC[BOTTOM])) for c in range(len(PROTOS))]).to(dev)

if ARM == "model":
    NET = TransitionModel(VTAB.shape[1], 5, arm="Rb").to(dev)
    NET.load_state_dict(torch.load(ROOT / "runs" / "105.6" / "tokens_Rb_s1021_net.pt"))
    NET.eval()
CALLS = Counter()


@torch.no_grad()
def predict(S, U):
    """Next codes of the 63 world tiles for states S (B, 63) under actions U (B,)."""
    S, U = np.asarray(S), np.asarray(U)
    CALLS["imagined transitions"] += len(S)
    if ARM == "truth":
        out = []
        for s, u in zip(S, U):
            g = np.vectorize(KIND_OF_CODE.get)(s).reshape(H, W)
            out.append(np.vectorize(CODE_OF_KIND.get)(true_next(g, int(u)).reshape(-1)))
        return np.array(out)
    res = []
    for b in range(0, len(S), 2048):
        s = torch.as_tensor(S[b:b + 2048], device=dev)
        full = torch.cat([s, torch.full((len(s), NT - 63), BOTTOM, device=dev, dtype=s.dtype)], 1)
        V = VTAB[full]
        B = len(V)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt, _ = NET(V, P_TILE.expand(B, NT, 2), F_TILE.expand(B, NT), torch.ones(B, NT, dtype=torch.bool, device=dev),
                         torch.as_tensor(U[b:b + 2048], device=dev))
        res.append(torch.cdist(nxt[:, :63].float().reshape(-1, PROTOS.shape[1]), PROTOS).argmin(1).view(B, 63).cpu().numpy())
    return np.concatenate(res)


def true_next(grid, a):
    """World 1's step (card 102) on a 7 x 9 grid of kinds with the agent drawn in it (scoring and the upper bound)."""
    g = grid.copy()
    ag = np.argwhere(g == WD.AGENT)
    if len(ag) != 1 or a not in MOVES:
        light(g)
        return g
    y, x = ag[0]
    g[y, x] = WD.FLOOR
    dx, dy = MOVES[a]
    ny, nx = y + dy, x + dx
    if 0 <= ny < H and 0 <= nx < W and g[ny, nx] == WD.FLOOR:
        y, x = ny, nx
    elif 0 <= ny < H and 0 <= nx < W and g[ny, nx] == WD.BOX:
        by, bx = ny + dy, nx + dx
        if 0 <= by < H and 0 <= bx < W and g[by, bx] == WD.FLOOR:
            g[by, bx], g[ny, nx] = WD.BOX, WD.FLOOR
            y, x = ny, nx
    light(g)
    g[y, x] = WD.AGENT
    return g


def light(g):
    for y, x in zip(*np.nonzero((g == WD.LAMP) | (g == WD.LIT))):
        near = any(0 <= y + dy < H and 0 <= x + dx < W and g[y + dy, x + dx] == WD.BOX for dx, dy in MOVES.values())
        g[y, x] = WD.LIT if near else WD.LAMP


# ------------------------------------------------------------------------------------------------ 1. statements
def learn_statements(per=150):
    z = np.load(ROOT / "runs" / "card102" / "world1_train.npz")
    sem, act, ep, done = z["sem"].reshape(-1, 63), z["act"].astype(int), z["ep"], z["done"]
    codes = np.vectorize(CODE_OF_KIND.get)(sem)
    T = np.flatnonzero(np.r_[ep[1:] == ep[:-1], False] & ~done & np.isin(act, list(MOVES)))
    rng = np.random.default_rng(1021)
    rng.shuffle(T)
    T = T[:6000]
    pred = predict(codes[T], act[T])
    by_type = defaultdict(list)
    for n, t in enumerate(T):
        for i in np.flatnonzero(pred[n] != codes[t]):
            key = (int(codes[t, i]), int(pred[n, i]), int(act[t]))
            if len(by_type[key]) < per:
                by_type[key].append((t, int(i), int(pred[n, i])))
    st = defaultdict(Counter)
    for (a, b, u), evs in by_type.items():
        for t, i, b_ in evs:
            s = codes[t]
            present = sorted(set(s.tolist()) - {AGENT})
            var, js = [], []
            for j in range(63):
                if j == i:
                    continue
                for k in ([CODE_OF_KIND[WD.FLOOR]] if s[j] == AGENT else [k for k in present if k != s[j]]):
                    v = s.copy()
                    v[j] = k
                    var.append(v)
                    js.append(j)
            p = predict(np.array(var), np.full(len(var), u))[:, i]
            cond = sorted({j for j, x in zip(js, p) if x != b_})
            ix, iy = i % 9, i // 9
            st[(a, b)][(u, tuple((j % 9 - ix, j // 9 - iy, int(s[j])) for j in cond))] += 1
    return {k: [(u, conds, n) for (u, conds), n in c.most_common() if n >= 2] for k, c in st.items()}


# ------------------------------------------------------------------------------------------------ 2-3. planning
def walk(s, q):
    """Closeness to place q propagated over the moves the model predicts in state s; the steps there, imagined."""
    a0 = int(np.flatnonzero(s == AGENT)[0])
    if a0 == q:
        return [], s
    floor = CODE_OF_KIND[WD.FLOOR]
    places = [c for c in range(63) if s[c] == floor or c == a0]
    S, U, src = [], [], []
    for c in places:
        v = s.copy()
        v[a0], v[c] = floor, AGENT
        for u in MOVES:
            S.append(v)
            U.append(u)
            src.append(c)
    P = predict(np.array(S), np.array(U))
    edges = defaultdict(list)                                            # reverse edges: who reaches c in one move
    for n, (c, u) in enumerate(zip(src, U)):
        dx, dy = MOVES[u]
        d = c + dx + 9 * dy
        if 0 <= d < 63 and P[n, d] == AGENT and (P[n] != S[n]).sum() == 2:   # only the agent moved
            edges[d].append((c, u))
    close = {q: 0}
    dq = deque([q])
    while dq:
        x = dq.popleft()
        for c, u in edges[x]:
            if c not in close:
                close[c] = close[x] + 1
                dq.append(c)
    if a0 not in close:
        return None
    acts, cur, v = [], a0, s.copy()
    while cur != q:
        step = min((close[cur + MOVES[u][0] + 9 * MOVES[u][1]], u) for c, u in
                   [(c, u) for d_, lst in edges.items() for c, u in lst if c == cur and d_ in close])
        u = step[1]
        v = predict(v[None], [u])[0]                                     # forward: one imagined effect at a time
        nxt = int(np.flatnonzero(v == AGENT)[0]) if (v == AGENT).sum() == 1 else -1
        if close.get(nxt, 99) >= close[cur]:
            return None
        acts.append(u)
        cur = nxt
    return acts, v


def achieve(s, p, c, depth, budget, seen):
    if s[p] == c:
        return [], s
    if c == AGENT:
        return walk(s, p)
    if depth == 0 or budget[0] <= 0:
        return None
    key = (s.tobytes(), p, c, depth)
    if key in seen:
        return None
    seen.add(key)
    budget[0] -= 1
    px, py = p % 9, p // 9
    opts = []
    for u, conds, n in STATEMENTS.get((int(s[p]), c), []):
        places = [(px + dx, py + dy, k) for dx, dy, k in conds]
        if any(not (0 <= x < W and 0 <= y < H) for x, y, _ in places):
            continue
        need = [(y * 9 + x, k) for x, y, k in places]
        unmet = sum(s[q] != k for q, k in need)
        opts.append((unmet, -n, u, need))
    for unmet, _, u, need in sorted(opts)[:6]:
        need = sorted(need, key=lambda qk: qk[1] == AGENT)               # the agent's own condition last
        acts, v, ok = [], s, True
        for q, k in need:
            r = achieve(v, q, k, depth - 1, budget, seen)
            if r is None:
                ok = False
                break
            acts, v = acts + r[0], r[1]
        if not ok or any(v[q] != k for q, k in need):                   # a condition undone on the way: refuse
            continue
        v2 = predict(v[None], [u])[0]
        if v2[p] == c:
            return acts + [u], v2
    return None


# ------------------------------------------------------------------------------------------------ 4. acting
def episode(seed):
    rng = np.random.default_rng(10_000 + seed)
    while True:
        env = WD.World1(rng, test=True)
        lamps = np.flatnonzero(env.grid().reshape(-1) == WD.LAMP)
        if len(lamps):
            break
    L = int(lamps[0])
    lit = CODE_OF_KIND[WD.LIT]
    see = lambda: np.vectorize(CODE_OF_KIND.get)(env.grid().reshape(-1))
    plan, imagined, t_plan, calls = [], [], 0.0, 0
    for step in range(MAXSTEPS):
        s = see()
        if env.grid().reshape(-1)[L] == WD.LIT:
            return True, step, t_plan, calls
        if ARM == "random":
            env.step(int(np.random.default_rng(seed * 1000 + step).integers(1, 5)))
            continue
        if not plan or (imagined and not np.array_equal(imagined[0], s)):
            t0 = time.monotonic()
            r = achieve(s, L, lit, DEPTH, [BUDGET], set())
            t_plan += time.monotonic() - t0
            calls += 1
            if r is None:
                return False, step, t_plan, calls
            plan = list(r[0])
            imagined, v = [], s
            for u in plan:                                               # what each step should show beforehand
                imagined.append(v)
                v = predict(v[None], [u])[0]
            if not plan:
                return env.grid().reshape(-1)[L] == WD.LIT, step, t_plan, calls
        u = plan.pop(0)
        imagined.pop(0)
        env.step(u)
    return bool(env.grid().reshape(-1)[L] == WD.LIT), MAXSTEPS, t_plan, calls


def main():
    global STATEMENTS
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    STATEMENTS = {} if ARM == "random" else learn_statements()
    t_learn = time.monotonic() - t0
    print(f"statements: {sum(len(v) for v in STATEMENTS.values())} over {len(STATEMENTS)} changes ({t_learn:.0f} s)",
          flush=True)
    res = []
    for e in range(EPISODES):
        res.append(episode(e))
        if e % 20 == 0:
            print(f"episode {e}: success so far {np.mean([r[0] for r in res]):.3f} ({time.monotonic() - t0:.0f} s)",
                  flush=True)
    ok = np.array([r[0] for r in res])
    steps = np.array([r[1] for r in res])
    rep = {"arm": ARM, "episodes": EPISODES, "success (lamp lit within 50 steps)": round(float(ok.mean()), 4),
           "steps when successful (mean)": round(float(steps[ok].mean()), 2) if ok.any() else None,
           "planning seconds per episode (mean)": round(float(np.mean([r[2] for r in res])), 3),
           "planning calls per episode (mean)": round(float(np.mean([r[3] for r in res])), 2),
           "statements": {f"{KIND_OF_CODE.get(a, a)}->{KIND_OF_CODE.get(b, b)} (kinds, scoring only)":
                          [(u, list(c), n) for u, c, n in v[:6]] for (a, b), v in STATEMENTS.items()},
           "imagined transitions": int(CALLS["imagined transitions"]),
           "seconds to learn statements": round(t_learn, 1), "seconds": round(time.monotonic() - t0, 1)}
    (OUT / f"world1_{ARM}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "statements"}, indent=1))


if __name__ == "__main__":
    main()
