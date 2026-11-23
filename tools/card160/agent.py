"""Card 160: acting by chaining statements backward from a goal (the card's one new component).

  goal            an event seen at the example success steps: a token's code changes, or the agent arrives on a token
  want an effect  read its statement; an unmet condition on what is out of view becomes the goal of the effect that
                  makes it true; then be at a place the statement allows; then its action
  be at a place   closeness propagated over the learned move effect on the believed map (Dijkstra); a move the model
                  says is blocked may be taken at a cost if some known effect on a neighbouring token would unblock it,
                  and then that effect is wanted first
  forward check   an effect done only to bring a token back into view is imagined once, and refused if it would put a
                  new blocker between that place and a place the chain still needs
The plan is rebuilt from the present at every step; nothing is carried over but two failed tries of the same effect.
"""
import os, heapq, numpy as np
from collections import Counter
from grid import DIRV, OFF8, NA, FAR, OUT, gid_inv, Belief

C_BLOCK, N_TRY, FAIL, HERE = 50, 10, "fail", "here"
AFTER_TEST_FIX = os.environ.get("FIX") == "1"                        # off in the frozen test; see the card, "after the test"


def holds(f, items):
    return (f[1] in items) if f[0] == "it" else any(it[1] == f[1] for it in items)


def goal_events(examples):
    """the events at the example success steps: (map before, pose before, action, events, pose after)"""
    goals = []
    for mp, pose, a, ev, pose2 in examples:
        g = [("effect", c, c2, a) for _, _, c, c2 in ev] or ([("arrive", int(mp[pose2[0], pose2[1]]))] if pose2[:2] != pose[:2] else [])
        goals += [x for x in g if x not in goals]
    return goals


class Agent:
    def __init__(s, model, goals, seed=0):
        s.M = model; s.goals = goals; s.rng = np.random.default_rng(seed); s.acts = {}

    def reset(s, g, p):
        s.bel = Belief(g, p); s.fail = Counter(); s.tried = None; s.last = None; s.same = False; s.G = {}; s.D = {}; s.noplan = 0

    def observe(s, a, g, p):
        b = (s.bel.map, s.bel.pose, s.bel.items()); ev = s.bel.step(a, g, p)
        s.same = (b[0] == s.bel.map).all() and b[1] == s.bel.pose and b[2] == s.bel.items()
        if s.tried is not None and s.bel.map[s.tried[0]] != s.tried[1]:
            s.fail[s.tried] += 1                                       # the effect did not happen: not under these conditions
        s.tried = None
        return ev

    def act(s):
        a = s.plan()
        if a is None or (s.same and a == s.last):                      # no plan, or the same action in an unchanged state
            a = int(s.rng.integers(NA)); s.noplan += 1; s.tried = None
        s.last = a
        return a

    # ---- the chain
    def cells(s, c):
        x0, y0, _ = s.bel.pose; xs, ys = np.nonzero(s.bel.map == c)
        return sorted(((int(x), int(y)) for x, y in zip(xs, ys)), key=lambda q: abs(q[0] - x0) + abs(q[1] - y0))

    def plan(s):
        for goal in s.goals:
            if goal[0] == "arrive":
                for X in s.cells(goal[1]):
                    r = s.go([(X[0], X[1], d) for d in range(4)], (), ())
                    if isinstance(r, int):
                        return r
            else:
                for X in s.cells(goal[1]):
                    r = s.want_effect(X, goal[2], goal[3], (), ())
                    if isinstance(r, int):
                        return r
        return None

    def want_effect(s, X, o, a_req, stack, chain):
        c = int(s.bel.map[X]); items = s.bel.items()
        if (X, o) in chain or len(chain) > 8 or s.fail[(X, o, items)] >= 2:
            return FAIL
        st = s.M.statement(c, o, a_req)
        if st is None:
            return FAIL
        free = any(g < 0 or g == FAR for g in st["G"])
        targets = [] if free else [(X[0] - dx, X[1] - dy, d) for dx, dy, d in map(gid_inv, st["G"])]
        stack2 = stack + (tuple(targets),) if targets else stack; chain2 = chain + ((X, o),)
        for f in st["need"]:
            if not holds(f, items):
                return s.want_in(f, stack2, chain2)
        unmet = [f for f in st["avoid"] if holds(f, items)]
        if unmet and (free or s.free_reach(targets)):                  # put a thing down only once nothing stands between the agent and
            return s.want_out(unmet[0], stack2, chain2)                #   the place: the walk may still need what is held
        r = HERE if free else s.go(targets, stack, chain2)
        if r == HERE:
            s.tried = (X, o, items)
            return int(st["a"])
        return r

    def want_in(s, f, stack, chain):
        """make a token go out of view (by a given action): the condition 'holding it', in the agent's own terms"""
        ak = f[1][1] if f[0] == "it" else f[1]; cands = []
        for k in np.unique(s.bel.map):
            if f[0] == "it" and k != f[1][0]:
                continue
            for o in s.M.outcomes(int(k), ak):
                cands += [(X, o) for X in s.cells(k)]
        x0, y0, _ = s.bel.pose
        for X, o in sorted(cands, key=lambda q: abs(q[0][0] - x0) + abs(q[0][1] - y0)):
            r = s.want_effect(X, o, ak, stack, chain)
            if isinstance(r, int):
                return r
        return FAIL

    def want_out(s, f, stack, chain):
        """bring a token that is out of view back into view, somewhere that breaks nothing the chain needs"""
        items = s.bel.items()
        for k, ak in [it for it in items if (it == f[1] if f[0] == "it" else it[1] == f[1])]:
            for ce in np.unique(s.bel.map):
                ce = int(ce)
                if k not in s.M.outcomes(ce):
                    continue
                st = s.M.statement(ce, k)
                for Z in s.cells(ce)[:N_TRY]:
                    if s.forward_ok(Z, ce, k, st, stack):
                        r = s.want_effect(Z, k, None, stack, chain)
                        if isinstance(r, int):
                            return r
        return FAIL

    def forward_ok(s, Z, ce, k, st, stack):
        mp, items = s.bel.map, s.bel.items(); W, H = mp.shape; x, y, d = s.bel.pose
        dist, _ = s.paths(mp, items, (x * H + y) * 4 + d)
        tg = [(tx * H + ty) * 4 + td for tx, ty, td in ((Z[0] - dx, Z[1] - dy, dd) for dx, dy, dd in map(gid_inv, st["G"])) if 0 <= tx < W and 0 <= ty < H]
        tg = [t for t in tg if dist.get(t, 1e9) < C_BLOCK]
        if not tg:
            return False
        pe = min(tg, key=dist.get); mp2 = mp.copy(); mp2[Z] = k
        gone = [it for it in items if it[0] == k][:1]; items2 = frozenset((items - set(gone)) | {(ce, st["a"])})
        d1, _ = s.paths(mp, items, pe); d2, _ = s.paths(mp2, items2, pe)
        for T in stack:
            ts = [(tx * H + ty) * 4 + td for tx, ty, td in T if 0 <= tx < W and 0 <= ty < H]
            n1 = min((d1[t] for t in ts if t in d1), default=None); n2 = min((d2[t] for t in ts if t in d2), default=None)
            if n1 is not None and (n2 is None or n2 // C_BLOCK > n1 // C_BLOCK):
                return False
        return True

    # ---- being at a place
    def free_reach(s, targets):
        mp = s.bel.map; W, H = mp.shape; x, y, d = s.bel.pose; dist, _ = s.paths(mp, s.bel.items(), (x * H + y) * 4 + d)
        return any(dist.get((tx * H + ty) * 4 + td, 1e9) < C_BLOCK for tx, ty, td in targets if 0 <= tx < W and 0 <= ty < H)

    def go(s, targets, stack, chain):
        mp = s.bel.map; W, H = mp.shape; x, y, d = s.bel.pose; src = (x * H + y) * 4 + d
        ts = [(tx * H + ty) * 4 + td for tx, ty, td in targets if 0 <= tx < W and 0 <= ty < H]
        if src in ts:
            return HERE
        dist, prev = s.paths(mp, s.bel.items(), src); ts = [t for t in ts if t in dist]
        if not ts:
            return FAIL
        t = min(ts, key=dist.get); path = []
        while t != src:
            t, a, lab = prev[t]; path.append((a, lab))
        for a, lab in path[::-1]:
            if lab is not None:                                        # the walk is blocked here: first the effect that unblocks it
                return s.want_effect(lab[0], lab[1], None, stack + (tuple(targets),), chain)
        return int(path[-1][0])

    def paths(s, mp, items, src):
        key = (mp.tobytes(), items, src)
        if key not in s.D:
            edges = s.graph(mp, items); dist = {src: 0}; prev = {}; h = [(0, src)]
            while h:
                d0, u = heapq.heappop(h)
                if d0 > dist[u]:
                    continue
                for c, v, a, lab in edges[u]:
                    if d0 + c < dist.get(v, 1e9):
                        dist[v] = d0 + c; prev[v] = (u, a, lab); heapq.heappush(h, (d0 + c, v))
            s.D[key] = (dist, prev)
        return s.D[key]

    def graph(s, mp, items):
        """what the learned move effect says each action does from every pose on the believed map"""
        key = (mp.tobytes(), items)
        if key in s.G:
            return s.G[key]
        W, H = mp.shape; pad = np.full((W + 2, H + 2), OUT); pad[1:-1, 1:-1] = mp
        nb = np.stack([pad[1 + dx:1 + dx + W, 1 + dy:1 + dy + H] for dx, dy in OFF8], -1).reshape(-1, 8)
        edges = [[] for _ in range(W * H * 4)]; outs = [np.asarray(s.M.agent_pred(d, nb, items)) for d in range(4)]
        for d in range(4):
            mv, d2 = outs[d][..., 0], outs[d][..., 1]
            for cell, a in zip(*np.nonzero((mv != 0) | (d2 != d))):
                x, y = divmod(int(cell), H); m = int(mv[cell, a])
                x2, y2 = (x, y) if m == 0 else (x + DIRV[m - 1][0], y + DIRV[m - 1][1])
                if 0 <= x2 < W and 0 <= y2 < H:
                    edges[cell * 4 + d].append((1, (x2 * H + y2) * 4 + int(d2[cell, a]), int(a), None))
        for k in np.unique(mp):
            alts = s.M.outcomes(int(k))
            if not alts:
                continue
            for x, y in zip(*np.nonzero(mp == k)):
                for j, (dx, dy) in enumerate(OFF8):
                    px, py = int(x) - dx, int(y) - dy
                    if not (0 <= px < W and 0 <= py < H):
                        continue
                    cell = px * H + py
                    for d in range(4):
                        for a in s.M.move_actions:
                            if outs[d][cell, a, 0] != 0:
                                continue
                            for o in alts:                              # would the move go through if this token were changed by a known effect?
                                it2 = items
                                if AFTER_TEST_FIX:                      # imagine the effect whole: its trace in memory too (as forward_ok does)
                                    if (int(k), o) not in s.acts:
                                        s.acts[(int(k), o)] = s.M.statement(int(k), o)["a"]
                                    it2 = frozenset((items - set([q for q in items if q[0] == o][:1])) | {(int(k), s.acts[(int(k), o)])})
                                nb2 = list(nb[cell]); nb2[j] = o; m, dd = s.M.agent_one(d, a, tuple(int(v) for v in nb2), it2)
                                if m != 0:
                                    x2, y2 = px + DIRV[m - 1][0], py + DIRV[m - 1][1]
                                    if 0 <= x2 < W and 0 <= y2 < H:
                                        edges[cell * 4 + d].append((1 + C_BLOCK, (x2 * H + y2) * 4 + dd, int(a), ((int(x), int(y)), int(o))))
                                    break
        s.G[key] = edges
        return edges


class Random:
    def __init__(s, seed=0):
        s.rng = np.random.default_rng(seed); s.noplan = 0

    def reset(s, g, p): pass
    def observe(s, a, g, p): pass
    def act(s): return int(s.rng.integers(NA))


def episode(world, seed, agent, want_example=False):
    g, p = world.reset(seed); agent.reset(g, p); done = False; n = 0; ex = None
    while not done:
        a = agent.act(); before = (agent.bel.map, agent.bel.pose) if want_example else None
        g, p, succ, done = world.step(a); ev = agent.observe(a, g, p); n += 1
        if want_example and succ:
            ex = (before[0], before[1], a, ev, p)
    return (succ, n, agent.noplan, ex) if want_example else (succ, n, agent.noplan)
