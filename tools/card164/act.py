"""Card 164: System 2 chooses the next event; System 1 (learned q) brings it about.

Three agents sharing one System 1 execution rule (Sys1):
  - Handed: upper bound; handed the next true event of the rules agent's solution.
  - ArmA:   System 2 is card 162's condition-based planner returning the next target event (X, o, a) or (X, ARR, ARR_ACT);
            System 1 (q) chooses every primitive action (walking, orienting, executing, no-op recovery).
  - ArmB:   System 2 is event-level forward search over imagined events gated by the cell effect model (item conditions)
            and single-event q (spatial reachability) — zero Dijkstra, zero move model; System 1 (q) executes every action.
"""
import os, heapq, numpy as np, torch
from collections import deque
import agent, agent161
from grid import NA, UNK, OUT, FAR, DIRV, gid_inv, Belief
import q as Q

HOLD = int(os.environ.get("HOLD", 40))
TAU = float(os.environ.get("TAU", 0.15))
MAX_EXP = int(os.environ.get("MAX_EXP", 80))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 7))


class QModel:
    """the fitted single-event prediction q(situation, action; event), batched on the device"""
    def __init__(s, path):
        ck = torch.load(path, map_location=Q.dev, weights_only=False)
        s.NC = ck["NC"]; s.ARR = s.NC; s.ARR_ACT = Q.ARR_ACT; s.bg = ck["bg"]
        s.known_set = ck["known_set"]; s.single = ck.get("single", True)
        s.net = Q.Net(s.NC).to(Q.dev); s.net.load_state_dict(ck["net"]); s.net.eval()
        s._tok = {}

    def toks(s, mp):
        key = (mp.shape, mp.tobytes())
        if key not in s._tok:
            s._tok[key] = Q.tokens_of(mp, s.bg)
        return s._tok[key]

    def ask(s, rows):
        """rows: list of (mp, pose, items, X, o, ea) -> array [N, NA] in [0, 1]"""
        n = len(rows)
        if not n:
            return np.zeros((0, NA), np.float32)
        tk = [s.toks(r[0]) for r in rows]
        L = max(1, max(len(t[0]) for t in tk))
        M = max(1, max(len(r[2]) for r in rows))
        z = lambda *sh: np.zeros(sh, np.int64)
        pd, ac, adx, ady, oo, ea = z(n), z(n), z(n), z(n), z(n), z(n)
        tc, tdx, tdy, ic, ia = z(n, L), z(n, L), z(n, L), z(n, M), z(n, M)
        tm = np.zeros((n, L), bool); im = np.zeros((n, M), bool)
        for i, (mp, pose, items, X, o, a_ev) in enumerate(rows):
            xs, ys, cs = tk[i]; k = len(xs)
            tc[i, :k] = cs; tdx[i, :k] = xs - pose[0]; tdy[i, :k] = ys - pose[1]; tm[i, :k] = True
            for j, (kk, a) in enumerate(sorted(items)):
                ic[i, j] = kk; ia[i, j] = a; im[i, j] = True
            pd[i] = pose[2]; ac[i] = mp[X]; adx[i] = X[0] - pose[0]; ady[i] = X[1] - pose[1]
            oo[i] = o; ea[i] = a_ev
        cl = lambda x: np.minimum(x, s.NC); t = lambda x: torch.as_tensor(x, device=Q.dev); out = []
        with torch.no_grad():
            for b in range(0, n, 4096):
                e = slice(b, b + 4096)
                lg = s.net(t(pd[e]), t(cl(ac[e])), t(adx[e]), t(ady[e]), t(oo[e]), t(ea[e]),
                           t(cl(tc[e])), t(tdx[e]), t(tdy[e]), t(tm[e]), t(cl(ic[e])), t(ia[e]), t(im[e]))
                out.append(torch.sigmoid(lg).cpu().numpy())
        return np.concatenate(out)


class Sys1:
    """Base agent: System 2 (choose_event) picks the next event (X, o, ea); System 1 (Q) picks every primitive action."""
    def __init__(s, qmodel, goals, hold=HOLD):
        s.Q = qmodel; s.goals = goals; s.hold = hold

    def reset(s, g, p):
        s.bel = Belief(g, p); s.taken = {}; s.noplan = 0; s.held = None; s.age = 0
        s.first = None; s.plans = 0; s.expansions = 0

    def observe(s, a, g, p):
        items = s.bel.items(); ev = s.bel.step(a, g, p)
        if ev or s.bel.items() != items:
            s.held = None; s.age = 0
        return ev

    def goal_asks(s, mp):
        out = []
        for g in s.goals:
            if g[0] == "arrive":
                out += [((int(x), int(y)), s.Q.ARR, s.Q.ARR_ACT) for x, y in zip(*np.nonzero(mp == g[1]))]
            else:
                out += [((int(x), int(y)), int(g[2]), int(g[3])) for x, y in zip(*np.nonzero(mp == g[1]))]
        return out

    def choose_event(s):
        raise NotImplementedError

    def act(s):
        key = (s.bel.map.tobytes(), s.bel.pose, s.bel.items())
        taken = s.taken.setdefault(key, set())
        if len(taken) >= NA:
            taken.clear()
        if s.held is None or s.age >= s.hold:
            s.held = s.choose_event(); s.age = 0; s.plans += 1
            if s.first is None:
                s.first = s.held
        ev = s.held
        if ev is None:
            # fallback: ask System 1 about the goal directly (never random toggle!)
            G = s.goal_asks(s.bel.map)
            if G:
                qs = s.Q.ask([(s.bel.map, s.bel.pose, s.bel.items(), *e) for e in G])
                qa = qs[int(qs.max(1).argmax())].copy()
                qa[list(taken)] = -1; a = int(qa.argmax()); s.noplan += 1
            else:
                a = min(set(range(NA)) - taken); s.noplan += 1
        else:
            qa = s.Q.ask([(s.bel.map, s.bel.pose, s.bel.items(), *ev)])[0].copy()
            a = int(qa.argmax())
            if a in taken:
                # re-check System 2 once if the best action was already tried here
                s.held = s.choose_event(); s.age = 0; s.plans += 1
                ev2 = s.held or ev
                qa = s.Q.ask([(s.bel.map, s.bel.pose, s.bel.items(), *ev2)])[0].copy()
                a = int(qa.argmax())
                if a in taken:
                    s.noplan += 1; qa[list(taken)] = -1; a = int(qa.argmax())
        s.age += 1; taken.add(a)
        return a


class Handed(Sys1):
    """Upper bound: handed the sequence of true events ((x, y), o, ea) from the rules agent's solution."""
    def reset(s, g, p):
        super().reset(g, p); s.k = 0

    def observe(s, a, g, p):
        ev = super().observe(a, g, p)
        if s.k < len(s.todo):
            tx, to, tea = s.todo[s.k]
            if to == s.Q.ARR:
                if s.bel.pose[:2] == tx:
                    s.k += 1; s.held = None
            elif any(((x, y), c2, a) == (tx, to, tea) for x, y, c, c2 in ev):
                s.k += 1; s.held = None
        return ev

    def choose_event(s):
        return s.todo[s.k] if s.k < len(s.todo) else None


class Planner162(agent161.Agent):
    """Card 162's System 2 event selector: identical condition/blocker chaining, returning the target event (X, o, a)
    instead of executing primitive walk/action steps."""
    def plan_event(s):
        for goal in s.goals:
            if goal[0] == "arrive":
                for X in s.cells(goal[1]):
                    r = s.go_ev([(X[0], X[1], d) for d in range(4)], (), (), (X, -1, -1))
                    if r != agent.FAIL:
                        return r
            else:
                for X in s.cells(goal[1]):
                    r = s.want_ev(X, goal[2], goal[3], (), ())
                    if r != agent.FAIL:
                        return r
        return None

    def want_ev(s, X, o, a_req, stack, chain):
        c = int(s.bel.map[X]); items = s.bel.items()
        if (X, o) in chain or len(chain) > 8 or s.fail[(X, o, items)] >= 2:
            return agent.FAIL
        st = s.M.statement(c, o, a_req)
        if st is None:
            return agent.FAIL
        free = any(g < 0 or g == FAR for g in st["G"])
        targets = [] if free else [(X[0] - dx, X[1] - dy, d) for dx, dy, d in map(gid_inv, st["G"])]
        stack2 = stack + (tuple(targets),) if targets else stack; chain2 = chain + ((X, o),)
        for f in st["need"]:
            if not agent.holds(f, items):
                return s.want_in_ev(f, stack2, chain2)
        unmet = []
        refuted = os.environ.get("SURPRISE") == "1" and s.fail[(X, o, items)] >= 1
        if s.p(c, o, st["a"], st["G"], items) < 0.5 or refuted:
            unmet = [("it", it) for it in sorted(items) if s.p(c, o, st["a"], st["G"], items - {it}) >= 0.5]
            if not refuted:
                unmet = unmet[:1]
        if unmet and (free or s.free_reach(targets)):
            for u in unmet:
                r = s.want_out_ev(u, stack2, chain2)
                if r != agent.FAIL:
                    return r
        ev = (X, int(o), int(st["a"]))
        if free:
            return ev
        return s.go_ev(targets, stack, chain2, ev)

    def want_in_ev(s, f, stack, chain):
        ak = f[1][1] if f[0] == "it" else f[1]; cands = []
        for k in np.unique(s.bel.map):
            if f[0] == "it" and k != f[1][0]:
                continue
            for o in s.M.outcomes(int(k), ak):
                cands += [(X, o) for X in s.cells(k)]
        x0, y0, _ = s.bel.pose
        for X, o in sorted(cands, key=lambda q: abs(q[0][0] - x0) + abs(q[0][1] - y0)):
            r = s.want_ev(X, o, ak, stack, chain)
            if r != agent.FAIL:
                return r
        return agent.FAIL

    def want_out_ev(s, f, stack, chain):
        items = s.bel.items()
        for k, ak in [it for it in items if (it == f[1] if f[0] == "it" else it[1] == f[1])]:
            for ce in np.unique(s.bel.map):
                ce = int(ce)
                if k not in s.M.outcomes(ce):
                    continue
                st = s.M.statement(ce, k)
                for Z in s.cells(ce)[:agent.N_TRY]:
                    if s.forward_ok(Z, ce, k, st, stack):
                        r = s.want_ev(Z, k, None, stack, chain)
                        if r != agent.FAIL:
                            return r
        return agent.FAIL

    def go_ev(s, targets, stack, chain, ev):
        mp = s.bel.map; W, H = mp.shape; x, y, d = s.bel.pose; src = (x * H + y) * 4 + d
        ts = [(tx * H + ty) * 4 + td for tx, ty, td in targets if 0 <= tx < W and 0 <= ty < H]
        if src in ts:
            return ev
        dist, prev = s.paths(mp, s.bel.items(), src); ts = [t for t in ts if t in dist]
        if not ts:
            return agent.FAIL
        t = min(ts, key=dist.get); path = []
        while t != src:
            t, a, lab = prev[t]; path.append((a, lab))
        for a, lab in path[::-1]:
            if lab is not None:
                return s.want_ev(lab[0], lab[1], None, stack + (tuple(targets),), chain)
        return ev


class ArmA(Sys1):
    """Arm A: Card 162's condition-based System 2 chooses the next event; System 1 (q) chooses every primitive action."""
    def __init__(s, qmodel, emodel, goals, hold=HOLD):
        super().__init__(qmodel, goals, hold); s.P = Planner162(emodel, goals)

    def reset(s, g, p):
        super().reset(g, p); s.P.reset(g, p); s.bel = s.P.bel

    def observe(s, a, g, p):
        items = s.bel.items(); ev = s.P.observe(a, g, p)
        if ev or s.bel.items() != items:
            s.held = None; s.age = 0
        elif s.held is not None and s.held[1] != s.Q.ARR and a == s.held[2]:
            # System 1 attempted the achieving action at a pose the model's statement allows, and the expected event did not come
            X, o, ea = s.held; c = int(s.bel.map[X]); st = s.P.M.statement(c, o, ea)
            px, py, pd = s.bel.pose
            from grid import gid, R
            dx, dy = X[0] - px, X[1] - py
            in_G = st is not None and (any(g < 0 or g == FAR for g in st["G"]) or (abs(dx) <= R and abs(dy) <= R and gid(dx, dy, pd) in st["G"]))
            if in_G:
                s.P.fail[(X, o, items)] += 1; s.held = None; s.age = 0
        return ev

    def choose_event(s):
        r = s.P.plan_event()
        if r is None:
            return None
        X, o, a = r
        return (X, s.Q.ARR, s.Q.ARR_ACT) if o == -1 else (X, o, a)


class ArmB(Sys1):
    """Arm B: Pure System 2 + System 1 (ZERO Dijkstra, ZERO move model).
    System 2 runs event-level search over imagined states using ONLY:
      - the cell effect model M (which code changes exist and whether current items permit them), and
      - single-event q (which events are directly reachable now, V(s; E) >= tau).
    System 1 (q) executes every primitive action toward the first event of the found chain.
    """
    def __init__(s, qmodel, emodel, goals, hold=HOLD, tau=TAU, max_exp=MAX_EXP, max_depth=MAX_DEPTH):
        super().__init__(qmodel, goals, hold); s.M = emodel; s.tau = tau; s.max_exp = max_exp; s.max_depth = max_depth
        s._eff = {}; s._feas = {}; s.fail = {}
        # precompute known (c -> o, a) effects and their allowed relative agent placements G from the cell effect model
        for c in list(emodel.cell):
            effs = []
            for o in emodel.cell[c]:
                if o == c:
                    continue
                for a in range(NA):
                    st = emodel.statement(c, o, a)
                    if st is not None and st["a"] == a:
                        effs.append((int(o), int(a), tuple(st["G"])))
            if effs:
                s._eff[int(c)] = effs

    def reset(s, g, p):
        super().reset(g, p); s.fail = {}

    def observe(s, a, g, p):
        items = s.bel.items(); ev = super().observe(a, g, p)
        if not ev and s.held is not None and s.held[1] != s.Q.ARR and a == s.held[2]:
            X, o, ea = s.held
            if abs(X[0] - s.bel.pose[0]) + abs(X[1] - s.bel.pose[1]) == 1:
                k = (X, o, ea, items); s.fail[k] = s.fail.get(k, 0) + 1; s.held = None; s.age = 0
        return ev

    def feasible(s, c, o, a, G, items):
        key = (c, o, a, items)
        if key not in s._feas:
            s._feas[key] = max(s.M.p_cell(c, a, g, items).get(o, 0.0) for g in G) >= 0.5
        return s._feas[key]

    def imagine(s, mp, pose, items, X, o, a):
        c = int(mp[X]); mp2 = mp.copy(); mp2[X] = o; it = set(items)
        for i in sorted(it):
            if i[0] == o:
                it.discard(i); break
        it.add((c, a))
        # move imagined pose to an empty/passable 4-neighbor of X closest to current pose (so distance stays accurate)
        W, H = mp2.shape; p2 = pose; best = 1e9
        for d, (dx, dy) in enumerate(DIRV):
            nx, ny = X[0] - dx, X[1] - dy
            if 0 <= nx < W and 0 <= ny < H and mp2[nx, ny] == s.Q.bg:
                dist = abs(nx - pose[0]) + abs(ny - pose[1])
                if dist < best:
                    best = dist; p2 = (nx, ny, d)
        return mp2, p2, frozenset(it)

    def candidates(s, mp, pose, items, is_root):
        """feasible events on non-background tokens plus up to 3 open-floor drop targets per held item"""
        xs, ys, cs = s.Q.toks(mp); evs = []; drop_evs = []
        for x, y, c in zip(xs, ys, cs):
            c = int(c); X = (int(x), int(y))
            for o, a, G in s._eff.get(c, ()):
                if is_root and s.fail.get((X, o, a, items), 0) >= 2:
                    continue
                if s.feasible(c, o, a, G, items):
                    evs.append((X, o, a))
        # drop candidates: only for items that can reappear on background
        bg_effs = [(o, a, G) for o, a, G in s._eff.get(s.Q.bg, ()) if any(it[0] == o for it in items) and s.feasible(s.Q.bg, o, a, G, items)]
        if bg_effs:
            W, H = mp.shape; ex, ey = np.nonzero(mp == s.Q.bg)
            cells = [(int(x), int(y)) for x, y in zip(ex, ey) if (int(x), int(y)) != pose[:2]]
            for o, a, G in bg_effs:
                for Z in cells:
                    if not (is_root and s.fail.get((Z, o, a, items), 0) >= 2):
                        drop_evs.append((Z, o, a))
        return evs, drop_evs

    def choose_event(s):
        mp0, pose0, items0 = s.bel.map, s.bel.pose, s.bel.items()
        # 1. Fast path (System 1): is a goal event directly achievable right now?
        G0 = s.goal_asks(mp0)
        G0_ok = []
        for X, o, ea in G0:
            if o == s.Q.ARR:
                G0_ok.append((X, o, ea))
            else:
                c = int(mp0[X]); eff = [G for oo, aa, G in s._eff.get(c, ()) if oo == o and aa == ea]
                if eff and s.feasible(c, o, ea, eff[0], items0) and s.fail.get((X, o, ea, items0), 0) < 2:
                    G0_ok.append((X, o, ea))
        if G0_ok:
            vg = s.Q.ask([(mp0, pose0, items0, *e) for e in G0_ok]).max(1)
            j = int(vg.argmax())
            if vg[j] >= s.tau:
                return G0_ok[j]

        # 2. Event-level forward best-first search over directly reachable single events
        seen = {(mp0.tobytes(), items0)}
        # priority queue: (cost, tie, depth, mp, pose, items, first_event)
        pq = [(0.0, 0, 0, mp0, pose0, items0, None)]; tie = 1; exp = 0
        while pq and exp < s.max_exp:
            cost, _, depth, mp, pose, items, first = heapq.heappop(pq); exp += 1
            if depth >= s.max_depth:
                continue
            evs, drop_evs = s.candidates(mp, pose, items, depth == 0)
            goals = s.goal_asks(mp) if depth > 0 else []
            # keep only item-feasible goals
            goals_ok = []
            for X, o, ea in goals:
                if o == s.Q.ARR:
                    goals_ok.append((X, o, ea))
                else:
                    c = int(mp[X]); eff = [G for oo, aa, G in s._eff.get(c, ()) if oo == o and aa == ea]
                    if eff and s.feasible(c, o, ea, eff[0], items):
                        goals_ok.append((X, o, ea))
            all_q = goals_ok + evs + drop_evs
            if not all_q:
                continue
            vals = s.Q.ask([(mp, pose, items, *e) for e in all_q]).max(1)
            ng, ne = len(goals_ok), len(evs)
            if ng and vals[:ng].max() >= s.tau:
                s.expansions += exp
                return first if first is not None else goals_ok[int(vals[:ng].argmax())]
            # collect directly reachable non-drop events
            reach = [(float(vals[ng + i]), evs[i]) for i in range(ne) if vals[ng + i] >= s.tau]
            # for drop events, keep up to 3 open-floor reachable drop cells per (o, a) so branching stays compact
            by_drop = {}
            for i, e in enumerate(drop_evs):
                v = float(vals[ng + ne + i])
                if v >= s.tau:
                    X = e[0]; W, H = mp.shape
                    # count non-background 4-neighbors (prefer open cells that do not plug doorways)
                    occ = sum(1 for dx, dy in DIRV if not (0 <= X[0] + dx < W and 0 <= X[1] + dy < H and mp[X[0] + dx, X[1] + dy] == s.Q.bg))
                    by_drop.setdefault((e[1], e[2]), []).append((occ, -v, e))
            for k_drop, cands in by_drop.items():
                cands.sort()
                for _, neg_v, e in cands[:3]:
                    reach.append((-neg_v, e))
            for v, e in reach:
                mp2, pose2, it2 = s.imagine(mp, pose, items, *e)
                sk = (mp2.tobytes(), it2)
                if sk in seen:
                    continue
                seen.add(sk); tie += 1
                # each event link adds 1.0 + small tie-breaker -log(v) so shorter event chains win first
                heapq.heappush(pq, (cost + 1.0 - 0.1 * np.log(max(v, 1e-3)), tie, depth + 1, mp2, pose2, it2, e if first is None else first))
        s.expansions += exp
        return None
