"""Card 165: Zero-Dijkstra System 2 (event-level BFS over cell effect model + Neural Reachability RNet)
paired with Card 164's frozen System 1 (q_reach).

No Dijkstra (paths/graph), no move model (net.Rel), no hand-written want_in/want_out/free_reach/forward_ok/C_BLOCK,
no DIRV/OFF8 hardcoding anywhere in planning or reachability.
"""
import os, numpy as np, torch
from collections import deque
from grid import NA, UNK, OUT, FAR, gid, R, Belief
import act as act164
import reach as Rmod

HOLD = int(os.environ.get("HOLD", 40))
MAX_EXP = int(os.environ.get("MAX_EXP", 250))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 7))
R_TH = float(os.environ.get("R_TH", 0.5))


class RModel:
    """Wrapper around fitted RNet for fast batched reachability queries on GPU."""
    def __init__(s, path):
        ck = torch.load(path, map_location=Rmod.dev, weights_only=False)
        s.NC = ck["NC"]; s.net = Rmod.RNet(s.NC).to(Rmod.dev)
        s.net.load_state_dict(ck["net"]); s.net.eval()
        with torch.no_grad():
            s.al = s.net.alpha().cpu().numpy()
            s.bt = s.net.beta().cpu().numpy()
            s.gm = s.net.gamma().cpu().numpy()
        s.act_offs = [Rmod.OFFS_ALL[i] for i in range(len(s.gm)) if s.gm[i] >= R_TH]
        s.step_offs = [Rmod.OFFS_STEP[i] for i in range(len(s.bt)) if s.bt[i] >= R_TH]
        s._cache = {}

    def query(s, items_list):
        """items_list: list of (mp, (px, py)) all of same shape (W, H) -> list of (r_arr [W, H], r_act [W, H])."""
        out = [None] * len(items_list); todo = []; todo_ix = []
        for i, (mp, pxy) in enumerate(items_list):
            key = (mp.shape, mp.tobytes(), pxy)
            if key in s._cache:
                out[i] = s._cache[key]
            else:
                todo.append((mp, pxy, key)); todo_ix.append(i)
        if todo:
            mps = torch.as_tensor(np.stack([x[0] for x in todo]), dtype=torch.long, device=Rmod.dev)
            px = torch.as_tensor([x[1][0] for x in todo], dtype=torch.long, device=Rmod.dev)
            py = torch.as_tensor([x[1][1] for x in todo], dtype=torch.long, device=Rmod.dev)
            with torch.no_grad():
                ra, ract = s.net.propagate(mps, px, py, K=64, early_stop=True, binary=True)
            ra = ra.cpu().numpy(); ract = ract.cpu().numpy()
            for j, i in enumerate(todo_ix):
                res = (ra[j], ract[j]); s._cache[todo[j][2]] = res; out[i] = res
        return out


class ZeroDijkstraAgent(act164.Sys1):
    """System 2: batched event-level BFS over cell effect model (M) + neural reachability (RM),
    with held expected event and P14 surprise recovery.  System 1: Card 164's frozen q_reach."""
    def __init__(s, qmodel, emodel, rmodel, goals, hold=HOLD, max_exp=MAX_EXP, max_depth=MAX_DEPTH):
        super().__init__(qmodel, goals, hold)
        s.M = emodel; s.RM = rmodel; s.max_exp = max_exp; s.max_depth = max_depth
        s._eff = {}; s._feas = {}; s._g_of = {}
        for c in list(emodel.cell):
            effs = []
            for o in emodel.cell[c]:
                if o == c:
                    continue
                for a in range(NA):
                    st = emodel.statement(c, o, a)
                    if st is not None and st["a"] == a:
                        effs.append((int(o), int(a), tuple(st["G"])))
                        s._g_of[(int(c), int(o), int(a))] = tuple(st["G"])
            if effs:
                s._eff[int(c)] = effs
        # codes that can ever reappear on background (drop targets)
        s._bg_effs = list(s._eff.get(s.Q.bg, ()))

    def reset(s, g, p):
        super().reset(g, p); s.fail = {}; s.RM._cache.clear()

    def observe(s, a, g, p):
        items = s.bel.items(); ev = super().observe(a, g, p)
        if not ev and s.held is not None and s.held[1] != s.Q.ARR and a == s.held[2]:
            X, o, ea = s.held; c = int(s.bel.map[X])
            G = s._g_of.get((c, o, ea))
            px, py, pd = s.bel.pose; dx, dy = X[0] - px, X[1] - py
            in_G = G is not None and (any(gg < 0 or gg == FAR for gg in G) or (abs(dx) <= R and abs(dy) <= R and gid(dx, dy, pd) in G))
            if in_G:
                k = (X, o, ea, items); s.fail[k] = s.fail.get(k, 0) + 1; s.held = None; s.age = 0
        return ev

    def feasible(s, c, o, a, G, items):
        key = (c, o, a, items)
        if key not in s._feas:
            s._feas[key] = max(s.M.p_cell(c, a, g, items).get(o, 0.0) for g in G) >= 0.5
        return s._feas[key]

    def imagine(s, mp, pxy, items, r_arr, X, o, a):
        """Imagine event (X, o, a) whole: new code at X, memory updated, agent placed at a reachable learned interaction neighbor of X."""
        c = int(mp[X]); mp2 = mp.copy(); mp2[X] = o; it = set(items)
        for i in sorted(it):
            if i[0] == o:
                it.discard(i); break
        it.add((c, a))
        # pick a cell p' that was reachable (r_arr >= R_TH) AND within RM's learned interaction offsets of X
        W, H = mp.shape; p2 = pxy; best = 1e9
        for ox, oy in s.RM.act_offs:
            nx, ny = X[0] - ox, X[1] - oy
            if 0 <= nx < W and 0 <= ny < H and r_arr[nx, ny] >= R_TH:
                d = abs(nx - pxy[0]) + abs(ny - pxy[1])
                if d < best:
                    best = d; p2 = (nx, ny)
        return mp2, p2, frozenset(it)

    def is_refuted(s, X, o, ea, items):
        """True if (X, o, ea) previously failed under some items_f that is a subset of current items (interfering item still present)."""
        for (fx, fo, fea, fitems), cnt in s.fail.items():
            if cnt >= 1 and (fx, fo, fea) == (X, o, ea) and fitems.issubset(items):
                return True
        return False

    def check_goals(s, mp, items, r_arr, r_act, is_root):
        for X, o, ea in s.goal_asks(mp):
            if o == s.Q.ARR:
                if r_arr[X] >= R_TH:
                    return (X, o, ea)
            else:
                if r_act[X] < R_TH:
                    continue
                if s.is_refuted(X, o, ea, items):
                    continue
                c = int(mp[X]); G = s._g_of.get((c, o, ea))
                if G is not None and s.feasible(c, o, ea, G, items):
                    return (X, o, ea)
        return None

    def expand_events(s, mp, pxy, items, r_arr, r_act, is_root, changed_cells):
        """Directly reachable, item-feasible events on mp (non-background tokens + up to 2 open-space drops per held token)."""
        xs, ys, cs = s.Q.toks(mp); evs = []
        for x, y, c in zip(xs, ys, cs):
            c = int(c); X = (int(x), int(y))
            if r_act[X] < R_TH:
                continue
            for o, a, G in s._eff.get(c, ()):
                if o in changed_cells.get(X, ()):
                    continue  # do not revert X back to an earlier code in the same imagined chain
                if s.is_refuted(X, o, a, items):
                    continue
                if s.feasible(c, o, a, G, items):
                    evs.append((X, o, a))
        # drop events: bring an out-of-view token back onto reachable background
        bg_ok = [(o, a, G) for o, a, G in s._bg_effs if any(it[0] == o for it in items) and s.feasible(s.Q.bg, o, a, G, items)]
        if bg_ok:
            W, H = mp.shape; ex, ey = np.nonzero((mp == s.Q.bg) & (r_act >= R_TH))
            al = s.RM.al
            # prefer nearest reachable empty cells (minimum detour; RNet itself prunes any cell that blocks downstream reachability),
            # plus one open-floor fallback (lowest occ) in case all adjacent cells are chokepoints
            by_dist = []; by_open = []
            for x, y in zip(ex, ey):
                Z = (int(x), int(y))
                if Z == pxy:
                    continue
                d = abs(Z[0] - pxy[0]) + abs(Z[1] - pxy[1])
                occ = sum(1 for ox, oy in s.RM.step_offs
                          if not (0 <= Z[0] + ox < W and 0 <= Z[1] + oy < H and al[min(int(mp[Z[0] + ox, Z[1] + oy]), s.RM.NC - 1)] >= R_TH))
                by_dist.append((d, occ, Z)); by_open.append((occ, d, Z))
            by_dist.sort(); by_open.sort()
            picks = []
            for _, _, Z in by_dist[:2]:
                picks.append(Z)
            for _, _, Z in by_open:
                if Z not in picks:
                    picks.append(Z); break
            for o, a, G in bg_ok:
                for Z in picks:
                    if not (is_root and s.fail.get((Z, o, a, items), 0) >= 1):
                        evs.append((Z, o, a))
        return evs

    def choose_event(s):
        mp0, pxy0, items0 = s.bel.map, s.bel.pose[:2], s.bel.items()
        # Batched BFS over abstract event-states!
        layer = [(mp0, pxy0, items0, None, {})]
        seen = {(mp0.tobytes(), items0)}
        exp = 0
        for depth in range(s.max_depth + 1):
            if not layer or exp >= s.max_exp:
                break
            # One batched GPU reachability call for the ENTIRE BFS layer!
            reach_res = s.RM.query([(st[0], st[1]) for st in layer])
            next_layer = []
            for (mp, pxy, items, first, chg), (r_arr, r_act) in zip(layer, reach_res):
                exp += 1
                g = s.check_goals(mp, items, r_arr, r_act, depth == 0)
                if g is not None:
                    s.expansions += exp
                    return g if first is None else first
                if depth == s.max_depth or exp >= s.max_exp:
                    continue
                for e in s.expand_events(mp, pxy, items, r_arr, r_act, depth == 0, chg):
                    mp2, pxy2, it2 = s.imagine(mp, pxy, items, r_arr, *e)
                    sk = (mp2.tobytes(), it2)
                    if sk in seen:
                        continue
                    seen.add(sk)
                    chg2 = dict(chg); chg2[e[0]] = chg.get(e[0], ()) + (int(mp[e[0]]),)
                    next_layer.append((mp2, pxy2, it2, e if first is None else first, chg2))
            layer = next_layer
        s.expansions += exp
        return None
