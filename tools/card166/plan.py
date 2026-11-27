"""Card 166: Zero-Dijkstra System 2 powered by ONE unified factored relational world model (WNet),
paired with Card 164's frozen System 1 (q_reach).

Retires Card 160's separate model_mlp.pkl, Table, energy.py, gid/FAR binning, FI/FY memory table,
and statement() count guard — zero Dijkstra, zero move model, zero hand-written chaining procedures.
"""
import os, numpy as np, torch
import act as act164
import world as Wmod

HOLD = int(os.environ.get("HOLD", 40))
MAX_EXP = int(os.environ.get("MAX_EXP", 250))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 7))
R_TH = float(os.environ.get("R_TH", 0.5))


class WModel:
    """Fast batched GPU wrapper around ONE unified WNet for both reachability AND cell-effect queries."""
    def __init__(s, path, ablate_effects=False):
        ck = torch.load(path, map_location=Wmod.dev, weights_only=False)
        s.NC = ck["NC"]; s.ablate_effects = ablate_effects
        s.net = Wmod.WNet(s.NC, D=ck.get("D", 32), H=ck.get("H", 128)).to(Wmod.dev)
        s.net.load_state_dict(ck["net"]); s.net.eval()
        with torch.no_grad():
            s.al = s.net.alpha().cpu().numpy()
            s.bt = s.net.beta().cpu().numpy()
            s.gm = s.net.gamma().cpu().numpy()
        s.act_offs = [Wmod.OFFS_ALL[i] for i in range(len(s.gm)) if s.gm[i] >= R_TH]
        s.act_off_ix = [i for i in range(len(s.gm)) if s.gm[i] >= R_TH] + [Wmod.FAR_IX]
        s.step_offs = [Wmod.OFFS_STEP[i] for i in range(len(s.bt)) if s.bt[i] >= R_TH]
        # Distinct observed transition triples (c -> o, a) grouped by subject code c
        s.eff_by_code = {}
        for c, o, a in ck["obs_effs"]:
            s.eff_by_code.setdefault(int(c), []).append((int(o), int(a)))
        s._reach_cache = {}
        s._feas_cache = {}
        s._pat_cache = {}

    def clear_cache(s):
        s._reach_cache.clear()

    def query_reach(s, items_list):
        """items_list: list of (mp, (px, py)) all of same shape (W, H) -> list of (r_arr [W, H], r_act [W, H])."""
        out = [None] * len(items_list); todo = []; todo_ix = []
        for i, (mp, pxy) in enumerate(items_list):
            key = (mp.shape, mp.tobytes(), pxy)
            if key in s._reach_cache:
                out[i] = s._reach_cache[key]
            else:
                todo.append((mp, pxy, key)); todo_ix.append(i)
        if todo:
            mps = torch.as_tensor(np.stack([x[0] for x in todo]), dtype=torch.long, device=Wmod.dev)
            px = torch.as_tensor([x[1][0] for x in todo], dtype=torch.long, device=Wmod.dev)
            py = torch.as_tensor([x[1][1] for x in todo], dtype=torch.long, device=Wmod.dev)
            with torch.no_grad():
                ra, ract = s.net.propagate(mps, px, py, K=64, early_stop=True, binary=True)
            ra = ra.cpu().numpy(); ract = ract.cpu().numpy()
            for j, i in enumerate(todo_ix):
                res = (ra[j], ract[j]); s._reach_cache[todo[j][2]] = res; out[i] = res
        return out

    def _pack_items(s, items):
        lst = sorted(items)
        M = max(1, len(lst))
        ic = np.zeros((1, M), np.int64); ia = np.zeros((1, M), np.int64); im = np.zeros((1, M), bool)
        for j, (kk, aa) in enumerate(lst):
            ic[0, j] = min(int(kk), s.NC - 1); ia[0, j] = int(aa); im[0, j] = True
        return (
            torch.as_tensor(ic, dtype=torch.long, device=Wmod.dev),
            torch.as_tensor(ia, dtype=torch.long, device=Wmod.dev),
            torch.as_tensor(im, dtype=torch.bool, device=Wmod.dev),
        )

    def feasible(s, c, o, a, items):
        """True iff WNet predicts p(o | c, a, dx, dy, d, items) >= 0.5 at some learned interaction offset (or FAR) and heading."""
        if s.ablate_effects:
            return False
        key = (c, o, a, items)
        if key not in s._feas_cache:
            # Batch all (off_ix in act_off_ix) x (d in 0..3) in one tiny GPU pass (~20 rows)
            offs = s.act_off_ix
            n = len(offs) * 4
            ct = torch.full((n,), int(c), dtype=torch.long, device=Wmod.dev)
            at = torch.full((n,), int(a), dtype=torch.long, device=Wmod.dev)
            ot = torch.tensor([ox for ox in offs for _ in range(4)], dtype=torch.long, device=Wmod.dev)
            dt = torch.tensor([d for _ in offs for d in range(4)], dtype=torch.long, device=Wmod.dev)
            ic, ia, im = s._pack_items(items)
            ic, ia, im = ic.expand(n, -1), ia.expand(n, -1), im.expand(n, -1)
            with torch.no_grad():
                probs = torch.softmax(s.net.cell_logits(ct, at, ot, dt, ic, ia, im), -1)
                pmax = float(probs[:, int(o)].max().item())
            s._feas_cache[key] = (pmax >= 0.5)
        return s._feas_cache[key]

    def p_at(s, c, o, a, dx, dy, d, items):
        """Exact predicted probability p(o | c, a, dx, dy, d, items) at the agent's current relative pose (zero gid!)."""
        if s.ablate_effects:
            return 0.0
        is_far = (abs(dx) > Wmod.RAD) or (abs(dy) > Wmod.RAD)
        off_ix = Wmod.FAR_IX if is_far else Wmod.OFF_ALL_IX[(dx, dy)]
        key = (c, o, a, off_ix, d, items)
        if key not in s._pat_cache:
            ct = torch.tensor([int(c)], dtype=torch.long, device=Wmod.dev)
            at = torch.tensor([int(a)], dtype=torch.long, device=Wmod.dev)
            ot = torch.tensor([off_ix], dtype=torch.long, device=Wmod.dev)
            dt = torch.tensor([int(d)], dtype=torch.long, device=Wmod.dev)
            ic, ia, im = s._pack_items(items)
            with torch.no_grad():
                probs = torch.softmax(s.net.cell_logits(ct, at, ot, dt, ic, ia, im), -1)
                s._pat_cache[key] = float(probs[0, int(o)].item())
        return s._pat_cache[key]

    def predict_batch(s, rows):
        """rows: list of (c, a, dx, dy, d, items) -> np.ndarray [N, NC] probabilities (for diagnostics & gate)."""
        if not rows:
            return np.zeros((0, s.NC), np.float32)
        n = len(rows)
        M = max(1, max(len(r[5]) for r in rows))
        c_a = np.zeros(n, np.int64); a_a = np.zeros(n, np.int64); d_a = np.zeros(n, np.int64)
        pf = np.zeros((n, 3), np.float32)
        ic = np.zeros((n, M), np.int64); ia = np.zeros((n, M), np.int64); im = np.zeros((n, M), bool)
        for i, (c, a, dx, dy, d, items) in enumerate(rows):
            c_a[i] = min(int(c), s.NC - 1); a_a[i] = int(a); d_a[i] = int(d)
            if abs(dx) > Wmod.RAD or abs(dy) > Wmod.RAD:
                pf[i] = (0.0, 0.0, 1.0)
            else:
                pf[i] = (dx / float(Wmod.RAD), dy / float(Wmod.RAD), 0.0)
            for j, (kk, aa) in enumerate(sorted(items)):
                ic[i, j] = min(int(kk), s.NC - 1); ia[i, j] = int(aa); im[i, j] = True
        out = []
        with torch.no_grad():
            for b in range(0, n, 4096):
                e = slice(b, b + 4096)
                lg = s.net.cell_logits_pos(
                    torch.as_tensor(c_a[e], device=Wmod.dev),
                    torch.as_tensor(a_a[e], device=Wmod.dev),
                    torch.as_tensor(pf[e], device=Wmod.dev),
                    torch.as_tensor(d_a[e], device=Wmod.dev),
                    torch.as_tensor(ic[e], device=Wmod.dev),
                    torch.as_tensor(ia[e], device=Wmod.dev),
                    torch.as_tensor(im[e], device=Wmod.dev),
                )
                out.append(torch.softmax(lg, -1).cpu().numpy())
        return np.concatenate(out, 0)


class ZeroDijkstraAgent(act164.Sys1):
    """System 2: batched event-level BFS over ONE unified world model (WM),
    with held expected event and P14 superset surprise recovery. System 1: Card 164's frozen q_reach."""
    def __init__(s, qmodel, wmodel, goals, hold=HOLD, max_exp=MAX_EXP, max_depth=MAX_DEPTH):
        super().__init__(qmodel, goals, hold)
        s.WM = wmodel; s.max_exp = max_exp; s.max_depth = max_depth
        s._bg_effs = list(wmodel.eff_by_code.get(s.Q.bg, ()))

    def reset(s, g, p):
        super().reset(g, p); s.fail = {}; s.WM.clear_cache()

    def observe(s, a, g, p):
        items = s.bel.items(); ev = super().observe(a, g, p)
        if not ev and s.held is not None and s.held[1] != s.Q.ARR and a == s.held[2]:
            X, o, ea = s.held; c = int(s.bel.map[X])
            px, py, pd = s.bel.pose
            # Direct WNet surprise check at the agent's exact relative pose (zero gid/G/statement!):
            # trigger if WNet predicts the event at this pose (under current items or empty items, i.e., valid interaction pose)
            dx, dy = X[0] - px, X[1] - py
            if s.WM.p_at(c, o, ea, dx, dy, pd, items) >= 0.5 or s.WM.p_at(c, o, ea, dx, dy, pd, frozenset()) >= 0.5:
                k = (X, o, ea, items); s.fail[k] = s.fail.get(k, 0) + 1; s.held = None; s.age = 0
        return ev

    def imagine(s, mp, pxy, items, r_arr, X, o, a):
        """Imagine event (X, o, a) whole: new code at X, memory updated, agent placed at a reachable learned interaction neighbor of X."""
        c = int(mp[X]); mp2 = mp.copy(); mp2[X] = o; it = set(items)
        for i in sorted(it):
            if i[0] == o:
                it.discard(i); break
        it.add((c, a))
        W, H = mp.shape; p2 = pxy; best = 1e9
        for ox, oy in s.WM.act_offs:
            nx, ny = X[0] - ox, X[1] - oy
            if 0 <= nx < W and 0 <= ny < H and r_arr[nx, ny] >= R_TH:
                d = abs(nx - pxy[0]) + abs(ny - pxy[1])
                if d < best:
                    best = d; p2 = (nx, ny)
        return mp2, p2, frozenset(it)

    def is_refuted(s, X, o, ea, items):
        for (fx, fo, fea, fitems), cnt in s.fail.items():
            if cnt >= 1 and (fx, fo, fea) == (X, o, ea) and fitems.issubset(items):
                return True
        return False

    def check_goals(s, mp, items, r_arr, r_act):
        for X, o, ea in s.goal_asks(mp):
            if o == s.Q.ARR:
                if r_arr[X] >= R_TH:
                    return (X, o, ea)
            else:
                if r_act[X] < R_TH:
                    continue
                if s.is_refuted(X, o, ea, items):
                    continue
                c = int(mp[X])
                if s.WM.feasible(c, o, ea, items):
                    return (X, o, ea)
        return None

    def expand_events(s, mp, pxy, items, r_arr, r_act, is_root, changed_cells):
        xs, ys, cs = s.Q.toks(mp); evs = []
        for x, y, c in zip(xs, ys, cs):
            c = int(c); X = (int(x), int(y))
            if r_act[X] < R_TH:
                continue
            for o, a in s.WM.eff_by_code.get(c, ()):
                if o in changed_cells.get(X, ()):
                    continue
                if s.is_refuted(X, o, a, items):
                    continue
                if s.WM.feasible(c, o, a, items):
                    evs.append((X, o, a))
        # drop events: bring an out-of-view token back onto reachable background
        bg_ok = [(o, a) for o, a in s._bg_effs if any(it[0] == o for it in items) and s.WM.feasible(s.Q.bg, o, a, items)]
        if bg_ok:
            W, H = mp.shape; ex, ey = np.nonzero((mp == s.Q.bg) & (r_act >= R_TH))
            al = s.WM.al
            by_dist = []; by_open = []
            for x, y in zip(ex, ey):
                Z = (int(x), int(y))
                if Z == pxy:
                    continue
                d = abs(Z[0] - pxy[0]) + abs(Z[1] - pxy[1])
                occ = sum(1 for ox, oy in s.WM.step_offs
                          if not (0 <= Z[0] + ox < W and 0 <= Z[1] + oy < H and al[min(int(mp[Z[0] + ox, Z[1] + oy]), s.WM.NC - 1)] >= R_TH))
                by_dist.append((d, occ, Z)); by_open.append((occ, d, Z))
            by_dist.sort(); by_open.sort()
            picks = []
            for _, _, Z in by_dist[:2]:
                picks.append(Z)
            for _, _, Z in by_open:
                if Z not in picks:
                    picks.append(Z); break
            for o, a in bg_ok:
                for Z in picks:
                    if not (is_root and s.fail.get((Z, o, a, items), 0) >= 1):
                        evs.append((Z, o, a))
        return evs

    def choose_event(s):
        mp0, pxy0, items0 = s.bel.map, s.bel.pose[:2], s.bel.items()
        layer = [(mp0, pxy0, items0, None, {})]
        seen = {(mp0.tobytes(), items0)}
        exp = 0
        for depth in range(s.max_depth + 1):
            if not layer or exp >= s.max_exp:
                break
            reach_res = s.WM.query_reach([(st[0], st[1]) for st in layer])
            next_layer = []
            for (mp, pxy, items, first, chg), (r_arr, r_act) in zip(layer, reach_res):
                exp += 1
                g = s.check_goals(mp, items, r_arr, r_act)
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
