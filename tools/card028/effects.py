"""Card 028: one learned model of what every action does.

The agent's facts are card 016's egocentric view (13 x 13 tiles centred on it, facing up, plus the held
place), each tile an exact pixel appearance (card 027). From its own stored transitions it counts:

  moves (left, right, forward): where each tile goes when the view changes (one fixed map per move: for
      each place after, the place before whose appearance predicts it best, and how it predicts it),
      what tiles entering the view look like, and whether the move changes the view at all, by the
      appearance in front (changed, unchanged, episode ended);
  pick up, drop, toggle: which places change, and the outcome per appearance in front (and held, for
      drop); where one key has several outcomes, card 010's evidence finder explains which, over the
      atoms "the held tile is X" and "X is in view".

Thinking: the agent keeps its facts in the frame of its first view of the episode and places each new
view in it by matching (the poses are the compositions of the learned move maps). A new conditions class
answers card 026's two questions on facts, with closer.Approach's logic (card 024's step measure) and the
learned effects in place of the simulator. Tree growth (card 026), its labels, the depth-first search,
walking and card 027's target rule all ask it; nothing the agent thinks with reads the simulator. The
simulator still runs the real actions and renders the view; the evaluator reads it for the checks.

Stages, per world (key, switch):
  gate: facts from pixels (card 027's check); arm 1, the upper bound (card 027's final setup);
        the model learned from the stored transitions; criterion 1 on held-out transitions.
  main: arm 2's tree grown with learned conditions; criterion 2 (learned against exact conditions on
        held-out situations); acting: arm 2, arm 3 (learned conditions on card 027's tree), arm 4
        (effects without rules, on arm 2's tree).

Run: bin/prun python tools/card028/effects.py [--episodes 5000] [--layouts 500] [--heldout 1000]
     [--out runs/028_effects.json]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card027"))
import kinds as Kd  # noqa: E402

from worldmodel.envs.keydoor_render import encode_logic_states  # noqa: E402
from worldmodel.logic_conditions import evidence_rules  # noqa: E402

closer, D, dl, ld = Kd.closer, Kd.D, Kd.dl, Kd.ld

R = Kd.R
W13 = 2 * R + 1
NV = W13 * W13                       # the 13 x 13 view places
NPL = NV + W13                       # plus the held row
CENTRE, FRONT, HELD = R * W13 + R, (R - 1) * W13 + R, NV
LEFT, RIGHT, FWD, PICK, DROP, TOG = range(6)
MOVES = (LEFT, RIGHT, FWD)
CHANGED, UNCHANGED, ENDED = 0, 1, 2
SAME = 255                                     # in an outcome: this place keeps its appearance
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))      # (column, row) steps in turning order
BUDGET = 200
NAME = Kd.NAME_OF_APP                          # evaluator names, reports only


def pool20():
    return Kd.pool20()


def views(codes, st, chunk=50000):
    """Top-down codes and states -> card 016's egocentric views as appearances, (B, 182) uint8."""
    out = np.zeros((len(codes), NPL), np.uint8)
    for s in range(0, len(codes), chunk):
        out[s:s + chunk] = Kd.APP[Kd.ego(codes[s:s + chunk], st[s:s + chunk])].reshape(-1, NPL)
    return out


def see(lay, s):
    """The environment's observation of state s: the egocentric view as appearances."""
    return Kd.APP[Kd.ego(encode_logic_states(lay, [s]), np.array([s[:3]]))].reshape(NPL).astype(np.uint8)


def nm(u):
    return "same" if int(u) == SAME else NAME.get(int(u)) or f"app {int(u)}"


# ---------------------------------------------------------------- the model, learned by counting

class Model:
    """What the agent learned (plain arrays and dicts; the workers get it by fork)."""


class Entry:
    """Outcomes of one (action, front appearance[, held appearance]) key. An outcome is the tuple of
    appearances after at the places that change; None is "nothing changes"."""

    def __init__(self):
        self.single, self.multi, self.default, self.optimistic = None, None, None, None


M = None


def fit_map(V0, V1, dev):
    """One move's map: for each place after, the place before whose appearance predicts it best
    (counting every pair of places and appearances), and the appearance table that predicts it; places
    whose appearance after never varies are "entering" with that appearance."""
    import torch
    apps = np.unique(np.concatenate([V0.ravel(), V1.ravel()]))
    K = len(apps)
    lut = np.zeros(256, np.int64)
    lut[apps] = np.arange(K)
    N, P = V0.shape
    C = torch.zeros(P * K, P * K, device=dev)
    for s in range(0, N, 4096):
        o0 = torch.nn.functional.one_hot(torch.as_tensor(lut[V0[s:s + 4096]], device=dev), K).float()
        o1 = torch.nn.functional.one_hot(torch.as_tensor(lut[V1[s:s + 4096]], device=dev), K).float()
        C += o0.reshape(len(o0), P * K).T @ o1.reshape(len(o1), P * K)
    C4 = C.reshape(P, K, P, K)
    best = C4.amax(3).sum(1)                         # [p, q]: rows where p's appearance predicts q's
    count = C4[0].sum(0)                             # [q, v]: how often place q shows v after
    const = (count.amax(1) == N).cpu().numpy()
    src = best.argmax(0).cpu().numpy().astype(np.int16)
    share = (best.amax(0) / N).cpu().numpy()
    ent = apps[count.argmax(1).cpu().numpy()].astype(np.uint8)
    src[const] = -1
    ar = torch.arange(P, device=dev)
    Cs = C4[torch.as_tensor(src.clip(0).astype(np.int64), device=dev), :, ar, :]    # [q, u, v]
    seen, arg = (Cs.sum(2) > 0).cpu().numpy(), Cs.argmax(2).cpu().numpy()
    trans = np.tile(np.arange(256, dtype=np.uint8), (P, 1))
    changed = {}
    for q in range(P):
        if const[q]:
            continue
        for ui in np.flatnonzero(seen[q]):
            trans[q, apps[ui]] = apps[arg[q, ui]]
            if apps[arg[q, ui]] != apps[ui]:
                changed.setdefault(q, []).append((int(apps[ui]), int(apps[arg[q, ui]])))
    return src, ent, trans, share, const, changed


def rule_data(V0, y, w):
    """Card 010's data for one outcome: atoms "held = X" and "X in view" (positive atoms only)."""
    held = V0[:, HELD]
    pres = np.zeros((len(V0), 256), bool)
    pres[np.arange(len(V0))[:, None], V0[:, :NV]] = True
    hv = np.unique(held)
    pv = np.flatnonzero(pres.any(0))
    X = np.concatenate([held[:, None] == hv[None], pres[:, pv]], 1)
    uniq, inv = np.unique(X, axis=0, return_inverse=True)
    inv = inv.ravel()

    class Dat:
        pass
    d = Dat()
    d.X = uniq
    d.n = np.bincount(inv, w, len(uniq))
    d.k = np.bincount(inv, w * y, len(uniq))
    d.atoms = [(0, int(u)) for u in hv] + [(1, int(u)) for u in pv]      # 0: held = u; 1: u in view
    return d


def rule_rates(d, rules):
    out, remaining = [], np.ones(len(d.n), bool)
    for rule in rules:
        m = remaining & d.X[:, rule].all(1)
        out.append((tuple(d.atoms[j] for j in rule), float(d.k[m].sum() / d.n[m].sum()),
                    float(d.n[m].sum()), float(d.k[m].sum())))
        remaining &= ~m
    N, K = d.n[remaining].sum(), d.k[remaining].sum()
    return out, float(K / N) if N else 0.0, float(N), float(K)


def atom_name(a):
    return f"held = {nm(a[1])}" if a[0] == 0 else f"{nm(a[1])} in view"


def learn(tr, E, V0, V1, log, dev, sub=40000, seed=0):
    m = Model()
    rep = {}
    rng = np.random.default_rng(seed)
    act, w = E["act"], tr["w0"].astype(np.float64)
    out = np.where(tr["term1"].astype(bool), ENDED, np.where(E["view_changed"], CHANGED, UNCHANGED))
    m.src, m.ent, m.trans, m.move_out = {}, {}, {}, {}
    for a in MOVES:
        t0 = time.monotonic()
        rows = np.flatnonzero((act == a) & (out != UNCHANGED))
        n_all = len(rows)
        if len(rows) > sub:
            rows = np.sort(rng.choice(rows, sub, replace=False))
        src, ent, trans, share, const, changed = fit_map(V0[rows], V1[rows], dev)
        m.src[a], m.ent[a], m.trans[a] = src, ent, trans
        cnt = np.zeros((256, 3))
        sel = act == a
        np.add.at(cnt, (V0[sel, FRONT], out[sel]), w[sel])
        mo = np.full(256, UNCHANGED, np.int64)
        ok = cnt.sum(1) > 0
        mo[ok] = cnt[ok].argmax(1)
        m.move_out[a] = mo
        rep[dl.ACTIONS[a]] = {
            "rows_view_changed": int(n_all), "rows_used": int(len(rows)), "entering_places": int(const.sum()),
            "lowest_predicted_share": round(float(share[~const].min()), 5),
            "places_predicted_below_1": int((share[~const] < 1).sum()),
            "tables_that_change_appearance": {f"place {q} from {int(src[q])}": [(nm(u), nm(v)) for u, v in ch]
                                              for q, ch in changed.items()},
            "outcome_by_front": {nm(u): ["changed", "unchanged", "episode ended"][int(mo[u])]
                                 for u in np.flatnonzero(ok)},
            "seconds": round(time.monotonic() - t0, 1)}
    # evaluator: forward's map against a one-row shift toward the agent
    exp = np.full(NPL, -2)
    for r in range(1, W13):
        for c in range(W13):
            exp[r * W13 + c] = (r - 1) * W13 + c
    exp[HELD] = HELD
    chk = [q for q in range(NPL) if exp[q] != -2 and m.src[FWD][q] >= 0 and m.src[FWD][q] != exp[q]]
    rep["forward_map_places_not_a_one_row_shift"] = chk                 # entering (always-wall) places aside
    m.draw = m.trans[FWD][CENTRE].copy()                       # what stands in front, with the agent on it
    qb = np.flatnonzero(m.src[FWD] == CENTRE)
    m.undraw = m.trans[FWD][qb[0]].copy() if len(qb) else np.arange(256, dtype=np.uint8)
    ident = np.arange(256)
    for u, v in zip(ident[m.undraw != ident], m.undraw[m.undraw != ident]):
        if m.draw[v] == v:
            m.draw[v] = u                                      # a pair seen only one way round
    for u, v in zip(ident[m.draw != ident], m.draw[m.draw != ident]):
        if m.undraw[v] == v:
            m.undraw[v] = u
    m.free = (m.move_out[FWD] == CHANGED).tolist()             # forward onto it moves the agent
    m.standable = m.move_out[FWD] != UNCHANGED

    # pick up, drop, toggle
    nmr = np.flatnonzero(np.isin(act, (PICK, DROP, TOG)))
    cp = np.flatnonzero((V0[nmr] != V1[nmr]).any(0))
    m.change_places = cp
    rep["places_changed_by_pickup_drop_toggle"] = [int(q) for q in cp]
    keyarr = np.stack([act[nmr], V0[nmr, FRONT], np.where(act[nmr] == DROP, V0[nmr, HELD], 255)], 1).astype(np.int64)
    uk, inv = np.unique(keyarr, axis=0, return_inverse=True)
    inv = inv.ravel()
    O, B = V1[nmr][:, cp], V0[nmr][:, cp]
    O = np.where(O == B, SAME, O)                              # an outcome says what changed, place by place
    same = (O == SAME).all(1)
    m.table = {}
    rep["outcomes"], rep["rules"] = {}, []
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uk)))]
    for g in range(len(uk)):
        a, u, h = (int(v) for v in uk[g])
        key = (a, u, h) if a == DROP else (a, u)
        rows = nmr[order[bounds[g]:bounds[g + 1]]]
        loc = order[bounds[g]:bounds[g + 1]]
        outs = [None if same[i] else tuple(int(v) for v in O[i]) for i in loc]
        cnt = Counter()
        for o, x in zip(outs, w[rows]):
            cnt[o] += x
        e = Entry()
        kname = f"{dl.ACTIONS[a]} {nm(u)}" + (f", holding {nm(h)}" if a == DROP else "")
        rep["outcomes"][kname] = {("nothing" if o is None else " / ".join(nm(v) for v in o)): round(c, 1)
                                  for o, c in cnt.most_common()}
        if len(cnt) == 1:
            e.single = next(iter(cnt))
            if e.single is not None:
                m.table[key] = e
            continue
        e.default = None if None in cnt else cnt.most_common(1)[0][0]
        e.optimistic = next(o for o, _ in cnt.most_common() if o is not None)
        e.multi = []
        Vk = V0[rows]
        for o in [o for o, _ in cnt.most_common() if o != e.default]:
            y = np.array([oo == o for oo in outs], np.float64)
            d = rule_data(Vk, y, w[rows])
            rules, ev, trace = evidence_rules(d)
            rr, dr, dn, dk = rule_rates(d, rules)
            e.multi.append((o, [(conds, rate) for conds, rate, _, _ in rr], dr))
            rep["rules"].append({
                "key": kname, "outcome": " / ".join(nm(v) for v in o),
                "rules": [{"if": [atom_name(c) for c in conds], "rate": round(rate, 4), "tries_weighted": round(n, 1),
                           "happened_weighted": round(k, 1)} for conds, rate, n, k in rr],
                "otherwise": {"rate": round(dr, 4), "tries_weighted": round(dn, 1), "happened_weighted": round(dk, 1)},
                "tries_stored": int(len(rows)),
                "trace": trace})
        m.table[key] = e
    build_poses(m)
    rep["poses"] = int(len(m.A))
    rep["pose_route_conflicts"], rep["pose_sweeps"] = m.pose_conflicts, m.pose_sweeps
    return m, rep


NEIGH = ((FRONT, 0), (CENTRE + W13, 2), (CENTRE + 1, 1), (CENTRE - 1, 3))   # view place, turns from facing


def facing(A):
    """The direction the agent faces in the first view's grid, from the first of its four neighbours
    (in front, behind, right, left) whose place there is known; -1 if none is."""
    c = int(A[CENTRE])
    for q, k in NEIGH:
        if A[q] >= 0:
            d = (int(A[q]) % W13 - c % W13, int(A[q]) // W13 - c // W13)
            return (DIRS.index(d) - k) % 4 if d in DIRS else -1
    return -1


def build_poses(m):
    """Every pose reachable by composing the learned move maps from the first view: for each pose,
    which place of the first view each view place shows. A pose is where the agent stands and which
    way it faces (declared); routes to the same pose that lose different edge places (entering
    tiles) are merged, and places two routes disagree on are counted (0 if the maps are consistent)."""
    ident = np.arange(NPL, dtype=np.int16)
    poses, ents, index = [ident], [np.zeros(NPL, np.uint8)], {(CENTRE, 0): 0}
    for sweep in range(50):
        grew, conflicts, nxt = False, 0, []
        i = 0
        while i < len(poses):
            row = [-1] * 6
            for a in MOVES:
                src, ent = m.src[a], m.ent[a]
                ok = src >= 0
                A2 = np.where(ok, poses[i][src.clip(0)], -1).astype(np.int16)
                E2 = np.where(ok, ents[i][src.clip(0)], ent).astype(np.uint8)
                if not 0 <= A2[CENTRE] < NV or facing(A2) < 0:
                    continue
                k = (int(A2[CENTRE]), facing(A2))
                j = index.get(k)
                if j is None:
                    if len(poses) >= 5000:
                        raise RuntimeError("more than 5000 poses: the learned move maps do not compose")
                    j = index[k] = len(poses)
                    poses.append(A2)
                    ents.append(E2)
                else:
                    A1 = poses[j]
                    both = (A1 >= 0) & (A2 >= 0)
                    conflicts += int((A1[both] != A2[both]).sum())
                    add = (A1 < 0) & (A2 >= 0)
                    if add.any():
                        A1[add] = A2[add]
                        grew = True
                row[a] = j
            nxt.append(row)
            i += 1
        if not grew:
            break
    m.pose_conflicts, m.pose_sweeps = conflicts, sweep + 1
    m.A, m.Ent, m.nxt = np.stack(poses), np.stack(ents), nxt
    m.cidx = [int(A[CENTRE]) for A in poses]
    m.fidx = [int(A[FRONT]) for A in poses]
    m.efront = [int(E[FRONT]) for E in ents]
    m.hidx = [int(A[HELD]) for A in poses]
    m.chg = [tuple(int(A[q]) for q in m.change_places) for A in poses]
    m.px = [c % W13 for c in m.cidx]
    m.py = [c // W13 for c in m.cidx]
    m.pd = [facing(A) for A in poses]
    m.poses_at = [[] for _ in range(NV)]
    m.facing = [[] for _ in range(NV)]
    for p, (c, f) in enumerate(zip(m.cidx, m.fidx)):
        m.poses_at[c].append(p)
        if 0 <= f < NV:
            m.facing[f].append(p)
    width = max(len(x) for x in m.poses_at)
    m.all_poses = np.arange(len(poses))
    m.at_arr = np.full((NV, width), -1, np.int64)
    for c, ps in enumerate(m.poses_at):
        m.at_arr[c, :len(ps)] = ps


# ---------------------------------------------------------------- conditions on facts

def _turns(d, e):
    return min((e - d) % 4, (d - e) % 4)


class Think:
    """Card 026's two questions ("does node n hold here?", "does action a achieve goal g from here?")
    on facts, with the learned effects: closer.Approach's logic with card 024's step measure. A
    situation is (facts id, pose id, ended): the facts in the frame of the episode's first view, the
    agent's pose in it, and whether an imagined step has ended the episode."""

    def __init__(self, parent, action, rules=True):
        self.parent, self.action, self.rules = parent, action, rules
        self.facts, self.fid = [], {}
        self.memo, self.rsets, self.appr, self.steps, self.pres = {}, {}, {}, {}, {}
        self.computed = 0

    def intern(self, b):
        i = self.fid.get(b)
        if i is None:
            i = self.fid[b] = len(self.facts)
            self.facts.append(b)
        return i

    # -- seeing
    def observe(self, world, V, ended=False, prefer=None):
        """Place a view in the facts: the pose whose predicted view matches it best (the agent's own
        tile aside), then the facts updated from it. Returns (facts array, situation, mismatches, ties)."""
        if world is None:
            f = V.copy()
            f[CENTRE] = M.undraw[V[CENTRE]]
            pose, miss, ties = 0, 0, 1
        else:
            cand = M.all_poses
            G = np.where(M.A >= 0, world[np.maximum(M.A, 0)], M.Ent)
            mism = (G != V).sum(1) - (G[:, CENTRE] != V[CENTRE])
            best = mism.min()
            tied = cand[mism == best]
            pose = prefer if prefer is not None and (tied == prefer).any() else int(tied[0])
            f = world.copy()
            Ap = M.A[pose]
            sel = Ap >= 0
            sel[CENTRE] = False
            f[Ap[sel]] = V[sel]
            f[Ap[CENTRE]] = M.undraw[V[CENTRE]]
            miss, ties = int(best), len(tied)
        return f, (self.intern(f.tobytes()), int(pose), bool(ended)), miss, ties

    def view(self, st):
        f = np.frombuffer(self.facts[st[0]], np.uint8)
        A = M.A[st[1]]
        v = np.where(A >= 0, f[np.maximum(A, 0)], M.Ent[st[1]]).astype(np.uint8)
        v[CENTRE] = M.draw[v[CENTRE]]
        return v

    def presence(self, fid, pose):
        k = (fid, pose)
        p = self.pres.get(k)
        if p is None:
            p = self.pres[k] = frozenset(self.view((fid, pose, False))[:NV].tolist())
        return p

    # -- the learned effects
    def outcome(self, key, fid, pose, held):
        e = M.table.get(key)
        if e is None:
            return None
        if e.multi is None:
            return e.single
        if not self.rules:
            return e.optimistic                     # arm 4: an outcome ever seen counts as possible
        pres = self.presence(fid, pose)
        best, bp = e.default, 0.5                   # declared: reachable when its rate is at least one half
        for o, rules, dr in e.multi:
            p = dr
            for conds, rate in rules:
                if all((held == u) if kind == 0 else (u in pres) for kind, u in conds):
                    p = rate
                    break
            if p >= bp:
                best, bp = o, p
        return best

    def step(self, st, a):
        k = (st, a)
        r = self.steps.get(k)
        if r is not None:
            return r
        fid, pose, ended = st
        if ended:
            r = st
        elif a in MOVES:
            f = self.facts[fid]
            fi = M.fidx[pose]
            u = f[fi] if fi >= 0 else M.efront[pose]
            o = M.move_out[a][u]
            p2 = M.nxt[pose][a]
            r = st if o == UNCHANGED or p2 < 0 else (fid, p2, o == ENDED)
        else:
            f = self.facts[fid]
            fi, hi = M.fidx[pose], M.hidx[pose]
            u, h = (f[fi] if fi >= 0 else M.efront[pose]), f[hi]
            o = self.outcome((a, u, h) if a == DROP else (a, u), fid, pose, h)
            if o is None:
                r = st
            else:
                b = bytearray(f)
                for i, v in zip(M.chg[pose], o):
                    if v != SAME:
                        b[i] = v
                r = (self.intern(bytes(b)), pose, False)
        self.steps[k] = r
        return r

    def with_tile(self, st, i, u):
        b = bytearray(self.facts[st[0]])
        b[i] = u
        return (self.intern(bytes(b)), st[1], st[2])

    # -- conditions (closer.Approach's logic)
    def holds(self, n, st):
        k = (n, st)
        v = self.memo.get(k)
        if v is None:
            self.computed += 1
            v = self.memo[k] = self._holds(n, st)
        return v

    def _holds(self, n, st):
        if n == 0:
            return st[2]                            # forward's predicted effect has ended the episode
        if self.holds(self.parent[n], st):
            return True
        return self.approach(n, st)

    def ready(self, g, a, st):
        return not self.holds(g, st) and self.holds(g, self.step(st, a))

    def ready_set(self, n, fid):
        """Poses (on a free tile, any direction) where way n's action achieves its parent."""
        k = (n, fid)
        r = self.rsets.get(k)
        if r is None:
            f = self.facts[fid]
            p, a = self.parent[n], self.action[n]
            free, at = M.free, M.poses_at
            r = frozenset(q for i in range(NV) if free[f[i]] for q in at[i] if self.ready(p, a, (fid, q, False)))
            self.rsets[k] = r
        return r

    def free_at(self, f, x, y):
        return 0 <= x < W13 and 0 <= y < W13 and M.free[f[y * W13 + x]]

    def closeness(self, st, rset):
        """Card 024's step measure in the frame's grid: tiles to the pose's tile, then turns until facing
        a direction where a step forward onto a free tile brings the agent closer (3: none)."""
        f = self.facts[st[0]]
        px, py, pd = M.px, M.py, M.pd
        x, y, d = px[st[1]], py[st[1]], pd[st[1]]
        best, cd = None, {}
        for q in rset:
            qx, qy = px[q], py[q]
            dist = abs(qx - x) + abs(qy - y)
            if dist == 0:
                k = (0, _turns(d, pd[q]))
            else:
                c = cd.get((qx, qy))
                if c is None:
                    c = cd[(qx, qy)] = [e for e in range(4)
                                        if abs(qx - x - DIRS[e][0]) + abs(qy - y - DIRS[e][1]) < dist
                                        and self.free_at(f, x + DIRS[e][0], y + DIRS[e][1])]
                k = (dist, min((_turns(d, e) for e in c), default=3))
            if best is None or k < best:
                best = k
        return best

    def closer_move(self, n, st):
        rset = self.ready_set(n, st[0])
        if not rset:
            return None
        here = self.closeness(st, rset)
        for a in closer.MOVE_ORDER:
            t = self.step(st, a)
            if t[2]:
                continue                            # stepping onto the goal square is not walking
            if self.closeness(t, rset) < here:
                return a
        return None

    def approach(self, n, st):
        seen, cur = [], st
        while True:
            k = (n, cur)
            if k in self.appr:
                out, cur = self.appr[k]
                break
            seen.append(k)
            if cur[1] in self.ready_set(n, cur[0]):
                out = True
                break
            a = self.closer_move(n, cur)
            if a is None:
                out = False
                break
            cur = self.step(cur, a)
        for k in seen:
            self.appr[k] = (out, cur)
        return out

    def move_closer(self, st, poses, keep=None, refused=None):
        """Card 027's move toward target poses, with its revision rule (refuse moves that end keep)."""
        if not poses:
            return None
        here = self.closeness(st, poses)
        for a in closer.MOVE_ORDER:
            t = self.step(st, a)
            if t[2]:
                continue
            if self.closeness(t, poses) < here:
                if keep is not None and not self.holds(keep, t):
                    if refused is not None:
                        refused[0] += 1
                    continue
                return a
        return None


# ---------------------------------------------------------------- labels on stored views

def _label_job(job):
    parent, action, nodes, ep, arrays, endeds, rules = job
    outs = [np.zeros((len(ep), len(parent)), bool) for _ in arrays]
    st_ = {"views": 0, "max_mismatch": 0, "ties": 0, "mismatch_over_2": 0}
    cur = th = world = None
    for i in range(len(ep)):
        if ep[i] != cur:
            cur, th, world = ep[i], Think(parent, action, rules), None
        for k in range(len(arrays)):
            world, st, miss, ties = th.observe(world, arrays[k][i], bool(endeds[k][i]))
            st_["views"] += 1
            st_["max_mismatch"] = max(st_["max_mismatch"], miss)
            st_["ties"] += ties > 1
            st_["mismatch_over_2"] += miss > 2
            for n in nodes:
                outs[k][i, n] = th.holds(n, st)
    return outs, st_


LABEL_STATS = Counter()


def labels(pool, parent, action, ep, arrays, endeds, nodes=None, rules=True, jobs=96):
    """Learned conditions of the given nodes on stored views (rows sorted by episode, in time order)."""
    nodes = list(range(len(parent))) if nodes is None else list(nodes)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    sel = starts[np.linspace(0, len(starts) - 1, min(jobs, len(starts))).astype(int)]
    bounds = sorted(set(sel.tolist()) | {len(ep)})
    tasks = [(list(parent), list(action), nodes, ep[a:b], [x[a:b] for x in arrays], [e[a:b] for e in endeds], rules)
             for a, b in zip(bounds, bounds[1:])]
    parts = pool.map(_label_job, tasks)
    for _, s in parts:
        for k, v in s.items():
            LABEL_STATS[k] = max(LABEL_STATS[k], v) if k == "max_mismatch" else LABEL_STATS[k] + v
    return [np.concatenate([p[0][k] for p in parts]) for k in range(len(arrays))]


# ---------------------------------------------------------------- growing the tree with learned conditions

def _play_job(job):
    """Card 026's depth-first acting (walk toward the poses where the way's action works), learned."""
    idx, layouts, parent, action, depth, kids, expanded = job
    out = []
    for i, lay in zip(idx, layouts):
        rng = np.random.default_rng(1000 + int(i))
        th = Think(parent, action)
        s = ld.start_state(lay)
        world, st, _, _ = th.observe(None, see(lay, s))
        rec = {"done": False, "steps": BUDGET, "pause": None, "checks": [], "used": Counter(),
               "random_no_way": 0, "random_no_move": 0, "pred_wrong": 0}
        for t in range(BUDGET):
            checks = [0]
            c, need = D.dfs(kids, depth, expanded, lambda n: th.holds(n, st), 0, checks)
            if need is not None:
                rec["pause"] = need
                break
            rec["checks"].append(checks[0])
            a = None
            if c is None:
                rec["random_no_way"] += 1
            else:
                rec["used"][c] += 1
                g, aw = parent[c], action[c]
                a = aw if th.ready(g, aw, st) else th.closer_move(c, st)
                if a is None:
                    rec["random_no_move"] += 1
            if a is None:
                a = int(rng.integers(len(dl.ACTIONS)))
            pred = th.step(st, a)
            s, end = ld.step(lay, s, a)
            world, st, _, _ = th.observe(world, see(lay, s), end, prefer=pred[1])
            rec["pred_wrong"] += st != pred
            if end:
                rec["done"], rec["steps"] = True, t + 1
                break
        out.append(rec)
    return out


def play(pool, test, tree, expanded):
    kids = {g: tree.children(g) for g in expanded}
    chunks = [c.tolist() for c in np.array_split(np.arange(len(test)), 40) if len(c)]
    parts = pool.map(_play_job, [(c, [test[i] for i in c], tree.parent, tree.action, tree.depth, kids,
                                  set(expanded)) for c in chunks])
    return [r for p in parts for r in p]


def expand_learned(pool, tree, G, expanded, S, log):
    """Card 026's evidence test and self-check for goals G, on learned labels."""
    tr = S["tr"]
    act, w0 = tr["act"], tr["w0"]
    L0, L1 = labels(pool, tree.parent, tree.action, tr["ep"], [S["V0"], S["V1"]], [S["no0"], S["end1"]], G)

    def gains_of(g):
        m = ~L0[:, g]
        return dl.way_gains(act[m], L1[m, g], L1[m, g], w0[m])

    new = dl.expand(tree, G, gains_of, log)
    expanded.update(G)
    if not new:
        return
    need = sorted(set(new) | {tree.parent[c] for c in new})
    seq, st0 = S["seq"], S["st0"]
    Ls, = labels(pool, tree.parent, tree.action, seq["ep"], [S["Vseq"]], [S["endseq"]], need)
    Lst, = labels(pool, tree.parent, tree.action, st0["ep"], [S["Vst0"]], [np.zeros(len(st0["ep"]), bool)], new)

    def check(c):
        g, a = tree.parent[c], tree.action[c]
        y = dl.walk_then_act(seq["act"], seq["ep"], Ls[:, g], a)
        m = ~Ls[:, g]
        return dl.check_gain(Ls[m, c], y[m]), {"rate_on": round(float(y[m & Ls[:, c]].mean()), 4),
                                               "rate_off": round(float(y[m & ~Ls[:, c]].mean()), 4)}

    dl.settle(tree, new, lambda c: Lst[:, c].mean(), check, log)
    for c in new:
        if tree.status[c] == "frontier":
            tree.status[c] = "not expanded"


def grow_learned(pool, test, S, log):
    tree, expanded = dl.Tree(), set()
    tree.status[0] = "not expanded"
    rounds = []
    while True:
        t0 = time.monotonic()
        recs = play(pool, test, tree, expanded)
        paused = Counter(r["pause"] for r in recs if r["pause"] is not None)
        rnd = {"round": len(rounds), "layouts_paused": int(sum(paused.values())),
               "goals_needed": {"/".join(tree.path(g)) or "(goal square)": int(k) for g, k in sorted(paused.items())}}
        if not paused or len(rounds) >= 40:
            rnd["seconds"] = round(time.monotonic() - t0, 1)
            rounds.append(rnd)
            log(f"  round {rnd['round']}: {json.dumps(rnd)}")
            return tree, expanded, recs, rounds
        expand_learned(pool, tree, sorted(paused), expanded, S, lambda m: None)
        rnd["seconds"] = round(time.monotonic() - t0, 1)
        rounds.append(rnd)
        log(f"  round {rnd['round']}: {json.dumps(rnd)}")


# ---------------------------------------------------------------- acting with card 027's target rule, learned

def _act_job(job):
    idx, layouts, parent, action, depth, kids, expanded, things, K, rules = job
    code_of_app, way_codes, floor_app = K
    out = []
    for i, lay in zip(idx, layouts):
        t0 = time.monotonic()
        rng = np.random.default_rng(1000 + int(i))
        th = Think(parent, action, rules)
        s = ld.start_state(lay)
        world, st, _, _ = th.observe(None, see(lay, s))
        failed, refused = set(), [0]
        rec = {"done": False, "steps": BUDGET, "random_no_way": 0, "random_no_move": 0, "move_way_steps": 0,
               "dep_checks": 0, "failed_acts": 0, "ways_used": Counter(), "pred_wrong": 0, "checks": 0,
               "first_key": None, "picked_other": False, "max_mismatch": 0}
        for t in range(BUDGET):
            chk = [0]
            c, need = D.dfs(kids, depth, expanded, lambda n: th.holds(n, st), 0, chk)
            rec["checks"] += chk[0]
            a, poses = None, set()
            if c is None:
                rec["random_no_way"] += 1
            elif c not in things:
                rec["move_way_steps"] += 1
                g, aw = parent[c], action[c]
                a = aw if th.ready(g, aw, st) else th.closer_move(c, st)
                if a is None:
                    rec["random_no_move"] += 1
            else:
                aw = action[c]
                f = th.facts[st[0]]
                here = M.cidx[st[1]]
                cand = {j for j in range(NV) if j != here and code_of_app[f[j]] in way_codes[c]}
                targets = cand
                if len(cand) > 1:
                    dep = set()
                    for j in cand:
                        if f[j] == floor_app:
                            continue
                        rec["dep_checks"] += 1
                        if not th.approach(c, th.with_tile(st, j, floor_app)):
                            dep.add(j)
                    targets = dep or cand
                poses = {p for j in targets for p in M.facing[j] if M.free[f[M.cidx[p]]]}
                poses = {p for p in poses if (c, M.fidx[p], p) not in failed}
                if st[1] in poses:
                    a = aw
                    rec["ways_used"][c] += 1
                else:
                    a = th.move_closer(st, poses, c, refused)
                if a is None:
                    rec["random_no_move"] += 1
            if a is None:
                a = int(rng.integers(len(dl.ACTIONS)))
            pred = th.step(st, a)
            s2, end = ld.step(lay, s, a)
            world, st2, miss, _ = th.observe(world, see(lay, s2), end, prefer=pred[1])
            rec["pred_wrong"] += st2 != pred
            rec["max_mismatch"] = max(rec["max_mismatch"], miss)
            if c in things and a == action[c] and st[1] in poses and not end and not th.holds(parent[c], st2):
                failed.add((c, M.fidx[st[1]], st[1]))          # acting here did not turn the parent true
                rec["failed_acts"] += 1
            if s2[3] and not s[3] and rec["first_key"] is None:  # evaluator only
                rec["first_key"] = "matching" if s2[3] == 1 else "other"
            if s2[3] == 2:
                rec["picked_other"] = True
            s, st = s2, st2
            if end:
                rec["done"], rec["steps"] = True, t + 1
                break
        rec["moves_refused"] = refused[0]
        rec["computed"] = th.computed
        rec["seconds"] = time.monotonic() - t0
        out.append(rec)
    return out


def act_arm(pool, test, tree, expanded, K, rules=True):
    kids = {g: tree.children(g) for g in expanded}
    things = set(Kd.thing_ways(tree))
    chunks = [c.tolist() for c in np.array_split(np.arange(len(test)), 40) if len(c)]
    parts = pool.map(_act_job, [(c, [test[i] for i in c], tree.parent, tree.action, tree.depth, kids,
                                 set(expanded), things, K, rules) for c in chunks])
    recs = [r for p in parts for r in p]
    done = np.array([r["done"] for r in recs])
    steps = np.array([r["steps"] for r in recs])
    moves = int(steps.sum())
    rnd = sum(r["random_no_way"] + r["random_no_move"] for r in recs)
    res = {"success": round(float(done.mean()), 4),
           "mean_steps_when_successful": round(float(steps[done].mean()), 1) if done.any() else None,
           "random_steps": int(rnd), "moves": moves, "random_share": round(rnd / max(moves, 1), 4),
           "random_no_way": int(sum(r["random_no_way"] for r in recs)),
           "move_way_steps": int(sum(r["move_way_steps"] for r in recs)),
           "dependence_checks_per_move": round(sum(r["dep_checks"] for r in recs) / max(moves, 1), 3),
           "failed_acts": int(sum(r["failed_acts"] for r in recs)),
           "moves_refused": int(sum(r["moves_refused"] for r in recs)),
           "predictions_wrong": int(sum(r["pred_wrong"] for r in recs)),
           "largest_view_mismatch_placing_the_agent": int(max(r["max_mismatch"] for r in recs)),
           "tree_checks_per_move": round(sum(r["checks"] for r in recs) / max(moves, 1), 2),
           "conditions_computed_per_move": round(sum(r["computed"] for r in recs) / max(moves, 1), 1),
           "seconds_per_layout_mean": round(float(np.mean([r["seconds"] for r in recs])), 3),
           "seconds_per_layout_max": round(float(np.max([r["seconds"] for r in recs])), 2)}
    used = Counter()
    for r in recs:
        used.update(r["ways_used"])
    res["ways_used"] = {"/".join(tree.path(n)): k for n, k in sorted(used.items())}
    fk = [r["first_key"] for r in recs]
    if any(fk):
        res["failing_layouts"] = int((~done).sum())
        res["failing_that_picked_other_key"] = int(sum(r["picked_other"] for r, d in zip(recs, done) if not d))
    return res


# ---------------------------------------------------------------- criteria 1 and 2

def effects_eval(V0, V1, act, term1, w, rules=True):
    """Criterion 1: from each held-out view, the predicted next facts (every view tile and the held
    tile, and whether the episode ends) against the real next view."""
    th = Think([-1], [-1], rules)
    ok = np.zeros(len(act), bool)
    errs = Counter()
    for i in range(len(act)):
        _, st, _, _ = th.observe(None, V0[i])
        nxt = th.step(st, int(act[i]))
        pv = th.view(nxt)
        ok[i] = bool((pv == V1[i]).all()) and nxt[2] == bool(term1[i])
        if not ok[i]:
            bad = np.flatnonzero(pv != V1[i])
            errs[(dl.ACTIONS[act[i]], nm(V0[i, FRONT]), nm(V0[i, HELD]),
                  tuple((int(q), nm(pv[q]), nm(V1[i, q])) for q in bad[:3]), nxt[2], bool(term1[i]))] += 1
        if len(th.facts) > 20000:
            th = Think([-1], [-1], rules)
    out = {}
    for a in range(6):
        m = act == a
        if m.any():
            out[dl.ACTIONS[a]] = {"rows": int(m.sum()), "exact_share": round(float(ok[m].mean()), 6),
                                  "exact_share_weighted": round(float((ok[m] * w[m]).sum() / w[m].sum()), 6)}
    return {"per_action": out, "all": round(float(ok.mean()), 6),
            "lowest": min(v["exact_share"] for v in out.values()),
            "errors_top": [[list(map(str, k)), v] for k, v in errs.most_common(8)]}


def rules_check(world, rep):
    """Criterion 1's rules: for each closed door, the rules found for its opening."""
    found = {}
    for r in rep["rules"]:
        if r["key"].startswith("toggle closed door"):
            found[r["key"]] = [x["if"] for x in r["rules"]]
    ok = bool(found)
    for key, rules in found.items():
        colour = key.split()[-1]
        want = [[f"held = key {colour}"]] if world == "key" else [["switch on in view"]]
        ok &= rules == want
    return {"door_rules": found, "exactly_as_expected": bool(ok and len(found) == 3)}


def agreement(Lx, Ll, nodes, tree):
    out = {}
    for n in nodes:
        a = np.concatenate([Lx[k][:, n] == Ll[k][:, n] for k in range(len(Lx))])
        on = np.concatenate([Lx[k][:, n] for k in range(len(Lx))])
        out["/".join(tree.path(n)) or "(goal square)"] = {"agree": round(float(a.mean()), 5),
                                                         "exact_on_share": round(float(on.mean()), 4)}
    return out


def used_nodes(tree, expanded):
    return sorted({c for g in expanded for c in tree.children(g)})


# ---------------------------------------------------------------- main

def main():
    global M
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    out = Path(args.get("--out", "runs/028_effects.json"))
    episodes, n_test = int(args.get("--episodes", 5000)), int(args.get("--layouts", 500))
    heldout = int(args.get("--heldout", 1000))
    worlds = args.get("--worlds", "key,switch").split(",")
    closer.MEASURE = "step"
    dl.MAX_GOALS = 10 ** 9
    dl.START_SHARE = 2.0
    dl.Exact = closer.Approach                       # the evaluator's exact conditions, read by the workers
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    res = {"note": "Card 028, tools/card028/effects.py; conditions, walking and tree growth on a model learned by "
                   "counting from the agent's own views.", "episodes": episodes, "layouts": n_test,
           "heldout_episodes": heldout, "worlds": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    for world in worlds:
        r = res["worlds"][world] = {}
        tw = time.monotonic()
        log(f"=== {world} world: gate")
        M = None
        pool = pool20()
        layouts, tr, seq, st0 = dl.collect(pool, world, episodes, 11, 0.0, seq_episodes=300)
        hlay, htr, _, _ = dl.collect(pool, world, heldout, 12, 0.0, seq_episodes=4)
        rng = np.random.default_rng(777)
        test = [ld.make_layout(8, world, rng) for _ in range(n_test)]
        E = Kd.effects(tr)
        r["facts_check"] = Kd.effects_check(tr, E)
        log(f"  facts from pixels against the simulator: {r['facts_check']}")
        # arm 1: card 027's final setup (simulator conditions, counted kinds); its tree is arm 3's
        if world == "key":
            rep = json.loads(Path("runs/026_dfs.json").read_text())["tree"]
            tree27 = dl.Tree.from_report(rep)
            exp27 = {t["id"] for t in rep if t["status"] == "expanded" or t["status"].startswith("leaf: no action")}
            rounds27 = "rebuilt from card 026"
        else:
            tree27, exp27, _, rounds27 = Kd.grow_tree(pool, test, layouts, tr, seq, st0, log)
        exact = Kd.exact_kinds(E, log)
        r["counted_kinds"] = Kd.partition_check(exact)["groups"]
        n_app = int(Kd.APP.max()) + 1
        code_exact = np.full(256, -1, np.int64)
        for u, k in exact.items():
            code_exact[u] = k
        wc = np.bincount(E["f0"], weights=tr["w0"].astype(np.float64))
        floor_app, wall_app = [int(u) for u in np.argsort(-wc)[:2]]
        ways27 = Kd.thing_ways(tree27)
        L0x, L1x = dl.exact_labels(pool, layouts, tree27.parent, tree27.action, tr["ep"], [tr["s0"], tr["s1"]],
                                   sorted({tree27.parent[n] for n in ways27}))
        wk = Kd.way_kinds(tree27, ways27, L0x, L1x, tr, E, exact)
        K27 = (code_exact[:n_app], {n: set(v.get("codes", [])) for n, v in wk.items()}, floor_app, wall_app)
        r["tree_027"] = {"nodes": len(tree27.parent), "goals_expanded": len(exp27), "rounds": rounds27,
                         "tree": [(t["path"], t["status"]) for t in tree27.report()]}
        r["arm1_upper_bound"] = Kd.act_arm(pool, test, tree27, exp27, "main", K27)
        log(f"  arm 1, upper bound (card 027's final setup): {json.dumps(r['arm1_upper_bound'])}")
        pool.close()
        save()

        # the model
        t0 = time.monotonic()
        V0, V1 = views(tr["c0"], tr["s0"]), views(tr["c1"], tr["s1"])
        Vseq, Vst0 = views(seq["codes"], seq["st"]), views(st0["codes"], st0["st"])
        Vh0, Vh1 = views(htr["c0"], htr["s0"]), views(htr["c1"], htr["s1"])
        term_eps = set(tr["ep"][tr["term1"].astype(bool)].tolist())      # episodes that ended (the agent saw it)
        last = np.r_[seq["ep"][1:] != seq["ep"][:-1], True]
        endseq = last & np.isin(seq["ep"], list(term_eps))
        log(f"  views: {len(V0)} stored transitions, {len(Vseq)} sequence frames ({time.monotonic() - t0:.0f}s)")
        t0 = time.monotonic()
        M, mrep = learn(tr, E, V0, V1, log, dev)
        mrep["seconds"] = round(time.monotonic() - t0, 1)
        r["model"] = mrep
        log(f"  model learned in {mrep['seconds']}s: {mrep['poses']} poses; places changed by pick up/drop/toggle "
            f"{mrep['places_changed_by_pickup_drop_toggle']}; forward map off a one-row shift at "
            f"{mrep['forward_map_places_not_a_one_row_shift']}")
        for a in ("left", "right", "forward"):
            x = mrep[a]
            log(f"    {a}: entering {x['entering_places']}, lowest predicted share {x['lowest_predicted_share']}, "
                f"tables changing appearance {x['tables_that_change_appearance']}")
        for x in mrep["rules"]:
            log(f"    rule: {x['key']} -> {x['outcome']}: {[(y['if'], y['rate'], y['tries_weighted']) for y in x['rules']]}; "
                f"otherwise {x['otherwise']}")
        t0 = time.monotonic()
        c1 = effects_eval(Vh0, Vh1, htr["act"].astype(np.int64), htr["term1"].astype(bool), htr["w0"].astype(np.float64))
        c1["rules"] = rules_check(world, mrep)
        c1["pass"] = c1["lowest"] >= 0.999 and c1["rules"]["exactly_as_expected"]
        c1["seconds"] = round(time.monotonic() - t0, 1)
        r["criterion_1_effects"] = c1
        log(f"  RESULT criterion 1 (effects on {len(Vh0)} held-out transitions): {json.dumps(c1)}")
        save()

        # main: grow with learned conditions, check them, act
        S = {"tr": tr, "V0": V0, "V1": V1, "no0": np.zeros(len(V0), bool), "end1": tr["term1"].astype(bool),
             "seq": seq, "Vseq": Vseq, "endseq": endseq, "st0": st0, "Vst0": Vst0}
        pool = pool20()
        t0 = time.monotonic()
        LABEL_STATS.clear()
        tree, expanded, recs, rounds = grow_learned(pool, test, S, log)
        g = D.summary(recs, tree)
        r["tree_learned"] = {"nodes": len(tree.parent), "goals_expanded": len(expanded), "rounds": rounds,
                             "seconds": round(time.monotonic() - t0, 1), "placing_views": dict(LABEL_STATS),
                             "tree": [(t["path"], t["status"], t["self_check"]) for t in tree.report()],
                             "last_round_play": {k: g[k] for k in ("success", "mean_steps_when_successful",
                                                                   "random_share")},
                             "same_ways_as_027": sorted(t["path"] for t in tree.report()) ==
                             sorted(t["path"] for t in tree27.report())}
        log(f"  tree grown with learned conditions: {json.dumps({k: v for k, v in r['tree_learned'].items() if k != 'tree'})}")
        log(f"  tree: {[(t['path'], t['status']) for t in tree.report()]}")
        save()

        t0 = time.monotonic()
        nodes = used_nodes(tree, expanded)
        Lx = dl.exact_labels(pool, hlay, tree.parent, tree.action, htr["ep"], [htr["s0"], htr["s1"]], nodes)
        Ll = labels(pool, tree.parent, tree.action, htr["ep"], [Vh0, Vh1], [np.zeros(len(Vh0), bool),
                                                                            htr["term1"].astype(bool)], nodes)
        ag = agreement(Lx, Ll, nodes, tree)
        nodes27 = used_nodes(tree27, exp27)
        Lx27 = dl.exact_labels(pool, hlay, tree27.parent, tree27.action, htr["ep"], [htr["s0"], htr["s1"]], nodes27)
        Ll27 = labels(pool, tree27.parent, tree27.action, htr["ep"], [Vh0, Vh1],
                      [np.zeros(len(Vh0), bool), htr["term1"].astype(bool)], nodes27)
        r["criterion_2_conditions"] = {"nodes_of_learned_tree": ag,
                                       "lowest": min(v["agree"] for v in ag.values()),
                                       "pass": min(v["agree"] for v in ag.values()) >= 0.99,
                                       "nodes_of_027_tree": agreement(Lx27, Ll27, nodes27, tree27),
                                       "seconds": round(time.monotonic() - t0, 1)}
        log(f"  RESULT criterion 2 (conditions on held-out situations): {json.dumps(r['criterion_2_conditions'])}")
        save()

        t0 = time.monotonic()
        ways = Kd.thing_ways(tree)
        L0l, L1l = labels(pool, tree.parent, tree.action, tr["ep"], [V0, V1], [S["no0"], S["end1"]],
                          sorted({tree.parent[n] for n in ways}))
        wkl = Kd.way_kinds(tree, ways, L0l, L1l, tr, E, exact)
        K2 = (code_exact, {n: set(v.get("codes", [])) for n, v in wkl.items()}, floor_app)
        r["way_kinds_learned_tree"] = {v["path"]: {k: v[k] for k in ("successes", "codes", "kinds")}
                                       for v in wkl.values() if v.get("successes")}
        r["arm2_main"] = act_arm(pool, test, tree, expanded, K2)
        log(f"  RESULT arm 2 (main, all learned): {json.dumps(r['arm2_main'])}")
        save()
        L0m, L1m = labels(pool, tree27.parent, tree27.action, tr["ep"], [V0, V1], [S["no0"], S["end1"]],
                          sorted({tree27.parent[n] for n in ways27}))
        wk3 = Kd.way_kinds(tree27, ways27, L0m, L1m, tr, E, exact)
        K3 = (code_exact, {n: set(v.get("codes", [])) for n, v in wk3.items()}, floor_app)
        r["arm3_learned_on_027_tree"] = act_arm(pool, test, tree27, exp27, K3)
        log(f"  RESULT arm 3 (learned conditions, card 027's tree): {json.dumps(r['arm3_learned_on_027_tree'])}")
        r["arm4_no_rules"] = act_arm(pool, test, tree, expanded, K2, rules=False)
        log(f"  RESULT arm 4 (effects without rules): {json.dumps(r['arm4_no_rules'])}")
        pool.close()
        r["acting_seconds"] = round(time.monotonic() - t0, 1)
        m2, ub = r["arm2_main"], r["arm1_upper_bound"]
        r["gate"] = {"facts_agree": all(v == 1.0 for v in r["facts_check"].values()),
                     "upper_bound_98": ub["success"] >= 0.98,
                     "baseline_below_98": r["arm4_no_rules"]["success"] < 0.98}
        r["criterion_3_acting"] = {"success_98": m2["success"] >= 0.98, "random_share_1pct": m2["random_share"] <= 0.01,
                                   "steps_within_1_5x": (m2["mean_steps_when_successful"] or 1e9)
                                   <= 1.5 * ub["mean_steps_when_successful"]}
        r["criterion_3_acting"]["pass"] = all(r["criterion_3_acting"].values())
        r["seconds"] = round(time.monotonic() - tw, 1)
        log(f"  GATE {world}: {r['gate']}; criterion 3: {r['criterion_3_acting']}; world took {r['seconds']}s")
        save()
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
