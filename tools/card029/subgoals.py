"""Card 029: subgoals worked out from the learned model.

The agent is given one goal, "the episode has ended" (it ends on the goal square), and card 028's
counted model. At every step it works backward from the goal through what the model says each action
does (means-ends analysis, as STRIPS):

  a condition's achievers are the actions whose learned outcome makes it true; an achiever's needs are
      what its outcome is keyed on besides the thing in front (the held tile, for drop), the atoms of
      the outcome's rule ("held = X", "X in view"), and last, facing a tile the action acts on;
  the first unmet need becomes the subgoal; among achievers, the fewest unmet needs first, then the
      one whose walking target is nearest (card 024's closeness); if one finds no action, the next;
  facing is reached by walking: moving closer (card 024), then move ways computed in the head (card
      026's, up to two moves deep), then a tile in the way that an action can make passable, whose
      change becomes a subgoal ("tile j shows the open door").

Card 027's revision rules, generalised: an action is refused when, in the head, it undoes a condition
already met higher in the chain, or leaves a facing higher in the chain unreachable; an (action, tile,
pose) is marked failed for the episode when acting there does not give the predicted result. It is
recomputed from the top at every step (teleo-reactive). Nothing the agent thinks with reads the
simulator; the evaluator reads it for the shortest route and the checks.

Arms, per world (key, switch, either, both): 1 the shortest route (evaluator, breadth first over full
states); 2 main; 3 card 028's pipeline (run for the worlds in --arm3; key and switch are card 028's
run, same data and layouts); 4 the same working backward without rules (an outcome ever seen counts as
possible).

Run: bin/prun python tools/card029/subgoals.py [--episodes 5000] [--layouts 500] [--heldout 1000]
     [--worlds key,switch,either,both] [--arm3 either,both] [--out runs/029_subgoals.json]
"""
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card028"))
import effects as F  # noqa: E402

Kd, dl, ld, closer, D = F.Kd, F.dl, F.ld, F.closer, F.D
NV, FRONT, HELD, SAME = F.NV, F.FRONT, F.HELD, F.SAME
LEFT, RIGHT, FWD, PICK, DROP, TOG = range(6)
MOVES = F.MOVES
BUDGET = 200
MAXD = 6                    # declared: depth of subgoals
HERE = -1                   # walking: already at the target
nm = F.nm


# ---------------------------------------------------------------- the model's actions, read as needs and results

class Op:
    """One learned outcome read as an action: act on a tile showing u (holding h, for drop); the tile
    in front becomes front and the hand held (None: unchanged); end: the episode ends. alts: one list
    of rule atoms per way the outcome happens ((0, X): held = X; (1, X): X in view)."""

    def __init__(self, a, u, h, front, held, alts, end=False):
        self.a, self.u, self.h, self.front, self.held, self.alts, self.end = a, u, h, front, held, alts, end


def build_ops(rules=True):
    M = F.M
    cp = [int(q) for q in M.change_places]
    assert set(cp) <= {FRONT, HELD}, cp
    ops = []
    for key, e in M.table.items():
        a, u = key[0], key[1]
        h = key[2] if a == DROP else None
        if e.multi is None:
            outs = [(e.single, [[]])]
        elif not rules:
            outs = [(e.optimistic, [[]])]          # arm 4: an outcome ever seen counts as possible
        else:
            outs = []
            for o, rl, dr in e.multi:
                alts = [list(conds) for conds, rate in rl if rate >= 0.5]     # declared threshold, as card 028
                if dr >= 0.5:
                    alts.append([])
                if alts:
                    outs.append((o, alts))
            if e.default is not None:
                outs.append((e.default, [[]]))
        for o, alts in outs:
            if o is None:
                continue
            ch = dict(zip(cp, o))
            fr, he = ch.get(FRONT, SAME), ch.get(HELD, SAME)
            ops.append(Op(a, u, h, None if fr == SAME else fr, None if he == SAME else he, alts))
    for u in range(256):
        if M.move_out[FWD][u] == F.ENDED:
            ops.append(Op(FWD, u, None, None, None, [[]], end=True))
    return ops


def cname(c):
    """A condition in words (reports)."""
    k = c[0]
    if k == "end":
        return "episode ended"
    if k == "held":
        return f"held = {nm(c[1])}"
    if k == "view":
        return f"{nm(c[1])} in view"
    if k == "tile":
        return f"tile {c[1]} shows {nm(c[2])}"
    if k == "face":
        return f"facing {'tile ' + str(c[3]) if c[3] is not None else nm(c[2])} (for {dl.ACTIONS[c[1]]})"
    if k == "act":
        return f"do {dl.ACTIONS[c[1]]}"
    if k == "walk":
        return f"walk: {c[1]}"
    return str(c)


class Result:
    def __init__(self, action, trace, near):
        self.action, self.trace, self.near = action, trace, near


# ---------------------------------------------------------------- working backward on facts

class Plan(F.Think):
    """Card 028's facts and learned effects (observe, step, closeness), with conditions about things
    and the means-ends procedure above."""

    def __init__(self, ops, rules=True):
        super().__init__([-1], [-1], rules)
        self.by = {}
        for op in ops:
            if op.end:
                self.by.setdefault(("end",), []).append(op)
            if op.held is not None:
                self.by.setdefault(("held", op.held), []).append(op)
            if op.front is not None:
                self.by.setdefault(("front", op.front), []).append(op)
        self.passable = {}                       # appearance -> ops that make it a tile one can walk onto
        for op in ops:
            if op.front is not None and F.M.free[op.front] and not F.M.free[op.u]:
                self.passable.setdefault(op.u, []).append(op)
        self.failed, self.refused, self.version = set(), set(), 0
        self.psets, self.pids = [], {}
        self.fmemo, self.amemo, self.rmemo, self.cmemo, self.canmemo, self.premo = {}, {}, {}, {}, {}, {}
        self.n_refused, self.walk_kind = 0, Counter()

    # -- conditions
    def hold(self, c, st):
        self.computed += 1
        k = c[0]
        if k == "end":
            return st[2]
        f = self.facts[st[0]]
        if k == "held":
            return f[F.M.hidx[st[1]]] == c[1]
        if k == "view":
            return c[1] in self.presence(st[0], st[1])
        if k == "tile":
            return f[c[1]] == c[2]
        if k == "face":
            return st[1] in self.psets[self.face_pid(st[0], c)]
        raise ValueError(c)

    def intern_p(self, P):
        P = frozenset(P)
        i = self.pids.get(P)
        if i is None:
            i = self.pids[P] = len(self.psets)
            self.psets.append(P)
        return i

    def face_pid(self, fid, need):
        """Poses facing a tile the need names (one tile, or any tile showing u), standing on a free tile,
        without refused tiles or failed poses."""
        k = (fid, need, self.version)
        r = self.fmemo.get(k)
        if r is None:
            _, a, u, j = need
            M = F.M
            farr = np.frombuffer(self.facts[fid], np.uint8)
            T = [j] if j is not None else np.flatnonzero(farr[:NV] == u).tolist()
            P = [p for i in T if (fid, a, i) not in self.refused for p in M.facing[i]
                 if M.free[farr[M.cidx[p]]] and (a, i, p) not in self.failed]
            r = self.fmemo[k] = self.intern_p(P)
        return r

    def achievers(self, c, st):
        k = c[0]
        if k == "end":
            ops, j = self.by.get(("end",), []), None
        elif k == "held":
            ops, j = self.by.get(("held", c[1]), []), None
        elif k == "view":
            ops, j = self.by.get(("front", c[1]), []), None
        elif k == "tile":
            f = self.facts[st[0]]
            ops, j = [op for op in self.by.get(("front", c[2]), []) if op.u == f[c[1]]], c[1]
        else:
            return []
        out = []
        for op in ops:
            pre = [("held", op.h)] if op.h is not None else []
            for alt in op.alts:
                needs = pre + [("held", x) for kind, x in alt if kind == 0] + [("view", x) for kind, x in alt if kind == 1]
                out.append((op, needs + [("face", op.a, op.u, j)]))
        return out

    # -- walking
    def closer_move_p(self, st, P):
        here = self.closeness(st, P)
        for a in closer.MOVE_ORDER:
            t = self.step(st, a)
            if t[2]:
                continue                                # stepping onto the goal square is not walking
            if self.closeness(t, P) < here:
                return a
        return None

    def approach_p(self, st, pid):
        """Moving closer (card 024's step measure) reaches one of the poses."""
        P = self.psets[pid]
        if not P:
            return False
        seen, cur = [], st
        while True:
            k = (pid, cur)
            if k in self.amemo:
                out = self.amemo[k]
                break
            seen.append(k)
            if cur[1] in P:
                out = True
                break
            a = self.closer_move_p(cur, P)
            if a is None:
                out = False
                break
            cur = self.step(cur, a)
        for k in seen:
            self.amemo[k] = out
        return out

    def standing(self, fid):
        k = ("standing", fid)
        r = self.canmemo.get(k)
        if r is None:
            farr = np.frombuffer(self.facts[fid], np.uint8)
            r = self.canmemo[k] = [p for p in F.M.all_poses.tolist() if F.M.free[farr[F.M.cidx[p]]]]
        return r

    def connected(self, st):
        """Poses reachable by moves at all (in the head). Moves and move ways only take moves, so where
        no pose of the target is reachable they cannot succeed; this only saves their computation."""
        k = (st[0], st[1])
        r = self.cmemo.get(k)
        if r is None:
            seen, todo = {st[1]}, [st[1]]
            while todo:
                p = todo.pop()
                for a in MOVES:
                    t = self.step((st[0], p, False), a)
                    if not t[2] and t[1] not in seen:
                        seen.add(t[1])
                        todo.append(t[1])
            r = frozenset(seen)
            for p in seen:
                self.cmemo[(st[0], p)] = r
        return r

    def can_approach(self, fid, pid):
        k = (fid, pid)
        r = self.canmemo.get(k)
        if r is None:
            r = self.canmemo[k] = frozenset(q for q in self.standing(fid) if self.approach_p((fid, q, False), pid))
        return r

    def pre(self, fid, m, A):
        """Poses (standing on a free tile) from which move m leads into the set A."""
        k = (fid, m, A)
        r = self.premo.get(k)
        if r is None:
            out = []
            for q in self.standing(fid):
                t = self.step((fid, q, False), m)
                if not t[2] and t[1] != q and t[1] in A:
                    out.append(q)
            r = self.premo[k] = self.intern_p(out)
        return r

    def reach(self, st, pid):
        """Walking toward the poses: HERE, the move to take, or None. Moving closer; else a move way
        (a pose from which one move lets moving closer succeed), then two moves deep."""
        k = (st, pid)
        if k in self.rmemo:
            return self.rmemo[k]
        P = self.psets[pid]
        r, kind = None, None
        if not P:
            pass
        elif st[1] in P:
            r, kind = HERE, "here"
        elif self.approach_p(st, pid):
            r, kind = self.closer_move_p(st, P), "closer"
        elif self.connected(st) & P:
            fid = st[0]
            A = self.can_approach(fid, pid)
            q1 = {}
            for m in closer.MOVE_ORDER:
                qid = q1[m] = self.pre(fid, m, A)
                if st[1] in self.psets[qid]:
                    r, kind = m, "move way"
                    break
                if self.approach_p(st, qid):
                    r, kind = self.closer_move_p(st, self.psets[qid]), "move way"
                    break
            if r is None:
                for m1 in closer.MOVE_ORDER:
                    A1 = self.can_approach(fid, q1[m1])
                    for m2 in closer.MOVE_ORDER:
                        qid = self.pre(fid, m2, A1)
                        if st[1] in self.psets[qid]:
                            r, kind = m2, "move way, two deep"
                            break
                        if self.approach_p(st, qid):
                            r, kind = self.closer_move_p(st, self.psets[qid]), "move way, two deep"
                            break
                    if r is not None:
                        break
        self.rmemo[k] = (r, kind)
        return r, kind

    def walk(self, need, st, depth, chain, protect, faces):
        pid = self.face_pid(st[0], need)
        r, kind = self.reach(st, pid)
        if r is not None and r != HERE:
            return Result(r, [need, ("walk", kind)], self.closeness(st, self.psets[pid]))
        if depth + 1 > MAXD:
            return None
        # a tile in the way that an action can make passable
        f = self.facts[st[0]]
        cands = []
        for j in range(NV):
            for op in self.passable.get(f[j], ()):
                st2 = self.with_tile(st, j, op.front)
                pid2 = self.face_pid(st2[0], need)
                if not (self.connected(st2) & self.psets[pid2]):
                    continue
                if self.reach(st2, pid2)[0] is None:
                    continue
                near = self.closeness(st, F.M.facing[j]) or (99, 9)
                cands.append((near, j, op.front))
        for _, j, v in sorted(set(cands)):
            c = ("tile", j, v)
            if c in chain:
                continue
            res = self.solve(c, st, depth + 1, chain + [need], protect, faces + [need])
            if res is not None:
                res.trace = [need, ("walk", "blocked")] + res.trace
                return res
        return None

    # -- means-ends
    def solve(self, c, st, depth, chain, protect, faces):
        alts = []
        for op, needs in self.achievers(c, st):
            if any(n in chain or n == c for n in needs if n[0] != "face"):
                continue                                           # a need already pursued above
            alts.append((sum(not self.hold(n, st) for n in needs), op, needs))
        alts.sort(key=lambda x: x[0])
        i = 0
        while i < len(alts):
            grp = [x for x in alts if x[0] == alts[i][0]]
            found = []
            for _, op, needs in grp:
                res = self.pursue(c, op, needs, st, depth, chain, protect, faces)
                if res is not None:
                    found.append(res)
            if found:
                best = min(found, key=lambda r: r.near)
                best.trace = [c] + best.trace
                return best
            i += len(grp)
        return None

    def pursue(self, c, op, needs, st, depth, chain, protect, faces):
        met = [n for n in needs if n[0] != "face" and self.hold(n, st)]
        prot = protect + met
        mine = [n for n in needs if n[0] == "face"]
        for n in needs:
            if self.hold(n, st):
                continue
            if n[0] == "face":
                return self.walk(n, st, depth, chain + [c], prot, faces)
            if depth + 1 > MAXD:
                return None
            return self.solve(n, st, depth + 1, chain + [c], prot, faces + mine)
        # every need met: act, unless the model's own prediction or a revision rule says no
        a = op.a
        st2 = self.step(st, a)
        if not self.hold(c, st2):
            return None
        bad = any(self.hold(p, st) and not self.hold(p, st2) for p in protect)      # its own needs it may use up
        if not bad and st2[0] != st[0]:
            for fn in faces:
                if (self.reach(st, self.face_pid(st[0], fn))[0] is not None
                        and self.reach(st2, self.face_pid(st2[0], fn))[0] is None):
                    bad = True
                    break
        if bad:
            self.refused.add((st[0], a, F.M.fidx[st[1]]))
            self.version += 1
            self.n_refused += 1
            return None
        return Result(a, [("act", a)], (0, 0))

    def choose(self, st):
        res = None
        for _ in range(12):
            v = self.version
            res = self.solve(("end",), st, 0, [], [], [])
            if res is not None or self.version == v:
                break
        return res


# ---------------------------------------------------------------- the evaluator's reading of facts

def world_cell(lay, i):
    """The simulator's cell of place i of the episode's first view (evaluator only)."""
    x0, y0, d0 = lay.start
    r, c = divmod(int(i), F.W13)
    return (x0 + int(Kd._DX[d0][r, c]), y0 + int(Kd._DY[d0][r, c]))


def walkable(lay, s, frm, extra=None):
    """Cells reachable on foot in the true state (evaluator only); extra: one more cell counted free."""
    seen, todo = {frm}, [frm]
    while todo:
        x, y = todo.pop()
        for dx, dy in ld.DIR_VEC:
            p = (x + dx, y + dy)
            if p in seen or p == lay.goal:
                continue
            c = ld.cell(lay, s, p)
            if c == "empty" or (c == "door" and s[6]) or p == extra:
                seen.add(p)
                todo.append(p)
    return seen


# ---------------------------------------------------------------- acting

def required(world, lay):
    c = lay.door_colour
    key, sw = f"held = key {c}", "switch on in view"
    need = {"key": [{key}], "switch": [{sw}], "either": [{key}, {sw}], "both": [{key, sw}]}[world]
    forbid = {f"held = key {x}" for x in ("red", "green", "blue") if x != c}
    if world == "key":
        forbid.add(sw)
    if world == "switch":
        forbid.add(key)
    return need, forbid


def _act_job(job):
    idx, layouts, rules, world, ops = job
    M = F.M
    out = []
    for i, lay in zip(idx, layouts):
        t0 = time.monotonic()
        rng = np.random.default_rng(1000 + int(i))
        pl = Plan(ops, rules)
        s = ld.start_state(lay)
        facts, st, _, _ = pl.observe(None, F.see(lay, s))
        rec = {"done": False, "steps": BUDGET, "random": 0, "pred_wrong": 0, "failed_acts": 0, "max_mismatch": 0,
               "pursued": set(), "tiles": [], "opened_by": None, "chain0": None, "walk": Counter(), "door_step": None}
        for t in range(BUDGET):
            res = pl.choose(st)
            if res is None:
                a = int(rng.integers(len(dl.ACTIONS)))
                rec["random"] += 1
            else:
                a = res.action
                rec["walk"]["act" if res.trace[-1][0] == "act" else res.trace[-1][1]] += 1
                if t == 0:
                    rec["chain0"] = [cname(x) for x in res.trace]
                if not s[6]:                                     # evaluator: before the door opened
                    for x in res.trace:
                        if x[0] in ("held", "view"):
                            rec["pursued"].add(cname(x))
                    for k, x in enumerate(res.trace):
                        if x[0] == "tile" and k >= 2 and res.trace[k - 1] == ("walk", "blocked"):
                            if all(tt["tile"] != x[1] for tt in rec["tiles"]):
                                need = res.trace[k - 2]
                                P = pl.psets[pl.face_pid(st[0], need)]
                                P2 = pl.psets[pl.face_pid(pl.with_tile(st, x[1], x[2])[0], need)]
                                stand = {world_cell(lay, M.cidx[p]) for p in P}
                                stand2 = {world_cell(lay, M.cidx[p]) for p in P2}     # with the tile changed
                                here = (s[0], s[1])
                                cell = world_cell(lay, x[1])
                                now = walkable(lay, s, here)
                                gone = walkable(lay, s, here, extra=cell) | {cell}
                                rec["tiles"].append({"tile": x[1], "shows": nm(pl.facts[st[0]][x[1]]),
                                                     "made": nm(x[2]), "is_door": cell == lay.door,
                                                     "confirmed_blocking": not (now & stand) and bool(gone & stand2)})
            pred = pl.step(st, a)
            s2, end = ld.step(lay, s, a)
            facts, st2, miss, _ = pl.observe(facts, F.see(lay, s2), end, prefer=pred[1])
            rec["pred_wrong"] += st2 != pred
            rec["max_mismatch"] = max(rec["max_mismatch"], miss)
            if a not in MOVES and st2 != pred:
                pl.failed.add((a, M.fidx[st[1]], st[1]))          # acting here did not give the predicted result
                pl.version += 1
                rec["failed_acts"] += 1
            if s2[6] and not s[6] and rec["opened_by"] is None:  # evaluator only
                rec["opened_by"] = ("key" if s2[3] == 1 else "") + ("+" if s2[3] == 1 and s2[7] else "") + \
                                   ("switch" if s2[7] else "")
                rec["door_step"] = t + 1
            s, st = s2, st2
            if end:
                rec["done"], rec["steps"] = True, t + 1
                break
        need, forbid = required(world, lay)
        rec["subgoals_ok"] = any(n <= rec["pursued"] for n in need) and not (forbid & rec["pursued"])
        rec["forbidden_pursued"] = sorted(forbid & rec["pursued"])
        rec["pursued"] = sorted(rec["pursued"])
        rec["refused"] = pl.n_refused
        rec["computed"] = pl.computed
        rec["seconds"] = time.monotonic() - t0
        out.append(rec)
    return out


def act_arm(pool, test, world, rules=True):
    ops = build_ops(rules)
    chunks = [c.tolist() for c in np.array_split(np.arange(len(test)), 40) if len(c)]
    parts = pool.map(_act_job, [(c, [test[i] for i in c], rules, world, ops) for c in chunks])
    recs = [r for p in parts for r in p]
    done = np.array([r["done"] for r in recs])
    steps = np.array([r["steps"] for r in recs])
    moves = int(steps.sum())
    rnd = sum(r["random"] for r in recs)
    walk = Counter()
    for r in recs:
        walk.update(r["walk"])
    tiles = [t for r in recs for t in r["tiles"]]
    res = {"success": round(float(done.mean()), 4),
           "mean_steps_when_successful": round(float(steps[done].mean()), 2) if done.any() else None,
           "random_steps": int(rnd), "moves": moves, "random_share": round(rnd / max(moves, 1), 4),
           "subgoals_ok_share": round(float(np.mean([r["subgoals_ok"] for r in recs])), 4),
           "layouts_subgoals_not_ok": int(sum(not r["subgoals_ok"] for r in recs)),
           "forbidden_pursued": dict(Counter(x for r in recs for x in r["forbidden_pursued"])),
           "pursued_before_door_opened": dict(Counter(x for r in recs for x in r["pursued"])),
           "door_opened_by": dict(Counter(r["opened_by"] for r in recs)),
           "tiles_in_the_way": {"subgoals": len(tiles), "door": sum(t["is_door"] for t in tiles),
                                "other": dict(Counter(f"{t['shows']} -> {t['made']}" for t in tiles if not t["is_door"])),
                                "confirmed_blocking": sum(t["confirmed_blocking"] for t in tiles)},
           "steps_by_kind": dict(walk),
           "failed_acts": int(sum(r["failed_acts"] for r in recs)),
           "actions_refused": int(sum(r["refused"] for r in recs)),
           "predictions_wrong": int(sum(r["pred_wrong"] for r in recs)),
           "largest_view_mismatch_placing_the_agent": int(max(r["max_mismatch"] for r in recs)),
           "conditions_computed_per_move": round(sum(r["computed"] for r in recs) / max(moves, 1), 1),
           "seconds_per_layout_mean": round(float(np.mean([r["seconds"] for r in recs])), 3),
           "seconds_per_layout_max": round(float(np.max([r["seconds"] for r in recs])), 2),
           "chains_at_start": [r["chain0"] for r in recs[:3]]}
    return res, recs


# ---------------------------------------------------------------- arm 1: the shortest route (evaluator)

def _shortest(job):
    out = []
    for lay in job:
        row = {}
        for rule in (lay.rule,) + (("key", "switch") if lay.rule == "either" else ()):
            row[rule] = closer._shortest([dataclasses.replace(lay, rule=rule)])[0]
        out.append(row)
    return out


def shortest(pool, test, world):
    chunks = [list(c) for c in np.array_split(np.array(test, dtype=object), 40) if len(c)]
    rows = [r for p in pool.map(_shortest, chunks) for r in p]
    sh = np.array([r[world] if r[world] is not None else -1 for r in rows])
    res = {"solved_within_budget": round(float(((sh > 0) & (sh <= BUDGET)).mean()), 4),
           "mean_steps": round(float(sh[sh > 0].mean()), 2)}
    if world == "either":
        k = np.array([r["key"] or 10 ** 6 for r in rows])
        s = np.array([r["switch"] or 10 ** 6 for r in rows])
        res["shorter_option"] = {"key": int((k < s).sum()), "switch": int((s < k).sum()), "equal": int((s == k).sum())}
    return res, rows


# ---------------------------------------------------------------- the gate's rules check, four worlds

def rules_check(world, rep):
    found = {r["key"]: sorted(sorted(x["if"]) for x in r["rules"]) for r in rep["rules"]
             if r["key"].startswith("toggle closed door")}
    ok = len(found) == 3
    for key, rules in found.items():
        c = key.split()[-1]
        k, s = f"held = key {c}", "switch on in view"
        want = {"key": [[k]], "switch": [[s]], "either": sorted([[k], [s]]), "both": [sorted([k, s])]}[world]
        ok &= rules == want
    return {"door_rules": found, "exactly_as_expected": bool(ok)}


# ---------------------------------------------------------------- arm 3: card 028's pipeline

def arm3(pool_fn, test, tr, seq, st0, E, V0, V1, Vseq, Vst0, endseq, log):
    S = {"tr": tr, "V0": V0, "V1": V1, "no0": np.zeros(len(V0), bool), "end1": tr["term1"].astype(bool),
         "seq": seq, "Vseq": Vseq, "endseq": endseq, "st0": st0, "Vst0": Vst0}
    exact = Kd.exact_kinds(E, lambda m: None)
    code_exact = np.full(256, -1, np.int64)
    for u, k in exact.items():
        code_exact[u] = k
    wc = np.bincount(E["f0"], weights=tr["w0"].astype(np.float64))
    floor_app = int(np.argsort(-wc)[0])
    pool = pool_fn()
    t0 = time.monotonic()
    F.LABEL_STATS.clear()
    tree, expanded, recs, rounds = F.grow_learned(pool, test, S, log)
    grown = round(time.monotonic() - t0, 1)
    ways = Kd.thing_ways(tree)
    L0l, L1l = F.labels(pool, tree.parent, tree.action, tr["ep"], [V0, V1], [S["no0"], S["end1"]],
                        sorted({tree.parent[n] for n in ways}))
    wkl = Kd.way_kinds(tree, ways, L0l, L1l, tr, E, exact)
    K2 = (code_exact, {n: set(v.get("codes", [])) for n, v in wkl.items()}, floor_app)
    res = F.act_arm(pool, test, tree, expanded, K2)
    pool.close()
    res["tree"] = {"nodes": len(tree.parent), "goals_expanded": len(expanded), "rounds": len(rounds),
                   "seconds_growing": grown, "ways": [(t["path"], t["status"]) for t in tree.report()]}
    return res


# ---------------------------------------------------------------- main

def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    out = Path(args.get("--out", "runs/029_subgoals.json"))
    episodes, n_test = int(args.get("--episodes", 5000)), int(args.get("--layouts", 500))
    heldout = int(args.get("--heldout", 1000))
    worlds = args.get("--worlds", "key,switch,either,both").split(",")
    arm3_worlds = [w for w in args.get("--arm3", "either,both").split(",") if w]
    closer.MEASURE = "step"
    dl.MAX_GOALS = 10 ** 9
    dl.START_SHARE = 2.0
    dl.Exact = closer.Approach
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    res = {"note": "Card 029, tools/card029/subgoals.py; subgoals worked backward through card 028's learned model.",
           "episodes": episodes, "layouts": n_test, "heldout_episodes": heldout, "worlds": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    for world in worlds:
        r = res["worlds"][world] = {}
        tw = time.monotonic()
        log(f"=== {world} world")
        F.M = None
        pool = F.pool20()
        layouts, tr, seq, st0 = dl.collect(pool, world, episodes, 11, 0.0, seq_episodes=300)
        hlay, htr, _, _ = dl.collect(pool, world, heldout, 12, 0.0, seq_episodes=4)
        rng = np.random.default_rng(777)
        test = [ld.make_layout(8, world, rng) for _ in range(n_test)]
        t0 = time.monotonic()
        r["arm1_shortest"], sh_rows = shortest(pool, test, world)
        r["arm1_shortest"]["seconds"] = round(time.monotonic() - t0, 1)
        log(f"  arm 1, shortest route: {json.dumps(r['arm1_shortest'])}")
        pool.close()
        E = Kd.effects(tr)
        r["facts_check"] = Kd.effects_check(tr, E)
        V0, V1 = F.views(tr["c0"], tr["s0"]), F.views(tr["c1"], tr["s1"])
        Vh0, Vh1 = F.views(htr["c0"], htr["s0"]), F.views(htr["c1"], htr["s1"])
        t0 = time.monotonic()
        F.M, mrep = F.learn(tr, E, V0, V1, log, dev)
        r["model"] = {"seconds": round(time.monotonic() - t0, 1), "poses": mrep["poses"],
                      "pose_route_conflicts": mrep["pose_route_conflicts"], "rules": mrep["rules"]}
        c1 = F.effects_eval(Vh0, Vh1, htr["act"].astype(np.int64), htr["term1"].astype(bool),
                            htr["w0"].astype(np.float64))
        c1["rules"] = rules_check(world, mrep)
        r["model_check"] = c1
        log(f"  facts {r['facts_check']}; held-out effects exact {c1['all']} (lowest {c1['lowest']}) of {len(Vh0)}; "
            f"rules {json.dumps(c1['rules'])}")
        save()

        pool = F.pool20()
        t0 = time.monotonic()
        r["arm2_main"], recs2 = act_arm(pool, test, world)
        r["arm2_main"]["seconds"] = round(time.monotonic() - t0, 1)
        log(f"  RESULT arm 2 (main): {json.dumps(r['arm2_main'])}")
        t0 = time.monotonic()
        r["arm4_no_rules"], _ = act_arm(pool, test, world, rules=False)
        r["arm4_no_rules"]["seconds"] = round(time.monotonic() - t0, 1)
        log(f"  RESULT arm 4 (no rules): {json.dumps({k: v for k, v in r['arm4_no_rules'].items() if k != 'chains_at_start'})}")
        pool.close()
        if world == "either":
            agree = Counter()
            for rec, row in zip(recs2, sh_rows):
                k, s = row["key"] or 10 ** 6, row["switch"] or 10 ** 6
                shorter = "key" if k < s else ("switch" if s < k else "equal")
                agree[f"{rec['opened_by']} (shorter: {shorter})"] += 1
            r["either_choice_against_shorter"] = dict(agree)
        save()

        if world in arm3_worlds:
            t0 = time.monotonic()
            Vseq, Vst0 = F.views(seq["codes"], seq["st"]), F.views(st0["codes"], st0["st"])
            term_eps = set(tr["ep"][tr["term1"].astype(bool)].tolist())
            last = np.r_[seq["ep"][1:] != seq["ep"][:-1], True]
            endseq = last & np.isin(seq["ep"], list(term_eps))
            r["arm3_card028"] = arm3(F.pool20, test, tr, seq, st0, E, V0, V1, Vseq, Vst0, endseq, log)
            r["arm3_card028"]["seconds"] = round(time.monotonic() - t0, 1)
            log(f"  RESULT arm 3 (card 028's pipeline): {json.dumps(r['arm3_card028'])}")
            save()

        m2, ub = r["arm2_main"], r["arm1_shortest"]
        ratio = (m2["mean_steps_when_successful"] or 1e9) / ub["mean_steps"]
        lim = 1.1 if world in ("key", "switch") else 1.25
        r["gate"] = {"facts_agree": all(v == 1.0 for v in r["facts_check"].values()),
                     "effects_exact": c1["lowest"] >= 0.999, "rules_as_expected": c1["rules"]["exactly_as_expected"],
                     "shortest_solves_all": ub["solved_within_budget"] == 1.0,
                     "baseline_below_98": r["arm4_no_rules"]["success"] < 0.98}
        r["criterion_1_subgoals"] = {"share": m2["subgoals_ok_share"], "pass": m2["subgoals_ok_share"] >= 0.99}
        r["criterion_acting"] = {"which": 2 if world in ("key", "switch") else 3,
                                 "success_98": m2["success"] >= 0.98, "random_share_1pct": m2["random_share"] <= 0.01,
                                 "steps_ratio": round(ratio, 3), f"steps_within_{lim}x": ratio <= lim}
        r["criterion_acting"]["pass"] = all(v for k, v in r["criterion_acting"].items() if k not in ("which", "steps_ratio"))
        r["seconds"] = round(time.monotonic() - tw, 1)
        log(f"  GATE {world}: {r['gate']}; criterion 1: {r['criterion_1_subgoals']}; acting: {r['criterion_acting']}; "
            f"world took {r['seconds']}s")
        save()
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
