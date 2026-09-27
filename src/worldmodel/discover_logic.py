"""Card 012: discovery end to end in the logic-door worlds.

From pixels and one supplied goal (on the goal square), the agent finds each
goal's ways to succeed (every action the evidence supports), each way's
condition ("a ready state can be reached by walking", read from a reach
value that barely shrinks with distance), checks each condition on its own
experience, and treats each condition as a new goal, fine-tuning its encoder
on every goal found so far (card 008). The same procedure with the
simulator's exact values is the feasibility gate. The simulator is used only
for that gate and to check what the learned conditions mean and do.

Storage (declared, not part of the mechanism): of each world's random play,
every transition in which something other than the agent changed, every
goal arrival, and a uniform 1/8 of the rest are kept; weights keep counts
natural. Which transitions are kept depends only on the state and action,
so it does not bias what an action achieves from a state.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

from .envs import logicdoor as ld
from .envs.keydoor_render import encode_logic_states, render, tile_images
from .logic_conditions import Data, log_evidence

ACTIONS = ld.ACTION_NAMES
MOVES = (ld.LEFT, ld.RIGHT, ld.FORWARD)
GAMMA_WALK = 0.95          # walk value: for moving (closer is higher)
GAMMA_REACH = 0.99         # reach value: "can get there at all" (see card 012)
MAX_DEPTH = 6
MAX_GOALS = 16             # declared compute budget: goals in the tree
START_SHARE = 0.5
K_MAX = MAX_GOALS
CHUNK = 250                # episodes per collection job


# ---------------------------------------------------------------- states as arrays

def to_arr(s):
    x, y, d, c, k, ds, door, sw, v = s
    return (x, y, d, c, k[0] if k else -1, k[1] if k else -1, ds[0] if ds else -1, ds[1] if ds else -1,
            door, sw, v)


def to_tup(a):
    a = [int(v) for v in a]
    return (a[0], a[1], a[2], a[3], None if a[4] < 0 else (a[4], a[5]), None if a[6] < 0 else (a[6], a[7]),
            a[8], a[9], a[10])


# ---------------------------------------------------------------- data

def _collect_chunk(job):
    rule, n, seed, play, p_uniform, n_seq = job
    rng = np.random.default_rng(seed + 7)
    data = ld.collect(rule, n, seed, play_starts=play)
    tr = {k: [] for k in ("c0", "c1", "act", "w0", "ep", "s0", "s1", "term1")}
    seq = {k: [] for k in ("codes", "st", "act", "ep")}
    st0 = {k: [] for k in ("codes", "st", "ep")}
    layouts = []
    for e, ep in enumerate(data):
        lay, S, A = ep["layout"], ep["states"], np.array(ep["actions"], np.int8)
        layouts.append(lay)
        codes = encode_logic_states(lay, S)
        st = np.array([to_arr(s) for s in S], np.int8)
        T = len(A)
        obj = codes // 5
        changed = (obj[1:] != obj[:-1]).reshape(T, -1).any(1)
        term = np.array([(s[0], s[1]) == lay.goal for s in S[1:]])
        forced = changed | term
        i = np.flatnonzero(forced | (rng.random(T) < p_uniform))
        tr["c0"].append(codes[i]), tr["c1"].append(codes[i + 1]), tr["act"].append(A[i])
        tr["w0"].append(np.where(forced[i], 1.0, 1 / p_uniform).astype(np.float32))
        tr["ep"].append(np.full(len(i), e, np.int32)), tr["s0"].append(st[i]), tr["s1"].append(st[i + 1])
        tr["term1"].append(term[i])
        if e < n_seq:
            seq["codes"].append(codes), seq["st"].append(st), seq["act"].append(np.append(A, -1).astype(np.int8))
            seq["ep"].append(np.full(len(S), e, np.int32))
        if st[0, 3] == 0 and st[0, 9] == 0:       # not a play start
            st0["codes"].append(codes[:1]), st0["st"].append(st[:1]), st0["ep"].append(np.array([e], np.int32))
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return layouts, cat(tr), cat(seq), cat(st0)


def collect(pool, rule, episodes, seed, play, p_uniform=1 / 8, seq_episodes=1500):
    jobs, n_chunks = [], (episodes + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        n = min(CHUNK, episodes - c * CHUNK)
        jobs.append((rule, n, seed * 100003 + c, play, p_uniform, (seq_episodes + n_chunks - 1) // n_chunks))
    parts = pool.map(_collect_chunk, jobs)
    layouts, out = [], [{}, {}, {}]
    for c, part in enumerate(parts):
        base = len(layouts)
        layouts += part[0]
        for k, d in enumerate(part[1:]):
            d["ep"] = d["ep"] + base
            for key, v in d.items():
                out[k].setdefault(key, []).append(v)
    tr, seq, st0 = [{k: np.concatenate(v) for k, v in d.items()} for d in out]
    return layouts, tr, seq, st0


# ---------------------------------------------------------------- exact goals

class Exact:
    """Exact goals of a discovery tree on one layout (memoised). Goal 0 is
    the goal square; goal n (parent p, action a) holds when p holds or a
    state where a achieves p can be reached from here by walking."""

    def __init__(self, lay, parent, action):
        self.lay, self.parent, self.action = lay, parent, action
        self.memo, self.comps, self.reach, self.dist = {}, {}, {}, {}

    def component(self, rest):
        if rest not in self.comps:
            lay, n = self.lay, self.lay.size
            dummy = (0, 0, 0, *rest)
            cells = [(x, y) for x in range(1, n - 1) for y in range(1, n - 1) if (x, y) != lay.goal
                     and (ld.cell(lay, dummy, (x, y)) == "empty" or (ld.cell(lay, dummy, (x, y)) == "door" and rest[3]))]
            comp, members, edges = {}, [], {}
            for x, y in cells:
                for d in range(4):
                    if (x, y, d) in comp:
                        continue
                    cid = len(members)
                    members.append([])
                    comp[(x, y, d)] = cid
                    todo = [(x, y, d)]
                    while todo:
                        u = todo.pop()
                        members[cid].append(u)
                        edges[u] = []
                        for a in MOVES:
                            t = ld.step(lay, (*u, *rest), a)[0][:3]
                            if (t[0], t[1]) == lay.goal:
                                continue
                            edges[u].append((a, t))
                            if t not in comp:
                                comp[t] = cid
                                todo.append(t)
            self.comps[rest] = (comp, members, edges)
        return self.comps[rest]

    def holds(self, n, s):
        key = (n, s)
        v = self.memo.get(key)
        if v is None:
            v = self.memo[key] = self._holds(n, s)
        return v

    def _holds(self, n, s):
        if n == 0:
            return (s[0], s[1]) == self.lay.goal
        p = self.parent[n]
        if self.holds(p, s):
            return True
        rest = s[3:]
        comp, members, _ = self.component(rest)
        cid = comp.get(s[:3])
        if cid is None:
            return False
        k = (n, rest, cid)
        if k not in self.reach:
            a = self.action[n]
            self.reach[k] = any(self.ready(p, a, (*u, *rest)) for u in members[cid])
        return self.reach[k]

    def ready(self, g, a, s):
        return not self.holds(g, s) and self.holds(g, ld.step(self.lay, s, a)[0])

    def distances(self, n, s):
        """Walking steps from every state of s's component to a state where
        goal n's way (parent's goal, action) is ready (reverse BFS)."""
        rest = s[3:]
        comp, members, edges = self.component(rest)
        cid = comp.get(s[:3])
        k = (n, rest, cid)
        if k not in self.dist:
            p, a = self.parent[n], self.action[n]
            rev = {u: [] for u in members[cid]}
            for u in members[cid]:
                for _, t in edges[u]:
                    rev[t].append(u)
            dist = {u: 0 for u in members[cid] if self.ready(p, a, (*u, *rest))}
            todo = deque(dist)
            while todo:
                u = todo.popleft()
                for v in rev[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        todo.append(v)
            self.dist[k] = dist
        return self.dist[k]

    def move_towards(self, n, s):
        dist = self.distances(n, s)
        _, _, edges = self.component(s[3:])
        here = dist.get(s[:3])
        if here is None:
            return None
        for a, t in edges[s[:3]]:
            if dist.get(t, 10 ** 9) < here:
                return a
        return None


def _exact_labels(job):
    layouts, parent, action, ep, arrays, nodes = job
    outs = [np.zeros((len(ep), len(parent)), bool) for _ in arrays]
    cur, ex = None, None
    for i in range(len(ep)):
        if ep[i] != cur:
            cur = ep[i]
            ex = Exact(layouts[cur], parent, action)
        for out, states in zip(outs, arrays):
            s = to_tup(states[i])
            for n in nodes:
                out[i, n] = ex.holds(n, s)
    return outs


def exact_labels(pool, layouts, parent, action, ep, arrays, nodes=None, jobs=96):
    """Exact goals of the given nodes (default all) on rows (sorted by
    episode) of each array; other columns are left False."""
    nodes = list(range(len(parent))) if nodes is None else list(nodes)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    sel = starts[np.linspace(0, len(starts) - 1, min(jobs, len(starts))).astype(int)]
    bounds = sorted(set(sel.tolist()) | {len(ep)})
    tasks = []
    for a, b in zip(bounds, bounds[1:]):
        eps = np.unique(ep[a:b])
        tasks.append(({int(e): layouts[e] for e in eps}, parent, action, ep[a:b], [x[a:b] for x in arrays], nodes))
    parts = pool.map(_exact_labels, tasks)
    return [np.concatenate([p[k] for p in parts]) for k in range(len(arrays))]


# ---------------------------------------------------------------- evidence tests

def way_gains(act, ready, success, w):
    """For each action: evidence gained (nats) by the rule "this action, taken
    where it is predicted to succeed, achieves the goal" over no rule (card
    010's evidence and structure cost)."""
    counts = Counter()
    for a in range(len(ACTIONS)):
        for r in (False, True):
            for y in (False, True):
                m = (act == a) & (ready == r) & (success == y)
                if m.any():
                    counts[((ACTIONS[a], str(r)), y)] += float(w[m].sum())
    d = Data(counts, keys=("action", "ready"))
    base = log_evidence(d, [])
    idx = {a: j for j, a in enumerate(d.atoms)}
    out = {}
    for a in range(len(ACTIONS)):
        ja, jr = idx.get(("action", ACTIONS[a])), idx.get(("ready", "True"))
        out[a] = -np.inf if ja is None or jr is None else log_evidence(d, [[ja, jr]]) - base
    return out


def check_gain(cond, y):
    """Self-check: evidence (nats) that success is likelier where the
    condition is on (the rule must be a route to success)."""
    counts = Counter()
    for c in (False, True):
        for v in (False, True):
            m = (cond == c) & (y == v)
            if m.any():
                counts[((str(c),), v)] += int(m.sum())
    d = Data(counts, keys=("condition",))
    j = {a: i for i, a in enumerate(d.atoms)}.get(("condition", "True"))
    if j is None:
        return -np.inf
    return log_evidence(d, [[j]]) - log_evidence(d, [])


def walk_then_act(act, ep, goal, a):
    """From each frame of the agent's own play (goal off): did it walk (turns,
    forward) and then take action a, achieving the goal, before doing
    anything else?"""
    n = len(act)
    y = np.zeros(n, bool)
    nxt = False
    for t in range(n - 1, -1, -1):
        last = t == n - 1 or ep[t + 1] != ep[t]
        if last:
            y[t] = nxt = False
            continue
        hit = act[t] == a and goal[t + 1] and not goal[t]
        if hit:
            nxt = True
        elif act[t] in MOVES and not goal[t + 1]:
            nxt = nxt
        else:
            nxt = False
        y[t] = nxt
    return y


def persistence(on, ep):
    same = ep[1:] == ep[:-1]
    m = same & on[:-1]
    return float(on[1:][m].mean()) if m.any() else float("nan")


def auc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]          # average ranks for ties
    npos = label.sum()
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * (~label).sum()))


# ---------------------------------------------------------------- the tree

class Tree:
    def __init__(self):
        self.parent, self.action, self.depth, self.status = [-1], [-1], [1], ["expanded"]
        self.gain, self.check = [None], [None]

    def add(self, p, a, gain):
        self.parent.append(p), self.action.append(a), self.depth.append(self.depth[p] + 1)
        self.status.append("new"), self.gain.append(gain), self.check.append(None)
        return len(self.parent) - 1

    def path(self, n):
        out = []
        while n > 0:
            out.append(ACTIONS[self.action[n]])
            n = self.parent[n]
        return tuple(reversed(out))

    def accepted(self, n):
        while n > 0:
            if self.status[n] == "rejected by self-check":
                return False
            n = self.parent[n]
        return True

    def children(self, g):
        return [c for c in range(1, len(self.parent)) if self.parent[c] == g and self.accepted(c)]

    @classmethod
    def from_report(cls, rep):
        t = cls()
        ids = {(): 0}
        t.status[0] = rep[0]["status"]
        for r in rep[1:]:
            path = tuple(r["path"].split("/"))
            n = t.add(ids[path[:-1]], ACTIONS.index(path[-1]), r["way_evidence_nats"])
            t.status[n], t.check[n] = r["status"], r["self_check"]
            ids[path] = n
        return t

    def report(self):
        return [{"id": n, "path": "/".join(self.path(n)) or "(goal square)", "depth": self.depth[n],
                 "status": self.status[n],
                 "way_evidence_nats": None if self.gain[n] is None else round(float(self.gain[n]), 1),
                 "self_check": self.check[n]} for n in range(len(self.parent))]


def expand(tree, frontier, gains_of, log):
    """Add a child per evidence-supported way of each frontier goal."""
    new = []
    for g in frontier:
        gains = gains_of(g)
        ways = sorted((a for a in gains if gains[a] > 0), key=lambda a: -gains[a])
        log(f"  goal {g} {tree.path(g)}: way evidence (nats) "
            + ", ".join(f"{ACTIONS[a]} {gains[a]:.0f}" for a in sorted(gains) if np.isfinite(gains[a])))
        if not ways:
            tree.status[g] = "leaf: no action supported by evidence"
            continue
        tree.status[g] = "expanded"
        for a in ways:
            if len(tree.parent) >= MAX_GOALS:
                tree.status[g] += " (goal budget reached)"
                break
            new.append(tree.add(g, a, gains[a]))
    return new


def settle(tree, new, start_on, check, log):
    """Self-check and stopping rules for the new conditions; returns the next frontier."""
    frontier = []
    for c in new:
        gain, rates = check(c)
        tree.check[c] = {"evidence_nats": round(float(gain), 1), **rates}
        share = float(start_on(c))
        tree.check[c]["on_at_start"] = round(share, 4)
        if not gain > 0:
            tree.status[c] = "rejected by self-check"
        elif share > START_SHARE:
            tree.status[c] = "leaf: on at start"
        elif tree.depth[c] >= MAX_DEPTH:
            tree.status[c] = "leaf: depth limit"
        else:
            tree.status[c] = "frontier"
            frontier.append(c)
        log(f"  condition {c} {tree.path(c)}: {tree.status[c]}; {tree.check[c]}")
    return frontier


# ---------------------------------------------------------------- exact procedure (gate)

def run_exact(pool, layouts, tr, seq, st0, log, episodes, seq_episodes):
    """The same procedure with exact goals, on the first `episodes` episodes
    (self-check on the first `seq_episodes` of the sequence set)."""
    keep = tr["ep"] < episodes
    tr = {k: v[keep] for k, v in tr.items()}
    keep = seq["ep"] < np.unique(seq["ep"])[min(seq_episodes, len(np.unique(seq["ep"]))) - 1] + 1
    seq = {k: v[keep] for k, v in seq.items()}
    tree = Tree()
    frontier = [0]
    act, w0 = tr["act"], tr["w0"]
    while frontier:
        P, A = tree.parent, tree.action
        L0, L1 = exact_labels(pool, layouts, P, A, tr["ep"], [tr["s0"], tr["s1"]], frontier)

        def gains_of(g):
            m = ~L0[:, g]
            return way_gains(act[m], L1[m, g], L1[m, g], w0[m])

        new = expand(tree, frontier, gains_of, log)
        if not new:
            break
        need = sorted(set(new) | {tree.parent[c] for c in new})
        Ls, = exact_labels(pool, layouts, tree.parent, tree.action, seq["ep"], [seq["st"]], need)
        Lst, = exact_labels(pool, layouts, tree.parent, tree.action, st0["ep"], [st0["st"]], new)

        def check(c):
            g, a = tree.parent[c], tree.action[c]
            y = walk_then_act(seq["act"], seq["ep"], Ls[:, g], a)
            m = ~Ls[:, g]
            return check_gain(Ls[m, c], y[m]), {"rate_on": round(float(y[m & Ls[:, c]].mean()), 4),
                                                "rate_off": round(float(y[m & ~Ls[:, c]].mean()), 4)}

        frontier = settle(tree, new, lambda c: Lst[:, c].mean(), check, log)
    Ls, = exact_labels(pool, layouts, tree.parent, tree.action, seq["ep"], [seq["st"]])
    persist = [persistence(Ls[:, n], seq["ep"]) for n in range(len(tree.parent))]
    return tree, persist


EXPECTED = {   # goals (as action paths) the gate must find; leaves must be "on at start"
    "key": {("forward",): "", ("forward", "toggle"): "", ("forward", "toggle", "pickup"): "leaf"},
    "switch": {("forward",): "", ("forward", "toggle"): "", ("forward", "toggle", "toggle"): "leaf"},
    "either": {("forward",): "", ("forward", "toggle"): "", ("forward", "toggle", "pickup"): "leaf",
               ("forward", "toggle", "toggle"): "leaf"},
    "both": {("forward",): "", ("forward", "toggle"): "", ("forward", "toggle", "pickup"): "",
             ("forward", "toggle", "toggle"): "", ("forward", "toggle", "pickup", "toggle"): "leaf",
             ("forward", "toggle", "toggle", "pickup"): "leaf"},
}


def structure_check(world, tree):
    paths = {tree.path(n): n for n in range(1, len(tree.parent)) if tree.accepted(n)}
    out = {}
    for p, kind in EXPECTED[world].items():
        n = paths.get(p)
        ok = n is not None and (kind != "leaf" or tree.status[n] == "leaf: on at start")
        out["/".join(p)] = bool(ok)
    extra = sorted("/".join(p) for p in paths if p not in EXPECTED[world])
    return {"expected_found": out, "all_expected": all(out.values()), "extra_goals": extra}


# ---------------------------------------------------------------- learned model

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def make_model(width=256):
    torch, nn, F = _torch()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 9 * 8, width), nn.ReLU())
            self.ach = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, K_MAX * len(ACTIONS)))
            self.walk = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))

    return Model()


class Trainer:
    """Encoder + achievement outputs for every goal so far + the goal
    square's walk value (card 008's encoder G, then continual fine-tuning).
    Event rows are over-sampled per goal with importance weights."""

    def __init__(self, model, tr, device, seed=0):
        torch, nn, F = _torch()
        self.torch, self.device = torch, device
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.model = model.to(device)
        self.target = copy.deepcopy(self.model).requires_grad_(False)
        self.tiles = torch.from_numpy(tile_images()).to(device)
        self.c0 = torch.from_numpy(tr["c0"]).to(device)
        self.c1 = torch.from_numpy(tr["c1"]).to(device)
        self.act = torch.from_numpy(tr["act"].astype(np.int64)).to(device)
        self.term1 = torch.from_numpy(tr["term1"]).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=3e-4, fused=True)
        self.params, self.tparams = list(self.model.parameters()), list(self.target.parameters())
        model, target, tiles, c0, c1, act, term1 = self.model, self.target, self.tiles, self.c0, self.c1, self.act, self.term1
        nA = len(ACTIONS)

        def step(r, y, m, wm):
            img0 = render(c0[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
            img1 = render(c1[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
            a = act[r]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z0 = model.enc(img0)
                la = model.ach(z0)
                q = model.walk(z0)
                with torch.no_grad():
                    q1 = target.walk(target.enc(img1))
            la = la.float().view(-1, K_MAX, nA).gather(2, a.view(-1, 1, 1).expand(-1, K_MAX, 1)).squeeze(2)
            bce = F.binary_cross_entropy_with_logits(la, y, reduction="none")
            loss_a = ((bce * m).sum(0) / m.sum(0).clamp_min(1e-6)).sum()
            q, q1 = torch.sigmoid(q.float()), torch.sigmoid(q1.float())
            qa = q.gather(1, a.clamp(max=2).view(-1, 1)).squeeze(1)
            yq = torch.where(term1[r], 1.0, GAMMA_WALK * q1.max(1).values)
            loss_w = ((qa - yq) ** 2 * wm).sum() / wm.sum().clamp_min(1.0)
            (loss_a + loss_w).backward()
            return loss_a.detach(), loss_w.detach()

        self.step = torch.compile(step)

    def train(self, L0, L1, active, updates, log, batch=512):
        torch = self.torch
        dev = self.device
        M = len(self.act)
        ev = [(g, torch.nonzero(~L0[:, g] & L1[:, g]).squeeze(1)) for g in active]
        ev = [(g, e) for g, e in ev if len(e)]
        n_e = (batch // 2) // max(len(ev), 1)
        n_u = batch - n_e * len(ev)
        evm = torch.zeros(M, len(ev), device=dev)
        for j, (_, e) in enumerate(ev):
            evm[e, j] = 1.0 / len(e)
        mask = torch.zeros(K_MAX, device=dev)
        mask[list(active)] = 1.0
        wm_all = (self.act <= 2).float()
        started = time.monotonic()
        self.model.train()
        for u in range(updates):
            r = torch.cat([torch.randint(M, (n_u,), device=dev, generator=self.gen)]
                          + [e[torch.randint(len(e), (n_e,), device=dev, generator=self.gen)] for _, e in ev])
            mix = n_u / batch / M + (n_e / batch) * evm[r].sum(1)
            w = (1.0 / M) / mix
            y = L1[r].float()
            m = (~L0[r]).float() * mask * w[:, None]
            self.opt.zero_grad(set_to_none=True)
            la, lw = self.step(r, y, m, wm_all[r])
            self.opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(self.tparams, self.params, 0.005)
            if u % 5000 == 0 or u == updates - 1:
                log(f"  train {u}: loss_a {float(la):.4f} loss_w {float(lw):.4f} "
                    f"{(u + 1) / (time.monotonic() - started):.0f} upd/s")
        self.model.eval()

    def encode(self, codes, chunk=8192):
        torch = self.torch
        codes = torch.as_tensor(codes).to(self.device)
        out = torch.empty(len(codes), 256, dtype=torch.half, device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(codes), chunk):
                img = render(codes[s:s + chunk].long(), self.tiles).permute(0, 3, 1, 2).float() / 255.0
                out[s:s + chunk] = self.model.enc(img).half()
        return out

    def achieve(self, Z, chunk=262144):
        """P(goal on next | z, action) for every goal output: (N, K_MAX, 6), half."""
        torch = self.torch
        out = torch.empty(len(Z), K_MAX, len(ACTIONS), dtype=torch.half, device=self.device)
        with torch.no_grad():
            for s in range(0, len(Z), chunk):
                out[s:s + chunk] = torch.sigmoid(self.model.ach(Z[s:s + chunk].float())).view(-1, K_MAX, len(ACTIONS)).half()
        return out


def MinHead(n, width=256):
    """Two value networks; the smaller estimate is used, both as the
    bootstrap target and when read (clipped double Q-learning, Fujimoto et
    al. 2018): with a discount near 1, taking the max of one noisy estimate
    lets values of unreachable states creep up (card 012, run 1)."""
    torch, nn, F = _torch()

    class _MinHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.nets = nn.ModuleList(
                nn.Sequential(nn.Linear(256, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                              nn.Linear(width, n * 6)) for _ in range(2))
            with torch.no_grad():
                for net in self.nets:
                    net[-1].bias.fill_(-4.0)      # start near "cannot reach": values only rise from ready states

        def forward(self, z):
            return torch.minimum(self.nets[0](z), self.nets[1](z))

        def both(self, z):
            return self.nets[0](z), self.nets[1](z)

    return _MinHead()


def train_values(Z0, Z1, act, term1, ready0, ready1, device, updates, log, batch=4096, seed=0):
    """Walk (0.95) and reach (0.99) values to each way's ready states:
    Q-learning on the walking steps, one output pair per way."""
    torch, nn, F = _torch()
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    n = ready0.shape[1]
    head = MinHead(n).to(device)
    target = copy.deepcopy(head).requires_grad_(False)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4, fused=True)
    rows = torch.nonzero(act <= 2).squeeze(1)
    gam = torch.tensor([GAMMA_WALK, GAMMA_REACH], device=device).view(1, 1, 2)
    params, tparams = list(head.parameters()), list(target.parameters())
    started = time.monotonic()
    for u in range(updates):
        r = rows[torch.randint(len(rows), (batch,), device=device, generator=gen)]
        a = act[r]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qs = head.both(Z0[r])
            with torch.no_grad():
                q1 = target(Z1[r])
        with torch.no_grad():
            q1 = torch.sigmoid(q1.float()).view(-1, n, 2, 3).max(3).values
            y = torch.where(ready1[r][:, :, None], 1.0, torch.where(term1[r][:, None, None], 0.0, gam * q1))
        mk = (~ready0[r]).float()[:, :, None]
        loss = 0.0
        for q in qs:
            q = torch.sigmoid(q.float()).view(-1, n, 2, 3)
            qa = q.gather(3, a.view(-1, 1, 1, 1).expand(-1, n, 2, 1)).squeeze(3)
            loss = loss + ((qa - y) ** 2 * mk).sum() / mk.sum().clamp_min(1.0) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(tparams, params, 0.01)
        if u % 10000 == 0 or u == updates - 1:
            log(f"  values {u}: loss {float(loss.detach()):.5f} {(u + 1) / (time.monotonic() - started):.0f} upd/s")
    head.eval()
    return head


def values_of(head, Z, n, chunk=131072, with_q=True):
    """(walk, reach) best-move values and walk Q per move: (N, n), (N, n), (N, n, 3)."""
    torch, _, _ = _torch()
    W, R, Q = [], [], []
    with torch.no_grad():
        for s in range(0, len(Z), chunk):
            q = torch.sigmoid(head(Z[s:s + chunk].float())).view(-1, n, 2, 3)
            W.append(q[:, :, 0].max(2).values), R.append(q[:, :, 1].max(2).values)
            if with_q:
                Q.append(q[:, :, 0])
    return torch.cat(W), torch.cat(R), (torch.cat(Q) if with_q else None)


class LearnedTree:
    """Every goal of the discovered tree on any frames, from the network."""

    def __init__(self, tree, trainer, head, way_index):
        self.tree, self.trainer, self.head, self.way_index = tree, trainer, head, way_index

    def evaluate(self, Z, root_on):
        torch, _, _ = _torch()
        t = self.tree
        n_nodes = len(t.parent)
        P = self.trainer.achieve(Z)
        W, R, Q = values_of(self.head, Z, len(self.way_index))
        L = torch.zeros(len(Z), n_nodes, dtype=torch.bool, device=Z.device)
        score = torch.zeros(len(Z), n_nodes, device=Z.device)
        L[:, 0] = root_on
        for c in range(1, n_nodes):
            if c not in self.way_index:
                continue
            g, a, j = t.parent[c], t.action[c], self.way_index[c]
            A = P[:, g, a].float()
            ready = (A > 0.5) & ~L[:, g]
            L[:, c] = L[:, g] | ready | (R[:, j] > 0.5)
            score[:, c] = torch.where(L[:, g], 1.0, torch.maximum(A, R[:, j]))
        return {"L": L, "score": score, "P": P, "W": W, "R": R, "Q": Q}


# ---------------------------------------------------------------- learned procedure

def run_learned(tree, trainer, tr, seq, st0, layouts, args, log):
    torch, _, _ = _torch()
    dev = trainer.device
    M = len(tr["act"])
    act = trainer.act
    term1 = trainer.term1
    walking = (act <= 2) & ~term1
    L0 = torch.zeros(M, K_MAX, dtype=torch.bool, device=dev)
    L1 = torch.zeros(M, K_MAX, dtype=torch.bool, device=dev)
    L1[:, 0] = term1
    seq_root = torch.from_numpy(np.array([(s[0], s[1]) == layouts[e].goal for s, e in zip(seq["st"], seq["ep"])])).to(dev)
    Ls = torch.zeros(len(seq["act"]), K_MAX, dtype=torch.bool, device=dev)
    Ls[:, 0] = seq_root
    Lst = torch.zeros(len(st0["ep"]), K_MAX, dtype=torch.bool, device=dev)
    seq_act, seq_ep = seq["act"], seq["ep"]
    w0 = tr["w0"]
    act_np = tr["act"]
    timings = {}
    t0 = time.monotonic()
    trainer.train(L0, L1, [0], args.pretrain, log)
    timings["pretrain"] = round(time.monotonic() - t0, 1)
    frontier = [0]
    first = True
    while frontier:
        t0 = time.monotonic()
        if not first:
            active = [n for n in range(len(tree.parent)) if tree.accepted(n)]
            trainer.train(L0, L1, active, args.finetune, log)
        first = False
        Z0, Z1 = trainer.encode(tr["c0"]), trainer.encode(tr["c1"])
        Zs, Zst = trainer.encode(seq["codes"]), trainer.encode(st0["codes"])
        P0 = trainer.achieve(Z0)

        def gains_of(g):
            m = (~L0[:, g]).cpu().numpy()
            a = act_np[m]
            p = P0[:, g, :].float().cpu().numpy()[m]
            ready = p[np.arange(len(a)), a] > 0.5
            return way_gains(a, ready, L1[:, g].cpu().numpy()[m], w0[m])

        new = expand(tree, frontier, gains_of, log)
        if not new:
            break
        P1, Ps, Pst = trainer.achieve(Z1), trainer.achieve(Zs), trainer.achieve(Zst)
        gs = [tree.parent[c] for c in new]
        As = [tree.action[c] for c in new]
        r0 = torch.stack([(P0[:, g, a] > 0.5) & ~L0[:, g] for g, a in zip(gs, As)], 1)
        r1 = torch.stack([(P1[:, g, a] > 0.5) & ~L1[:, g] for g, a in zip(gs, As)], 1)
        del P0, P1
        head = train_values(Z0, Z1, act, term1, r0, r1, dev, args.value_updates, log)
        for Z, L, P, rdy in ((Z0, L0, None, r0), (Z1, L1, None, r1), (Zs, Ls, Ps, None), (Zst, Lst, Pst, None)):
            _, R, _ = values_of(head, Z, len(new), with_q=False)
            for j, (c, g, a) in enumerate(zip(new, gs, As)):
                ready = rdy[:, j] if rdy is not None else (P[:, g, a] > 0.5) & ~L[:, g]
                L[:, c] = L[:, g] | ready | (R[:, j] > 0.5)
        # Walking cannot change what can be reached by walking: on a walking
        # step the condition is the same on both frames (on if either reads on).
        for c in new:
            either = L0[walking, c] | L1[walking, c]
            L0[walking, c] = either
            L1[walking, c] = either
        Ls_np = Ls.cpu().numpy()

        def check(c):
            g, a = tree.parent[c], tree.action[c]
            y = walk_then_act(seq_act, seq_ep, Ls_np[:, g], a)
            m = ~Ls_np[:, g]
            return check_gain(Ls_np[m, c], y[m]), {"rate_on": round(float(y[m & Ls_np[:, c]].mean()), 4),
                                                   "rate_off": round(float(y[m & ~Ls_np[:, c]].mean()), 4)}

        frontier = settle(tree, new, lambda c: float(Lst[:, c].float().mean()), check, log)
        timings[f"depth_{tree.depth[new[0]] - 1}"] = round(time.monotonic() - t0, 1)
        del Z0, Z1, Zs, Zst
    # Final: every way's values refit on the final encoder (card 008).
    t0 = time.monotonic()
    ways = [c for c in range(1, len(tree.parent)) if tree.accepted(c)]
    if not ways:
        return LearnedTree(tree, trainer, None, {}), [], timings
    Z0, Z1 = trainer.encode(tr["c0"]), trainer.encode(tr["c1"])
    P0, P1 = trainer.achieve(Z0), trainer.achieve(Z1)
    r0 = torch.stack([(P0[:, tree.parent[c], tree.action[c]] > 0.5) & ~L0[:, tree.parent[c]] for c in ways], 1)
    r1 = torch.stack([(P1[:, tree.parent[c], tree.action[c]] > 0.5) & ~L1[:, tree.parent[c]] for c in ways], 1)
    del P0, P1
    head = train_values(Z0, Z1, act, term1, r0, r1, dev, args.value_updates, log)
    timings["final_values"] = round(time.monotonic() - t0, 1)
    Ls_np = Ls.cpu().numpy()
    persist = [persistence(Ls_np[:, n], seq_ep) for n in range(len(tree.parent))]
    return LearnedTree(tree, trainer, head, {c: j for j, c in enumerate(ways)}), persist, timings


# ---------------------------------------------------------------- evaluation

def meaning_labels(layouts, eps, states):
    names, cols = [], []
    V = [ld.variables(layouts[e], to_tup(s)) for e, s in zip(eps, states)]
    keys = sorted({(k, v) for d in V for k, v in d.items() if k not in ("x", "y", "dir")}, key=str)
    for k, v in keys:
        names.append(f"{k}={v}")
        cols.append([d[k] == v for d in V])
    c, sw = states[:, 3], states[:, 9]
    for name, col in (("key and switch on", (c == 1) & (sw == 1)), ("key or switch on", (c == 1) | (sw == 1)),
                      ("empty hands and switch on", (c == 0) & (sw == 1)),
                      ("key and switch off", (c == 1) & (sw == 0))):
        names.append(name)
        cols.append(col)
    return names, np.array(cols, bool).T


def encode_pairs(pairs):
    return np.concatenate([encode_logic_states(l, [s]) for l, s in pairs]) if pairs else np.zeros((0, 9, 8), np.uint8)


def learned_on_states(lt, pairs, trainer):
    torch, _, _ = _torch()
    Z = trainer.encode(encode_pairs(pairs))
    root = torch.tensor([(s[0], s[1]) == l.goal for l, s in pairs], device=trainer.device)
    return lt.evaluate(Z, root)


def behaviour(lt, trainer, layouts, eps, states, Ltest, n=200, seed=0):
    """Criterion 2: from test frames with a condition on (off), can walking
    (exact) to some state and taking the way's action achieve its goal
    (goal read by the network)?"""
    rng = np.random.default_rng(seed)
    t = lt.tree
    out = {}
    for c in lt.way_index:
        g, a = t.parent[c], t.action[c]
        res = {}
        for name, pick in (("condition_on", Ltest[:, c] & ~Ltest[:, g]), ("condition_off", ~Ltest[:, c] & ~Ltest[:, g])):
            idx = np.flatnonzero(pick)
            idx = rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else idx
            pairs, owner = [], []
            for k, i in enumerate(idx):
                lay, s = layouts[eps[i]], to_tup(states[i])
                ex = Exact(lay, [-1], [-1])
                comp, members, _ = ex.component(s[3:])
                cid = comp.get(s[:3])
                for u in (members[cid] if cid is not None else []):
                    pairs.append((lay, ld.step(lay, (*u, *s[3:]), a)[0]))
                    owner.append(k)
            hit = np.zeros(len(idx), bool)
            if pairs:
                L = learned_on_states(lt, pairs, trainer)["L"][:, g].cpu().numpy()
                np.logical_or.at(hit, np.array(owner), L)
            res[name] = {"frames": int(len(idx)), "achieved": round(float(hit.mean()), 4) if len(idx) else None}
        out["/".join(t.path(c))] = res
    return out


def choose(tree, on, R, W, persist):
    """The shallowest goal (off) with a way whose condition holds; ties: the
    more lasting goal first, then the higher reach, then the higher walk value."""
    best, todo, seen = None, [0], set()
    while todo:
        g = todo.pop()
        if g in seen:
            continue
        seen.add(g)
        for c in tree.children(g):
            if on(c):
                key = (tree.depth[g], -np.nan_to_num(persist[g]), -R(c), -W(c), c)
                if best is None or key < best:
                    best = key
            else:
                todo.append(c)
    return None if best is None else best[-1]


def act(world, layouts, lt, trainer, et, persist_l, persist_e, chooser, executor, budget=200, eps=0.05, seed=0,
        goal_only=False):
    """Criterion 3. chooser / executor: "learned" or "exact". et: the exact
    tree (from the gate); goals are matched between trees by action path."""
    torch, _, _ = _torch()
    rng = np.random.default_rng(seed)
    states = [ld.start_state(l) for l in layouts]
    exs = [Exact(l, et.parent, et.action) for l in layouts]
    done = np.zeros(len(layouts), bool)
    steps = np.full(len(layouts), budget)
    unmatched = 0
    lpath = {lt.tree.path(c): c for c in lt.way_index}
    epath = {et.path(c): c for c in range(1, len(et.parent)) if et.accepted(c)}
    root_way = lpath.get(("forward",))
    for step_i in range(budget):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        if chooser == "learned" or executor == "learned":
            out = learned_on_states(lt, [(layouts[i], states[i]) for i in live], trainer)
            L, P, W, R, Q = [out[k].float().cpu().numpy() for k in ("L", "P", "W", "R", "Q")]
        for j, i in enumerate(live):
            s, ex = states[i], exs[i]
            if goal_only:
                c = root_way
            elif chooser == "learned":
                wi = lt.way_index
                c = choose(lt.tree, lambda n: bool(L[j, n]), lambda n: R[j, wi[n]], lambda n: W[j, wi[n]], persist_l)
            else:
                def dist(n):
                    return ex.distances(n, s).get(s[:3], 10 ** 6)
                c = choose(et, lambda n: ex.holds(n, s), lambda n: float(dist(n) < 10 ** 6), lambda n: -dist(n),
                           persist_e)
            a = None
            if c is not None:
                path = (lt.tree if (chooser == "learned" or goal_only) else et).path(c)
                if executor == "learned":
                    cl = lpath.get(path)
                    if cl is None:
                        unmatched += 1
                    else:
                        g, aw, k = lt.tree.parent[cl], lt.tree.action[cl], lt.way_index[cl]
                        if P[j, g, aw] > 0.5 and not L[j, g]:
                            a = aw
                        elif rng.random() < eps:
                            a = int(rng.choice(MOVES))
                        else:
                            a = int(Q[j, k].argmax())
                else:
                    ce = epath.get(path)
                    if ce is None:
                        unmatched += 1
                    else:
                        g, aw = et.parent[ce], et.action[ce]
                        a = aw if ex.ready(g, aw, s) else ex.move_towards(ce, s)
            if a is None:
                a = int(rng.integers(len(ACTIONS)))
            states[i], end = ld.step(layouts[i], s, a)
            if end:
                done[i] = True
                steps[i] = step_i + 1
    return {"success": round(float(done.mean()), 4),
            "mean_steps_when_successful": round(float(steps[done].mean()), 1) if done.any() else None,
            "steps_without_a_matching_way": int(unmatched)}


def random_play(layouts, budget=200, seed=0):
    rng = np.random.default_rng(seed)
    ok = 0
    for l in layouts:
        s = ld.start_state(l)
        for a in rng.integers(len(ACTIONS), size=budget):
            s, end = ld.step(l, s, int(a))
            if end:
                ok += 1
                break
    return round(ok / len(layouts), 4)


def evaluate(pool, world, tree_e, persist_e, lt, persist_l, trainer, args, log):
    torch, _, _ = _torch()
    res = {}
    rng = np.random.default_rng(99)
    test_layouts, _, tseq, _ = collect(pool, world, args.test_episodes, 99, 0.0, seq_episodes=args.test_episodes)
    pick = np.sort(rng.choice(len(tseq["ep"]), min(args.test_frames, len(tseq["ep"])), replace=False))
    eps, states, codes = tseq["ep"][pick], tseq["st"][pick], tseq["codes"][pick]
    Le, = exact_labels(pool, test_layouts, tree_e.parent, tree_e.action, eps, [states])
    root = torch.from_numpy(Le[:, 0]).to(trainer.device)
    out = lt.evaluate(trainer.encode(codes), root)
    Ll, score = out["L"].cpu().numpy(), out["score"].float().cpu().numpy()
    names, labels = meaning_labels(test_layouts, eps, states)
    epath = {tree_e.path(n): n for n in range(len(tree_e.parent))}
    t = lt.tree
    crit1 = {}
    for c in lt.way_index:
        g = t.parent[c]
        m = ~Ll[:, g]
        e = epath.get(t.path(c))
        ent = {"on_share": round(float(Ll[m, c].mean()), 4)}
        if e is not None:
            me = ~Le[:, tree_e.parent[e]]
            ent["auc_vs_exact_counterpart"] = round(auc(score[me, c], Le[me, e]), 4)
            ent["agreement_with_exact"] = round(float((Ll[me, c] == Le[me, e]).mean()), 4)
        else:
            ent["auc_vs_exact_counterpart"] = None
        top = sorted(((auc(score[m, c], labels[m, j]), names[j]) for j in range(len(names))
                      if 0 < labels[m, j].sum() < m.sum()), key=lambda x: -np.nan_to_num(x[0]))[:3]
        ent["best_matches"] = [(round(a, 4), n) for a, n in top]
        crit1["/".join(t.path(c))] = ent
    res["criterion_1_meaning"] = crit1
    # Vase: does breaking the vase change any detector (learned; exact for reference)?
    vi = np.flatnonzero(states[:, 10] == 0)[:3000]
    flipped = states[vi].copy()
    flipped[:, 10] = 1
    pairs = [(test_layouts[eps[i]], to_tup(s)) for i, s in zip(vi, flipped)]
    Lf = learned_on_states(lt, pairs, trainer)["L"].cpu().numpy()
    Lef, = exact_labels(pool, test_layouts, tree_e.parent, tree_e.action, eps[vi], [flipped])
    res["vase_flip_changes"] = {"/".join(t.path(c)): {
        "learned": round(float((Lf[:, c] != Ll[vi, c]).mean()), 4),
        "exact": (round(float((Lef[:, epath[t.path(c)]] != Le[vi, epath[t.path(c)]]).mean()), 4)
                  if t.path(c) in epath else None)} for c in lt.way_index}
    # Duplicates: two conditions that agree on ≥ 99% of test frames.
    ws = sorted(lt.way_index)
    res["duplicates"] = [["/".join(t.path(a)), "/".join(t.path(b))] for i, a in enumerate(ws) for b in ws[i + 1:]
                         if (Ll[:, a] == Ll[:, b]).mean() >= 0.99]
    log("criterion 1 " + json.dumps(crit1, default=str))
    res["criterion_2_behaviour"] = behaviour(lt, trainer, test_layouts, eps, states, Ll)
    log("criterion 2 " + json.dumps(res["criterion_2_behaviour"]))
    lay_rng = np.random.default_rng(777)
    layouts = [ld.make_layout(8, world, lay_rng) for _ in range(args.act_layouts)]
    diag = layouts[:args.diag_layouts]
    res["criterion_3_acting"] = {
        "all_learned": act(world, layouts, lt, trainer, tree_e, persist_l, persist_e, "learned", "learned"),
        "goal_square_walk_value_only": act(world, layouts, lt, trainer, tree_e, persist_l, persist_e,
                                           "learned", "learned", goal_only=True),
        "random_play": random_play(layouts),
        f"diagnostic_{len(diag)}_layouts": {
            "exact_detectors_exact_rest": act(world, diag, lt, trainer, tree_e, persist_l, persist_e, "exact", "exact"),
            "exact_detectors_learned_rest": act(world, diag, lt, trainer, tree_e, persist_l, persist_e, "exact", "learned"),
            "learned_detectors_exact_reach": act(world, diag, lt, trainer, tree_e, persist_l, persist_e, "learned", "exact")},
    }
    log("criterion 3 " + json.dumps(res["criterion_3_acting"]))
    # Encoder gate (card 008): forward when facing the goal square.
    facing = np.array([ld.cell(test_layouts[e], to_tup(s), ld.front(to_tup(s))) == "goal" for e, s in zip(eps, states)])
    pf = out["P"][:, 0, ld.FORWARD].float().cpu().numpy()
    res["encoder_goal_square_check"] = {"facing_goal_right": round(float((pf[facing] > 0.5).mean()), 4) if facing.any() else None,
                                        "not_facing_right": round(float((pf[~facing & ~Le[:, 0]] <= 0.5).mean()), 4)}
    return res


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, choices=ld.RULES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--play-starts", type=float, default=None, help="default: 1/3 in 'both', else 0")
    parser.add_argument("--pretrain", type=int, default=30000)
    parser.add_argument("--finetune", type=int, default=12000)
    parser.add_argument("--value-updates", type=int, default=15000)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--test-frames", type=int, default=60000)
    parser.add_argument("--act-layouts", type=int, default=500)
    parser.add_argument("--diag-layouts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--gate-episodes", type=int, default=5000)
    parser.add_argument("--gate-from", type=Path, default=None, help="reuse the exact tree of an earlier result")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    play = args.play_starts if args.play_starts is not None else (1 / 3 if args.world == "both" else 0.0)
    pool = mp.get_context("fork").Pool(args.workers)      # before CUDA starts
    logf = open(args.out / "log.txt", "w")
    started = time.monotonic()

    def log(msg):
        msg = f"[{time.monotonic() - started:6.0f}s] {msg}"
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    result = {"world": args.world, "args": {k: str(v) for k, v in vars(args).items()}, "play_starts": play}

    def save():
        (args.out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")

    layouts, tr, seq, st0 = collect(pool, args.world, args.episodes, 11, play)
    result["data"] = {"transitions_kept": int(len(tr["act"])), "weighted_transitions": round(float(tr["w0"].sum())),
                      "goal_square_arrivals": int(tr["term1"].sum()), "sequence_frames": int(len(seq["act"])),
                      "start_frames_without_play_start": int(len(st0["ep"]))}
    log(f"data {result['data']}")
    t0 = time.monotonic()
    if args.gate_from:
        gate = json.loads(args.gate_from.read_text())["gate"]
        tree_e = Tree.from_report(gate["tree"])
        persist_e = gate.get("persistence") or [float("nan")] * len(tree_e.parent)
        log(f"gate reused from {args.gate_from}")
    else:
        tree_e, persist_e = run_exact(pool, layouts, tr, seq, st0, log, args.gate_episodes, 300)
    result["gate"] = {"tree": tree_e.report(), **structure_check(args.world, tree_e), "persistence": persist_e,
                      "seconds": round(time.monotonic() - t0, 1),
                      "reused_from": str(args.gate_from) if args.gate_from else None}
    log(f"gate {json.dumps({k: v for k, v in result['gate'].items() if k != 'tree'})}")
    save()
    if args.gate_only or not result["gate"]["all_expected"]:
        result["stopped"] = "gate only" if args.gate_only else "exact gate failed"
        save()
        return
    torch, _, _ = _torch()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    trainer = Trainer(make_model(), tr, device)
    tree_l = Tree()
    lt, persist_l, timings = run_learned(tree_l, trainer, tr, seq, st0, layouts, args, log)
    torch.save({"model": trainer.model.state_dict(), "values": lt.head.state_dict() if lt.head else None, "way_index": lt.way_index,
                "tree": tree_l.report()}, args.out / "nets.pt")
    result["learned"] = {"tree": tree_l.report(), **structure_check(args.world, tree_l), "timings": timings,
                         "persistence": {"/".join(tree_l.path(n)): round(p, 4) for n, p in enumerate(persist_l)}}
    log(f"learned {json.dumps({k: v for k, v in result['learned'].items() if k != 'tree'})}")
    save()
    if not lt.way_index:
        result["stopped"] = "no way learned for the goal square"
        save()
        return
    t0 = time.monotonic()
    result["evaluation"] = evaluate(pool, args.world, tree_e, persist_e, lt, persist_l, trainer, args, log)
    result["evaluation"]["seconds"] = round(time.monotonic() - t0, 1)
    result["seconds"] = round(time.monotonic() - started, 1)
    save()
    log("done")


if __name__ == "__main__":
    main()
