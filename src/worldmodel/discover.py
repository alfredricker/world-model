"""Card 007: discover a goal's conditions from pixels and one supplied goal.

For a goal g (level 1: on the goal square, supplied; below: discovered):
1. learn A(z, action) for g; the achieving action has the most frames with
   A > 0.5 (none if it achieves g fewer than 10 times in training);
2. ready states: A(z, achieving action) > 0.5;
3. learn a walk value to the ready states (Q-learning on movement steps);
4. the discovered condition c(z) = g holds, or ready, or walk value > 0.1;
5. c becomes the next level's goal.

z is card 005's frozen internal state. The same procedure with exact values
is the feasibility gate. The simulator's variables are used only to check
what a detector means and whether it behaves as a condition.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditions import collect, exact_chance
from .envs.keydoor import FORWARD, LEFT, RIGHT, Layout, start_state, step, variables
from .envs.keydoor_render import encode_states, render, tile_images
from .learn_keydoor import Frames, Net, auc, build
from .reach import walk_components

GAMMA = 0.95
ACTIONS = ("left", "right", "forward", "pickup", "toggle")
MAX_LEVELS = 4
MIN_SUCCESSES = 10


def on_goal(layout: Layout, s: tuple) -> bool:
    return (s[0], s[1]) == layout.goal


# ---------------------------------------------------------------- meaning

class Meaning:
    """What a detector tracks: best areas under the curve against the
    simulator's variable values, on a chosen subset of test states."""

    def __init__(self, vars_: list[dict]):
        keys = sorted({(k, v) for d in vars_ for k, v in d.items() if k not in ("x", "y", "dir")}, key=str)
        self.names = [f"{k}={v}" for k, v in keys]
        self.labels = np.array([[d[k] == v for k, v in keys] for d in vars_], bool)

    def __call__(self, score: np.ndarray, subset: np.ndarray, top: int = 3):
        res = []
        for j, name in enumerate(self.names):
            lab = self.labels[subset, j]
            if 0 < lab.sum() < len(lab):
                res.append((round(auc(score[subset], lab), 4), name))
        return sorted(res, reverse=True)[:top]


# ---------------------------------------------------------------- exact gate

class ExactLevel:
    def __init__(self, goal, action):
        self.goal, self.action = goal, action
        self.cache = {}

    def ready(self, layout, s):
        return not self.goal(layout, s) and self.goal(layout, step(layout, s, self.action)[0])

    def detector(self, layout, s):
        """Goal holds, or a ready state is within walking reach."""
        if self.goal(layout, s):
            return True
        key = (layout, s[3], s[4])
        if key not in self.cache:
            comp = walk_components(layout, s[3], s[4])
            self.cache[key] = (comp, {r for n, r in comp.items() if self.ready(layout, (*n, s[3], s[4]))})
        comp, roots = self.cache[key]
        return comp.get(s[:3]) in roots


def exact_gate(train_data, test_data, meaning: Meaning, test_pairs) -> dict:
    goal = on_goal
    levels = []
    for k in range(1, MAX_LEVELS + 1):
        counts = np.zeros(5, int)
        for ep in train_data:
            lay, S, A = ep["layout"], ep["states"], ep["actions"]
            for s, a, t in zip(S, A, S[1:]):
                if not goal(lay, s) and goal(lay, t):
                    counts[a] += 1
        info = {"level": k, "successes_per_action": dict(zip(ACTIONS, counts.tolist()))}
        if counts.max() < MIN_SUCCESSES:
            info["achieving_action"] = None
            levels.append(info)
            break
        lvl = ExactLevel(goal, int(counts.argmax()))
        info["achieving_action"] = ACTIONS[lvl.action]
        unmet = np.array([not goal(l, s) for l, s in test_pairs])
        det = np.array([lvl.detector(l, s) for l, s in test_pairs], float)
        info["detector_meaning"] = meaning(det, unmet)
        info["detector_on_share"] = round(float(det[unmet].mean()), 4)
        levels.append(info)
        goal = lvl.detector
    return {"levels": levels}


# ---------------------------------------------------------------- learned heads

class Head(nn.Module):
    def __init__(self, out: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, out))

    def forward(self, z):
        return self.net(z.float())


def train_achieve(Z, t_idx, t_act, goal_on, device, updates=15000, batch=2048, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = t_idx.new_tensor(np.flatnonzero(~goal_on[t_idx.cpu().numpy()]))
    gi = torch.from_numpy(goal_on).to(device)
    y_all = gi[t_idx[rows] + 1].float()
    events = rows[y_all.bool()]
    head = Head(5).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    m, n_u, n_e = len(rows), batch * 3 // 4, batch // 4
    w_event = 1.0 / (0.75 + 0.25 * m / max(len(events), 1))
    for _ in range(updates):
        r = torch.cat([rows[torch.randint(m, (n_u,), device=device, generator=gen)],
                       events[torch.randint(len(events), (n_e,), device=device, generator=gen)]])
        i = t_idx[r]
        y = gi[i + 1].float()
        logit = head(Z[i]).gather(1, t_act[r][:, None]).squeeze(1)
        w = torch.where(y.bool(), torch.tensor(w_event, device=device), torch.tensor(1 / 0.75, device=device))
        loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


def train_walk(Z, t_idx, t_act, ready, term, device, updates=40000, batch=2048, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    ri = torch.from_numpy(ready).to(device)
    ti = torch.from_numpy(term).to(device)
    moves = torch.from_numpy(np.flatnonzero((t_act.cpu().numpy() <= FORWARD)
                                            & ~ready[t_idx.cpu().numpy()])).to(device)
    head = Head(3).to(device)
    target = copy.deepcopy(head).requires_grad_(False)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(updates):
        r = moves[torch.randint(len(moves), (batch,), device=device, generator=gen)]
        i = t_idx[r]
        q = torch.sigmoid(head(Z[i])).gather(1, t_act[r][:, None]).squeeze(1)
        with torch.no_grad():
            nxt = torch.sigmoid(target(Z[i + 1])).max(1).values
            y = torch.where(ri[i + 1], 1.0, torch.where(ti[i + 1], 0.0, GAMMA * nxt))
        loss = F.mse_loss(q, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, pt in zip(head.parameters(), target.parameters()):
                pt.lerp_(p, 0.005)
    return head


@torch.no_grad()
def apply(head, Z, chunk=262144, sigmoid=True):
    out = [head(Z[s:s + chunk]) for s in range(0, len(Z), chunk)]
    out = torch.cat(out)
    return torch.sigmoid(out) if sigmoid else out


class LearnedLevel:
    def __init__(self, a_head, action, w_head):
        self.a_head, self.action, self.w_head = a_head, action, w_head

    @torch.no_grad()
    def parts(self, z, goal_on):
        """ready, walk value, detector for a batch of z with the level goal given."""
        a = torch.sigmoid(self.a_head(z))[:, self.action]
        w = torch.sigmoid(self.w_head(z)).max(1).values
        ready = (a > 0.5) & ~goal_on
        return ready, w, goal_on | ready | (w > 0.1)


def detectors(levels, z, on_goal_t):
    """Goal of each level and its detector, from the top down."""
    goals, dets, readies = [], [], []
    g = on_goal_t
    for lvl in levels:
        ready, _, c = lvl.parts(z, g)
        goals.append(g)
        readies.append(ready)
        dets.append(c)
        g = c
    return goals, readies, dets


# ---------------------------------------------------------------- behaviour checks

class Encoder:
    """Frames of arbitrary symbolic states -> card 005's z."""

    def __init__(self, net, device):
        self.net, self.device = net, device
        self.tiles = torch.from_numpy(tile_images()).to(device)

    @torch.no_grad()
    def __call__(self, pairs):
        codes = np.concatenate([encode_states(l, np.array([s])) for l, s in pairs])
        img = render(torch.from_numpy(codes).to(self.device).long(), self.tiles)
        return self.net.enc(img.permute(0, 3, 1, 2).float() / 255.0)


def walkable(layout, s):
    """Every state reachable from s by turns and forward steps."""
    seen, todo = {s}, deque([s])
    while todo:
        u = todo.popleft()
        if on_goal(layout, u):
            continue
        for a in (LEFT, RIGHT, FORWARD):
            t, _ = step(layout, u, a)
            if t not in seen:
                seen.add(t)
                todo.append(t)
    return list(seen)


def level_goal(levels, k, enc, pairs):
    """Does level k's goal hold in these states (k = 0: on the goal square)?"""
    g = torch.tensor([on_goal(l, s) for l, s in pairs], device=enc.device)
    if k == 0:
        return g
    z = enc(pairs)
    _, _, dets = detectors(levels[:k], z, g)
    return dets[-1]


def behaviour(levels, k, enc, test_pairs, det_k, unmet_k, n=300, seed=0):
    """From frames where level k's detector is on (off): can walking to some
    state and taking the achieving action achieve level k's goal?"""
    rng = np.random.default_rng(seed)
    out = {}
    for name, pick in (("detector_on", det_k & unmet_k), ("detector_off", ~det_k & unmet_k)):
        idx = np.flatnonzero(pick)
        idx = rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else idx
        hits = 0
        for i in idx:
            lay, s = test_pairs[i]
            nxt = [(lay, step(lay, u, levels[k].action)[0]) for u in walkable(lay, s)]
            hits += bool(level_goal(levels, k, enc, nxt).any())
        out[name] = {"frames": int(len(idx)), "achieved": round(hits / max(len(idx), 1), 4)}
    return out


def act(levels, enc, layouts, top_only: bool, budget=200, eps=0.05, seed=0):
    rng = np.random.default_rng(seed)
    states = [start_state(l) for l in layouts]
    done = np.zeros(len(layouts), bool)
    steps = np.full(len(layouts), budget)
    for t in range(budget):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        pairs = [(layouts[i], states[i]) for i in live]
        z = enc(pairs)
        g = torch.zeros(len(live), dtype=torch.bool, device=enc.device)
        _, readies, dets = detectors(levels, z, g)
        for j, i in enumerate(live):
            chosen = 0 if top_only else next((k for k, c in enumerate(dets) if c[j]), None)
            if chosen is None:
                a = int(rng.integers(5))
            elif readies[chosen][j]:
                a = levels[chosen].action
            elif rng.random() < eps:
                a = int(rng.integers(3))
            else:
                a = int(torch.sigmoid(levels[chosen].w_head(z[j:j + 1]))[0].argmax())
            states[i], end = step(layouts[i], states[i], a)
            if end:
                done[i] = True
                steps[i] = t + 1
    return {"success": round(float(done.mean()), 4),
            "mean_steps_when_successful": round(float(steps[done].mean()), 1) if done.any() else None}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, default=Path("runs/005_data10k_long/net.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--test-layouts", type=int, default=500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    started = time.monotonic()
    logf = open(args.out / "log.txt", "w")
    result = {}

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    def save():
        (args.out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")

    train_data = collect(8, args.episodes, 640, 3)
    test_data = collect(8, 500, 640, 99)
    test_pairs = [(ep["layout"], s) for ep in test_data for s in ep["states"]]
    meaning = Meaning([variables(l, s) for l, s in test_pairs])

    # Gate: the procedure with exact values.
    result["gate"] = exact_gate(train_data[:2000], test_data, meaning, test_pairs)
    save()
    log("gate " + json.dumps(result["gate"]))

    # Learned: internal state z of every training and test frame.
    net = Net().to(device)
    net.load_state_dict(torch.load(args.net))
    net.eval()
    d, dt = build(train_data), build(test_data)

    def all_z(dd):
        fr = Frames(dd["codes"], device)
        with torch.no_grad():
            return torch.cat([net.enc(fr(torch.arange(s, min(len(dd["codes"]), s + 16384), device=device))).half()
                              for s in range(0, len(dd["codes"]), 16384)])

    Z, Zt = all_z(d), all_z(dt)
    t_idx = torch.from_numpy(d["t_idx"]).to(device)
    t_act = torch.from_numpy(d["t_act"]).to(device)
    log(f"z ready: {len(Z)} train, {len(Zt)} test frames, {time.monotonic() - started:.0f} s")

    goal_on = d["hold"][:, 3].copy()           # level 1 goal: on the goal square (supplied)
    goal_on_t = dt["hold"][:, 3].copy()
    levels, result["levels"] = [], []
    for k in range(1, MAX_LEVELS + 1):
        i, a = d["t_idx"], d["t_act"]
        achieved = ~goal_on[i] & goal_on[i + 1]
        counts = np.bincount(a[achieved], minlength=5)
        info = {"level": k, "successes_per_action": dict(zip(ACTIONS, counts.tolist()))}
        if counts.max() < MIN_SUCCESSES:
            info["achieving_action"] = None
            result["levels"].append(info)
            save()
            log(f"level {k}: no achieving action " + json.dumps(info))
            break
        a_head = train_achieve(Z, t_idx, t_act, goal_on, device)
        A = apply(a_head, Z).cpu().numpy()
        frames_high = ((A > 0.5) & ~goal_on[:, None]).sum(0)
        action = int(frames_high.argmax())
        info["frames_with_A_above_half"] = dict(zip(ACTIONS, frames_high.tolist()))
        info["achieving_action"] = ACTIONS[action]
        ready = (A[:, action] > 0.5) & ~goal_on
        w_head = train_walk(Z, t_idx, t_act, ready, d["term"], device)
        lvl = LearnedLevel(a_head, action, w_head)
        levels.append(lvl)
        def run(ZZ, g):
            gg = torch.from_numpy(g).to(device)
            ws, cs = [], []
            for s0 in range(0, len(ZZ), 262144):
                _, w, c = lvl.parts(ZZ[s0:s0 + 262144].float(), gg[s0:s0 + 262144])
                ws.append(w)
                cs.append(c)
            return torch.cat(ws).cpu().numpy(), torch.cat(cs).cpu().numpy()

        _, C = run(Z, goal_on)
        Wt, Ct = run(Zt, goal_on_t)
        unmet_t = ~goal_on_t
        info["detector_on_share"] = round(float(Ct[unmet_t].mean()), 4)
        info["detector_meaning"] = meaning(np.where(Ct, np.maximum(Wt, 0.1001), Wt), unmet_t)
        # Behaviour check with the simulator (criterion 2).
        enc = Encoder(net, device)
        info["behaviour"] = behaviour(levels, k - 1, enc, test_pairs, Ct, unmet_t)
        result["levels"].append(info)
        save()
        log(f"level {k}: " + json.dumps(info))
        goal_on, goal_on_t = C, Ct

    # Acting (criterion 3) on new layouts.
    rng = np.random.default_rng(777)
    from .envs.keydoor import make_layout
    layouts = [make_layout(8, rng) for _ in range(args.test_layouts)]
    enc = Encoder(net, device)
    result["acting"] = {
        "discovered_conditions": act(levels, enc, layouts, top_only=False),
        "top_goal_only": act(levels, enc, layouts, top_only=True),
        "random_play_chance_200_steps": round(float(np.mean(
            [exact_chance(l, "on_goal", 200)[1][exact_chance(l, "on_goal", 200)[0][start_state(l)]]
             for l in layouts[:100]])), 4)}
    result["seconds"] = round(time.monotonic() - started, 1)
    save()
    log("acting " + json.dumps(result["acting"]))


if __name__ == "__main__":
    main()
