"""Card 005: learn achievement probabilities and walk values from frames.

A(frame, action, goal): will this action make the goal true now (logistic
loss on random play, event steps over-sampled with importance weights).
W(frame, target, move): Q-learning on movement steps only; W = 0.95^steps
to reach the target by turns and forward steps, 0 if it cannot be reached.
Goals and targets are supplied tasks with a success signal (C1, until P20).
The evaluation compares both with exact values on new layouts.
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

from .conditions import achievement_rows, collect, find_rules, holds
from .envs.keydoor import FORWARD, LEFT, PICKUP, RIGHT, TOGGLE, Layout, step, variables
from .envs.keydoor_render import encode_states, render, tile_images
from .reach import at_target, target_exists, walk_components

GOALS = ("door_unlocked", "door_open", "holding_matching_key", "on_goal")
TARGETS = ("goal", "door", "key", "distractor")
GAMMA = 0.95
INF = 10 ** 6


# ---------------------------------------------------------------- data

def build(data: list[dict]) -> dict:
    codes, hold, at, term, t_idx, t_act, owner = [], [], [], [], [], [], []
    offset = 0
    for e, ep in enumerate(data):
        lay, S, A = ep["layout"], ep["states"], ep["actions"]
        codes.append(encode_states(lay, np.array(S)))
        hold.append(np.array([[holds(g, lay, s) for g in GOALS] for s in S], bool))
        at.append(np.array([[at_target(lay, s, t) for t in TARGETS] for s in S], bool))
        term.append(np.array([(s[0], s[1]) == lay.goal for s in S], bool))
        t_idx.append(offset + np.arange(len(A)))
        t_act.append(A)
        owner.append(np.full(len(S), e))
        offset += len(S)
    d = {"codes": np.concatenate(codes), "hold": np.concatenate(hold),
         "at": np.concatenate(at), "term": np.concatenate(term),
         "t_idx": np.concatenate(t_idx), "t_act": np.concatenate(t_act).astype(np.int64),
         "owner": np.concatenate(owner)}
    i = d["t_idx"]
    d["achieved"] = d["hold"][i + 1] & ~d["hold"][i]
    d["event"] = d["achieved"].any(1)
    return d


def unlock_subset(data: list[dict], n: int) -> list[dict]:
    """All episodes without an unlock, plus the first n with one (P19 curve)."""
    keep, got = [], 0
    for ep in data:
        unlocked = any(s[4] != 0 for s in ep["states"])
        if not unlocked:
            keep.append(ep)
        elif got < n:
            keep.append(ep)
            got += 1
    return keep


# ---------------------------------------------------------------- model

class Net(nn.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 9 * 8, width), nn.ReLU())
        self.achieve = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, len(GOALS) * 5))
        self.walk = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, len(TARGETS) * 3))

    def forward(self, frames):
        z = self.enc(frames)
        return (self.achieve(z).view(-1, len(GOALS), 5),
                torch.sigmoid(self.walk(z).view(-1, len(TARGETS), 3)))


class Frames:
    def __init__(self, codes: np.ndarray, device):
        self.codes = torch.from_numpy(codes).to(device)          # uint8; indexed per batch
        self.tiles = torch.from_numpy(tile_images()).to(device)

    def __call__(self, idx):
        img = render(self.codes[idx].long(), self.tiles)
        return img.permute(0, 3, 1, 2).float() / 255.0


# ---------------------------------------------------------------- training

def train(d: dict, updates: int, device, batch: int = 512, lr: float = 3e-4, seed: int = 0, log=print):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    frames = Frames(d["codes"], device)
    net = Net().to(device)
    target = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t_idx = torch.from_numpy(d["t_idx"]).to(device)
    t_act = torch.from_numpy(d["t_act"]).to(device)
    achieved = torch.from_numpy(d["achieved"]).to(device).float()
    hold = torch.from_numpy(d["hold"]).to(device)
    at = torch.from_numpy(d["at"]).to(device)
    term = torch.from_numpy(d["term"]).to(device)
    events = torch.from_numpy(np.flatnonzero(d["event"])).to(device)
    moves = torch.from_numpy(np.flatnonzero(d["t_act"] <= FORWARD)).to(device)
    m = len(t_idx)
    n_u, n_e = batch * 3 // 4, batch // 4
    w_event = 1.0 / (0.75 + 0.25 * m / len(events))       # importance weight of an event row
    start = time.monotonic()
    for u in range(updates):
        ra = torch.cat([torch.randint(m, (n_u,), device=device, generator=gen),
                        events[torch.randint(len(events), (n_e,), device=device, generator=gen)]])
        rw = moves[torch.randint(len(moves), (batch,), device=device, generator=gen)]
        ia, iw = t_idx[ra], t_idx[rw]
        logits, _ = net(frames(ia))
        _, q = net(frames(iw))
        # A: logistic loss at the taken action, for goals that did not already hold.
        la = logits.gather(2, t_act[ra][:, None, None].expand(-1, len(GOALS), 1)).squeeze(2)
        mask = ~hold[ia]
        weight = torch.where(achieved[ra].bool().any(1), torch.tensor(w_event, device=device),
                             torch.tensor(1.0 / 0.75, device=device))
        bce = F.binary_cross_entropy_with_logits(la, achieved[ra], reduction="none")
        loss_a = (bce * mask * weight[:, None]).sum() / (mask * weight[:, None]).sum()
        # W: Q-learning on movement steps.
        qa = q.gather(2, t_act[rw][:, None, None].expand(-1, len(TARGETS), 1)).squeeze(2)
        with torch.no_grad():
            _, qn = target(frames(iw + 1))
            y = torch.where(at[iw + 1], 1.0, torch.where(term[iw + 1][:, None], 0.0, GAMMA * qn.max(2).values))
        mw = ~at[iw]
        loss_w = ((qa - y) ** 2 * mw).sum() / mw.sum()
        loss = loss_a + loss_w
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target.parameters()):
                pt.lerp_(p, 0.005)
        if u % 2000 == 0 or u == updates - 1:
            log(f"update {u} loss_a {loss_a.item():.4f} loss_w {loss_w.item():.4f} "
                f"{(u + 1) / (time.monotonic() - start):.0f} upd/s")
    return net


# ---------------------------------------------------------------- evaluation

@lru_cache(maxsize=20000)
def walk_distances(layout: Layout, carry: int, door: int, target: str) -> dict:
    """(x, y, dir) -> fewest turns and forward steps to the target (backward search)."""
    comp = walk_components(layout, carry, door)
    rev = {}
    for u in comp:
        if (u[0], u[1]) == layout.goal:
            continue
        for a in (LEFT, RIGHT, FORWARD):
            v = step(layout, (*u, carry, door), a)[0][:3]
            if v != u:
                rev.setdefault(v, []).append(u)
    dist = {u: 0 for u in comp if at_target(layout, (*u, carry, door), target)}
    todo = deque(dist)
    while todo:
        v = todo.popleft()
        for u in rev.get(v, []):
            if u not in dist:
                dist[u] = dist[v] + 1
                todo.append(u)
    return dist


def auc(score, label):
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1.0
    npos = label.sum()
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * (~label).sum()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def predict(net, d, device, chunk=8192):
    frames = Frames(d["codes"], device)
    n = len(d["codes"])
    A, W = [], []
    for s in range(0, n, chunk):
        idx = torch.arange(s, min(n, s + chunk), device=device)
        logits, q = net(frames(idx))
        A.append(torch.sigmoid(logits).cpu())
        W.append(q.max(2).values.cpu())
    return torch.cat(A).numpy(), torch.cat(W).numpy()    # (N, goals, 5), (N, targets)


CELLS = [  # name, goal, action, condition on variables, truth
    ("toggle, facing locked door, matching key", "door_unlocked", TOGGLE,
     lambda v: v["front"] == "door" and v["door"] == "locked" and v["carrying_matches_door"], 1),
    ("toggle, facing locked door, other key", "door_unlocked", TOGGLE,
     lambda v: v["front"] == "door" and v["door"] == "locked" and v["carrying"] == "key" and not v["carrying_matches_door"], 0),
    ("toggle, facing locked door, nothing", "door_unlocked", TOGGLE,
     lambda v: v["front"] == "door" and v["door"] == "locked" and v["carrying"] == "none", 0),
    ("pickup, facing matching key, empty hands", "holding_matching_key", PICKUP,
     lambda v: v["front_matches_door"] and v["carrying"] == "none", 1),
    ("pickup, facing other key, empty hands", "holding_matching_key", PICKUP,
     lambda v: v["front"] == "key" and not v["front_matches_door"] and v["carrying"] == "none", 0),
    ("forward, facing goal square", "on_goal", FORWARD, lambda v: v["front"] == "goal", 1),
]


def evaluate(net, test_data, d, device) -> dict:
    A, W = predict(net, d, device)
    states = [s for ep in test_data for s in ep["states"]]
    layouts = [ep["layout"] for ep in test_data for _ in ep["states"]]
    vars_ = [variables(l, s) for l, s in zip(layouts, states)]
    out = {"achievement": {}, "walk": {}}
    # 1. contrast cells
    i_all, a_all = d["t_idx"], d["t_act"]
    for name, goal, action, cond, truth in CELLS:
        g = GOALS.index(goal)
        rows = [i for i, a in zip(i_all, a_all) if a == action and not d["hold"][i, g] and cond(vars_[i])]
        p = A[rows, g, action]
        true = d["hold"][np.array(rows) + 1, g]
        out["achievement"][name] = {
            "n": len(rows), "truth": truth, "truth_checked": bool((true == truth).all()),
            "mean_p": round(float(p.mean()), 4) if rows else None,
            "right_side_of_0.5": round(float(((p > 0.5) == bool(truth)).mean()), 4) if rows else None}
    # overall area under the curve per goal at the taken action
    for g, goal in enumerate(GOALS):
        m = ~d["hold"][i_all, g]
        out["achievement"][f"auc_{goal}"] = round(auc(A[i_all[m], g, a_all[m]], d["achieved"][m, g]), 4)
    # 2. walk values
    exact = np.full((len(states), len(TARGETS)), -1)
    for k, (l, s) in enumerate(zip(layouts, states)):
        for t, target in enumerate(TARGETS):
            # Not scored: already at the target, target gone, or the episode
            # has ended (the agent stands on the goal square; no walking follows).
            if at_target(l, s, target) or not target_exists(s, target) or (s[0], s[1]) == l.goal:
                continue
            exact[k, t] = walk_distances(l, s[3], s[4], target).get(s[:3], INF)
    for t, target in enumerate(TARGETS):
        m = exact[:, t] >= 0
        reach = exact[m, t] < INF
        r = {"n": int(m.sum()), "reachable_share": round(float(reach.mean()), 4),
             "auc": round(auc(W[m, t], reach), 4)}
        if reach.any():
            r["rank_corr_with_steps"] = round(spearman(W[m, t][reach], -exact[m, t][reach]), 4)
            r["mean_abs_error_value"] = round(float(np.abs(W[m, t][reach] - GAMMA ** exact[m, t][reach]).mean()), 4)
        out["walk"][target] = r
    g = TARGETS.index("goal")
    left = np.array([v["side"] == "left" for v in vars_]) & (exact[:, g] >= 0)
    door_open = np.array([v["door"] == "open" for v in vars_])
    pred = W[:, g] > 0.1
    out["walk"]["goal_from_left_room"] = {
        "n": int(left.sum()),
        "correct_vs_reachable": round(float((pred[left] == (exact[left, g] < INF)).mean()), 4),
        "correct_vs_door_open": round(float((pred[left] == door_open[left]).mean()), 4),
        "mean_W_door_open": round(float(W[left & door_open, g].mean()), 4),
        "mean_W_door_closed": round(float(W[left & ~door_open, g].mean()), 4)}
    # 3. conditions from learned predictions versus true outcomes
    out["conditions"] = {}
    true_rows = {goal: achievement_rows(test_data, goal) for goal in ("door_unlocked", "holding_matching_key")}
    for goal, rows in true_rows.items():
        gi = GOALS.index(goal)
        m = ~d["hold"][i_all, gi]
        p = A[i_all[m], gi, a_all[m]]
        learned = [(v, a, bool(q > 0.5)) for (v, a, _), q in zip(rows, p)]
        out["conditions"][goal] = {"from_truth": find_rules(rows)[:1], "from_learned": find_rules(learned)[:1]}
    walk_true, walk_learned = [], []
    for k, v in enumerate(vars_[::2]):
        k *= 2
        if exact[k, g] < 0:
            continue
        v = {a: b for a, b in v.items() if a != "side"}
        walk_true.append((v, 0, bool(exact[k, g] < INF)))
        walk_learned.append((v, 0, bool(W[k, g] > 0.1)))
    names = ("walk to goal",)
    out["conditions"]["walk_to_goal"] = {"from_truth": find_rules(walk_true, action_names=names)[:1],
                                         "from_learned": find_rules(walk_learned, action_names=names)[:1]}
    # trivial baselines
    out["baseline"] = {
        "achievement_base_rate": {goal: round(float(d["achieved"][d["t_act"] == a, g].mean()), 5)
                                  for g, goal in enumerate(GOALS)
                                  for a in [{"door_unlocked": TOGGLE, "door_open": TOGGLE,
                                             "holding_matching_key": PICKUP, "on_goal": FORWARD}[goal]]},
        "walk_reachable_share": {t: out["walk"][t]["reachable_share"] for t in TARGETS}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--unlock-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rescore", action="store_true", help="evaluate the saved net in --out again")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    logf = open(args.out / ("rescore_log.txt" if args.rescore else "log.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    started = time.monotonic()
    if args.rescore:
        test_data = collect(8, args.test_episodes, 640, 99)
        net = Net().to(device)
        net.load_state_dict(torch.load(args.out / "net.pt"))
        res = evaluate(net, test_data, build(test_data), device)
        (args.out / "result_rescored.json").write_text(json.dumps(res, indent=1) + "\n")
        log(json.dumps(res["walk"]))
        log(json.dumps(res["conditions"]))
        return
    data = collect(8, args.episodes, 640, 3)
    if args.unlock_episodes is not None:
        data = unlock_subset(data, args.unlock_episodes)
    d = build(data)
    log(f"train: {len(data)} episodes, {len(d['t_idx'])} steps, {int(d['achieved'][:, 0].sum())} unlocks")
    test_data = collect(8, args.test_episodes, 640, 99)
    dt = build(test_data)
    net = train(d, args.updates, device, seed=args.seed, log=log)
    torch.save(net.state_dict(), args.out / "net.pt")
    res = evaluate(net, test_data, dt, device)
    res["train"] = {"episodes": len(data), "unlocks": int(d["achieved"][:, 0].sum()), "updates": args.updates,
                    "seconds": round(time.monotonic() - started, 1)}
    (args.out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    log(json.dumps(res["achievement"], indent=1))
    log(json.dumps(res["walk"], indent=1))
    log(json.dumps(res["conditions"]))


if __name__ == "__main__":
    main()
