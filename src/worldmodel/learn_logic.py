"""Card 010 part B: learn each world's achievement probabilities from pixels,
then read the rules back from the network's predictions.

Per world: card 005's encoder with one achievement head (goal x action),
trained on random play with supplied success signals (C1), card 008's
speed settings. On new layouts, the evidence finder (part A) runs twice on
the same test attempts: with the true outcomes and with the network's
predictions (A > 0.5). The two rule sets must match.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs.keydoor_render import encode_logic_states, encode_states, render, tile_images
from .logic_conditions import Data, World, describe, evidence_rules, expected, rule_sets, step_patterns


def frames_of(name: str, ep: dict) -> np.ndarray:
    if name == "nodrop":
        return encode_states(ep["layout"], np.array(ep["states"]))
    return encode_logic_states(ep["layout"], ep["states"])


def build(world: World, data: list[dict]) -> dict:
    goals = list(world.goals)
    codes, hold, t_idx, t_act = [], [], [], []
    offset = 0
    for ep in data:
        lay, S, A = ep["layout"], ep["states"], ep["actions"]
        codes.append(frames_of(world.name, ep))
        hold.append(np.array([[world.goals[g](lay, s) for g in goals] for s in S], bool))
        t_idx.append(offset + np.arange(len(A)))
        t_act.append(np.array(A, np.int64))
        offset += len(S)
    d = {"codes": np.concatenate(codes), "hold": np.concatenate(hold),
         "t_idx": np.concatenate(t_idx), "t_act": np.concatenate(t_act)}
    i = d["t_idx"]
    d["achieved"] = d["hold"][i + 1] & ~d["hold"][i]
    return d


class Net(nn.Module):
    def __init__(self, n_goals: int, n_actions: int, width: int = 256):
        super().__init__()
        self.n_goals, self.n_actions = n_goals, n_actions
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 9 * 8, width), nn.ReLU())
        self.achieve = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, n_goals * n_actions))

    def forward(self, x):
        return self.achieve(self.enc(x)).view(-1, self.n_goals, self.n_actions)


class Frames:
    def __init__(self, codes, device):
        self.codes = torch.from_numpy(codes).to(device)
        self.tiles = torch.from_numpy(tile_images()).to(device)

    def __call__(self, idx):
        return render(self.codes[idx].long(), self.tiles).permute(0, 3, 1, 2).float() / 255.0


def train(d, n_goals, n_actions, device, updates, batch=512, log=print, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    frames = Frames(d["codes"], device)
    net = Net(n_goals, n_actions).to(device)
    fnet = torch.compile(net)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, fused=True)
    t_idx = torch.from_numpy(d["t_idx"]).to(device)
    t_act = torch.from_numpy(d["t_act"]).to(device)
    achieved = torch.from_numpy(d["achieved"]).to(device).float()
    hold = torch.from_numpy(d["hold"]).to(device)
    events = torch.from_numpy(np.flatnonzero(d["achieved"].any(1))).to(device)
    m = len(t_idx)
    n_u, n_e = batch * 3 // 4, batch // 4
    w_event = 1.0 / (0.75 + 0.25 * m / len(events))
    started = time.monotonic()
    for u in range(updates):
        r = torch.cat([torch.randint(m, (n_u,), device=device, generator=gen),
                       events[torch.randint(len(events), (n_e,), device=device, generator=gen)]])
        i = t_idx[r]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fnet(frames(i))
        la = logits.float().gather(2, t_act[r][:, None, None].expand(-1, n_goals, 1)).squeeze(2)
        mask = (~hold[i]).float()
        w = torch.where(achieved[r].bool().any(1), torch.tensor(w_event, device=device),
                        torch.tensor(1 / 0.75, device=device))[:, None]
        loss = (F.binary_cross_entropy_with_logits(la, achieved[r], reduction="none") * mask * w).sum() / (mask * w).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % 5000 == 0 or u == updates - 1:
            log(f"  update {u} loss {loss.item():.5f} {(u + 1) / (time.monotonic() - started):.0f} upd/s")
    return net


@torch.no_grad()
def predict(net, d, device, chunk=16384):
    frames = Frames(d["codes"], device)
    out = []
    for s in range(0, len(d["t_idx"]), chunk):
        i = torch.from_numpy(d["t_idx"][s:s + chunk]).to(device)
        a = torch.from_numpy(d["t_act"][s:s + chunk]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(net(frames(i)).float())
        out.append(p.gather(2, a[:, None, None].expand(-1, p.shape[1], 1)).squeeze(2).cpu())
    return torch.cat(out).numpy()                  # (transitions, goals) at the taken action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", nargs="+", default=["key", "switch", "either", "both", "nodrop"])
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--updates", type=int, default=60000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    logf = open(args.out / "log.txt", "a")
    result_path = args.out / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    for name in args.worlds:
        started = time.monotonic()
        world = World(name)
        goals = list(world.goals)
        train_data = world.collect(args.episodes, 11)
        test_data = world.collect(args.test_episodes, 99)
        d, dt = build(world, train_data), build(world, test_data)
        log(f"{name}: {len(d['t_idx'])} training steps, successes per goal "
            f"{dict(zip(goals, d['achieved'].sum(0).tolist()))}")
        net = train(d, len(goals), len(world.actions), device, args.updates, log=log)
        torch.save(net.state_dict(), args.out / f"net_{name}.pt")
        P = predict(net, dt, device)
        # The same test attempts, labelled by truth and by the network.
        pats = []
        for ep in test_data:
            pats += step_patterns(world, ep)
        res = {}
        for g, goal in enumerate(goals):
            m = ~dt["hold"][dt["t_idx"], g]
            truth, learned = Counter(), Counter()
            for j in np.flatnonzero(m):
                truth[(pats[j], bool(dt["achieved"][j, g]))] += 1
                learned[(pats[j], bool(P[j, g] > 0.5))] += 1
            dT, dL = Data(truth), Data(learned)
            rT, rL = describe(dT, evidence_rules(dT)[0]), describe(dL, evidence_rules(dL)[0])
            y, p = dt["achieved"][m, g], P[m, g]
            res[goal] = {
                "rules_from_truth": sorted(map(sorted, rule_sets(rT))),
                "rules_from_network": sorted(map(sorted, rule_sets(rL))),
                "match": rule_sets(rT) == rule_sets(rL),
                "truth_matches_expected": rule_sets(rT) == expected(name, goal),
                "successes_in_test": int(y.sum()),
                "network_right_on_successes": round(float((p[y] > 0.5).mean()), 4) if y.any() else None,
                "network_right_on_failures": round(float((p[~y] <= 0.5).mean()), 5),
                "network_rules_detail": rL}
            log(f"  {goal}: match={res[goal]['match']} truth={res[goal]['rules_from_truth']} "
                f"network={res[goal]['rules_from_network']} "
                f"right on successes {res[goal]['network_right_on_successes']}, failures {res[goal]['network_right_on_failures']}")
        res["seconds"] = round(time.monotonic() - started, 1)
        result[name] = res
        result_path.write_text(json.dumps(result, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
