"""Evaluator-labelled component test: same local door view, opposite goal effect.

This tests readiness input sufficiency, not learned-from-experience walking.
It changes only readiness's spatial radius (1 versus 6), keeps the same
shared pixel encoder/context, and evaluates new layouts with wall column 4.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bench_shared import Ego, R, UP, bv
from audit_inputs import signatures
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld
from worldmodel.envs.keydoor_render import tile_images
from worldmodel.spatial_state import SharedSpatialLearner


def pairs(n_layouts, seed, wall_column, device):
    rng = np.random.default_rng(seed)
    ego = Ego(torch, device)
    codes, labels, layouts = [], [], []
    attempts = 0
    while len(layouts) < n_layouts:
        attempts += 1
        if attempts > n_layouts * 500:
            raise RuntimeError("not enough matched local-view pairs")
        layout = ld.make_layout(8, "key", rng)
        if layout.wall_x != wall_column:
            continue
        base = ld.start_state(layout, carry=1)
        states = [(layout.wall_x - 1, layout.door_y, 0, *base[3:]),
                  (layout.wall_x + 1, layout.door_y, 2, *base[3:])]
        ex = dl.Exact(layout, [-1, 0], [-1, ld.FORWARD])
        truth = [ex.ready(1, ld.TOGGLE, s) for s in states]
        if truth != [True, False]:
            continue
        topdown = torch.from_numpy(dl.encode_pairs([(layout, s) for s in states])).to(device)
        views = ego.codes(topdown, np.array([s[:3] for s in states])).cpu().numpy()
        valid = np.zeros((2, 4, 13, 13), bool)
        valid[:, UP, R, R] = True
        sig, _ = signatures(views, valid)
        if not np.array_equal(sig[0], sig[1]):
            continue
        codes.extend(views)
        labels.extend(truth)
        layouts.append({"wall_x": layout.wall_x, "door_y": layout.door_y, "colour": layout.door_colour})
    return torch.from_numpy(np.stack(codes)).to(device), torch.tensor(labels, device=device), layouts


def goal_pairs(n_layouts, seed, wall_columns, device):
    """Counterfactual evaluator pairs: change only the remote goal position."""
    rng = np.random.default_rng(seed)
    ego = Ego(torch, device)
    codes, labels, layouts = [], [], []
    attempts = 0
    while len(layouts) < n_layouts:
        attempts += 1
        if attempts > n_layouts * 500:
            raise RuntimeError("not enough remote-goal pairs")
        layout = ld.make_layout(8, "key", rng)
        if layout.wall_x not in wall_columns:
            continue
        base = ld.start_state(layout, carry=1)
        states = [(layout.wall_x - 1, layout.door_y, 0, *base[3:]),
                  (layout.wall_x + 1, layout.door_y, 2, *base[3:])]
        # Goal changes stay outside either local query. No objects are moved.
        free = [(x, y) for x in range(1, 7) for y in range(1, 7)
                if ld.cell(layout, base, (x, y)) == "empty"
                and all(max(abs(x-s[0]), abs(y-s[1])) > 1 for s in states)]
        left = [p for p in free if p[0] < layout.wall_x]
        right = [p for p in free if p[0] > layout.wall_x]
        if not left or not right:
            continue
        goals = [left[int(rng.integers(len(left)))], right[int(rng.integers(len(right)))]]
        alternatives = [replace(layout, goal=goal) for goal in goals]
        group, truth = [], []
        for state in states:
            for alternative in alternatives:
                ex = dl.Exact(alternative, [-1, 0], [-1, ld.FORWARD])
                truth.append(ex.ready(1, ld.TOGGLE, state))
                group.append((alternative, state))
        if truth != [False, True, True, False]:
            continue
        topdown = torch.from_numpy(dl.encode_pairs(group)).to(device)
        views = ego.codes(topdown, np.array([s[:3] for _, s in group])).cpu().numpy()
        valid = np.zeros((4, 4, 13, 13), bool)
        valid[:, UP, R, R] = True
        sig, _ = signatures(views, valid)
        assert np.array_equal(sig[0], sig[1]) and np.array_equal(sig[2], sig[3])
        codes.extend(views)
        labels.extend(truth)
        layouts.append({"wall_x": layout.wall_x, "door_y": layout.door_y,
                        "colour": layout.door_colour, "goals": goals})
    return torch.from_numpy(np.stack(codes)).to(device), torch.tensor(labels, device=device), layouts


def metrics(model, codes, labels, tiles):
    outputs, condition = [], []
    with torch.no_grad():
        for i in range(0, len(codes), 32):
            z = model.encode(bv.images(codes[i:i+32], tiles))
            outputs.append(model.ready_logits(z)[:, 0, UP, R, R])
            condition.append(model.condition_logits(z)[:, 1])
    logits, holds = torch.cat(outputs), torch.cat(condition)
    return {"frames": len(labels), "positive": int(labels.sum()), "negative": int((~labels).sum()),
            "ready_accuracy": float(((logits > 0) == labels).float().mean()),
            "ready_recall": float((logits[labels] > 0).float().mean()),
            "ready_false_positive": float((logits[~labels] > 0).float().mean()),
            "condition_accuracy": float(((holds > 0) == ~labels).float().mean()),
            "max_within_pair_logit_difference": float((logits[::2] - logits[1::2]).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--layouts", type=int, default=128)
    parser.add_argument("--seconds", type=int, default=210)
    parser.add_argument("--out", type=Path, default=Path("runs/017_context_gate"))
    parser.add_argument("--counterfactual-goals", action="store_true")
    parser.add_argument("--recurrent-comparison", action="store_true")
    parser.add_argument("--condition-comparison", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 540:
        parser.error("screen must be at most 540 seconds")
    torch.set_num_threads(4)
    torch.manual_seed(17)
    device = torch.device("cuda")
    start = time.monotonic()
    log = lambda m: print(f"[{time.monotonic()-start:6.1f}s] {m}", flush=True)
    tiles = torch.from_numpy(tile_images()).to(device)
    if args.counterfactual_goals:
        tr, yt, tr_layouts = goal_pairs(args.layouts, 417, (3, 4), device)
        te, ye, te_layouts = goal_pairs(args.layouts, 917, (5,), device)
    else:
        tr, yt, tr_layouts = pairs(args.layouts, 417, 3, device)
        te, ye, te_layouts = pairs(args.layouts, 917, 4, device)
    log(f"matched pairs: {len(tr)} training / {len(te)} held-out frames")
    ck = torch.load("runs/012run5_key/nets.pt", map_location=device, weights_only=False)["model"]
    enc = {k.removeprefix("enc."): v for k, v in ck.items() if k.startswith(("enc.0.", "enc.2.", "enc.4."))}
    result = {"kind": "oracle_readiness_context_gate", "arch_version": 2,
              "counterfactual_goals": args.counterfactual_goals,
              "training_layouts": tr_layouts, "test_layouts": te_layouts,
              "exact_local_pair_equivalence": True,
              "baselines": {"always_false_accuracy": .5, "always_true_accuracy": .5,
                            "any_local_only_readiness_accuracy_ceiling": .5}, "arms": {}}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.condition_comparison:
        arms = [("flat", 6, "recurrent", "flat"), ("spatial", 6, "recurrent", "spatial")]
    elif args.recurrent_comparison:
        arms = [("wide", 6, "wide", "flat"), ("recurrent", 6, "recurrent", "flat")]
    else:
        arms = [("1", 1, "wide", "flat"), ("6", 6, "wide", "flat")]
    for name, radius, mode, condition_mode in arms:
        torch.manual_seed(17)
        net = SharedSpatialLearner(2, 1, readiness_radius=radius, readiness_mode=mode,
                                   condition_mode=condition_mode).to(device)
        net.patch_encoder.load_state_dict(enc)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        # Identical sampled rows and objectives in both arms. Only the spatial
        # extent of readiness changes. Conditions have full map access in both.
        gen = torch.Generator(device=device).manual_seed(17)
        completed = 0
        for u in range(args.updates):
            if time.monotonic() - start > args.seconds - 10:
                break
            # Keep opposite-label matched pairs together, preventing imbalance.
            p = torch.randint(len(tr) // 2, (8,), device=device, generator=gen)
            rows = (2*p[:, None] + torch.arange(2, device=device)).flatten()
            z = net.encode(bv.images(tr[rows], tiles))
            ready = net.ready_logits(z)[:, 0, UP, R, R]
            condition = net.condition_logits(z)[:, 1]
            loss_r = F.binary_cross_entropy_with_logits(ready, yt[rows].float())
            loss_c = F.binary_cross_entropy_with_logits(condition, (~yt[rows]).float())
            opt.zero_grad(set_to_none=True)
            (loss_r + loss_c).backward()
            opt.step()
            completed += 1
            if u % 250 == 0 or u == args.updates - 1:
                log(f"{name}, radius {radius}, {u}: ready={loss_r.item():.4f}, conditions={loss_c.item():.4f}")
        result["arms"][name] = {"updates": completed, "train": metrics(net, tr, yt, tiles),
                                       "test": metrics(net, te, ye, tiles)}
        arm_version = 3 if mode == "recurrent" and condition_mode == "spatial" else 2
        torch.save({"model": net.state_dict(), "arch_version": arm_version, "readiness_radius": radius,
                    "readiness_mode": mode, "condition_mode": condition_mode,
                    "oracle_trained": True, "n_conditions": 2, "n_ways": 1},
                   args.out/(f"{name}.pt" if args.recurrent_comparison or args.condition_comparison else f"radius{radius}.pt"))
        log(json.dumps(result["arms"][name]))
    selected_name = "spatial" if args.condition_comparison else "recurrent" if args.recurrent_comparison else "6"
    selected = result["arms"][selected_name]["test"]
    result["passed"] = selected["ready_accuracy"] >= .99 and selected["condition_accuracy"] >= .99
    if not args.recurrent_comparison and not args.condition_comparison:
        result["passed"] = result["passed"] and result["arms"]["1"]["test"]["ready_accuracy"] <= .5
    result["seconds"] = time.monotonic() - start
    (args.out/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    log(f"context component gate passed: {result['passed']}")


if __name__ == "__main__":
    main()
