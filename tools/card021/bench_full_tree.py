"""Bounded full-tree gate. Constructed states belong only to the evaluator/upper bound."""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card017"))
import bench_shared as bs
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld


INACTIVE = "forward/toggle/toggle/toggle"


def configurations(layout, rng):
    """Legal evaluator configurations, including moved objects and terminal poses.

    These are constructed states, not claimed to be trajectories or experience.
    At most one key is carried; ground objects never overlap fixed objects.
    """
    floor = [(x, y) for x in range(1, 7) for y in range(1, 7)
             if x != layout.wall_x and (x, y) not in (layout.switch, layout.vase, layout.goal)]
    rows = []
    for carry in range(3):
        for door in range(2):
            for vase in range(2):
                if rng.random() < .5:
                    key, distractor = layout.key, layout.distractor
                else:
                    a, b = rng.choice(len(floor), 2, replace=False)
                    key, distractor = floor[a], floor[b]
                rest = (carry, None if carry == 1 else key, None if carry == 2 else distractor,
                        door, int(rng.integers(2)), vase)
                poses = [(x, y, d) for x, y in floor
                         if ld.cell(layout, (0, 0, 0, *rest), (x, y)) == "empty" for d in range(4)]
                rows.append((*poses[int(rng.integers(len(poses)))], *rest))
    # Audit every distractor placement while holding the matching key. A vase
    # and the other object can trap the agent in a corner: breaking the vase
    # and opening the door then require two toggles. Uniform configurations
    # under-cover this genuine branch, especially with a wide left room.
    for distractor in floor:
        rest = (1, None, distractor, 0, int(rng.integers(2)), 0)
        poses = [(x, y, d) for x, y in floor if (x, y) != distractor for d in range(4)]
        rows.append((*poses[int(rng.integers(len(poses)))], *rest))
    for direction in range(4):
        rows.append((*layout.goal, direction, *rows[direction][3:]))
    return rows


def new_source(data, layouts, eps, states):
    states = np.asarray(states, dtype=np.int8)
    topdown = torch.from_numpy(dl.encode_pairs([(layouts[e], dl.to_tup(s))
                                               for e, s in zip(eps, states)])).to(data.device)
    return {"layouts": layouts, "ep": np.asarray(eps), "states": states,
            "codes": data.ego.codes(topdown, states),
            "wall": np.asarray([layouts[e].wall_x for e in eps])}


def teacher_labels(data, source):
    """Report teacher errors on the evaluator set; never supply its hidden state."""
    ck = torch.load("runs/012run5_key/nets.pt", map_location=data.device, weights_only=False)
    model = dl.make_model().to(data.device)
    model.load_state_dict(ck["model"])
    model.eval()
    wi = {int(k): v for k, v in ck["way_index"].items()}
    holds, ready = [], []
    with torch.no_grad():
        for start in range(0, len(source["states"]), 1024):
            pairs = [(source["layouts"][e], dl.to_tup(s)) for e, s in
                     zip(source["ep"][start:start+1024], source["states"][start:start+1024])]
            codes = torch.from_numpy(dl.encode_pairs(pairs)).to(data.device)
            z = model.enc(bs.bv.images(codes, data.tiles))
            p = model.ach(z).reshape(-1, dl.K_MAX, len(dl.ACTIONS)).sigmoid()
            reach = model.val(z).reshape(-1, dl.VAL_SLOTS, 2, 3).sigmoid()[:, :, 1].max(-1).values
            on = torch.zeros(len(pairs), len(data.tree.parent), dtype=torch.bool, device=data.device)
            on[:, 0] = torch.tensor([(s[0], s[1]) == lay.goal for lay, s in pairs], device=data.device)
            rr = {}
            for n in range(1, len(data.tree.parent)):
                if n not in wi:
                    continue
                parent, action = data.tree.parent[n], data.tree.action[n]
                rr[n] = (p[:, parent, action] > .5) & ~on[:, parent]
                on[:, n] = on[:, parent] | rr[n] | (reach[:, wi[n]] > .5)
            holds.append(on)
            ready.append(torch.stack([rr[n] for n in data.ways], 1))
    source["holds"], source["ready"] = torch.cat(holds), torch.cat(ready)


def augment(data, split, layouts_per_split, seed):
    source = data.splits[split]
    rng = np.random.default_rng(seed)
    allowed = [i for i, lay in enumerate(source["layouts"]) if split != "train" or lay.wall_x != 5]
    selected = rng.choice(allowed, min(layouts_per_split, len(allowed)), replace=False)
    eps, states = list(source["ep"]), list(source["states"])
    for e in selected:
        extra = configurations(source["layouts"][e], rng)
        eps.extend([int(e)] * len(extra))
        states.extend(dl.to_arr(s) for s in extra)
    data.splits[split] = new_source(data, source["layouts"], eps, states)
    data.oracle(split)
    enlarged = data.splits[split]
    truth = enlarged["oracle"]["ready"].cpu().numpy()
    # Move the displayed avatar to rare positive query poses, without changing
    # the rest of the configuration. Exact labels are recomputed afterwards.
    added, seen = [], set()
    for j in range(len(data.ways)):
        for wall in (3, 4, 5):
            frame, direction, row, column = np.where(truth[:, j] &
                (enlarged["wall"] == wall)[:, None, None, None])
            branch_seen = set()
            for i in rng.permutation(len(frame)):
                index, ed, ey, ex = (int(a[i]) for a in (frame, direction, row, column))
                arr = enlarged["states"][index].copy()
                x, y, d = (int(v) for v in arr[:3])
                fx, fy = ld.DIR_VEC[d]
                rx, ry = ld.DIR_VEC[(d + 1) % 4]
                arr[:3] = (x + (bs.R-ey)*fx + (ex-bs.R)*rx,
                           y + (bs.R-ey)*fy + (ex-bs.R)*ry, (d+ed-bs.UP) % 4)
                e = int(enlarged["ep"][index])
                signature = (e, *arr.tolist())
                branch_seen.add(signature)
                if signature not in seen:
                    seen.add(signature)
                    added.append((e, arr))
                # Each branch/column contributes at most 64 distinct candidates.
                if len(branch_seen) >= 64:
                    break
    eps.extend(e for e, _ in added)
    states.extend(s for _, s in added)
    data.splits[split] = new_source(data, source["layouts"], eps, states)
    data.oracle(split)
    teacher_labels(data, data.splits[split])
    data.log(f"{split}: {len(selected)} evaluator layouts, {len(added)} re-centred poses, "
             f"{len(states)} total evaluation frames")


def coverage(data):
    result = bs.coverage(data)
    unsupported = []
    for split, source in data.splits.items():
        oracle = source["oracle"]
        root = oracle["holds"][:, 0].cpu().numpy()
        counts = {"positive": int(root.sum()), "negative": int((~root).sum())}
        result["counts"][split]["(goal square)"] = {"condition": counts}
        if min(counts.values()) < 20:
            result["insufficient"].append({"split": split, "way": "(goal square)",
                                           "metric": "condition", **counts})
        j = next(j for j, n in enumerate(data.ways) if "/".join(data.tree.path(n)) == INACTIVE)
        unsupported.append(int(oracle["ready"][:, j].sum()))
    result["inactive_readiness"] = {"way": INACTIVE, "positive_counts": unsupported,
        "certificate": "Single vase removal and single door opening suffice for walking plus toggles in key worlds.",
        "verified_on_audit": not any(unsupported)}
    if not any(unsupported):
        result["insufficient"] = [m for m in result["insufficient"]
            if not (m["way"] == INACTIVE and m["metric"] != "condition"
                    and m["positive"] == 0 and m["negative"] >= 20)]
    result["sufficient"] = not result["insufficient"] and not any(unsupported)
    return result


def evaluate(data, model, split="test"):
    result = bs.evaluate(data, model, split)
    source = data.splits[split]
    scores = []
    with torch.no_grad():
        for start in range(0, len(source["codes"]), 128):
            state = model.encode(bs.bv.images(source["codes"][start:start+128], data.tiles))
            scores.append(model.condition_logits(state).cpu().numpy())
    scores = np.concatenate(scores)
    exact = source["oracle"]["holds"].cpu().numpy()
    result["conditions"]["(goal square)"] = bs.confusion(scores[:, 0], exact[:, 0])
    result["teacher_conditions"] = {
        "/".join(data.tree.path(n)) or "(goal square)": bs.confusion(
            source["holds"][:, n].cpu().numpy().astype(float)*2-1, exact[:, n]) for n in data.nodes}
    return result


def gate_passes(result, upper, training=False):
    recall, fp = (.99, .01) if upper else (.95, .05)
    failures = []
    for name, row in result["readiness"].items():
        for tag in (("agent", "other_poses") if training else ("agent", "other_poses", "unseen_wall")):
            m = row[tag]
            passed = (m["positive"] == 0 and m["negative"] >= 20 and m["false_positive"] <= fp
                      if name == INACTIVE else bs.passes(m, recall, fp))
            if not passed:
                failures.append({"way": name, "metric": tag, **m})
    for name, m in result["conditions"].items():
        if not bs.passes(m, recall, fp):
            failures.append({"way": name, "metric": "condition", **m})
    return not failures, failures


def duration_diagnostic(data, args, log):
    """One uninterrupted optimizer trajectory; only the update budget changes."""
    start = time.monotonic()
    torch.manual_seed(args.seed)
    model = data.model(24)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    source = data.splits["train"]
    result = {"arch_version": 3, "training_card": "021", "oracle_trained": True,
              "kind": "duration_diagnostic", "requested_updates": 10000,
              "seed": args.seed, "batch": args.batch, "learning_rate": 3e-4,
              "data": str(args.out / "data.pt"), "checkpoints": {}}
    completed = 0
    for update in range(10000):
        if time.monotonic() - start >= 420:
            break
        ids = torch.randint(len(source["codes"]), (args.batch,), device=data.device)
        state = model.encode(bs.bv.images(source["codes"][ids], data.tiles))
        ready_loss = bs.balanced_bce(model.ready_logits(state), source["oracle"]["ready"][ids],
                                     source["oracle"]["valid"][ids, None])
        condition_loss = bs.balanced_bce(model.condition_logits(state)[:, data.nodes],
                                         source["oracle"]["holds"][ids][:, data.nodes])
        opt.zero_grad(set_to_none=True)
        (ready_loss + condition_loss).backward()
        opt.step()
        completed = update + 1
        if completed % 1000 == 0:
            log(f"duration {completed}: readiness={ready_loss.item():.4f}, "
                f"conditions={condition_loss.item():.4f}")
        if completed in (2000, 6000, 10000):
            snapshot = {}
            for split in ("train", "test"):
                scores = evaluate(data, model, split)
                passed, failures = gate_passes(scores, True, training=split == "train")
                snapshot[split] = {"passed": passed, "failures": failures, "scores": scores}
                cond = list(scores["conditions"].values())
                log(f"duration {completed} {split}: passed={passed}, failures={len(failures)}, "
                    f"condition recall min={min(m['recall'] for m in cond):.4f}, "
                    f"false positives max={max(m['false_positive'] for m in cond):.4f}")
            snapshot["seconds"] = time.monotonic()-start
            result["checkpoints"][str(completed)] = snapshot
            result["completed_updates"] = completed
            (args.out / "duration_result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            model.train()
    result["completed_updates"] = completed
    result["seconds"] = time.monotonic()-start
    result["passed"] = (completed == 10000 and all(result["checkpoints"]["10000"][s]["passed"]
                                                   for s in ("train", "test")))
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "tree": data.tree.report(), "ways": data.ways,
                "arch_version": 3, "oracle_trained": True, "component_only": True,
                "readiness_mode": model.readiness_mode, "readiness_radius": model.readiness_radius,
                "condition_mode": model.condition_mode, "training_card": "021",
                "completed_updates": completed}, args.out / "upper_duration.pt")
    (args.out / "duration_result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    log(f"duration diagnostic passed: {result['passed']}")


def convert(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: convert(v, device) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "duration"))
    parser.add_argument("--out", type=Path, default=Path("runs/021_full_tree"))
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--layouts", type=int, default=192)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seconds", type=int, default=480)
    args = parser.parse_args()
    if not 0 < args.seconds <= 480:
        parser.error("this gate is capped at 480 seconds, with an external 540-second timeout")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    log = lambda m: print(f"[{time.monotonic()-start:6.1f}s] {m}", flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    result_path = args.out / "result.json"
    def save(result):
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.stage == "prepare":
        with mp.get_context("spawn").Pool(args.workers) as pool:
            data = bs.Data(pool, args, log)
            experience = data.splits.copy()
            for split, seed in (("train", 2117), ("test", 2199)):
                augment(data, split, args.layouts, seed)
            result = {"arch_version": 3, "stage": "coverage", "coverage": coverage(data),
                "baselines": {"always_false": {"recall": 0., "false_positive": 0.},
                              "always_true": {"recall": 1., "false_positive": 1.}},
                "prepare_seconds": time.monotonic()-start,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
            save(result)
            torch.save(convert({"splits": data.splits, "experience": experience,
                                "tree": data.tree, "ways": data.ways, "nodes": data.nodes,
                                "initial_encoder": data.initial_encoder}, "cpu"), args.out / "data.pt")
            log(f"coverage sufficient: {result['coverage']['sufficient']}; "
                f"missing: {result['coverage']['insufficient']}")
        return
    result = json.loads(result_path.read_text())
    if not result["coverage"]["sufficient"]:
        log("coverage failed; fitting and later stages remain stopped")
        return
    data = bs.Data.__new__(bs.Data)
    data.__dict__.update(convert(torch.load(args.out / "data.pt", weights_only=False), args.device))
    data.device, data.log = torch.device(args.device), log
    data.tiles = torch.from_numpy(bs.tile_images()).to(data.device)
    if args.stage == "duration":
        duration_diagnostic(data, args, log)
        return
    evaluation = data.splits
    for arm in ("upper", "learned"):
        torch.manual_seed(args.seed)
        model = data.model(24)
        data.splits = evaluation if arm == "upper" else data.experience
        result[arm+"_fit"] = bs.fit(data, model, args.updates, arm == "upper", args.batch,
                                    log, start+args.seconds-45)
        data.splits = evaluation
        result[arm] = evaluate(data, model)
        result[arm+"_passed"], result[arm+"_failures"] = gate_passes(result[arm], arm == "upper")
        result["stage"] = arm
        result["fit_seconds"] = time.monotonic()-start
        torch.save({"model": model.state_dict(), "tree": data.tree.report(), "ways": data.ways,
                    "arch_version": 3, "oracle_trained": arm == "upper", "component_only": True,
                    "readiness_mode": model.readiness_mode, "readiness_radius": model.readiness_radius,
                    "condition_mode": model.condition_mode, "training_card": "021"}, args.out / (arm+".pt"))
        save(result)
        log(f"{arm} passed: {result[arm+'_passed']}; failures: {len(result[arm+'_failures'])}")
        if not result[arm+"_passed"]:
            result["stopped"] = f"{arm} full-tree gate failed; no later stage started"
            save(result)
            return
        del model
    result["stopped"] = "component gates passed; walking is the next separately gated stage"
    save(result)


if __name__ == "__main__":
    main()
