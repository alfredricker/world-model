"""Card 017 component gates for the shared spatial learner.

Run from the repository root with bin/prun python tools/card017/bench_shared.py.
The oracle is confined to the upper bound and evaluation. The learned arm
uses cached predictions of card 012's teacher; the teacher is then deleted.
Neither arm is an autonomous-play result or a rerun of condition discovery.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card014"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card016"))
import bench_vin as bv
from bench_ego import Ego, H, W, R, UP
from worldmodel import discover_logic as dl
from worldmodel.envs.keydoor_render import tile_images
from worldmodel.spatial_state import SharedSpatialLearner


def oracle_chunk(job):
    layouts, tree, ways, items = job
    shape = (len(items), len(ways), 4, 9, 8)
    distance = np.full(shape, -1, np.int16)
    valid = np.zeros((len(items), 4, 9, 8), bool)
    holds = np.zeros((len(items), len(tree.parent)), bool)
    cache = {}
    for i, (e, arr) in enumerate(items):
        ex = cache.setdefault(e, dl.Exact(layouts[e], tree.parent, tree.action))
        s = dl.to_tup(arr)
        holds[i] = [ex.holds(n, s) for n in range(len(tree.parent))]
        comp, members, _ = ex.component(s[3:])
        for x, y, d in comp:
            valid[i, d, y, x] = True
        for j, n in enumerate(ways):
            for group in members:
                for (x, y, d), steps in ex.distances(n, (*group[0], *s[3:])).items():
                    distance[i, j, d, y, x] = steps
    return distance, valid, holds


def balanced_bce(logits, truth, mask=None):
    """Equal positive and negative weight per output; absent classes are omitted."""
    # Class/output is axis 1. All remaining dimensions are observations.
    loss = F.binary_cross_entropy_with_logits(logits, truth.float(), reduction="none")
    mask = torch.ones_like(truth, dtype=torch.bool) if mask is None else mask.expand_as(truth)
    dims = (0, *range(2, logits.ndim))
    terms = []
    for label in (False, True):
        use = mask & (truth == label)
        count = use.sum(dims)
        terms.append(((loss * use).sum(dims) / count.clamp_min(1), count > 0))
    total = sum((value * present).sum() for value, present in terms)
    return total / sum(present.sum() for _, present in terms).clamp_min(1)


def confusion(score, truth, mask=None):
    use = np.ones_like(truth, bool) if mask is None else np.broadcast_to(mask, truth.shape)
    pred = score > 0
    pos, neg = use & truth, use & ~truth
    return {"positive": int(pos.sum()), "negative": int(neg.sum()),
            "recall": float(pred[pos].mean()) if pos.any() else None,
            "false_positive": float(pred[neg].mean()) if neg.any() else None}


def passes(metrics, recall, false_positive):
    return (metrics["positive"] >= 20 and metrics["negative"] >= 20
            and metrics["recall"] >= recall and metrics["false_positive"] <= false_positive)


def coverage(data):
    """Require evidence for both rates before spending any fitting budget."""
    report, missing = {}, []
    for split_name, source in data.splits.items():
        oracle = {k: v.cpu().numpy() for k, v in source["oracle"].items()}
        centre = np.zeros_like(oracle["valid"])
        centre[:, UP, R, R] = True
        report[split_name] = {}
        for j, node in enumerate(data.ways):
            name = "/".join(data.tree.path(node))
            metrics = {}
            masks = [("agent", centre), ("other_poses", ~centre)]
            if split_name == "test":
                masks.append(("unseen_wall", (source["wall"] == 5)[:, None, None, None]))
            for tag, mask in masks:
                valid = mask & oracle["valid"]
                metrics[tag] = {"positive": int((valid & oracle["ready"][:, j]).sum()),
                                "negative": int((valid & ~oracle["ready"][:, j]).sum())}
            truth = oracle["holds"][:, node]
            metrics["condition"] = {"positive": int(truth.sum()), "negative": int((~truth).sum())}
            report[split_name][name] = metrics
            for tag, counts in metrics.items():
                if min(counts.values()) < 20:
                    missing.append({"split": split_name, "way": name, "metric": tag, **counts})
    return {"counts": report, "sufficient": not missing, "insufficient": missing}


class Data:
    def __init__(self, pool, args, log):
        self.device = torch.device(args.device)
        self.pool, self.log = pool, log
        self.tiles = torch.from_numpy(tile_images()).to(self.device)
        ck = torch.load("runs/012run5_key/nets.pt", map_location=self.device, weights_only=False)
        saved = json.loads(Path("runs/012run5_key/result.json").read_text())
        self.tree = dl.Tree.from_report(saved["learned"]["tree"])
        self.ways = sorted(int(n) for n in ck["way_index"])
        self.nodes = [0] + self.ways
        self.ego = Ego(torch, self.device)
        self.splits = {}
        model = dl.make_model().to(self.device)
        model.load_state_dict(ck["model"])
        model.eval()
        # Only weights initialize the new encoder. No teacher features are
        # injected into the shared state or retained at acting time.
        self.initial_encoder = {k.removeprefix("enc."): v.detach().clone()
                                for k, v in ck["model"].items()
                                if k.startswith(("enc.0.", "enc.2.", "enc.4."))}
        wi = {int(k): v for k, v in ck["way_index"].items()}
        for split, count, seed in (("train", args.episodes, 11), ("test", args.test_episodes, 999)):
            layouts, tr, _, _ = dl.collect(pool, "key", count, seed, 1 / 3, seq_episodes=1)
            # Both endpoints ground held-state labels, including the supplied
            # terminal goal signal. Splits are independent episodes/layouts.
            codes = torch.from_numpy(np.concatenate([tr["c0"], tr["c1"]])).to(self.device)
            states = np.concatenate([tr["s0"], tr["s1"]])
            eps = np.tile(tr["ep"], 2)
            root = np.concatenate([np.zeros(len(tr["act"]), bool), tr["term1"]])
            labels, ready = [], []
            with torch.no_grad():
                for start in range(0, len(codes), 2048):
                    end = start + 2048
                    z = model.enc(bv.images(codes[start:end], self.tiles))
                    p = model.ach(z).reshape(-1, dl.K_MAX, len(dl.ACTIONS)).sigmoid()
                    reach = model.val(z).reshape(-1, dl.VAL_SLOTS, 2, 3).sigmoid()[:, :, 1].max(-1).values
                    on = torch.zeros(len(z), len(self.tree.parent), dtype=torch.bool, device=self.device)
                    on[:, 0] = torch.as_tensor(root[start:end], device=self.device)
                    rr = {}
                    for n in range(1, len(self.tree.parent)):
                        if n not in wi:
                            continue
                        parent, action = self.tree.parent[n], self.tree.action[n]
                        rr[n] = (p[:, parent, action] > .5) & ~on[:, parent]
                        on[:, n] = on[:, parent] | rr[n] | (reach[:, wi[n]] > .5)
                    labels.append(on.cpu().numpy())
                    ready.append(torch.stack([rr[n] for n in self.ways], 1).cpu().numpy())
            on, rd = np.concatenate(labels), np.concatenate(ready)
            wall = np.array([layouts[e].wall_x for e in eps])
            allowed = wall != 5 if split == "train" else np.ones(len(codes), bool)
            rng = np.random.default_rng(args.seed + (split == "test"))
            # Component screen intentionally covers teacher-positive examples,
            # plus uniform rows. Every metric reports its class counts.
            rows = [rng.choice(np.flatnonzero(allowed), min(args.frames, int(allowed.sum())), replace=False)]
            for j in range(len(self.ways)):
                positive = np.flatnonzero(allowed & rd[:, j])
                rows.append(rng.choice(positive, min(64, len(positive)), replace=False))
            rows = np.unique(np.concatenate(rows))
            ego_codes = self.ego.codes(codes[torch.as_tensor(rows, device=self.device)], states[rows])
            self.splits[split] = {"layouts": layouts, "ep": eps[rows], "states": states[rows],
                                  "codes": ego_codes, "holds": torch.from_numpy(on[rows]).to(self.device),
                                  "ready": torch.from_numpy(rd[rows]).to(self.device), "wall": wall[rows]}
            log(f"{split}: {len(codes)} endpoint observations; screen {len(rows)}; "
                f"ready positives {rd[rows].sum(0).tolist()}")
        # Teacher lifetime ends here; each experimental arm has one learner.
        del model, ck
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def oracle(self, split):
        data = self.splits[split]
        tasks = []
        for ids in np.array_split(np.arange(len(data["states"])), 32):
            if not len(ids):
                continue
            tasks.append(({int(e): data["layouts"][e] for e in np.unique(data["ep"][ids])},
                          self.tree, self.ways, [(int(data["ep"][i]), data["states"][i]) for i in ids]))
        result = self.pool.map(oracle_chunk, tasks)
        distance, valid, holds = [torch.from_numpy(np.concatenate([x[j] for x in result])).to(self.device)
                                  for j in range(3)]
        # Ego.maps also transforms facing channels. A dummy successor array
        # suffices: this component gate trains no movement targets.
        dummy = distance[:, :, None].expand(-1, -1, 3, -1, -1, -1)
        dm = self.ego.maps(distance, dummy, data["states"])[0][..., :H - 1, :]
        vm = self.ego.maps(valid[:, None].long(), valid[:, None, None].long().expand(-1, -1, 3, -1, -1, -1),
                           data["states"])[0][:, 0, :, :H - 1, :] > 0
        data["oracle"] = {"ready": dm == 0, "valid": vm, "holds": holds, "distance": dm}

    def model(self, horizons):
        model = SharedSpatialLearner(len(self.tree.parent), len(self.ways), horizons=horizons).to(self.device)
        model.patch_encoder.load_state_dict(self.initial_encoder)
        return model


def fit(data, model, updates, upper, batch, log, deadline):
    split = data.splits["train"]
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    start = time.monotonic()
    completed = 0
    for update in range(updates):
        if time.monotonic() >= deadline:
            break
        ids = torch.randint(len(split["codes"]), (batch,), device=data.device)
        state = model.encode(bv.images(split["codes"][ids], data.tiles))
        field = model.ready_logits(state)
        on = model.condition_logits(state)[:, data.nodes]
        if upper:
            oracle = split["oracle"]
            ready_loss = balanced_bce(field, oracle["ready"][ids], oracle["valid"][ids, None])
            condition_loss = balanced_bce(on, oracle["holds"][ids][:, data.nodes])
        else:
            # Only the actual agent's entry receives learned readiness labels.
            ready_loss = balanced_bce(field[:, :, UP, R, R], split["ready"][ids])
            condition_loss = balanced_bce(on, split["holds"][ids][:, data.nodes])
        loss = ready_loss + condition_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        completed += 1
        if update % 250 == 0 or update == updates - 1:
            log(f"{'upper' if upper else 'learned'} {update}: ready={ready_loss.item():.4f} "
                f"conditions={condition_loss.item():.4f} {completed/(time.monotonic()-start):.1f} updates/s")
    return {"updates": completed, "requested_updates": updates, "seconds": time.monotonic() - start}


def evaluate(data, model, split="test"):
    source = data.splits[split]
    field, holds = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source["codes"]), 128):
            state = model.encode(bv.images(source["codes"][start:start+128], data.tiles))
            field.append(model.ready_logits(state).cpu().numpy())
            holds.append(model.condition_logits(state).cpu().numpy())
    field, holds = np.concatenate(field), np.concatenate(holds)
    exact = {k: v.cpu().numpy() for k, v in source["oracle"].items()}
    centre = np.zeros_like(exact["valid"])
    centre[:, UP, R, R] = True
    result = {"readiness": {}, "conditions": {}, "teacher": {}}
    for j, n in enumerate(data.ways):
        name = "/".join(data.tree.path(n))
        result["readiness"][name] = {}
        for tag, mask in (("agent", centre), ("other_poses", ~centre),
                          ("unseen_wall", np.broadcast_to((source["wall"] == 5)[:, None, None, None], centre.shape))):
            result["readiness"][name][tag] = confusion(field[:, j], exact["ready"][:, j], mask & exact["valid"])
        result["conditions"][name] = confusion(holds[:, n], exact["holds"][:, n])
        result["teacher"][name] = confusion(source["ready"][:, j].cpu().numpy().astype(float) * 2 - 1,
                                             exact["ready"][:, j, UP, R, R], exact["valid"][:, UP, R, R])
        result["readiness"][name]["teacher_agreement_at_agent"] = confusion(
            field[:, j, UP, R, R], source["ready"][:, j].cpu().numpy())
    return result


def gate_passes(result, upper):
    recall, fp = (.99, .01) if upper else (.95, .05)
    ready = all(passes(row[tag], recall, fp) for row in result["readiness"].values()
                for tag in ("agent", "other_poses", "unseen_wall"))
    conditions = all(passes(row, recall, fp) for row in result["conditions"].values())
    return ready and conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=250)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seconds", type=int, default=480)
    parser.add_argument("--out", type=Path, default=Path("runs/017_shared_gate"))
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 540:
        parser.error("this component screen is capped at 540 seconds; use an approved card for a longer run")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    log = lambda message: print(f"[{time.monotonic()-start:6.1f}s] {message}", flush=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    result = {"arch_version": 3, "kind": "component_gate", "args": {k: str(v) if isinstance(v, Path) else v
                                                                                for k, v in vars(args).items()}}
    result["baselines"] = {"always_false": {"recall": 0., "false_positive": 0.},
                           "always_true": {"recall": 1., "false_positive": 1.},
                           "note": "Rates are defined only when the corresponding class is present; see coverage counts."}
    def save():
        result["seconds"] = time.monotonic() - start
        (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        data = Data(pool, args, log)
        for split in ("train", "test"):
            data.oracle(split)
            log(f"{split} oracle ready; evaluator labels isolated from the learned arm")
        result["data"] = {name: {"frames": len(s["codes"]), "ready_positive": s["ready"].sum(0).tolist(),
                                  "unseen_wall_frames": int((s["wall"] == 5).sum())}
                          for name, s in data.splits.items()}
        result["coverage"] = coverage(data)
        save()
        if args.coverage_only or not result["coverage"]["sufficient"]:
            result["stopped"] = ("coverage only" if result["coverage"]["sufficient"] else
                                 "insufficient class coverage; no model fitting or later stage started")
            result["upper_passed"] = None
            log(result["stopped"])
            save()
            return
        upper = data.model(24)
        result["upper_fit"] = fit(data, upper, args.updates, True, args.batch, log, start + args.seconds - 30)
        result["upper"] = evaluate(data, upper)
        result["upper_passed"] = gate_passes(result["upper"], True)
        torch.save({"model": upper.state_dict(), "tree": data.tree.report(), "ways": data.ways,
                    "arch_version": 3, "readiness_mode": upper.readiness_mode, "condition_mode": upper.condition_mode,
                    "readiness_radius": upper.readiness_radius, "oracle_trained": True}, args.out / "upper.pt")
        save()
        log(f"upper gate passed: {result['upper_passed']}")
        if not result["upper_passed"]:
            result["stopped"] = "component upper bound failed or lacked class coverage; no learned or walking run"
            save()
            return
        del upper
        learned = data.model(24)
        result["learned_fit"] = fit(data, learned, args.updates, False, args.batch, log, start + args.seconds - 30)
        result["learned"] = evaluate(data, learned)
        result["learned_passed"] = gate_passes(result["learned"], False)
        torch.save({"model": learned.state_dict(), "tree": data.tree.report(), "ways": data.ways,
                    "arch_version": 3, "readiness_mode": learned.readiness_mode, "condition_mode": learned.condition_mode,
                    "readiness_radius": learned.readiness_radius, "oracle_trained": False}, args.out / "learned.pt")
        result["stopped"] = "walking upper bound and joint training are subsequent stages; no acting claim"
        save()


if __name__ == "__main__":
    main()
