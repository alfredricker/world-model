"""Evaluator-only audit: can card 017's readiness inputs identify its labels?

Exact equality of patch-code signatures implies identical RGB information
at this architecture's readiness input, for EVERY choice of encoder weights.
No codes or oracle labels are supplied to a learned model by this audit.
"""
import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench_shared import Data, H, R, UP
from worldmodel.envs.keydoor_render import N_CODES
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld


def signatures(codes, valid):
    """Oriented eight-neighbor code values, room histogram, inventory code."""
    h = H - 1
    room = codes[:, :h]
    # Out-of-image padding has zero features, not the features of floor.
    padded = np.pad(room.astype(np.int16), ((0, 0), (1, 1), (1, 1)), constant_values=-1)
    neighbors = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    hist = np.stack([np.bincount(x.ravel(), minlength=N_CODES) for x in room]).astype(np.int16)
    ctx = np.concatenate([hist, codes[:, h, 0:1]], 1)
    all_signatures, all_indices = [], []
    for facing in range(4):
        n, y, x = np.nonzero(valid[:, facing])
        local = neighbors[n, y, x]
        local = np.rot90(local, (facing - UP) % 4, axes=(1, 2)).reshape(-1, 9)
        local = local[:, [0, 1, 2, 3, 5, 6, 7, 8]]
        all_signatures.append(np.concatenate([local, ctx[n]], 1))
        all_indices.append(np.stack([n, np.full_like(n, facing), y, x], 1))
    return np.concatenate(all_signatures), np.concatenate(all_indices)


def audit(data):
    codes = np.concatenate([d["codes"].cpu().numpy() for d in data.splits.values()])
    valid = np.concatenate([d["oracle"]["valid"].cpu().numpy() for d in data.splits.values()])
    ready = np.concatenate([d["oracle"]["ready"].cpu().numpy() for d in data.splits.values()])
    sig, indices = signatures(codes, valid)
    _, group = np.unique(sig, axis=0, return_inverse=True)
    ng = int(group.max()) + 1
    count = np.bincount(group, minlength=ng)
    n, facing, y, x = indices.T
    origins = [(source, i) for source in data.splits.values() for i in range(len(source["codes"]))]

    def witness(index, node):
        frame, direction, row, col = indices[index]
        source, i = origins[frame]
        layout = source["layouts"][int(source["ep"][i])]
        actual = dl.to_tup(source["states"][i])
        fx, fy = ld.DIR_VEC[actual[2]]
        rx, ry = ld.DIR_VEC[(actual[2] + 1) % 4]
        query = (int(actual[0] + (R-row)*fx + (col-R)*rx),
                 int(actual[1] + (R-row)*fy + (col-R)*ry),
                 int((actual[2]+direction-UP) % 4), *actual[3:])
        ex = dl.Exact(layout, data.tree.parent, data.tree.action)
        parent, action = data.tree.parent[node], data.tree.action[node]
        successor = ld.step(layout, query, action)[0]
        return {"layout": asdict(layout), "actual_state": dl.to_arr(actual), "query_state": dl.to_arr(query),
                "parent_before": bool(ex.holds(parent, query)), "parent_after": bool(ex.holds(parent, successor)),
                "direct_ready": bool(ex.ready(parent, action, query))}
    result = {"poses": len(group), "distinct_input_signatures": ng, "ways": {}}
    for j, node in enumerate(data.ways):
        labels = ready[n, j, facing, y, x]
        positive = np.bincount(group, weights=labels, minlength=ng).astype(int)
        negative = count - positive
        mixed = (positive > 0) & (negative > 0)
        np_, nn = int(positive.sum()), int(negative.sum())
        # With <=1% false positives, any positive group costing more than
        # the entire FP allowance is individually unaffordable. Ignoring
        # competition among affordable groups makes this a generous ceiling.
        allowance = .01 * nn
        max_recall = float(positive[negative <= allowance].sum() / np_) if np_ else None
        row = {"positive": np_, "negative": nn, "conflicting_input_groups": int(mixed.sum()),
               "positive_in_conflicts": int(positive[mixed].sum()),
               "negative_in_conflicts": int(negative[mixed].sum()),
               "minimum_classification_errors": int(np.minimum(positive, negative).sum()),
               "optimistic_recall_ceiling_at_1pct_false_positives": max_recall}
        if mixed.any():
            g = np.flatnonzero(mixed)[np.argmax(np.minimum(positive[mixed], negative[mixed]))]
            ids = np.flatnonzero(group == g)
            a = ids[labels[ids]][0]
            b = ids[~labels[ids]][0]
            # Retain explicit RGB-renderable witnesses, not just aggregate rates.
            row["witness"] = {"positive_pose": indices[a].tolist(), "negative_pose": indices[b].tolist(),
                              "positive_codes": codes[n[a]].tolist(), "negative_codes": codes[n[b]].tolist(),
                              "same_input_signature": bool(np.array_equal(sig[a], sig[b])),
                              "positive_truth": witness(a, node), "negative_truth": witness(b, node)}
            assert row["witness"]["positive_truth"]["direct_ready"]
            assert not row["witness"]["negative_truth"]["direct_ready"]
        result["ways"]["/".join(data.tree.path(node))] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/017_input_audit.json"))
    args = parser.parse_args()
    start = time.monotonic()
    log = lambda m: print(f"[{time.monotonic()-start:6.1f}s] {m}", flush=True)
    config = SimpleNamespace(device="cuda", episodes=1000, test_episodes=250, seed=17, frames=512)
    with mp.get_context("spawn").Pool(8) as pool:
        data = Data(pool, config, log)
        for split in data.splits:
            data.oracle(split)
        result = audit(data)
    result["seconds"] = time.monotonic() - start
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for name, row in result["ways"].items():
        log(f"{name}: {json.dumps({k: v for k, v in row.items() if k != 'witness'})}")


if __name__ == "__main__":
    main()
