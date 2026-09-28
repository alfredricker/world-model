"""Guard the gate itself: empty classes must not turn into perfect rates."""
import importlib.util
from pathlib import Path

import numpy as np
import torch


spec = importlib.util.spec_from_file_location("card017_bench", Path(__file__).resolve().parents[1] /
                                              "tools/card017/bench_shared.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_confusion_respects_evaluator_validity_mask():
    score = np.array([2., -2., 2., 2.])
    truth = np.array([True, True, False, False])
    m = bench.confusion(score, truth, np.array([True, True, True, False]))
    assert m == {"positive": 2, "negative": 1, "recall": .5, "false_positive": 1.}


def test_absent_class_and_small_samples_do_not_pass():
    empty_positive = bench.confusion(np.full(100, -2.), np.zeros(100, bool))
    assert empty_positive["recall"] is None
    assert not bench.passes(empty_positive, .99, .01)
    small = bench.confusion(np.array([2., -2.]), np.array([True, False]))
    assert not bench.passes(small, .99, .01)


def test_balanced_loss_does_not_reward_majority_prediction():
    truth = torch.zeros(100, 1, dtype=torch.bool)
    truth[0] = True
    wrong = bench.balanced_bce(torch.full((100, 1), -10.), truth)
    neutral = bench.balanced_bce(torch.zeros(100, 1), truth)
    assert wrong > neutral
    # The inventory row is excluded from the evaluator masks; padding cannot
    # become easy negative examples in the map loss.
    logits = torch.zeros(1, 1, 4, 2, 2, requires_grad=True)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[..., 0, 0] = True
    bench.balanced_bce(logits, torch.zeros_like(mask), mask).backward()
    assert (logits.grad[~mask] == 0).all()


def test_local_door_view_cannot_determine_goal_relative_readiness():
    from worldmodel import discover_logic as dl
    from worldmodel.envs import logicdoor as ld
    from worldmodel.envs.keydoor_render import tile_images
    from worldmodel.spatial_state import SharedSpatialLearner

    # Directly checked witness from the exact audit: matching key held and
    # identical local door view, but the goal is on different sides.
    a = ld.Layout(8, 4, 2, "blue", "red", (1, 4), (3, 3), (2, 6), (1, 1), (6, 6), (2, 1, 0), "key")
    b = ld.Layout(8, 3, 4, "blue", "red", (1, 4), (2, 6), (2, 5), (2, 1), (4, 2), (1, 2, 2), "key")
    actual_a = (1, 5, 3, 1, None, (3, 4), 0, 0, 1)
    actual_b = (2, 4, 1, 1, None, (2, 6), 0, 0, 1)
    query_a = (3, 2, 0, *actual_a[3:])
    query_b = (4, 4, 2, *actual_b[3:])
    ex_a = dl.Exact(a, [-1, 0], [-1, ld.FORWARD])
    ex_b = dl.Exact(b, [-1, 0], [-1, ld.FORWARD])
    assert ex_a.ready(1, ld.TOGGLE, query_a)
    assert not ex_b.ready(1, ld.TOGGLE, query_b)
    ego = bench.Ego(torch, torch.device("cpu"))
    actual = (actual_a, actual_b)
    poses = []
    for s, q in zip(actual, (query_a, query_b)):
        fx, fy = ld.DIR_VEC[s[2]]
        rx, ry = ld.DIR_VEC[(s[2] + 1) % 4]
        dx, dy = q[0] - s[0], q[1] - s[1]
        poses.append(((q[2] - s[2] + bench.UP) % 4, bench.R - (dx*fx + dy*fy), bench.R + dx*rx + dy*ry))
    codes = torch.from_numpy(dl.encode_pairs([(a, actual_a), (b, actual_b)]))
    view = ego.codes(codes, np.array([s[:3] for s in actual]))
    rgb = bench.bv.images(view, torch.from_numpy(tile_images()))
    torch.manual_seed(31)
    local = SharedSpatialLearner(2, 1, readiness_radius=1, readiness_mode="wide", condition_mode="flat")
    wide = SharedSpatialLearner(2, 1, readiness_radius=6, readiness_mode="wide", condition_mode="flat")
    wide.patch_encoder.load_state_dict(local.patch_encoder.state_dict())
    wide.context_encoder.load_state_dict(local.context_encoder.state_dict())
    with torch.no_grad():
        state = local.encode(rgb)
        torch.testing.assert_close(state.context[0], state.context[1])
        lo, hi = local.ready_logits(state), wide.ready_logits(state)
        l = [lo[i, 0, d, y, x] for i, (d, y, x) in enumerate(poses)]
        h = [hi[i, 0, d, y, x] for i, (d, y, x) in enumerate(poses)]
        torch.testing.assert_close(l[0], l[1])
        assert abs(float(h[0] - h[1])) > 1e-5
