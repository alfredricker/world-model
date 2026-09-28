import torch
import pytest

from worldmodel.spatial_state import SharedSpatialLearner, SpatialState


def learner(mode="recurrent"):
    torch.manual_seed(7)
    return SharedSpatialLearner(3, 2, height=5, width=5, channels=8, horizons=3, readiness_mode=mode)


def frame(patches):
    b, h, w = patches.shape[:3]
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, h * 8, w * 8)


def test_context_is_invariant_to_room_patch_permutation():
    net = learner()
    patches = torch.rand(2, 6, 5, 3, 8, 8)
    changed = patches.clone()
    permutation = torch.randperm(25)
    changed[:, :5] = patches[:, :5].reshape(2, 25, 3, 8, 8)[:, permutation].reshape(2, 5, 5, 3, 8, 8)
    a, b = net.encode(frame(patches)), net.encode(frame(changed))
    torch.testing.assert_close(a.context, b.context)
    torch.testing.assert_close(a.spatial.flatten(2)[:, :, permutation], b.spatial.flatten(2))


def test_inventory_is_context_and_not_a_walkable_cell():
    net = learner()
    rgb = torch.rand(2, 3, 48, 40)
    changed = rgb.clone()
    changed[:, :, 40:, :8] = 0
    a, b = net.encode(rgb), net.encode(changed)
    torch.testing.assert_close(a.spatial, b.spatial)
    assert not torch.allclose(a.context, b.context)
    irrelevant = rgb.clone()
    irrelevant[:, :, 40:, 8:] = 0
    torch.testing.assert_close(a.context, net.encode(irrelevant).context)


@pytest.mark.parametrize("mode", ["wide", "recurrent"])
def test_readiness_rotates_locations_and_facing_together(mode):
    net = learner(mode)
    m, c = torch.randn(2, 8, 5, 5), torch.randn(2, 8)
    original = net.ready_logits(SpatialState(m, c))
    rotated = net.ready_logits(SpatialState(torch.rot90(m, 1, (-2, -1)), c))
    expected = torch.rot90(original.roll(-1, dims=2), 1, (-2, -1))
    torch.testing.assert_close(rotated, expected)


@pytest.mark.parametrize("mode", ["wide", "recurrent"])
def test_each_task_trains_the_same_visual_encoder(mode):
    net = learner(mode)
    rgb, way = torch.rand(2, 3, 48, 40), torch.tensor([0, 1])
    for task in ("conditions", "ready", "q"):
        net.zero_grad(set_to_none=True)
        output = net(rgb, way)[task]
        torch.nn.functional.binary_cross_entropy_with_logits(output, torch.ones_like(output)).backward()
        grad = net.patch_encoder[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert sum(name == "patch_encoder.0.weight" for name, _ in net.named_parameters()) == 1


def test_conditions_and_readiness_share_spatial_processor():
    net = learner()
    rgb = torch.rand(2, 3, 48, 40)
    for task in ("conditions", "ready"):
        net.zero_grad(set_to_none=True)
        out = net(rgb, torch.tensor([0, 1]))[task]
        torch.nn.functional.binary_cross_entropy_with_logits(out, torch.ones_like(out)).backward()
        grad = net.ready_recurrence.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
