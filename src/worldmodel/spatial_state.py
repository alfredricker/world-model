"""One pixel state for condition detection, local readiness, and walking.

Cards 017–020 isolate information loss and spatial generalization; version 3
uses shared spatial features for readiness and condition readouts.
No simulator codes, object labels, or teacher network enter this module.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SpatialState:
    spatial: torch.Tensor                 # B, C, H, W
    context: torch.Tensor                 # B, C


class SharedSpatialLearner(nn.Module):
    def __init__(self, n_conditions, n_ways, height=13, width=13, channels=64, horizons=24,
                 readiness_radius=6, readiness_mode="recurrent", condition_mode="spatial"):
        super().__init__()
        self.height, self.width = height, width
        self.n_conditions, self.n_ways = n_conditions, n_ways
        self.channels, self.horizons = channels, horizons
        if readiness_radius < 1:
            raise ValueError("readiness_radius must be positive")
        self.readiness_radius = readiness_radius
        if readiness_mode not in ("wide", "recurrent"):
            raise ValueError("readiness_mode must be wide or recurrent")
        self.readiness_mode = readiness_mode
        if condition_mode not in ("flat", "spatial"):
            raise ValueError("condition_mode must be flat or spatial")
        self.condition_mode = condition_mode
        # The same encoder processes each room patch and the inventory patch.
        # Pool BEFORE spatial mixing: context cannot encode relative positions.
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, channels, 4, 2, 1), nn.ReLU())
        self.context_encoder = nn.Sequential(nn.Linear(2 * channels, channels), nn.ReLU())
        self.conditions = nn.Sequential(
            nn.Linear(channels * height * width + channels, 256), nn.ReLU(),
            nn.Linear(256, n_conditions))
        # Readiness is relative to a GOAL, so a door's local appearance alone
        # is insufficient (card 017's exact input-collision audit). Read the
        # spatial arrangement around each queried pose, with radius 1 kept
        # as the failed local-only control. The query's own tile is excluded.
        spatial_radius = readiness_radius if readiness_mode == "wide" else 1
        side = 2 * spatial_radius + 1
        self.ready_spatial = nn.Conv2d(channels, 128, side, padding=spatial_radius)
        self.ready_context = nn.Linear(channels, 128, bias=False)
        self.ready_output = nn.Conv2d(128, n_ways, 1)
        self.ready_recurrence = nn.Conv2d(128, 128, 3, padding=1) if readiness_mode == "recurrent" else None
        mask = torch.ones(1, 1, side, side)
        mask[..., spatial_radius, spatial_radius] = 0
        self.register_buffer("ready_kernel_mask", mask)
        self.features = nn.Conv2d(channels, 32, 1)
        self.context_step = nn.Linear(channels, 16)
        # Card 016's learned recurrence; its sigmoid is explicit here. This is
        # not an exact Bellman operator and carries no convergence guarantee.
        self.step = nn.Sequential(nn.Conv2d(4 + 32 + 16, 64, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(64, 12, 1))
        nn.init.constant_(self.ready_output.bias, -4.)
        nn.init.constant_(self.step[-1].bias, -4.)
        if condition_mode == "spatial":
            # Construct after all shared modules so matched seeds initialize
            # the shared processor identically in flat/spatial comparisons.
            self.conditions = nn.Linear(128, n_conditions)

    def encode(self, rgb):
        """RGB B,3,(H+1)*8,W*8 in [0,1]; final row's first patch is inventory."""
        b, ch, ih, iw = rgb.shape
        h, w, c = self.height, self.width, self.channels
        if (ch, ih, iw) != (3, (h + 1) * 8, w * 8):
            raise ValueError(f"expected RGB (*,3,{(h+1)*8},{w*8}), got {tuple(rgb.shape)}")
        patches = rgb.reshape(b, 3, h + 1, 8, w, 8).permute(0, 2, 4, 1, 3, 5)
        room = patches[:, :h].reshape(b, h * w, 3, 8, 8)
        held = patches[:, h, 0].unsqueeze(1)
        tokens = self.patch_encoder(torch.cat([room, held], 1).reshape(-1, 3, 8, 8))
        tokens = tokens.reshape(b, h * w + 1, c)
        m = tokens[:, :-1].transpose(1, 2).reshape(b, c, h, w)
        context = self.context_encoder(torch.cat([tokens[:, :-1].mean(1), tokens[:, -1]], 1))
        return SpatialState(m, context)

    def condition_logits(self, state):
        if self.condition_mode == "spatial":
            hidden = self.ready_features(state, 3)
            return self.conditions(hidden[:, :, self.height // 2, self.width // 2]).float()
        return self.conditions(torch.cat([state.spatial.flatten(1), state.context], 1)).float()

    def ready_features(self, state, facing):
        """Shared spatial processor, expressed in the original map coordinates."""
        m, c = state.spatial, state.context
        context = self.ready_context(c)[:, :, None, None]
        turns = (facing - 3) % 4
        rotated = torch.rot90(m, turns, (-2, -1))
        hidden = F.conv2d(rotated, self.ready_spatial.weight * self.ready_kernel_mask,
                          self.ready_spatial.bias, padding=self.ready_spatial.padding)
        initial = F.relu(hidden + context)
        hidden = initial
        if self.ready_recurrence is not None:
            # Feature updates share weights across space and distance. They
            # are not simulated time steps, supplied paths or object rules.
            for _ in range(self.readiness_radius - 1):
                hidden = F.relu(initial + self.ready_recurrence(hidden))
        return torch.rot90(hidden, -turns, (-2, -1))

    def ready_logits(self, state):
        """B,ways,4,H,W. Directions are right, down, left, up (renderer convention)."""
        outputs = []
        for facing in range(4):
            outputs.append(self.ready_output(self.ready_features(state, facing)))
        return torch.stack(outputs, 2).float()

    def walk(self, state, way, full=False):
        """Use the same state and readiness field; return logits, not probabilities.

        Normal output is ready B, action values B,K,3, state values B,K+1
        at the egocentric centre. Full output retains the facing and map axes.
        """
        m, c = state.spatial, state.context
        b, _, h, w = m.shape
        target = self.ready_logits(state)[torch.arange(b, device=m.device), way]
        f = torch.cat([F.relu(self.features(m)), self.context_step(c)[:, :, None, None].expand(-1, -1, h, w)], 1)
        v, values, actions = target, [target], []
        for _ in range(self.horizons):
            q = self.step(torch.cat([v.sigmoid().to(f.dtype), f], 1)).float().reshape(b, 3, 4, h, w)
            v = torch.maximum(target, q.max(1).values)
            values.append(v)
            actions.append(q)
        q, v = torch.stack(actions, 1), torch.stack(values, 1)
        if full:
            return target, q, v
        return target[:, 3, h // 2, w // 2], q[:, :, :, 3, h // 2, w // 2], v[:, :, 3, h // 2, w // 2]

    def forward(self, rgb, way):
        state = self.encode(rgb)
        ready, q, v = self.walk(state, way)
        return {"conditions": self.condition_logits(state), "ready": ready, "q": q, "v": v}
