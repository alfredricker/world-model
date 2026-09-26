"""Hub H of card 001: encoder, latent transition T(z, a) and quasimetric
reachability head d(z, g), on one learned state (C2).

Sources: QRL (structure, actions scored by d(T(z, a), g)), MRN (the head),
LeWM (T with AdaLN, SIGReg), HILP / OGBench GCIVL (action-free expectile
regression of d with an EMA target). See experiments/001 and LITERATURE.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

ACTIONS = 5


@dataclass
class HubConfig:
    latent: int = 192
    trunk: tuple[int, int, int] = (32, 64, 64)
    hidden: int = 256            # encoder projector and T width
    t_blocks: int = 2
    head_hidden: int = 256
    head_out: int = 64           # K: size of the symmetric and asymmetric embeddings
    obs: str = "frames"          # "frames"; "egocentric" or "symbolic" (feasibility gate only)


class FrameTrunk(nn.Module):
    """42x42x3 uint8 frame -> features. Three stride-2 convolutions, then the
    flattened map (keeps where things are in the egocentric view) joined with
    a global max over positions (what is anywhere in view)."""

    def __init__(self, channels):
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1), nn.GELU(),     # 21
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GELU(),    # 11
            nn.Conv2d(c2, c3, 3, 2, 1), nn.GELU(),    # 6
        )
        self.out = c3 * 6 * 6 + c3

    def forward(self, x):                             # (B, 42, 42, 3) uint8
        x = x.permute(0, 3, 1, 2).float() / 255
        h = self.conv(x)
        return torch.cat([h.flatten(1), h.amax((2, 3))], 1)


class SymbolicTrunk(nn.Module):
    """Simulator state as a grid of categorical channels plus the carried
    object -> features. Same three convolutions (first at stride 1); the
    carried object is embedded and joined. Feasibility gate only: the
    egocentric 7x7 view (the upper bound) or the allocentric 8x29 world."""

    def __init__(self, channels, sizes=(8, 7, 5, 4, 7, 5), carry_sizes=(3, 7), shape=(8, 29), embed=16):
        super().__init__()
        c1, c2, c3 = channels
        self.tables = nn.ModuleList(nn.Embedding(n, embed) for n in sizes)
        self.carry = nn.ModuleList(nn.Embedding(n, embed) for n in carry_sizes)
        self.conv = nn.Sequential(
            nn.Conv2d(embed, c1, 3, 1, 1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(c2, c3, 3, 2, 1), nn.GELU(),
        )
        h, w = shape
        for _ in range(2):
            h, w = (h + 1) // 2, (w + 1) // 2
        self.out = c3 * h * w + c3 + embed

    def forward(self, obs):
        grid, carried = obs                              # (B, 8, 29, 6), (B, 2) uint8
        grid, carried = grid.long(), carried.long()
        e = sum(t(grid[..., i]) for i, t in enumerate(self.tables)).permute(0, 3, 1, 2)
        h = self.conv(e)
        c = sum(t(carried[:, i]) for i, t in enumerate(self.carry))
        return torch.cat([h.flatten(1), h.amax((2, 3)), c], 1)


class Encoder(nn.Module):
    def __init__(self, cfg: HubConfig):
        super().__init__()
        if cfg.obs == "frames":
            self.trunk = FrameTrunk(cfg.trunk)
        elif cfg.obs == "egocentric":
            self.trunk = SymbolicTrunk(cfg.trunk, sizes=(11, 6, 3), carry_sizes=(11, 6), shape=(7, 7))
        else:
            self.trunk = SymbolicTrunk(cfg.trunk)
        self.proj = nn.Sequential(nn.Linear(self.trunk.out, cfg.hidden), nn.BatchNorm1d(cfg.hidden), nn.GELU(),
                                  nn.Linear(cfg.hidden, cfg.latent))

    def forward(self, obs):
        return self.proj(self.trunk(obs))


class AdaLNBlock(nn.Module):
    """x + gate(a) * MLP(LN(x) * (1 + scale(a)) + shift(a)); the modulation is
    zero-initialised, so the block starts as the identity (LeWM / DiT)."""

    def __init__(self, width, cond):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.mod = nn.Linear(cond, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        shift, scale, gate = self.mod(c).chunk(3, -1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class Transition(nn.Module):
    """T(z, a) = z + delta(z, a). Starts as "nothing happens"."""

    def __init__(self, cfg: HubConfig):
        super().__init__()
        self.action = nn.Embedding(ACTIONS, 64)
        self.inp = nn.Linear(cfg.latent, cfg.hidden)
        self.blocks = nn.ModuleList(AdaLNBlock(cfg.hidden, 64) for _ in range(cfg.t_blocks))
        self.out = nn.Linear(cfg.hidden, cfg.latent)

    def forward(self, z, a):
        c = self.action(a)
        h = self.inp(z)
        for block in self.blocks:
            h = block(h, c)
        return z + self.out(h)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class MRNHead(nn.Module):
    """d(x, y) = ||phi(x) - phi(y)|| + max_i relu(h_i(x) - h_i(y)): a
    quasimetric by MRN's Proposition 1 (d(x, x) = 0, triangle inequality).

    Goal sets: ``goal_features`` pools examples. The symmetric part takes the
    minimum over examples; the asymmetric part uses the coordinate-wise
    maximum of h over examples. Both are lower bounds on the distance to every
    example, and asymmetric coordinates on which the examples disagree stop
    counting (card 001, "Goals")."""

    def __init__(self, cfg: HubConfig):
        super().__init__()
        self.phi = mlp(cfg.latent, cfg.head_hidden, cfg.head_out)
        self.h = mlp(cfg.latent, cfg.head_hidden, cfg.head_out)

    def features(self, z):
        return self.phi(z), self.h(z)

    def distance(self, x, y):
        """x, y: feature pairs (phi, h), each (..., K). Returns (...)."""
        sym = (x[0] - y[0]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        asym = F.relu(x[1] - y[1]).amax(-1)
        return sym + asym

    def set_distance(self, x, ys, rule="pooled"):
        """x: features (B, K); ys: features of K_g examples (B, n, K)."""
        if rule == "min":
            return self.distance((x[0][:, None], x[1][:, None]), ys).amin(1)
        sym = (x[0][:, None] - ys[0]).pow(2).sum(-1).clamp_min(1e-12).sqrt().amin(1)
        asym = F.relu(x[1] - ys[1].amax(1)).amax(-1)
        if rule == "pooled":
            return sym + asym
        if rule == "asym":
            return asym
        raise ValueError(rule)


class Hub(nn.Module):
    def __init__(self, cfg: HubConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.transition = Transition(cfg)
        self.head = MRNHead(cfg)

    def encode(self, obs):
        return self.encoder(obs)

    @torch.no_grad()
    def fork_scores(self, obs, goal_obs, rule="pooled"):
        """Predicted steps-to-goal after each action, (B, 5), and from the
        current state, (B,). goal_obs holds n examples per fork, flattened
        to (B * n, ...)."""
        z = self.encode(obs)
        b = z.shape[0]
        g = self.head.features(self.encode(goal_obs))
        n = g[0].shape[0] // b
        g = (g[0].view(b, n, -1), g[1].view(b, n, -1))
        now = self.head.set_distance(self.head.features(z), g, rule)
        nxt = self.transition(z.repeat_interleave(ACTIONS, 0),
                              torch.arange(ACTIONS, device=z.device).repeat(b))
        fx = self.head.features(nxt)
        gg = (g[0].repeat_interleave(ACTIONS, 0), g[1].repeat_interleave(ACTIONS, 0))
        after = self.head.set_distance(fx, gg, rule).view(b, ACTIONS)
        return after, now


def sigreg(z: torch.Tensor, projections: int = 1024, knots: int = 17) -> torch.Tensor:
    """LeWM's SIGReg: Epps-Pulley statistic of random 1-D projections of the
    batch against N(0, 1), averaged over projections."""
    n, dim = z.shape
    u = torch.randn(dim, projections, device=z.device)
    u = u / u.norm(dim=0, keepdim=True)
    x = z @ u                                                  # (n, M)
    t = torch.linspace(0, 3, knots, device=z.device)
    target = torch.exp(-0.5 * t ** 2)
    xt = x[..., None] * t                                      # (n, M, knots)
    cos, sin = torch.cos(xt).mean(0), torch.sin(xt).mean(0)    # empirical characteristic function
    err = (cos - target) ** 2 + sin ** 2
    stat = torch.trapezoid(err * target, t, dim=-1) * n
    return stat.mean()


def expectile(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric squared error. With tau > 0.5, overestimates of a distance
    cost more, so d tends toward the lower expectile (the best next state)."""
    u = target - pred
    w = torch.where(u < 0, tau, 1 - tau)
    return (w * u ** 2).mean()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
