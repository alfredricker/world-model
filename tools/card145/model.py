"""Card 145: the state layer.

  x_t   patch features of step t, [1024, 64] (principal components, mean frame removed, one overall scale)
  p_t   the arm's own pose, 12 numbers (standardised)
  z_t   the state, S numbers

  expected  = decoder(z_{t-1}, p_t)                 what the old state says this frame should look like
  candidate, s = encoder(x_t, x_t − expected, p_t, z_{t-1})
  gate      = max(0, tanh(s + noise))               per dimension (GateL0RD); forced open at a segment's first step
  z_t       = z_{t-1} + gate × (candidate − z_{t-1})
  x̂_t      = decoder(z_t, p_t)
  loss      = squared error of x̂_t (summed over a patch's 64 numbers, mean over patches)
              + λ × the number of gates open per step (a step function, its slope passed straight through)

The decoder draws every patch from a few tokens made of the state and of the pose: a patch's query (its place in
the image) attends to them, with no attention between patches. The encoder's few queries attend to the patches.
"""
import math

import torch
import torch.nn as nn

N, FEAT, POSE = 1024, 64, 12


def fourier(n=32, bands=6):
    g = torch.linspace(-1, 1, n)
    xy = torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
    f = torch.cat([xy * (2 ** k) * math.pi for k in range(bands)], -1)
    return torch.cat([xy, f.sin(), f.cos()], -1)                           # [1024, 2 + 4·bands]


class Block(nn.Module):
    def __init__(self, W, self_attn=False):
        super().__init__()
        self.n1, self.att = nn.LayerNorm(W), nn.MultiheadAttention(W, 4, batch_first=True)
        self.sa = (nn.LayerNorm(W), nn.MultiheadAttention(W, 4, batch_first=True)) if self_attn else None
        if self_attn:
            self.ns, self.sat = self.sa
        self.n2, self.mlp = nn.LayerNorm(W), nn.Sequential(nn.Linear(W, 2 * W), nn.GELU(), nn.Linear(2 * W, W))

    def forward(self, q, mem):
        q = q + self.att(self.n1(q), mem, mem, need_weights=False)[0]
        if self.sa:
            h = self.ns(q)
            q = q + self.sat(h, h, h, need_weights=False)[0]
        return q + self.mlp(self.n2(q))


class Decoder(nn.Module):
    def __init__(self, S, W=256, L=4, ns=8, npose=8):
        super().__init__()
        self.W, self.ns, self.npose = W, ns, npose
        self.register_buffer('ff', fourier())
        self.pos = nn.Sequential(nn.Linear(self.ff.shape[1], W), nn.GELU(), nn.Linear(W, W))
        self.free = nn.Parameter(torch.zeros(N, W))
        self.st = nn.Sequential(nn.Linear(S, W), nn.GELU(), nn.Linear(W, ns * W))
        self.po = nn.Sequential(nn.Linear(POSE, W), nn.GELU(), nn.Linear(W, npose * W))
        self.kind = nn.Parameter(torch.randn(ns + npose, W) * 0.02)
        self.blocks = nn.ModuleList([Block(W) for _ in range(L)])
        self.norm, self.out = nn.LayerNorm(W), nn.Linear(W, FEAT)

    def forward(self, z, pose, use_pose=True, idx=None):                   # [B, S], [B, 12] → [B, 1024, 64], or only the patches idx [B, n]
        B = z.shape[0]
        mem = torch.cat([self.st(z).view(B, self.ns, self.W), self.po(pose).view(B, self.npose, self.W) * float(use_pose)], 1) + self.kind
        q = self.pos(self.ff) + self.free
        q = q.expand(B, -1, -1) if idx is None else q[idx]
        for b in self.blocks:
            q = b(q, mem)
        return self.out(self.norm(q))


class Encoder(nn.Module):
    def __init__(self, S, W=192, M=8, L=2):
        super().__init__()
        self.W, self.M = W, M
        self.register_buffer('ff', fourier())
        self.inp = nn.Linear(2 * FEAT, W)
        self.pos = nn.Sequential(nn.Linear(self.ff.shape[1], W), nn.GELU(), nn.Linear(W, W))
        self.q = nn.Parameter(torch.randn(M, W) * 0.02)
        self.cond = nn.Sequential(nn.Linear(S + POSE, W), nn.GELU(), nn.Linear(W, M * W))
        self.blocks = nn.ModuleList([Block(W, self_attn=True) for _ in range(L)])
        self.head = nn.Sequential(nn.LayerNorm(M * W), nn.Linear(M * W, 512), nn.GELU(), nn.Linear(512, 2 * S))

    def forward(self, x, res, pose, z):
        B = x.shape[0]
        mem = self.inp(torch.cat([x, res], -1)) + self.pos(self.ff)
        q = self.q + self.cond(torch.cat([z, pose], -1)).view(B, self.M, self.W)
        for b in self.blocks:
            q = b(q, mem)
        cand, s = self.head(q.reshape(B, -1)).chunk(2, -1)
        return cand, s


class State(nn.Module):
    def __init__(self, S=32, use_pose=True, noise=0.1):
        super().__init__()
        self.S, self.use_pose, self.noise = S, use_pose, noise
        self.enc, self.dec = Encoder(S), Decoder(S)

    def step(self, x, pose, z, first=False):                               # first: True, False, or a mask [B] of samples whose gate is forced open
        with torch.no_grad():
            res = x - self.dec(z, pose, self.use_pose)
        cand, s = self.enc(x, res, pose, z)
        if self.training:
            s = s + self.noise * torch.randn_like(s)
        g = torch.relu(torch.tanh(s))
        if first is True:
            g = torch.ones_like(s)
        elif torch.is_tensor(first):
            g = torch.where(first[:, None], torch.ones_like(g), g)
        return z + g * (cand - z), g

    def forward(self, x, pose, z=None, idx=None, first=None):              # x [B, T, 1024, 64], pose [B, T, 12] → x̂ (at patches idx [B·T, n] if given), z, gates [B, T, S]
        B, T = x.shape[:2]
        if z is None:
            z, first = x.new_zeros(B, self.S), True
        zs, gs = [], []
        for t in range(T):
            z, g = self.step(x[:, t], pose[:, t], z, first if t == 0 else False)
            zs.append(z); gs.append(g)
        zs, gs = torch.stack(zs, 1), torch.stack(gs, 1)
        rec = self.dec(zs.reshape(B * T, -1), pose.reshape(B * T, -1), self.use_pose, idx)
        rec = rec.view(B, T, -1, FEAT)
        return rec, zs, gs


@torch.no_grad()
def run_filter(model, x, pose):
    """Whole episodes, no reconstruction kept: x [B, T, 1024, 64] (any float type), pose [B, T, 12] → z, gates [B, T, S]."""
    z = x.new_zeros(x.shape[0], model.S, dtype=torch.float32)
    zs, gs = [], []
    for t in range(x.shape[1]):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z, g = model.step(x[:, t].float(), pose[:, t], z, t == 0)
        z, g = z.float(), g.float()
        zs.append(z); gs.append(g)
    return torch.stack(zs, 1), torch.stack(gs, 1)


def open_count(g):
    """The number of gates open, [..., S] → [...]; a step function forward, slope 1 backward."""
    return (g + ((g > 0).float() - g).detach()).sum(-1)
