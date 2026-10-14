"""Series 105's transition model: one model over a set of token vectors, for every action and every token.

A state is a set of tokens. Each token is one vector (what perception gives: the encoder's vector), a position in its
frame (world or screen) and whether it is present. The action is one more token. A transformer reads the set: its
attention logits are the content term plus a learned function of the two tokens' position difference when both are
in the same frame, and a learned constant per pair of frames otherwise. No absolute position is an input (arm R).
Arm A gives raw positions as inputs instead, as card 102.1's first run did. The output is every token's next vector,
as a change added to its current one, and its presence. Nothing is named: no event types, places or categories.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as fn


class RelBias(nn.Module):
    """Per-head attention bias from the position difference p_j - p_i (same frame), else a constant per frame pair.
    bounded (card 105.6, Swin Transformer V2's form): differences enter log-spaced, sign(d) log2(1 + |d|) / log2(8),
    and the bias is 16 sigmoid(output)."""

    def __init__(self, heads, nframes=3, hidden=64, scale=8.0, bounded=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, heads))
        self.cross = nn.Parameter(torch.zeros(nframes, nframes, heads))
        self.scale, self.bounded = scale, bounded

    def forward(self, P, F):
        d = P[:, None, :, :] - P[:, :, None, :]                         # (B, N, N, 2): j's position minus i's
        B = d.shape[0]
        if B > 1 and bool((F == F[:1]).all()):
            same = (F[:1, :, None] == F[:1, None, :])[..., None]        # only differences within a frame are read
            if bool((torch.where(same, d, 0) == torch.where(same, d[:1], 0)).all()):
                return self.forward(P[:1], F[:1]).expand(B, -1, -1, -1)  # the same differences: computed once (exact)
        d = torch.sign(d) * torch.log2(1 + d.abs()) / math.log2(8) if self.bounded else d / self.scale
        same = (F[:, :, None] == F[:, None, :])
        b = self.mlp(d).permute(0, 3, 1, 2)                              # (B, H, N, N)
        cross = self.cross
        if self.bounded:
            b, cross = 16 * torch.sigmoid(b), 16 * torch.sigmoid(cross)
        oh = fn.one_hot(F, cross.shape[0]).to(b.dtype)                   # a product, not a gather: no atomic adds
        c = torch.einsum("bif,bjg,fgh->bhij", oh, oh, cross.to(b.dtype))
        return torch.where(same[:, None], b, c)


class Block(nn.Module):
    def __init__(self, d, heads, rel, bounded=False):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.bias = RelBias(heads, bounded=bounded) if rel else None

    def forward(self, x, P, F, M):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        logit = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if self.bias is not None:
            logit = logit + self.bias(P, F)
        logit = logit.masked_fill(~M[:, None, None, :], float("-inf"))
        att = logit.softmax(-1)
        x = x + self.o((att @ v).transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.n2(x)), att


class TransitionModel(nn.Module):
    """next vectors = V + delta(V, P, F, action); presence logits. Arm "R" (relative), "Rb" (relative, bounded and
    log-spaced, card 105.6) or "A" (raw positions in)."""

    def __init__(self, dv, na, d=128, layers=4, heads=4, arm="R", nframes=3):
        super().__init__()
        self.arm = arm
        self.inp = nn.Linear(dv + 1 + (2 + nframes if arm == "A" else 0), d)
        self.act = nn.Embedding(na, d)
        self.blocks = nn.ModuleList(Block(d, heads, rel=arm.startswith("R"), bounded=arm == "Rb") for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dv + 1))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)
        self.nframes = nframes

    def forward(self, V, P, F, M, a, return_att=False):
        """V (B, N, dv), P (B, N, 2), F (B, N) frame ids (long), M (B, N) present, a (B,). The action token is added
        as frame nframes - 1 at position 0."""
        B, N, _ = V.shape
        feats = [V, M[..., None].float()]
        if self.arm == "A":
            feats += [P / 8.0, fn.one_hot(F, self.nframes).float()]
        x = self.inp(torch.cat(feats, -1))
        x = torch.cat([self.act(a)[:, None], x], 1)
        P = torch.cat([torch.zeros_like(P[:, :1]), P], 1)
        F = torch.cat([torch.full_like(F[:, :1], self.nframes - 1), F], 1)
        M = torch.cat([torch.ones_like(M[:, :1]), M], 1)
        atts = []
        for b in self.blocks:
            x, att = b(x, P, F, M)
            atts.append(att)
        y = self.out(x[:, 1:])
        nxt = V + y[..., :-1]
        return (nxt, y[..., -1], atts) if return_att else (nxt, y[..., -1])


def weighted_loss(pred, pres, Vn, Mn, V, valid, w_change=None, plain=False):
    """Squared distance to the next vectors, with tokens whose vector changed carrying half the weight (the change is
    read from the data: target differs from input), plus presence by cross-entropy. valid: tokens with a target.
    plain (card 105.7): every token with a target weighs the same, so the minimiser is the expected next vector; the
    weighting above favours change wherever changes are random (a cow that may step in)."""
    err = ((pred - Vn) ** 2).sum(-1)
    if plain:
        w = valid / valid.sum().clamp(min=1)
        return (w * err).sum() + fn.binary_cross_entropy_with_logits(pres[valid], Mn[valid].float())
    changed = (((Vn - V) ** 2).sum(-1) > 1e-3) & valid
    n_c, n_u = changed.sum().clamp(min=1), (valid & ~changed).sum().clamp(min=1)
    w = torch.where(changed, 0.5 / n_c, 0.5 / n_u) * valid
    return (w * err).sum() + fn.binary_cross_entropy_with_logits(pres[valid], Mn[valid].float())


def code_loss(pred, pres, Vn, Mn, valid, protos, log_tau):
    """Card 105.8: the log-likelihood of each token's next code (the prototype nearest its next vector, perception's
    own name for it) under p(c) proportional to exp(-|pred - proto_c|^2 / tau), tau learned; every token with a target
    weighs the same. A proper scoring rule over the same space: no outcome favoured, and a confident miss costs without
    bound. Presence by cross-entropy, as before."""
    D = pred.shape[-1]
    d2 = torch.cdist(pred.reshape(-1, D), protos) ** 2
    with torch.no_grad():
        tgt = torch.cdist(Vn.reshape(-1, D), protos).argmin(1)
    ce = fn.cross_entropy(-d2 / log_tau.exp(), tgt, reduction="none").view(valid.shape)
    w = valid / valid.sum().clamp(min=1)
    return (w * ce).sum() + fn.binary_cross_entropy_with_logits(pres[valid], Mn[valid].float())
