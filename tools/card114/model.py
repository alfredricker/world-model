"""Card 114: DINO-WM's predictor, shared by the patch model and the state upper bound.

Each frame is a set of N tokens of D numbers (patches: N=196, D=384; state: N=1, D=40). The 5 actions of a model
step (25 numbers) are embedded by an MLP to 10 numbers and appended to every token of the frame they follow
(DINO-WM). A ViT of depth 6 over the tokens of the last H <= 3 frames, with a causal mask over frames, predicts each
frame's successor (teacher forcing in training; open loop at scoring). A linear map in and out sets the width
(384) for both inputs; otherwise it is DINO-WM's design.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 3


class Block(nn.Module):
    def __init__(self, w, heads, mlp):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.qkv, self.proj = nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.mlp = nn.Sequential(nn.Linear(w, mlp), nn.GELU(), nn.Linear(mlp, w))
        self.heads = heads

    def forward(self, x, mask):
        B, L, W = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, W // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, W))
        return x + self.mlp(self.n2(x))


class Predictor(nn.Module):
    def __init__(self, n_tok, dim, act_dim=25, act_emb=10, width=384, depth=6, heads=6, mlp=2048):
        super().__init__()
        self.n_tok, self.dim = n_tok, dim
        self.act = nn.Sequential(nn.Linear(act_dim, 128), nn.GELU(), nn.Linear(128, act_emb))
        self.inp = nn.Linear(dim + act_emb, width)
        self.pos = nn.Parameter(torch.randn(1, H * n_tok, width) * 0.02)
        self.blocks = nn.ModuleList([Block(width, heads, mlp) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        f = torch.arange(H).repeat_interleave(n_tok)
        self.register_buffer('mask', f[:, None] >= f[None, :], persistent=False)

    def forward(self, z, a):
        """z: [B, h, N, D] frames, a: [B, h, 25] the actions after each frame → [B, h, N, D] each frame's successor."""
        B, h, N, D = z.shape
        e = self.act(a)[:, :, None].expand(B, h, N, -1)
        x = self.inp(torch.cat([z, e], -1)).reshape(B, h * N, -1) + self.pos[:, :h * N]
        m = self.mask[:h * N, :h * N]
        for blk in self.blocks:
            x = blk(x, m)
        return self.out(self.norm(x)).reshape(B, h, N, D)

    @torch.no_grad()
    def rollout(self, z_hist, a_all):
        """Open loop. z_hist: [B, H, N, D] real frames, oldest first. a_all: [B, H-1+K, 25], the action chunk after
        each frame (the H-1 real ones between history frames, then the K to imagine) → [B, K, N, D]."""
        seq = z_hist
        K = a_all.shape[1] - (z_hist.shape[1] - 1)
        for k in range(K):
            nxt = self(seq[:, -H:], a_all[:, k:k + H])[:, -1:]
            seq = torch.cat([seq, nxt], 1)
        return seq[:, z_hist.shape[1]:]
