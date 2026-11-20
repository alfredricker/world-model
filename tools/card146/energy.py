"""Terms 1 and 2 of the candidate energy (theory/thousand_brains_energy_formulation.md, section 7),
in two forms of a patch's state.  J is [B, P, P], symmetric, zero on the diagonal: J_ij > 0 pulls
patches i and j into one object, J_ij < 0 pushes them into different ones.

settle_labels   the state is a belief p_i over K objects, p_i = softmax(beta v_i):
                E = -1/2 sum_{i != j} J_ij <p_i, p_j> + (1/beta) sum_i <p_i, log p_i>
                This is the Hopfield-Tank energy that CRAM learns (its equation 1), with patches as the
                cells, object identities as the values and -J_ij as the cost of "same value".  Two
                patches that differ have <p_i, p_j> = 0.
settle_vectors  the state is a unit vector x_i with K fixed orthonormal object phases u_k (section 7 as
                first written):  E = -1/2 sum J_ij <x_i, x_j> - (alpha/beta) sum_i log sum_k exp(beta <x_i, u_k>)
Both descend the energy with noise that decays to zero while beta rises from b0 to b1 (annealing).
K is a ceiling on the number of objects, not their number.
"""
import torch
import torch.nn.functional as F


def carriers(K, N, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(N, N, generator=g))
    return q[:K].to(device)                                    # [K, N], orthonormal rows


def settle_labels(J, K=8, T=200, eta=0.3, b0=1.0, b1=30.0, noise=0.01, seed=0):
    B, P, _ = J.shape
    g = torch.Generator(device=J.device).manual_seed(seed)
    v = noise * torch.randn(B, P, K, device=J.device, generator=g)
    for t in range(T):
        beta = b0 * (b1 / b0) ** (t / (T - 1))
        p = torch.softmax(beta * v, -1)
        v = (1 - eta) * v + eta * (J @ p) + noise * (1 - t / T) * torch.randn(v.shape, device=v.device, generator=g)
    p = torch.softmax(b1 * v, -1)
    e = -0.5 * torch.einsum("bik,bij,bjk->b", p, J, p) + (p * torch.log(p.clamp(min=1e-9))).sum((-1, -2)) / b1
    return p.argmax(-1), e, p


def settle_vectors(J, K=8, N=16, T=300, gamma=0.3, b0=1.0, b1=30.0, alpha=4.0, noise=0.05, seed=0):
    B, P, _ = J.shape
    u = carriers(K, N, J.device)
    g = torch.Generator(device=J.device).manual_seed(seed)
    x = F.normalize(torch.randn(B, P, N, device=J.device, generator=g), dim=-1)
    for t in range(T):
        beta = b0 * (b1 / b0) ** (t / (T - 1))
        m = torch.softmax(beta * x @ u.T, -1)
        f = J @ x + alpha * (m @ u) + noise * (1 - t / T) * torch.randn(x.shape, device=x.device, generator=g)
        f = f - (f * x).sum(-1, keepdim=True) * x              # tangent to the sphere
        x = F.normalize(x + gamma * f, dim=-1)
    e = -0.5 * torch.einsum("bin,bij,bjn->b", x, J, x) - (alpha / b1) * torch.logsumexp(b1 * x @ u.T, -1).sum(-1)
    return (x @ u.T).argmax(-1), e, x


def grid(G, device):
    yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    return torch.stack([yy, xx], -1).reshape(-1, 2).float().to(device)


def couplings(feat, pos, tau, radius=None, far=0.0, sharp=1.0, scale=1 / 16):
    """J_ij = scale * (cos(feat_i, feat_j)^sharp - tau) for patches within `radius` patches of each
    other (all pairs if None); pairs farther apart get -scale * far.  feat: [B, P, D]; pos: [P, 2]."""
    f = F.normalize(feat, dim=-1)
    J = (f @ f.transpose(1, 2)).clamp(min=0) ** sharp - tau
    P = pos.shape[0]
    eye = torch.eye(P, dtype=torch.bool, device=feat.device)
    if radius is not None:
        near = torch.cdist(pos, pos) <= radius + 1e-6
        J = torch.where(near, J, torch.full_like(J, -far))
    return scale * J.masked_fill(eye, 0.0)
