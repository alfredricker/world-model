"""Card 110's shared parts: the model's per-token vector, familiarity by recall, the confidence fit and the scores.

- vector: the transition model's last-layer vector of each token, after its final LayerNorm (the input to the output
  layer): the model's own state for that token, in the one latent space.
- familiarity: the mean distance from a token's vector to its k nearest stored vectors of the same action (training
  memory), searched exactly in chunks on the GPU.
- confidence (arm R): p_T(outcome) x a(d), a(d) = sigmoid(alpha + beta log d), with alpha, beta fitted by the likelihood
  of "the predicted next code is right" on held-out real tokens; k in (1, 4, 16) chosen by the likelihood on a second
  held-out half. p_T: the model's probability with its temperature fitted the same way (Guo et al. 2017).
"""
import numpy as np
import torch

dev = "cuda"
KS = (1, 4, 16)


class Vectors:
    """Captures the last-layer token vectors of every forward pass of net (B, N, d); the output layer reads only the
    state tokens, so the action token is not among them."""

    def __init__(self, net):
        self.v = None
        net.out[0].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.v = out.detach().to(torch.float16)


class Memory:
    """Stored vectors with their action and source state; search by the action."""

    def __init__(self, vec, act, state):
        self.vec, self.act, self.state = vec, np.asarray(act), np.asarray(state)
        self.by_act = {}
        for a in np.unique(self.act):
            m = torch.as_tensor(self.vec[self.act == a]).to(dev)
            self.by_act[int(a)] = (m, (m.float() ** 2).sum(1))

    def subset(self, keep_states):
        m = np.isin(self.state, keep_states)
        return Memory(self.vec[m], self.act[m], self.state[m])

    @torch.no_grad()
    def knn(self, q, a, ks=KS, chunk_q=1024, chunk_m=400_000):
        """Mean distance to the k nearest stored vectors of action a[i], for each k in ks. q (n, d) fp16 numpy/tensor."""
        q = torch.as_tensor(q).to(dev)
        a = np.asarray(a)
        out = np.zeros((len(q), len(ks)), np.float32)
        K = max(ks)
        for act in np.unique(a):
            idx = np.flatnonzero(a == act)
            if int(act) not in self.by_act:
                out[idx] = np.inf
                continue
            M, M2 = self.by_act[int(act)]
            for s in range(0, len(idx), chunk_q):
                qi = q[idx[s:s + chunk_q]]
                q2 = (qi.float() ** 2).sum(1, keepdim=True)
                best = None
                for c in range(0, len(M), chunk_m):
                    d2 = (q2 + M2[None, c:c + chunk_m] - 2 * (qi @ M[c:c + chunk_m].T).float()).clamp(min=0)
                    v = d2.topk(min(K, d2.shape[1]), dim=1, largest=False).values
                    best = v if best is None else torch.cat([best, v], 1).topk(min(K, best.shape[1] + v.shape[1]),
                                                                               dim=1, largest=False).values
                d = best.sqrt()
                out[idx[s:s + chunk_q]] = np.stack([d[:, :k].mean(1).cpu().numpy() for k in ks], 1)
        return out


def fit_temperature(neg_d2, right_code):
    """neg_d2 (n, C) = -squared distances to the codes; right_code (n,) the true next code. Returns tau maximising the
    likelihood of the true code under softmax(neg_d2 / tau)."""
    x = torch.as_tensor(neg_d2, device=dev).float()
    y = torch.as_tensor(right_code, device=dev).long()
    lt = torch.zeros((), device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([lt], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / lt.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(lt.exp())


def fit_applicability(p, logd, right):
    """conf = p * sigmoid(alpha + beta log d), fitted by the Bernoulli likelihood of right. Returns (alpha, beta)."""
    p = torch.as_tensor(p, device=dev).float().clamp(1e-6, 1 - 1e-6)
    x = torch.as_tensor(logd, device=dev).float()
    y = torch.as_tensor(right, device=dev).float()
    w = torch.tensor([3.0, 0.0], device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=300)

    def closure():
        opt.zero_grad()
        c = (p * torch.sigmoid(w[0] + w[1] * x)).clamp(1e-6, 1 - 1e-6)
        loss = -(y * c.log() + (1 - y) * (1 - c).log()).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(w[0]), float(w[1])


def applic(logd, ab):
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * logd)))


def bern_ll(c, y):
    c = np.clip(c, 1e-6, 1 - 1e-6)
    return float(np.mean(y * np.log(c) + (1 - y) * np.log(1 - c)))


def choose_k(p, D, right):
    """D (n, len(KS)) distances. Fit on one half, score on the other; returns (k index, (alpha, beta), table)."""
    n = len(p)
    rng = np.random.default_rng(0)
    h = rng.permutation(n)
    f, e = h[: n // 2], h[n // 2:]
    tab = {}
    for j, k in enumerate(KS):
        ld = np.log(D[:, j] + 1e-6)
        ab = fit_applicability(p[f], ld[f], right[f])
        tab[k] = {"alpha, beta": [round(ab[0], 3), round(ab[1], 3)],
                  "held-out log-likelihood": round(bern_ll(p[e] * applic(ld[e], ab), right[e]), 5)}
    j = int(np.argmax([tab[k]["held-out log-likelihood"] for k in KS]))
    ld = np.log(D[:, j] + 1e-6)
    return j, fit_applicability(p, ld, right), tab


def ece(c, y, bins=15):
    c, y = np.asarray(c, float), np.asarray(y, float)
    b = np.clip((c * bins).astype(int), 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(c[m].mean() - y[m].mean())
    return round(float(e), 4)


def reliability(c, y, bins=10):
    c, y = np.asarray(c, float), np.asarray(y, float)
    b = np.clip((c * bins).astype(int), 0, bins - 1)
    return [[round(float(c[b == k].mean()), 3), round(float(y[b == k].mean()), 3), int((b == k).sum())]
            for k in range(bins) if (b == k).any()]


def auroc(score, pos):
    """AUROC of score for the positive class (ties count half)."""
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return None
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    npos, nneg = pos.sum(), (~pos).sum()
    return round(float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)), 4)


def prf(tp, fp, fn):
    f = lambda x, y: round(x / max(x + y, 1), 4)
    return f(tp, fp), f(tp, fn)
