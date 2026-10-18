"""Card 111: conditions of an event by contrast over memory (greedy conditional contrasts, Cheng and Novick 1990).

Selection: the fact with the smallest p (ties by power); see greedy().

Memory for one situation (the same action, and the anchored token's code): items, each with a value at every cell of
a window (codes, or the evaluator's labels for the upper bound; PAD outside), and whether the event's outcome followed
(success). The facts of an event e: for every cell, "the cell holds e's value", and "the cell does not hold v" for
every other value v seen there. Greedy: among the items that satisfy the facts kept so far, add the fact with the
smallest one-sided Fisher p for raising the success (causal power, Cheng 1997, is reported), if both sides hold at least N_MIN items and a one-sided Fisher
test gives p < ALPHA / (number of candidate facts); stop when none qualifies.
"""
import torch

dev = "cuda"
N_MIN, ALPHA, MAX_STEPS, TAIL = 5, 0.01, 16, 4000


class Situation:
    """vals (n, cells) long in [0, K]; K is PAD. y (n,) bool."""

    def __init__(self, vals, y, K):
        self.K = K + 1
        self.n, self.cells = vals.shape
        self.M = torch.zeros(self.n, self.cells * self.K, device=dev)
        v = torch.as_tensor(vals, device=dev).long()
        self.M.scatter_(1, v + torch.arange(self.cells, device=dev)[None] * self.K, 1.0)
        self.y = torch.as_tensor(y, device=dev).float()


def log_comb(n, k):
    return torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)


def fisher_upper(k, N, S, n):
    """P(X >= k), X hypergeometric: n draws from N items of which S are successes (all float tensors)."""
    x = k[:, None] + torch.arange(TAIL, device=dev, dtype=torch.float64)[None]
    N, S, n = N[:, None], S[:, None], n[:, None]
    valid = (x <= torch.minimum(n, S)) & (n - x <= N - S)
    xc = torch.minimum(x, torch.minimum(n, S))
    lp = log_comb(S, xc) + log_comb(N - S, (n - xc).clamp(min=0)) - log_comb(N, n)
    lp = torch.where(valid, lp, torch.full_like(lp, -float("inf")))
    return torch.logsumexp(lp, 1).exp()


@torch.no_grad()
def miss_reason(sit, m, cell, val):
    """Why a true condition at cell (e's value val) was not kept, in the final subset m: no failures (or no
    successes) left, the cell never varies there, or enough cases each side (the method is short)."""
    N, S = float(m.sum()), float((m * sit.y).sum())
    if S == N or S == 0:
        return "no contrast left (all successes)" if S == N else "no successes left"
    n_eq = float((m * sit.M[:, cell * sit.K + val]).sum())
    if n_eq < N_MIN or N - n_eq < N_MIN:
        return "the cell does not vary (< 5 a side)"
    return "enough cases each side"


def greedy(sit, e_vals, keep=None, final=None):
    """e_vals (cells,) long. keep: optional (n,) bool, the items memory holds (P19). Returns the kept facts:
    (cell, value, is_equal, power, p, n_present, n_absent)."""
    K, cells = sit.K, sit.cells
    m = torch.ones(sit.n, device=dev) if keep is None else torch.as_tensor(keep, device=dev).float()
    e = torch.as_tensor(e_vals, device=dev).long()
    eq_idx = torch.arange(cells, device=dev) * K + e                                   # positive facts
    out = []
    for _ in range(MAX_STEPS):
        N, S = m.sum().double(), (m * sit.y).sum().double()
        if N < 2 * N_MIN or S == 0 or S == N:
            break
        n1 = (sit.M.T @ m).double()                                                     # items holding each value
        s1 = (sit.M.T @ (m * sit.y)).double()
        # equality facts at e's values; negations at every other value seen
        neg = torch.ones(cells * K, dtype=torch.bool, device=dev)
        neg[eq_idx] = False
        neg &= n1 > 0
        pres = torch.cat([n1[eq_idx], N - n1[neg]])
        sp = torch.cat([s1[eq_idx], S - s1[neg]])
        idx = torch.cat([eq_idx, torch.nonzero(neg).squeeze(1)])
        is_eq = torch.cat([torch.ones(cells, dtype=torch.bool, device=dev), torch.zeros(int(neg.sum()), dtype=torch.bool, device=dev)])
        absn, sa = N - pres, S - sp
        p1, p0 = sp / pres.clamp(min=1), sa / absn.clamp(min=1)
        dp = p1 - p0
        ok = (pres >= N_MIN) & (absn >= N_MIN) & (dp > 0)
        for f in out:                                                                   # a kept fact is not re-added
            ok &= idx != f[0] * K + f[1]
        if not ok.any():
            break
        cand = torch.nonzero(ok).squeeze(1)
        p = fisher_upper(sp[cand], N.expand(len(cand)), S.expand(len(cand)), pres[cand])
        sig = p < ALPHA / len(idx)
        if not sig.any():
            break
        cand, p = cand[sig], p[sig]
        power = dp[cand] / (1 - p0[cand]).clamp(min=1e-12)
        # the most strongly supported fact (smallest p; ties by power). Largest power alone (the card's first rule)
        # picked small, accidentally pure subsets in the smoke test (2026-10-17)
        lp = torch.log(p.clamp(min=1e-300))
        best = int(torch.argmin(lp - 1e-6 * power))
        j = int(cand[best])
        cell, val = int(idx[j]) // K, int(idx[j]) % K
        col = sit.M[:, int(idx[j])]
        m = m * (col if bool(is_eq[j]) else 1 - col)
        out.append((cell, val, bool(is_eq[j]), round(float(power[best]), 4), float(p[best]), int(pres[j]), int(absn[j])))
    if final is not None:
        final.append(m)
    return out
