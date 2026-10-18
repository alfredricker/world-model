"""Card 111.2 (111.1's method; closed now needs N_CLOSE cases): conditions of an event by contrast over memory (greedy conditional contrasts, Cheng and Novick 1990),
revised from card 111 (tools/card111/contrast.py).

Memory for one situation (the same action, and the anchored token's code): items, each with a value at every cell of
a window (codes, or the evaluator's labels for the upper bound; PAD outside), and whether the event's outcome followed
(success). The facts of an event e: for every cell, "the cell holds e's value", and "the cell does not hold v" for
every other value v seen there. Greedy: among the items that satisfy the facts kept so far, add the fact with the
smallest one-sided Fisher p for raising the success, if both sides hold at least N_MIN items and p passes the
threshold; stop when none qualifies. Changes from 111:
  (a) p in log space (111 clamped p at 1e-300, so with large memories every strong fact tied);
  (b) tie set: the facts holding on exactly the same remembered items as the chosen one are returned with it;
  (c) threshold ALPHA / k by Tarone (1990): k counts only facts whose smallest possible p, from their counts, could
      pass (111: ALPHA / every fact);
  (d) closed when the kept facts leave only successes (at least N_CLOSE; 111.1: N_MIN), otherwise open, with near misses: the cells
      where the remaining failures differ from e more often than the remaining successes do (Winston 1970).
"""
import math

import numpy as np

import torch

dev = "cuda"
N_MIN, ALPHA, MAX_STEPS, TAIL = 5, 0.01, 16, 4000
N_CLOSE = 29                                       # (1 - 0.1)^29 < 0.05: a 10% failure rate would likely have shown


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
    """log P(X >= k), X hypergeometric: n draws from N items of which S are successes (all float tensors)."""
    x = k[:, None] + torch.arange(TAIL, device=dev, dtype=torch.float64)[None]
    N, S, n = N[:, None], S[:, None], n[:, None]
    valid = (x <= torch.minimum(n, S)) & (n - x <= N - S)
    xc = torch.minimum(x, torch.minimum(n, S))
    lp = log_comb(S, xc) + log_comb(N - S, (n - xc).clamp(min=0)) - log_comb(N, n)
    lp = torch.where(valid, lp, torch.full_like(lp, -float("inf")))
    return torch.logsumexp(lp, 1)                                                     # log p: p underflows past 1e-300


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


def greedy(sit, e_vals, keep=None, final=None, tarone=True):
    """e_vals (cells,) long. keep: optional (n,) bool, the items memory holds (P19). Returns the kept facts:
    (cell, value, is_equal, power, log p, n_present, n_absent, tie cells). final, if given, receives the final items
    (mask), the steps' tie sets, and the answer: closed or open, the remaining counts and the near misses."""
    K, cells = sit.K, sit.cells
    m = torch.ones(sit.n, device=dev) if keep is None else torch.as_tensor(keep, device=dev).float()
    e = torch.as_tensor(e_vals, device=dev).long()
    eq_idx = torch.arange(cells, device=dev) * K + e                                   # positive facts
    out, steps = [], []
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
        lp = fisher_upper(sp[cand], N.expand(len(cand)), S.expand(len(cand)), pres[cand])
        sig = lp < (log_threshold(pres, absn, N, S) if tarone else math.log(ALPHA / len(idx)))
        if not sig.any():
            break
        cand, lp = cand[sig], lp[sig]
        p = lp.exp()
        power = dp[cand] / (1 - p0[cand]).clamp(min=1e-12)
        # the most strongly supported fact: smallest log p (card 111 clamped p at 1e-300, so large memories tied)
        best = int(torch.argmin(lp - 1e-9 * power))
        j = int(cand[best])
        cell, val = int(idx[j]) // K, int(idx[j]) % K
        col = sit.M[:, int(idx[j])]
        chosen = m * (col if bool(is_eq[j]) else 1 - col)
        # tie set: every fact (equality or negation) that holds on exactly the same remembered cases as the chosen one
        ov = (sit.M.T @ chosen).double()                                               # |M_j ∩ chosen|
        nc = chosen.sum().double()
        tie_eq = (n1 == nc) & (ov == nc)
        tie_neg = ((N - n1) == nc) & ((nc - ov) == nc) & (n1 > 0)
        tie = torch.nonzero(tie_eq).squeeze(1).tolist(), torch.nonzero(tie_neg).squeeze(1).tolist()
        steps.append((m, tie))
        m = chosen
        tie_cells = sorted({c // K for c in tie[0] + tie[1]} | {cell})
        out.append((cell, val, bool(is_eq[j]), round(float(power[best]), 4), float(lp[best]), int(pres[j]), int(absn[j]),
                    tie_cells))
    if final is not None:
        final.append(m)
        final.append(steps)
        final.append(answer(sit, m, eq_idx))
    return out


def log_threshold(pres, absn, N, S):
    """Tarone (1990): log(ALPHA / k), k the smallest number with at most k facts whose smallest possible p (all
    present items successes, as far as the counts allow) is below ALPHA / k. Facts below N_MIN a side are untestable."""
    t = (pres >= N_MIN) & (absn >= N_MIN)
    if not t.any():
        return -float("inf")
    n = pres[t]
    lpm = torch.sort(fisher_upper(torch.minimum(n, S.expand_as(n)), N.expand_as(n), S.expand_as(n), n)).values
    ks = torch.arange(1, len(n) + 1, device=dev, dtype=torch.float64)
    m_k = torch.searchsorted(lpm.contiguous(), (math.log(ALPHA) - torch.log(ks)).contiguous())
    k = int(ks[torch.nonzero(m_k <= ks)[0, 0]])
    return math.log(ALPHA / k)


@torch.no_grad()
def answer(sit, m, eq_idx, top=10):
    """Closed when the kept facts leave only successes (at least N_CLOSE); otherwise open, with near misses."""
    y = sit.y
    N, S = float(m.sum()), float((m * y).sum())
    out = {"closed": bool(S == N and N >= N_CLOSE), "closed on 5 (111.1)": bool(S == N and N >= N_MIN),
           "items left": int(N), "failures left": int(N - S), "near misses": []}
    F, Sx = m * (1 - y), m * y
    if N - S > 0:
        f_same = (sit.M[:, eq_idx].T @ F).double()                                      # failures holding e's value
        s_same = (sit.M[:, eq_idx].T @ Sx).double()
        score = (1 - f_same / (N - S)) - (1 - s_same / max(S, 1.0))
        order = torch.argsort(score, descending=True)[:top]
        out["near misses"] = [int(c) for c in order.tolist() if float(score[c]) > 0]
    return out


@torch.no_grad()
def in_tie(sit, steps, cell, val):
    """Was a fact about cell (e's value val, or a negation at that cell) tied with a chosen fact at some step?"""
    K = sit.K
    for _, (eqs, negs) in steps:
        if any(c // K == cell for c in eqs) or any(c // K == cell for c in negs):
            return True
    return False


@torch.no_grad()
def why_enough(sit, m, cell, val):
    """For a miss with enough cases each side: the equality fact's effect size and p in the final subset."""
    K = sit.K
    col = sit.M[:, cell * K + val]
    N, S = m.sum().double(), (m * sit.y).sum().double()
    n1, s1 = (m * col).sum().double(), (m * col * sit.y).sum().double()
    dp = s1 / n1 - (S - s1) / (N - n1)
    if dp <= 0:
        return "e's value does not raise success here"
    lp = float(fisher_upper(s1[None], N[None], S[None], n1[None])[0])
    import math
    return "raises success, not significant" if lp >= math.log(ALPHA / (sit.cells * K)) else "significant, but not chosen (step limit)"


@torch.no_grad()
def contrastable(sit, cells, vals, keep=None):
    """Scoring only. For each true condition (cell, e's value): among remembered cases that match e on every other
    true condition, does this cell hold e's value in >= N_MIN cases and another value in >= N_MIN? A condition that
    is not contrastable cannot come from memory; it is experiment's."""
    if not len(cells):
        return []
    cols = torch.as_tensor([c * sit.K + v for c, v in zip(cells, vals)], device=dev)
    Mt = sit.M[:, cols]                                                                # (n, t)
    tot = Mt.sum(1, keepdim=True)
    others = (tot - Mt) == (len(cells) - 1)                                            # matches e on all the others
    if keep is not None:
        others &= torch.as_tensor(keep, device=dev).bool()[:, None]
    n_eq = (others & (Mt == 1)).sum(0)
    n_ne = (others & (Mt == 0)).sum(0)
    return ((n_eq >= N_MIN) & (n_ne >= N_MIN)).tolist()


def score_sets(C, tok_of, tc, judged, ties=True):
    """Scoring only. Each kept fact is one stated condition: its cell, or its tie set's cells (ties=True), mapped to
    world tokens by tok_of (None: not counted, e.g. outside the grid or the event's own token). A condition is right
    if a member is a true condition, wrong if no member is and some member is judged, unscored otherwise.
    Returns (right, wrong, found): found marks the true conditions covered."""
    sets = set()
    for f in C:
        toks = frozenset(t for t in (map(tok_of, f[7] if ties else [f[0]])) if t is not None)
        if toks:
            sets.add(toks)
    true = tc & judged
    found = np.zeros_like(tc)
    right = wrong = 0
    for s in sets:
        s = list(s)
        if true[s].any():
            right += 1
            found[[t for t in s if true[t]]] = True
        elif judged[s].any():
            wrong += 1
    return right, wrong, found


@torch.no_grad()
def contrastable_sig(sit, cells, vals, keep=None, level=ALPHA / 50):
    """Scoring only (111.2). For each true condition (cell, e's value): among remembered cases that match e on every
    other true condition, does e's value raise success with a one-sided Fisher p < level? Then memory holds the
    evidence for it."""
    if not len(cells):
        return []
    cols = torch.as_tensor([c * sit.K + v for c, v in zip(cells, vals)], device=dev)
    Mt = sit.M[:, cols].bool()
    tot = Mt.sum(1, keepdim=True)
    others = (tot - Mt.long()) == (len(cells) - 1)
    if keep is not None:
        others &= torch.as_tensor(keep, device=dev).bool()[:, None]
    y = sit.y.bool()[:, None]
    N = others.sum(0).double(); S = (others & y).sum(0).double()
    n = (others & Mt).sum(0).double(); k = (others & Mt & y).sum(0).double()
    lp = fisher_upper(k, N, S, n)
    raises = (k / n.clamp(min=1)) > ((S - k) / (N - n).clamp(min=1))
    return ((lp < math.log(level)) & raises & (n > 0) & (N - n > 0)).tolist()
