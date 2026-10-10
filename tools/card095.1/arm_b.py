"""Card 095.1, arm B: exemplars. Every token of every stored try is an instance; nothing is trained.

An instance: the action, the token, its place (relative to the agent; the hand is place 169), and the tokens at the
places around the agent (tok.py's near places, within 2 steps, and the hand: 13 places), with whether the token
changed and what it became. Per action and world (its own training episodes, as recall uses its world's memory):

- **which tokens change:** a vote over instances, each weighted by exp(-sum over admitted conditions of lambda x
  distance) (card 049). Candidates: the token's own parts (4: L1 per part of its vector), its place (same or not),
  and the parts of the token at each of the 13 places (52). Admitted greedily while a candidate raises the
  leave-one-try-out log-likelihood of the training instances by more than log(number of candidates). Instances
  equal on the admitted conditions' inputs are grouped with summed counts (card 051's index), which keeps tier 3's
  1.2 million distinct instances per action tractable.
- **what it becomes:** per part, the relation most supported by the weighted changed instances: kept, copied from
  the token at place r, shifted by a stored change, or set to a stored part (card 038's four ways, chosen here by
  support among the neighbours rather than leave-one-pair-out). Applied to the query's own tokens; snapped to the
  nearest known token.

  bin/prun python tools/card095.1/arm_b.py tier2        → runs/095.1/b_tier2.npz, runs/095.1/b_tier2.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
import tok                                             # noqa: E402

dev = "cuda"
NP, PW = 4, 8
GRID = (0.5, 1, 2, 4, 8, 16)                           # lambda, in units of 1 / the condition's mean distance
ALPHA = 1.0
QMAX = 4096
MAX_ADMIT = 10
TOPK = 64
CHUNK = 16384


def fields(H, P, near_n):
    """Per token slot: [own handle, place, the handles at the 13 context places] (absent: -1)."""
    ctx = np.concatenate([H[:, :near_n], H[:, -1:]], 1)
    r, c = np.nonzero(H >= 0)
    F = np.concatenate([H[r, c][:, None], P[r, c][:, None], ctx[r]], 1)
    return F, r, c


class Arm:
    def __init__(self, arr, F, nc, n, tc, tu):
        """F (m, 15) instance fields (distinct); nc / n weighted changed / all counts; tc / tu changed / unchanged
        tries (unweighted, for leaving one try out with its own weight)."""
        self.F, self.nc, self.n, self.tc, self.tu = F, nc, n, tc, tu
        V = np.concatenate([arr, np.zeros((1, 32), np.float32)])       # row -1: absent
        self.V = V
        self.cand = [("own", k) for k in range(NP)] + [("place",)] + \
                    [("at", r, k) for r in range(F.shape[1] - 2) for k in range(NP)]
        # mean distance per candidate over random instance pairs (its scale)
        rng = np.random.default_rng(0)
        a, b = rng.integers(len(F), size=(2, 20000))
        self.scale = np.array([max(self.dist_np(c, F[a], F[b]).mean(), 1e-3) for c in self.cand])
        self.f = (nc.sum() + 1) / (n.sum() + 2)

    def col(self, c):
        return 0 if c[0] == "own" else 1 if c[0] == "place" else 2 + c[1]

    def dist_np(self, c, Fa, Fb):
        j = self.col(c)
        if c[0] == "place":
            return (Fa[:, j] != Fb[:, j]).astype(np.float32)
        k = c[-1]
        return np.abs(self.V[Fa[:, j]][:, k * PW:(k + 1) * PW] - self.V[Fb[:, j]][:, k * PW:(k + 1) * PW]).sum(1)

    def dist_t(self, c, Fq, Fk):
        """(|q|, |k|) distance matrix on the GPU."""
        j = self.col(c)
        if c[0] == "place":
            return (Fq[:, j][:, None] != Fk[:, j][None]).float()
        k = c[-1]
        Vt = self.Vt[:, k * PW:(k + 1) * PW]
        return torch.cdist(Vt[Fq[:, j]], Vt[Fk[:, j]], p=1)

    def groups(self, cols):
        """Instances grouped on the given field columns: (group fields, nc, n, nt)."""
        if not cols:
            return (self.F[:1], self.nc.sum(keepdims=True), self.n.sum(keepdims=True), self.tc.sum(keepdims=True),
                    self.tu.sum(keepdims=True))
        u, inv = np.unique(self.F[:, sorted(set(cols))], axis=0, return_inverse=True)
        inv = inv.ravel()
        G = np.zeros((len(u), self.F.shape[1]), np.int64)
        G[:, sorted(set(cols))] = u
        s = lambda x: np.bincount(inv, weights=x, minlength=len(u))
        return G, s(self.nc), s(self.n), s(self.tc), s(self.tu)

    def ll(self, adm, lam, extra=None, grid=(None,)):
        """Leave-one-try-out log-likelihood of the training instances (summed over tries), grouped on the
        admitted conditions' inputs (and extra's); one value per lambda in grid for extra."""
        conds = adm + ([extra] if extra else [])
        key = tuple(sorted({self.col(c) for c in conds}))
        g = self._gc.get(key)
        if g is None:
            G, nc, n, tc, tu = self.groups(list(key))
            if len(G) > QMAX:
                qi = np.unique(np.random.default_rng(1).choice(len(G), QMAX, p=n / n.sum()))
                scale = n.sum() / n[qi].sum()
            else:
                qi, scale = np.arange(len(G)), 1.0
            g = self._gc[key] = (G, nc, n, tc, tu, qi, scale)
        G, nc, n, tc, tu, qi, scale = g
        Gt = torch.as_tensor(G, device=dev)
        q = torch.as_tensor(qi, device=dev)
        nct, nt_ = torch.as_tensor(nc, device=dev).double(), torch.as_tensor(n, device=dev).double()
        tc_, tu_ = torch.as_tensor(tc, device=dev).double()[q], torch.as_tensor(tu, device=dev).double()[q]
        fq, nq = nct[q], nt_[q]
        wc = torch.where(tc_ > 0, fq / tc_.clamp_min(1), torch.zeros_like(fq))          # a changed try's weight
        wu = torch.where(tu_ > 0, (nq - fq) / tu_.clamp_min(1), torch.zeros_like(fq))   # an unchanged try's
        Nc = torch.zeros(len(grid), len(qi), device=dev, dtype=torch.float64)
        N = torch.zeros_like(Nc)
        for k0 in range(0, len(G), CHUNK):             # keys in chunks (GPU memory)
            Gk = Gt[k0:k0 + CHUNK]
            D = torch.zeros(len(qi), len(Gk), device=dev)
            for c, l in zip(adm, lam):
                D += (l / self.scale[self.cand.index(c)]) * self.dist_t(c, Gt[q], Gk)
            dx = self.dist_t(extra, Gt[q], Gk) / self.scale[self.cand.index(extra)] if extra else None
            for i, gl in enumerate(grid):
                K = torch.exp(-(D if dx is None else D + gl * dx)).double()
                Nc[i] += K @ nct[k0:k0 + CHUNK]
                N[i] += K @ nt_[k0:k0 + CHUNK]
        out = []
        for i in range(len(grid)):
            pc_ch = ((Nc[i] - wc + ALPHA * self.f) / (N[i] - wc + ALPHA)).clamp(1e-9, 1 - 1e-9)   # own try left out
            pc_un = ((Nc[i] + ALPHA * self.f) / (N[i] - wu + ALPHA)).clamp(1e-9, 1 - 1e-9)
            l = fq * torch.log(pc_ch) + (nq - fq) * torch.log(1 - pc_un)               # weighted tries
            out.append(float(l.sum()) * scale)
        return out

    def admit(self, log):
        self.Vt = torch.as_tensor(self.V, device=dev)
        self._gc = {}
        adm, lam = [], []
        cur = self.ll(adm, lam)[0]
        trace = []
        while len(adm) < MAX_ADMIT:
            best = None
            for c in self.cand:
                if c in adm:
                    continue
                for v, g in zip(self.ll(adm, lam, c, GRID), GRID):
                    if best is None or v > best[0]:
                        best = (v, c, g)
            gain = best[0] - cur
            log(f"   best {best[1]} lambda {best[2]} gain {gain:.2f}")
            if gain <= math.log(len(self.cand)):
                break
            adm.append(best[1])
            lam.append(best[2])
            cur = best[0]
            trace.append({"cond": str(best[1]), "lambda": best[2], "gain": round(gain, 2)})
        self.adm, self.lam = adm, lam
        G, nc, n, _, _ = self.groups([self.col(c) for c in adm])
        self.G, self.Gnc, self.Gn = G, nc, n
        return trace

    def kernel(self, Fq, k0=0, k1=None):
        """Kernel weights of queries Fq over the groups k0:k1."""
        Gt, Qt = torch.as_tensor(self.G[k0:k1], device=dev), torch.as_tensor(Fq, device=dev)
        D = torch.zeros(len(Fq), len(Gt), device=dev)
        for c, l in zip(self.adm, self.lam):
            D += (l / self.scale[self.cand.index(c)]) * self.dist_t(c, Qt, Gt)
        return torch.exp(-D).double()

    def p_change(self, Fq, bs=1024, totals=False):
        """P(change) for query fields Fq; with totals, also the kernel-weighted changed and all counts."""
        out, tot = [], []
        nc, n = torch.as_tensor(self.Gnc, device=dev), torch.as_tensor(self.Gn, device=dev)
        for s in range(0, len(Fq), bs):
            Nc = torch.zeros(len(Fq[s:s + bs]), device=dev, dtype=torch.float64)
            N = torch.zeros_like(Nc)
            for k0 in range(0, len(self.G), CHUNK):
                K = self.kernel(Fq[s:s + bs], k0, k0 + CHUNK)
                Nc += K @ nc[k0:k0 + CHUNK]
                N += K @ n[k0:k0 + CHUNK]
            out.append(((Nc + ALPHA * self.f) / (N + ALPHA)).cpu().numpy())
            tot.append(torch.stack([Nc, N], 1).cpu().numpy())
        p = np.concatenate(out) if out else np.zeros(0)
        return (p, np.concatenate(tot) if tot else np.zeros((0, 2))) if totals else p


def results(arm, Fq, Fch, wch, Ach, known, arr):
    """What each query token becomes: per part, the relation most supported by the kernel-weighted changed
    training instances (Fch fields, wch weights, Ach handles after)."""
    V = arm.V
    Kq = []
    gi = {tuple(r): i for i, r in enumerate(arm.G[:, sorted({arm.col(c) for c in arm.adm})].tolist())} if arm.adm else None
    cols = sorted({arm.col(c) for c in arm.adm})
    gof = np.array([gi[tuple(r)] for r in Fch[:, cols].tolist()]) if arm.adm else np.zeros(len(Fch), np.int64)
    Kt = known
    out = np.full(len(Fq), -1)
    ways = np.zeros((len(Fq), NP), np.int64)           # 0 keep, 1 copy, 2 shift, 3 set (report)
    src = np.full((len(Fq), NP), -1)                   # copied from context place index
    for s in range(0, len(Fq), 256):
        K = np.concatenate([arm.kernel(Fq[s:s + 256], k0, k0 + CHUNK).cpu().numpy()
                            for k0 in range(0, len(arm.G), CHUNK)], 1)   # (q, groups)
        for qi in range(K.shape[0]):
            wk = K[qi][gof] * wch                       # weight of each changed instance
            top = np.argsort(-wk)[:TOPK]
            top = top[wk[top] > 0]
            if not len(top):
                continue
            q = Fq[s + qi]
            after = arr[Ach[top]]
            res = np.zeros(32, np.float32)
            for k in range(NP):
                sl = slice(k * PW, (k + 1) * PW)
                a, own = after[:, sl], V[Fch[top, 0]][:, sl]
                score = {}
                score[("keep",)] = wk[top][np.abs(a - own).sum(1) < 1e-5].sum()
                for r in range(Fch.shape[1] - 2):
                    cr = V[Fch[top, 2 + r]][:, sl]
                    score[("copy", r)] = wk[top][(np.abs(a - cr).sum(1) < 1e-5) & (Fch[top, 2 + r] >= 0)].sum()
                d = a - own
                for vals, kind in ((d, "shift"), (a, "set")):
                    u, inv = np.unique(np.round(vals, 5), axis=0, return_inverse=True)
                    sw = np.bincount(inv.ravel(), weights=wk[top], minlength=len(u))
                    j = int(sw.argmax())
                    score[(kind, j)] = sw[j]
                    if kind == "shift":
                        best_shift = u[j]
                    else:
                        best_set = u[j]
                way = max(score, key=score.get)
                if way[0] == "keep":
                    res[sl] = V[q[0]][sl]; ways[s + qi, k] = 0
                elif way[0] == "copy":
                    res[sl] = V[q[2 + way[1]]][sl]; ways[s + qi, k] = 1; src[s + qi, k] = way[1]
                elif way[0] == "shift":
                    res[sl] = V[q[0]][sl] + best_shift; ways[s + qi, k] = 2
                else:
                    res[sl] = best_set; ways[s + qi, k] = 3
            out[s + qi] = known[np.abs(arr[known] - res).sum(1).argmin()]
    return out, ways, src


if __name__ == "__main__":
    world = sys.argv[1]
    t0 = time.monotonic()
    log = lambda m: print(m, flush=True)
    z = tok.load(world)
    H, P, C, A = tok.build(z)
    tr, te = tok.split(z)
    near_n = int((np.abs(z["wh"][:169]).sum(1) <= tok.DMAX).sum()) - 1
    F, r, c = fields(H, P, near_n)
    ch, af, w = C[r, c], A[r, c], z["w"][r].astype(np.float64)
    act, train = z["act"][r], tr[r]
    arr, known = z["arr"], np.unique(z["app"])
    pc = np.zeros(len(F))
    tot = np.zeros((len(F), 2))                        # kernel-weighted changed and all counts (arm C's evidence)
    after = np.full(len(F), -1)
    # arm A's held-out predictions (arm C's prior), per instance
    za = np.load(tok.RUNS / f"a_{world}.npz")
    pa = np.zeros(len(F))
    te_idx = np.flatnonzero(te)
    m = ~train
    pa[m] = za["p"][np.searchsorted(te_idx, r[m]), c[m]]
    loo = {k: [] for k in ("r", "c", "nc", "n", "ch", "w")}
    WH = z["wh"][:169]
    order = [p for p in np.lexsort((np.arange(169), np.abs(WH).sum(1))).tolist() if p != int(z["centre"])]
    ctx_places = order[:near_n] + [tok.HAND]           # context index -> place (71: the front; 169: the hand)
    rep = {"world": world, "context_places": ctx_places}
    W = {}
    for a in (3, 4, 5):
        s = np.flatnonzero(train & (act == a))
        u, inv = np.unique(F[s], axis=0, return_inverse=True)
        inv = inv.ravel()
        b = lambda x: np.bincount(inv, weights=x, minlength=len(u))
        arm = Arm(arr, u, b(w[s] * ch[s]), b(w[s]), b(ch[s].astype(float)), b((~ch[s]).astype(float)))
        log(f"{world} action {a}: training instances {len(s)}, distinct {len(u)}")
        trace = arm.admit(log)
        log(f"   admitted {[str(x) for x in arm.adm]} groups {len(arm.G)} ({round(time.monotonic() - t0, 1)} s)")
        q = np.flatnonzero(~train & (act == a))
        uq, qinv = np.unique(F[q], axis=0, return_inverse=True)
        pq, tq = arm.p_change(uq, totals=True)
        pc[q], tot[q] = pq[qinv.ravel()], tq[qinv.ravel()]
        # arm C: training instances' totals with their own try left out (to fit the prior's weight)
        smp = np.random.default_rng(a).choice(s, min(20000, len(s)), replace=False, p=w[s] / w[s].sum())
        us, sinv = np.unique(F[smp], axis=0, return_inverse=True)
        _, ts = arm.p_change(us, totals=True)
        ts = ts[sinv.ravel()]
        loo["r"].append(r[smp]), loo["c"].append(c[smp]), loo["ch"].append(ch[smp]), loo["w"].append(w[smp])
        loo["nc"].append(ts[:, 0] - w[smp] * ch[smp]), loo["n"].append(ts[:, 1] - w[smp])
        # results, for held-out tokens that changed or that arm B or arm A predicts to change
        need = q[ch[q] | (pc[q] > 0.5) | (pa[q] > 0.5)]
        un, ninv = np.unique(F[need], axis=0, return_inverse=True)
        sc = s[ch[s]]
        res, ways, src = results(arm, un, F[sc], w[sc], af[sc], known, arr)
        after[need] = res[ninv.ravel()]
        hand_pick = (un[:, 1] == tok.HAND)
        rep[str(a)] = {"admitted": trace, "groups": int(len(arm.G)), "distinct_instances": int(len(u)),
                       "ways_on_hand_tokens": np.bincount(ways[hand_pick].ravel(), minlength=4).tolist(),
                       "copy_source_on_hand_tokens": {str(k): int(v) for k, v in zip(*np.unique(src[hand_pick], return_counts=True))}}
        log(f"   results for {len(un)} distinct held-out tokens ({round(time.monotonic() - t0, 1)} s)")
    sel = ~train
    np.savez_compressed(tok.RUNS / f"b_{world}.npz", r=r[sel], c=c[sel], p=pc[sel], after=after[sel],
                        nc=tot[sel, 0], n=tot[sel, 1], f_alpha=ALPHA)
    np.savez_compressed(tok.RUNS / f"b_loo_{world}.npz", **{k: np.concatenate(v) for k, v in loo.items()},
                        act=np.concatenate([np.full(min(20000, int((train & (act == a)).sum())), a) for a in (3, 4, 5)]))
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (tok.RUNS / f"b_{world}.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(f"{world} saved {rep['seconds']} s")
