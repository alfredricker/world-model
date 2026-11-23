"""Card 160: the transition energy of a token (card 158's form), the same with a multilayer perceptron, and the
statements read from either.

  token (a cell)   p(next code | code c, action, where the agent is relative to the token, tokens out of view)
                   = softmax over the next codes seen for c of  b[c] + the sum of one coupling per input token.
                   As in card 158 the couplings are a set shared by all codes, toward any change of code, plus each
                   code's own sparse difference; L1 on both, in nats over the whole experience (card 158: 1 nat)
  the agent        p(move and heading | heading, action, the codes of the eight cells around it, tokens out of view),
                   the same form, each heading fitted by itself
A token out of view enters twice: by its code with the action at which it went, and by that action alone.
The perceptron (the ordinary comparison): one network for all codes, the token's code among its inputs.
"""
import numpy as np, torch
from collections import defaultdict
from grid import NA, NG, FAR, OFF8, UNK, OUT, gid_inv

SEEN = 3
dev = "cuda" if torch.cuda.is_available() else "cpu"


def fit_tokens(idx, own, Y, mask, stay, F, kind, l1, seed=0, log=None):
    """idx [n, L] input numbers padded with F; own [n] the token's own state (a cell's code, the agent's heading); Y [n, O]
    counts of the next state; mask [K, O] the next states ever seen from each own state; stay [K] the next state that
    means no change.  The no-change state is the reference (logit 0)."""
    K, O = mask.shape; g = torch.Generator(device="cpu").manual_seed(seed); n = len(idx)
    X = torch.zeros(n, F + 1, device=dev); X.scatter_add_(1, torch.as_tensor(idx, device=dev), torch.ones(idx.shape, device=dev)); X = X[:, :F]
    own_t = torch.as_tensor(own, device=dev); Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev); groups = []
    for k in range(K):
        outs = np.nonzero(mask[k])[0]
        if len(outs) > 1 and (own == k).any():
            r = torch.nonzero(own_t == k)[:, 0]; ch = torch.as_tensor(outs != stay[k], dtype=torch.float32, device=dev)
            groups.append((k, outs, X[r], Yt[r][:, outs], ch))
    if kind == "sparse":
        dW = [torch.zeros(F, len(o), device=dev, requires_grad=True) for _, o, _, _, _ in groups]
        b = [(torch.log((y.sum(0) + 0.5) / (y[:, ch == 0].sum() + 0.5)) * ch).clone().requires_grad_(True) for _, _, _, y, ch in groups]
        W0 = torch.zeros(F, device=dev, requires_grad=True); P = dW + [W0] + b; m = [torch.zeros_like(p) for p in P]; v = [torch.zeros_like(p) for p in P]; steps = 3000
        for t in range(1, steps + 1):
            sh = None; loss = 0
            for (k, outs, x, y, ch), w, bi in zip(groups, dW, b):
                loss = loss - (y * torch.log_softmax((x @ w + bi + (x @ W0)[:, None]) * ch, -1)).sum()
            gr = torch.autograd.grad(loss, P); lr = 0.1 if t < 0.5 * steps else 0.02 if t < 0.85 * steps else 0.004
            with torch.no_grad():
                for i, (p, gi, mi, vi) in enumerate(zip(P, gr, m, v)):
                    mi.mul_(0.9).add_(gi, alpha=0.1); vi.mul_(0.99).addcmul_(gi, gi, value=0.01); den = (vi / (1 - 0.99 ** t)).sqrt() + 1e-8
                    p.sub_(lr * (mi / (1 - 0.9 ** t)) / den)
                    if i <= len(dW):                                   # the couplings, shared and own, pay l1 nats each; the biases do not
                        p.copy_(torch.sign(p) * torch.clamp(p.abs() - lr * l1 / den, min=0))
            if log is not None and t % 500 == 0:
                log.append(round(float(loss.detach()) + l1 * float((sum(p.abs().sum() for p in dW) + W0.abs().sum()).detach()), 1))
        out = dict(b=np.zeros((K, O)), dW=np.zeros((K, F + 1, O)), W0=np.append(W0.detach().cpu().numpy().astype(np.float64), 0.0), nll=float(loss.detach()))
        for (k, outs, x, y, ch), w, bi in zip(groups, dW, b):
            c = ch.cpu().numpy(); out["b"][k, outs] = bi.detach().cpu().numpy() * c; out["dW"][k][:F, outs] = w.detach().cpu().numpy() * c
        return out
    Hd = 128; E = (torch.randn(F, Hd, generator=g) * 0.1).to(dev).requires_grad_(True); Eo = (torch.randn(K, Hd, generator=g) * 0.1).to(dev).requires_grad_(True)
    b1 = torch.zeros(Hd, device=dev, requires_grad=True); V = (torch.randn(Hd, O, generator=g) * 0.1).to(dev).requires_grad_(True); bo = torch.zeros(O, device=dev, requires_grad=True)
    opt = torch.optim.Adam([E, Eo, b1, V, bo], lr=0.003); neg = (1 - torch.as_tensor(mask, dtype=torch.float32, device=dev)) * -1e9; tot = Yt.sum()
    for t in range(1, 6001):
        loss = -(Yt * torch.log_softmax(torch.relu(X @ E + Eo[own_t] + b1) @ V + bo + neg[own_t], -1)).sum()
        opt.zero_grad(); (loss / tot * 1000).backward(); opt.step()
        if log is not None and t % 1000 == 0:
            log.append(round(float(loss.detach()), 1))
    cp = lambda z: z.detach().cpu().numpy().astype(np.float64)
    return dict(E=np.vstack([cp(E), np.zeros((1, Hd))]), Eo=cp(Eo), b1=cp(b1), V=cp(V), bo=cp(bo), nll=float(loss.detach()))


def tok_logits(w, k, idx, stay_k):
    """idx [..., L] padded input numbers of a token whose own state is k -> [..., O]"""
    if "dW" in w:
        ch = np.ones(w["b"].shape[1]); ch[stay_k] = 0
        return w["b"][k] + w["dW"][k][idx].sum(-2) + w["W0"][idx].sum(-1)[..., None] * ch
    return np.maximum(w["E"][idx].sum(-2) + w["Eo"][k] + w["b1"], 0) @ w["V"] + w["bo"]


class Energy:
    def __init__(s, T, kind="sparse", l1=1.0, seed=0):
        s.kind = kind; s.codes = T.codes; s.NC = NC = len(T.codes.triples); sets = T.item_sets(); s.sets = sets
        s.items = sorted(set().union(*sets)); s.item_ix = {it: i for i, it in enumerate(s.items)}
        s.FG = NA; s.FI = NA + NG + 1; s.FY = s.FI + len(s.items); s.F = s.FY + NA                    # a token's inputs: action, place of the agent, tokens out of view, their actions
        s.AI = NA + 8 * NC; s.AY = s.AI + len(s.items); s.AF = s.AY + NA                                # the agent's inputs: action, eight neighbours' codes, the same memory
        s.cell, s.sit, s.log = {}, {}, {}; s._st, s._p = {}, {}
        c, a, g, it, o, n = T.rows(); L = 2 + max(len(x) for x in sets) * 2
        setf = [s.mem_feats(x, s.FI, s.FY) for x in sets]
        sk, inv = np.unique(((c * 8 + a) * 128 + g) * (1 << 20) + it, return_inverse=True); Y = np.zeros((len(sk), NC)); np.add.at(Y, (inv, o), n)
        si = sk % (1 << 20); q = sk // (1 << 20); sg = q % 128; q = q // 128; sa = q % 8; sc = q // 8
        idx = np.full((len(sk), L), s.F); idx[:, 0] = sa; idx[:, 1] = s.FG + sg
        for r, i in enumerate(si):
            idx[r, 2:2 + len(setf[i])] = setf[i]
        mask = np.eye(NC, dtype=bool)
        for ci in np.unique(sc):
            mask[ci] |= Y[sc == ci].sum(0) > 0
        s.log["cells"] = []; s.w = fit_tokens(idx, sc, Y, mask, np.arange(NC), s.F, kind, l1, seed, s.log["cells"]); s.mask = mask
        for ci in np.unique(sc):
            m = sc == ci; outs = np.nonzero(mask[ci])[0].tolist(); s.sit[int(ci)] = (sa[m], sg[m], si[m], Y[m][:, outs], outs, None)
            if len(outs) > 1:
                s.cell[int(ci)] = outs
        s.aouts = [(mv, d2) for mv in range(5) for d2 in range(4)]; setA = [s.mem_feats(x, s.AI, s.AY) for x in sets]; sits = {}; s.move_actions = set()
        for (d, a_, nb, it_, out), cnt in T.agent.items():
            sits.setdefault((d, a_, nb, it_), np.zeros(20))[out[0] * 4 + out[1]] += cnt
            if out[0] != 0:
                s.move_actions.add(a_)
        s.move_actions = sorted(s.move_actions); idx = np.full((len(sits), 9 + L), s.AF); own = np.zeros(len(sits), int); Y = np.zeros((len(sits), 20))
        for r, ((d, a_, nb, it_), y) in enumerate(sits.items()):
            f = [a_] + [NA + j * NC + k for j, k in enumerate(nb)] + setA[it_]; idx[r, :len(f)] = f; own[r] = d; Y[r] = y
        s.amask = np.zeros((4, 20), bool); s.amask[np.arange(4), np.arange(4)] = True
        for d in range(4):
            s.amask[d] |= Y[own == d].sum(0) > 0
        s.log["agent"] = []; s.wa = fit_tokens(idx, own, Y, s.amask, np.arange(4), s.AF, kind, l1, seed, s.log["agent"]); s.nll = s.w["nll"] + s.wa["nll"]

    def mem_feats(s, items, fi, fy):
        return [fi + s.item_ix[x] for x in sorted(items) if x in s.item_ix] + [fy + a for a in sorted({x[1] for x in items})]

    # ---- predictions
    def cell_logits(s, c, feats):
        """over the next codes seen for c"""
        return tok_logits(s.w, c, np.asarray(feats, dtype=int), c)[s.cell[c]]

    def p_cell(s, c, a, g, items):
        """{next code: probability}"""
        key = (c, a, g, items)
        if key not in s._p:
            if c not in s.cell:
                s._p[key] = {c: 1.0}
            else:
                lg = s.cell_logits(c, [a, s.FG + g] + s.mem_feats(items, s.FI, s.FY)); p = np.exp(lg - lg.max()); p /= p.sum()
                s._p[key] = dict(zip(s.cell[c], p.tolist()))
        return s._p[key]

    def cell(s, c, a, g, items):
        p = s.p_cell(c, a, g, items)
        return max(p, key=p.get)

    def agent_pred(s, d, neigh_rows, items):
        """for each row of eight neighbour codes and each action: the most probable (move, heading)"""
        nb = np.asarray(neigh_rows); n = len(nb); mf = s.mem_feats(items, s.AI, s.AY)
        f = np.where(nb < s.NC, NA + np.arange(8) * s.NC + nb, s.AF); idx = np.full((n, NA, 9 + len(mf)), s.AF)
        idx[:, :, 0] = np.arange(NA); idx[:, :, 1:9] = f[:, None]; idx[:, :, 9:] = mf
        best = np.where(s.amask[d], tok_logits(s.wa, d, idx, d), -1e9).argmax(-1)
        return np.stack([best // 4, best % 4], -1)

    def agent_one(s, d, a, neigh, items):
        return tuple(int(x) for x in s.agent_pred(d, [neigh], items)[0, a])

    # ---- statements (card 158's read-out)
    def statement(s, c, o, a_req=None):
        key = (c, o, a_req)
        if key not in s._st:
            s._st[key] = s._statement(c, o, a_req)
        return s._st[key]

    def p_raw(s, c, o, a, g, mem):
        lg = s.cell_logits(c, [a, s.FG + g] + sorted(mem)); e = np.exp(lg - lg.max())
        return e[s.cell[c].index(o)] / e.sum()

    def _statement(s, c, o, a_req):
        """Start from the commonest experienced success of the effect.  Its places: those the model allows (the couplings
        to places are shared by all codes).  A token out of view is a condition if the model says the effect fails without it (needed) or
        with it (to be avoided) and the experience holds tries of the effect in that case."""
        if c not in s.cell or o not in s.cell[c] or o == c:
            return None
        outs = s.cell[c]; io = outs.index(o); sa, sg, si, Y, _, _ = s.sit[c]
        p = lambda a, g, items: s.p_cell(c, a, g, items).get(o, 0.0)
        cand = [(Y[r, io], r) for r in np.nonzero(Y[:, io] >= SEEN)[0] if (a_req is None or sa[r] == a_req) and p(int(sa[r]), int(sg[r]), s.sets[si[r]]) >= 0.5]
        if not cand:
            return None
        n, r = max(cand); a, g, items = int(sa[r]), int(sg[r]), s.sets[si[r]]; ma = sa == a
        G = [g2 for g2 in range(NG + 1) if p(a, g2, items) >= 0.5]
        att = ma & np.isin(sg, G); tries = Y[att].sum(1); isets = [s.sets[i] for i in si[att]]
        seen = lambda pred: sum(t for t, x in zip(tries, isets) if pred(x)) >= SEEN
        need, avoid = [], []; base = set(s.mem_feats(items, s.FI, s.FY))
        for it in sorted(items):
            if p(a, g, items - {it}) < 0.5 and seen(lambda x: it not in x):
                other = s.p_raw(c, o, a, g, set(s.mem_feats(items - {it}, s.FI, s.FY)) | {s.FY + it[1]})      # anything else gone by the same action instead?
                need.append(("any", it[1]) if other >= 0.5 and seen(lambda x: it not in x and any(q[1] == it[1] for q in x)) else ("it", it))
        for y in range(NA):
            if not any(q[1] == y for q in items) and s.p_raw(c, o, a, g, base | {s.FY + y}) < 0.5 and seen(lambda x: any(q[1] == y for q in x)):
                avoid.append(("any", y))
        for it in s.items:
            if it not in items and ("any", it[1]) not in avoid and p(a, g, items | {it}) < 0.5 and seen(lambda x: it in x):
                avoid.append(("it", it))
        return dict(a=a, G=G, need=need, avoid=avoid, n=int(n))

    def outcomes(s, c, a_req=None):
        if c not in s.cell:
            return []
        return [o for o in s.cell[c] if o != c and s.statement(c, o, a_req) is not None]
