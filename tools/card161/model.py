"""Card 161: one outcome, "nothing changes", shared by every token.

Card 160's perceptron gave each code its own output for staying as it is, so "this action does nothing here" was learned
code by code.  Here the network has one output for no change, used by every token (a cell of any code, the agent at any
heading); the outputs for changes are as before.  Importing this module adds the kind "shared" to card 160's code.

GUARD says what evidence a condition on a token out of view needs before it is stated:
  code   at least 3 tries of this code's effect in that case (card 160)
  pool   at least 3 tries, by the same action, of any code's effect in that case
  none   the model's answer alone
"""
import os, numpy as np, torch
import energy
from energy import SEEN, dev
from grid import NA, NG

GUARD = os.environ.get("GUARD", "code")
_fit0, _logits0 = energy.fit_tokens, energy.tok_logits


def fit_tokens(idx, own, Y, mask, stay, F, kind, l1, seed=0, log=None):
    if kind != "shared":
        return _fit0(idx, own, Y, mask, stay, F, kind, l1, seed, log)
    K, O = mask.shape; g = torch.Generator(device="cpu").manual_seed(seed); n = len(idx); Hd = 128
    X = torch.zeros(n, F + 1, device=dev); X.scatter_add_(1, torch.as_tensor(idx, device=dev), torch.ones(idx.shape, device=dev)); X = X[:, :F]
    own_t = torch.as_tensor(own, device=dev); Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    st = torch.as_tensor(np.asarray(stay)[own], device=dev)[:, None]
    E = (torch.randn(F, Hd, generator=g) * 0.1).to(dev).requires_grad_(True); Eo = (torch.randn(K, Hd, generator=g) * 0.1).to(dev).requires_grad_(True)
    b1 = torch.zeros(Hd, device=dev, requires_grad=True); V = (torch.randn(Hd, O + 1, generator=g) * 0.1).to(dev).requires_grad_(True); bo = torch.zeros(O + 1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([E, Eo, b1, V, bo], lr=0.003); neg = (1 - torch.as_tensor(mask, dtype=torch.float32, device=dev)) * -1e9; tot = Yt.sum()
    for t in range(1, 6001):
        z = torch.relu(X @ E + Eo[own_t] + b1) @ V + bo; lg = z[:, :O].scatter(1, st, z[:, O:])          # the token's own state takes the shared output
        loss = -(Yt * torch.log_softmax(lg + neg[own_t], -1)).sum()
        opt.zero_grad(); (loss / tot * 1000).backward(); opt.step()
        if log is not None and t % 1000 == 0:
            log.append(round(float(loss.detach()), 1))
    cp = lambda z: z.detach().cpu().numpy().astype(np.float64)
    return dict(E=np.vstack([cp(E), np.zeros((1, Hd))]), Eo=cp(Eo), b1=cp(b1), Vs=cp(V), bo=cp(bo), nll=float(loss.detach()))


def tok_logits(w, k, idx, stay_k):
    if "Vs" not in w:
        return _logits0(w, k, idx, stay_k)
    z = np.maximum(w["E"][idx].sum(-2) + w["Eo"][k] + w["b1"], 0) @ w["Vs"] + w["bo"]; lg = z[..., :-1].copy(); lg[..., stay_k] = z[..., -1]
    return lg


def pooled(s):
    """tries of any code's effect by an action, at the places that effect allows: (action, token sets) with their counts"""
    if not hasattr(s, "_pool"):
        s._pool = {a: [] for a in range(NA)}
        for c in s.cell:
            for a in range(NA):
                for o in s.cell[c]:
                    st = s.statement_places(c, o, a) if o != c else None
                    if st is None:
                        continue
                    sa, sg, si, Y, _, _ = s.sit[c]; att = (sa == a) & np.isin(sg, st)
                    s._pool[a] += [(float(t), s.sets[i]) for t, i in zip(Y[att].sum(1), si[att])]
                    break
    return s._pool


def statement_places(s, c, o, a_req):
    """the places the model allows for the commonest experienced success of the effect, or None"""
    outs = s.cell[c]; io = outs.index(o); sa, sg, si, Y, _, _ = s.sit[c]
    p = lambda a, g, items: s.p_cell(c, a, g, items).get(o, 0.0)
    cand = [(Y[r, io], r) for r in np.nonzero(Y[:, io] >= SEEN)[0] if sa[r] == a_req and p(int(sa[r]), int(sg[r]), s.sets[si[r]]) >= 0.5]
    if not cand:
        return None
    n, r = max(cand); a, items = int(sa[r]), s.sets[si[r]]
    return [g2 for g2 in range(NG + 1) if p(a, g2, items) >= 0.5]


def _statement(s, c, o, a_req):
    if c not in s.cell or o not in s.cell[c] or o == c:
        return None
    outs = s.cell[c]; io = outs.index(o); sa, sg, si, Y, _, _ = s.sit[c]
    p = lambda a, g, items: s.p_cell(c, a, g, items).get(o, 0.0)
    cand = [(Y[r, io], r) for r in np.nonzero(Y[:, io] >= SEEN)[0] if (a_req is None or sa[r] == a_req) and p(int(sa[r]), int(sg[r]), s.sets[si[r]]) >= 0.5]
    if not cand:
        return None
    n, r = max(cand); a, g, items = int(sa[r]), int(sg[r]), s.sets[si[r]]; ma = sa == a
    G = [g2 for g2 in range(NG + 1) if p(a, g2, items) >= 0.5]
    if GUARD == "none":
        seen = lambda pred: True
    elif GUARD == "pool":
        pl = pooled(s)[a]; seen = lambda pred: sum(t for t, x in pl if pred(x)) >= SEEN
    else:
        att = ma & np.isin(sg, G); tries = Y[att].sum(1); isets = [s.sets[i] for i in si[att]]
        seen = lambda pred: sum(t for t, x in zip(tries, isets) if pred(x)) >= SEEN
    need, avoid = [], []; base = set(s.mem_feats(items, s.FI, s.FY))
    for it in sorted(items):
        if p(a, g, items - {it}) < 0.5 and seen(lambda x: it not in x):
            other = s.p_raw(c, o, a, g, set(s.mem_feats(items - {it}, s.FI, s.FY)) | {s.FY + it[1]})
            need.append(("any", it[1]) if other >= 0.5 and seen(lambda x: it not in x and any(q[1] == it[1] for q in x)) else ("it", it))
    for y in range(NA):
        if not any(q[1] == y for q in items) and s.p_raw(c, o, a, g, base | {s.FY + y}) < 0.5 and seen(lambda x: any(q[1] == y for q in x)):
            avoid.append(("any", y))
    for it in s.items:
        if it not in items and ("any", it[1]) not in avoid and p(a, g, items | {it}) < 0.5 and seen(lambda x: it in x):
            avoid.append(("it", it))
    return dict(a=a, G=G, need=need, avoid=avoid, n=int(n))


energy.fit_tokens, energy.tok_logits = fit_tokens, tok_logits
energy.Energy._statement = _statement; energy.Energy.statement_places = statement_places
