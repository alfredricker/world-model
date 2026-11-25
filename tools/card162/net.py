"""Card 162: one effect network in which a neighbour is what it is and where it lies.

Card 160's perceptron took each neighbour of the agent as one input per (position, code) pair and kept separate weights
per heading, so what a code does was learned again for every position and heading.  Here every token in a situation is
two parts: what it is (an embedding of its code, one table for every use) and where it lies relative to the token asked
about (an embedding of the offset, or "far", or "out of view by action a").  Attention combines them: where a token lies,
with the asked token and the action, decides how much it counts; what it contributes is what it is, plus the bare fact
that something lies there (as card 160 gave a token out of view twice: by its code, and by the action alone).

  asked about the agent (its heading):  the eight cells around it and the tokens out of view  -> its move and next heading
  asked about a cell (its code):        the agent, where it stands, and the tokens out of view -> the cell's next code
Arm A uses the network for the agent's move only (cells: card 160's perceptron).  Arm B uses one network for both.
No rotation is built in: which offset matters for which heading is learned.
"""
import pickle, numpy as np, torch
from grid import NA, NG, FAR, OFF8, R, SIDE, gid_inv
from energy import SEEN

dev = "cuda" if torch.cuda.is_available() else "cpu"
D, H, DK, DV, HD = 32, 8, 8, 8, 128
OFFIX = lambda dx, dy: (dx + R) * SIDE + (dy + R)                       # where the other token lies, relative to the token asked about
R_FAR, R_OUT, NR = SIDE * SIDE, SIDE * SIDE + 1, SIDE * SIDE + 1 + NA


def tables(M, T, cells):
    """the experience as rows: asked token, action, the other tokens (what, where), counts of what followed"""
    NC = M.NC; AG, AFAR, PAD = NC, NC + 4, NC + 5; sets = T.item_sets(); L = 8 + max(len(x) for x in sets); rows = []
    mem = [[(k, R_OUT + a) for k, a in sorted(x) if k < NC] for x in sets]
    sit = {}
    for (d, a, nb, it, out), cnt in T.agent.items():
        sit.setdefault((d, a, nb, it), np.zeros(20))[out[0] * 4 + out[1]] += cnt
    for (d, a, nb, it), y in sit.items():
        ctx = [(k, OFFIX(*OFF8[j])) for j, k in enumerate(nb) if k < NC] + mem[it]; yy = np.zeros(NC + 20); yy[NC:] = y; rows.append((AG + d, a, ctx, yy))
    if cells:
        c, a, g, it, o, n = T.rows(); key = ((c * 8 + a) * 128 + g) * (1 << 20) + it; sk, inv = np.unique(key, return_inverse=True); Y = np.zeros((len(sk), NC + 20)); np.add.at(Y, (inv, o), n)
        si = sk % (1 << 20); q = sk // (1 << 20); sg = q % 128; q = q // 128; sa = q % 8; sc = q // 8
        for r in range(len(sk)):
            if sg[r] == FAR:
                first = (AFAR, R_FAR)
            else:
                dx, dy, d = gid_inv(int(sg[r])); first = (AG + d, OFFIX(-dx, -dy))
            rows.append((int(sc[r]), int(sa[r]), [first] + mem[si[r]], Y[r]))
    mask = np.zeros((NC + 5, NC + 20), bool); mask[:NC, :NC] = M.mask; mask[AG:AG + 4, NC:] = M.amask; mask[AFAR] = True; groups = []
    for grp in (rows[:len(sit)], rows[len(sit):]):                      # the agent's rows, the cells' rows: each padded to its own length
        if not grp:
            continue
        n = len(grp); L = max(len(r[2]) for r in grp); subj = np.array([r[0] for r in grp]); act = np.array([r[1] for r in grp]); ce = np.full((n, L), PAD); cr = np.zeros((n, L), int); cm = np.zeros((n, L), bool)
        for i, r in enumerate(grp):
            m = len(r[2])
            if m:
                ce[i, :m] = [x[0] for x in r[2]]; cr[i, :m] = [x[1] for x in r[2]]; cm[i, :m] = True
        groups.append((subj, act, ce, cr, cm, np.stack([r[3] for r in grp])))
    return groups, mask


def fit(M, T, cells, seed=0, steps=5000, log=None):
    groups, mask = tables(M, T, cells); NC = M.NC; NE = NC + 6; NO = NC + 20; g = torch.Generator(device="cpu").manual_seed(seed)
    rn = lambda *sh: (torch.randn(*sh, generator=g) * 0.1).to(dev).requires_grad_(True); z = lambda *sh: torch.zeros(*sh, device=dev, requires_grad=True)
    P = dict(Ee=rn(NE, D), Ea=rn(NA, D), Kr=rn(NR, H, DK), null=z(H), Q1=rn(2 * D, 64), q1=z(64), Q2=rn(64, H * DK), Wv=rn(D, H * DV), Ur=z(NR, H * DV),
             W1=rn(2 * D + H * DV, HD), b1=z(HD), W2=rn(HD, HD), b2=z(HD), W3=rn(HD, NO), b3=z(NO))
    t = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=dev); neg = (1 - t(mask, torch.float32)) * -1e9; oh = torch.nn.functional.one_hot
    G = [(t(subj), t(act), oh(t(ce), NE).float(), oh(t(cr), NR).float(), t(cm, torch.bool), t(Y, torch.float32)) for subj, act, ce, cr, cm, Y in groups]; tot = sum(x[5].sum() for x in G)
    opt = torch.optim.Adam(list(P.values()), lr=0.003)
    for it in range(1, steps + 1):                                       # every row at every step
        for gp in opt.param_groups:
            gp["lr"] = 0.003 if it < 0.6 * steps else 0.001 if it < 0.85 * steps else 0.0003
        loss = 0; val = P["Ee"] @ P["Wv"]
        for subj, act, ce, cr, cm, Y in G:
            n, L = cm.shape; es, ea = P["Ee"][subj], P["Ea"][act]; q = (torch.relu(torch.cat([es, ea], 1) @ P["Q1"] + P["q1"]) @ P["Q2"]).view(n, H, DK)
            att = torch.einsum("nhr,nlr->nhl", torch.einsum("nhd,rhd->nhr", q, P["Kr"]) / DK ** 0.5, cr).masked_fill(~cm[:, None, :], -1e9)
            w = torch.softmax(torch.cat([att, P["null"].expand(n, H)[..., None]], -1), -1)[..., :-1]
            ctx = torch.einsum("nhl,nlhd->nhd", w, (ce @ val + cr @ P["Ur"]).view(n, L, H, DV)).reshape(n, H * DV)
            lg = torch.relu(torch.relu(torch.cat([es, ea, ctx], 1) @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]) @ P["W3"] + P["b3"]
            loss = loss - (Y * torch.log_softmax(lg + neg[subj], -1)).sum()
        opt.zero_grad(); (loss / tot * 1000).backward(); opt.step()
        if log is not None and it % 1000 == 0:
            log.append(round(float(loss.detach()), 1))
    w = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in P.items()}; w["nll"] = float(loss.detach()); w["rows"] = sum(len(x[0]) for x in G)
    return w


def forward(w, subj, act, ce, cr, cm):
    n, L = ce.shape; es, ea = w["Ee"][subj], w["Ea"][act]; q = (np.maximum(np.concatenate([es, ea], 1) @ w["Q1"] + w["q1"], 0) @ w["Q2"]).reshape(n, H, DK)
    att = np.einsum("nhd,nlhd->nhl", q, w["Kr"][cr]) / DK ** 0.5; att = np.where(cm[:, None, :], att, -1e9); att = np.concatenate([att, np.broadcast_to(w["null"], (n, H))[..., None]], -1)
    att = np.exp(att - att.max(-1, keepdims=True)); a = (att / att.sum(-1, keepdims=True))[..., :-1]
    ctx = np.einsum("nhl,nlhd->nhd", a, (w["Ee"][ce] @ w["Wv"] + w["Ur"][cr]).reshape(n, L, H, DV)).reshape(n, H * DV)
    return np.maximum(np.maximum(np.concatenate([es, ea, ctx], 1) @ w["W1"] + w["b1"], 0) @ w["W2"] + w["b2"], 0) @ w["W3"] + w["b3"]


class Rel:
    """card 160's model with the agent's move (arm A), or every token (arm B), predicted by the network above"""
    def __init__(s, base, w, cells):
        s.base, s.w, s.cells = base, w, cells; s._a1, s._pc, s._stc = {}, {}, {}; s.NC = base.NC; s.PAD = base.NC + 5

    def __getattr__(s, k):
        return getattr(s.__dict__["base"], k)

    def mem(s, items):
        return [(k, R_OUT + a) for k, a in sorted(items) if k < s.NC]

    def agent_pred(s, d, neigh_rows, items):
        nb = np.asarray(neigh_rows); n = len(nb); m = s.mem(items); L = 8 + len(m); ce = np.full((n, L), s.PAD); cr = np.zeros((n, L), int); cm = np.zeros((n, L), bool)
        ce[:, :8] = np.where(nb < s.NC, nb, s.PAD); cr[:, :8] = [OFFIX(*o) for o in OFF8]; cm[:, :8] = nb < s.NC
        for j, (k, r) in enumerate(m):
            ce[:, 8 + j] = k; cr[:, 8 + j] = r; cm[:, 8 + j] = True
        rep = lambda x: np.repeat(x, NA, 0); lg = forward(s.w, np.full(n * NA, s.NC + d), np.tile(np.arange(NA), n), rep(ce), rep(cr), rep(cm))[:, s.NC:]
        best = np.where(s.base.amask[d], lg, -1e9).argmax(-1).reshape(n, NA)
        return np.stack([best // 4, best % 4], -1)

    def agent_one(s, d, a, neigh, items):
        key = (d, neigh, items)
        if key not in s._a1:
            s._a1[key] = s.agent_pred(d, [neigh], items)[0]
        return tuple(int(x) for x in s._a1[key][a])

    def p_cell(s, c, a, g, items):
        if not s.cells:
            return s.base.p_cell(c, a, g, items)
        key = (c, a, g, items)
        if key not in s._pc:
            if c not in s.base.cell:
                s._pc[key] = {c: 1.0}
            else:
                if g == FAR or g < 0:
                    first = (s.NC + 4, R_FAR)
                else:
                    dx, dy, d = gid_inv(int(g)); first = (s.NC + d, OFFIX(-dx, -dy))
                ctx = [first] + s.mem(items); outs = s.base.cell[c]
                lg = forward(s.w, np.array([c]), np.array([a]), np.array([[x[0] for x in ctx]]), np.array([[x[1] for x in ctx]]), np.ones((1, len(ctx)), bool))[0, outs]
                p = np.exp(lg - lg.max()); p /= p.sum(); s._pc[key] = dict(zip(outs, p.tolist()))
        return s._pc[key]

    def statement(s, c, o, a_req=None):
        if not s.cells:
            return s.base.statement(c, o, a_req)
        key = (c, o, a_req)
        if key not in s._stc:
            s._stc[key] = s._statement(c, o, a_req)
        return s._stc[key]

    def _statement(s, c, o, a_req):
        """card 160's read-out of the action, the places and the needed tokens, asked of this network.  Whether any token gone
        by the same action would do is asked by putting another such token in its place."""
        B = s.base
        if c not in B.cell or o not in B.cell[c] or o == c:
            return None
        outs = B.cell[c]; io = outs.index(o); sa, sg, si, Y, _, _ = B.sit[c]
        p = lambda a, g, items: s.p_cell(c, a, g, items).get(o, 0.0)
        cand = [(Y[r, io], r) for r in np.nonzero(Y[:, io] >= SEEN)[0] if (a_req is None or sa[r] == a_req) and p(int(sa[r]), int(sg[r]), B.sets[si[r]]) >= 0.5]
        if not cand:
            return None
        n, r = max(cand); a, g, items = int(sa[r]), int(sg[r]), B.sets[si[r]]; ma = sa == a
        G = [g2 for g2 in range(NG + 1) if p(a, g2, items) >= 0.5]
        att = ma & np.isin(sg, G); tries = Y[att].sum(1); isets = [B.sets[i] for i in si[att]]
        seen = lambda pred: sum(t for t, x in zip(tries, isets) if pred(x)) >= SEEN
        need = []
        for it in sorted(items):
            if p(a, g, items - {it}) < 0.5 and seen(lambda x: it not in x):
                other = any(p(a, g, (items - {it}) | {i2}) >= 0.5 for i2 in B.items if i2[1] == it[1] and i2 != it and i2 not in items)
                need.append(("any", it[1]) if other and seen(lambda x: it not in x and any(q[1] == it[1] for q in x)) else ("it", it))
        return dict(a=a, G=G, need=need, avoid=[], n=int(n))

    def outcomes(s, c, a_req=None):
        if c not in s.base.cell:
            return []
        return [o for o in s.base.cell[c] if o != c and s.statement(c, o, a_req) is not None]


def load(exp, arm, T):
    base = pickle.load(open("runs/160/model_%s_mlp.pkl" % exp, "rb")); base.codes = T.codes; base._st = {}
    if arm == "base":
        return base
    return Rel(base, pickle.load(open("runs/162/net_%s_%s.pkl" % (arm, exp), "rb")), arm == "B")
