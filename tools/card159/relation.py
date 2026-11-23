"""Card 159: contact as a learned relation between the hand's place and the object's.

The relative place r_t of an object's box and the hand's box (the gap along each axis, cut off at -2 and 3: 36 tokens)
enters the transition of the object's latent state (card 158) in place of the contact rule:
  p(h_t = 1 | h_{t-1}, r_t, grip_t, moved_t; class k) = sigmoid(b_k + w_k . (h_{t-1}, grip_t, moved_t) + u[r_t]),   u shared, sparse
"In reach" is the set of relative places from which "taken" is at least half as probable as from the best place.  A first fit with one class for all objects gives the two-valued
contact token that card 157's rule for the dynamics classes needs; then the classes; then the fit with the classes.

usage: relation.py <name> <by: shape|colour> <world seeds...>     writes runs/159/<name>_<by>.json
"""
import sys, json, time, importlib.util, numpy as np, torch
_s = importlib.util.spec_from_file_location("events158", "tools/card158/events.py"); Ev = importlib.util.module_from_spec(_s); _s.loader.exec_module(Ev)
C, W, R, pc = Ev.C, Ev.W, Ev.R, Ev.pc
T, H, HAND, DISP = Ev.T, Ev.H, Ev.HAND, Ev.DISP
LO, HI = -2, 3; NV = HI - LO + 1; V = NV * NV; SEEN = Ev.SEEN
WHERE = "box"            # an object's place: its frame carried by its own displacement ("box": the box around its points)


def rel_code(hand, box):
    hy, hx = hand; y0, x0, y1, x1 = box
    gy = max(y0 - (hy + HAND), hy - y1); gx = max(x0 - (hx + HAND), hx - x1)
    return (int(np.clip(gy, LO, HI)) - LO) * NV + int(np.clip(gx, LO, HI)) - LO


def name(v):
    return "(%d, %d)" % (v // NV + LO, v % NV + LO)


TOUCH = np.array([(v // NV + LO == 0 and v % NV + LO < 0) or (v % NV + LO == 0 and v // NV + LO < 0) for v in range(V)])      # the simulator's rule, for scoring
RULE = lambda hand, box: bool(TOUCH[rel_code(hand, box)])                                                                 # ... which is cards 157-158's rule on boxes


# ------------------------------------------------------------------------------------------------ sequences
def own_rows(X):
    """one row per (episode, pattern); `obj` and `true` are for scoring only"""
    rows = []
    for e, ep in enumerate(X["tr"]):
        slot_of = {int(p): i for i, p in enumerate(X["own0"][e]) if p >= 0}
        dh = np.diff(ep["hand"], axis=0); moved = dh.any(1)
        for s in X["seen"][e]:
            t0 = s["t0"]
            if s["n"][t0] < 0.5 * max(s["n_all"][t0], 1):                    # most of its points lie on the hand's box: it is the hand
                continue
            valid = (np.arange(T) >= t0) & (s["n"] >= 4) & (s["code"] >= 0)
            d = np.clip(DISP[s["code"].clip(0)], -4, 4); came = valid & moved & (d == dh).all(1)
            rel = np.array([rel_code(ep["hand"][t], s[WHERE][t]) if valid[t] else V - 1 for t in range(T)])
            q = X["mem"].info[s["p"]]; i = slot_of.get(s["p"], q["tid"] - 3 if q["episode"] == e and q.get("tid", 0) >= 3 else -1)
            rows.append(dict(e=e, p=s["p"], obj=i, t0=t0, valid=valid, rel=rel, came=came, moved=moved, grip=ep["acts"][:, 2], disp=d, true=C.true_kind(X, s["p"])))
    return rows


def sim_rows(X):
    rows = []
    for e, ep in enumerate(X["tr"]):
        dh = np.diff(ep["hand"], axis=0); moved = dh.any(1)
        for i, o in enumerate(ep["objs"]):
            d = np.diff(ep["pos"][:, i], axis=0); s = o["size"]
            rel = np.array([rel_code(ep["hand"][t], (ep["pos"][t, i, 0], ep["pos"][t, i, 1], ep["pos"][t, i, 0] + s, ep["pos"][t, i, 1] + s)) for t in range(T)])
            rows.append(dict(e=e, p=-1, obj=i, t0=0, valid=np.ones(T, bool), rel=rel, came=moved & (d == dh).all(1), moved=moved, grip=ep["acts"][:, 2], true=o["kind"]))
    return rows


def stack(rows, cls, rel=None):
    D = {k: np.stack([r[k] for r in rows]).astype(int) for k in ("valid", "came", "moved", "grip")}
    D["rel"] = np.stack([r["rel"] for r in rows]) if rel is None else rel
    D["t0"] = np.array([r["t0"] for r in rows]); D["cls"] = np.asarray(cls); D["live"] = (np.arange(T)[None] >= D["t0"][:, None]).astype(int)
    return D


# ------------------------------------------------------------------------------------------------ the model
class States(Ev.States):
    """card 158's latent state with a relative-place token of nv values in place of the contact token"""
    def __init__(self, K, nv=V):
        self.K, self.nv = K, nv; self.emit = np.array([[0.02, 0.1], [0.02, 0.8]])
        self.w0 = np.array([-2.0, 4.0, 0.0, 0.0]); self.dw = np.zeros((K, 4)); self.u = np.zeros(nv)          # bias, state before, grip, moved | relative place

    def p1(self, cls, hb, rel, grip, moved):
        w = self.w0 + self.dw[cls]
        return 1 / (1 + np.exp(-(w[..., 0] + w[..., 1] * hb + w[..., 2] * grip + w[..., 3] * moved + self.u[rel])))

    def factors(self, D):
        S = len(D["cls"]); A = np.zeros((S, T, 2, 2)); cls = D["cls"][:, None]
        for hb in (0, 1):
            p = self.p1(cls, hb, D["rel"], D["grip"], D["moved"]); A[:, :, hb, 1] = p; A[:, :, hb, 0] = 1 - p
        A = np.where(D["live"][..., None, None] == 1, A, np.eye(2))
        pe = self.emit[:, D["moved"]].transpose(1, 2, 0)
        B = np.where(D["came"][..., None] == 1, pe, 1 - pe); B = np.where(D["valid"][..., None] == 1, B, 1.0)
        return A, B

    def fit(self, D, iters=30, l1=1.0, steps=300):
        K, nv = self.K, self.nv; live = D["live"] == 1; self.log = []; sp = torch.nn.functional.softplus
        f = torch.tensor(np.stack(np.meshgrid(np.arange(2), np.arange(nv), np.arange(2), np.arange(2), indexing="ij"), -1).reshape(-1, 4))     # (hb, rel, grip, moved)
        fx = f[:, [0, 2, 3]].double(); fr = f[:, 1]
        for it in range(iters):
            g, xi, ll = self.posteriors(D); self.log.append(round(float(ll), 1))
            for mv in (0, 1):
                sel = (D["valid"] == 1) & (D["moved"] == mv)
                for h in (0, 1):
                    self.emit[h, mv] = (g[..., h][sel & (D["came"] == 1)].sum() + 0.01) / (g[..., h][sel].sum() + 1.0)
            N = np.zeros((K, 2, nv, 2, 2, 2)); idx = (np.broadcast_to(D["cls"][:, None], live.shape)[live], D["rel"][live], D["grip"][live], D["moved"][live])
            for hb in (0, 1):
                for h in (0, 1):
                    np.add.at(N[:, hb, :, :, :, h], idx, xi[..., hb, h][live])
            Nt = torch.tensor(N).reshape(K, -1, 2); w0 = torch.tensor(self.w0, requires_grad=True); dw = torch.tensor(self.dw, requires_grad=True); u = torch.tensor(self.u, requires_grad=True)
            opt = torch.optim.Adam([w0, dw, u], lr=0.05); tot = float(N.sum())
            for _ in range(steps):
                w = w0[None] + dw; lg = w[:, :1] + w[:, 1:] @ fx.T + u[fr][None]
                loss = ((Nt[..., 1] * sp(-lg) + Nt[..., 0] * sp(lg)).sum() + l1 * (w0[1:].abs().sum() + dw.abs().sum() + u.abs().sum())) / tot
                opt.zero_grad(); loss.backward(); opt.step()
            self.w0, self.dw, self.u = w0.detach().numpy(), dw.detach().numpy(), u.detach().numpy(); self.N6 = N; self.N = N.sum(-1)
        self.up = int(self.emit[1, 1] > self.emit[0, 1]); self.far = int(self.N.sum((0, 1, 3, 4)).argmax())      # the commonest relative place
        up, free = self.up, 1 - self.up; pt = np.zeros(nv)        # for each relative place: how probable "taken" is at best (any class, grip, move the experience produced)
        for v in range(nv):
            c = [(lambda q: q if up == 1 else 1 - q)(float(self.p1(k, free, v, g, mv))) for k in range(K) for g in (0, 1) for mv in (0, 1) if self.N[k, free, v, g, mv] >= SEEN]
            pt[v] = max(c) if c else 0.0
        self.reach = pt >= 0.5 * pt.max()                         # in reach: "taken" is at least half as probable there as at the best place
        self.best = int(self.N6[:, free, :, :, :, up].sum((0, 2, 3)).argmax())      # the relative place at which objects were most often taken
        return self


# ------------------------------------------------------------------------------------------------ tests
def contact_agreement(rows, X, reach):
    """steps on true objects: the model's "in reach" against the simulator's touching"""
    both = sim = own = 0
    for r in rows:
        if r["obj"] < 0:
            continue
        ep = X["tr"][r["e"]]; o = ep["objs"][r["obj"]]
        for t in np.nonzero(r["valid"])[0]:
            a = W.touching(ep["hand"][t], ep["pos"][t, r["obj"]], o["size"]); b = bool(reach[r["rel"][t]])
            both += a and b; sim += a and not b; own += b and not a
    return dict(caught=pc([both / max(both + sim, 1)]), false=pc([own / max(both + own, 1)]), contacts_simulator=int(both + sim), contacts_model=int(both + own))


def conditions(model, k):
    """as card 158's, with the relative place in place of contact: it is a condition if being at the commonest relative
    place instead (far from the hand) makes the event improbable"""
    out = {}
    for name_, hb, want in (("taken", 1 - model.up, model.up), ("released", model.up, 1 - model.up)):
        pr = lambda v, g, mv: (lambda q: q if want == 1 else 1 - q)(float(model.p1(k, hb, v, g, mv)))
        cand = [(float(model.N6[k, hb, v, g, mv, want]), (v, g, mv)) for v in range(model.nv) for g in (0, 1) for mv in (0, 1) if model.N[k, hb, v, g, mv] >= SEEN and pr(v, g, mv) >= 0.5]
        if not cand:
            out[name_] = None; continue
        _, (v, g, mv) = max(cand); cond = {}; untested = []
        for tok, alt in (("place", (model.far, g, mv)), ("grip", (v, 1 - g, mv)), ("moved", (v, g, 1 - mv))):
            if tok == "place" and v == model.far:
                continue
            if model.N[k, hb, alt[0], alt[1], alt[2]] < SEEN:
                untested.append(tok)
            elif pr(*alt) < 0.5:
                cond[tok] = dict(place=name(v) if model.nv == V else int(v), grip=g, moved=mv)[tok]
        out[name_] = cond; out[name_ + " untested"] = untested
    return out


def statements(model, table):
    st = {}
    for b, nm, g_right in ((0, "takeable", 1), (2, "sticky", 0)):
        k = int(table[:, b].argmax()); c = conditions(model, k); tk, rl = c["taken"], c["released"]
        st[nm] = dict(model_class=k, taken=tk, released=rl, taken_right=tk is not None and set(tk) == {"place", "grip"} and tk["grip"] == g_right, released_right=rl == dict(grip=1 - g_right))
    return st


def own_tokens(X, rows, contact):
    """card 157's token arrays (for its rule for the classes), with the given two-valued contact per row and step"""
    by_ep = {}
    for j, r in enumerate(rows):
        by_ep.setdefault(r["e"], []).append(j)
    Gs = W.tokens(X["tr"])[0]; Gr, Or, Mr, Pr = [], [], [], []
    for e, js in sorted(by_ep.items()):
        for c in range(0, len(js), 4):
            G = Gs[e * T:(e + 1) * T].copy(); O = np.zeros((T, 4, 5), int); m = np.zeros((T, 4), bool); pp = np.full((T, 4), -1)
            for a, j in enumerate(js[c:c + 4]):
                r = rows[j]; v = r["valid"]
                O[v, a, 0] = contact[j][v]; O[v, a, 2:4] = r["disp"][v] + 4; O[v, a, 4] = r["came"][v]; m[:, a] = v; pp[v, a] = r["p"]
            Gr.append(G); Or.append(O); Mr.append(m); Pr.append(pp)
    return np.concatenate(Gr), np.concatenate(Or), np.concatenate(Mr), np.concatenate(Pr)


def run_world(seed, by):
    t0 = time.time(); X = torch.load("runs/159/exp_%s_%d.pt" % (by, seed), weights_only=False); world = W.make_world(seed, by)
    log = dict(world=seed, by=by, kind_of=world["kind_of"].tolist(), patterns=len(X["mem"]), tally=X["tally"])
    rows = own_rows(X); one = np.zeros(len(rows), int); D1 = stack(rows, one)
    first = States(1).fit(D1)                                                # one class for all: which relative places are in reach
    contact = [first.reach[r["rel"]] for r in rows]
    cache = "runs/159/fit_%s_%d.pt" % (by, seed); cl = {}
    try:
        L, cl, te, views, fs = torch.load(cache, weights_only=False)
    except FileNotFoundError:
        G, O, m, pat = own_tokens(X, rows, contact); L = C.fit(X, G, O, m, pat, cl, with_bounds=False)
        te, views, fs = C.first_sights(world, X, 100, seed * 100 + 30, "fresh"); te = [{k: ep[k] for k in ("objs", "order", "hand", "pos")} for ep in te]
        torch.save((L, cl, te, views, fs), cache)
    table = np.array(cl["classes_by_true_behaviour_then_ground_hand"]); keep = [j for j, r in enumerate(rows) if r["p"] in L["idx_of"]]
    rows = [rows[j] for j in keep]; cls = [int(L["kind"][L["idx_of"][r["p"]]]) for r in rows]; K = L["dyn"].K
    log["classes"] = dict(n=cl["classes"], sizes=cl["class_sizes"], purity=cl["class_purity"], table=table.tolist()); log["first_sight"] = fs
    log["sequences"] = dict(n=len(rows), on_true_objects=sum(r["obj"] >= 0 for r in rows), all_true_events=Ev.all_true_events(X))
    D = stack(rows, cls); model = States(K).fit(D)                           # the method
    rule = np.stack([TOUCH[r["rel"]].astype(int) for r in rows]); Dr = stack(rows, cls, rule); m_rule = States(K, 2).fit(Dr)       # the rule of cards 157-158 on the same boxes
    tk = np.array([1 if r["true"] >= 3 else r["true"] for r in rows]); Dt = stack(rows, tk); m_true = States(3).fit(Dt)          # the method with true behaviours given
    srows = sim_rows(X); Ds = stack(srows, [r["true"] for r in srows]); m_sim = States(3).fit(Ds, l1=5.0)                         # the bound: true boxes, true behaviours
    log["contact"] = {"in reach, first fit (one class)": contact_agreement(rows, X, first.reach), "in reach": contact_agreement(rows, X, model.reach), "the rule on the same boxes": contact_agreement(rows, X, TOUCH),
                      "true boxes, in reach": contact_agreement(srows, X, m_sim.reach)}
    log["in_reach"] = {"first fit": [name(v) for v in np.nonzero(first.reach)[0]], "with classes": [name(v) for v in np.nonzero(model.reach)[0]], "true boxes": [name(v) for v in np.nonzero(m_sim.reach)[0]],
                       "couplings": {name(v): round(float(model.u[v]), 2) for v in range(V) if abs(model.u[v]) > 1e-3}}
    log["events"] = {"learned relation": Ev.events_f1(rows, model.viterbi(D), X, model.up), "the rule on the same boxes": Ev.events_f1(rows, m_rule.viterbi(Dr), X, m_rule.up),
                     "learned relation, true behaviours given": Ev.events_f1(rows, m_true.viterbi(Dt), X, m_true.up), "state read from what is seen": Ev.events_f1(rows, Ev.seen_baseline(D), X, 1),
                     "true boxes, true behaviours": Ev.events_f1(srows, m_sim.viterbi(Ds), X, m_sim.up)}

    def by_appearance(e, i, o):
        v = views.get((e, i))
        return None if v is None else R.kind_belief(L, [L["app"][v["p"]]] if v["p"] >= 0 else [v["v"]])[0]
    take = lambda mdl, belief, at: Ev.test_take(te, views, L, type("M", (), dict(K=mdl.K, up=mdl.up, p1=lambda s, k, hb, c, g, mv: mdl.p1(k, hb, at, g, mv), seen=lambda s, k, hb, c, g, mv: mdl.N[k, hb, at, g, mv] >= SEEN))(), belief)
    log["take"] = {"learned relation": take(model, by_appearance, model.best), "the rule on the same boxes": take(m_rule, by_appearance, 1),
                   "learned relation, true behaviours given": take(m_true, lambda e, i, o: np.eye(3)[o["kind"]], m_true.best), "true boxes, true behaviours": take(m_sim, lambda e, i, o: np.eye(3)[o["kind"]], m_sim.best)}
    log["conditions"] = statements(model, table); log["conditions_rule"] = statements(m_rule, table); log["conditions_bound"] = statements(m_sim, np.eye(3))
    log["model"] = dict(up=model.up, emit=model.emit.round(3).tolist(), by_class=(model.w0 + model.dw).round(2).tolist(), far=name(model.far), best=name(model.best), log_likelihood=model.log[::5])
    log["seconds"] = round(time.time() - t0)
    return log


if __name__ == "__main__":
    nm, by = sys.argv[1], sys.argv[2]; res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by)); r = res[-1]
        print(by, s, "classes", r["classes"]["n"], r["classes"]["purity"], r["sequences"], "seconds", r["seconds"])
        for k, v in r["contact"].items(): print("   contact %-36s" % k, v)
        print("   in reach", r["in_reach"])
        for k, v in r["events"].items(): print("   events  %-42s" % k, v)
        for k, v in r["take"].items(): print("   take    %-42s" % k, v)
        for k in ("conditions", "conditions_rule", "conditions_bound"): print("   %s" % k, r[k])
        print("   model", r["model"], flush=True)
        json.dump(res, open("runs/159/%s_%s.json" % (nm, by), "w"), indent=1)
