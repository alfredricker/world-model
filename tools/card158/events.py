"""Card 158: a latent state kept in each object module, its events and their conditions.

Each object (a stored pattern in an episode, with card 157's own tokens) has a sequence of steps.  The module keeps
h_t in {0, 1}:
  emission     p(came along_t | h_t, the hand moved_t)                                   a table, learned
  transition   p(h_t = 1 | h_{t-1}, contact_t, grip_t, moved_t; class k) = sigmoid(b_k + sum of one coupling per token),
               a shared set of couplings plus each class's sparse difference
fitted by expectation-maximisation (forward-backward).  An event is a change in the Viterbi sequence of states.

usage: events.py <name> <experience: script|agent> <by: shape|colour> <world seeds...>   writes runs/158/<name>_<experience>_<by>.json
"""
import sys, json, time, numpy as np, torch
import importlib.util
_spec = importlib.util.spec_from_file_location("classes157", "tools/card157/classes.py"); C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
W, R, P, pc = C.W, C.R, C.P, C.pc
T, H, HAND, DISP = C.T, C.H, C.HAND, C.DISP
TOK = ("state before", "contact", "grip", "moved")
SEEN = 3


# ------------------------------------------------------------------------------------------------ sequences
def own_sequences(X, L):
    """one row per (episode, pattern): [T] arrays of card 157's own tokens; `obj` (the true object, -1 for none) is for scoring only"""
    rows = []
    for e, ep in enumerate(X["tr"]):
        slot_of = {int(p): i for i, p in enumerate(X["own0"][e]) if p >= 0}
        dh = np.diff(ep["hand"], axis=0); moved = dh.any(1)
        for s in X["seen"][e]:
            t0 = s["t0"]; y0, x0, y1, x1 = s["box"][t0]; hy, hx = ep["hand"][t0]
            inter = max(0, min(y1, hy + HAND) - max(y0, hy)) * max(0, min(x1, hx + HAND) - max(x0, hx))
            if s["n"][t0] == 0 or inter > 0.5 * (y1 - y0) * (x1 - x0) or s["p"] not in L["idx_of"]:
                continue
            valid = (np.arange(T) >= t0) & (s["n"] >= 4) & (s["code"] >= 0)
            d = np.clip(DISP[s["code"].clip(0)], -4, 4)
            came = valid & moved & (d == dh).all(1)
            contact = np.array([valid[t] and C.touching(ep["hand"][t], s["box"][t]) for t in range(T)])
            q = X["mem"].info[s["p"]]; i = slot_of.get(s["p"], q["tid"] - 3 if q["episode"] == e and q.get("tid", 0) >= 3 else -1)
            rows.append(dict(e=e, obj=i, t0=t0, valid=valid, contact=contact, came=came, moved=moved, grip=ep["acts"][:, 2], cls=int(L["kind"][L["idx_of"][s["p"]]]),
                             true=C.true_kind(X, s["p"])))
    return rows


def sim_sequences(X):
    """the upper bound's input: every true object, the simulator's contact and displacement, its true behaviour as the class"""
    rows = []
    for e, ep in enumerate(X["tr"]):
        dh = np.diff(ep["hand"], axis=0); moved = dh.any(1)
        for i, o in enumerate(ep["objs"]):
            d = np.diff(ep["pos"][:, i], axis=0)
            rows.append(dict(e=e, obj=i, t0=0, valid=np.ones(T, bool), contact=np.array([W.touching(ep["hand"][t], ep["pos"][t, i], o["size"]) for t in range(T)]),
                             came=moved & (d == dh).all(1), moved=moved, grip=ep["acts"][:, 2], cls=o["kind"], true=o["kind"]))
    return rows


def stack(rows, cls=None):
    D = {k: np.stack([r[k] for r in rows]).astype(int) for k in ("valid", "contact", "came", "moved", "grip")}
    D["t0"] = np.array([r["t0"] for r in rows]); D["cls"] = np.array([r["cls"] for r in rows]) if cls is None else np.asarray(cls)
    D["live"] = (np.arange(T)[None] >= D["t0"][:, None]).astype(int)                       # steps of the sequence (before t0 the pattern did not exist)
    return D


# ------------------------------------------------------------------------------------------------ the model
class States:
    def __init__(self, K):
        self.K = K
        self.emit = np.array([[0.02, 0.1], [0.02, 0.8]])                 # p(came | h, moved): a start that only breaks the symmetry
        self.w0 = np.array([-2.0, 4.0, 0.0, 0.0, 0.0]); self.dw = np.zeros((K, 5))         # bias, state before, contact, grip, moved

    def p1(self, cls, hb, contact, grip, moved):
        w = self.w0 + self.dw[cls]
        return 1 / (1 + np.exp(-(w[..., 0] + w[..., 1] * hb + w[..., 2] * contact + w[..., 3] * grip + w[..., 4] * moved)))

    def factors(self, D):
        """A [S,T,2,2] (from h_{t-1} to h_t) and B [S,T,2] (the emission of what was seen)"""
        S = len(D["cls"]); A = np.zeros((S, T, 2, 2)); cls = D["cls"][:, None]
        for hb in (0, 1):
            p = self.p1(cls, hb, D["contact"], D["grip"], D["moved"]); A[:, :, hb, 1] = p; A[:, :, hb, 0] = 1 - p
        A = np.where(D["live"][..., None, None] == 1, A, np.eye(2))
        pe = self.emit[:, D["moved"]].transpose(1, 2, 0)                                    # [S,T,2]: p(came=1 | h)
        B = np.where(D["came"][..., None] == 1, pe, 1 - pe); B = np.where(D["valid"][..., None] == 1, B, 1.0)
        return A, B

    def posteriors(self, D):
        A, B = self.factors(D); S = len(A); al = np.zeros((S, T, 2)); be = np.ones((S, T, 2)); prev = np.full((S, 2), 0.5); ll = 0.0
        for t in range(T):
            a = np.einsum("si,sij->sj", prev, A[:, t]) * B[:, t]; z = a.sum(1, keepdims=True); al[:, t] = a / z; prev = al[:, t]; ll += np.log(z).sum()
        for t in range(T - 2, -1, -1):
            b = np.einsum("sij,sj->si", A[:, t + 1], B[:, t + 1] * be[:, t + 1]); be[:, t] = b / b.sum(1, keepdims=True)
        g = al * be; g /= g.sum(-1, keepdims=True)
        prev = np.concatenate([np.full((S, 1, 2), 0.5), al[:, :-1]], 1)
        xi = prev[..., :, None] * A * (B * be)[..., None, :]; xi /= xi.sum((-1, -2), keepdims=True)
        return g, xi, ll

    def fit(self, D, iters=30, l1=1.0, steps=300):
        K = self.K; live = D["live"] == 1; self.log = []
        for it in range(iters):
            g, xi, ll = self.posteriors(D); self.log.append(round(float(ll), 1))
            for mv in (0, 1):                                                               # emission: counts
                sel = (D["valid"] == 1) & (D["moved"] == mv)
                for h in (0, 1):
                    self.emit[h, mv] = (g[..., h][sel & (D["came"] == 1)].sum() + 0.01) / (g[..., h][sel].sum() + 1.0)
            N = np.zeros((K, 2, 2, 2, 2, 2))                                                # transition: expected counts, then a weighted logistic fit
            idx = (np.broadcast_to(D["cls"][:, None], live.shape)[live], D["contact"][live], D["grip"][live], D["moved"][live])
            for hb in (0, 1):
                for h in (0, 1):
                    np.add.at(N[:, hb, :, :, :, h], idx, xi[..., hb, h][live])
            Nt = torch.tensor(N); w0 = torch.tensor(self.w0, requires_grad=True); dw = torch.tensor(self.dw, requires_grad=True)
            f = torch.tensor(np.stack(np.meshgrid(*[np.arange(2)] * 4, indexing="ij"), -1).reshape(-1, 4), dtype=torch.float64)      # (hb, contact, grip, moved)
            opt = torch.optim.Adam([w0, dw], lr=0.05); tot = float(N.sum())
            for _ in range(steps):
                w = w0[None] + dw; lg = w[:, :1] + w[:, 1:] @ f.T                           # [K,16]
                n1 = Nt[..., 1].reshape(K, 16); n0 = Nt[..., 0].reshape(K, 16)
                loss = ((n1 * torch.nn.functional.softplus(-lg) + n0 * torch.nn.functional.softplus(lg)).sum() + l1 * (w0[1:].abs().sum() + dw.abs().sum())) / tot     # the penalty in nats: a coupling is kept if it saves more than l1 nats per unit, however rare its event
                opt.zero_grad(); loss.backward(); opt.step()
            self.w0, self.dw = w0.detach().numpy(), dw.detach().numpy(); self.N6 = N; self.N = N.sum(-1)     # how often each situation (class, state before, contact, grip, moved) occurred, and with which state after
        self.up = int(self.emit[1, 1] > self.emit[0, 1])                                    # the state in which the object comes along when the hand moves
        return self

    def seen(self, k, hb, c, g, mv):
        """a situation counts only if the experience produced it (LESSONS: check a condition only in a situation the learned effects produce)"""
        return self.N[k, hb, c, g, mv] >= SEEN

    def viterbi(self, D):
        A, B = self.factors(D); S = len(A); lA, lB = np.log(A + 1e-12), np.log(B + 1e-12); d = np.full((S, 2), np.log(0.5)); back = np.zeros((S, T, 2), int)
        for t in range(T):
            c = d[:, :, None] + lA[:, t]; back[:, t] = c.argmax(1); d = c.max(1) + lB[:, t]
        h = np.zeros((S, T), int); h[:, -1] = d.argmax(1)
        for t in range(T - 1, 0, -1):
            h[:, t - 1] = back[np.arange(S), t, h[:, t]]
        return h


# ------------------------------------------------------------------------------------------------ tests
def events_f1(rows, h, X, up):
    """model events (changes of h) against the simulator's taken / released, per sequence, from the step after its first"""
    tp = fp = fn = 0; late = []
    for r, hs in zip(rows, h):
        st = (hs == up).astype(int); t0 = r["t0"]; pred = {(t, st[t]) for t in range(t0 + 1, T) if st[t] != st[t - 1]}
        true = set()
        if r["obj"] >= 0:
            held = (X["tr"][r["e"]]["held"][1:] == r["obj"]).astype(int)                    # held during step t
            true = {(t, held[t]) for t in range(t0 + 1, T) if held[t] != held[t - 1]}
        tp += len(pred & true); fp += len(pred - true); fn += len(true - pred)
    return dict(f1=pc([2 * tp / max(2 * tp + fp + fn, 1)]), precision=pc([tp / max(tp + fp, 1)]), recall=pc([tp / max(tp + fn, 1)]), events_true=tp + fn, events_model=tp + fp)


def seen_baseline(D):
    """the state read from what is seen: the last value of "came along" at a step in which the hand moved"""
    h = np.zeros(D["came"].shape, int); cur = np.zeros(len(h), int)
    for t in range(T):
        see = (D["valid"][:, t] == 1) & (D["moved"][:, t] == 1); cur = np.where(see, D["came"][:, t], cur); h[:, t] = cur
    return h


def all_true_events(X):
    return int(sum((np.diff((ep["held"][1:] == i).astype(int)) != 0).sum() for ep in X["tr"] for i in range(len(ep["objs"]))))


def best_action(model, w, want, hb=0, contact=1):
    """the grip and move that most probably bring state `want` from state hb, under a belief w over classes, among the
    situations the experience produced; None ("cannot") if it is more probable that no action does"""
    acts = [(g, mv) for g in (0, 1) for mv in (0, 1)]
    pk = np.array([[(lambda q: q if want == 1 else 1 - q)(float(model.p1(k, hb, contact, g, mv))) if model.seen(k, hb, contact, g, mv) else 0.0 for g, mv in acts] for k in range(model.K)])
    w = np.asarray(w, float); pa = w @ pk; none = float(w @ (1 - pk.max(1)))
    return (float(pa.max()), acts[int(pa.argmax())]) if pa.max() > none else (none, None)


def test_take(te, views, L, model, belief):
    """card 152's take test: the hand at the object's side; the action that makes the object held, scored in the simulator"""
    ok, kinds, bound = [], [], []
    for e, ep in enumerate(te):
        st = dict(hand=ep["hand"][0], pos=ep["pos"][0], order=ep["order"], held=-1)
        for i, o in enumerate(ep["objs"]):
            spot = W.side_spot(st["hand"], st["pos"][i], o["size"])
            if any(W.touching(spot, st["pos"][j], q["size"]) for j, q in enumerate(ep["objs"]) if j != i):
                continue
            s1 = dict(st, hand=spot); w = belief(e, i, o); a = None
            if w is not None:
                p, act = best_action(model, w, model.up, hb=1 - model.up)
                if act is not None:
                    d = np.sign(np.array([H // 2 - 3, H // 2 - 3]) - spot) * 3; d[d == 0] = 3
                    a = (int(d[0]) * act[1], int(d[1]) * act[1], act[0])
            ok.append(a is None if o["kind"] == 1 else (a is not None and W.step(s1, a, ep["objs"])["held"] == i)); kinds.append(o["kind"]); bound.append(w is not None)
    ok, kinds = np.array(ok), np.array(kinds)
    return dict(n=len(ok), all=pc(ok), takeable=pc(ok[kinds == 0]), fixed=pc(ok[kinds == 1]), sticky=pc(ok[kinds == 2]))


def conditions(model, k):
    """for class k: the conditions of "taken" (to the state that comes along) and of "released".  Start from the
    situation in which the event most often happened; a token is a condition if changing it alone makes the event
    improbable; a change that leads to a situation the experience never produced is left untested"""
    out = {}
    for name, hb, want in (("taken", 1 - model.up, model.up), ("released", model.up, 1 - model.up)):
        pr = lambda c, g, mv: (lambda q: q if want == 1 else 1 - q)(float(model.p1(k, hb, c, g, mv)))
        cand = [(float(model.N6[k, hb, c, g, mv, want]), (c, g, mv)) for c in (0, 1) for g in (0, 1) for mv in (0, 1) if model.seen(k, hb, c, g, mv) and pr(c, g, mv) >= 0.5]
        if not cand:
            out[name] = None; continue
        _, (c, g, mv) = max(cand); cond = {}; untested = []
        for tok, alt in (("contact", (1 - c, g, mv)), ("grip", (c, 1 - g, mv)), ("moved", (c, g, 1 - mv))):
            if not model.seen(k, hb, *alt):
                untested.append(tok)
            elif pr(*alt) < 0.5:
                cond[tok] = dict(contact=c, grip=g, moved=mv)[tok]
        out[name] = cond; out[name + " untested"] = untested
    return out


def run_world(exp, seed, by):
    t0 = time.time(); X = torch.load(("runs/158/exp_script_%s_%d.pt" if exp == "script" else "runs/157/exp_%s_%d.pt") % (by, seed), weights_only=False); world = W.make_world(seed, by)
    log = dict(world=seed, by=by, experience=exp, kind_of=world["kind_of"].tolist(), patterns=len(X["mem"]), tally=X["tally"]); cl = {}
    cache = "runs/158/fit_%s_%s_%d.pt" % (exp, by, seed)
    try:
        L, cl, te, views, fs = torch.load(cache, weights_only=False)
    except FileNotFoundError:
        G, O, m, pat, agree = C.own_tokens(X); L = C.fit(X, G, O, m, pat, cl, with_bounds=False)
        te, views, fs = C.first_sights(world, X, 100, seed * 100 + 30, "fresh"); te = [{k: ep[k] for k in ("objs", "order", "hand", "pos")} for ep in te]
        torch.save((L, cl, te, views, fs), cache)
    log["classes"] = dict(n=cl["classes"], sizes=cl["class_sizes"], purity=cl["class_purity"], table=cl["classes_by_true_behaviour_then_ground_hand"])
    rows = own_sequences(X, L); D = stack(rows); K = L["dyn"].K
    takes = sum(int(((ep["held"][1:] >= 0) & (ep["held"][:-1] < 0)).sum()) for ep in X["tr"])
    still = sum(int(((ep["held"][1:] >= 0) & (ep["held"][:-1] < 0) & ~np.diff(ep["hand"], axis=0).any(1)).sum()) for ep in X["tr"])
    log["experience_events"] = dict(taken=takes, taken_without_a_move=still, all_true_events=all_true_events(X), sequences=len(rows), sequences_on_true_objects=sum(r["obj"] >= 0 for r in rows))
    model = States(K).fit(D); h = model.viterbi(D)
    tk = np.array([1 if r["true"] >= 3 else r["true"] for r in rows]); Dt = stack(rows, tk); m_true = States(3).fit(Dt)       # own tokens, true behaviours given
    D1 = stack(rows, np.zeros(len(rows), int)); m_one = States(1).fit(D1)                                                   # own tokens, one class for all
    srows = sim_sequences(X); Ds = stack(srows); m_sim = States(3).fit(Ds)                                                  # the simulator's tokens, true behaviours
    m_sim5 = States(3).fit(Ds, l1=5.0)                                                                                      # the same with a penalty of 5 nats
    log["events"] = {"latent state": events_f1(rows, h, X, model.up), "state read from what is seen": events_f1(rows, seen_baseline(D), X, 1),
                     "latent state, true behaviours given": events_f1(rows, m_true.viterbi(Dt), X, m_true.up), "latent state, one class": events_f1(rows, m_one.viterbi(D1), X, m_one.up),
                     "simulator's tokens, true behaviours": events_f1(srows, m_sim.viterbi(Ds), X, m_sim.up), "simulator's tokens, true behaviours, penalty 5": events_f1(srows, m_sim5.viterbi(Ds), X, m_sim5.up), "simulator's tokens, state read from what is seen": events_f1(srows, seen_baseline(Ds), X, 1)}
    log["first_sight"] = fs

    def by_appearance(e, i, o):
        v = views.get((e, i))
        if v is None:
            return None
        return R.kind_belief(L, [L["app"][v["p"]]] if v["p"] >= 0 else [v["v"]])[0]
    log["take"] = {"latent state": test_take(te, views, L, model, by_appearance), "own tokens, true behaviours given": test_take(te, views, L, m_true, lambda e, i, o: np.eye(3)[o["kind"]]),
                   "own tokens, one class": test_take(te, views, L, m_one, lambda e, i, o: np.ones(1)), "simulator's tokens, true behaviours": test_take(te, views, L, m_sim, lambda e, i, o: np.eye(3)[o["kind"]]),
                   "simulator's tokens, true behaviours, penalty 5": test_take(te, views, L, m_sim5, lambda e, i, o: np.eye(3)[o["kind"]])}
    table = np.array(cl["classes_by_true_behaviour_then_ground_hand"]); st = {}
    for b, nm, g_right in ((0, "takeable", 1), (2, "sticky", 0)):
        k = int(table[:, b].argmax()); c = conditions(model, k)
        st[nm] = dict(model_class=k, taken=c["taken"], released=c["released"], taken_right=c["taken"] == dict(contact=1, grip=g_right), released_right=c["released"] == dict(grip=1 - g_right))
    log["conditions"] = st; log["conditions_bound"] = {nm: conditions(m_sim5, b) for b, nm in ((0, "takeable"), (2, "sticky"))}
    log["model"] = dict(up=model.up, emit=model.emit.round(3).tolist(), shared=dict(zip(("bias",) + TOK, model.w0.round(2).tolist())), by_class=(model.w0 + model.dw).round(2).tolist(), log_likelihood=model.log[::5])
    log["seconds"] = round(time.time() - t0)
    return log


if __name__ == "__main__":
    name, exp, by = sys.argv[1], sys.argv[2], sys.argv[3]; res = []
    for s in sys.argv[4:]:
        res.append(run_world(exp, int(s), by)); r = res[-1]
        print(exp, by, s, "classes", r["classes"]["n"], r["classes"]["purity"], "events in experience", r["experience_events"], "seconds", r["seconds"])
        for k, v in r["events"].items(): print("   events %-52s" % k, v)
        for k, v in r["take"].items(): print("   take   %-52s" % k, v)
        print("   conditions", r["conditions"]); print("   conditions with the simulator's tokens", r["conditions_bound"]); print("   model", r["model"], flush=True)
        json.dump(res, open("runs/158/%s_%s_%s.json" % (name, exp, by), "w"), indent=1)
