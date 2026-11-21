"""Card 155: the object modules and dynamics classes of cards 152-153.1, fed by card 154's binding instead of the
simulator's masks.

Learning reads pixels and no mask.  An object is a stored pattern of card 154's memory (a group of points that moved
together, or that pattern recognised at the first sight of a later episode).  Its appearance features are read from
the pattern (card 152's shape and colour memories); its transitions are the simulator's tokens (card 152's, still
given) of the true object the pattern lies on, from the step in which the group first moved (all steps when the
pattern was recognised at first sight).  The dynamics classes and the appearance prior are card 153.1's.

Tests are card 152's, with the same episodes as cards 153 and 153.1:
  known objects   no mask anywhere.  A pattern recognised at first sight gives the object its module (appearance and
                  its own transitions).  What no pattern explains is not an object for the model: it is predicted to
                  stay where it is, and the answer to "take it" is "cannot".
  new objects     (a) no mask anywhere, as above;  (b) the true mask given at test only, to ask what was learned.

usage: pipeline.py <name> <by: shape|colour> <world seeds...>    writes runs/155/<name>_<by>.json   ("_nNN": episodes, 100)
"""
import sys, json, time, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "tools/card154")
import feedback as Fb
sys.path.insert(0, "tools/card153.1"); sys.path.insert(0, "tools/card153")
import classes as Cl, sprites as S
Kd, W, M, R = S.Kd, S.W, S.M, S.R
assert W is Fb.W
pc = R.pc


class NoOne:
    """stands in for card 152's memory of individuals where the test gives masks: no one is recognised"""
    def settle(self, vs):
        return [dict(e=9.0, k=-1)] * len(vs)


def appearance(mem, p):
    m = mem.M[p].cpu(); ys, xs = torch.nonzero(m, as_tuple=True); h, w = int(ys.max()) + 1, int(xs.max()) + 1
    return dict(sil=F.adaptive_avg_pool2d(m[:h, :w].float()[None], 8)[0].reshape(-1), colour=mem.Tx[p].cpu()[m].mean(0))


def owner(rec, mask, n_obj):
    """for each true object of a scene, the recognised pattern that explains most of it (a quarter at least), or -1"""
    out = np.full(n_obj, -1)
    for i in range(n_obj):
        m = mask == 3 + i
        ov = [int((r["pix"] & m).sum()) for r in rec]
        if ov and max(ov) >= max(0.25 * m.sum(), 1):
            out[i] = rec[int(np.argmax(ov))]["m"]
    return out


def learn(world, n_train, seed, log):
    tr = Fb.episodes(world, n_train, seed); mem = Fb.Memory(); G, O, m = W.tokens(tr); T = W.T
    pat = np.full((len(tr), T, 4), -1); true_of = {}; t0 = time.time()
    for e, ep in enumerate(tr):
        P0 = len(mem); n = len(ep["objs"])
        _, rec, _ = Fb.run_episode(ep, e, mem, seed=e)
        best = {i: (0, -1, 0) for i in range(n)}
        for i, p in enumerate(owner(rec, ep["masks"][0], n)):
            if p >= 0:
                best[i] = (10 ** 6, int(p), 0)                         # recognised at first sight: every step of the episode
        for p in range(P0, len(mem)):
            q = mem.info[p]
            if q.get("tid", 0) >= 3 and q["n"] * q["iou"] > best[q["tid"] - 3][0]:
                best[q["tid"] - 3] = (q["n"] * q["iou"], p, q["first"] - 1)   # stored in this episode: from its first move
        for i, (_, p, a) in best.items():
            if p >= 0:
                pat[e, a:, i] = p; true_of[p] = ep["objs"][i]
    used = np.unique(pat[pat >= 0]); idx_of = {int(p): j for j, p in enumerate(used)}; n_ind = len(used)
    ind = np.vectorize(lambda p: idx_of.get(int(p), -1))(pat).reshape(-1, 4); m2 = m & (ind >= 0)
    app = [appearance(mem, p) for p in range(len(mem))]
    sil = torch.stack([app[p]["sil"] for p in used]); col = torch.stack([app[p]["colour"] for p in used])
    shape = M.Proto(R.TH_SHAPE).learn(sil); colour = M.Proto(R.TH_COLOUR).learn(col)
    s_code = shape.energy(sil).argmin(1).numpy(); c_code = colour.energy(col).argmin(1).numpy(); klog = []
    bind_s = round(time.time() - t0)
    dyn, kind, E = Cl.learn_classes(S.make, G, O, m2, ind, n_ind, (s_code, c_code), (len(shape.P), len(colour.P)), log=klog)
    K = dyn.K; looks, w = Kd.fit_looks(s_code, c_code, E, [len(shape.P), len(colour.P), K]) if K > 1 else (None, np.ones(n_ind))
    true = np.array([true_of[int(p)]["kind"] for p in used])
    table = np.array([[float(w[(kind == k) & (true == t)].sum()) for t in range(3)] for k in range(K)])
    all_true = {}
    for e, ep in enumerate(tr):
        for i, o in enumerate(ep["objs"]):
            all_true[Fb.key_of(ep, e, 3 + i)] = o["kind"]
    log.update(train_episodes=n_train, patterns=len(mem), objects_with_transitions=n_ind, objects_by_true_behaviour=np.bincount(true, minlength=3).tolist(),
               true_objects_in_training_by_behaviour=np.bincount(list(all_true.values()), minlength=3).tolist(),
               transitions_used=int(m2.sum()), transitions_with_masks=int(m.sum()), informative_used=int(M.informative(O, m2).sum()), informative_with_masks=int(M.informative(O, m).sum()),
               shape_codes=len(shape.P), colour_codes=len(colour.P), classes=K, class_sizes=np.bincount(kind, minlength=K).tolist(), class_log=klog,
               classes_by_true_behaviour=table.round(1).tolist(), class_purity=pc([table.max(1).sum() / max(table.sum(), 1e-9)]),
               appearance_share=looks.share() if looks else None, binding_seconds=bind_s)
    return dict(me=NoOne(), of_uid={}, shape=shape, colour=colour, dyn=dyn, kind=kind, mis=E, eps=1 / (1 + np.e), looks=looks, G=G, O=O, m=m, tr=tr,
                true_of_kind=table.argmax(1), mem=mem, idx_of=idx_of, app=app)


def test_no_masks(world, L, n, seed, source, bounds):
    """card 152's test of effects and of the take action, with the objects found by feedback at first sight"""
    sizes = W.SIZES + W.NEW_SIZES if source == "fresh" else W.SIZES
    te = Fb.episodes(world, n, seed, source, sizes); G, O, m = W.tokens(te); K = L["dyn"].K; T = W.T
    info = M.informative(O, m)
    per_obj = lambda f: np.array([[f(o) for o in ep["objs"]] + [0] * (4 - len(ep["objs"])) for ep in te])
    kind_e, exc_e = per_obj(lambda o: o["kind"]), per_obj(lambda o: o["kind"] != o["rule"]).astype(bool)
    bound = np.zeros((len(te), 4), bool); w_app = np.full((len(te), 4, K), 1.0 / K, np.float32); w_all = w_app.copy(); right_one = []
    for e, ep in enumerate(te):
        _, rec, _, _ = Fb.first_sight(ep["frames"][0], L["mem"], True, 5000 + e)
        for i, p in enumerate(owner(rec, ep["masks"][0], len(ep["objs"]))):
            if p >= 0:
                bound[e, i] = True; v = [L["app"][p]]
                w_app[e, i] = R.kind_belief(L, v)[0]; w_all[e, i] = R.kind_belief(L, v, [L["idx_of"].get(int(p), -1)])[0]
                right_one.append(L["mem"].info[p]["key"] == Fb.key_of(ep, "x", 3 + i))
    rep = lambda a: a.repeat(T, 0)
    still = (O[..., 2:] != np.array([4, 4, 0])).any(-1) & m                       # wrong if predicted to stay put and not be held
    bad = lambda w: np.where(rep(bound), (M.vote(L["dyn"], G, O, m, rep(w)) != O[..., 2:]).any(-1) & m, still)
    arms = {"appearance and own transitions": bad(w_all), "appearance only": bad(w_app),
            "true behaviour given": M.wrong(bounds["true"], G, O, m, np.eye(3, dtype=np.float32)[rep(kind_e)]),
            "one class for all": M.wrong(bounds["one"], G, O, m, np.ones((len(G), 4, 1), np.float32))}
    out = dict(objects=int(sum(len(ep["objs"]) for ep in te)), objects_bound=pc([bound[e, i] for e, ep in enumerate(te) for i in range(len(ep["objs"]))]),
               bound_to_its_own_pattern=pc(right_one), rows={}, bound_by_behaviour={})
    for k, nm in enumerate(("takeable", "fixed", "sticky")):
        sel = [bound[e, i] for e, ep in enumerate(te) for i, o in enumerate(ep["objs"]) if o["kind"] == k]
        out["bound_by_behaviour"][nm] = dict(n=len(sel), bound=pc(sel))
    cuts = {"all": info, "follow their appearance": info & ~rep(exc_e), "exceptions": info & rep(exc_e), "bound": info & rep(bound), "not bound": info & ~rep(bound)}
    for k, nm in enumerate(("takeable", "fixed", "sticky")):
        cuts[nm] = info & (rep(kind_e) == k)
    for name, c in cuts.items():
        if c.sum():
            out["rows"][name] = dict(n=int(c.sum()), **{a: pc(~wr[c]) for a, wr in arms.items()})
    out["take"] = test_take(te, L, bounds, w_app, w_all, bound)
    return out


def test_take(te, L, bounds, w_app, w_all, bound):
    """card 152's test_take; an object that is not bound gets the answer "cannot" """
    cases = []
    for e, ep in enumerate(te):
        st = dict(hand=ep["hand"][0], pos=ep["pos"][0], order=ep["order"], held=-1)
        for i, o in enumerate(ep["objs"]):
            spot = W.side_spot(st["hand"], st["pos"][i], o["size"])
            if any(W.touching(spot, st["pos"][j], q["size"]) for j, q in enumerate(ep["objs"]) if j != i):
                continue
            cases.append((e, i, dict(st, hand=spot)))
    N = len(cases); G = np.zeros((N, 5), int); O = np.zeros((N, 4, 5), int); m = np.zeros((N, 4), bool)
    G[:, :2] = 4; O[..., 2:4] = 4
    for n, (e, i, st) in enumerate(cases):
        m[n, :len(te[e]["objs"])] = True; O[n, i] = (1, 0, 4, 4, 1)

    def answers(model):
        out = []
        for k in range(model.K):
            a = M.propose(model, G, O, m, k); Ga = G.copy(); Ga[:, 2:] = a
            one = np.zeros((N, 4, model.K), np.float32); one[..., k] = 1
            held = model.forward(Ga, O, m, one)[..., 2]
            out.append([tuple(int(v) for v in a[n] - np.array([4, 4, 0])) if held[n, cases[n][1]] == 1 else None for n in range(N)])
        return out

    def score(model, w, use_bound):
        ans = answers(model); ok = []
        for n, (e, i, st) in enumerate(cases):
            tally = {}
            for k in range(model.K):
                tally[ans[k][n]] = tally.get(ans[k][n], 0) + float(w[n][k])
            a = max(tally, key=tally.get) if (not use_bound or bound[e, i]) else None
            kind = te[e]["objs"][i]["kind"]
            ok.append(a is None if kind == 1 else (a is not None and W.step(st, a, te[e]["objs"])["held"] == i))
        return np.array(ok)
    kinds = np.array([te[e]["objs"][i]["kind"] for e, i, _ in cases]); exc = np.array([te[e]["objs"][i]["kind"] != te[e]["objs"][i]["rule"] for e, i, _ in cases])
    bd = np.array([bound[e, i] for e, i, _ in cases])
    arms = {"appearance and own transitions": score(L["dyn"], [w_all[e, i] for e, i, _ in cases], True), "appearance only": score(L["dyn"], [w_app[e, i] for e, i, _ in cases], True),
            "true behaviour given": score(bounds["true"], [np.eye(3)[k] for k in kinds], False), "one class for all": score(bounds["one"], [[1.0]] * N, False)}
    out = {}
    for name, c in (("all", kinds >= 0), ("takeable", kinds == 0), ("fixed", kinds == 1), ("sticky", kinds == 2), ("exceptions", exc), ("bound", bd), ("not bound", ~bd),
                    ("bound takeable", bd & (kinds == 0)), ("bound sticky", bd & (kinds == 2))):
        if c.sum():
            out[name] = dict(n=int(c.sum()), **{a: pc(v[c]) for a, v in arms.items()})
    return out


def run_world(seed, by, n_train):
    t0 = time.time(); world = W.make_world(seed, by); log = dict(world=seed, by=by, kind_of=world["kind_of"].tolist())
    L = learn(world, n_train, seed * 100 + 10, log)
    kd = np.array([[o["kind"] for o in ep["objs"]] + [0] * (4 - len(ep["objs"])) for ep in L["tr"]]).repeat(W.T, 0)
    bounds = dict(true=M.Dyn().fit(L["G"], L["O"], L["m"], kd, 3), one=M.Dyn().fit(L["G"], L["O"], L["m"], 0 * kd, 1))
    log["known_objects"] = test_no_masks(world, L, 100, seed * 100 + 40, "pool", bounds)
    log["new_objects"] = test_no_masks(world, L, 100, seed * 100 + 30, "fresh", bounds)
    log["new_objects_mask_at_test"] = R.test_effects(world, L, 100, seed * 100 + 30, "fresh", bounds)
    log["seconds"] = round(time.time() - t0)
    return log


if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]; n_train = int(name.split("_n")[1].split("_")[0]) if "_n" in name else 100; res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by, n_train)); r = res[-1]
        print(json.dumps({q: v for q, v in r.items() if q not in ("new_objects", "known_objects", "new_objects_mask_at_test", "class_log")}))
        for part in ("known_objects", "new_objects"):
            a = r[part]; print("  %s: bound %s (own pattern %s) by behaviour %s" % (part, a["objects_bound"], a["bound_to_its_own_pattern"], a["bound_by_behaviour"]))
            for k, v in a["rows"].items(): print("     effects %-24s" % k, v)
            for k, v in a["take"].items(): print("     take    %-24s" % k, v)
        a = r["new_objects_mask_at_test"]
        for k, v in a["rows"].items(): print("  new, mask at test: effects %-20s" % k, v)
        for k, v in a["take"].items(): print("  new, mask at test: take    %-20s" % k, v)
        sys.stdout.flush()
        json.dump(res, open("runs/155/%s_%s.json" % (name, by), "w"), indent=1)
