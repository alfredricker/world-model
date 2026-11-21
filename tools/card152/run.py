"""Card 152: learn the object module's memories from the agent's own episodes in one world, then test it.

usage: run.py <name> <by: shape|colour> <world seeds...>     writes runs/152/<name>_<by>.json
"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "tools/card152")
import world as W, modules as M

TH_SHAPE, TH_COLOUR, TH_SELF = 1.0, 0.05, 0.004
pc = lambda x: round(100 * float(np.mean(x)), 1) if len(x) else None


def first_views(eps):
    return [M.views(ep["frame"], ep["mask"], len(ep["objs"])) for ep in eps]


def learn(world, n_train, seed, log):
    tr = W.collect(world, n_train, seed); V = first_views(tr)
    # what: itself.  Each object at first sight is matched against every stored individual, and stored if none fits.
    me = M.Itself(TH_SELF); ind = np.full((len(tr), 4), -1); of_uid = {}; first = []; mistakes = 0
    for e, ep in enumerate(tr):
        got = me.settle(V[e]) if len(me) else [dict(e=9.0)] * len(V[e])
        for i, (v, g) in enumerate(zip(V[e], got)):
            if g["e"] < TH_SELF:
                ind[e, i] = g["k"]
            else:
                ind[e, i] = me.store(v); first.append(v)
            u = ep["objs"][i]["uid"]
            if u >= 0:
                mistakes += of_uid.setdefault(u, ind[e, i]) != ind[e, i]
            else:
                mistakes += g["e"] < TH_SELF
    n_ind = len(me)
    # what: shape and colour, from the same first sights
    shape = M.Proto(TH_SHAPE).learn(torch.stack([v["sil"] for v in first])); colour = M.Proto(TH_COLOUR).learn(torch.stack([v["colour"] for v in first]))
    s_code = shape.energy(torch.stack([v["sil"] for v in first])).argmin(1).numpy(); c_code = colour.energy(torch.stack([v["colour"] for v in first])).argmin(1).numpy()
    # does: kinds of behaviour from the records
    G, O, m = W.tokens(tr); ind_rows = ind.repeat(W.T, 0); klog = []
    dyn, kind, mis, n_info, eps = M.learn_kinds(G, O, m, ind_rows, n_ind, log=klog)
    K = dyn.K; srt = np.sort(mis, 1)
    sure = (srt[:, 1] - srt[:, 0] >= 1) if K > 1 else np.zeros(n_ind, bool)
    looks = M.Looks().fit(np.stack([s_code, c_code, kind], 1)[sure], [len(shape.P), len(colour.P), K]) if sure.sum() > 5 else None
    # how the learned kinds line up with the true behaviours (for the report only)
    true = np.full(n_ind, -1)
    for e, ep in enumerate(tr):
        for i, o in enumerate(ep["objs"]):
            true[ind[e, i]] = o["kind"]
    table = np.array([[int(((kind == k) & (true == t) & sure).sum()) for t in range(3)] for k in range(K)])
    log.update(train_episodes=n_train, individuals_stored=n_ind, identity_mistakes_in_training=int(mistakes), shape_codes=len(shape.P), colour_codes=len(colour.P),
               kinds=K, kind_rounds=klog, individuals_with_settled_kind=int(sure.sum()), kinds_by_true_behaviour=table.tolist(),
               kind_purity=pc([table.max(1).sum() / max(table.sum(), 1)]), record_noise=round(eps, 4), looks_share=looks.share() if looks else None)
    return dict(me=me, of_uid=of_uid, shape=shape, colour=colour, dyn=dyn, kind=kind, mis=mis, eps=eps, looks=looks, G=G, O=O, m=m, tr=tr, true_of_kind=table.argmax(1))


def kind_belief(L, vs, known=None):
    """belief over kinds for each view: from its looks, plus the records of the individual it is recognised as"""
    K = L["dyn"].K
    ps = L["shape"].belief(torch.stack([v["sil"] for v in vs])); pcl = L["colour"].belief(torch.stack([v["colour"] for v in vs]))
    f = L["looks"].field(ps, pcl) if L["looks"] else torch.zeros(len(vs), K)
    if known is not None:
        lam = float(np.log((1 - L["eps"]) / L["eps"]))
        for j, k in enumerate(known):
            if k >= 0:
                f[j] = f[j] - lam * torch.tensor(L["mis"][k], dtype=torch.float32)
    return torch.softmax(f, -1).numpy()


def test_recognition(world, L, n, seed):
    sc = W.scenes(world, n, seed); rows = []
    for s in sc:
        vs = M.views(s["frame"], s["mask"], len(s["objs"]))
        keep = [i for i, v in enumerate(vs) if v is not None and v["n"] >= 8 and v["n"] >= 0.1 * s["objs"][i]["mask"].sum()]
        if not keep:
            continue
        got = L["me"].settle([vs[i] for i in keep])
        for i, g in zip(keep, got):
            o = s["objs"][i]; truth = L["of_uid"].get(o["uid"], -1)
            ans = lambda e, k: k if e < TH_SELF else -1
            a = ans(g["e"], g["k"])
            place = (vs[i]["box"][0] + g["shift"][0], vs[i]["box"][1] + g["shift"][1])
            rows.append((truth >= 0, 1 - vs[i]["n"] / o["mask"].sum(), a == truth, ans(g["e0"], g["k0"]) == truth, ans(g["ea"], g["ka"]) == truth,
                         a == truth and truth >= 0 and place == tuple(s["pos"][i]), g["e"]))
    r = np.array(rows, float); known = r[:, 0] == 1; hid = r[:, 1]
    cut = {"all": hid >= 0, "wholly seen": hid == 0, "up to a quarter hidden": (hid > 0) & (hid < 0.25), "a quarter or more hidden": hid >= 0.25, "half or more hidden": hid >= 0.5}
    out = {"views": int(len(r)), "known": int(known.sum()), "new": int((~known).sum())}
    for name, c in cut.items():
        out[name] = dict(n=int(c.sum()), right=pc(r[c, 2]), right_known=pc(r[c & known, 2]), right_new=pc(r[c & ~known, 2]),
                         where_not_settled=pc(r[c, 3]), no_object_frame=pc(r[c, 4]), no_object_frame_known=pc(r[c & known, 4]), place_right_known=pc(r[c & known, 5]))
    out["energy_of_right_known"] = [round(float(np.percentile(r[known & (r[:, 2] == 1), 6], q)), 5) for q in (50, 99, 100)]
    out["energy_of_new"] = [round(float(np.percentile(r[~known, 6], q)), 5) for q in (0, 1, 50)]
    return out


def test_effects(world, L, n, seed, source, bounds):
    """effects predicted before any interaction in the episode: the kind comes from the first frame alone"""
    sizes = W.SIZES + W.NEW_SIZES if source == "fresh" else W.SIZES
    te = W.collect(world, n, seed, source=source, sizes=sizes); V = first_views(te); G, O, m = W.tokens(te); K = L["dyn"].K
    info = M.informative(O, m)
    per_obj = lambda f: np.array([[f(o) for o in ep["objs"]] + [0] * (4 - len(ep["objs"])) for ep in te]).repeat(W.T, 0)
    kind_true, exc, new_size = per_obj(lambda o: o["kind"]), per_obj(lambda o: o["kind"] != o["rule"]).astype(bool), per_obj(lambda o: o["size"] in W.NEW_SIZES).astype(bool)
    w_looks = np.zeros((len(te), 4, K), np.float32); w_all = np.zeros((len(te), 4, K), np.float32); recog = []
    for e, ep in enumerate(te):
        got = L["me"].settle(V[e]); known = [g["k"] if g["e"] < TH_SELF else -1 for g in got]
        w_looks[e, :len(V[e])] = kind_belief(L, V[e]); w_all[e, :len(V[e])] = kind_belief(L, V[e], known)
        recog += [(k >= 0) == (o["uid"] >= 0) and (k < 0 or k == L["of_uid"].get(o["uid"], -2)) for k, o in zip(known, ep["objs"])]
    rep = lambda w: w.repeat(W.T, 0)
    hard = lambda w: np.eye(K, dtype=np.float32)[w.argmax(-1)]
    bad = lambda w: (M.vote(L["dyn"], G, O, m, rep(w)) != O[..., 2:]).any(-1) & m
    arms = {"looks and own records": bad(w_all), "looks only": bad(w_looks),
            "looks only, winner passed": M.wrong(L["dyn"], G, O, m, rep(hard(w_looks))), "looks only, couplings averaged": M.wrong(L["dyn"], G, O, m, rep(w_looks)),
            "true behaviour given": M.wrong(bounds["true"], G, O, m, np.eye(3, dtype=np.float32)[kind_true]),
            "one kind for all": M.wrong(bounds["one"], G, O, m, np.ones((len(G), 4, 1), np.float32))}
    kind_right = (L["true_of_kind"][rep(w_looks).argmax(-1)] == kind_true)
    out = {"informative_object_steps": int(info.sum()), "recognised_right": pc(recog), "kind_from_looks_right": pc(kind_right[info]), "rows": {}}
    cuts = {"all": info, "follow their looks": info & ~exc, "exceptions": info & exc, "a width never seen": info & new_size, "a width seen": info & ~new_size}
    for k, nm in enumerate(("takeable", "fixed", "sticky")):
        cuts[nm] = info & (kind_true == k)
    for name, c in cuts.items():
        if c.sum():
            out["rows"][name] = dict(n=int(c.sum()), **{a: pc(~wr[c]) for a, wr in arms.items()})
    out["take"] = test_take(te, L, bounds, w_looks, w_all)
    return out


def test_take(te, L, bounds, w_looks, w_all):
    """The hand is put at the side of each object of each episode's first state and the module is asked for the action
    that leaves the object held.  Each kind proposes an action and checks it forward with its own couplings ("cannot"
    if the object does not come out held); the answers are weighted by the belief over kinds.  Scored in the simulator:
    the object is held after the action, or, for a fixed object, the answer is "cannot"."""
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

    def score(model, w):
        ans = answers(model); ok = []
        for n, (e, i, st) in enumerate(cases):
            tally = {}
            for k in range(model.K):
                tally[ans[k][n]] = tally.get(ans[k][n], 0) + float(w[n][k])
            a = max(tally, key=tally.get); kind = te[e]["objs"][i]["kind"]
            ok.append(a is None if kind == 1 else (a is not None and W.step(st, a, te[e]["objs"])["held"] == i))
        return np.array(ok)
    kinds = np.array([te[e]["objs"][i]["kind"] for e, i, _ in cases]); exc = np.array([te[e]["objs"][i]["kind"] != te[e]["objs"][i]["rule"] for e, i, _ in cases])
    arms = {"looks and own records": score(L["dyn"], [w_all[e, i] for e, i, _ in cases]), "looks only": score(L["dyn"], [w_looks[e, i] for e, i, _ in cases]),
            "true behaviour given": score(bounds["true"], [np.eye(3)[k] for k in kinds]), "one kind for all": score(bounds["one"], [[1.0]] * N)}
    out = {}
    for name, c in (("all", kinds >= 0), ("takeable", kinds == 0), ("fixed", kinds == 1), ("sticky", kinds == 2), ("exceptions", exc)):
        if c.sum():
            out[name] = dict(n=int(c.sum()), **{a: pc(v[c]) for a, v in arms.items()})
    return out


def run_world(seed, by, n_train=300):
    t0 = time.time(); world = W.make_world(seed, by); log = dict(world=seed, by=by, kind_of=world["kind_of"].tolist())
    L = learn(world, n_train, seed * 100 + 10, log)
    kd = np.array([[o["kind"] for o in ep["objs"]] + [0] * (4 - len(ep["objs"])) for ep in L["tr"]]).repeat(W.T, 0)
    bounds = dict(true=M.Dyn().fit(L["G"], L["O"], L["m"], kd, 3), one=M.Dyn().fit(L["G"], L["O"], L["m"], 0 * kd, 1))
    log["recognition"] = test_recognition(world, L, 600, seed * 100 + 20)
    log["new_objects"] = test_effects(world, L, 100, seed * 100 + 30, "fresh", bounds)
    log["known_objects"] = test_effects(world, L, 100, seed * 100 + 40, "pool", bounds)
    log["seconds"] = round(time.time() - t0)
    return log


if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]
    n_train = int(name.split("_n")[1]) if "_n" in name else 300
    res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by, n_train)); r = res[-1]
        print(json.dumps({k: v for k, v in r.items() if k not in ("recognition", "new_objects", "known_objects")}))
        print("  recognition", json.dumps(r["recognition"]))
        print("  new objects", json.dumps(r["new_objects"]))
        print("  known objects", json.dumps(r["known_objects"]), flush=True)
        json.dump(res, open("runs/152/%s_%s.json" % (name, by), "w"), indent=1)
