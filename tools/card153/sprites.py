"""Card 153 on card 152's sprite worlds: the same object module and tests, with the kinds found by tools/card153/kinds.py.

usage: sprites.py <name> <by: shape|colour> <world seeds...>     writes runs/153/<name>_<by>.json
       "_nNN" in the name sets the training episodes (300); "old" in the name uses card 152's kind learner
       (with this card's looks-to-kind couplings), to tell the two changes apart.
"""
import sys, json, time, math, importlib.util, numpy as np, torch
sys.path.insert(0, "tools/card153")
import kinds as Kd
import world as W, modules as M
spec = importlib.util.spec_from_file_location("run152", "tools/card152/run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

L1_OBJ = 1e-3
make = lambda: Kd.Couplings(M.GV, M.OV, (2, 3, 4))


def learn(world, n_train, seed, log, rule):
    tr = W.collect(world, n_train, seed); V = R.first_views(tr)
    me = M.Itself(R.TH_SELF); ind = np.full((len(tr), 4), -1); of_uid = {}; first = []          # what: as card 152
    for e, ep in enumerate(tr):
        got = me.settle(V[e]) if len(me) else [dict(e=9.0)] * len(V[e])
        for i, (v, g) in enumerate(zip(V[e], got)):
            if g["e"] < R.TH_SELF:
                ind[e, i] = g["k"]
            else:
                ind[e, i] = me.store(v); first.append(v)
            if ep["objs"][i]["uid"] >= 0:
                of_uid.setdefault(ep["objs"][i]["uid"], ind[e, i])
    n_ind = len(me)
    shape = M.Proto(R.TH_SHAPE).learn(torch.stack([v["sil"] for v in first])); colour = M.Proto(R.TH_COLOUR).learn(torch.stack([v["colour"] for v in first]))
    s_code = shape.energy(torch.stack([v["sil"] for v in first])).argmin(1).numpy(); c_code = colour.energy(torch.stack([v["colour"] for v in first])).argmin(1).numpy()
    G, O, m = W.tokens(tr); ind_rows = ind.repeat(W.T, 0); klog = []
    if rule == "new":
        dyn, kind, E = Kd.learn_kinds(make, G, O, m, ind_rows, n_ind, L1_OBJ, log=klog)
    else:
        dyn, kind, mis, _, eps = M.learn_kinds(G, O, m, ind_rows, n_ind, log=klog); E = math.log((1 - eps) / eps) * mis
    K = dyn.K; looks, w = Kd.fit_looks(s_code, c_code, E, [len(shape.P), len(colour.P), K]) if K > 1 else (None, np.ones(n_ind))
    true = np.full(n_ind, -1)
    for e, ep in enumerate(tr):
        for i, o in enumerate(ep["objs"]):
            true[ind[e, i]] = o["kind"]
    table = np.array([[float(w[(kind == k) & (true == t)].sum()) for t in range(3)] for k in range(K)])
    log.update(train_episodes=n_train, individuals_stored=n_ind, kinds=K, kind_sizes=np.bincount(kind, minlength=K).tolist(), kind_log=klog,
               individuals_whose_records_tell=round(float(w.sum()), 1), kinds_by_true_behaviour=table.round(1).tolist(),
               kind_purity=R.pc([table.max(1).sum() / max(table.sum(), 1e-9)]), looks_share=looks.share() if looks else None,
               looks_strength=round(looks.strength, 2) if looks else None)
    return dict(me=me, of_uid=of_uid, shape=shape, colour=colour, dyn=dyn, kind=kind, mis=E, eps=1 / (1 + math.e), looks=looks, G=G, O=O, m=m, tr=tr,
                true_of_kind=table.argmax(1))


def run_world(seed, by, n_train, rule):
    t0 = time.time(); world = W.make_world(seed, by); log = dict(world=seed, by=by, rule=rule, kind_of=world["kind_of"].tolist())
    L = learn(world, n_train, seed * 100 + 10, log, rule)
    kd = np.array([[o["kind"] for o in ep["objs"]] + [0] * (4 - len(ep["objs"])) for ep in L["tr"]]).repeat(W.T, 0)
    bounds = dict(true=M.Dyn().fit(L["G"], L["O"], L["m"], kd, 3), one=M.Dyn().fit(L["G"], L["O"], L["m"], 0 * kd, 1))
    log["new_objects"] = R.test_effects(world, L, 100, seed * 100 + 30, "fresh", bounds)
    log["known_objects"] = R.test_effects(world, L, 100, seed * 100 + 40, "pool", bounds)
    log["seconds"] = round(time.time() - t0)
    return log


if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]
    n_train = int(name.split("_n")[1].split("_")[0]) if "_n" in name else 300
    res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by, n_train, "old" if "old" in name else "new")); r = res[-1]; a = r["new_objects"]; k = r["known_objects"]
        print(json.dumps({q: v for q, v in r.items() if q not in ("new_objects", "known_objects", "kind_log")}))
        print("  kind log", json.dumps(r["kind_log"]))
        print("  new objects: effects", a["rows"]["all"], "| take", a["take"]["all"])
        print("  known exceptions: effects", k["rows"].get("exceptions"), "| take", k["take"].get("exceptions"), flush=True)
        json.dump(res, open("runs/153/%s_%s.json" % (name, by), "w"), indent=1)
