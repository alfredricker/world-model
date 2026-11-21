"""Card 153.1 on the sprite worlds: card 153's run with the dynamics classes inferred by tools/card153.1/classes.py.
usage: sprites.py <name> <by: shape|colour> <world seeds...>     writes runs/153.1/<name>_<by>.json   ("_nNN": training episodes)
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "tools/card153.1"); sys.path.insert(0, "tools/card153")
import classes as Cl, sprites as S
Kd, W, M, R = S.Kd, S.W, S.M, S.R


def learn(world, n_train, seed, log, rule):
    tr = W.collect(world, n_train, seed); V = R.first_views(tr)
    me = M.Itself(R.TH_SELF); ind = np.full((len(tr), 4), -1); of_uid = {}; first = []
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
    G, O, m = W.tokens(tr); klog = []
    dyn, kind, E = Cl.learn_classes(S.make, G, O, m, ind.repeat(W.T, 0), n_ind, (s_code, c_code), (len(shape.P), len(colour.P)), log=klog)
    K = dyn.K; looks, w = Kd.fit_looks(s_code, c_code, E, [len(shape.P), len(colour.P), K]) if K > 1 else (None, np.ones(n_ind))
    true = np.full(n_ind, -1)
    for e, ep in enumerate(tr):
        for i, o in enumerate(ep["objs"]):
            true[ind[e, i]] = o["kind"]
    table = np.array([[float(w[(kind == k) & (true == t)].sum()) for t in range(3)] for k in range(K)])
    log.update(train_episodes=n_train, individuals_stored=n_ind, kinds=K, kind_sizes=np.bincount(kind, minlength=K).tolist(), kind_log=klog,
               individuals_whose_records_tell=round(float(w.sum()), 1), kinds_by_true_behaviour=table.round(1).tolist(),
               kind_purity=R.pc([table.max(1).sum() / max(table.sum(), 1e-9)]), looks_share=looks.share() if looks else None)
    return dict(me=me, of_uid=of_uid, shape=shape, colour=colour, dyn=dyn, kind=kind, mis=E, eps=1 / (1 + np.e), looks=looks, G=G, O=O, m=m, tr=tr, true_of_kind=table.argmax(1))


S.learn = learn
if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]; n_train = int(name.split("_n")[1].split("_")[0]) if "_n" in name else 300; res = []
    for s in sys.argv[3:]:
        res.append(S.run_world(int(s), by, n_train, "153.1")); r = res[-1]
        print(json.dumps({q: v for q, v in r.items() if q not in ("new_objects", "known_objects", "kind_log")}))
        print("  log", json.dumps(r["kind_log"])); print("  new objects: effects", r["new_objects"]["rows"]["all"], "| take", r["new_objects"]["take"]["all"], flush=True)
        json.dump(res, open("runs/153.1/%s_%s.json" % (name, by), "w"), indent=1)
