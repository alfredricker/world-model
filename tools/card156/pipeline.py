"""Card 156, end to end: experience (the agent's own, or card 152's script) -> binding with feedback and appearance
couplings -> stored patterns -> object modules and dynamics classes (cards 152-153.1, wired as in card 155) -> card
152's tests on objects never met and on known ones, with no mask anywhere.

At test an object is whatever the binding layer makes of the first frame: a stored pattern recognised there (its
module has its own transitions), or a group bound by the appearance couplings (its appearance features are read
from the group).  What is left in the ground is not an object: predicted to stay put, "take it" answered "cannot".

usage: pipeline.py <name> <by: shape|colour> <world seeds...>     writes runs/156/<name>_<by>.json
       name holds the arm: "agent" (its own experience), "script" (card 152's script), "bound" (script, with the
       appearance couplings taught from the simulator's masks); "_nNN" sets the training episodes (100)
"""
import sys, json, time, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "tools/card156")
import agent as Ag
Le, Ap, Fb = Ag.Le, Ag.Ap, Ag.Fb
sys.path.insert(0, "tools/card155")
import pipeline as P
Cl, S, Kd, W, M, R = P.Cl, P.S, P.Kd, P.W, P.M, P.R
assert W is Fb.W
pc = R.pc; H = Fb.H


def group_appearance(frame, m):
    m = Fb.in_frame(m); ys, xs = np.nonzero(m); y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    img = torch.tensor(frame.astype(np.float32) / 255)
    return dict(sil=F.adaptive_avg_pool2d(torch.tensor(m[y0:y1, x0:x1], dtype=torch.float32)[None], 8)[0].reshape(-1), colour=img[torch.tensor(m)].mean(0))


def learn(world, n_train, seed, log, arm):
    mem = Fb.Memory(); net = Ap.Pairs(); T = W.T; t0 = time.time(); tr = []; recs = []; new = []; tally = {}
    rng = np.random.default_rng(seed)
    script = Fb.episodes(world, n_train, seed) if arm != "agent" else None
    for e in range(n_train):
        P0 = len(mem)
        if arm == "agent":
            ep, Sn, lg = Ag.live_episode(world, e, rng, mem, net, seed=e); rec = Sn.rec
        else:
            ep = script[e]; _, rec, lg = Le.run_episode(ep, e, mem, net, seed=e, labels="true" if arm == "bound" else "own")
        tr.append(ep); recs.append(rec); new.append((P0, len(mem)))
        for k, v in lg.items():
            tally[k] = tally.get(k, 0) + v
        if e % 10 == 9:
            net.fit(300)
    G, O, m = W.tokens(tr); pat = np.full((len(tr), T, 4), -1); true_of = {}
    for e, ep in enumerate(tr):
        n = len(ep["objs"]); best = {i: (0, -1, 0) for i in range(n)}
        for i, p in enumerate(P.owner(recs[e], ep["masks"][0], n)):
            if p >= 0:
                best[i] = (10 ** 6, int(p), 0)
        for p in range(*new[e]):
            q = mem.info[p]
            if q.get("tid", 0) >= 3 and q["n"] * q["iou"] > best[q["tid"] - 3][0]:
                best[q["tid"] - 3] = (q["n"] * q["iou"], p, q["first"] - 1)
        for i, (_, p, a) in best.items():
            if p >= 0:
                pat[e, a:, i] = p; true_of[p] = ep["objs"][i]
    used = np.unique(pat[pat >= 0]); idx_of = {int(p): j for j, p in enumerate(used)}; n_ind = len(used)
    ind = np.vectorize(lambda p: idx_of.get(int(p), -1))(pat).reshape(-1, 4); m2 = m & (ind >= 0)
    app = [P.appearance(mem, p) for p in range(len(mem))]
    sil = torch.stack([app[p]["sil"] for p in used]); col = torch.stack([app[p]["colour"] for p in used])
    shape = M.Proto(R.TH_SHAPE).learn(sil); colour = M.Proto(R.TH_COLOUR).learn(col)
    s_code = shape.energy(sil).argmin(1).numpy(); c_code = colour.energy(col).argmin(1).numpy(); klog = []
    exp_s = round(time.time() - t0)
    dyn, kind, E = Cl.learn_classes(S.make, G, O, m2, ind, n_ind, (s_code, c_code), (len(shape.P), len(colour.P)), log=klog)
    K = dyn.K; looks, w = Kd.fit_looks(s_code, c_code, E, [len(shape.P), len(colour.P), K]) if K > 1 else (None, np.ones(n_ind))
    true = np.array([true_of[int(p)]["kind"] for p in used])
    table = np.array([[float(w[(kind == k) & (true == t)].sum()) for t in range(3)] for k in range(K)])
    moved = sum(int(np.abs(np.diff(ep["pos"][:, i].astype(int), axis=0)).sum() > 0) for ep in tr for i in range(len(ep["objs"])))
    movable = sum(o["kind"] != 1 for ep in tr for o in ep["objs"])
    log.update(arm=arm, train_episodes=n_train, patterns=len(mem), patterns_of_still_groups=sum(bool(q.get("still")) for q in mem.info), experience=tally,
               objects_in_training=sum(len(ep["objs"]) for ep in tr), movable_objects_in_training=int(movable), objects_moved_in_training=int(moved),
               objects_with_transitions=n_ind, objects_by_true_behaviour=np.bincount(true, minlength=3).tolist(),
               informative_used=int(M.informative(O, m2).sum()), informative_with_masks=int(M.informative(O, m).sum()),
               shape_codes=len(shape.P), colour_codes=len(colour.P), classes=K, class_sizes=np.bincount(kind, minlength=K).tolist(), class_log=klog,
               classes_by_true_behaviour=table.round(1).tolist(), class_purity=pc([table.max(1).sum() / max(table.sum(), 1e-9)]),
               appearance_share=looks.share() if looks else None, experience_seconds=exp_s)
    return dict(me=P.NoOne(), of_uid={}, shape=shape, colour=colour, dyn=dyn, kind=kind, mis=E, eps=1 / (1 + np.e), looks=looks, G=G, O=O, m=m, tr=tr,
                true_of_kind=table.argmax(1), mem=mem, net=net, idx_of=idx_of, app=app)


def test(world, L, n, seed, source, bounds):
    """card 152's tests with no mask: the objects are what the binding layer makes of the first frame"""
    sizes = W.SIZES + W.NEW_SIZES if source == "fresh" else W.SIZES
    te = Fb.episodes(world, n, seed, source, sizes); G, O, m = W.tokens(te); K = L["dyn"].K; T = W.T
    info = M.informative(O, m)
    per_obj = lambda f: np.array([[f(o) for o in ep["objs"]] + [0] * (4 - len(ep["objs"])) for ep in te])
    kind_e, exc_e = per_obj(lambda o: o["kind"]), per_obj(lambda o: o["kind"] != o["rule"]).astype(bool)
    bound = np.zeros((len(te), 4), bool); w_app = np.full((len(te), 4, K), 1.0 / K, np.float32); w_all = w_app.copy(); found = []; ground = []; by_pattern = 0
    for e, ep in enumerate(te):
        lab, rec, _, _ = Ap.first_sight(ep["frames"][0], L["mem"], L["net"], 5000 + e); mask = ep["masks"][0]; th = Fb.things_of(ep["objs"], "x")
        rows, sp = Fb.score_frame(lab, rec, L["mem"], mask, th); ground.append(sp)
        found += [(r["found"], ep["objs"][i - 3]["kind"], ep["objs"][i - 3]["size"]) for r, (i, _, _) in zip(rows, th) if i >= 3]
        bgl = np.bincount(lab.ravel()).argmax(); own = P.owner(rec, mask, len(ep["objs"]))
        for i in range(len(ep["objs"])):
            mi = mask == 3 + i; k = np.bincount(lab[mi]).argmax()
            if own[i] >= 0:
                p = int(own[i]); v = [L["app"][p]]; known = [L["idx_of"].get(p, -1)]; by_pattern += 1
            elif k != bgl and (lab[mi] == k).sum() >= 0.25 * mi.sum():
                v = [group_appearance(ep["frames"][0], lab == k)]; known = None
            else:
                continue
            bound[e, i] = True; w_app[e, i] = R.kind_belief(L, v)[0]; w_all[e, i] = R.kind_belief(L, v, known)[0] if known else w_app[e, i]
    rep = lambda a: a.repeat(T, 0)
    still = (O[..., 2:] != np.array([4, 4, 0])).any(-1) & m
    bad = lambda w: np.where(rep(bound), (M.vote(L["dyn"], G, O, m, rep(w)) != O[..., 2:]).any(-1) & m, still)
    arms = {"appearance and own transitions": bad(w_all), "appearance only": bad(w_app),
            "true behaviour given": M.wrong(bounds["true"], G, O, m, np.eye(3, dtype=np.float32)[rep(kind_e)]),
            "one class for all": M.wrong(bounds["one"], G, O, m, np.ones((len(G), 4, 1), np.float32))}
    n_obj = int(sum(len(ep["objs"]) for ep in te))
    out = dict(objects=n_obj, objects_bound=pc([bound[e, i] for e, ep in enumerate(te) for i in range(len(ep["objs"]))]), bound_by_a_stored_pattern=round(100 * by_pattern / n_obj, 1),
               labels_on_the_ground_per_scene=round(float(np.mean(ground)), 3), found={}, bound_by_behaviour={}, rows={})
    for k, nm in enumerate(("takeable", "fixed", "sticky")):
        out["found"][nm] = dict(n=sum(q[1] == k for q in found), found=pc([q[0] for q in found if q[1] == k]))
        out["bound_by_behaviour"][nm] = pc([bound[e, i] for e, ep in enumerate(te) for i, o in enumerate(ep["objs"]) if o["kind"] == k])
    mv = [q for q in found if q[1] != 1]
    out["found"]["movable"] = dict(n=len(mv), found=pc([q[0] for q in mv])); out["found"]["movable, a width never seen"] = dict(n=sum(q[2] in W.NEW_SIZES for q in mv), found=pc([q[0] for q in mv if q[2] in W.NEW_SIZES]))
    cuts = {"all": info, "exceptions": info & rep(exc_e), "bound": info & rep(bound), "not bound": info & ~rep(bound)}
    for k, nm in enumerate(("takeable", "fixed", "sticky")):
        cuts[nm] = info & (rep(kind_e) == k)
    for name, c in cuts.items():
        if c.sum():
            out["rows"][name] = dict(n=int(c.sum()), **{a: pc(~wr[c]) for a, wr in arms.items()})
    out["take"] = P.test_take(te, L, bounds, w_app, w_all, bound)
    return out


def run_world(seed, by, n_train, arm):
    t0 = time.time(); world = W.make_world(seed, by); log = dict(world=seed, by=by, kind_of=world["kind_of"].tolist())
    L = learn(world, n_train, seed * 100 + 10, log, arm)
    kd = np.array([[o["kind"] for o in ep["objs"]] + [0] * (4 - len(ep["objs"])) for ep in L["tr"]]).repeat(W.T, 0)
    bounds = dict(true=M.Dyn().fit(L["G"], L["O"], L["m"], kd, 3), one=M.Dyn().fit(L["G"], L["O"], L["m"], 0 * kd, 1))
    log["new_objects"] = test(world, L, 100, seed * 100 + 30, "fresh", bounds)
    log["known_objects"] = test(world, L, 100, seed * 100 + 40, "pool", bounds)
    log["seconds"] = round(time.time() - t0); log["peak_gpu_mb"] = int(torch.cuda.max_memory_allocated() // 2 ** 20)
    return log


if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]; n_train = int(name.split("_n")[1].split("_")[0]) if "_n" in name else 100
    arm = "agent" if "agent" in name else "bound" if "bound" in name else "script"; res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by, n_train, arm)); r = res[-1]
        print(json.dumps({q: v for q, v in r.items() if q not in ("new_objects", "known_objects", "class_log")}))
        for part in ("new_objects", "known_objects"):
            a = r[part]; A = "appearance and own transitions"
            print("  %s: bound %s (by a pattern %s) found %s ground labels %s" % (part, a["objects_bound"], a["bound_by_a_stored_pattern"], {k: v["found"] for k, v in a["found"].items()}, a["labels_on_the_ground_per_scene"]))
            print("     effects", {k: (v[A], v["true behaviour given"]) for k, v in a["rows"].items()})
            print("     take   ", {k: (v[A], v["one class for all"]) for k, v in a["take"].items()})
        sys.stdout.flush()
        json.dump(res, open("runs/156/%s_%s.json" % (name, by), "w"), indent=1)
