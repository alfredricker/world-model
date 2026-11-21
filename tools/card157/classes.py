"""Card 157, stage 2: the dynamics classes fitted on one saved experience (stage 1) with each object's transitions
read from the binding layer's own states ("own") and, on the same experience, from the simulator ("sim", card 156's
rule), and the test of the action that brings a never-met object along.

own tokens, for a stored pattern present in an episode, at each step (tokens as card 152's, the two "held" dropped):
  contact        the box around the pattern's points lies side by side with the hand's box
  displacement   the commonest displacement its points were seen to make
  came along     that displacement is the hand's own and is not zero
A pattern whose box lies on the hand's box is the hand, and is left out (the agent knows where its hand is).

usage: classes.py <name> <by: shape|colour> <world seeds...>     writes runs/157/<name>_<by>.json
"""
import sys, json, time, importlib.util, numpy as np, torch
spec = importlib.util.spec_from_file_location("pipeline156", "tools/card156/pipeline.py"); P6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P6)
Ag, Ap, Fb, P, Cl, S, Kd, W, M, R = P6.Ag, P6.Ap, P6.Fb, P6.P, P6.Cl, P6.S, P6.Kd, P6.W, P6.M, P6.R
pc = R.pc; T, H, HAND = W.T, W.H, W.HAND; DISP = np.array(Ag.M.DISP)
TOL = 0
CHECK_NO_GROUND = bool(__import__("os").environ.get("NO_GROUND"))


def true_kind(X, p):
    """for scoring only: the behaviour of the true object a pattern was stored on; 3 for the ground, 4 for the hand"""
    q = X["mem"].info[p]; tid = q.get("tid", 0)
    return X["tr"][q["episode"]]["objs"][tid - 3]["kind"] if tid >= 3 else 3 if tid == 0 else 4


def sim_tokens(X):
    tr, mem = X["tr"], X["mem"]; G, O, m = W.tokens(tr); pat = np.full((len(tr), T, 4), -1)
    for e, ep in enumerate(tr):
        n = len(ep["objs"]); best = {i: (0, -1, 0) for i in range(n)}
        for i, p in enumerate(X["own0"][e]):
            if p >= 0:
                best[i] = (10 ** 6, int(p), 0)
        for p in range(*X["new"][e]):
            q = mem.info[p]
            if q.get("tid", 0) >= 3 and q["n"] * q["iou"] > best[q["tid"] - 3][0]:
                best[q["tid"] - 3] = (q["n"] * q["iou"], p, q["first"] - 1)
        for i, (_, p, a) in best.items():
            if p >= 0:
                pat[e, a:, i] = p
    pat = pat.reshape(-1, 4)
    return G, O, m & (pat >= 0), pat, {}


def touching(hand, box, tol=TOL):
    hy, hx = hand; y0, x0, y1, x1 = box
    rows = hy < y1 and y0 < hy + HAND; cols = hx < x1 and x0 < hx + HAND
    near = lambda g: 0 <= g <= tol
    return bool((rows and (near(x0 - hx - HAND) or near(hx - x1))) or (cols and (near(y0 - hy - HAND) or near(hy - y1))))


def own_tokens(X, tol=TOL):
    tr = X["tr"]; Gs, Os, _ = W.tokens(tr); Gr, Or, Mr, Pr = [], [], [], []
    agree = dict(steps=0, contact_both=0, contact_sim_only=0, contact_own_only=0, moved_sim=0, moved_sim_same=0, still_sim=0, still_sim_same=0); hands = 0
    for e, ep in enumerate(tr):
        objs = []; slot_of = {int(p): i for i, p in enumerate(X["own0"][e]) if p >= 0}
        for s in X["seen"][e]:
            t0 = s["t0"]; y0, x0, y1, x1 = s["box"][t0]; hy, hx = ep["hand"][t0]
            inter = max(0, min(y1, hy + HAND) - max(y0, hy)) * max(0, min(x1, hx + HAND) - max(x0, hx))
            if s["n"][t0] == 0 or inter > 0.5 * (y1 - y0) * (x1 - x0):
                hands += s["n"][t0] > 0; continue
            if CHECK_NO_GROUND and true_kind(X, s["p"]) >= 3:          # a check only (it reads the simulator): patterns on the ground left out
                continue
            objs.append(s)
        for c in range(0, len(objs), 4):
            G = Gs[e * T:(e + 1) * T].copy(); O = np.zeros((T, 4, 5), int); m = np.zeros((T, 4), bool); pp = np.full((T, 4), -1)
            for j, s in enumerate(objs[c:c + 4]):
                q = X["mem"].info[s["p"]]; i = slot_of.get(s["p"], q["tid"] - 3 if q["episode"] == e and q.get("tid", 0) >= 3 else -1)     # scoring only
                for t in range(s["t0"], T):
                    if s["n"][t] < 4 or s["code"][t] < 0:
                        continue
                    d = np.clip(DISP[s["code"][t]], -4, 4); dh = G[t, :2] - 4
                    O[t, j] = (touching(ep["hand"][t], s["box"][t], tol), 0, d[0] + 4, d[1] + 4, int((d == dh).all() and dh.any())); m[t, j] = True; pp[t, j] = s["p"]
                    if i >= 0:
                        so = Os[e * T + t, i]; agree["steps"] += 1; mv = (so[2:4] != 4).any()
                        agree["contact_both"] += int(so[0] and O[t, j, 0]); agree["contact_sim_only"] += int(so[0] and not O[t, j, 0]); agree["contact_own_only"] += int(O[t, j, 0] and not so[0])
                        agree["moved_sim" if mv else "still_sim"] += 1; agree["moved_sim_same" if mv else "still_sim_same"] += int((so[2:4] == O[t, j, 2:4]).all())
            Gr.append(G); Or.append(O); Mr.append(m); Pr.append(pp)
    agree["patterns_on_the_hand_left_out"] = int(hands)
    return np.concatenate(Gr), np.concatenate(Or), np.concatenate(Mr), np.concatenate(Pr), {k: int(v) for k, v in agree.items()}


def fit(X, G, O, m, pat, log, with_bounds=True):
    mem = X["mem"]; used = np.unique(pat[m]); idx_of = {int(p): j for j, p in enumerate(used)}; n_ind = len(used)
    ind = np.vectorize(lambda p: idx_of.get(int(p), -1))(pat); m = m & (ind >= 0); t0 = time.time()
    app = [P.appearance(mem, p) for p in range(len(mem))]
    sil = torch.stack([app[p]["sil"] for p in used]); col = torch.stack([app[p]["colour"] for p in used])
    shape = M.Proto(R.TH_SHAPE).learn(sil); colour = M.Proto(R.TH_COLOUR).learn(col)
    s_code = shape.energy(sil).argmin(1).numpy(); c_code = colour.energy(col).argmin(1).numpy(); klog = []
    dyn, kind, E = Cl.learn_classes(S.make, G, O, m, ind, n_ind, (s_code, c_code), (len(shape.P), len(colour.P)), log=klog)
    K = dyn.K; looks, w = Kd.fit_looks(s_code, c_code, E, [len(shape.P), len(colour.P), K]) if K > 1 else (None, np.ones(n_ind))
    true = np.array([true_kind(X, int(p)) for p in used]); tk = np.where(true >= 3, 1, true)            # the ground counted as fixed
    table = np.array([[float(w[(kind == k) & (true == t)].sum()) for t in range(5)] for k in range(K)])
    t3 = np.array([[float(w[(kind == k) & (tk == t)].sum()) for t in range(3)] for k in range(K)])
    kd = np.where(m, tk[ind.clip(0)], 0)
    log.update(objects_with_transitions=n_ind, objects_by_true_behaviour_then_ground_hand=np.bincount(true, minlength=5).tolist(), transitions=int(m.sum()), in_contact=int((m & (O[..., 0] == 1)).sum()),
               came_along=int((m & (O[..., 4] == 1)).sum()), shape_codes=len(shape.P), colour_codes=len(colour.P), classes=K, class_sizes=np.bincount(kind, minlength=K).tolist(), class_log=klog,
               classes_by_true_behaviour_then_ground_hand=table.round(1).tolist(), class_purity=pc([t3.max(1).sum() / max(t3.sum(), 1e-9)]),
               appearance_share=looks.share() if looks else None, fit_seconds=round(time.time() - t0))
    bounds = dict(true=M.Dyn().fit(G, O, m, kd, 3), one=M.Dyn().fit(G, O, m, 0 * kd, 1)) if with_bounds else None
    return dict(me=P.NoOne(), of_uid={}, shape=shape, colour=colour, dyn=dyn, kind=kind, mis=E, eps=1 / (1 + np.e), looks=looks, idx_of=idx_of, app=app, bounds=bounds, table=t3)


def first_sights(world, X, n, seed, source):
    """what the binding layer makes of each test scene's first frame (the same for both arms)"""
    sizes = W.SIZES + W.NEW_SIZES if source == "fresh" else W.SIZES
    te = Fb.episodes(world, n, seed, source, sizes); views = {}; found = []; ground = []
    for e, ep in enumerate(te):
        lab, rec, _, _ = Ap.first_sight(ep["frames"][0], X["mem"], X["net"], 5000 + e); mask = ep["masks"][0]; th = Fb.things_of(ep["objs"], "x")
        rows, sp = Fb.score_frame(lab, rec, X["mem"], mask, th); ground.append(sp)
        found += [(r["found"], ep["objs"][i - 3]["kind"], ep["objs"][i - 3]["size"]) for r, (i, _, _) in zip(rows, th) if i >= 3]
        bgl = np.bincount(lab.ravel()).argmax(); own = P.owner(rec, mask, len(ep["objs"]))
        for i in range(len(ep["objs"])):
            mi = mask == 3 + i; k = np.bincount(lab[mi]).argmax()
            if own[i] >= 0:
                views[e, i] = dict(p=int(own[i]))
            elif k != bgl and (lab[mi] == k).sum() >= 0.25 * mi.sum():
                views[e, i] = dict(p=-1, v=P6.group_appearance(ep["frames"][0], lab == k))
    mv = [q for q in found if q[1] != 1]
    stats = dict(objects=int(sum(len(ep["objs"]) for ep in te)), bound=len(views), labels_on_the_ground_per_scene=round(float(np.mean(ground)), 3),
                 found_movable=pc([q[0] for q in mv]), found_fixed=pc([q[0] for q in found if q[1] == 1]))
    return te, views, stats


def test_come(te, views, L):
    """the action that brings the object along: the hand at the object's side, the wanted effect is the object and the
    hand displaced by 3 pixels toward the middle of the frame; right if the object has moved in the simulator
    ("cannot" for a fixed object)"""
    K = L["dyn"].K; cases = []
    for e, ep in enumerate(te):
        st = dict(hand=ep["hand"][0], pos=ep["pos"][0], order=ep["order"], held=-1)
        for i, o in enumerate(ep["objs"]):
            spot = W.side_spot(st["hand"], st["pos"][i], o["size"])
            if any(W.touching(spot, st["pos"][j], q["size"]) for j, q in enumerate(ep["objs"]) if j != i):
                continue
            cases.append((e, i, dict(st, hand=spot)))
    N = len(cases); G = np.zeros((N, 5), int); O = np.zeros((N, 4, 5), int); m = np.zeros((N, 4), bool); O[..., 2:4] = 4
    w_app = np.full((N, K), 1.0 / K, np.float32); w_all = w_app.copy(); bound = np.zeros(N, bool)
    for n, (e, i, st) in enumerate(cases):
        d = np.sign(np.array([H // 2 - 3, H // 2 - 3]) - st["hand"]) * 3; d[d == 0] = 3
        G[n, :2] = 4 + d; m[n, 0] = True; O[n, 0] = (1, 0, 4 + d[0], 4 + d[1], 1)        # the object asked about, alone in its row
        v = views.get((e, i))
        if v is not None:
            bound[n] = True; vs = [L["app"][v["p"]]] if v["p"] >= 0 else [v["v"]]
            w_app[n] = R.kind_belief(L, vs)[0]; w_all[n] = R.kind_belief(L, vs, [L["idx_of"].get(v["p"], -1)])[0] if v["p"] >= 0 else w_app[n]

    def answers(model):
        out = []
        for k in range(model.K):
            a = M.propose(model, G, O, m, k); Ga = G.copy(); Ga[:, 2:] = a
            one = np.zeros((N, 4, model.K), np.float32); one[..., k] = 1
            came = model.forward(Ga, O, m, one)[..., 2]
            out.append([tuple(int(v) for v in a[n] - np.array([4, 4, 0])) if came[n, 0] == 1 else None for n in range(N)])
        return out

    def score(model, w, use_bound):
        ans = answers(model); ok = []
        for n, (e, i, st) in enumerate(cases):
            tally = {}
            for k in range(model.K):
                tally[ans[k][n]] = tally.get(ans[k][n], 0) + float(w[n][k])
            a = max(tally, key=tally.get) if (not use_bound or bound[n]) else None
            kind = te[e]["objs"][i]["kind"]
            ok.append(a is None if kind == 1 else (a is not None and (W.step(st, a, te[e]["objs"])["pos"][i] != st["pos"][i]).any()))
        return np.array(ok)
    kinds = np.array([te[e]["objs"][i]["kind"] for e, i, _ in cases]); exc = np.array([te[e]["objs"][i]["kind"] != te[e]["objs"][i]["rule"] for e, i, _ in cases])
    arms = {"appearance and own transitions": score(L["dyn"], w_all, True), "appearance only": score(L["dyn"], w_app, True),
            "true behaviour given": score(L["bounds"]["true"], np.eye(3)[kinds], False), "one class for all": score(L["bounds"]["one"], np.ones((N, 1)), False)}
    out = {}
    for name, c in (("all", kinds >= 0), ("takeable", kinds == 0), ("fixed", kinds == 1), ("sticky", kinds == 2), ("exceptions", exc), ("bound", bound), ("not bound", ~bound)):
        if c.sum():
            out[name] = dict(n=int(c.sum()), **{a: pc(v[c]) for a, v in arms.items()})
    return out


def run_world(seed, by, arms=("sim", "own")):
    t0 = time.time(); X = torch.load("runs/157/exp_%s_%d.pt" % (by, seed), weights_only=False); world = W.make_world(seed, by)
    log = dict(world=seed, by=by, kind_of=world["kind_of"].tolist(), train_episodes=X["n_train"], patterns=len(X["mem"]), experience=X["tally"], experience_seconds=X["seconds"])
    sights = {part: first_sights(world, X, 100, seed * 100 + s, src) for part, s, src in (("new_objects", 30, "fresh"), ("known_objects", 40, "pool"))}
    log["first_sight"] = {part: v[2] for part, v in sights.items()}
    for arm in arms:
        G, O, m, pat, agree = (own_tokens if arm == "own" else sim_tokens)(X); a = dict(agreement_with_the_simulator=agree)
        L = fit(X, G, O, m, pat, a)
        for part, (te, views, _) in sights.items():
            a[part] = test_come(te, views, L)
        log[arm] = a
    log["seconds"] = round(time.time() - t0)
    return log


if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]; res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by)); r = res[-1]; A = "appearance and own transitions"
        print(by, s, "first sight", r["first_sight"], "seconds", r["seconds"])
        for arm in ("sim", "own"):
            a = r[arm]
            print("  %s: objects %d %s classes %d %s purity %s share %s | contact %d came %d" % (arm, a["objects_with_transitions"], a["objects_by_true_behaviour_then_ground_hand"], a["classes"], a["classes_by_true_behaviour_then_ground_hand"],
                                                                                         a["class_purity"], a["appearance_share"], a["in_contact"], a["came_along"]))
            if a["agreement_with_the_simulator"]: print("     agreement", a["agreement_with_the_simulator"])
            for part in ("new_objects", "known_objects"):
                print("     %s:" % part, {k: (v["n"], v[A], v["appearance only"], v["true behaviour given"], v["one class for all"]) for k, v in a[part].items()})
        sys.stdout.flush()
        json.dump(res, open("runs/157/%s_%s.json" % (name, by), "w"), indent=1)
