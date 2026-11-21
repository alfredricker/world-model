"""Card 154: learn stored patterns from motion binding in card 152's sprite world, then test feedback on scenes where
nothing has moved.

usage: run.py <name> <by: shape|colour> <world seeds...>      writes runs/154/<name>_<by>.json
       "_nNN" in the name sets the number of training episodes (100)
Arms at test: feedback with the learned memory; no feedback (motion only: at first sight nothing is bound);
feedback with a memory stored from the true masks (the upper bound).
"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "tools/card154")
import feedback as Fb
W, B = Fb.W, Fb.B
pc = lambda v: round(100 * float(np.mean(v)), 1) if len(v) else None


def moved_steps(ep, i):
    pos = ep["hand"] if i == 1 else ep["pos"][:, i - 3]
    return np.abs(np.diff(pos.astype(int), axis=0)).sum(1) > 0


def summary(rows, known):
    """rows of score_frame with the scene's spurious count; known: the keys of things that moved in training"""
    kn = [r for r in rows if r["key"] in known]; un = [r for r in rows if r["key"] not in known]
    cut = {"all": lambda h: True, "wholly seen": lambda h: h == 0, "up to a quarter hidden": lambda h: 0 < h < 0.25,
           "a quarter to a half hidden": lambda h: 0.25 <= h < 0.5, "half or more hidden": lambda h: h >= 0.5}
    out = {}
    for name, c in cut.items():
        s = [r for r in kn if c(r["hidden"])]; ob = [r for r in s if r["key"] != "hand"]
        out[name] = dict(n=len(s), found=pc([r["found"] for r in s]), itself=pc([r["itself"] for r in s]), n_objects=len(ob), objects_found=pc([r["found"] for r in ob]),
                         objects_itself=pc([r["itself"] for r in ob]), mean_iou=round(float(np.mean([r["iou"] for r in s])), 3) if s else None)
    out["not known"] = dict(n=len(un), found=pc([r["found"] for r in un]), wrongly_recognised=pc([r["claimed"] for r in un]))
    return out


def test_stills(frames, masks, things, mem, feedback, known, seed):
    rows = []; sp = []
    for j, (fr, mk, th) in enumerate(zip(frames, masks, things)):
        lab, rec, _, _ = Fb.first_sight(fr, mem, feedback, seed + j)
        r, s = Fb.score_frame(lab, rec, mem, mk, th); rows += r; sp.append(s)
    out = summary(rows, known); out["labels_on_the_ground_per_scene"] = round(float(np.mean(sp)), 3)
    return out


def test_moving(eps, mem, feedback, known, seed):
    """whole episodes: things that have moved (card 147's measure), and known things that have not moved yet"""
    mv, still = [], []
    for e, ep in enumerate(eps):
        L, rec, _ = Fb.run_episode(ep, "t%d" % e, mem, feedback=feedback, learn=False, seed=seed + e)
        th = Fb.things_of(ep["objs"], "t%d" % e)
        first = {i: (np.nonzero(moved_steps(ep, i))[0][0] + 1 if moved_steps(ep, i).any() else 10 ** 6) for i, _, _ in th}
        for t in range(1, Fb.T + 1):
            rows, _ = Fb.score_frame(L[t], [], mem, ep["masks"][t], th)
            for r, (i, key, full) in zip(rows, [x for x in th if (ep["masks"][t] == x[0]).sum() >= 8]):
                if first[i] <= t:
                    mv.append((r["iou"] >= 0.5, r["found"]))
                elif key in known:
                    still.append(r["found"])
    return dict(moved_rows=len(mv), moved_found_iou=pc([a for a, _ in mv]), moved_found_alone=pc([b for _, b in mv]), known_not_yet_moved_rows=len(still), known_not_yet_moved_found=pc(still))


def run_world(seed, by, n_train):
    t0 = time.time(); world = W.make_world(seed, by); log = dict(world=seed, by=by, train_episodes=n_train)
    tr = Fb.episodes(world, n_train, seed * 100 + 10)
    mem = Fb.Memory(); bound = Fb.Memory(); tot = dict(stored=0, known=0, too_big=0); moved = {}; never = set()
    a, b = B.ArmC(N=4, K=2, dev="cpu").a, B.ArmC(N=4, K=2, dev="cpu").b
    for e, ep in enumerate(tr):
        _, _, lg = Fb.run_episode(ep, e, mem, seed=e)
        for k in tot:
            tot[k] += lg[k]
        for i in [1] + [3 + j for j in range(len(ep["objs"]))]:
            key = Fb.key_of(ep, e, i); st = int(moved_steps(ep, i).sum())
            if st >= Fb.N_MIN:
                moved[key] = moved.get(key, 0) + 1
                if not any(q["key"] == key for q in bound.info):
                    m = ep["masks"][0] == i; ys, xs = np.nonzero(m); y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
                    bound.store(m[y0:y1, x0:x1], ep["frames"][0][y0:y1, x0:x1].astype(np.float32) / 255, 10 * a, -10 * b, dict(key=key, iou=1.0, n=int(m.sum()), episode=e))
            elif key.startswith("pool"):
                never.add(key)
    never -= set(moved); known = set(moved)
    keys = [q["key"] for q in mem.info]; iou = np.array([q["iou"] for q in mem.info])
    pool_known = [k for k in known if k.startswith("pool")]
    log["memory"] = dict(patterns=len(mem), **tot, pattern_iou_quartiles=np.percentile(iou, [5, 25, 50, 75]).round(3).tolist(), things_that_moved=len(known), pool_objects_that_moved=len(pool_known),
                         pool_objects_never_moved=len(never), moved_with_a_pattern=pc([k in keys for k in known]),
                         moved_with_a_good_pattern=pc([any(q["key"] == k and q["iou"] >= 0.8 for q in mem.info) for k in known]),
                         pool_moved_with_a_good_pattern=pc([any(q["key"] == k and q["iou"] >= 0.8 for q in mem.info) for k in pool_known]),
                         patterns_of_the_hand=keys.count("hand"), stored_couplings=dict(inside_median=round(float(np.median(mem.Jin)), 1), outside_median=round(float(np.median(mem.Jout)), 1)),
                         train_seconds=round(time.time() - t0))
    arms = {"feedback": (mem, True), "no feedback": (mem, False), "feedback, memory from true masks": (bound, True)}
    te = Fb.episodes(world, 100, seed * 100 + 30, "pool")
    fr = [ep["frames"][0] for ep in te]; mk = [ep["masks"][0] for ep in te]; th = [Fb.things_of(ep["objs"], "a%d" % e) for e, ep in enumerate(te)]
    log["first_sight"] = {n: test_stills(fr, mk, th, m, f, known, 1000) for n, (m, f) in arms.items()}
    nv = []
    for j in range(len(te)):                                               # pool objects that never moved in training (the fixed ones)
        lab, rec, _, _ = Fb.first_sight(fr[j], mem, True, 1000 + j)
        nv += [r["found"] for r in Fb.score_frame(lab, rec, mem, mk[j], th[j])[0] if r["key"] in never]
    log["first_sight"]["pool objects that never moved in training"] = dict(n=len(nv), found=pc(nv))
    sc = W.scenes(world, 300, seed * 100 + 50)
    fr = [s["frame"] for s in sc]; mk = [s["mask"].astype(int) for s in sc]; th = [Fb.things_of(s["objs"], "b%d" % e) for e, s in enumerate(sc)]
    log["covered"] = {n: test_stills(fr, mk, th, m, f, known, 2000) for n, (m, f) in arms.items()}
    log["moving"] = {n: test_moving(te[:20], m, f, known, 3000) for n, (m, f) in list(arms.items())[:2]}
    log["seconds"] = round(time.time() - t0); log["peak_gpu_mb"] = int(torch.cuda.max_memory_allocated() // 2 ** 20)
    return log


if __name__ == "__main__":
    name, by = sys.argv[1], sys.argv[2]
    n_train = int(name.split("_n")[1].split("_")[0]) if "_n" in name else 100
    res = []
    for s in sys.argv[3:]:
        res.append(run_world(int(s), by, n_train)); r = res[-1]
        print(json.dumps({k: r[k] for k in ("world", "by", "train_episodes", "memory", "seconds", "peak_gpu_mb")}))
        for part in ("first_sight", "covered", "moving"):
            for arm, v in r[part].items():
                print("  %-12s %-34s %s" % (part, arm, json.dumps(v)))
        sys.stdout.flush()
        json.dump(res, open("runs/154/%s_%s.json" % (name, by), "w"), indent=1)
