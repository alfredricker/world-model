"""usage: run_arm.py <split> <episodes> <arm: C | M | step> <motion: pixels | true> [K] [first episode]
Scores one arm on one split and writes runs/147/<arm>_<motion>_<split>[_K<K>].json and .npy (the rows)."""
import sys, json, time, numpy as np
sys.path.insert(0, "tools/card147")
import binding as B


def summary(r):
    """r: rows as an array, with the episode in column 0"""
    f = r[:, 4] >= 0.5; ob = r[:, 2] >= 3; hd = r[:, 7] == 1
    pc = lambda s, v=None: None if s.sum() == 0 else round(100 * float((f if v is None else v)[s].mean()), 1)
    out = dict(rows=int(len(r)), found=pc(f | True), mean_iou=round(float(r[:, 4].mean()), 3), hand=pc(~ob), objects=pc(ob),
               held=pc(ob & hd), held_with_other=pc(ob & hd, r[:, 5] == 1), released=pc(ob & ~hd),
               released_in_background=pc(ob & ~hd, (r[:, 6] == 1) & ~f), first_frame=pc(ob & (r[:, 8] == 0)), third_frame=pc(ob & (r[:, 8] == 2)))
    for s in sorted(set(r[ob, 9].astype(int))):
        out["width_%d" % s] = pc(ob & (r[:, 9] == s))
    return out


if __name__ == "__main__":
    split, n, arm, source = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    K = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    e0 = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    D = "data/sprite147/%s_" % split
    fr = np.load(D + "frames.npy", mmap_mode="r"); mk = np.load(D + "masks.npy", mmap_mode="r")
    hand = np.load(D + "hand.npy"); op = np.load(D + "obj_pos.npy"); held = np.load(D + "held.npy"); meta = json.load(open(D + "meta.json"))
    rows = []; t0 = time.time(); alike = []; full = 0
    for e in range(e0, e0 + n):
        L, model = B.run_episode(arm, np.asarray(fr[e]), np.asarray(mk[e]), hand[e], op[e], source, K=K, seed=e)
        full += getattr(model, "full", 0)
        rows += [(e,) + x for x in B.score(L, np.asarray(mk[e]), hand[e], op[e], held[e], meta[e])]
    r = np.array(rows, float)
    out = summary(r)
    al = np.array([meta[int(e)]["alike"] and i in (3, 4) for e, i in zip(r[:, 0], r[:, 2])])
    out["look_alikes"] = None if al.sum() == 0 else round(100 * float((r[al, 4] >= 0.5).mean()), 1)
    out.update(arm=arm, motion=source, split=split, episodes=n, K=K, seconds=round(time.time() - t0), no_free_file=full)
    name = "runs/147/%s_%s_%s%s" % (arm, source, split, "" if K == 8 else "_K%d" % K)
    if e0 == 0:
        json.dump(out, open(name + ".json", "w"), indent=1); np.save(name + ".npy", r)
    print(json.dumps(out))
