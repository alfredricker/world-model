"""Card 159, stage 1: the script's experience as tools/card158/experience.py, with an object's place read from those of
its points that do not lie on the hand's own box.

usage: experience.py <by: shape|colour> <world seed> [episodes=100]     writes runs/159/exp_<by>_<seed>.pt
"""
import sys, time, numpy as np, torch
sys.path.insert(0, "tools/card157")
import experience as E7
Ag, Le, Ap, Fb, W, P = E7.Ag, E7.Le, E7.Ap, E7.Fb, E7.W, E7.P
M, dev, T, H, HAND = Le.M, Le.dev, W.T, W.H, W.HAND


DISP = np.array(M.DISP)


def seen_of(S, ids, t0, hand, ta, box_a):
    """per step: how many of the group's points are seen (n_all) and how many off the hand's box (n); the box around the
    latter; the commonest displacement index the latter were seen to make (-1: none read); and the object's frame: the
    box (ta, box_a) at which the pattern was recognised or stored, carried by those displacements (path integration)"""
    out = dict(t0=t0, n_all=np.zeros(T, int), n=np.zeros(T, int), box=np.zeros((T, 4), int), code=np.full(T, -1), n_read=np.zeros(T, int))
    if box_a is None:
        pix = np.isin(S.toks[ta], ids); ys, xs = np.nonzero(pix); box_a = (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1) if len(ys) else (0, 0, 0, 0)
    for t in range(t0, T):
        pix = np.isin(S.toks[t], ids); out["n_all"][t] = pix.sum(); hy, hx = hand[t]
        pix[hy:hy + HAND, hx:hx + HAND] = False; ys, xs = np.nonzero(pix); out["n"][t] = len(ys)
        if len(ys) == 0:
            continue
        out["box"][t] = (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)
        o = S.obs[t][np.unique(S.toks[t][pix])]; o = o[o >= 0]; out["n_read"][t] = len(o)
        if len(o):
            out["code"][t] = np.bincount(o).argmax()
    d = np.where(out["code"][:, None] >= 0, DISP[out["code"].clip(0)], 0); cum = np.concatenate([np.zeros((1, 2), int), np.cumsum(d, 0)])[:T]      # displacement before step t
    out["frame"] = np.asarray(box_a)[None] + np.tile(cum - cum[min(ta, T - 1)], 2)
    return out


def box_of(mem, r):
    ys, xs = torch.nonzero(mem.M[int(r["m"])], as_tuple=True)
    return (r["y"] + int(ys.min()), r["x"] + int(xs.min()), r["y"] + int(ys.max()) + 1, r["x"] + int(xs.max()) + 1)


def run(by, seed, n_train):
    world = W.make_world(seed, by); mem = Fb.Memory(); net = Ap.Pairs(); eps = Fb.episodes(world, n_train, seed * 100 + 10)
    tr, seen, own0, new, tally = [], [], [], [], {}; t0 = time.time()
    for e, ep in enumerate(eps):
        P0 = len(mem); frames = ep["frames"]; img = frames.astype(np.float32) / 255
        f = torch.tensor(frames).to(dev).permute(0, 3, 1, 2).float() / 255
        idx, valid = M.flow(*M.cost_volume(f[:-1], f[1:])); idx, valid = idx.cpu().numpy(), valid.cpu().numpy()
        S = Le.Seen(frames[0], mem, net, e)
        for t in range(T):
            S.step(img[t], img[t + 1], idx[t], valid[t])
        lg = Le.finish(ep, e, mem, net, S, "own")
        present = [(int(r["m"]), r["ids"], 0, 0, box_of(mem, r)) for r in S.rec] + [(w["p"], w["ids"], w["first"] - 1, w["ts"], None) for w in S.who]
        seen.append([dict(p=p, **seen_of(S, ids, a, ep["hand"], ta, ba)) for p, ids, a, ta, ba in present])
        own0.append(P.owner(S.rec, ep["masks"][0], len(ep["objs"]))); new.append((P0, len(mem)))
        tr.append({k: ep[k] for k in ("objs", "order", "hand", "pos", "held", "acts")})
        for k, v in lg.items():
            tally[k] = tally.get(k, 0) + v
        if e % 10 == 9:
            net.fit(300)
    torch.save(dict(by=by, seed=seed, n_train=n_train, mem=mem, net=net, tr=tr, seen=seen, own0=own0, new=new, tally=tally, seconds=round(time.time() - t0)),
               "runs/159/exp_%s_%d.pt" % (by, seed))
    print(by, seed, "patterns", len(mem), "tally", tally, "seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 100)
