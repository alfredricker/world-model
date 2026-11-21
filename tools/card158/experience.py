"""Card 158, stage 1 for the script's experience: as tools/card157/experience.py, with card 152's script acting
(binding, appearance couplings and memory as in card 156's script arm).

usage: experience.py <by: shape|colour> <world seed> [episodes=100]     writes runs/158/exp_script_<by>_<seed>.pt
"""
import sys, time, numpy as np, torch
sys.path.insert(0, "tools/card157")
import experience as E7
Ag, Le, Ap, Fb, W, P = E7.Ag, E7.Le, E7.Ap, E7.Fb, E7.W, E7.P
M, dev, T = Le.M, Le.dev, W.T


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
        present = [(int(r["m"]), r["ids"], 0) for r in S.rec] + [(w["p"], w["ids"], w["first"] - 1) for w in S.who]
        seen.append([dict(p=p, **E7.seen_of(S, ids, a)) for p, ids, a in present])
        own0.append(P.owner(S.rec, ep["masks"][0], len(ep["objs"]))); new.append((P0, len(mem)))
        tr.append({k: ep[k] for k in ("objs", "order", "hand", "pos", "held", "acts")})
        for k, v in lg.items():
            tally[k] = tally.get(k, 0) + v
        if e % 10 == 9:
            net.fit(300)
    torch.save(dict(by=by, seed=seed, n_train=n_train, mem=mem, net=net, tr=tr, seen=seen, own0=own0, new=new, tally=tally, seconds=round(time.time() - t0)),
               "runs/158/exp_script_%s_%d.pt" % (by, seed))
    print(by, seed, "patterns", len(mem), "tally", tally, "seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 100)
