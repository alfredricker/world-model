"""Card 157, stage 1: the agent's experience in one world (card 156's agent, binding and appearance couplings,
unchanged), saved for stage 2.  For every stored pattern present in an episode it records what the binding layer
saw of the pattern's points at each step: how many, the box around them, and the displacement most of them made.

usage: experience.py <by: shape|colour> <world seed> [episodes=100]     writes runs/157/exp_<by>_<seed>.pt
"""
import sys, time, numpy as np, torch
sys.path.insert(0, "tools/card156")
import agent as Ag
Le, Ap, Fb, W = Ag.Le, Ag.Ap, Ag.Fb, Ag.W
sys.path.insert(0, "tools/card155")
import pipeline as P
T, H = W.T, W.H


def seen_of(S, ids, t0):
    """what the binding layer saw of one group of points: per step the count, the box (y0, x0, y1, x1), the commonest
    displacement index of the motion modules (-1: none read) and how many points showed it"""
    out = dict(t0=t0, n=np.zeros(T, int), box=np.zeros((T, 4), int), code=np.full(T, -1), n_code=np.zeros(T, int), n_read=np.zeros(T, int))
    for t in range(t0, T):
        pix = np.isin(S.toks[t], ids); ys, xs = np.nonzero(pix); out["n"][t] = len(ys)
        if len(ys) == 0:
            continue
        out["box"][t] = (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)
        o = S.obs[t][np.unique(S.toks[t][pix])]; o = o[o >= 0]; out["n_read"][t] = len(o)
        if len(o):
            c = np.bincount(o); out["code"][t] = c.argmax(); out["n_code"][t] = c.max()
    return out


def run(by, seed, n_train):
    world = W.make_world(seed, by); mem = Fb.Memory(); net = Ap.Pairs(); rng = np.random.default_rng(seed * 100 + 10)
    tr, seen, own0, new, tally = [], [], [], [], {}; t0 = time.time()
    for e in range(n_train):
        P0 = len(mem)
        ep, S, lg = Ag.live_episode(world, e, rng, mem, net, seed=e)
        present = [(int(r["m"]), r["ids"], 0) for r in S.rec] + [(w["p"], w["ids"], w["first"] - 1) for w in S.who]
        seen.append([dict(p=p, **seen_of(S, ids, a)) for p, ids, a in present])
        own0.append(P.owner(S.rec, ep["masks"][0], len(ep["objs"]))); new.append((P0, len(mem)))      # for the simulator-token arm and for scoring
        tr.append({k: ep[k] for k in ("objs", "order", "hand", "pos", "held", "acts")})
        for k, v in lg.items():
            tally[k] = tally.get(k, 0) + v
        if e % 10 == 9:
            net.fit(300)
    torch.save(dict(by=by, seed=seed, n_train=n_train, mem=mem, net=net, tr=tr, seen=seen, own0=own0, new=new, tally=tally, seconds=round(time.time() - t0)),
               "runs/157/exp_%s_%d.pt" % (by, seed))
    print(by, seed, "patterns", len(mem), "tally", tally, "seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 100)
