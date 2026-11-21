"""check: how the agent's takes differ from the script's.  dev5.py <seed> <by> <n>"""
import sys, numpy as np
sys.path.insert(0, "tools/card156")
import agent as Ag
Le, Ap, Fb, W = Ag.Le, Ag.Ap, Ag.Fb, Ag.W
seed, by, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
world = W.make_world(seed, by)
def takes(eps):
    G, O, m = W.tokens(eps); t = m & (O[..., 0] == 1) & (O[..., 1] == 0) & (O[..., 4] == 1); z = (O[..., 2] == 4) & (O[..., 3] == 4)
    return dict(takes=int(t.sum()), without_a_move=int((t & z).sum()), with_a_move=int((t & ~z).sum()))
print("script", takes(Fb.episodes(world, n, seed * 100 + 10)))
mem = Fb.Memory(); net = Ap.Pairs(); rng = np.random.default_rng(seed * 100 + 10); eps = []
for e in range(n):
    ep, S, lg = Ag.live_episode(world, e, rng, mem, net, seed=e); eps.append(ep)
    if e % 10 == 9: net.fit(300)
print("agent ", takes(eps))
