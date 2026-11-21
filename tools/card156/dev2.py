"""dev: how the binding layer's own labels compare with the simulator's, over the final frames of n episodes. dev2.py <seed> <by> <n>"""
import sys, numpy as np, torch
sys.path.insert(0, "tools/card156")
import learn as Le
Ap, Fb, W = Le.Ap, Le.Fb, Le.W
seed, by, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
world = W.make_world(seed, by); tr = Fb.episodes(world, n, seed * 100 + 10); mem = Fb.Memory()
C = torch.zeros(6, 3, dtype=torch.float64, device="cuda")       # rows: true pair type; columns: own none / together / apart
names = ["ground-ground", "same moved thing", "same thing never moved in the episode", "moved thing - ground", "unmoved thing - ground", "two different things"]
orig = Ap.evidence
def spy(model, tok):
    ev = orig(model, tok); spy.last = ev; return ev
Ap.evidence = spy
class Sink:
    trained = False
    def add(self, *a): pass
for e, ep in enumerate(tr):
    Le.run_episode(ep, e, mem, Sink(), seed=e)
    m = torch.as_tensor(ep["masks"][-1].ravel(), device="cuda"); ev = spy.last
    mv = torch.zeros(16, dtype=torch.bool, device="cuda")
    for i in [1] + [3 + j for j in range(len(ep["objs"]))]:
        pos = ep["hand"] if i == 1 else ep["pos"][:, i - 3]
        mv[i] = bool(np.abs(np.diff(pos.astype(int), axis=0)).sum() > 0)
    same = m[:, None] == m[None]; g = m == 0; pm = mv[m]
    typ = torch.full(same.shape, -1, device="cuda")
    typ[g[:, None] & g[None]] = 0
    typ[same & ~g[:, None] & pm[:, None]] = 1; typ[same & ~g[:, None] & ~pm[:, None]] = 2
    gx = (g[:, None] ^ g[None]); anym = (pm[:, None] | pm[None])
    typ[gx & anym] = 3; typ[gx & ~anym] = 4; typ[~same & ~g[:, None] & ~g[None]] = 5
    typ.fill_diagonal_(-1)
    for a in range(6):
        for b in range(3):
            C[a, b] += ((typ == a) & (ev == b)).sum()
C = C.cpu().numpy()
print("%-40s %12s %8s %8s %8s" % ("true pair type", "pairs", "none", "together", "apart"))
for a in range(6): print("%-40s %12d %8.1f %8.1f %8.1f" % (names[a], C[a].sum(), *(100 * C[a] / max(C[a].sum(), 1))))
print("of own 'together' pairs, truly the same thing: %.1f%%; of own 'apart' pairs, truly different or thing-ground: %.1f%%, ground-ground: %.1f%%" % (100 * (C[1, 1] + C[2, 1]) / C[:, 1].sum(), 100 * (C[3, 2] + C[4, 2] + C[5, 2]) / C[:, 2].sum(), 100 * C[0, 2] / C[:, 2].sum()))
