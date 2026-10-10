"""Card 095.2: arm B's nearest stored groups by radius search, exact within the radius.

Arm B's distance between a query and a stored group is a sum over admitted conditions, each a distance between
discrete values (a token's part, or a place). Per field column, the admitted conditions on it give a table of
distances between its distinct values. A query lists, per column, the stored values within radius R of its own
value, walks their combinations (depth first, stopping a branch once its distance passes R) and looks each
combination up in a hash of the groups. Groups beyond R carry kernel weight below exp(-R) each.
"""
import numpy as np


class Radius:
    def __init__(self, arm, R):
        self.arm, self.R = arm, R
        cols = sorted({arm.col(c) for c in arm.adm})
        strength = {col: sum(l / arm.scale[arm.cand.index(c)] for c, l in zip(arm.adm, arm.lam) if arm.col(c) == col)
                    for col in cols}
        cols = sorted(cols, key=lambda col: -strength[col])   # strongest first: branches end soonest
        self.cols = cols
        G = arm.G[:, cols]
        self.key = {tuple(r): i for i, r in enumerate(G.tolist())}
        self.vals = [np.unique(G[:, j]) for j in range(len(cols))]
        self.conds = [[(c, l / arm.scale[arm.cand.index(c)]) for c, l in zip(arm.adm, arm.lam) if arm.col(c) == col]
                      for col in cols]
        self.memo = [{} for _ in cols]

    def _dist(self, j, v):
        """Distances from value v to every stored value of column j, as (values, distances) within R, sorted."""
        m = self.memo[j].get(v)
        if m is None:
            vals = self.vals[j]
            d = np.zeros(len(vals))
            V = self.arm.V
            for c, w in self.conds[j]:
                if c[0] == "place":
                    d += w * (vals != v)
                else:
                    k = c[-1]
                    sl = slice(k * 8, (k + 1) * 8)
                    d += w * np.abs(V[vals][:, sl] - V[v][sl]).sum(1)
            keep = d < self.R
            o = np.argsort(d[keep])
            m = self.memo[j][v] = (vals[keep][o].tolist(), d[keep][o].tolist())
        return m

    def query(self, f):
        """Groups within R of query fields f: (group indices, distances)."""
        lists = [self._dist(j, int(f[c])) for j, c in enumerate(self.cols)]
        out_i, out_d = [], []
        key = [0] * len(lists)
        R, look = self.R, self.key

        def walk(j, acc):
            if j == len(lists):
                g = look.get(tuple(key))
                if g is not None:
                    out_i.append(g)
                    out_d.append(acc)
                return
            vs, ds = lists[j]
            for v, d in zip(vs, ds):
                if acc + d >= R:
                    break
                key[j] = v
                walk(j + 1, acc + d)

        walk(0, 0.0)
        return np.array(out_i, np.int64), np.array(out_d)
