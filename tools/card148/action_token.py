"""Card 148: the action as an output token, completed from records of what actions did.

One step of experience is a small set of tokens:
  the hand's effect      e_y, e_x          how far the hand moved (9 values each)
  the action             a_y, a_x, a_g     the commanded move (9 values each) and the grip (2)
  per object             c, h, o_y, o_x, h2  in contact before, held before, how far it moved, held after
A query clamps everything but the action (the state now and the wanted effect) and asks for the action.

Arm P (couplings, as CRAM): an energy of pairwise couplings between token values, the same for every
  object, with one action shared by all objects of the scene; learned by pseudolikelihood (each token
  predicted from the others), no backpropagation through settling.  The action is read by settling.
Arm R (records, a modern Hopfield memory): every (step, object) is stored whole in one shot; a query
  retrieves by the softmax of the number of clamped tokens that match, and the retrieved records vote.
The chosen action is run in the simulator from the same state; it succeeds if the hand, every object
and what is held end exactly as wanted.
"""
import sys, json, itertools, numpy as np, torch
sys.path.insert(0, "tools/card148")
import env

GV, OV = [9, 9, 9, 9, 2], [2, 2, 9, 9, 2]
GO, OO = np.cumsum([0] + GV), np.cumsum([0] + OV)
ACT = slice(GO[2], GO[5])                      # the action's part of the hand-and-action vector


def tokens(data):
    N = len(data)
    G = np.zeros((N, 5), int); O = np.zeros((N, 4, 5), int); m = np.zeros((N, 4), bool); kind = []
    for k, (s, a, n) in enumerate(data):
        e = n["hand"] - s["hand"]
        G[k] = (e[0] + 4, e[1] + 4, a[0] + 4, a[1] + 4, a[2])
        for i in range(len(s["size"])):
            o = n["pos"][i] - s["pos"][i]
            O[k, i] = (env.touching(s["hand"], s["pos"][i], s["size"][i]), s["held"] == i, o[0] + 4, o[1] + 4, n["held"] == i)
            m[k, i] = True
        kind.append("take" if s["held"] < 0 <= n["held"] else "let go" if n["held"] < 0 <= s["held"] else "carry" if s["held"] >= 0
                    else "touching, not taken" if O[k, :, 0].any() else "move")
    return G, O, m, np.array(kind)


def onehot(x, V):
    return torch.cat([torch.nn.functional.one_hot(torch.as_tensor(x[..., i]), v) for i, v in enumerate(V)], -1).float()


def block_mask(V):
    off = np.cumsum([0] + V); M = torch.ones(off[-1], off[-1])
    for i in range(len(V)):
        M[off[i]:off[i + 1], off[i]:off[i + 1]] = 0
    return M


class Pairwise:
    def fit(self, G, O, m, l1=1e-3, steps=500, seed=0):
        torch.manual_seed(seed)
        xg, xo = onehot(G, GV), onehot(O, OV) * torch.as_tensor(m)[..., None]
        P = [torch.zeros(s, requires_grad=True) for s in ((GO[-1],), (OO[-1],), (GO[-1], GO[-1]), (OO[-1], OO[-1]), (GO[-1], OO[-1]))]
        opt = torch.optim.Adam(P, lr=0.05)
        Mg, Mo = block_mask(GV), block_mask(OV)
        Gt, Ot, mt = torch.as_tensor(G), torch.as_tensor(O), torch.as_tensor(m)
        ce = torch.nn.functional.cross_entropy
        for _ in range(steps):
            bg, bo, Wgg, Woo, Wgo = P
            Wg, Wo = (Wgg + Wgg.T) / 2 * Mg, (Woo + Woo.T) / 2 * Mo
            fg = bg + xg @ Wg + (xo @ Wgo.T).sum(1)
            fo = bo + xo @ Wo + (xg @ Wgo)[:, None]
            loss = sum(ce(fg[:, GO[i]:GO[i + 1]], Gt[:, i]) for i in range(5))
            loss = loss + sum((ce(fo[..., OO[i]:OO[i + 1]].reshape(-1, OV[i]), Ot[..., i].reshape(-1), reduction="none") * mt.reshape(-1)).sum() / len(G) for i in range(5))
            loss = loss + l1 * (Wg.abs().sum() + Wo.abs().sum() + Wgo.abs().sum())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            self.bg, self.Wg, self.Wgo = P[0].clone(), ((P[2] + P[2].T) / 2 * Mg).clone(), P[4].clone()
            self.bo, self.Wo = P[1].clone(), ((P[3] + P[3].T) / 2 * Mo).clone()        # kept for card 149
        return self

    def field(self, G, O, m):
        """what the clamped tokens say about each action value"""
        xg, xo = onehot(G, GV), onehot(O, OV) * torch.as_tensor(m)[..., None]
        xg[:, ACT] = 0
        return (self.bg + xg @ self.Wg + (xo @ self.Wgo.T).sum(1))[:, ACT], self.Wg[ACT, ACT]

    def predict(self, G, O, m, iters=30):
        """settle the three action tokens (mean field, temperature falling from 1 to 1/8)"""
        f, Waa = self.field(G, O, m)
        p = torch.cat([torch.full((len(G), v), 1.0 / v) for v in GV[2:]], 1)
        cut = np.cumsum([0] + GV[2:])
        for t in range(iters):
            beta = 8 ** (t / (iters - 1))
            for i in range(3):
                s = slice(cut[i], cut[i + 1])
                p[:, s] = torch.softmax(beta * (f + p @ Waa)[:, s], -1)
        return np.stack([p[:, cut[i]:cut[i + 1]].argmax(1).numpy() for i in range(3)], 1) - np.array([4, 4, 0])

    def predict_exhaustive(self, G, O, m):
        """the lowest-energy action by trying all 162"""
        f, Waa = self.field(G, O, m)
        combos = np.array(list(itertools.product(range(9), range(9), range(2))))
        X = onehot(combos, GV[2:])
        score = f @ X.T + 0.5 * ((X @ Waa) * X).sum(1)
        return combos[score.argmax(1).numpy()] - np.array([4, 4, 0])


class Records:
    def fit(self, G, O, m, **kw):
        k, o = np.nonzero(m)
        self.keys = torch.as_tensor(np.concatenate([G[k, :2], O[k, o]], 1)).cuda()
        self.acts = torch.as_tensor(G[k, 2:]).cuda()
        return self

    def predict(self, G, O, m, beta=4.0):
        k, o = np.nonzero(m)
        q = torch.as_tensor(np.concatenate([G[k, :2], O[k, o]], 1)).cuda()
        logp = [torch.zeros(len(G), v).cuda() for v in GV[2:]]
        kt = torch.as_tensor(k).cuda()
        for s in range(0, len(q), 4000):
            sim = (q[s:s + 4000, None] == self.keys[None]).sum(-1).float()
            w = torch.softmax(beta * sim, 1)
            for i, v in enumerate(GV[2:]):
                d = w @ torch.nn.functional.one_hot(self.acts[:, i], v).float()
                logp[i].index_add_(0, kt[s:s + 4000], torch.log(d + 1e-6))
        return np.stack([l.argmax(1).cpu().numpy() for l in logp], 1) - np.array([4, 4, 0])


def success(data, acts):
    ok = np.zeros(len(data), bool)
    for k, ((s, a, n), b) in enumerate(zip(data, acts)):
        r = env.step(s, tuple(int(v) for v in b))
        ok[k] = np.all(r["hand"] == n["hand"]) and np.all(r["pos"] == n["pos"]) and r["held"] == n["held"]
    return ok


def by_kind(ok, kind):
    out = {"all": round(100 * float(ok.mean()), 1)}
    for k in ("move", "touching, not taken", "take", "carry", "let go"):
        if (kind == k).any():
            out[k] = round(100 * float(ok[kind == k].mean()), 1)
    return out


NS = (10, 30, 100, 300, 1000, 3000)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "gate":
        data = env.collect(100, 201); G, O, m, kind = tokens(data)
        print("held-out validation steps:", len(data), {k: int((kind == k).sum()) for k in np.unique(kind)})
        rule = np.stack([G[:, 0] - 4, G[:, 1] - 4, (O[..., 4] * m).any(1)], 1)
        tr = tokens(env.collect(50, 200))[0][:, 2:]
        vals, cnt = np.unique(tr, axis=0, return_counts=True)
        common = np.tile(vals[cnt.argmax()] - np.array([4, 4, 0]), (len(data), 1))
        rng = np.random.default_rng(0)
        rand = np.stack([rng.integers(-4, 5, len(data)), rng.integers(-4, 5, len(data)), rng.integers(0, 2, len(data))], 1)
        out = {"steps": len(data), "kinds": {k: int((kind == k).sum()) for k in np.unique(kind)},
               "the recorded action run again": by_kind(success(data, G[:, 2:] - np.array([4, 4, 0])), kind),
               "a hand-written rule (move = the hand's wanted move; grip = some object wanted held)": by_kind(success(data, rule), kind),
               "the commonest action in the records": by_kind(success(data, common), kind), "a random action": by_kind(success(data, rand), kind)}
        json.dump(out, open("runs/148/gate.json", "w"), indent=1); print(json.dumps(out, indent=1))
    else:
        ev_seed, tr_seeds = (201, [200]) if mode == "dev" else (300, [100, 101, 102, 103, 104])
        data = env.collect(100, ev_seed); G, O, m, kind = tokens(data)
        out = {"steps": len(data), "kinds": {k: int((kind == k).sum()) for k in np.unique(kind)}, "runs": []}
        settings = [("P", dict(l1=l)) for l in (0.0, 1e-3, 1e-2)] + [("R", dict(beta=b)) for b in (1.0, 2.0, 4.0, 8.0, 16.0)] if mode == "dev" else \
                   [("P", dict(l1=float(sys.argv[2]))), ("R", dict(beta=float(sys.argv[3])))]
        for seed in tr_seeds:
            Gt, Ot, mt, _ = tokens(env.collect(50, seed))
            for n in NS:
                for arm, kw in settings:
                    if arm == "P":
                        model = Pairwise().fit(Gt[:n], Ot[:n], mt[:n], **kw)
                        acts = model.predict(G, O, m); ex = model.predict_exhaustive(G, O, m)
                        r = dict(by_kind(success(data, acts), kind), settled_is_lowest=round(100 * float((acts == ex).all(1).mean()), 1),
                                 exhaustive=by_kind(success(data, ex), kind)["all"])
                    else:
                        r = by_kind(success(data, Records().fit(Gt[:n], Ot[:n], mt[:n]).predict(G, O, m, **kw)), kind)
                    out["runs"].append(dict(arm=arm, n=n, seed=seed, **kw, **r))
                    print(json.dumps(out["runs"][-1]), flush=True)
        json.dump(out, open("runs/148/%s.json" % mode, "w"), indent=1)
