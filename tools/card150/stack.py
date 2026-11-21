"""Card 150: three feed-forward layers of energy modules on the sprite world's pixel stream.

  layer 0  motion modules, one per pixel.  State: a belief over 121 displacements.  Energy: the matching
           cost of each displacement (card 147's front end).  Each settles alone, in one step.
  layer 1  binding modules, one per followed point.  State: a belief over at most K objects.  Energy:
           pairwise couplings learned from layer 0's settled states (card 147's arm C).  They settle
           together, `ticks` updates per frame, carried on from the last frame.
  layer 2  object modules, one per object.  State: the object's displacement.  It reads layer 1 (which
           points are its own) and layer 0, in one of three ways:
             state    every point's settled displacement; the object takes the commonest
             gated    the same, but only from points whose layer-0 energy is low (the scalar as confidence)
             belief   the sum of its points' layer-0 energies over all displacements; the object takes the lowest
Nothing flows downward.  Scored against the simulator: is the displacement of the hand and of each
object that has moved right, at every step?

usage: stack.py <split> <episodes> <ticks> <fraction updated per tick> [true]   ("true": layer 1 replaced by the true masks)
"""
import sys, json, numpy as np, torch
sys.path.insert(0, "tools/card147")
import motion as M, binding as B

H = M.H


class Binding(B.ArmC):
    """card 147's arm C with the number of updates per frame and the share of points updated per tick exposed"""
    def __init__(self, ticks=40, frac=1.0, **kw):
        super().__init__(**kw)
        self.ticks, self.frac = ticks, frac

    def update(self, obs, born, dead, tokens):
        J, v, dev = self.J, self.v, self.dev
        gone = torch.as_tensor(np.concatenate([dead, born]), device=dev)
        J[gone] = 0; J[:, gone] = 0; v[gone] = 0
        ids = np.nonzero(obs >= 0)[0]; o = obs[ids]; mv = o != B.ZERO
        if mv.any():
            ti = torch.as_tensor(ids, device=dev); to = torch.as_tensor(o, device=dev); tm = torch.as_tensor(mv, device=dev)
            same = (to[tm][:, None] == to[None]).float()
            block = self.a * same - self.b * (1 - same)
            J[ti[tm][:, None], ti[None]] += block
            J[ti[~tm][:, None], ti[tm][None]] += block[:, ~tm].T
        V = torch.as_tensor(tokens.tok.ravel(), device=dev)
        Jv = J[V][:, V] + self.j0
        Jv.fill_diagonal_(0)
        x = v[V]
        for t in range(self.ticks):
            beta = self.b1 if self.ticks == 1 else self.b0 * (self.b1 / self.b0) ** (t / (self.ticks - 1))
            p = torch.softmax(beta * x, -1)
            new = (1 - self.eta) * x + self.eta * (Jv @ p) + self.noise * (1 - t / self.ticks) * torch.randn(x.shape, device=dev, generator=self.g)
            if self.frac < 1:                                    # asynchronous: only a random share of the points update
                new = torch.where(torch.rand(len(x), 1, device=dev, generator=self.g) < self.frac, new, x)
            x = new
        v[V] = x
        return x.argmax(-1).reshape(H, H).cpu().numpy()


def object_layer(lab, C, idx, valid, K):
    """lab [H,W] labels; C [121,H,W] layer-0 energies; idx, valid: layer 0's settled state and whether its energy is low.
    Returns, per label, the displacement by the three readings (-1: no point to read)."""
    out = np.full((K, 3), -1)
    onek = torch.nn.functional.one_hot(lab, K).permute(2, 0, 1).float()              # [K,H,W]
    votes = torch.nn.functional.one_hot(idx, len(M.DISP)).float()                    # [H,W,121]
    hard = torch.einsum("khw,hwd->kd", onek, votes)
    gated = torch.einsum("khw,hwd->kd", onek * valid.float(), votes)
    soft = torch.einsum("khw,dhw->kd", onek, C.clamp(max=0.05))
    n = onek.sum((1, 2))
    out[:, 0] = hard.argmax(1).cpu().numpy(); out[:, 1] = gated.argmax(1).cpu().numpy(); out[:, 2] = soft.argmin(1).cpu().numpy()
    out[(n == 0).cpu().numpy()] = -1
    out[(gated.sum(1) == 0).cpu().numpy(), 1] = -1
    return out


def run(split, episodes, ticks, frac, true_masks=False, dev="cuda", K=8):
    D = "data/sprite147/%s_" % split
    fr = np.load(D + "frames.npy", mmap_mode="r"); mk = np.load(D + "masks.npy", mmap_mode="r")
    hand = np.load(D + "hand.npy"); op = np.load(D + "obj_pos.npy"); meta = json.load(open(D + "meta.json"))
    rows = []                                # episode, step, thing, moving, found, right by state, gated, belief, pixels
    for e in range(episodes):
        frames = np.asarray(fr[e]); masks = np.asarray(mk[e]).astype(int); img = frames.astype(np.float32) / 255
        f = torch.tensor(frames).to(dev).permute(0, 3, 1, 2).float() / 255
        C, P = M.cost_volume(f[:-1], f[1:]); idx, valid = M.flow(C, P)
        idn, vn = idx.cpu().numpy(), valid.cpu().numpy()
        tokens = B.Tokens(); model = Binding(ticks=ticks, frac=frac, K=K, dev=dev, seed=e)
        things = [(1, hand[e])] + [(3 + i, op[e][:, i]) for i in range(meta[e]["n"])]
        first = {}
        for i, pos in things:
            ch = np.nonzero(np.abs(np.diff(pos.astype(int), axis=0)).sum(1))[0]
            if len(ch):
                first[i] = ch[0]
        for t in range(len(frames) - 1):
            old = tokens.tok.copy()
            obs, born, dead = tokens.step(img[t], img[t + 1], idn[t], vn[t])
            model.update(obs, born, dead, tokens)
            if true_masks == "one":
                lab = torch.zeros(H, H, dtype=torch.long, device=dev); k_of = lambda i, m: 0
            elif true_masks:
                lab = torch.as_tensor(masks[t], device=dev); k_of = lambda i, m: i
            else:
                lab = model.v[torch.as_tensor(old.ravel(), device=dev)].argmax(-1).reshape(H, H)     # frame t's points, after this step
                labn = lab.cpu().numpy(); k_of = lambda i, m: np.bincount(labn[m]).argmax()
            ans = object_layer(lab, C[t], idx[t], valid[t], 8 if true_masks else K)
            for i, pos in things:
                m = masks[t] == i
                if i not in first or first[i] > t or not m.any():
                    continue
                d = pos[t + 1].astype(int) - pos[t].astype(int)
                truth = (d[0] + M.R) * (2 * M.R + 1) + (d[1] + M.R)
                k = k_of(i, m)
                found = True if true_masks else ((labn == k) & m).sum() / ((labn == k) | m).sum() >= 0.5
                rows.append((e, t, i, bool(d.any()), found, ans[k, 0] == truth, ans[k, 1] == truth, ans[k, 2] == truth, m.sum()))
    r = np.array(rows, float)
    mv = r[:, 3] == 1; ob = r[:, 2] >= 3; small = r[:, 8] < 40
    pc = lambda s, c: round(100 * float(r[s, c].mean()), 1)
    out = dict(split=split, episodes=episodes, ticks=ticks, share_updated=frac, layer1="one module" if true_masks == "one" else "true masks" if true_masks else "binding", rows=int(len(r)), moving_rows=int(mv.sum()),
               found=pc(mv, 4), moving={"state": pc(mv, 5), "gated": pc(mv, 6), "belief": pc(mv, 7)},
               moving_objects={"state": pc(mv & ob, 5), "gated": pc(mv & ob, 6), "belief": pc(mv & ob, 7)},
               moving_under_40_pixels={"state": pc(mv & small, 5), "gated": pc(mv & small, 6), "belief": pc(mv & small, 7)} if (mv & small).any() else None,
               still={"state": pc(~mv, 5), "gated": pc(~mv, 6), "belief": pc(~mv, 7)})
    return out


if __name__ == "__main__":
    split, episodes, ticks, frac = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    true_masks = (sys.argv[5] if sys.argv[5] == "one" else True) if len(sys.argv) > 5 else False
    out = run(split, episodes, ticks, frac, true_masks)
    json.dump(out, open("runs/150/%s_%d_%d_%g%s.json" % (split, episodes, ticks, frac, "_one" if true_masks == "one" else "_true" if true_masks else ""), "w"), indent=1)
    print(json.dumps(out))
