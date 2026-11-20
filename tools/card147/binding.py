"""Card 147: which tracked points make one object, learned from how they move as the agent acts.

Shared by the arms: `Tokens` follows every visible point from frame to frame (a token is a point with
an identity that persists), and remembers a static point that gets covered so it can come back.
Each step every token that was seen gives one observation: the displacement it made, possibly zero.

Arm C (couplings): one number per pair of tokens, the accumulated log-odds that the two are parts of
  one object: both moved the same way +a; one moved and the other did not, or moved otherwise, -b.
  The grouping is read by settling a belief over at most K objects per token on the energy of card
  146.1, carried from frame to frame.
Arm M (motion prediction): at most K object files.  A file moves the way most of its tokens moved;
  each token keeps, per file, the accumulated count of steps in which that file failed to predict
  its motion, and belongs to the file with the lowest count.  Tokens that no file predicts open a
  new file.
Baseline (per step): tokens that moved the same way in this step are one object; all else is one.

Neither arm reads the true masks, appearance (beyond re-finding a covered point) or the action.
"""
import numpy as np, torch
from scipy import ndimage
import motion as M

H = M.H
ZERO = M.ZERO
DISP = np.array(M.DISP)
YS, XS = np.mgrid[0:H, 0:H]
EIGHT = np.ones((3, 3), bool)
N_TOK = 6144


class Tokens:
    def __init__(self, N=N_TOK):
        self.N = N
        self.tok = np.arange(H * H).reshape(H, H)            # the token seen at each pixel
        self.alive = np.zeros(N, bool); self.alive[:H * H] = True
        self.under_tok = np.full((H, H), -1)                 # a static token hidden at this place, and its look
        self.under_img = np.zeros((H, H, 3), np.float32)

    def step(self, a, b, idx, valid):
        """a, b: frames t and t+1 [H, W, 3] in 0..1; idx, valid: the motion read at frame t.
        Returns obs[N] (the displacement index each token was seen to make, -1 for none), born, dead."""
        N, tok, ut = self.N, self.tok, self.under_tok
        moving = valid & (idx != ZERO); static = valid & (idx == ZERO)
        new = np.full((H, H), -1); obs = np.full(N, -1)
        new[static] = tok[static]; obs[tok[static]] = ZERO
        ty, tx = YS + DISP[idx][..., 0], XS + DISP[idx][..., 1]
        inb = moving & (ty >= 0) & (ty < H) & (tx >= 0) & (tx < H)
        inb &= ~static[ty.clip(0, H - 1), tx.clip(0, H - 1)]  # a place shows one thing: nothing moved to where a point
        moving = inb                                         # was seen to stay, so that reading is dropped
        my, mx = ty[inb], tx[inb]
        cover = ~moving[my, mx]                              # what a mover lands on, unless that moved away itself,
        cy, cx = my[cover], mx[cover]                        # is remembered as hidden there
        ut[cy, cx] = tok[cy, cx]; self.under_img[cy, cx] = a[cy, cx]
        new[my, mx] = tok[inb]; obs[tok[inb]] = idx[inb]
        placed = np.zeros(N, bool); placed[new[new >= 0]] = True; placed[tok[cy, cx]] = True
        un = new < 0                                         # a place nothing was carried to:
        back = un & (ut >= 0) & (((b - self.under_img) ** 2).mean(-1) < M.THR_P)
        new[back] = ut[back]; placed[ut[back]] = True; ut[back] = -1          # the hidden token has come back,
        stay = (new < 0) & ~moving & (((b - a) ** 2).mean(-1) < M.THR_P)
        new[stay] = tok[stay]; placed[tok[stay]] = True                       # or what was there is still there,
        for dv in np.unique(idx[inb]):                                         # or it is the rim of a mover whose own
            dy, dx = DISP[dv]                                                  # motion could not be read
            front = np.zeros((H, H), bool); sel = idx[inb] == dv; front[my[sel], mx[sel]] = True
            sy, sx = YS - dy, XS - dx
            ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < H)
            sy, sx = sy.clip(0, H - 1), sx.clip(0, H - 1)
            src = tok[sy, sx]
            ok &= ((b - a[sy, sx]) ** 2).mean(-1) < M.THR_P
            for _ in range(3):
                cand = (new < 0) & ndimage.binary_dilation(front, EIGHT) & ok & ~placed[src]
                if not cand.any():
                    break
                new[cand] = src[cand]; placed[src[cand]] = True; obs[src[cand]] = dv; front |= cand
        alive = placed.copy(); alive[ut[ut >= 0]] = True
        dead = np.nonzero(self.alive & ~alive)[0]
        free = np.concatenate([np.nonzero(~alive & ~self.alive)[0], dead])
        hole = new < 0                                       # otherwise it is a point not seen before
        born = free[:hole.sum()]
        assert len(born) == hole.sum(), "out of token slots"
        new[hole] = born; alive[born] = True
        obs[dead] = -1; obs[born] = -1
        self.alive, self.tok = alive, new
        return obs, born, dead


class PerStep:
    """the trivial baseline: no memory at all"""
    def __init__(self, N=N_TOK, K=8, **kw):
        self.K = K

    def update(self, obs, born, dead, tokens):
        lab = np.zeros(len(obs), int)
        for n, dv in enumerate(np.unique(obs[(obs >= 0) & (obs != ZERO)])[:self.K - 1]):
            lab[obs == dv] = n + 1
        return lab[tokens.tok]


class ArmM:
    def __init__(self, N=N_TOK, K=8, n_min=3, **kw):
        self.K, self.n_min = K, n_min
        self.S = np.zeros((N, K)); self.z = np.zeros(N, int)
        self.active = np.zeros(K, bool); self.active[0] = True
        self.full = 0

    def update(self, obs, born, dead, tokens):
        S, z, K = self.S, self.z, self.K
        S[dead] = 0; z[dead] = 0
        ids = np.nonzero(obs >= 0)[0]; o = obs[ids]
        d = np.full(K, -1)                                   # each file moves the way most of its tokens moved
        for k in np.nonzero(self.active)[0]:
            m = o[z[ids] == k]
            if len(m):
                d[k] = np.bincount(m, minlength=len(DISP)).argmax()
        own = S[ids, z[ids]].copy()
        S[ids] += (o[:, None] != d[None]) & (d[None] >= 0)
        # tokens that every file now predicts worse than their own file did before this step
        orphan = np.where(self.active[None], S[ids], np.inf).min(1) > own
        for dv in np.unique(o[orphan]):
            f = orphan & (o == dv)
            if f.sum() < self.n_min:
                continue
            free = np.nonzero(~self.active)[0]
            if len(free) == 0:
                self.full += 1; continue
            k = free[0]; self.active[k] = True
            S[:, k] = S[np.arange(len(S)), z]                # for every other token the new file starts level with its own
            S[ids, k] = own + (o != dv)
        cost = np.where(self.active[None], S[ids], np.inf)
        size = np.bincount(z[tokens.alive], minlength=K)
        cost = cost - 1e-6 * size[None] / max(1, size.max())  # a tie goes to the file the token is in, then to the largest
        cost[np.arange(len(ids)), z[ids]] -= 0.5
        z[ids] = cost.argmin(1)
        size = np.bincount(z[tokens.alive & (np.arange(len(z))[:, None] != born[None]).all(1)] if len(born) else z[tokens.alive], minlength=K)
        S[born] = 0; z[born] = size.argmax()                 # a point not seen before: no evidence, the largest file
        size = np.bincount(z[tokens.alive], minlength=K)
        self.active = size > 0
        return z[tokens.tok]


class ArmC:
    def __init__(self, N=N_TOK, K=8, a=1.2, b=4.6, j0=0.05, iters=40, eta=0.3, b0=1.0, b1=20.0, noise=0.1, seed=0, dev="cuda", **kw):
        self.K, self.a, self.b, self.j0, self.iters, self.eta, self.b0, self.b1, self.noise, self.dev = K, a, b, j0, iters, eta, b0, b1, noise, dev
        self.J = torch.zeros(N, N, device=dev)
        self.v = torch.zeros(N, K, device=dev); self.v[:H * H, 0] = 1.0     # at first everything is one object
        self.g = torch.Generator(device=dev).manual_seed(seed)

    def update(self, obs, born, dead, tokens):
        J, v, dev = self.J, self.v, self.dev
        gone = torch.as_tensor(np.concatenate([dead, born]), device=dev)
        J[gone] = 0; J[:, gone] = 0; v[gone] = 0
        ids = np.nonzero(obs >= 0)[0]; o = obs[ids]; mv = o != ZERO
        if mv.any():
            ti = torch.as_tensor(ids, device=dev); to = torch.as_tensor(o, device=dev); tm = torch.as_tensor(mv, device=dev)
            same = (to[tm][:, None] == to[None]).float()
            block = self.a * same - self.b * (1 - same)      # moved the same way; or one moved and the other did otherwise
            J[ti[tm][:, None], ti[None]] += block
            J[ti[~tm][:, None], ti[tm][None]] += block[:, ~tm].T
        V = torch.as_tensor(tokens.tok.ravel(), device=dev)
        Jv = J[V][:, V] + self.j0                            # no evidence yet: a weak pull to be one object
        Jv.fill_diagonal_(0)
        x = v[V]
        for t in range(self.iters):
            beta = self.b0 * (self.b1 / self.b0) ** (t / (self.iters - 1))
            p = torch.softmax(beta * x, -1)
            x = (1 - self.eta) * x + self.eta * (Jv @ p) + self.noise * (1 - t / self.iters) * torch.randn(x.shape, device=dev, generator=self.g)
        v[V] = x
        return x.argmax(-1).reshape(H, H).cpu().numpy()


def score(L, masks, hand, obj_pos, held, meta):
    """One row per frame and per thing (the hand, each object) that has moved by that frame and is in view.
    found: the label that covers most of it has intersection-over-union of at least 0.5 with it.
    with_other: that label also covers at least half of another thing that has moved.
    in_bg: that label is the one that covers most of the true background.
    Rows: frame, thing (1 hand, 3 + i object i), pixels in view, IoU, with_other, in_bg, held now, frames since it first moved, width."""
    rows = []
    T1 = len(L)
    things = [(1, hand)] + [(3 + i, obj_pos[:, i]) for i in range(meta["n"])]
    first = {}
    for i, pos in things:
        ch = np.nonzero(np.abs(np.diff(pos.astype(int), axis=0)).sum(1))[0]
        if len(ch):
            first[i] = ch[0] + 1
    for t in range(1, T1):
        now = [i for i in first if first[i] <= t and (masks[t] == i).any()]
        bgl = np.bincount(L[t][masks[t] == 0]).argmax()
        for i in now:
            m = masks[t] == i
            k = np.bincount(L[t][m]).argmax()
            e = L[t] == k
            iou = (e & m).sum() / (e | m).sum()
            other = any((e & (masks[t] == j)).sum() >= 0.5 * (masks[t] == j).sum() for j in now if j != i)
            rows.append((t, i, m.sum(), iou, other, k == bgl, i >= 3 and held[t] == i - 3, t - first[i], 7 if i == 1 else meta["sizes"][i - 3]))
    return rows


ARMS = dict(C=ArmC, M=ArmM, step=PerStep)


def run_episode(arm, frames, masks, hand, obj_pos, source="pixels", K=8, dev="cuda", **kw):
    img = frames.astype(np.float32) / 255
    if source == "pixels":
        f = torch.tensor(frames).to(dev).permute(0, 3, 1, 2).float() / 255
        idx, valid = M.flow(*M.cost_volume(f[:-1], f[1:]))
        idx, valid = idx.cpu().numpy(), valid.cpu().numpy()
    else:
        idx, valid = M.true_flow(masks, hand, obj_pos)
    tokens = Tokens(); model = ARMS[arm](K=K, dev=dev, **kw)
    L = np.zeros((len(frames), H, H), int)
    for t in range(len(frames) - 1):
        obs, born, dead = tokens.step(img[t], img[t + 1], idx[t], valid[t])
        L[t + 1] = model.update(obs, born, dead, tokens)
    return L, model
