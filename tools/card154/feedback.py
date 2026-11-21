"""Card 154: feedback from the object modules to the binding modules, on sprite pixels with no masks.

  layer 0  motion modules, one per pixel (card 147's front end, no learning).
  layer 1  binding modules, one per followed point p.  State: a belief b_p over at most K objects.  Energy:
           -1/2 sum_pq J_pq b_p.b_q, J_pq the log-odds that p and q are parts of one object, accumulated from
           whether they moved together (card 147's couplings, settled as in card 150).
  layer 2  object modules.  Their memory holds stored patterns: for an object that the binding layer has found
           by its motion, where its points lie in its own frame, what they look like, and how strongly they
           were coupled to each other (J_in) and to everything else (J_out) when it was stored.

Feed-forward (cards 147, 150): a pattern is stored from a group of points that moved together.
Feedback (this card): at the first sight of a scene every stored pattern is tried at every place.  A pattern
recognised at a place writes its stored couplings back into layer 1: J_in between the points it explains, J_out
between them and every other point; and its module's belief that those points are its own.  Layer 1 then settles
as it always does.  So an object that has not moved
in this scene is bound by the couplings it earned when it last moved.

Recognition is a likelihood ratio with no hand-set threshold.  A pattern pixel "matches" when its error is within
sensor noise (card 147's THR_P).  Per pattern pixel not yet explained by another pattern: a match scores
-log q0 (q0: the share of pattern pixels that match at all places of this frame, i.e. by chance), a miss scores
log 1/2 (a pattern pixel is as likely hidden as seen).  A pattern is taken at a place when its score exceeds
log(number of patterns x number of places), the prior against any one hypothesis; the best is taken first and
the pixels it explains are no longer evidence for the others (explaining away).

Nothing here reads the true masks; they are used for scoring, and for the upper bound's memory.
"""
import sys, json, time, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "tools/card147"); sys.path.insert(0, "tools/card152")
import motion as M, binding as B
import world as W

dev = "cuda"
H, PS, T = M.H, 16, W.T
K = 12                    # objects a scene may hold (the scenes of the test hold up to 8 with the hand and the ground)
N_MIN = 3                 # a group is stored once it has moved in this many steps
LAM = float(np.log(2.0))  # the cost of a pattern pixel that is not seen: as likely hidden as not


# ---------------------------------------------------------------- the world: card 152's episodes with every frame
def episodes(world, n, seed, source="mixed", sizes=None):
    bgs = []; orig = W.background

    def rec(rng):
        b = orig(rng); bgs.append(b); return b
    W.background = rec
    try:
        eps = W.collect(world, n, seed, source) if sizes is None else W.collect(world, n, seed, source, sizes)
    finally:
        W.background = orig
    rng = np.random.default_rng(seed + 7919)
    for ep, bg in zip(eps, bgs):
        fr, mk = [ep["frame"]], [ep["mask"]]
        for t in range(1, T + 1):
            f, m = W.render(rng, bg, dict(hand=ep["hand"][t], pos=ep["pos"][t], order=ep["order"], held=ep["held"][t]), ep["objs"])
            fr.append(f); mk.append(m)
        ep["frames"] = np.stack(fr); ep["masks"] = np.stack(mk).astype(int)
    return eps


def key_of(ep, e, i):
    """a name for a true thing, for scoring only: the hand, a pool object, or a new object of one episode"""
    if i == 1:
        return "hand"
    o = ep["objs"][i - 3]
    return "pool%d" % o["uid"] if o["uid"] >= 0 else "new%s_%d" % (e, i - 3)


# ---------------------------------------------------------------- layer 2's memory of stored patterns
class Memory:
    def __init__(self):
        self.M = torch.zeros(0, PS, PS, dtype=torch.bool, device=dev); self.Tx = torch.zeros(0, PS, PS, 3, device=dev)
        self.Jin, self.Jout, self.info = [], [], []

    def __len__(self):
        return len(self.M)

    def store(self, mask, img, jin, jout, info):
        """mask [h, w] bool and img [h, w, 3] of the group's box"""
        h, w = mask.shape
        m = torch.zeros(1, PS, PS, dtype=torch.bool, device=dev); tx = torch.zeros(1, PS, PS, 3, device=dev)
        m[0, :h, :w] = torch.as_tensor(mask, device=dev); tx[0, :h, :w] = torch.as_tensor(img, device=dev).float() * m[0, :h, :w, None]
        self.M = torch.cat([self.M, m]); self.Tx = torch.cat([self.Tx, tx])
        self.Jin.append(float(jin)); self.Jout.append(float(jout)); self.info.append(info)

    @torch.no_grad()
    def match(self, frame, most=K - 1):
        """frame [H, H, 3] in 0..1 on the device.  Returns the patterns recognised, best first:
        dict(m, y, x, pix [H, H] bool: the pixels explained, score, matched, size)."""
        P = len(self)
        if P == 0:
            return []
        O = PS - 1; Hp = H + 2 * O; Q = H + O; NP = Q * Q               # a pattern's frame may start above or left of the image
        pad = torch.full((Hp, Hp, 3), 10.0, device=dev); pad[O:O + H, O:O + H] = frame
        U = F.unfold(pad.permute(2, 0, 1)[None], PS)[0].reshape(3, PS * PS, NP).permute(2, 1, 0)         # [places, 256, 3]
        Mf = self.M.reshape(P, -1); size = Mf.sum(1).float()
        g = torch.zeros(P, NP, PS * PS, dtype=torch.bool, device=dev)
        for s in range(0, P, 8):
            tx = self.Tx[s:s + 8].reshape(-1, 1, PS * PS, 3)
            g[s:s + 8] = (((U[None] - tx) ** 2).mean(-1) < M.THR_P) & Mf[s:s + 8, None]
        # a match counts for what it is worth: -log of how often this pattern pixel matches by chance, read from its
        # matches at all places of this frame (a pixel that looks like the ground is weak evidence)
        q0 = (g.float().mean(1) if P <= 64 else torch.cat([g[s:s + 64].float().mean(1) for s in range(0, P, 64)])).clamp(min=1.0 / NP)
        Wg = (LAM - torch.log(q0)) * Mf                                                             # [P, 256]
        thr = float(np.log(P * NP))
        claimed = torch.zeros(Hp, Hp, dtype=torch.bool, device=dev); claimed[:O] = True; claimed[O + H:] = True; claimed[:, :O] = True; claimed[:, O + H:] = True
        free = (F.unfold(claimed[None, None].float(), PS)[0].T < 0.5).float()                       # [places, 256]: in the image and not explained

        def scores(idx):
            return torch.stack([((g[i] & (free > 0)).float() * Wg[i]).sum(-1) for i in idx]) - LAM * (Mf[idx].float() @ free.T)
        score = scores(range(P))
        cand = torch.nonzero(score.max(1).values > thr)[:, 0]
        out = []; used = torch.zeros(len(cand), dtype=torch.bool, device=dev)
        score = score[cand]
        while len(cand) and len(out) < most:
            best = int(score.argmax()); c, s = best // NP, best % NP
            if float(score[c, s]) <= thr:
                break
            y, x = s // Q, s % Q; m = int(cand[c])
            ex = (g[m, s] & (free[s] > 0)).reshape(PS, PS)
            pix = torch.zeros(Hp, Hp, dtype=torch.bool, device=dev); pix[y:y + PS, x:x + PS] = ex
            out.append(dict(m=m, y=y - O, x=x - O, pix=pix[O:O + H, O:O + H].cpu().numpy(), score=round(float(score[c, s]), 1), matched=int(ex.sum()), size=int(size[m])))
            claimed |= pix; used[c] = True
            free = (F.unfold(claimed[None, None].float(), PS)[0].T < 0.5).float()
            score = scores(cand.tolist()); score[used] = -1e9
        return out


# ---------------------------------------------------------------- layer 1 with a way in for layer 2
class Binding(B.ArmC):
    def __init__(self, ticks=40, **kw):
        super().__init__(**kw); self.ticks = ticks

    def recall(self, ids, jin, jout, visible, slot):
        """a recognised pattern writes its stored couplings between the points it explains, and from them to the rest,
        and its module's belief that these points are its own (the module is object `slot` of the scene)"""
        J = self.J; N = len(J)
        ti = torch.as_tensor(ids, device=self.dev)
        self.v[ti] = 0; self.v[ti, slot] = 1
        rest = torch.zeros(N, dtype=torch.bool, device=self.dev); rest[torch.as_tensor(visible, device=self.dev)] = True; rest[ti] = False
        to = torch.nonzero(rest)[:, 0]
        J[ti[:, None], ti[None]] += jin; J[ti[:, None], to[None]] += jout; J[to[:, None], ti[None]] += jout

    def settle(self, tokens):
        J, v, dev = self.J, self.v, self.dev
        V = torch.as_tensor(tokens.tok.ravel(), device=dev)
        Jv = J[V][:, V] + self.j0
        Jv.fill_diagonal_(0)
        x = v[V]
        for t in range(self.ticks):
            beta = self.b0 * (self.b1 / self.b0) ** (t / (self.ticks - 1))
            p = torch.softmax(beta * x, -1)
            x = (1 - self.eta) * x + self.eta * (Jv @ p) + self.noise * (1 - t / self.ticks) * torch.randn(x.shape, device=dev, generator=self.g)
        v[V] = x
        return x.argmax(-1).reshape(H, H).cpu().numpy()

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
        return self.settle(tokens)


# ---------------------------------------------------------------- one scene, with or without feedback
def first_sight(frame, mem, feedback, seed):
    """the binding of a still frame.  Returns the label map, the patterns recognised, the token store and layer 1."""
    tokens = B.Tokens(); model = Binding(K=K, dev=dev, seed=seed); rec = []
    if feedback and len(mem):
        f = torch.as_tensor(frame, device=dev).float() / 255
        for j, h in enumerate(mem.match(f)):
            ids = tokens.tok[h["pix"]]
            model.recall(ids, mem.Jin[h["m"]], mem.Jout[h["m"]], tokens.tok.ravel(), j + 1)
            h["ids"] = ids; rec.append(h)
    lab = model.settle(tokens) if rec else np.zeros((H, H), int)
    return lab, rec, tokens, model


def in_frame(m):
    """the points of a group that fall in the PS x PS frame that holds the most of them"""
    c = np.pad(m.astype(int), ((1, PS), (1, PS))).cumsum(0).cumsum(1)
    box = c[PS:PS + H, PS:PS + H] - c[:H, PS:PS + H] - c[PS:PS + H, :H] + c[:H, :H]          # points in the frame starting at (y, x)
    y, x = np.unravel_index(box.argmax(), box.shape)
    out = np.zeros_like(m); out[y:y + PS, x:x + PS] = m[y:y + PS, x:x + PS]
    return out


def run_episode(ep, e, mem, feedback=True, learn=True, bound=False, seed=0):
    """the binding of every frame of an episode; with `learn`, groups that moved and that no stored pattern explains
    are stored.  `bound`: the memory is filled from the true masks instead (the upper bound's memory)."""
    frames = ep["frames"]; img = frames.astype(np.float32) / 255
    f = torch.tensor(frames).to(dev).permute(0, 3, 1, 2).float() / 255
    idx, valid = M.flow(*M.cost_volume(f[:-1], f[1:])); idx, valid = idx.cpu().numpy(), valid.cpu().numpy()
    L = np.zeros((T + 1, H, H), int); toks = np.zeros((T + 1, H, H), int)
    L[0], rec, tokens, model = first_sight(frames[0], mem, feedback, seed)
    toks[0] = tokens.tok
    moved = np.zeros(K, int); last = np.zeros(K, int); first = np.zeros(K, int)
    for t in range(T):
        obs, born, dead = tokens.step(img[t], img[t + 1], idx[t], valid[t])
        L[t + 1] = model.update(obs, born, dead, tokens); toks[t + 1] = tokens.tok
        lab = model.v.argmax(-1).cpu().numpy(); ids = obs >= 0
        n = np.bincount(lab[ids], minlength=K); mv = np.bincount(lab[ids & (obs != B.ZERO)], minlength=K)
        go = (mv >= 4) & (mv > 0.5 * n)
        first[go & (moved == 0)] = t + 1; moved += go; last[go] = t + 1
    log = dict(stored=0, known=0, too_big=0)
    if learn and bound:
        for i in [1] + [3 + j for j in range(len(ep["objs"]))]:
            pos = ep["hand"] if i == 1 else ep["pos"][:, i - 3]
            key = key_of(ep, e, i)
            if not np.abs(np.diff(pos.astype(int), axis=0)).sum() or any(q["key"] == key for q in mem.info):
                continue
            m = ep["masks"][0] == i; ys, xs = np.nonzero(m); y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            mem.store(m[y0:y1, x0:x1], img[0][y0:y1, x0:x1], 10 * model.a, -10 * model.b, dict(key=key, iou=1.0, n=int(m.sum()), episode=e))
            log["stored"] += 1
    elif learn:
        bgl = np.bincount(L[T].ravel(), minlength=K).argmax()
        vis = toks[T].ravel()
        for k in range(K):
            if k == bgl or moved[k] < N_MIN:
                continue
            ts = last[k]; m = L[ts] == k; ys, xs = np.nonzero(m)
            if len(ys) < 4:
                continue
            m = in_frame(m)                                       # a module's frame is PS x PS: it is put where it holds most of the group
            if m.sum() < 0.8 * len(ys):
                log["too_big"] += 1; continue
            ys, xs = np.nonzero(m); y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            ids = toks[ts][m]
            if any(np.isin(ids, r["ids"]).mean() >= 0.5 for r in rec):
                log["known"] += 1; continue
            g = toks[T][L[T] == k]
            if len(g) < 4:
                continue
            tg = torch.as_tensor(g, device=dev); rest = torch.as_tensor(np.setdiff1d(vis, g), device=dev)
            Jg = model.J[tg]; n = len(g)
            jin = float((Jg[:, tg].sum() - Jg[:, tg].diagonal().sum()) / (n * (n - 1))); jout = float(Jg[:, rest].mean())
            tm = ep["masks"][ts]; ti = int(np.bincount(tm[m]).argmax()); tr = tm == ti
            mem.store(m[y0:y1, x0:x1], img[ts][y0:y1, x0:x1], jin, jout,
                      dict(key=key_of(ep, e, ti) if ti else None, iou=round(float((m & tr).sum() / (m | tr).sum()), 3), n=int(m.sum()), episode=e, moved=int(moved[k]), first=int(first[k]), tid=ti))
            log["stored"] += 1
    return L, rec, log


# ---------------------------------------------------------------- scoring against the simulator
def score_frame(lab, rec, mem, mask, things):
    """things: list of (true id in the mask, key, pixels of the whole thing).  One row per thing with 8 pixels in view:
    key, share hidden, found (the label that covers most of it has IoU >= 0.5 with it, is not the ground's label and
    does not cover half of another thing), IoU, recognised as itself (the pattern that explains most of its pixels was
    stored from the same thing).  Also the number of labels that lie mostly on the ground."""
    rows = []
    bgl = np.bincount(lab[mask == 0]).argmax() if (mask == 0).any() else -1
    owner = np.full((H, H), -1)
    for j, r in enumerate(rec):
        owner[r["pix"]] = j
    for i, key, full in things:
        m = mask == i
        if m.sum() < 8:
            continue
        k = np.bincount(lab[m]).argmax(); e = lab == k
        iou = (e & m).sum() / (e | m).sum()
        other = any((e & (mask == j)).sum() >= 0.5 * (mask == j).sum() for j, _, _ in things if j != i and (mask == j).any())
        own = owner[m]; who = np.bincount(own[own >= 0]).argmax() if (own >= 0).sum() >= 0.25 * m.sum() else -1
        rows.append(dict(key=key, hidden=round(1 - m.sum() / full, 3), found=bool(iou >= 0.5 and k != bgl and not other), iou=round(float(iou), 3),
                         itself=bool(who >= 0 and mem.info[rec[who]["m"]]["key"] == key), claimed=bool(who >= 0)))
    spurious = sum(1 for k in np.unique(lab) if k != bgl and (lab == k).sum() >= 8 and (mask[lab == k] == 0).mean() > 0.5)
    return rows, spurious


def things_of(objs, e, ep=None):
    hm, _ = W.hand_sprite()
    out = [(1, "hand", int(hm.sum()))]
    for i, o in enumerate(objs):
        out.append((3 + i, "pool%d" % o["uid"] if o["uid"] >= 0 else "new%s_%d" % (e, i), int(o["mask"].sum())))
    return out
