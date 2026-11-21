"""Card 156: one episode of experience for the binding layer, its memory of patterns and the appearance couplings.

As card 154's run_episode, with three additions:
  - at first sight the couplings predicted from appearance are added (appearance.first_sight);
  - at the end of the episode the binding layer's own couplings are the appearance network's training labels;
  - a group that appearance bound at first sight and that did not move is stored as a pattern too (with no
    couplings of its own: nothing was seen of it), so that the object modules can keep its transitions.
"""
import sys, numpy as np, torch
sys.path.insert(0, "tools/card156")
import appearance as Ap
Fb, B, M, W = Ap.Fb, Ap.B, Ap.M, Ap.W
dev, H, T, K, PS, N_MIN = Ap.dev, Fb.H, Fb.T, Fb.K, Fb.PS, Fb.N_MIN


def run_episode(ep, e, mem, net, learn=True, seed=0, labels="own", stills=True):
    """an episode already recorded (the script's).  labels: "own" (the binding layer's couplings), "true" (from the
    simulator's masks: the upper bound) or None"""
    frames = ep["frames"]; img = frames.astype(np.float32) / 255
    f = torch.tensor(frames).to(dev).permute(0, 3, 1, 2).float() / 255
    idx, valid = M.flow(*M.cost_volume(f[:-1], f[1:])); idx, valid = idx.cpu().numpy(), valid.cpu().numpy()
    S = Seen(frames[0], mem, net, seed)
    for t in range(T):
        S.step(img[t], img[t + 1], idx[t], valid[t])
    return S.L, S.rec, (finish(ep, e, mem, net, S, labels, stills) if learn else {})


class Seen:
    """what the binding layer makes of an episode, frame by frame"""
    def __init__(self, frame0, mem, net, seed):
        self.L = np.zeros((T + 1, H, H), int); self.toks = np.zeros((T + 1, H, H), int); self.t = 0
        self.L[0], self.rec, self.tokens, self.model = Ap.first_sight(frame0, mem, net, seed)
        self.toks[0] = self.tokens.tok
        self.moved = np.zeros(K, int); self.last = np.zeros(K, int); self.first = np.zeros(K, int)
        self.obs = []; self.who = []          # card 157: each step's readings per point; the patterns stored from this episode

    def step(self, a, b, idx, valid):
        t = self.t
        obs, born, dead = self.tokens.step(a, b, idx, valid); self.obs.append(obs)
        self.L[t + 1] = self.model.update(obs, born, dead, self.tokens); self.toks[t + 1] = self.tokens.tok
        lab = self.model.v.argmax(-1).cpu().numpy(); ids = obs >= 0
        n = np.bincount(lab[ids], minlength=K); mv = np.bincount(lab[ids & (obs != B.ZERO)], minlength=K)
        go = (mv >= 4) & (mv > 0.5 * n)
        self.first[go & (self.moved == 0)] = t + 1; self.moved += go; self.last[go] = t + 1; self.t += 1


def finish(ep, e, mem, net, S, labels="own", stills=True):
    """the learning at the end of an episode: patterns stored, and the appearance network's labels"""
    frames = ep["frames"]; img = frames.astype(np.float32) / 255
    L, toks, rec, model, moved, last, first = S.L, S.toks, S.rec, S.model, S.moved, S.last, S.first
    log = dict(stored=0, known=0, too_big=0, still=0)
    bg0 = np.bincount(L[0].ravel(), minlength=K).argmax()
    at_first = [toks[0][L[0] == k] for k in range(K) if k != bg0 and (L[0] == k).sum() >= 8]      # the groups of the first sight

    def info(m, t, **kw):
        tm = ep["masks"][t]; ti = int(np.bincount(tm[m]).argmax()); tr = tm == ti
        return dict(key=Fb.key_of(ep, e, ti) if ti else None, iou=round(float((m & tr).sum() / (m | tr).sum()), 3), n=int(m.sum()), episode=e, tid=ti, **kw)
    bgl = np.bincount(L[T].ravel(), minlength=K).argmax(); vis = toks[T].ravel(); taken = []
    for k in range(K):
        if k == bgl or moved[k] < N_MIN:
            continue
        ts = last[k]; m = L[ts] == k; ys, xs = np.nonzero(m)
        if len(ys) < 4:
            continue
        m = Fb.in_frame(m)
        if m.sum() < 0.8 * len(ys):
            log["too_big"] += 1; continue
        ys, xs = np.nonzero(m); y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        ids = toks[ts][m]; taken.append(ids)
        if any(np.isin(ids, r["ids"]).mean() >= 0.5 for r in rec):
            log["known"] += 1; continue
        g = toks[T][L[T] == k]
        if len(g) < 4:
            continue
        tg = torch.as_tensor(g, device=dev); rest = torch.as_tensor(np.setdiff1d(vis, g), device=dev)
        Jg = model.J[tg]; n = len(g)
        jin = float((Jg[:, tg].sum() - Jg[:, tg].diagonal().sum()) / (n * (n - 1))); jout = float(Jg[:, rest].mean())
        seen0 = any(np.isin(ids, a).mean() >= 0.5 for a in at_first)                       # it was already a group at first sight
        mem.store(m[y0:y1, x0:x1], img[ts][y0:y1, x0:x1], jin, jout, info(m, ts, moved=int(moved[k]), first=1 if seen0 else int(first[k]), still=False))
        log["stored"] += 1; S.who.append(dict(p=len(mem) - 1, ids=ids, first=1 if seen0 else int(first[k])))
    if stills:
        for a in at_first:
            if any(np.isin(a, r["ids"]).mean() >= 0.5 for r in rec) or any(np.isin(a, ids).mean() >= 0.5 for ids in taken):
                continue
            m = Fb.in_frame(np.isin(toks[0], a))
            if m.sum() < 0.8 * len(a):
                continue
            ys, xs = np.nonzero(m); y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            mem.store(m[y0:y1, x0:x1], img[0][y0:y1, x0:x1], 0.0, 0.0, info(m, 0, moved=0, first=1, still=True))
            log["still"] += 1; S.who.append(dict(p=len(mem) - 1, ids=toks[0][m], first=1))
    if net is not None and labels:
        net.add(frames[T], Ap.evidence(model, toks[T]) if labels == "own" else Ap.truth(ep["masks"][T]))
    return log
