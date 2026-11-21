"""Card 156: binding couplings as a learned function of appearance, trained by what the agent's own interaction shows.

Card 154's binding layer has couplings J_pq between points, accumulated from whether p and q moved together as the
hand acted, and (feedback) written back by a recognised stored pattern.  Both need the object to have moved once.

Here a network predicts, for any two points of a frame, what motion will show about them:
    f(appearance around p, appearance around q, their offset)  ->  P(together), P(apart), P(no evidence)
The training signal is the binding layer's own J at the end of each episode of the agent's experience.  Evidence
counts from three steps of motion on, the number card 154 asks before it stores a pattern (a is the gain of one
step moved together, b the loss of one step in which one moved and the other did not):
    J_pq >=  3a   the two were seen to move together        -> "together"
    J_pq <= -3b   one moved and the other did not            -> "apart"
    otherwise     nothing, or too little, was seen            -> "no evidence"
No mask, no designed similarity, no grouping prior: the label is what happened after the agent acted.

At the first sight of a scene the predicted couplings   A_pq = J_in P(together) + J_out P(apart)
(J_in, J_out: the medians of the couplings stored with the memory's patterns) are added to the binding energy,
    E(b) = -1/2 sum_pq (J_pq + A_pq) b_p.b_q ,
and the layer settles as in cards 147 and 154.  A is never used as a training label.
"""
import sys, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "tools/card154")
import feedback as Fb
W, B, M = Fb.W, Fb.B, Fb.M
dev = "cuda"; H = Fb.H; HW = H * H
YX = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij"), -1).reshape(-1, 2).float().to(dev) / H


class Pairs(nn.Module):
    def __init__(self, d=16, h=64, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.emb = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, d, 3, padding=1))
        self.head = nn.Sequential(nn.Linear(2 * d + 3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 3))
        self.to(dev); self.opt = torch.optim.Adam(self.parameters(), 2e-3); self.frames = []; self.pairs = []; self.labels = []; self.trained = False
        self.counts = torch.zeros(3, dtype=torch.float64, device=dev); self.prior = True; self.hard = False; self.base = True

    def embed(self, img):                                    # [B, H, H, 3] in 0..1 -> [B, HW, d]
        return self.emb(img.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)

    def logits(self, ep, eq, p, q):
        off = (YX[p] - YX[q]).abs()
        return self.head(torch.cat([(ep - eq).abs(), (ep + eq) / 2, off, off.norm(dim=-1, keepdim=True)], -1))

    def add(self, frame, cls, n=1500):
        """frame [H, H, 3] uint8; cls [HW, HW] on the device: 0 no evidence, 1 together, 2 apart.  The same number of
        pairs of each class."""
        ps, ls = [], []; self.counts += torch.bincount(cls.reshape(-1), minlength=3)
        for c in (1, 2):
            nz = torch.nonzero(cls == c)
            if len(nz):
                nz = nz[torch.randint(len(nz), (n,), device=dev)]; ps.append(nz); ls.append(torch.full((n,), c, device=dev))
        r = torch.randint(HW, (8 * n, 2), device=dev); r = r[(cls[r[:, 0], r[:, 1]] == 0) & (r[:, 0] != r[:, 1])][:n]
        ps.append(r); ls.append(torch.zeros(len(r), dtype=torch.long, device=dev))
        self.frames.append(torch.as_tensor(frame, device=dev).float() / 255); self.pairs.append(torch.cat(ps)); self.labels.append(torch.cat(ls))

    def fit(self, steps=300, batch=16):
        n = len(self.frames); g = torch.Generator(device="cpu").manual_seed(n)
        for _ in range(steps):
            idx = torch.randint(n, (min(batch, n),), generator=g).tolist()
            e = self.embed(torch.stack([self.frames[i] for i in idx])); loss = 0
            for j, i in enumerate(idx):
                p, q = self.pairs[i][:, 0], self.pairs[i][:, 1]
                loss = loss + F.cross_entropy(self.logits(e[j][p], e[j][q], p, q), self.labels[i])
            self.opt.zero_grad(); (loss / len(idx)).backward(); self.opt.step()
        self.trained = True
        return float(loss.detach()) / len(idx)

    @torch.no_grad()
    def couplings(self, frame, jin, jout):
        """the predicted couplings between every two pixels of a frame [H, H, 3] uint8: [HW, HW]"""
        e = self.embed((torch.as_tensor(frame, device=dev).float() / 255)[None])[0]; A = torch.zeros(HW, HW, device=dev); q = torch.arange(HW, device=dev); none = torch.zeros(HW, HW, dtype=torch.bool, device=dev)
        # the classes were shown in equal numbers; how often each occurs among all pairs is put back (the prior)
        shift = torch.log(self.counts / self.counts.sum()).float() if self.prior else torch.zeros(3, device=dev)
        for s in range(0, HW, 256):
            p = torch.arange(s, min(s + 256, HW), device=dev)
            pr = torch.softmax(self.logits(e[p][:, None].expand(-1, HW, -1), e[None].expand(len(p), -1, -1), p[:, None].expand(-1, HW), q[None].expand(len(p), -1)) + shift, -1)
            none[p] = pr.argmax(-1) == 0
            if self.hard:                                            # each pair takes its likeliest state
                c = pr.argmax(-1); A[p] = jin * (c == 1) + jout * (c == 2)
            else:
                A[p] = jin * pr[..., 1] + jout * pr[..., 2]
        if self.base and not self.hard and none.any():           # pairs expected to give no evidence carry none on average
            A[none] -= A[none].mean()
        A = (A + A.T) / 2; A.fill_diagonal_(0)
        return A


class Binding(Fb.Binding):
    """card 154's binding layer with a second set of couplings, the ones predicted from appearance"""
    def __init__(self, **kw):
        super().__init__(**kw); self.A = torch.zeros_like(self.J)

    def settle(self, tokens):
        J, v = self.J, self.v
        V = torch.as_tensor(tokens.tok.ravel(), device=self.dev)
        Jv = J[V][:, V] + self.A[V][:, V] + self.j0
        Jv.fill_diagonal_(0)
        x = v[V]
        for t in range(self.ticks):
            beta = self.b0 * (self.b1 / self.b0) ** (t / (self.ticks - 1))
            p = torch.softmax(beta * x, -1)
            x = (1 - self.eta) * x + self.eta * (Jv @ p) + self.noise * (1 - t / self.ticks) * torch.randn(x.shape, device=self.dev, generator=self.g)
        v[V] = x
        return x.argmax(-1).reshape(H, H).cpu().numpy()

    def update(self, obs, born, dead, tokens):
        gone = torch.as_tensor(np.concatenate([dead, born]), device=self.dev)
        self.A[gone] = 0; self.A[:, gone] = 0
        return super().update(obs, born, dead, tokens)


def stored_medians(mem, model):
    jin = [j for j in mem.Jin if j > 0]; jout = [j for j in mem.Jout if j < 0]
    if jin and jout:
        return float(np.median(jin)), float(np.median(jout))
    return 10 * model.a, -10 * model.b


def first_sight(frame, mem, net, seed, feedback=True):
    """card 154's first sight, with the couplings predicted from appearance added"""
    tokens = B.Tokens(); model = Binding(K=Fb.K, dev=dev, seed=seed); rec = []
    if feedback and len(mem):
        f = torch.as_tensor(frame, device=dev).float() / 255
        for j, h in enumerate(mem.match(f)):
            ids = tokens.tok[h["pix"]]
            model.recall(ids, mem.Jin[h["m"]], mem.Jout[h["m"]], tokens.tok.ravel(), j + 1)
            h["ids"] = ids; rec.append(h)
    if net is not None and net.trained:
        model.A[:HW, :HW] = net.couplings(frame, *stored_medians(mem, model))
    lab = model.settle(tokens) if (rec or (net is not None and net.trained)) else np.zeros((H, H), int)
    return lab, rec, tokens, model


def evidence(model, tok):
    """what the binding layer's own couplings say about every two visible points: 0 no evidence, 1 together, 2 apart"""
    V = torch.as_tensor(tok.ravel(), device=dev); JV = model.J[V][:, V]
    return (JV >= Fb.N_MIN * model.a).long() + 2 * (JV <= -Fb.N_MIN * model.b).long()


def truth(mask):
    """the same three classes from the simulator's mask, for the upper bound only: same thing, different things, both ground"""
    m = torch.as_tensor(mask.ravel(), device=dev); same = m[:, None] == m[None]; thing = (m[:, None] > 0) | (m[None] > 0)
    return (same & thing).long() + 2 * (~same).long()
