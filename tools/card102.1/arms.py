"""Card 102.1: arm C asked about every token, with relations learned from raw positions.

Arm A, a pair network. Each token's input is its what (encoder vector) and its raw where: the tile's column and row
in the observation's 9 x 9 grid of tiles (the carried rows included), as two numbers. Nothing else is declared. Two
layers; in each, a pair function (one hidden layer on token i's and token j's states, both raw wheres and the action,
entering linearly) gives every pair 4 attention logits and a message, and token i adds the attention-weighted
messages to its state. Heads per token: P(it changes), and what it becomes by card 095.1's four ways per part (keep,
copy from another token by a pair pointer, shift, set), snapped to the nearest known token. Trained on the training
steps by the agent's own labels: a token changed when its code at the same place differs one step later.

Arm B, the vote. Stored token instances of the same action and the same code (identity first) count by the nearness
of arm A's final state for them (the token's context): instances per (action, code) are grouped into at most 256
context centroids with their counts (card 095.2's grouping), and a query counts each centroid by
exp(-||context_q - centroid|| / h). Arm C: P = (changed count + alpha * P_A) / (count + alpha); the result is the
vote's result with the largest weighted count where that count exceeds alpha_r (arm A's pseudo-count for its own
result), else arm A's. alpha_r is chosen like alpha, on the last tenth (the share of changed tokens whose result is
right); alpha_r = 0 is "the vote's result wherever it has one". Results are stored as relations: a result equal to the code
of the token arm A's copy pointer picks is stored as "copy from it" (card 095.1's copy way), any other as its code. Arm A is fitted on the first nine tenths of the training
episodes; h and alpha are chosen by the own-label likelihood of the last tenth; the store then holds every training
step.

Forms: tokens (card 098.3's codes and vectors); truth (the upper bound: the evaluator's kinds, heading and item
counts as one-hot tokens); twoplace (the baseline: tokens, each token's attention limited to itself, the token in
front and the carried rows, version 27's front and hand).

Crafter's steps: the curriculum's, awake, not a move (a move shifts the view; the agent's own motion is card
102.5's). Breaking world 1: every step. Scored (evaluator): Crafter's non-creature tiles in view and carried items
other than the vitals; world 1's lamps.

  bin/prun python tools/card102.1/arms.py crafter|world1 tokens|truth|twoplace [--steps N] [--worlds K] [--seed S] [--tag T]
      [--rel]   (revision 1: positions only as differences between two tokens' wheres)
      [--gates] (revision 2: independent gates per pair, closed by default, fewest open within 5% of the loss)
      [--locus] (revision 2, arm GL: each pair also reads both tokens' offsets from the action's locus)
      [--soft]  (revision 3: gates as in revision 2, never pushed shut: no cost and no pruning phase)
      → runs/102.1/<world>_<form>.json, _pred.npz, _net.pt
  bin/prun python tools/card102.1/arms.py compare      → runs/102.1/compare.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "102.1"
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
NT = 81
WHERE = torch.as_tensor(np.array([(k % 9, k // 9) for k in range(NT)], np.float32) / 8.0)
MOVES = (1, 2, 3, 4)
CREATURES = (14, 15, 16, 17)                            # cow, zombie, skeleton, arrow
GRASS, TABLE = 2, 11
MAKE_SLOT = {11: 73, 12: 74, 13: 75, 14: 76, 15: 77, 16: 78}   # make action → its tool's carried slot
W1_STEP = {1: -1, 2: 1, 3: -9, 4: 9}                   # world 1: a move's target, as a token index offset
NP_, HEADS = 4, 4
SOFT = "--soft" in sys.argv                            # revision 3: the gates are never pushed shut (no cost, no pruning)
GATES = "--gates" in sys.argv or SOFT                  # revision 2: independent gates per pair, closed by default
LOCUS = "--locus" in sys.argv                          # revision 2, arm GL: positions also measured from the action's locus
REL = "--rel" in sys.argv or GATES or LOCUS            # revision 1: positions enter only as differences between tokens
HC_BETA, HC_GAMMA, HC_ZETA = 2 / 3, -0.1, 1.1          # hard-concrete gates (Louizos, Welling and Kingma 2018)
SLACK = 0.05                                           # revision 2: the loss may rise 5% above its ungated level
LAM = float(arg("--lam", "nan")) if "--lam" in sys.argv else None   # or: a fixed cost in nats per open gate per state


# ------------------------------------------------------------------------------------------------ data
def vec_truth(sem, face, inv):
    n = len(sem)
    V = np.zeros((n, NT, 49), np.float16)
    r, c = np.nonzero(sem >= 0)
    V[r, c, sem[r, c]] = 1
    hd = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    V[np.arange(n), 31, 19 + np.array([hd.get((int(a), int(b)), 0) for a, b in face])] = 1
    for k in range(16):
        V[:, 63 + k, 23 + k] = 1
        V[np.arange(n), 63 + k, 39 + inv[:, k].clip(0, 9)] = 1
    return V


def crafter_steps(ws, form):
    keys = ("V", "Va", "code", "codea", "act", "lab0", "lab1", "scored", "ep", "front")
    out = {k: [] for k in keys}
    for w in ws:
        z = np.load(RUNS / f"curr_w{w}.npz")
        act, ep, done, sleep = z["act"].astype(np.int64), z["ep"], z["done"], z["sleep"]
        n = len(act)
        nxt = np.r_[ep[1:] == ep[:-1], False] & ~done
        t = np.flatnonzero(nxt & ~sleep & ~np.r_[sleep[1:], True] & ~np.isin(act, MOVES))
        sem = z["sem"].reshape(n, 63).astype(np.int64)
        inv, face = z["inv"].astype(np.int64), z["face"].astype(np.int64)
        lab = np.full((n, NT), -1, np.int64)
        lab[:, :63] = sem
        lab[:, 63:79] = 100 + 10 * np.arange(16)[None] + inv.clip(0, 9)
        lab[:, 31] = 1300 + (face[:, 0] + 1) * 3 + (face[:, 1] + 1)        # the agent and its heading
        if form == "truth":
            V0, V1 = vec_truth(sem[t], face[t], inv[t]), vec_truth(sem[t + 1], face[t + 1], inv[t + 1])
            code = np.where(lab < 0, 9998, lab)
        else:
            V = np.load(RUNS / f"curr_vec_w{w}.npy")
            V0, V1 = V[t], V[t + 1]
            code = np.load(RUNS / f"curr_code_w{w}.npy").astype(np.int64)
        scored = np.zeros((len(t), NT), bool)
        scored[:, :63] = (sem[t] >= 0) & ~np.isin(sem[t], CREATURES) & ~np.isin(sem[t + 1], CREATURES)
        scored[:, 31] = False
        scored[:, 67:79] = True                         # carried items; the vitals (slots 63-66) are left out
        for k, v in (("V", V0), ("Va", V1), ("code", code[t]), ("codea", code[t + 1]), ("act", act[t]),
                     ("lab0", lab[t]), ("lab1", lab[t + 1]), ("scored", scored), ("ep", ep[t]),
                     ("front", (3 + face[t, 1]) * 9 + (4 + face[t, 0]))):
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def world1_steps(split, form):
    z = np.load(ROOT / "runs" / "card102" / f"world1_{split}.npz")
    act, ep, done = z["act"].astype(np.int64), z["ep"], z["done"]
    n = len(act)
    t = np.flatnonzero(np.r_[ep[1:] == ep[:-1], False] & ~done)
    sem = z["sem"].reshape(n, 63).astype(np.int64)
    lab = np.full((n, NT), 99, np.int64)
    lab[:, :63] = sem
    if form == "truth":
        def vt(i):
            V = np.zeros((len(i), NT, 8), np.float16)
            V[np.arange(len(i))[:, None], np.arange(63)[None], sem[i]] = 1
            V[:, 63:, 7] = 1
            return V
        V0, V1, code = vt(t), vt(t + 1), lab
    else:
        V = np.load(RUNS / f"world1_vec_{split}.npy")
        V0, V1 = V[t], V[t + 1]
        code = np.load(RUNS / f"world1_code_{split}.npy").astype(np.int64)
    agent = np.argmax(sem == 2, 1)[t]
    front = agent + np.array([W1_STEP.get(int(a), 0) for a in act[t]])
    scored = np.zeros((len(t), NT), bool)
    scored[:, :63] = np.isin(sem[t], (4, 5))            # lamps, lit or not: the world's question
    return {"V": V0, "Va": V1, "code": code[t], "codea": code[t + 1], "act": act[t], "lab0": lab[t],
            "lab1": lab[t + 1], "scored": scored, "ep": ep[t], "front": front, "push": z["push"][t],
            "locus": (np.stack([agent % 9, agent // 9], 1) / 8.0).astype(np.float32)}   # declared: removed by 102.5


def load(world, form, nworlds=8):
    f = "truth" if form == "truth" else "tokens"
    if world == "crafter":
        return crafter_steps(range(nworlds), f), crafter_steps((8, 9), f)
    return world1_steps("train", f), world1_steps("test", f)


def sub(d, m):
    return {k: v[m] for k, v in d.items()}


# ------------------------------------------------------------------------------------------------ arm A
class PairLayer(nn.Module):
    def __init__(self, d, na, rel=False, gates=False, locus=False):
        super().__init__()
        if locus:                                       # each token's offset from the action's locus
            self.li, self.lj = nn.Linear(2, 2 * d, bias=False), nn.Linear(2, 2 * d, bias=False)
        self.gates, self.locus = gates, locus
        self.hi, self.hj = nn.Linear(d, 2 * d), nn.Linear(d, 2 * d, bias=False)
        if rel:                                         # one linear map of where_j - where_i
            self.wd = nn.Linear(2, 2 * d, bias=False)
        else:
            self.wi, self.wj = nn.Linear(2, 2 * d, bias=False), nn.Linear(2, 2 * d, bias=False)
        self.act = nn.Embedding(na, 2 * d)
        self.out = nn.Linear(2 * d, d + HEADS + 1)
        self.norm = nn.LayerNorm(d)
        self.rel = rel

    def forward(self, h, W, a, mask, locus=None):
        n, L, d = h.shape
        if self.rel:                                    # wd(W_j - W_i) = wd(W_j) - wd(W_i), as wd is linear
            pw = self.wd(W)
            pi, pj = self.hi(h) - pw + self.act(a)[:, None], self.hj(h) + pw
        else:
            pi = self.hi(h) + self.wi(W) + self.act(a)[:, None]
            pj = self.hj(h) + self.wj(W)
        if self.locus:
            off = W - locus[:, None]
            pi, pj = pi + self.li(off), pj + self.lj(off)
        o = self.out(torch.relu(pi[:, :, None] + pj[:, None]))         # (n, L, L, d + HEADS + 1)
        logit = o[..., d:d + HEADS]
        m = o[..., :d].view(n, L, L, HEADS, d // HEADS)
        if self.gates:                                  # one gate per pair and head; all closed = no outside cause
            bad = torch.eye(L, dtype=torch.bool, device=h.device)[None]
            if mask is not None:
                bad = bad | ~mask
            if self.training:
                u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
                s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logit) / HC_BETA)
            else:
                s = torch.sigmoid(logit)
            if SOFT:                                    # revision 3: a binary-concrete gate, never exactly shut
                g = s.masked_fill(bad[..., None], 0)
                self.p_open = g
            else:
                g = (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0, 1).masked_fill(bad[..., None], 0)
                self.p_open = torch.sigmoid(logit - HC_BETA * np.log(-HC_GAMMA / HC_ZETA)).masked_fill(bad[..., None], 0)
            msg = ((g[..., None] * m).sum(2) / (1 + g.sum(2))[..., None]).reshape(n, L, d)
            return self.norm(h + msg), o[..., -1], g
        if mask is not None:
            logit = logit.masked_fill(~mask[..., None], -1e9)
        att = torch.softmax(logit, 2)                                   # over j, per head
        msg = (att[..., None] * m).sum(2).reshape(n, L, d)
        return self.norm(h + msg), o[..., -1], att


class PairNet(nn.Module):
    def __init__(self, dv, na=17, d=64, rel=None, gates=None, locus=None):
        super().__init__()
        self.rel = REL if rel is None else rel
        self.gates = GATES if gates is None else gates
        self.locus = LOCUS if locus is None else locus
        self.what = nn.Sequential(nn.Linear(dv, d), nn.ReLU(), nn.Linear(d, d))
        if not self.rel:                                # revision 1: a token's state holds no absolute where
            self.where = nn.Linear(2, d, bias=False)
        self.layers = nn.ModuleList([PairLayer(d, na, self.rel, self.gates, self.locus) for _ in range(2)])
        self.change = nn.Linear(d, 1)
        self.gate = nn.Linear(d, NP_ * 4)
        self.shift, self.set = nn.Linear(d, dv), nn.Linear(d, dv)
        self.dv = dv

    def forward(self, X, a, mask=None, W=None, locus=None):
        n, L = X.shape[:2]
        if W is None:
            W = WHERE.to(X.device)[None].expand(n, -1, -1)
        if self.locus and locus is None:                # Crafter: the view's centre, where its adapter puts the agent
            locus = WHERE[31].to(X.device)[None].expand(n, -1)
        h = self.what(X) if self.rel else self.what(X) + self.where(W)
        atts = []
        for layer in self.layers:
            h, ptr, att = layer(h, W, a, mask, locus)
            atts.append(att)
        if self.gates:                                  # expected open gates per state (the L0 cost)
            self.open = sum(layer.p_open.sum((1, 2, 3)) for layer in self.layers)
        bad = torch.eye(L, dtype=torch.bool, device=X.device)[None]
        if mask is not None:
            bad = bad | ~mask
        ptr = ptr.masked_fill(bad, -1e9)
        copy = torch.softmax(ptr, -1) @ X
        pw = self.dv // NP_
        g = torch.softmax(self.gate(h).view(n, L, NP_, 4), -1)
        ways = torch.stack([X, copy, X + self.shift(h), self.set(h)], -1)[..., :NP_ * pw, :]
        after = (ways.reshape(n, L, NP_, pw, 4) * g[..., None, :]).sum(-1).reshape(n, L, NP_ * pw)
        after = torch.cat([after, X[..., NP_ * pw:]], -1)
        return self.change(h)[..., 0], after, h, atts, ptr.argmax(-1)


def mask_of(d, i, form):
    """Two-place: each token reads itself, the token in front and the carried rows."""
    if form != "twoplace":
        return None
    n = len(i)
    m = torch.zeros(n, NT, NT, dtype=torch.bool, device=dev)
    m[:, torch.arange(NT), torch.arange(NT)] = True
    m[torch.arange(n), :, torch.as_tensor(d["front"][i], device=dev)] = True
    m[:, :, 63:] = True
    return m


def tensors(d, i, form):
    X = torch.as_tensor(d["V"][i], device=dev).float()
    a = torch.as_tensor(d["act"][i], device=dev)
    m = mask_of(d, i, form)
    locus = torch.as_tensor(d["locus"][i], device=dev) if "locus" in d else None
    if "W" in d:                                        # token sets of any size (MiniGrid): wheres given, pads masked
        valid = torch.as_tensor(d["valid"][i], device=dev)
        vm = valid[:, None, :].expand(-1, valid.shape[1], -1)
        return X, a, (vm if m is None else m & vm), torch.as_tensor(d["W"][i], device=dev), locus
    return X, a, m, None, locus


def train_a(d, form, steps, seed=1021, bs=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PairNet(d["V"].shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(d["act"])
    t0 = time.monotonic()
    pw = d["w"] / d["w"].sum() if "w" in d else None   # MiniGrid: tries drawn by memory's try weights
    # revision 2: the first half trains without a cost on gates and sets the loss level tau; the second half finds
    # the fewest open gates whose loss stays within SLACK of tau (a multiplier mu raised while the loss is within it,
    # lowered while it is not; SPARTAN, Lei, Schoelkopf and Posner 2025; GECO, Rezende and Viola 2018)
    mu, tau, ema, half = 1e-4, None, None, steps // 2
    for step in range(steps):
        i = np.sort(rng.integers(0, n, bs) if pw is None else rng.choice(n, bs, p=pw))
        T = tensors(d, i, form)
        logit, after, _, _, _ = net(*T)
        own = torch.as_tensor(d["code"][i] != d["codea"][i], device=dev)
        bce = fn.binary_cross_entropy_with_logits(logit, own.float(), reduction="none")
        if "valid" in d:
            bce = bce * torch.as_tensor(d["valid"][i], device=dev)
        lc = bce.sum(1).mean()
        tgt = torch.as_tensor(d["Va"][i], device=dev).float()
        la = ((after - tgt).abs().sum(-1) * own).sum(1).mean()
        loss = lc + 0.1 * la
        if net.gates and not SOFT:
            ema = lc.item() if ema is None else 0.99 * ema + 0.01 * lc.item()
            if step == half:
                tau = ema
            if LAM is not None and step >= half:       # a fixed cost per open gate, paid in the state where it is open
                loss = loss + LAM * min(1.0, (step - half) / 1000) * net.open.mean()
            elif tau is not None:
                mu = min(10.0, mu * 1.01) if ema <= tau * (1 + SLACK) else max(1e-7, mu / 1.01)
                loss = loss + mu * net.open.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0 or step == 99:
            print({"step": step + 1, "change": round(lc.item(), 4), "after": round(la.item(), 4),
                   "s": round(time.monotonic() - t0, 1)} | ({"open gates per state": round(net.open.mean().item(), 1),
                                                            "mu": float(f"{mu:.3g}"), "tau": tau and round(tau, 4)}
                                                           if net.gates else {}), flush=True)
    return net


@torch.no_grad()
def run_a(net, d, form, bs=256):
    P, H, A, S = [], [], [], []
    for s in range(0, len(d["act"]), bs):
        i = np.arange(s, min(s + bs, len(d["act"])))
        logit, after, h, _, src = net(*tensors(d, i, form))
        P.append(torch.sigmoid(logit).cpu().numpy())
        H.append(h.half().cpu().numpy())
        A.append(after.half().cpu().numpy())
        S.append(src.cpu().numpy().astype(np.int16))
    return np.concatenate(P), np.concatenate(H), np.concatenate(A), np.concatenate(S)


# ------------------------------------------------------------------------------------------------ arm B and C
def kmeans(X, k, iters=10, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device=dev)[:k]].clone()
    for _ in range(iters):
        j = torch.cdist(X, C).argmin(1)
        s = torch.zeros_like(C).index_add_(0, j, X)
        c = torch.bincount(j, minlength=k).float()
        C = torch.where(c[:, None] > 0, s / c.clamp_min(1)[:, None], C)
    return C, torch.cdist(X, C).argmin(1)


def groups(key):
    order = np.argsort(key, kind="stable")
    ks, starts = np.unique(key[order], return_index=True)
    return order, ks, starts, np.r_[starts[1:], len(order)]


COPY = -7                                              # a result stored as a relation: the code of arm A's pointer


def relation(code, codea, src):
    """A changed token's result: COPY when it is the code of the token arm A's pointer picks, else the code."""
    at_src = np.take_along_axis(code, src.astype(np.int64), 1)
    return np.where(codea == at_src, COPY, codea)


def build_store(d, H, S, K=256):
    """Per (action, code): context centroids with counts, changed counts and result histograms (codes or COPY)."""
    store = {}
    own = (d["code"] != d["codea"]).reshape(-1)
    res = relation(d["code"], d["codea"], S).reshape(-1)
    Hf = H.reshape(-1, H.shape[-1])
    order, ks, starts, ends = groups((d["act"][:, None] * 100000 + d["code"]).reshape(-1))
    for kk, s, e in zip(ks, starts, ends):
        idx = order[s:e]
        X = torch.as_tensor(Hf[idx], device=dev).float()
        C, j = kmeans(X, K) if len(idx) > K else (X, torch.arange(len(idx), device=dev))
        j = j.cpu().numpy()
        cnt = np.bincount(j, minlength=len(C)).astype(np.float32)
        chg = np.bincount(j, weights=own[idx].astype(np.float32), minlength=len(C)).astype(np.float32)
        rv, ri = np.unique(res[idx][own[idx]], return_inverse=True)
        M = np.zeros((len(C), len(rv)), np.float32)    # per centroid: counts of each result
        np.add.at(M, (j[own[idx]], ri), 1)
        store[int(kk)] = (C, cnt, chg, rv, M)
    return store


def vote(store, act, code, H, hw, results=True):
    """Per token: weighted count, weighted changed count, the vote's best result (-1 none) and its weighted count."""
    n, L = code.shape
    N, Nc, R = np.zeros(n * L, np.float32), np.zeros(n * L, np.float32), np.full(n * L, -1, np.int64)
    RW = np.zeros(n * L, np.float32)
    Hf = H.reshape(-1, H.shape[-1])
    order, ks, starts, ends = groups((act[:, None] * 100000 + code).reshape(-1))
    for kk, s, e in zip(ks, starts, ends):
        if int(kk) not in store:
            continue
        C, cnt, chg, rv, M = store[int(kk)]
        idx = order[s:e]
        for b in range(0, len(idx), 16384):
            ii = idx[b:b + 16384]
            k = torch.exp(-torch.cdist(torch.as_tensor(Hf[ii], device=dev).float(), C) / hw).cpu().numpy()
            N[ii], Nc[ii] = k @ cnt, k @ chg
            if results and len(rv):
                Wr = k @ M
                bi = Wr.argmax(1)
                R[ii], RW[ii] = rv[bi], Wr[np.arange(len(ii)), bi]
    return N.reshape(n, L), Nc.reshape(n, L), R.reshape(n, L), RW.reshape(n, L)


def ll(p, y, mask):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return ((y * np.log(p) + (1 - y) * np.log(1 - p)) * mask).sum(1)


def snap(after, known_V, known_code):
    K = torch.as_tensor(known_V, device=dev).float()
    A = after.reshape(-1, after.shape[-1])
    out = [torch.cdist(torch.as_tensor(A[s:s + 100000], device=dev).float(), K, p=1).argmin(1).cpu()
           for s in range(0, len(A), 100000)]
    return known_code[torch.cat(out).numpy()].reshape(after.shape[:2])


def results_of(d, S, R, RW, resA, alpha_r):
    """The vote's result where its weighted count exceeds arm A's pseudo-count alpha_r, else arm A's."""
    resV = np.where(R == COPY, np.take_along_axis(d["code"], S.astype(np.int64), 1), R)
    return np.where((R != -1) & (RW > alpha_r), resV, resA)


ALPHA_R = (0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0, np.inf)


class ArmC:
    def __init__(self, net, store, hw, alpha, form, known, alpha_r=0.0):
        self.net, self.store, self.hw, self.alpha, self.form, self.known = net, store, hw, alpha, form, known
        self.alpha_r = alpha_r

    def __call__(self, d):
        PA, H, AF, S = run_a(self.net, d, self.form)
        N, Nc, R, RW = vote(self.store, d["act"], d["code"], H, self.hw)
        P = (Nc + self.alpha * PA) / (N + self.alpha)
        resA = snap(AF, *self.known)
        return P, PA, N, results_of(d, S, R, RW, resA, self.alpha_r), resA


def choose_alpha_r(store, vd, H_v, S_v, resA, hw):
    """alpha_r by the share of the validation tokens that changed whose result is right (own labels)."""
    _, _, R, RW = vote(store, vd["act"], vd["code"], H_v, hw)
    ch = vd["code"] != vd["codea"]
    acc = {a: float((results_of(vd, S_v, R, RW, resA, a) == vd["codea"])[ch].mean()) for a in ALPHA_R}
    return max(ALPHA_R, key=lambda a: (acc[a], -a)), acc


def fit(tr, form, steps):
    """Arm A on nine tenths of the training episodes, h and alpha on the last tenth, the store on all."""
    eps = np.unique(tr["ep"])
    val = np.isin(tr["ep"], eps[int(0.9 * len(eps)):])
    ft, vd = sub(tr, ~val), sub(tr, val)
    net = train_a(ft, form, steps, seed=int(arg("--seed", "1021")))
    _, H_f, _, S_f = run_a(net, ft, form)
    PA_v, H_v, AF_v, S_v = run_a(net, vd, form)
    store = build_store(ft, H_f, S_f)
    del H_f
    rng = np.random.default_rng(0)
    a = H_v.reshape(-1, H_v.shape[-1])[rng.integers(0, H_v.shape[0] * H_v.shape[1], 4000)].astype(np.float32)
    scale = float(np.median(np.linalg.norm(a[:2000] - a[2000:], axis=1)))
    y = (vd["code"] != vd["codea"]).astype(np.float32)
    grid = []
    for hw in (0.02, 0.05, 0.1, 0.25):
        N, Nc, _, _ = vote(store, vd["act"], vd["code"], H_v, hw * scale, results=False)
        for alpha in (0.3, 1.0, 3.0, 10.0, 30.0):
            grid.append((float(ll((Nc + alpha * PA_v) / (N + alpha), y, np.ones_like(y)).mean()), hw, alpha))
    best = max(grid)
    allc = np.concatenate([tr["code"].reshape(-1), tr["codea"].reshape(-1)])
    u, first = np.unique(allc, return_index=True)
    u, first = u[u >= 0], first[u >= 0]                 # pads (MiniGrid) are not tokens
    nt = tr["code"].size
    V0, V1 = tr["V"].reshape(-1, tr["V"].shape[-1]), tr["Va"].reshape(-1, tr["V"].shape[-1])
    known_V = np.stack([V0[f] if f < nt else V1[f - nt] for f in first]).astype(np.float32)
    alpha_r, acc = choose_alpha_r(store, vd, H_v, S_v, snap(AF_v, known_V, u), best[1] * scale)
    del store
    _, H_t, _, S_t = run_a(net, tr, form)
    store = build_store(tr, H_t, S_t)
    return ArmC(net, store, best[1] * scale, best[2], form, (known_V, u), alpha_r), {
        "kernel width (x median context distance)": best[1], "median context distance": round(scale, 4),
        "alpha": best[2], "validation own-label log-likelihood per step": round(best[0], 4),
        "alpha_r (results)": alpha_r, "validation results right, by alpha_r": {str(k): round(v, 4) for k, v in acc.items()}}


# ------------------------------------------------------------------------------------------------ scoring
def label_of_code(tr):
    """To score results only: each (code, place)'s most common evaluator label in training, and each code's for pairs
    not seen. A code's meaning can depend on its place: Crafter's carried slots draw the count, and the slot names the
    item, so one code is "count 2" of whichever item its slot holds."""
    c = np.concatenate([tr["code"], tr["codea"]])
    l = np.concatenate([tr["lab0"], tr["lab1"]])
    pl = np.broadcast_to(np.arange(c.shape[1]), c.shape)
    out = []
    for key in ((c * 1000 + pl).reshape(-1), c.reshape(-1)):
        order, ks, starts, ends = groups(key)
        lf = l.reshape(-1)[order]
        m = {}
        for k, s, e in zip(ks, starts, ends):
            v, n = np.unique(lf[s:e], return_counts=True)
            m[int(k)] = int(v[n.argmax()])
        out.append(m)
    return out


def labels(res, m):
    by_pair, by_code = m
    key = res * 1000 + np.arange(res.shape[1])[None]
    u, inv = np.unique(key, return_inverse=True)
    return np.array([by_pair.get(int(k), by_code.get(int(k) // 1000, -2)) for k in u])[inv].reshape(res.shape)


def score(world, tr, te, P, res, m=None):
    y, sc = te["lab1"] != te["lab0"], te["scored"]
    L = ll(P, y.astype(np.float32), sc)
    rl = labels(res, m if m is not None else label_of_code(tr))
    pred = P >= 0.5
    right = np.where(y, pred & (rl == te["lab1"]), ~pred)
    whole = (right | ~sc).all(1)
    out = {}
    gs = {"make": (11, 12, 13, 14, 15, 16), "do": (5,), "place": (7, 8, 9, 10), "noop": (0,), "sleep": (6,)} \
        if world == "crafter" else {"all": (0, 1, 2, 3, 4)}
    for g, acts in gs.items():
        s = np.isin(te["act"], acts)
        out[g] = {"steps": int(s.sum()), "log-likelihood per step": round(float(L[s].mean()), 4),
                  "whole scored next state right": round(float(whole[s].mean()), 4),
                  "scored tokens that changed": int((y & sc)[s].sum()),
                  "changed tokens right (change and result)": round(float(right[y & sc & s[:, None]].mean()), 4)
                  if (y & sc & s[:, None]).any() else None}
    if world == "crafter":
        mk = np.isin(te["act"], list(MAKE_SLOT))
        slot = np.array([MAKE_SLOT.get(int(a), 73) for a in te["act"]])
        made = mk & y[np.arange(len(y)), slot]
        out["makes that made a tool"] = {"tries": int(made.sum()),
                                         "whole scored next state right": round(float(whole[made].mean()), 4)}
    else:
        lamp, push = sc, te["push"][:, None]
        for name, s in (("lamps at pushes, changed", lamp & push & y), ("lamps at pushes, unchanged", lamp & push & ~y),
                        ("lamps lit", lamp & y & (te["lab1"] == 5))):
            out[name] = {"tokens": int(s.sum()), "P >= 0.5 iff changed": round(float((pred == y)[s].mean()), 4)}
    return out, L, whole


# ------------------------------------------------------------------------------------------------ criterion 2 (Crafter)
NEAR8 = [r * 9 + c for r in (2, 3, 4) for c in (3, 4, 5) if (r, c) != (3, 4)]
FAR16 = [r * 9 + c for r in range(1, 6) for c in range(2, 7) if max(abs(r - 3), abs(c - 4)) == 2]
SQ5 = NEAR8 + FAR16


def role_test(arm, te):
    """Held-out makes that made a tool, with exactly one table in the 5 x 5 square: the table's token moved to each
    of the 8 near and 16 far places (its old place given a grass token of the same frame from outside the square)."""
    y = te["lab1"] != te["lab0"]
    slot = np.array([MAKE_SLOT.get(int(a), 73) for a in te["act"]])
    mk = np.isin(te["act"], list(MAKE_SLOT)) & y[np.arange(len(y)), slot]
    rows, skipped = [], 0
    for s in np.flatnonzero(mk):
        tab = [p for p in SQ5 if te["lab0"][s, p] == TABLE]
        grass = [p for p in range(63) if p not in SQ5 and te["lab0"][s, p] == GRASS]
        if len(tab) != 1 or not grass:
            skipped += 1
            continue
        p0, g = tab[0], grass[0]
        v = {k: np.repeat(te[k][s:s + 1], 24, 0) for k in ("V", "code", "act", "front")}
        for j, q in enumerate(NEAR8 + FAR16):
            v["V"][j, p0], v["code"][j, p0] = te["V"][s, g], te["code"][s, g]
            v["V"][j, q], v["code"][j, q] = te["V"][s, p0], te["code"][s, p0]
        rows.append(arm(v)[0][:, slot[s]])
    R = np.array(rows).reshape(-1, 24)
    ok = (R[:, :8] >= 0.5).all(1) & (R[:, 8:] < 0.5).all(1)
    f = lambda x: round(float(x), 4) if len(R) else None
    return {"tests": len(R), "skipped (not one table in the 5 x 5 square, or no grass outside)": skipped,
            "share passing (all 8 near >= 0.5, all 16 far < 0.5)": f(ok.mean()) if len(R) else None,
            "mean P(made) near": f(R[:, :8].mean()) if len(R) else None,
            "mean P(made) far": f(R[:, 8:].mean()) if len(R) else None,
            "near places >= 0.5": f((R[:, :8] >= 0.5).mean()) if len(R) else None,
            "far places < 0.5": f((R[:, 8:] < 0.5).mean()) if len(R) else None}


@torch.no_grad()
def attention_report(arm, te):
    """Make steps that made a tool: the tool slot's attention (both layers, heads averaged) over the 81 places."""
    y = te["lab1"] != te["lab0"]
    slot = np.array([MAKE_SLOT.get(int(a), 73) for a in te["act"]])
    s = np.flatnonzero(np.isin(te["act"], list(MAKE_SLOT)) & y[np.arange(len(y)), slot])[:256]
    if not len(s):
        return {}
    _, _, _, atts, _ = arm.net(*tensors(te, s, arm.form))
    out = {}
    for li, att in enumerate(atts):
        a = att.mean(-1)[torch.arange(len(s)), torch.as_tensor(slot[s], device=dev)].cpu().numpy()   # (n, 81)
        lab = te["lab0"][s]
        if arm.net.gates:                               # gates: shares of the open gates' weight
            out[f"layer {li + 1} open gates (all heads)"] = round(float((att[torch.arange(len(s)), torch.as_tensor(
                slot[s], device=dev)] > (0.5 if SOFT else 0)).sum((1, 2)).float().mean()), 2)
            a = a / (a.sum(1, keepdims=True) + 1e-9)
        out[f"layer {li + 1}"] = {
            "on the table tokens": round(float((a * (lab == TABLE)).sum(1).mean()), 4),
            "on the agent": round(float(a[:, 31].mean()), 4),
            "on carried slots": round(float(a[:, 63:].sum(1).mean()), 4),
            "on itself": round(float(a[np.arange(len(s)), slot[s]].mean()), 4),
            "map over view rows (9 x 7)": np.round(a[:, :63].mean(0).reshape(7, 9), 3).tolist()}
    return out


@torch.no_grad()
def lamp_report(arm, te):
    """World 1: where lamps that changed at a push put their layer-1 weight, by the evaluator's kind."""
    names = {0: "floor", 1: "wall", 2: "agent", 3: "box", 4: "lamp", 5: "lit lamp", 99: "empty row"}
    y = te["lab1"] != te["lab0"]
    rows = np.flatnonzero((te["scored"] & te["push"][:, None] & y).any(1))[:300]
    _, _, _, atts, _ = arm.net(*tensors(te, rows, arm.form))
    a = atts[0].mean(-1).cpu().numpy()
    tot = dict.fromkeys(names.values(), 0.0)
    for r, t in enumerate(rows):
        for i in np.flatnonzero(te["scored"][t] & y[t]):
            for k, v in names.items():
                tot[v] += float(a[r, i][te["lab0"][t] == k].sum())
    z = sum(tot.values()) or 1.0
    return {k: round(v / z, 3) for k, v in tot.items()}


@torch.no_grad()
def open_gates(arm, te, n=512):
    _, _, _, atts, _ = arm.net(*tensors(te, np.arange(min(n, len(te["act"]))), arm.form))
    return {f"layer {li + 1}": round(float((g > (0.5 if SOFT else 0)).sum((1, 2, 3)).float().mean()), 1)
            for li, g in enumerate(atts)}


# ------------------------------------------------------------------------------------------------ main
def main():
    world, form = sys.argv[1], sys.argv[2]
    steps, nworlds = int(arg("--steps", "20000")), int(arg("--worlds", "8"))
    tag = arg("--tag", "")
    t0 = time.monotonic()
    tr, te = load(world, form, nworlds)
    print(f"{world} {form}: train steps {len(tr['act'])}, test {len(te['act'])} ({time.monotonic() - t0:.0f} s)",
          flush=True)
    arm, rep = fit(tr, form, steps)
    print("fitted", rep, f"({time.monotonic() - t0:.0f} s)", flush=True)
    t1 = time.monotonic()
    P, PA, N, res, resA = arm(te)
    rep["query seconds per step"] = round((time.monotonic() - t1) / len(te["act"]), 5)
    sc, L, whole = score(world, tr, te, P, res)
    rep.update({"world": world, "form": form, "steps": steps, "training worlds": nworlds, "results": sc})
    rep["arm A alone"] = score(world, tr, te, PA, resA)[0]
    if world == "crafter":
        rep["role test (criterion 2)"] = role_test(arm, te)
        rep["attention of the made tool's slot"] = attention_report(arm, te)
    else:
        rep["lamps that changed at pushes: layer-1 weight by kind"] = lamp_report(arm, te)
    if arm.net.gates:
        rep["open gates per state (test, both layers, all heads)"] = open_gates(arm, te)
    rep["store instances (centroids)"] = int(sum(len(v[1]) for v in arm.store.values()))
    rep["seconds"] = round(time.monotonic() - t0, 1)
    name = f"{world}_{form}{tag}"
    np.savez_compressed(RUNS / f"{name}_pred.npz", P=P, PA=PA, N=N, res=res, resA=resA, L=L, whole=whole,
                        ep=te["ep"], act=te["act"])
    torch.save(arm.net.state_dict(), RUNS / f"{name}_net.pt")
    (RUNS / f"{name}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "attention of the made tool's slot"}, indent=1), flush=True)


def compare():
    """Criterion 1: make steps' log-likelihood, tokens against the two-place query, paired bootstrap over episodes."""
    tag = arg("--tag", "")
    a, b = np.load(RUNS / f"crafter_tokens{tag}_pred.npz"), np.load(RUNS / "crafter_twoplace_pred.npz")
    out = {}
    for g, acts in (("make", list(MAKE_SLOT)), ("do", [5]), ("place", [7, 8, 9, 10])):
        s = np.isin(a["act"], acts)
        dl, ep = a["L"][s] - b["L"][s], a["ep"][s]
        u, inv = np.unique(ep, return_inverse=True)
        tot, cnt = np.bincount(inv, dl), np.bincount(inv)
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            k = rng.integers(0, len(u), len(u))
            boot.append(tot[k].sum() / cnt[k].sum())
        boot = np.array(boot)
        out[g] = {"steps": int(s.sum()), "episodes": len(u), "difference per step (tokens - two-place)":
                  round(float(dl.mean()), 4), "95% interval": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
                  "p (difference <= 0)": float((boot <= 0).mean())}
    (RUNS / f"compare{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def rescore():
    """Arm C's results scored again from the saved predictions (runs scored before results were labelled by code and
    place); arm A alone is rescored where its results were saved."""
    world, form = sys.argv[2], sys.argv[3]
    tr, te = load(world, form)
    m = label_of_code(tr)
    z = np.load(RUNS / f"{world}_{form}_pred.npz")
    rep = json.loads((RUNS / f"{world}_{form}.json").read_text())
    sc, L, whole = score(world, tr, te, z["P"], z["res"], m)
    rep["results"] = sc
    if "resA" in z.files:
        rep["arm A alone"] = score(world, tr, te, z["PA"], z["resA"], m)[0]
    else:
        rep["arm A alone (results not saved: change log-likelihood only)"] = {
            g: {"log-likelihood per step": v["log-likelihood per step"]} for g, v in rep.pop("arm A alone").items()
            if "log-likelihood per step" in v}
    rep["scoring"] = "results labelled by (code, place)"
    np.savez_compressed(RUNS / f"{world}_{form}_pred.npz", **{k: z[k] for k in z.files if k != "whole"}, whole=whole)
    (RUNS / f"{world}_{form}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(sc, indent=1))


if __name__ == "__main__":
    {"compare": compare, "rescore": rescore}.get(sys.argv[1], main)()
