"""Card 076: a learned router from the situation's tokens to remembered orders, scored beside the planner.

Data: runs/076/*.json (tools/card076/collect.py). A weighing is a query (the two needs' kinds and actions, whether A
holds now), the situation as tokens (handle, dx, dy, marks; offsets from the achiever's target) and the planner's
outcome ("n first" is an order).

Recall, two levels (cards 042, 050):
- **own:** stored weighings of the identical situation (the same query, the same code at every offset with the same
  marks) are the evidence, the router's vote their prior (weight ALPHA_OWN);
- **router:** e = rho(sum_tokens phi(token, the named tokens, the query)); a stored weighing counts by
  exp(-|w * (e_query - e_stored)|_1); P(order) = (sum k y + alpha p0) / (sum k + alpha).
Fitted by the leave-one-out likelihood of the training outcomes, leaving out whole layouts (card 071), real orders
weighted by their rarity. Memory at test: every training weighing, plus the episode's earlier weighings (each added
after it is predicted; card 074's protocol). An order is predicted when P >= 0.5 (recall's convention).

Baselines: always "no order"; card 074's recall (own: the query and the codes of the two named tiles and the held
tile; else neighbours by one fitted width over their vectors); a hand-built recall (own key adds the offset between
the named tiles and the codes next to each; else card 074's neighbours). Simplification declared: the hand-built
recall is an exact key, not card 049's admission.

  bin/prun python tools/card076/router.py --out runs/076/router.json
"""
import json
import sys
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D = Path("runs/076")
OUT = Path(next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == "--out"), D / "router.json"))
HUES = ("red", "green", "blue", "purple", "yellow", "grey")
ALPHA_OWN = 1e-3
STEPS, BATCH, BANK, LR = 3000, 256, 4096, 1e-3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(76)
RNG = np.random.default_rng(76)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


# ---------------------------------------------------------------- data

def load(files):
    """Weighings as dicts with layout, episode, source; a global handle table (code, vector)."""
    W, H = [], {}
    for f, src in files:
        if not (D / f).exists():
            continue
        d = json.load(open(D / f))
        for ep in d["per_episode"]:
            for h, (code, vec) in ep.get("handles", {}).items():
                H[int(h)] = (code, vec)
            lay = (src.split(" ")[0], int(ep["seed"]))         # one layout per seed and world (decoy folds share it)
            for k, s in enumerate(ep.get("situations", [])):
                W.append({**s, "layout": lay, "episode": (f, int(ep["seed"])), "k": k, "source": src})
    return W, H


TRAIN = [("train_tier1.json", "tier1"), ("train_tier2.json", "tier2"), ("train_tier2b.json", "tier2"),
         ("train_decoy.json", "decoy key known")]
TEST = ([("tier1.json", "tier1"), ("tier2.json", "tier2")] + [(f"known_{h}.json", "decoy key known") for h in HUES]
        + [(f"decoy_{h}.json", "decoy trying") for h in HUES])


def query_key(w):
    return (w["A"][0], w["A"][1], w["n"][0], w["n"][1], w["A_holds"])


def named(w, bit):
    """The token a need names: the hand if named, else the one nearest the target."""
    ts = [t for t in w["tokens"] if t[3] & bit]
    if not ts:
        return None
    hand = [t for t in ts if t[3] & 1]
    return (hand or sorted(ts, key=lambda t: abs(t[1]) + abs(t[2])))[0]


class Feat:
    """Padded token tensors for a list of weighings."""

    def __init__(self, H, qvocab):
        self.hid = {h: i for i, h in enumerate(sorted(H))}
        self.V = torch.tensor(np.array([H[h][1] for h in sorted(H)], np.float32), device=DEV)
        self.code = {h: H[h][0] for h in H}
        self.qv = qvocab

    def q(self, w):
        v = np.zeros(len(self.qv), np.float32)
        for k, x in enumerate(query_key(w)):
            j = self.qv.get((k, x))
            if j is not None:
                v[j] = 1
        return v

    def build(self, ws):
        n, m = len(ws), max(len(w["tokens"]) for w in ws)
        idx = np.zeros((n, m), np.int64)
        xy = np.zeros((n, m, 2), np.float32)
        mk = np.zeros((n, m, 4), np.float32)
        mask = np.zeros((n, m), np.float32)
        Q = np.stack([self.q(w) for w in ws])
        for i, w in enumerate(ws):
            t = np.array(w["tokens"], np.int64)
            k = len(t)
            idx[i, :k] = [self.hid[h] for h in t[:, 0]]
            xy[i, :k] = t[:, 1:3]
            mk[i, :k] = (t[:, 3:4] >> np.arange(4)) & 1
            mask[i, :k] = 1
        g = lambda a: torch.tensor(a, device=DEV)
        return {"idx": g(idx), "xy": g(xy), "mk": g(mk), "mask": g(mask), "q": g(Q)}


class Router(nn.Module):
    def __init__(self, nq, h=64, de=32):
        super().__init__()
        din = 32 + 3 + 4 + 2 * (32 + 2) + nq
        self.phi = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h + nq, h), nn.ReLU(), nn.Linear(h, de))
        self.w = nn.Parameter(torch.zeros(de))
        self.la = nn.Parameter(torch.tensor(0.0))

    def embed(self, V, b):
        z = V[b["idx"]]                                          # (n, m, 32)
        xy = b["xy"] / 6.0
        mk, mask, q = b["mk"], b["mask"], b["q"]
        tok = torch.cat([z, xy, xy.abs().sum(-1, keepdim=True), mk], -1)
        ctx = []
        for bit in (1, 2):                                       # the tokens A names, then n
            sel = mk[..., bit:bit + 1] * mask[..., None]
            cnt = sel.sum(1).clamp(min=1)
            ctx += [(z * sel).sum(1) / cnt, (xy * sel).sum(1) / cnt]
        ctx = torch.cat(ctx + [q], -1)
        x = torch.cat([tok, ctx[:, None].expand(-1, tok.shape[1], -1)], -1)
        s = (self.phi(x) * mask[..., None]).sum(1)
        return self.rho(torch.cat([s, q], -1))

    def kernel(self, ea, eb):
        w = nn.functional.softplus(self.w)
        return torch.exp(-(torch.abs(ea[:, None] - eb[None]) * w).sum(-1))

    def sums(self, ea, eb, yb, lower=False, chunk=128):
        """(sum_j k_ij y_j, sum_j k_ij) over eb for each row of ea, in chunks; lower: only j < i (ea is eb)."""
        num, den = [], []
        for s in range(0, len(ea), chunk):
            K = self.kernel(ea[s:s + chunk], eb)
            if lower:
                i = torch.arange(s, s + K.shape[0], device=K.device)[:, None]
                K = K * (torch.arange(len(eb), device=K.device)[None] < i)
            num.append(K @ yb)
            den.append(K.sum(1))
        return torch.cat(num), torch.cat(den)

    def vote(self, K, y, p0):
        a = nn.functional.softplus(self.la)
        return (K @ y + a * p0) / (K.sum(1) + a)


def sub(b, i):
    return {k: v[i] for k, v in b.items()}


def embed_all(model, V, b, chunk=2048):
    out = []
    with torch.no_grad():
        for s in range(0, len(b["idx"]), chunk):
            out.append(model.embed(V, sub(b, slice(s, s + chunk))))
    return torch.cat(out)


def fit(train, F, steps=STEPS, log=print, leave_layout=True):
    b = F.build(train)
    y = torch.tensor([float(w["outcome"] == "n first") for w in train], device=DEV)
    lay = {l: i for i, l in enumerate(sorted({w["layout"] for w in train}))}
    L = torch.tensor([lay[w["layout"]] for w in train], device=DEV)
    p0 = float(y.mean())
    wpos = (1 - p0) / max(p0, 1e-6)
    model = Router(len(F.qv)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(train)
    for t in range(steps):
        qi = torch.randint(n, (min(BATCH, n),), device=DEV)
        bi = torch.randint(n, (min(BANK, n),), device=DEV)
        eq, eb = model.embed(F.V, sub(b, qi)), model.embed(F.V, sub(b, bi))
        K = model.kernel(eq, eb)
        K = K * ((L[qi][:, None] != L[bi][None]) if leave_layout else (qi[:, None] != bi[None]))   # layouts, or itself
        P = model.vote(K, y[bi], p0).clamp(1e-6, 1 - 1e-6)
        yq = y[qi]
        loss = -(wpos * yq * torch.log(P) + (1 - yq) * torch.log(1 - P)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % 500 == 0:
            log(f"  step {t} loss {loss.item():.4f}")
    model.eval()
    return model, b, y, p0


def sit_key(w, F):
    return query_key(w) + tuple(sorted((F.code[t[0]], t[1], t[2], t[3]) for t in w["tokens"]))


# ---------------------------------------------------------------- recall over stored weighings

class OwnRouter:
    def __init__(self, model, F, train, btrain, ytrain, p0):
        self.m, self.F, self.p0 = model, F, p0
        self.etr = embed_all(model, F.V, btrain)
        self.ytr = ytrain
        self.own = defaultdict(lambda: [0, 0])
        for w in train:
            o = self.own[sit_key(w, F)]
            o[0] += 1
            o[1] += w["outcome"] == "n first"

    def episode(self, ws):
        """P(order) for an episode's weighings in order, each added to memory after it is predicted."""
        b = self.F.build(ws)
        e = embed_all(self.m, self.F.V, b)
        with torch.no_grad():
            ye = torch.tensor([float(w["outcome"] == "n first") for w in ws], device=DEV)
            nt, dt = self.m.sums(e, self.etr, self.ytr)
            ne, de = self.m.sums(e, e, ye, lower=True)
            a = nn.functional.softplus(self.m.la)
            P = ((nt + ne + a * self.p0) / (dt + de + a)).cpu().numpy()
        local = defaultdict(lambda: [0, 0])
        out = []
        for w, p in zip(ws, P):
            k = sit_key(w, self.F)
            n0, o0 = self.own.get(k, (0, 0))
            n1, o1 = local[k]
            nn_, oo = n0 + n1, o0 + o1
            out.append((oo + ALPHA_OWN * p) / (nn_ + ALPHA_OWN) if nn_ else float(p))
            local[k][0] += 1
            local[k][1] += w["outcome"] == "n first"
        return out


class KeyRecall:
    """Card 074's recall (hand=False) or the hand-built recall (hand=True): an exact key, else neighbours by one
    fitted width over the named tiles' and the held tile's vectors."""

    def __init__(self, train, F, hand, lam=None):
        self.F, self.hand = F, hand
        self.Vn = F.V.cpu().numpy()
        self.own = defaultdict(lambda: [0, 0])
        self.X, self.y, self.lay = [], [], []
        for w in train:
            self.add(w, self.own)
            self.X.append(self.vec(w))
            self.y.append(float(w["outcome"] == "n first"))
            self.lay.append(w["layout"])
        self.X, self.y, self.lay = np.array(self.X), np.array(self.y), np.array([hash(l) for l in self.lay])
        self.p0 = float(self.y.mean())
        self.lam = lam if lam is not None else self.fit_lam()

    def key(self, w):
        code = lambda t: None if t is None else self.F.code[t[0]]
        a, n = named(w, 2), named(w, 4)
        held = next(t for t in w["tokens"] if t[3] & 1)
        k = query_key(w) + (code(a), code(n), code(held))
        if self.hand:
            off = None if a is None or n is None else (n[1] - a[1], n[2] - a[2])
            at = {(t[1], t[2]): self.F.code[t[0]] for t in w["tokens"] if not t[3] & 1}
            nb = lambda t: () if t is None else tuple(at.get((t[1] + dx, t[2] + dy))
                                                      for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            k = k + (off, nb(a), nb(n))
        return k

    def vec(self, w):
        v = lambda t: np.zeros(32) if t is None else self.Vn[self.F.hid[t[0]]]
        held = next(t for t in w["tokens"] if t[3] & 1)
        return np.concatenate([v(named(w, 2)), v(named(w, 4)), v(held)])

    def add(self, w, own):
        o = own[self.key(w)]
        o[0] += 1
        o[1] += w["outcome"] == "n first"

    def neigh(self, x, lam, exclude=None):
        k = np.exp(-lam * np.abs(self.X - x).sum(1))
        if exclude is not None:
            k = k * (self.lay != exclude)
        return (k @ self.y + self.p0) / (k.sum() + 1)

    def fit_lam(self):
        idx = RNG.choice(len(self.X), min(1500, len(self.X)), replace=False)
        best = None
        for lam in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            ll = 0.0
            for i in idx:
                p = min(max(self.neigh(self.X[i], lam, self.lay[i]), 1e-6), 1 - 1e-6)
                ll += np.log(p if self.y[i] else 1 - p)
            if best is None or ll > best[0]:
                best = (ll, lam)
        return best[1]

    def episode(self, ws):
        local = defaultdict(lambda: [0, 0])
        out = []
        for w in ws:
            k = self.key(w)
            n0, o0 = self.own.get(k, (0, 0))
            n1, o1 = local[k]
            if n0 + n1:
                out.append((o0 + o1) / (n0 + n1))
            else:
                out.append(float(self.neigh(self.vec(w), self.lam)))
            self.add(w, local)
        return out


# ---------------------------------------------------------------- scoring

def score(rows, pred):
    """rows: weighings; pred: P(order). Agreement, the no-order baseline, McNemar, orders caught and precision."""
    y = np.array([w["outcome"] == "n first" for w in rows], dtype=bool)
    p = np.array(pred) >= 0.5
    right, none_right = p == y, ~y
    b, c = int((right & ~none_right).sum()), int((~right & none_right).sum())
    return {"weighings": int(len(y)), "agree": round(float(right.mean()), 4) if len(y) else None,
            "always_none": round(float(none_right.mean()), 4) if len(y) else None,
            "mcnemar_vs_none": {"b": b, "c": c, "p": round(mcnemar(b, c), 5)},
            "orders": int(y.sum()), "caught": round(float(p[y].mean()), 4) if y.any() else None,
            "precision": round(float(y[p].mean()), 4) if p.any() else None, "predicted_orders": int(p.sum())}


def by_episode(ws):
    eps = defaultdict(list)
    for w in ws:
        eps[w["episode"]].append(w)
    for k in eps:
        eps[k].sort(key=lambda w: w["k"])
    return eps


def run_recall(rec, test):
    pred = {}
    for k, ws in by_episode(test).items():
        for w, p in zip(ws, rec.episode(ws)):
            pred[id(w)] = p
    return [pred[id(w)] for w in test]


def report(test, preds, subsets):
    out = {}
    for name, p in preds.items():
        r = {"all": score(test, p)}
        for s in sorted({w["source"] for w in test}):
            ii = [i for i, w in enumerate(test) if w["source"] == s]
            r[s] = score([test[i] for i in ii], [p[i] for i in ii])
        for sn, ii in subsets.items():
            r[sn] = score([test[i] for i in ii], [p[i] for i in ii])
        out[name] = r
    return out


def main():
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    train, H1 = load(TRAIN)
    test, H2 = load(TEST)
    H = {**H1, **H2}
    qkeys = sorted({(k, x) for w in train + test for k, x in enumerate(query_key(w))}, key=str)
    F = Feat(H, {k: i for i, k in enumerate(qkeys)})
    res = {"note": "Card 076, tools/card076/router.py", "train": len(train), "test": len(test),
           "train_orders": int(sum(w["outcome"] == "n first" for w in train)),
           "test_orders": int(sum(w["outcome"] == "n first" for w in test)),
           "train_layouts": len({w["layout"] for w in train}),
           "train_by_source": {s: [sum(w["source"] == s for w in train),
                                   sum(w["source"] == s and w["outcome"] == "n first" for w in train)]
                               for s in sorted({w["source"] for w in train})}}
    log(json.dumps(res))
    # gate (a): do identical inputs ever disagree?
    grp = defaultdict(list)
    for w in train:
        grp[sit_key(w, F)].append(w["outcome"] == "n first")
    major = sum(max(sum(v), len(v) - sum(v)) for v in grp.values())
    res["gate_identical_inputs_consistent"] = round(major / len(train), 4)
    # fit
    t0 = time.monotonic()
    model, btr, ytr, p0 = fit(train, F, log=log)
    res["fit_seconds"] = round(time.monotonic() - t0, 1)
    # gate (b): a router fitted and scored on the training weighings, no layout left out (the weighing itself excluded)
    gm, _, _, _ = fit(train, F, log=lambda m: None, leave_layout=False)
    with torch.no_grad():
        e = embed_all(gm, F.V, btr)
        num, den = gm.sums(e, e, ytr)
        a = nn.functional.softplus(gm.la)
        P = ((num - ytr + a * p0) / (den - 1 + a)).cpu().numpy()      # the weighing itself (k = 1) left out
    res["gate_router_on_training"] = score(train, P)
    log(f"gate: identical inputs consistent {res['gate_identical_inputs_consistent']}; router on training "
        f"{res['gate_router_on_training']}")
    # test
    trained_named = {F.code[t[0]] for w in train for t in w["tokens"] if t[3] & 6}
    off = lambda w: (query_key(w), None if named(w, 2) is None or named(w, 4) is None else
                     (named(w, 4)[1] - named(w, 2)[1], named(w, 4)[2] - named(w, 2)[2]))
    trained_arr = {off(w) for w in train}
    subsets = {"unseen colour": [i for i, w in enumerate(test) if any(F.code[t[0]] not in trained_named
                                                                      for t in w["tokens"] if t[3] & 6)],
               "unseen arrangement": [i for i, w in enumerate(test) if off(w) not in trained_arr]}
    t0 = time.monotonic()
    preds = {"router": run_recall(OwnRouter(model, F, train, btr, ytr, p0), test)}
    res["router_seconds_per_prediction"] = round((time.monotonic() - t0) / max(1, len(test)), 6)
    preds["always none"] = [0.0] * len(test)
    preds["card 074 recall"] = run_recall(KeyRecall(train, F, hand=False), test)
    preds["hand-built recall"] = run_recall(KeyRecall(train, F, hand=True), test)
    res["report"] = report(test, preds, subsets)
    for name, r in res["report"].items():
        log(f"{name}: {json.dumps(r['all'])}")
    # P19's curve: training layouts added until the real orders reach 10, 30, 100, 300
    curve = []
    lays = sorted({w["layout"] for w in train}, key=lambda l: RNG.random())
    for target in (10, 30, 100, 300):
        keep, n = set(), 0
        for l in lays:
            if n >= target:
                break
            keep.add(l)
            n += sum(w["layout"] == l and w["outcome"] == "n first" for w in train)
        sub_ = [w for w in train if w["layout"] in keep]
        if n < target:
            curve.append({"orders": target, "available": n})
            continue
        m2, b2, y2, q2 = fit(sub_, F, steps=STEPS, log=lambda m: None)
        sc = score(test, run_recall(OwnRouter(m2, F, sub_, b2, y2, q2), test))
        curve.append({"orders": target, "train_weighings": len(sub_), **sc})
        log(f"curve {target}: {sc}")
    res["p19_curve"] = curve
    OUT.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
