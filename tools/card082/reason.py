"""Card 082: a router and a network that reason over recalled tries (report only).

  WM_REL_ENCODER=runs/070/encoder.pt bin/prun python tools/card082/reason.py none [--seeds 79,80,81] [--arms 1]
  ... reason.py red                                    # one fold: red's door openings removed from memory

Per fold (card 079's setup): version 18's memory; each stored try of pick up, drop and toggle is a row (front, held,
view set; outcome counts). Inputs are relations only: tile vectors from the encoder, normalised over the action's
tiles, enter only through similarities <Phi_h z_a, Phi_h z_b> (8 heads); no tile vector passes forward. The router
scores every stored row against a query from the between-row relations and retrieves the top 64; their log weights bias
the reasoner's attention. The reasoner: 3 attention layers over the retrieved rows and the query, between-row relations
as attention biases; retrieved rows attend to each other, the query to them. One network for all three actions, no
action label. Trained by hiding 32 (front, held) combinations per step and predicting their rows from the rest.
Arms at seed 79 (WM_ARMS=1): "no router" (every row in context) and "no memory" (the query row alone).
Diagnostics of the gate: WM_STEPS (training steps), WM_MIX=1 (half the steps hide single rows, not combinations),
WM_QVIEW=empty (the table's queries with no tiles in view), WM_TAG (suffix of the output file).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
HUE = sys.argv[1]
sys.argv = ["run.py", "decoy", "--fold", HUE if HUE != "none" else "red", "--hold", "opening", "--online", "0",
            "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

T, R = RUN.T, RUN.R
VP, TK, SP = T.VP, T.TK, sys.modules["slot_planner"]
DEV = "cuda"
t00 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)

T.ENVS[1] = R.register()
T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
_memory = T.memory


def memory(tier, pool=None):
    D, stats = _memory(tier, pool)
    if HUE == "none":
        return D, stats
    f = D["ego0"][:, T.FRONT].astype(np.int64) // 5 * 5
    h = D["ego0"][:, T.HELD].astype(np.int64) // 5 * 5
    drop = (D["act"] == VP.TOG) & (f == T.KR.code(("door", HUE, 0))) & (h == T.KR.code(("key", HUE)))
    inter = np.isin(D["act"], T.INTER)
    keep_t = ~drop[inter]
    D = {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}
    return D, {**stats, "opening_tries_removed": int(drop.sum()), "rows": int(len(D["act"]))}


T.memory = memory
W, info = T.setup(1, log)
kd = W.kinds[VP.TOG]
app = lambda o: int(T.Kd.APP[T.KR.code(o)])
pl = T.S7.Plan047(W)
env = T.make(1)
env.reset(seed=T.SEED_TEST + 999)
codes = T.now_codes(env)
_, st, _, _ = pl.observe(None, T.PV.crop(T.Kd.APP[codes], codes))
ctx, empty = pl.ctx_id(st[0], st[1]), int(pl.facts[st[0]][VP.HELD])
P = torch.load(os.environ.get("WM_REL_ENCODER", "runs/070/encoder.pt"))["P"].double().numpy()
Z = W.S.arr

# ---------------------------------------------------------------- the table and version 18's answers
doors = {c: app(("door", c, 0)) for c in T.HUES}
keys = {c: app(("key", c)) for c in T.HUES}
NAME = {}
for h in range(len(Z)):                                # the evaluator's names, to build the table only
    try:
        NAME.setdefault(W.judge.name(h), h)
    except Exception:                                  # noqa: BLE001
        pass
fronts = [("door " + c, doors[c]) for c in T.HUES] + [(nm, NAME[nm]) for nm in ("wall", "floor") if nm in NAME]
helds = [("key " + c, keys[c]) for c in T.HUES] + [("nothing", empty)]
cells = [(fn, fu, hn, hh, fn.startswith("door") and hn.startswith("key") and fn[5:] == hn[4:])
         for fn, fu in fronts for hn, hh in helds]
v18 = [int(W.outcome(VP.TOG, fu, hh, ctx)[0]) for _, fu, _, hh, _ in cells]


SEEDS = [int(s) for s in os.environ.get("WM_SEEDS", "79,80,81").split(",")]
EXTRA_ARMS = os.environ.get("WM_ARMS", "1") == "1"
STEPS = int(os.environ.get("WM_STEPS", "3000"))
K, H, DH, WIDTH, LAYERS, HEADS = 64, 8, 16, 64, 3, 4
ACTS = [("pick up", VP.PICK), ("drop", VP.DROP), ("toggle", VP.TOG)]

# ---------------------------------------------------------------- the tables (evaluator's truth) and version 18
names = {**{f"door {c}": doors[c] for c in T.HUES}, **{f"key {c}": keys[c] for c in T.HUES},
         **{nm: NAME[nm] for nm in ("wall", "floor") if nm in NAME}, "nothing": empty}
tile_fronts = [n for n in names if n != "nothing"]
TABLES = {
    "toggle": [(fn, hn, fn.startswith("door") and hn.startswith("key") and fn[5:] == hn[4:])
               for fn in [f"door {c}" for c in T.HUES] + ["wall", "floor"] for hn in [f"key {c}" for c in T.HUES] + ["nothing"]],
    "pick up": [(fn, hn, fn.startswith("key") and hn == "nothing") for fn in tile_fronts for hn in ("nothing", "key red")],
    "drop": [(fn, hn, fn == "floor" and hn != "nothing") for fn in tile_fronts for hn in ("nothing", "key red", "key blue")],
}
V18 = {an: [int(W.outcome(a, names[f], names[h], ctx)[0]) for f, h, _ in TABLES[an]] for an, a in ACTS}
fold_cells = [i for i, (f, h, t) in enumerate(TABLES["toggle"]) if f == f"door {HUE}" and h == f"key {HUE}"]


# ---------------------------------------------------------------- rows
class Rows:
    """One action's stored tries as rows, plus the table's queries appended after them."""

    def __init__(self, kd, queries):
        keys = list(kd.keys) + [(names[f], names[h], ctx) for f, h, _ in queries]
        self.n = len(kd.keys)
        self.m = max(1, max(len(SP.SETS[k[2]]) for k in keys))
        self.front = torch.tensor([k[0] for k in keys], device=DEV)
        self.held = torch.tensor([k[1] for k in keys], device=DEV)
        view = np.zeros((len(keys), self.m), np.int64)
        vmask = np.zeros((len(keys), self.m), np.float32)
        for i, k in enumerate(keys):
            s = list(SP.SETS[k[2]]) if not (i >= self.n and os.environ.get("WM_QVIEW") == "empty") else []
            view[i, :len(s)] = s
            vmask[i, :len(s)] = 1
        self.view, self.vmask = torch.tensor(view, device=DEV), torch.tensor(vmask, device=DEV)
        cnt = np.zeros((len(keys), kd.counts.shape[1]), np.float32)
        cnt[:self.n] = kd.counts
        self.dist = torch.tensor(cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-9), device=DEV)
        self.lcnt = torch.tensor(np.log1p(cnt.sum(1)) / 10, device=DEV)
        self.combo = torch.tensor([hash(k[:2]) % (1 << 61) for k in keys], device=DEV)
        self.own = [kd.index.get(k) for k in keys[self.n:]]
        used = torch.unique(torch.cat([self.front, self.held, self.view[self.vmask > 0]]))
        zu = torch.tensor(Z, dtype=torch.float32, device=DEV)[used]
        self.mu, self.sd = zu.mean(0), zu.std(0) + 1e-6


class Net(nn.Module):
    def __init__(self, ncat=4):
        super().__init__()
        self.phi = nn.Linear(Z.shape[1], H * DH, bias=False)
        self.router = nn.Sequential(nn.Linear(5 * H, 32), nn.ReLU(), nn.Linear(32, 1))
        self.bias = nn.Sequential(nn.Linear(5 * H, 32), nn.ReLU(), nn.Linear(32, LAYERS * HEADS))
        self.emb = nn.Linear(5 * H + 2 + ncat + 2, WIDTH)
        self.ln1 = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in range(LAYERS))
        self.ln2 = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in range(LAYERS))
        self.qkv = nn.ModuleList(nn.Linear(WIDTH, 3 * WIDTH) for _ in range(LAYERS))
        self.o = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(LAYERS))
        self.ff = nn.ModuleList(nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))
                                for _ in range(LAYERS))
        self.out = nn.Linear(WIDTH, ncat)

    def proj(self, R):
        """Per row: projected front, held, view tiles (rows, slots, H, DH) and the view mask."""
        P = self.phi((torch.tensor(Z, dtype=torch.float32, device=DEV) - R.mu) / R.sd).view(-1, H, DH)
        return P[R.front], P[R.held], P[R.view], R.vmask

    @staticmethod
    def within(Pf, Ph, Pv, vm):
        s = lambda a, b: (a * b).sum(-1) / DH ** 0.5
        fh = s(Pf, Ph)
        vf, vh = s(Pv, Pf[:, None]), s(Pv, Ph[:, None])
        cnt = vm.sum(1, keepdim=True).clamp(min=1)[..., None]
        mean = lambda x: (x * vm[..., None]).sum(1) / cnt[:, 0]
        mx = lambda x: (x - 1e4 * (1 - vm[..., None])).max(1).values * (vm.sum(1, keepdim=True) > 0)
        return torch.cat([fh, mean(vf), mx(vf), mean(vh), mx(vh), vm.sum(1, keepdim=True) / 10], -1)

    @staticmethod
    def between(A, B):
        """Relations between every row of A (Bt, Ta) and every row of B (Bt, Tb), as (Bt, Ta, Tb, 5H): front-front,
        held-held, front-held, held-front, view-view (mean of the view's projections)."""
        (fa, ha, va), (fb, hb, vb) = A, B
        s = lambda a, b: torch.einsum("bihd,bjhd->bijh", a, b) / DH ** 0.5
        return torch.cat([s(fa, fb), s(ha, hb), s(fa, hb), s(ha, fb), s(va, vb)], -1)

    def reason(self, tok, rel, logw, allow, attn_out=False):
        """tok (B, T, F) inputs; rel (B, T, T, 5H); logw (B, T) router bias; allow (B, T, T) attention mask."""
        x = self.emb(tok)
        b = self.bias(rel).view(*rel.shape[:3], LAYERS, HEADS)
        Bn, Tn, _ = x.shape
        att = None
        for l in range(LAYERS):
            q, k, v = self.qkv[l](self.ln1[l](x)).view(Bn, Tn, 3, HEADS, WIDTH // HEADS).unbind(2)
            lg = torch.einsum("bihd,bjhd->bhij", q, k) / (WIDTH // HEADS) ** 0.5
            lg = lg + b[..., l, :].permute(0, 3, 1, 2) + logw[:, None, None, :]
            lg = lg.masked_fill(~allow[:, None], -1e9)
            a = torch.softmax(lg, -1)
            att = a
            x = x + self.o[l](torch.einsum("bhij,bjhd->bihd", a, v).reshape(Bn, Tn, WIDTH))
            x = x + self.ff[l](self.ln2[l](x))
        return (self.out(x), att) if attn_out else self.out(x)


def forward(net, R, qi, hidden, mode, shuffle=False, attn=False, gen=None):
    """Predict rows qi of R from the stored rows not in `hidden` (a bool mask over stored rows).
    mode: "router" (top K per query), "all" (every allowed row in one shared context), "none" (query alone)."""
    Pf, Ph, Pv, vm = net.proj(R)
    W_ = net.within(Pf, Ph, Pv, vm)
    Pvm = (Pv * vm[..., None, None]).sum(1) / vm.sum(1).clamp(min=1)[:, None, None]
    rowv = lambda idx: (Pf[idx], Ph[idx], Pvm[idx])
    feat = lambda idx, qflag: torch.cat([W_[idx], qflag[..., None],
                                         R.dist[idx] * (1 - qflag[..., None]), (R.lcnt[idx] * (1 - qflag))[..., None],
                                         (1 - qflag)[..., None]], -1)
    cand = torch.nonzero(~hidden).squeeze(1)
    Q = len(qi)
    if mode == "router":
        sc = net.router(net.between(tuple(t[None] for t in rowv(qi)), tuple(t[None] for t in rowv(cand))))[0, ..., 0]
        k = min(K, len(cand))
        top, ix = sc.topk(k, 1)
        logw = torch.log_softmax(top, 1)
        idx = torch.cat([cand[ix], qi[:, None]], 1)                    # (Q, K+1); the query last
        qflag = torch.zeros(Q, k + 1, device=DEV)
        qflag[:, -1] = 1
        lw = torch.cat([logw, torch.zeros(Q, 1, device=DEV)], 1)
    elif mode == "all":
        idx = torch.cat([cand, qi])[None]
        qflag = torch.cat([torch.zeros(len(cand), device=DEV), torch.ones(Q, device=DEV)])[None]
        lw = torch.zeros_like(qflag)
    else:
        idx, qflag, lw = qi[:, None], torch.ones(Q, 1, device=DEV), torch.zeros(Q, 1, device=DEV)
    tok = feat(idx, qflag)
    if shuffle:                                                         # outcomes permuted among the context rows
        for bi in range(tok.shape[0]):
            ctxi = torch.nonzero(qflag[bi] == 0).squeeze(1)
            perm = ctxi[torch.randperm(len(ctxi), generator=gen, device="cpu").to(DEV)]
            tok[bi, ctxi, -(R.dist.shape[1] + 2):] = tok[bi, perm, -(R.dist.shape[1] + 2):]
    rv = rowv(idx)
    rel = net.between(rv, rv)
    eye = torch.eye(idx.shape[1], dtype=torch.bool, device=DEV)[None]
    allow = (qflag[:, None, :] == 0) | eye                              # attend to context rows, and to itself
    allow = allow & ~((qflag[:, :, None] == 0) & (qflag[:, None, :] == 1))
    out = net.reason(tok, rel, lw, allow, attn_out=attn)
    logits, a = out if attn else (out, None)
    if mode == "all":
        sel = logits[0, len(cand):]
        return (sel, None) if attn else sel
    sel = logits[:, -1]
    if attn:
        return sel, (a[:, :, -1, :].mean(1), idx)
    return sel


LOSS = []


def train(mode, seed, RS):
    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = np.random.default_rng(seed)
    t0 = time.monotonic()
    for step in range(STEPS):
        R = RS[g.integers(len(RS))]
        stored = R.combo[:R.n]
        uc = torch.unique(stored)
        hid_c = uc[torch.tensor(g.choice(len(uc), min(32, len(uc) - 1), replace=False), device=DEV)]
        hidden = torch.isin(stored, hid_c)
        hrows = torch.nonzero(hidden).squeeze(1)
        qi = hrows[torch.tensor(g.choice(len(hrows), min(128, len(hrows)), replace=False), device=DEV)]
        if os.environ.get("WM_MIX") == "1" and g.random() < 0.5:     # single rows hidden: the rest of the combination stays
            qi = torch.tensor(g.choice(R.n, 128, replace=False), device=DEV)
            hidden = torch.zeros(R.n, dtype=torch.bool, device=DEV)
            hidden[qi] = True
        logp = torch.log_softmax(forward(net, R, qi, hidden, mode), -1)
        loss = -(R.dist[qi] * logp).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        LOSS.append(float(loss))
    return net, time.monotonic() - t0


@torch.no_grad()
def evaluate(net, mode, RS, shuffle=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    res = {}
    for (an, _), R in zip(ACTS, RS):
        qi = torch.arange(R.n, R.n + len(TABLES[an]), device=DEV)
        hidden = torch.zeros(R.n, dtype=torch.bool, device=DEV)
        p = torch.softmax(forward(net, R, qi, hidden, mode, shuffle=shuffle, gen=gen), -1).cpu().numpy()
        cats = []
        for j, own in enumerate(R.own):
            pj = p[j] if own is None else R.dist[own].cpu().numpy()          # own tries first (card 050)
            cats.append(int(pj.argmax()) if pj.max() >= 0.5 else 0)
        right = [bool((c != 0) == t) for c, (_, _, t) in zip(cats, TABLES[an])]
        res[an] = {"right": sum(right), "of": len(right), "cats": cats,
                   "wrong": [TABLES[an][i][:2] for i in range(len(right)) if not right[i]]}
        if an == "toggle":
            res[an]["fold_pair"] = [cats[i] for i in fold_cells]
            res[an]["fold_pair_p"] = [[round(float(x), 3) for x in p[i]] for i in fold_cells]
    return res


@torch.no_grad()
def loco(net, RS):
    """Each stored row predicted with its own (front, held) combination hidden: accuracy per action."""
    out = {}
    for (an, _), R in zip(ACTS, RS):
        ok = n = 0
        for c in torch.unique(R.combo[:R.n]):
            hidden = R.combo[:R.n] == c
            qi = torch.nonzero(hidden).squeeze(1)
            p = forward(net, R, qi, hidden, "router")
            ok += int((p.argmax(1) == R.dist[qi].argmax(1)).sum())
            n += len(qi)
        out[an] = round(ok / n, 4)
    return out


@torch.no_grad()
def probe(net, RS):
    """The left-out pair's query: is a stored opening among its retrieved rows, and how much attention goes to them."""
    R = RS[2]
    if not fold_cells:
        return None
    qi = torch.tensor([R.n + fold_cells[0]], device=DEV)
    _, (a, idx) = forward(net, R, qi, torch.zeros(R.n, dtype=torch.bool, device=DEV), "router", attn=True)
    opens = (R.dist[idx[0, :-1]][:, 1:].sum(1) > 0)
    return {"openings_retrieved": int(opens.sum()), "of": K,
            "attention_on_openings": round(float(a[0, :-1][opens].sum()), 3)}


RS = [Rows(W.kinds[a], TABLES[an]) for an, a in ACTS]
data = {an: {"rows": R.n, "combinations": int(len(torch.unique(R.combo[:R.n]))),
             "categories": [int(x) for x in np.nonzero(W.kinds[a].counts.sum(0))[0]]} for (an, a), R in zip(ACTS, RS)}
log(f"data {json.dumps(data)}")
log(f"query view: {[W.judge.name(h) for h in SP.SETS[ctx]]}")
v18r = {an: sum(bool((c != 0) == t) for c, (_, _, t) in zip(V18[an], TABLES[an])) for an, _ in ACTS}
log(f"v18 right {v18r}; fold pair {[V18['toggle'][i] for i in fold_cells]}")
runs = []
plan = [("router", s) for s in SEEDS] + ([("none", 79), ("all", 79)] if EXTRA_ARMS else [])
for mode, seed in plan:
    net, secs = train(mode, seed, RS)
    r = {"mode": mode, "seed": seed, "fit_seconds": round(secs, 1), "tables": evaluate(net, mode, RS),
         "loss_by_fifth": [round(float(np.mean(c)), 4) for c in np.array_split(LOSS[-STEPS:], 5)]}
    if mode == "router":
        r["shuffled"] = evaluate(net, mode, RS, shuffle=True, seed=seed)["toggle"]
        r["loco"] = loco(net, RS)
        r["probe"] = probe(net, RS)
    runs.append(r)
    log(json.dumps({"mode": mode, "seed": seed, "secs": r["fit_seconds"], "loss": r["loss_by_fifth"],
                    **{an: [t["right"], t["of"], t["wrong"]] for an, t in r["tables"].items()},
                    "fold_pair": r["tables"]["toggle"].get("fold_pair"),
                    "shuffled_fold_pair": r.get("shuffled", {}).get("fold_pair"), "loco": r.get("loco"),
                    "probe": r.get("probe")}))
res = {"note": "Card 082, tools/card082/reason.py", "fold": HUE, "data": data, "v18_right": v18r,
       "v18_fold_pair": [V18["toggle"][i] for i in fold_cells], "tables": TABLES, "runs": runs}
out = ROOT / "runs" / "082" / f"fold_{HUE}{os.environ.get('WM_TAG', '')}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1, default=str) + "\n")
log(f"wrote {out}")
