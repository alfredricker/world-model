"""Card 031: codes from pixels.

Each 8 x 8 x 3 tile becomes a tuple of discrete codes, one per codebook, from a small encoder trained on
the agent's own tiles and in-place changes (a door opening, the agent drawn in a doorway). The counted
model of card 028 is stated over the code tuples: each entry is keyed by the fewest codebooks that lose
no evidence, so tiles that behave alike share one, and outcomes are stated per codebook (unchanged,
taken from the other changed place, or set to a code). Card 029's subgoals read the table unchanged.

Purple keys and doors never occur in training (the renderer's object list is extended here, src/
unchanged); the test worlds use them.

Arms: 1 codes from the simulator's labels (object with state and agent; colour); 2 learned codes,
4 codebooks of 8; 3 exact tile names (card 029 unchanged); 4 one codebook of 64; 5 four codebooks of
8 without the pairs. Arms 2, 4 and 5 run ten encoder seeds.

Run: bin/prun python tools/card031/codes.py [--episodes 5000] [--layouts 500] [--heldout 1000]
     [--test-episodes 1000] [--seeds 10] [--out runs/031_codes.json]
"""
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# purple keys and doors, appended to the renderer's object list before any tile table is built
from worldmodel.envs import keydoor_render as KR  # noqa: E402

PURPLE_OBJECTS = [("key", "purple")] + [("door", "purple", st) for st in range(3)]
for _o in PURPLE_OBJECTS:
    if _o not in KR.OBJ_INDEX:
        KR.OBJ_INDEX[_o] = len(KR.OBJECTS)
        KR.OBJECTS.append(_o)
KR.N_CODES = len(KR.OBJECTS) * 5

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card029"))
import subgoals as G  # noqa: E402
from worldmodel.logic_conditions import evidence_rules, log_evidence  # noqa: E402

F = G.F
Kd, dl, ld, closer = F.Kd, F.dl, F.ld, F.closer
NV, NPL, W13, R = F.NV, F.NPL, F.W13, F.R
CENTRE, FRONT, HELD, SAME = F.CENTRE, F.FRONT, F.HELD, F.SAME
LEFT, RIGHT, FWD, PICK, DROP, TOG = range(6)
MOVES = F.MOVES
CHANGED, UNCHANGED, ENDED = F.CHANGED, F.UNCHANGED, F.ENDED
BEHIND = CENTRE + W13                        # after a step forward, where the agent stood
NEW = -1                                     # a piece far from every code of its codebook
UP = Kd.UP
TILES = Kd.TILES                             # (codes, 8, 8, 3), purple included
APP0 = Kd.APP.copy()                         # exact pixel identity (card 027): arm 3, and the encoder's data


def obj_of(c):
    return KR.OBJECTS[int(c) // 5]


def agent_of(c):
    return int(c) % 5 != 0


def label_of(c):
    """Evaluator's labels of a tile code: (object with state and agent, colour)."""
    o, ag = obj_of(c), agent_of(c)
    if o is None:
        obj, col = "floor", "none"
    elif o in ("wall", "goal"):
        obj, col = o, "none"
    elif o[0] == "door":
        obj, col = ("closed door" if o[2] == 0 else "shut door" if o[2] == 1 else "open door"), o[1]
    else:
        obj, col = o[0], o[1]
    if ag:
        obj = "agent in doorway" if obj == "open door" else f"agent on {obj}"
    return obj, col


def name_of_code(c):
    obj, col = label_of(c)
    if obj in ("ball",):
        return "switch on" if col == "yellow" else "switch off"
    if obj == "box":
        return "vase"
    return obj if col == "none" else f"{obj} {col}"


# the tiles an egocentric view can show: every object without the agent, and with the agent facing up
VIEW_CODES = np.array(sorted([i * 5 for i in range(len(KR.OBJECTS))] + [i * 5 + UP + 1 for i in range(len(KR.OBJECTS))]))


# ---------------------------------------------------------------- the agent's tiles and in-place pairs

def ego_codes(codes, st, chunk=50000):
    out = np.zeros((len(codes), NPL), np.int64)
    for s in range(0, len(codes), chunk):
        out[s:s + chunk] = Kd.ego(codes[s:s + chunk], st[s:s + chunk]).reshape(-1, NPL)
    return out


def tiles_and_pairs(worlds):
    """From the stored transitions of the training worlds: the distinct tiles seen (one code each, by
    pixel identity) and the distinct in-place pairs (before, after), each once. Declared data policy: a
    pair is taken at the places a pick up, drop or toggle changed (in front, held), and for forward, from
    the tile in front to the centre (the agent drawn on it) and from the centre to the place behind (the
    agent gone); it is left out when the thing moved: its "before" tile shows after at the other changed
    place, or its "after" tile showed there before."""
    app = APP0
    seen, pairs = set(), set()
    for w in worlds:
        e0, e1, act, vc = w["ego0"], w["ego1"], w["act"], w["view_changed"]
        seen.update(np.unique(app[e0]).tolist())
        seen.update(np.unique(app[e1]).tolist())
        inter = np.isin(act, (PICK, DROP, TOG))
        fwd = (act == FWD) & vc & ~w["term1"]
        for mask, (p, bp), (o, bo) in ((inter, (FRONT, FRONT), (HELD, HELD)), (fwd, (FRONT, CENTRE), (CENTRE, BEHIND))):
            # place p before -> place bp after; the other changed place: o before -> bo after
            b_p, a_p = app[e0[mask, p]], app[e1[mask, bp]]
            b_o, a_o = app[e0[mask, o]], app[e1[mask, bo]]
            for (x0, x1, y0, y1) in ((b_p, a_p, b_o, a_o), (b_o, a_o, b_p, a_p)):
                keep = (x0 != x1) & (x0 != y1) & (x1 != y0)
                pairs.update(zip(x0[keep].tolist(), x1[keep].tolist()))
    rep = {}
    for c in range(len(app) - 1, -1, -1):
        rep[int(app[c])] = c
    seen = sorted(seen)
    return [rep[u] for u in seen], sorted((rep[a], rep[b]) for a, b in pairs)


# ---------------------------------------------------------------- the encoder

def train_codes(tile_codes, pair_codes, K=4, M=8, dim=8, pair_w=0.1, seed=0, updates=5000, dev="cuda",
                pair_rule="adaptive", reseed=True, dl_w=0.0, tau=0.1):
    """Sliced vector quantisation: an encoder maps a tile to K unit-length pieces, each replaced by the
    nearest of its codebook's M unit-length vectors; a decoder rebuilds the pixels from them. Loss:
    rebuild (mean squared error) + codebook and commitment terms + pair_w x the pair term. Pair term, per
    pair, from the distance between the two tiles' pieces in each codebook (before quantisation):
    "adaptive" (as built, 2002_02886's rule): the codebooks whose distance is below the midpoint of the
    pair's smallest and largest are pulled together, the others left free; "sum" (as declared): the sum
    of all four. reseed (as built): every 250 updates, while two training tiles share every code, an
    unused code of each codebook is moved onto one of those tiles' pieces. dl_w (card 032): weight of
    the code length, the sum over codebooks of the entropy of code use over the training tiles, from
    soft assignments (softmax of minus the squared distance / tau). Returns code -> tuple (NEW where a
    piece is far from every used code), and a report."""
    import torch
    import torch.nn as nn
    fn = torch.nn.functional
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False    # a seed gives one result
    torch.manual_seed(seed)
    x_all = torch.as_tensor(TILES / 255.0, dtype=torch.float32, device=dev).permute(0, 3, 1, 2)   # codes, 3, 8, 8
    enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                        nn.ReLU(), nn.Flatten(), nn.Linear(32 * 16, K * dim)).to(dev)
    dec = nn.Sequential(nn.Linear(K * dim, 32 * 16), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)),
                        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()).to(dev)
    books = nn.Parameter(torch.randn(K, M, dim, device=dev))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [books], lr=1e-3)
    tc = torch.as_tensor(tile_codes, device=dev)
    pa = torch.as_tensor([p[0] for p in pair_codes], device=dev, dtype=torch.long)
    pb = torch.as_tensor([p[1] for p in pair_codes], device=dev, dtype=torch.long)

    def pieces(x):
        return fn.normalize(enc(x).reshape(len(x), K, dim), dim=-1)

    def quant(z):
        e = fn.normalize(books, dim=-1)                                   # K, M, dim
        d = torch.cdist(z.transpose(0, 1), e)                              # K, B, M
        idx = d.argmin(2).T                                                # B, K
        zq = e[torch.arange(K, device=dev)[None], idx]                     # B, K, dim
        return zq, idx

    t0 = time.monotonic()
    for step in range(updates):
        x = x_all[tc]
        z = pieces(x)
        zq, _ = quant(z)
        st = z + (zq - z).detach()
        rebuild = fn.mse_loss(dec(st.reshape(len(x), -1)), x)
        vq = fn.mse_loss(zq, z.detach()) + 0.25 * fn.mse_loss(z, zq.detach())
        if len(pa) and pair_w > 0:
            za, zb = pieces(x_all[pa]), pieces(x_all[pb])
            d = (za - zb).norm(dim=-1)                                     # pairs, K
            if pair_rule == "adaptive":
                mid = (d.amax(1, keepdim=True) + d.amin(1, keepdim=True)).detach() / 2
                pair = (d * (d < mid)).sum(1).mean()
            else:
                pair = d.sum(1).mean()
        else:
            pair = torch.zeros((), device=dev)
        loss = rebuild + vq + pair_w * pair
        if dl_w > 0:
            e = fn.normalize(books, dim=-1)
            d2 = torch.cdist(z.transpose(0, 1), e) ** 2                   # K, tiles, M
            q = torch.softmax(-d2 / tau, 2).mean(1)                        # K, M: share of tiles per code
            length = -(q * torch.log(q + 1e-12)).sum()
            loss = loss + dl_w * length
        opt.zero_grad()
        loss.backward()
        opt.step()
        if reseed and step % 250 == 249 and step < updates - 500:
            with torch.no_grad():
                zt = pieces(x_all[tc])
                _, it = quant(zt)
                tup = [tuple(r) for r in it.tolist()]
                dup = [i for i, t in enumerate(tup) if tup.count(t) > 1]
                if dup:
                    for k in range(K):
                        unused = sorted(set(range(M)) - set(it[:, k].tolist()))
                        if unused:
                            i = dup[int(torch.randint(len(dup), (1,)))]
                            books[k, unused[0]] = zt[i, k]
    with torch.no_grad():
        e = fn.normalize(books, dim=-1)
        z_tr = pieces(x_all[tc])
        _, idx_tr = quant(z_tr)
        used = [sorted(set(idx_tr[:, k].tolist())) for k in range(K)]
        # "new": farther from the nearest used code than half its distance to the nearest other used code
        z_all = pieces(x_all)
        out = np.full((len(x_all), K), NEW, np.int64)
        for k in range(K):
            ek = e[k, used[k]]                                             # U, dim
            d = torch.cdist(z_all[:, k], ek)                               # codes, U
            near, j = d.min(1)
            if len(used[k]) > 1:
                dd = torch.cdist(ek, ek) + torch.eye(len(used[k]), device=dev) * 1e9
                half = dd.min(1).values[j] / 2
            else:
                half = torch.full_like(near, 1.0)
            ok = (near <= half).cpu().numpy()
            out[ok, k] = np.array(used[k])[j.cpu().numpy()[ok]]
        hard = 0.0
        for k in range(K):
            cnt = np.bincount(idx_tr[:, k].cpu().numpy(), minlength=M) / len(idx_tr)
            hard += float(-(cnt[cnt > 0] * np.log(cnt[cnt > 0])).sum())
        rep = {"rebuild_mse": round(float(rebuild), 6), "pair_term": round(float(pair), 4),
               "codes_used": [len(u) for u in used], "code_length_nats": round(hard, 3),
               "seconds": round(time.monotonic() - t0, 1)}
    return out, rep


def oracle_codes():
    """Arm 1: two codebooks from the simulator's labels (object with its state and the agent; colour)."""
    labels = [label_of(c) for c in range(len(TILES))]
    objs = sorted({o for o, _ in labels})
    cols = sorted({c for _, c in labels})
    return np.array([[objs.index(o), cols.index(c)] for o, c in labels], np.int64)


class Tuples:
    """Code tuples interned as small integers (the model's arrays are uint8; 255 is SAME)."""

    def __init__(self, code_tuples):
        self.tup, self.id = [], {}
        self.lut = np.array([self.intern(tuple(int(v) for v in t)) for t in code_tuples], np.uint8)

    def intern(self, t):
        i = self.id.get(t)
        if i is None:
            i = self.id[t] = len(self.tup)
            if i >= SAME:
                raise RuntimeError("more than 254 code tuples")
            self.tup.append(t)
        return i

    def codes(self, i):
        return self.tup[int(i)]


# ---------------------------------------------------------------- entries over codes

U, C = ("U",), ("C",)                        # unchanged; taken from the other changed place before


def canon(a, b, o):
    return U if a == b else C if a == o else ("S", a)


class Kind:
    """The tries of one kind of entry, merged into distinct types. keys: the tile ids the entry is keyed
    on (the tile in front; for drop also the held tile). For each outcome place p: the tile before
    (b), after (a) and at the other changed place before (o); bkey[p] / okey[p]: which key tile b / o
    is (None: not a key tile). cat: for moves, the outcome (changed, unchanged, ended)."""

    def __init__(self, name, nkeys, bkey, okey, keys, b, a, o, w, rows=None, cat=None):
        self.name, self.nkeys, self.bkey, self.okey = name, nkeys, bkey, okey
        cols = [keys] + ([cat[:, None]] if cat is not None else [b, a, o])
        arr = np.concatenate([np.asarray(c, np.int64).reshape(len(w), -1) for c in cols], 1)
        uniq, inv = np.unique(arr, axis=0, return_inverse=True)
        inv = inv.ravel()
        self.keys = uniq[:, :nkeys]
        self.w = np.bincount(inv, w, len(uniq))
        self.moves = cat is not None
        if self.moves:
            self.cat = uniq[:, nkeys]
        else:
            n = b.shape[1]
            self.b, self.a, self.o = uniq[:, nkeys:nkeys + n], uniq[:, nkeys + n:nkeys + 2 * n], uniq[:, nkeys + 2 * n:]
        self.rows = None
        if rows is not None:
            order = np.argsort(inv, kind="stable")
            bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
            self.rows = [rows[order[bounds[t]:bounds[t + 1]]] for t in range(len(uniq))]


def describe(kind, tidx, T):
    """Categories of a group's tries: split by the places that changed; then, per changed place and
    codebook, the first of unchanged, taken from the other place, set to a code that holds for all of
    them (where none does, split by each try's own). Returns {category: [type indices]}."""
    out = {}
    by_mask = {}
    for t in tidx:
        mask = tuple(bool(kind.a[t, p] != kind.b[t, p]) for p in range(kind.b.shape[1]))
        by_mask.setdefault(mask, []).append(t)
    K = len(T.codes(0))
    for mask, ts in by_mask.items():
        groups = [((), ts)]
        for p in [p for p in range(len(mask)) if mask[p]]:
            for k in range(K):
                nxt = []
                for desc, g in groups:
                    A = [T.codes(kind.a[t, p])[k] for t in g]
                    B = [T.codes(kind.b[t, p])[k] for t in g]
                    O = [T.codes(kind.o[t, p])[k] for t in g]
                    if A == B:
                        nxt.append((desc + (U,), g))
                    elif A == O:
                        nxt.append((desc + (C,), g))
                    elif len(set(A)) == 1:
                        nxt.append((desc + (("S", A[0]),), g))
                    else:
                        split = {}
                        for t, x, y, z in zip(g, A, B, O):
                            split.setdefault(canon(x, y, z), []).append(t)
                        nxt += [(desc + (opt,), sub) for opt, sub in split.items()]
                groups = nxt
        for desc, g in groups:
            out[(mask, desc)] = out.get((mask, desc), []) + g
    return out


def n_categories(kind, T):
    """Categories under each try's own description (an upper bound on any group's), for the prior."""
    if kind.moves:
        return 3
    cats = set()
    K = len(T.codes(0))
    for t in range(len(kind.w)):
        mask = tuple(bool(kind.a[t, p] != kind.b[t, p]) for p in range(kind.b.shape[1]))
        cats.add((mask, tuple(canon(T.codes(kind.a[t, p])[k], T.codes(kind.b[t, p])[k], T.codes(kind.o[t, p])[k])
                              for p in range(len(mask)) if mask[p] for k in range(K))))
    return max(2, len(cats))


def rule_data(V0, y, w):
    """card 028's rule data, with each try's row in it."""
    held = V0[:, HELD]
    pres = np.zeros((len(V0), 256), bool)
    pres[np.arange(len(V0))[:, None], V0[:, :NV]] = True
    hv = np.unique(held)
    pv = np.flatnonzero(pres.any(0))
    X = np.concatenate([held[:, None] == hv[None], pres[:, pv]], 1)
    uniq, inv = np.unique(X, axis=0, return_inverse=True)
    inv = inv.ravel()
    d = type("Dat", (), {})()
    d.X = uniq
    d.n = np.bincount(inv, w, len(uniq))
    d.k = np.bincount(inv, w * y, len(uniq))
    d.atoms = [(0, int(u)) for u in hv] + [(1, int(u)) for u in pv]
    return d, inv


def logev(counts, k):
    counts = list(counts) + [0.0] * (k - len(counts))
    return Kd._logev(counts)


def subsets(K):
    import itertools
    return [s for n in range(K + 1) for s in itertools.combinations(range(K), n)]


class Pooler:
    """Chooses, for one kind, the codebooks S its entries are keyed on, by evidence (appendix A)."""

    def __init__(self, kind, T, V0, w, cache):
        self.kind, self.T, self.V0, self.w, self.cache = kind, T, V0, w, cache
        self.K = len(T.codes(0))
        self.ncat = n_categories(kind, T)

    def groups(self, S):
        """S: one tuple of codebooks per key tile. Groups of types by the keys' codes in S."""
        g = {}
        for t in range(len(self.kind.w)):
            key = tuple(tuple(self.T.codes(self.kind.keys[t, j])[k] for k in S[j]) for j in range(self.kind.nkeys))
            g.setdefault(key, []).append(t)
        return g

    def score_group(self, tidx):
        kind = self.kind
        if kind.moves:
            cnt = np.bincount(kind.cat[tidx], kind.w[tidx], 3)
            return logev(cnt, 3), None
        ck = (kind.name, frozenset(tuple(kind.keys[t]) for t in tidx))
        hit = self.cache.get(ck)
        if hit is not None:
            return hit
        cats = describe(kind, tidx, self.T)
        wt = {c: float(kind.w[ts].sum()) for c, ts in cats.items()}
        ev = logev(sorted(wt.values(), reverse=True), self.ncat)
        info = {"cats": cats, "weights": wt, "rules": {}}
        if len(cats) > 1:
            nothing = [c for c in cats if not any(c[0])]
            default = nothing[0] if nothing else max(wt, key=wt.get)
            info["default"] = default
            rows = np.concatenate([kind.rows[t] for t in tidx])
            lab = np.concatenate([np.full(len(kind.rows[t]), i) for i, t in enumerate(tidx)])
            cat_of = {}
            for c, ts in cats.items():
                for t in ts:
                    cat_of[t] = c
            order = list(cats)
            y_all = np.array([order.index(cat_of[tidx[i]]) for i in lab])
            cell = np.zeros((len(rows), 0), np.int64)
            cost = 0.0
            for c in cats:
                if c == default:
                    continue
                y = (y_all == order.index(c)).astype(np.float64)
                d, inv = rule_data(self.V0[rows], y, self.w[rows])
                rules, best, trace = evidence_rules(d)
                base = log_evidence(d, [])
                cost += sum(np.log(2) + len(r) * (np.log(2) + np.log(len(d.atoms))) for r in rules)
                fired = np.full(len(rows), -1)
                for j, r in enumerate(rules):
                    fired[(fired < 0) & d.X[inv][:, r].all(1)] = j
                cell = np.concatenate([cell, fired[:, None]], 1)
                info["rules"][c] = (d, rules, F.rule_rates(d, rules), round(float(best - base), 2))
            # every set of tries the rules single out is scored as a group is (Dirichlet, all categories)
            _, ci = np.unique(cell, axis=0, return_inverse=True)
            ci = ci.ravel()
            ev = -cost
            for j in range(ci.max() + 1):
                m = ci == j
                ev += logev(np.bincount(y_all[m], self.w[rows][m], len(order)), self.ncat)
            info["cells"] = int(ci.max() + 1)
        self.cache[ck] = (ev, info)
        return ev, info

    def cands(self):
        """Splits a group can take: a non-empty set of one key tile's codebooks (smallest first)."""
        return [(j, S) for S in subsets(self.K)[1:] for j in range(self.kind.nkeys)]

    def split(self, tidx, j, S):
        g = {}
        for t in tidx:
            c = self.T.codes(self.kind.keys[t, j])
            g.setdefault(tuple(c[k] for k in S), []).append(t)
        return g

    def tree(self, tidx=None):
        """As built: each group keyed on the fewest codebooks that lose no evidence. A group is split by
        the set of codebooks (of one key tile) that raises the evidence most, if any does, and each part
        is treated the same way. Returns ("leaf", types) or ("split", j, S, {codes: subtree})."""
        if tidx is None:
            tidx = list(range(len(self.kind.w)))
        best, arg = self.score_group(tidx)[0], None
        for j, S in self.cands():
            g = self.split(tidx, j, S)
            if len(g) < 2:
                continue
            sc = sum(self.score_group(ts)[0] for ts in g.values())
            if sc > best + 1e-9:
                best, arg = sc, (j, S, g)
        if arg is None:
            return ("leaf", tidx)
        j, S, g = arg
        return ("split", j, S, {k: self.tree(ts) for k, ts in g.items()})

    def choose(self):
        nk = self.kind.nkeys
        cands = subsets(self.K)
        if nk == 2:
            import itertools
            cands = sorted(itertools.product(cands, cands), key=lambda s: len(s[0]) + len(s[1]))
        else:
            cands = [(s,) for s in cands]
        scored = []
        for S in cands:
            g = self.groups(S)
            scored.append((sum(self.score_group(ts)[0] for ts in g.values()), S))
        best = max(v for v, _ in scored)
        S = next(s for v, s in scored if v >= best - 1e-9)          # ties: the smallest, then codebook order
        ties = [s for v, s in scored if v >= best - 1e-9 and s != S and sum(map(len, s)) == sum(map(len, S))]
        return S, best, scored, ties


def _key_of(T, S, ids):
    """A tile tuple's group key under S, or None if one of its codes there is new."""
    key = []
    for j, u in enumerate(ids):
        c = T.codes(u)
        v = tuple(c[k] for k in S[j])
        if NEW in v:
            return None
        key.append(v)
    return tuple(key)


def concretise(kind, T, cat, ts, key_ids):
    """A category's outcome for one key tile (tuple of ids at the outcome places, SAME where
    unchanged), or None for nothing."""
    mask, desc = cat
    if not any(mask):
        return None
    K = len(T.codes(0))
    w = kind.w[ts]

    def rep(arr, p):                        # the category's most common tile at a place that is not a key
        vals = arr[ts, p]
        u, inv = np.unique(vals, return_inverse=True)
        return int(u[np.bincount(inv.ravel(), w).argmax()])

    out, i = [], 0
    for p in range(len(mask)):
        bid = key_ids[kind.bkey[p]] if kind.bkey[p] is not None else rep(kind.b, p)
        if not mask[p]:
            out.append(SAME)
            continue
        oid = key_ids[kind.okey[p]] if kind.okey[p] is not None else rep(kind.o, p)
        bc, oc = T.codes(bid), T.codes(oid)
        new = []
        for k in range(K):
            opt = desc[i]
            i += 1
            new.append(bc[k] if opt == U else oc[k] if opt == C else opt[1])
        aid = T.intern(tuple(new))
        out.append(SAME if aid == bid else aid)
    return tuple(out)


def leaves(node):
    if node[0] == "leaf":
        return [node[1]]
    return [x for sub in node[3].values() for x in leaves(sub)]


def lookup(node, T, key_ids):
    """A tile's leaf in the tree: None where one of its codes on the way is new or unseen."""
    while node[0] == "split":
        _, j, S, ch = node
        c = T.codes(key_ids[j])
        v = tuple(c[k] for k in S)
        if NEW in v or v not in ch:
            return None
        node = ch[v]
    return node[1]


def tree_report(node, kind):
    if node[0] == "leaf":
        return sorted({" + ".join(F.nm(u) for u in kind.keys[t]) for t in node[1]})
    _, j, S, ch = node
    return {f"{'front' if j == 0 else 'held'} codebooks {list(S)} = {list(k)}": tree_report(sub, kind) for k, sub in ch.items()}


def pooled_model(m, V0, V1, act, w, out, T, log, mode="tree"):
    """Replaces the per-tile entries of card 028's model m by entries keyed on codebooks (appendix A),
    written back for every tile tuple the view can show and every tuple an outcome creates. mode
    "tree" (as built): codebooks chosen per group; "global" (as declared): one set per action."""
    rep = {}
    cache = {}
    assert [int(q) for q in m.change_places] == [FRONT, HELD], m.change_places
    kinds = {}
    for a in MOVES:
        sel = np.flatnonzero(act == a)
        kinds[dl.ACTIONS[a]] = Kind(dl.ACTIONS[a], 1, None, None, V0[sel, FRONT], None, None, None, w[sel],
                                    cat=out[sel])
    fw = np.flatnonzero((act == FWD) & (out != UNCHANGED))
    kinds["draw"] = Kind("draw", 1, [0], [None], V0[fw, FRONT], V0[fw, FRONT][:, None], V1[fw, CENTRE][:, None],
                         V0[fw, CENTRE][:, None], w[fw], rows=fw)
    kinds["undraw"] = Kind("undraw", 1, [0], [None], V0[fw, CENTRE], V0[fw, CENTRE][:, None], V1[fw, BEHIND][:, None],
                           V0[fw, FRONT][:, None], w[fw], rows=fw)
    for a in (PICK, TOG, DROP):
        sel = np.flatnonzero(act == a)
        b = V0[sel][:, [FRONT, HELD]]
        keys = b if a == DROP else b[:, :1]
        kinds[dl.ACTIONS[a]] = Kind(dl.ACTIONS[a], 2 if a == DROP else 1, [0, 1] if a == DROP else [0, None],
                                    [1, 0] if a == DROP else [None, 0], keys, b, V1[sel][:, [FRONT, HELD]],
                                    b[:, ::-1], w[sel], rows=sel)
    chosen = {}
    for name, kind in kinds.items():
        t0 = time.monotonic()
        pl = Pooler(kind, T, V0, w, cache)
        if mode == "tree":
            node = pl.tree()
            best = sum(pl.score_group(ts)[0] for ts in leaves(node))
            chosen[name] = (pl, node)
            rep[name] = {"tree": tree_report(node, kind), "log_evidence": round(float(best), 2),
                         "one_group": round(float(pl.score_group(list(range(len(kind.w))))[0]), 2),
                         "seconds": round(time.monotonic() - t0, 2)}
            continue
        S, best, scored, ties = pl.choose()
        g = pl.groups(S)
        node = ("flat", S, g)
        chosen[name] = (pl, node)
        top = sorted(scored, key=lambda x: -x[0])[:4]
        rep[name] = {"codebooks": [list(s) for s in S], "log_evidence": round(float(best), 2),
                     "next_best": [[[list(s) for s in S2], round(float(v), 2)] for v, S2 in top[1:]],
                     "equal_evidence_same_size": [[list(s) for s in S2] for S2 in ties],
                     "groups": [sorted({" + ".join(F.nm(u) for u in kind.keys[t]) for t in ts}) for ts in g.values()],
                     "seconds": round(time.monotonic() - t0, 2)}
    # write-back
    universe = set(int(v) for v in T.lut[VIEW_CODES])
    done = set()
    m.move_out = {a: np.full(256, UNCHANGED, np.int64) for a in MOVES}
    draw, undraw = np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8)
    table = {}
    group_entry = {}

    def entry_for(name, key_ids):
        pl, node = chosen[name]
        if node[0] == "flat":
            _, S, g = node
            k = _key_of(T, S, key_ids)
            ts = None if k is None else g.get(k)
        else:
            ts = lookup(node, T, key_ids)
        if ts is None:
            return None, None
        if pl.kind.moves:
            return int(np.bincount(pl.kind.cat[ts], pl.kind.w[ts], 3).argmax()), None
        return ts, pl.score_group(ts)[1]

    for sweep in range(6):
        todo = set(range(len(T.tup))) - done
        if not todo:
            break
        for u in sorted(todo):
            for a in MOVES:
                o, _ = entry_for(dl.ACTIONS[a], (u,))
                if o is not None:
                    m.move_out[a][u] = o
            for name, arr in (("draw", draw), ("undraw", undraw)):
                ts, info = entry_for(name, (u,))
                if ts is None:
                    continue
                kind = chosen[name][0].kind
                c = max(info["cats"], key=lambda c: info["weights"][c])
                o = concretise(kind, T, c, info["cats"][c], (u,))
                arr[u] = u if o is None or o[0] == SAME else o[0]
        pairs = [(u, None) for u in sorted(todo)] + [(u, h) for u in range(len(T.tup)) for h in range(len(T.tup))
                                                       if u in todo or h in todo]
        for u, h in pairs:
            for a in ((PICK, TOG) if h is None else (DROP,)):
                name = dl.ACTIONS[a]
                key_ids = (u,) if h is None else (u, h)
                ts, info = entry_for(name, key_ids)
                if ts is None:
                    continue
                kind = chosen[name][0].kind
                e = F.Entry()
                cats = info["cats"]
                conc = {c: concretise(kind, T, c, cats[c], key_ids) for c in cats}
                if len(cats) == 1:
                    e.single = next(iter(conc.values()))
                    if e.single is None:
                        continue
                else:
                    wt = info["weights"]
                    e.default = conc[info["default"]]
                    e.optimistic = next(conc[c] for c in sorted(cats, key=lambda c: -wt[c]) if conc[c] is not None)
                    e.multi = []
                    for c, (d, rules, rr, gain) in info["rules"].items():
                        rl, dr, _, _ = rr
                        e.multi.append((conc[c], [(conds, rate) for conds, rate, _, _ in rl], dr))
                table[(a, u, h) if a == DROP else (a, u)] = e
        done.update(todo)
    new_ids = sorted(set(range(len(T.tup))) - universe)
    ident = np.arange(256)
    for x, y in zip(ident[undraw != ident], undraw[undraw != ident]):
        if draw[y] == y:
            draw[y] = x                                            # a pair seen only one way round, as card 028
    for x, y in zip(ident[draw != ident], draw[draw != ident]):
        if undraw[y] == y:
            undraw[y] = x
    m.draw, m.undraw = draw, undraw
    qb = np.flatnonzero(m.src[FWD] == CENTRE)
    m.trans[FWD][CENTRE] = draw
    if len(qb):
        m.trans[FWD][qb[0]] = undraw
    m.free = (m.move_out[FWD] == CHANGED).tolist()
    m.standable = m.move_out[FWD] != UNCHANGED
    m.table = table
    rep["tuples_created_by_outcomes"] = [F.nm(u) for u in new_ids]
    rep["rules"] = []
    for name in ("pickup", "toggle", "drop"):
        pl, node = chosen[name]
        for ts in (node[2].values() if node[0] == "flat" else leaves(node)):
            info = pl.score_group(ts)[1]
            for c, (d, rules, rr, gain) in info["rules"].items():
                rl, dr, dn, dk = rr
                rep["rules"].append({
                    "kind": name, "group": sorted({" + ".join(F.nm(u) for u in pl.kind.keys[t]) for t in ts}),
                    "outcome": describe_cat(c, T),
                    "rules": [{"if": [F.atom_name(x) for x in conds], "rate": round(rate, 4), "tries_weighted": round(n, 1)}
                              for conds, rate, n, kk in rl],
                    "otherwise": {"rate": round(dr, 4), "tries_weighted": round(dn, 1)}, "evidence_gain": gain})
    return rep


def describe_cat(c, T):
    mask, desc = c
    if not any(mask):
        return "nothing"
    K = len(T.codes(0))
    out, i = [], 0
    for p, ch in zip(("front", "held"), mask):
        if not ch:
            continue
        parts = []
        for k in range(K):
            o = desc[i]
            i += 1
            parts.append("=" if o == U else "other" if o == C else str(o[1]))
        out.append(f"{p}: " + ",".join(parts))
    return "; ".join(out)


# ---------------------------------------------------------------- the purple test worlds (src/ unchanged)

_MAKE, _START = ld.make_layout, ld.start_state
TESTS = {"a": ("key", 1), "b": ("switch", 0), "c": ("key", 0)}          # world, purple door open at the start
EXCLUDED = "toggle the closed purple door holding the purple key (excluded)"


def purple_layout(size, rule, rng):
    return dataclasses.replace(_MAKE(size, rule, rng), door_colour="purple")      # the matching key is purple too


def open_start(lay, carry=0, switch_on=0):
    return (*lay.start, carry, None if carry == 1 else lay.key, None if carry == 2 else lay.distractor, 1, switch_on, 0)


def patch(purple=False, door_open=False):
    """Collection, acting and the shortest route read ld.make_layout and ld.start_state at call time;
    pools forked after this see the patch."""
    ld.make_layout = purple_layout if purple else _MAKE
    ld.start_state = open_start if door_open else _START


def cases(layouts, tr, world):
    """Section 4's events on test transitions, from the simulator's states (evaluator)."""
    s0, s1 = tr["s0"].astype(np.int64), tr["s1"].astype(np.int64)
    act = tr["act"].astype(np.int64)
    dv = np.array(ld.DIR_VEC)
    fx, fy = s0[:, 0] + dv[s0[:, 2], 0], s0[:, 1] + dv[s0[:, 2], 1]
    door = np.array([layouts[e].door for e in tr["ep"]])
    at_door = (fx == door[:, 0]) & (fy == door[:, 1])
    in_door = (s0[:, 0] == door[:, 0]) & (s0[:, 1] == door[:, 1])
    at_key = (fx == s0[:, 4]) & (fy == s0[:, 5])
    opened, closed = s0[:, 8] == 1, s0[:, 8] == 0
    moved = (s1[:, :2] != s0[:, :2]).any(1)
    c = {"pick up the purple key": (act == PICK) & (s0[:, 3] == 0) & (s1[:, 3] == 1),
         "drop the purple key": (act == DROP) & (s0[:, 3] == 1) & (s1[:, 3] == 0),
         "forward into the purple key (blocked)": (act == FWD) & at_key,
         "forward into the closed purple door (blocked)": (act == FWD) & at_door & closed,
         "forward onto the open purple door": (act == FWD) & at_door & opened,
         "toggle the open purple door (closes)": (act == TOG) & at_door & opened}
    tc = (act == TOG) & at_door & closed
    if world == "switch":
        c["toggle the closed purple door, switch on (opens)"] = tc & (s0[:, 9] == 1)
        c["toggle the closed purple door, switch off (stays)"] = tc & (s0[:, 9] == 0)
    else:
        c["toggle the closed purple door without the purple key (stays)"] = tc & (s0[:, 3] != 1)
        c[EXCLUDED] = tc & (s0[:, 3] == 1)
    extra = {"forward out of the purple doorway (reported only)": (act == FWD) & in_door & moved,
             "turn in the purple doorway (reported only)": np.isin(act, (LEFT, RIGHT)) & in_door}
    return c, extra


# ---------------------------------------------------------------- evaluation

def _eval_job(job):
    V0, V1, act, term1 = job
    th = F.Think([-1], [-1])
    ok = np.zeros(len(act), bool)
    errs = Counter()
    for i in range(len(act)):
        _, st, _, _ = th.observe(None, V0[i])
        nxt = th.step(st, int(act[i]))
        pv = th.view(nxt)
        ok[i] = bool((pv == V1[i]).all()) and nxt[2] == bool(term1[i])
        if not ok[i]:
            bad = np.flatnonzero(pv != V1[i])
            errs[(dl.ACTIONS[act[i]], F.nm(V0[i, FRONT]), F.nm(V0[i, HELD]),
                  tuple((int(q), F.nm(pv[q]), F.nm(V1[i, q])) for q in bad[:3]), nxt[2], bool(term1[i]))] += 1
        if len(th.facts) > 20000:
            th = F.Think([-1], [-1])
    return ok, errs


def eval_rows(pool, V0, V1, act, term1):
    """Card 028's check (every view tile, the held tile, and whether the episode ends), in parallel."""
    if len(act) == 0:
        return np.zeros(0, bool), Counter()
    chunks = [c for c in np.array_split(np.arange(len(act)), 40) if len(c)]
    parts = pool.map(_eval_job, [(V0[c], V1[c], act[c], term1[c]) for c in chunks])
    errs = Counter()
    for _, e in parts:
        errs.update(e)
    return np.concatenate([p[0] for p in parts]), errs


def per_action(ok, act):
    out = {}
    for a in range(6):
        m = act == a
        if m.any():
            out[dl.ACTIONS[a]] = {"rows": int(m.sum()), "exact_share": round(float(ok[m].mean()), 6)}
    return out


def names_for(T):
    """Evaluator names of tuple ids (reports only): every renderer tile the view can show; tuples no
    tile has are named by their codes."""
    names = {}
    for c in VIEW_CODES:
        names.setdefault(int(T.lut[c]), []).append(name_of_code(c))
    out = {u: "|".join(sorted(set(v), key=v.index)) for u, v in names.items()}
    for u in range(len(T.tup)):
        out.setdefault(u, "tuple " + ",".join("new" if x == NEW else str(x) for x in T.tup[u]))
    return out


def door_rules(world, T):
    """The written-back rules for opening each closed door, in card 029's check."""
    rules = []
    for c in ("red", "green", "blue"):
        u = int(T.lut[KR.code(("door", c, 0))])
        e = F.M.table.get((TOG, u))
        if e is None or e.multi is None:
            continue
        for o, rl, dr in e.multi:
            rules.append({"key": f"toggle closed door {c}", "rules": [{"if": [F.atom_name(x) for x in conds]}
                                                                       for conds, rate in rl if rate >= 0.5]})
    return G.rules_check(world, {"rules": rules})


# ---------------------------------------------------------------- codes against the simulator's labels

def sim_labels(c):
    o, ag = obj_of(c), agent_of(c)
    if o is None or o in ("wall", "goal"):
        return {"object": o or "floor", "state": "none", "agent on it": ag, "colour": "none"}
    if o[0] == "door":
        return {"object": "door", "state": "open" if o[2] == 2 else "closed", "agent on it": ag, "colour": o[1]}
    if o[0] == "ball":
        return {"object": "switch", "state": "on" if o[1] == "yellow" else "off", "agent on it": ag, "colour": o[1]}
    return {"object": {"box": "vase"}.get(o[0], o[0]), "state": "none", "agent on it": ag, "colour": o[1]}


def nmi(x, y):
    """Normalised mutual information (arithmetic mean of the entropies)."""
    import math
    n = len(x)
    px, py, pxy = Counter(x), Counter(y), Counter(zip(x, y))
    hx = -sum(v / n * math.log(v / n) for v in px.values())
    hy = -sum(v / n * math.log(v / n) for v in py.values())
    i = sum(v / n * math.log(v * n / (px[a] * py[b])) for (a, b), v in pxy.items())
    return round(i / ((hx + hy) / 2), 3) if hx > 0 and hy > 0 else 0.0


PURPLE_TILES = {"key purple": KR.code(("key", "purple")), "closed door purple": KR.code(("door", "purple", 0)),
                "open door purple": KR.code(("door", "purple", 2)),
                "agent in doorway purple": KR.code(("door", "purple", 2)) + UP + 1}


def codes_report(codes_arr, tiles):
    """The gate's look at the codes: collisions among the tiles seen in training, new codes on them,
    the purple tiles' codes (with the seen tiles sharing each code), and each codebook against labels."""
    K = codes_arr.shape[1]
    by = {}
    for c in tiles:
        by.setdefault(tuple(int(v) for v in codes_arr[c]), []).append(name_of_code(c))
    rep = {"tiles_seen": len(tiles), "collisions": [v for v in by.values() if len(v) > 1],
           "seen_tiles_with_a_new_code": [name_of_code(c) for c in tiles if (codes_arr[c] == NEW).any()],
           "codes_of_seen_tiles": {name_of_code(c): [int(v) for v in codes_arr[c]] for c in tiles},
           "purple": {}}
    for nmk, c in PURPLE_TILES.items():
        row = [int(v) for v in codes_arr[c]]
        rep["purple"][nmk] = {"codes": ["new" if v == NEW else v for v in row],
                              "shared_per_codebook": [[name_of_code(t) for t in tiles if codes_arr[t, k] == row[k]]
                                                      if row[k] != NEW else [] for k in range(K)],
                              "same_tuple_as": by.get(tuple(row), [])}
    allc = list(tiles) + list(PURPLE_TILES.values())
    labs = [sim_labels(c) for c in allc]
    rep["nmi_codebook_vs_label"] = {f"codebook {k}": {lab: nmi([int(codes_arr[c, k]) for c in allc], [l[lab] for l in labs])
                                                      for lab in labs[0]} for k in range(K)}
    return rep


# ---------------------------------------------------------------- one arm (one set of codes) in every world

WORLDS = ("key", "switch", "either", "both")


def run_codes(label, codes_arr, pooled, data, tests, dev, log, ref=None, mode="tree"):
    """Learn, check and act in the four familiar worlds and the purple ones, with the tiles read
    through these codes. ref: arm 3's steps per world (criterion 1)."""
    T = Tuples(codes_arr)
    lut = np.zeros(256, np.uint8)
    lut[:len(T.lut)] = T.lut
    Kd.APP = lut
    F.NAME = names_for(T)
    out = {"tuples": len(T.tup)}
    real_rules = F.evidence_rules
    for world in WORLDS:
        t0 = time.monotonic()
        D = data[world]
        r = out[world] = {}
        V0, V1 = lut[D["ego0"]], lut[D["ego1"]]
        E = {"act": D["act"], "view_changed": (V0[:, :NV] != V1[:, :NV]).any(1)}
        tr = {"w0": D["w"], "term1": D["term1"]}
        if pooled:                         # the per-tile rules are replaced below; skip their search
            F.evidence_rules = lambda d: ([], log_evidence(d, []), [])
        try:
            F.M, mrep = F.learn(tr, E, V0, V1, lambda m: None, dev)
        finally:
            F.evidence_rules = real_rules
        if pooled:
            outc = np.where(D["term1"], ENDED, np.where(E["view_changed"], CHANGED, UNCHANGED))
            r["entries"] = pooled_model(F.M, V0, V1, D["act"], D["w"], outc, T, log, mode)
            F.NAME = names_for(T)
        else:
            r["entries"] = {"rules": mrep["rules"]}
        r["door_rules"] = door_rules(world, T)
        r["learn_seconds"] = round(time.monotonic() - t0, 1)
        pool = F.pool20()
        Vh0, Vh1 = lut[D["hego0"]], lut[D["hego1"]]
        ok, errs = eval_rows(pool, Vh0, Vh1, D["hact"], D["hterm1"])
        pa = per_action(ok, D["hact"])
        r["heldout_effects"] = {"per_action": pa, "all": round(float(ok.mean()), 6),
                                "lowest": min(v["exact_share"] for v in pa.values()),
                                "errors_top": [[list(map(str, k)), v] for k, v in errs.most_common(6)]}
        res, _ = G.act_arm(pool, D["test"], world)
        pool.close()
        r["acting"] = {k: res[k] for k in ("success", "mean_steps_when_successful", "random_share",
                                           "predictions_wrong", "failed_acts", "seconds_per_layout_mean")}
        r["acting"]["steps_ratio_to_shortest"] = round((res["mean_steps_when_successful"] or 1e9) / D["shortest"], 3)
        if ref is not None:
            r["acting"]["steps_ratio_to_arm3"] = round((res["mean_steps_when_successful"] or 1e9) / ref[world], 3)
        r["criterion_1"] = {"effects_exact": r["heldout_effects"]["lowest"] >= 0.999,
                            "success_99": res["success"] >= 0.99,
                            "steps_within_5pct": ref is None or r["acting"]["steps_ratio_to_arm3"] <= 1.05}
        r["criterion_1"]["pass"] = all(r["criterion_1"].values())
        for tk, (tw, door_open) in TESTS.items():
            if tw != world:
                continue
            X = tests[tk]
            q = r[f"test_{tk}"] = {}
            W0, W1 = lut[X["ego0"]], lut[X["ego1"]]
            patch(door_open=bool(door_open))
            pool = F.pool20()
            ok, errs = eval_rows(pool, W0, W1, X["act"], X["term1"])
            res, recs = G.act_arm(pool, X["test"], tw)
            pool.close()
            patch()
            q["cases"] = {}
            for nmk, rows in X["cases"].items():
                q["cases"][nmk] = {"rows": int(len(rows)), "exact_share": round(float(ok[rows].mean()), 4) if len(rows) else None}
            q["errors_top"] = [[list(map(str, k)), v] for k, v in errs.most_common(8)]
            q["acting"] = {k: res[k] for k in ("success", "mean_steps_when_successful", "random_share", "predictions_wrong",
                                               "failed_acts", "door_opened_by", "pursued_before_door_opened")}
            q["acting"]["steps_ratio_to_shortest"] = round((res["mean_steps_when_successful"] or 1e9) / X["shortest"], 3)
            q["acting"]["chains_at_start"] = res["chains_at_start"]
            crit = [v["exact_share"] for k, v in q["cases"].items() if k in X["criterion_cases"]]
            q["criterion_2"] = {"lowest_case": min((x if x is not None else 0.0 for x in crit), default=None),
                                "pass": bool(crit) and all(x is not None and x >= 0.99 for x in crit)}
            q["criterion_3"] = {"success_98": res["success"] >= 0.98,
                                "steps_within_1_15x": q["acting"]["steps_ratio_to_shortest"] <= 1.15}
            q["criterion_3"]["pass"] = all(q["criterion_3"].values())
        r["seconds"] = round(time.monotonic() - t0, 1)
        log(f"  {label} {world}: effects lowest {r['heldout_effects']['lowest']}, acting {r['acting']['success']} "
            f"({r['acting']['mean_steps_when_successful']} steps); door rules ok {r['door_rules']['exactly_as_expected']}; "
            + "; ".join(f"test {tk}: cases lowest {r[f'test_{tk}']['criterion_2']['lowest_case']}, acting "
                        f"{r[f'test_{tk}']['acting']['success']}" for tk in TESTS if f"test_{tk}" in r)
            + f" ({r['seconds']}s)")
    Kd.APP = APP0
    return out


def verdicts(o):
    """The three criteria for one set of codes."""
    c1 = all(o[w]["criterion_1"]["pass"] for w in WORLDS)
    c2 = all(o[w][f"test_{tk}"]["criterion_2"]["pass"] for tk, w in (("a", "key"), ("b", "switch")))
    c3 = all(o[w][f"test_{tk}"]["criterion_3"]["pass"] for tk, w in (("a", "key"), ("b", "switch")))
    return {"criterion_1": c1, "criterion_2": c2, "criterion_3": c3}


# ---------------------------------------------------------------- main

def setup(episodes, heldout, n_test, n_tep, res, save, log):
    """The experience, held-out transitions and test layouts of the four familiar worlds, the purple
    test worlds, and the encoder's tiles and pairs."""
    # experience, held-out transitions and test layouts in the four familiar worlds
    data = {}
    patch()
    pool = F.pool20()
    for world in WORLDS:
        t0 = time.monotonic()
        layouts, tr, _, _ = dl.collect(pool, world, episodes, 11, 0.0, seq_episodes=4)
        hlay, htr, _, _ = dl.collect(pool, world, heldout, 12, 0.0, seq_episodes=4)
        rng = np.random.default_rng(777)
        test = [ld.make_layout(8, world, rng) for _ in range(n_test)]
        sh, _ = G.shortest(pool, test, world)
        data[world] = {"act": tr["act"].astype(np.int64), "w": tr["w0"].astype(np.float64),
                       "term1": tr["term1"].astype(bool),
                       "ego0": ego_codes(tr["c0"], tr["s0"]).astype(np.uint8),
                       "ego1": ego_codes(tr["c1"], tr["s1"]).astype(np.uint8),
                       "hact": htr["act"].astype(np.int64), "hterm1": htr["term1"].astype(bool),
                       "hego0": ego_codes(htr["c0"], htr["s0"]).astype(np.uint8),
                       "hego1": ego_codes(htr["c1"], htr["s1"]).astype(np.uint8),
                       "test": test, "shortest": sh["mean_steps"]}
        res.setdefault("data", {})[world] = {"stored_transitions": int(len(tr["act"])), "heldout": int(len(htr["act"])),
                                             "shortest_route": sh, "seconds": round(time.monotonic() - t0, 1)}
        log(f"{world}: {len(tr['act'])} stored transitions, {len(htr['act'])} held out; shortest {sh}")
    pool.close()

    # the purple test worlds: every transition of the test episodes, the cases of section 4
    tests = {}
    for tk, (world, door_open) in TESTS.items():
        t0 = time.monotonic()
        patch(purple=True, door_open=bool(door_open))
        pool = F.pool20()
        layouts, tr, _, _ = dl.collect(pool, world, n_tep, 31, 0.0, p_uniform=1.0, seq_episodes=4)
        rng = np.random.default_rng(777)
        test = [ld.make_layout(8, world, rng) for _ in range(n_test)]
        sh, _ = G.shortest(pool, test, world)
        pool.close()
        patch()
        crit, extra = cases(layouts, tr, world)
        allc = {**crit, **extra}
        other = np.flatnonzero(~np.any(np.stack(list(allc.values())), 0))
        other = np.sort(np.random.default_rng(31).choice(other, min(20000, len(other)), replace=False))
        allc["other transitions (a sample of 20,000; reported only)"] = np.isin(np.arange(len(tr["act"])), other)
        sel = np.flatnonzero(np.any(np.stack(list(allc.values())), 0))
        pos = np.full(len(tr["act"]), -1)
        pos[sel] = np.arange(len(sel))
        tests[tk] = {"act": tr["act"][sel].astype(np.int64), "term1": tr["term1"][sel].astype(bool),
                     "ego0": ego_codes(tr["c0"][sel], tr["s0"][sel]).astype(np.uint8),
                     "ego1": ego_codes(tr["c1"][sel], tr["s1"][sel]).astype(np.uint8),
                     "cases": {k: pos[np.flatnonzero(m)] for k, m in allc.items()},
                     "criterion_cases": [k for k in crit if k != EXCLUDED] if tk != "c" else [],
                     "test": test, "shortest": sh["mean_steps"]}
        res.setdefault("data", {})[f"test_{tk}"] = {"world": world, "door_open_at_start": bool(door_open),
                                                     "transitions": int(len(tr["act"])),
                                                     "cases": {k: int(m.sum()) for k, m in allc.items()},
                                                     "shortest_route": sh, "seconds": round(time.monotonic() - t0, 1)}
        log(f"test {tk} ({world}, door open {bool(door_open)}): {res['data'][f'test_{tk}']['cases']}; shortest {sh}")
    save()

    # the encoder's data: the tiles seen and the in-place pairs, from all four worlds
    worlds_seen = [{"ego0": data[w]["ego0"], "ego1": data[w]["ego1"], "act": data[w]["act"],
                    "view_changed": (APP0[data[w]["ego0"]][:, :NV] != APP0[data[w]["ego1"]][:, :NV]).any(1),
                    "term1": data[w]["term1"]} for w in WORLDS]
    tiles, pairs = tiles_and_pairs(worlds_seen)
    res["encoder_data"] = {"tiles": [name_of_code(c) for c in tiles],
                           "pairs": [[name_of_code(a), name_of_code(b)] for a, b in pairs]}
    log(f"encoder data: {len(tiles)} tiles, {len(pairs)} pairs: {res['encoder_data']['pairs']}")
    save()

    return data, tests, tiles, pairs


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    out = Path(args.get("--out", "runs/031_codes.json"))
    episodes, n_test = int(args.get("--episodes", 5000)), int(args.get("--layouts", 500))
    heldout, n_tep = int(args.get("--heldout", 1000)), int(args.get("--test-episodes", 1000))
    seeds = int(args.get("--seeds", 10))
    arms = args.get("--arms", "3,1,1g,2,4,5").split(",")
    closer.MEASURE = "step"
    dl.MAX_GOALS = 10 ** 9
    dl.START_SHARE = 2.0
    dl.Exact = closer.Approach
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    res = {"note": "Card 031, tools/card031/codes.py; card 029's model over codes learned from pixels.",
           "episodes": episodes, "layouts": n_test, "heldout_episodes": heldout, "test_episodes": n_tep,
           "seeds": seeds, "arms": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    assert KR.N_CODES <= 255

    data, tests, tiles, pairs = setup(episodes, heldout, n_test, n_tep, res, save, log)

    ref = None
    for arm in arms:
        if arm == "3":
            o = run_codes("arm 3", APP0[:, None].astype(np.int64), False, data, tests, dev, log)
            ref = {w: o[w]["acting"]["mean_steps_when_successful"] for w in WORLDS}
            res["arms"]["3_exact_tile_names"] = o
            save()
            continue
        if arm in ("1", "1g"):
            ca = oracle_codes()
            mode = "global" if arm == "1g" else "tree"
            o = run_codes(f"arm {arm}", ca, True, data, tests, dev, log, ref, mode)
            o["codes"] = codes_report(ca, tiles)
            o["verdicts"] = verdicts(o)
            res["arms"]["1_labels" + ("_one_set_per_action_as_declared" if arm == "1g" else "")] = o
            log(f"arm {arm} verdicts {o['verdicts']}")
            save()
            continue
        setting = {"2": dict(K=4, M=8, dim=8, pair_w=0.1), "4": dict(K=1, M=64, dim=32, pair_w=0.1),
                   "5": dict(K=4, M=8, dim=8, pair_w=0.0)}[arm]
        name = {"2": "2_main", "4": "4_one_codebook_of_64", "5": "5_no_pairs"}[arm]
        runs = res["arms"][name] = {"setting": setting, "seeds": []}
        for seed in range(seeds):
            ca, crep = train_codes(tiles, pairs, seed=seed, dev=dev, **setting)
            o = run_codes(f"arm {arm} seed {seed}", ca, True, data, tests, dev, log, ref)
            o["encoder"] = crep
            o["codes"] = codes_report(ca, tiles)
            o["verdicts"] = verdicts(o)
            runs["seeds"].append(o)
            log(f"arm {arm} seed {seed}: encoder {crep}; collisions {o['codes']['collisions']}; "
                f"purple {[(k, v['codes']) for k, v in o['codes']['purple'].items()]}; verdicts {o['verdicts']}")
            save()
        runs["seeds_passing"] = {c: sum(s["verdicts"][c] for s in runs["seeds"]) for c in ("criterion_1", "criterion_2", "criterion_3")}
        log(f"arm {arm}: seeds passing {runs['seeds_passing']}")
        save()
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
