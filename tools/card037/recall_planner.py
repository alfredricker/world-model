"""Card 037: recall in the planner.

Card 031's pooled model builds each planner entry from the stored tries whose tiles share the query's codes
in the codebooks chosen by evidence. Here every stored try of the kind counts, weighted by recall: the
query's own tries at weight 1, another tile t's tries scaled by k_t / n_t, with k_t = exp(-sum_j lambda_j
|x_q,j - x_t,j|) on the encoder's pieces (card 033's vote), lambda fitted per world and kind by leave-one-out
on the stored tiles. An entry exists only if some outcome's probability (W_c + 1/4) / (W + 1) is at least one
half (card 033's prior): with no tries of its own, a tile needs a vote of at least 0.5. The entry is then
built as card 031 builds it (categories per codebook, default, card 010's rules), from the weighted tries.
Tuples an outcome creates get vectors piece by piece.

Steps:
  bin/prun python tools/card037/recall_planner.py --encoders --seeds 400-404        card 035's recipe, saved
  bin/prun python tools/card037/recall_planner.py --gate                            alpha, fresh codes, names
  bin/prun python tools/card037/recall_planner.py --main --arm 2 --seeds 400-402 --out runs/037_arm2_a.json
  bin/prun python tools/card037/recall_planner.py --labels --out runs/037_arm1.json
  bin/prun python tools/card037/recall_planner.py --dev --seeds 399-399 [--alpha 4]      familiar worlds only (shakedown)
Arms: 2 recall entries; 3 card 031's counted entries on the same codes; 4 recall with lambda at its start.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card036"))
import fresh as FR  # noqa: E402  (card 036 -> card 035 -> card 034 -> card 033 -> card 031)

N = FR.N
T = N.T
R = N.R
CO = N.C                                        # card 031's module (CO.C is its "taken from the other place")
F, dl = CO.F, CO.dl
FRONT, HELD, CENTRE, BEHIND, SAME = CO.FRONT, CO.HELD, CO.CENTRE, CO.BEHIND, CO.SAME
MOVES, FWD, PICK, DROP, TOG = CO.MOVES, CO.FWD, CO.PICK, CO.DROP, CO.TOG
UNCHANGED, CHANGED = CO.UNCHANGED, CO.CHANGED
KD = 4                                          # codebooks
ROOT = Path(__file__).resolve().parents[2]
ENC = ROOT / "runs" / "037_enc"
MU = 0.01
PRIOR = 0.25                                    # card 033's prior per outcome
KMIN = 0.01                                     # tiles with a smaller vote are left out of the weighted tries
EPS = 0.01                                      # outcomes below this share of an entry are left out
CTX = {}                                        # per seed: z, books, fresh vectors, fit flag, reports


# ---------------------------------------------------------------- vectors of tuples

class Vectors:
    """A vector (4 pieces) for every tuple id: real tiles from the encoder, created tuples piece by piece."""

    def __init__(self, T_):
        self.T, self.v = T_, {}
        self.gen = {}                               # created tuples: 1 if made from real tiles, 2 if from a made one
        z = CTX["z"].reshape(len(CTX["z"]), KD, -1)
        view = set(int(c) for c in CO.VIEW_CODES)
        by = {}
        for c in range(len(T_.lut)):
            by.setdefault(int(T_.lut[c]), []).append(c)
        for u, cs in by.items():
            pick = [c for c in cs if c in view] or cs
            self.v[u] = z[pick].mean(0)

    def code_vec(self, k, code):
        if code == N.NEW:
            return None
        if code < 8:
            return CTX["books"][k][code]
        return CTX["fresh"].get((k, code))

    def compose(self, aid, bid, oid, desc_k):
        if aid in self.v:
            return
        pieces = []
        for k, opt in enumerate(desc_k):
            if opt == CO.U:
                p = self.v[bid][k]
            elif opt == CO.C:
                p = self.v[oid][k]
            else:
                p = self.code_vec(k, opt[1])
                if p is None:
                    p = self.v[bid][k]
            pieces.append(p)
        self.v[aid] = np.stack(pieces)

    def key(self, ids):
        return np.concatenate([self.v[int(u)].ravel() for u in ids])


def concretise(kind, T_, cat, ts, key_ids, V):
    """Card 031's concretise, also giving every created tuple its vector."""
    mask, desc = cat
    if not any(mask):
        return None
    K = len(T_.codes(0))
    w = kind.w[ts]

    def rep(arr, p):
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
        bc, oc = T_.codes(bid), T_.codes(oid)
        new, dk = [], []
        for k in range(K):
            opt = desc[i]
            i += 1
            dk.append(opt)
            new.append(bc[k] if opt == CO.U else oc[k] if opt == CO.C else opt[1])
        n0 = len(T_.tup)
        aid = T_.intern(tuple(new))
        if len(T_.tup) > n0:
            CTX.setdefault("made", {})[aid] = (kind.name, tuple(int(u) for u in key_ids), p, tuple(dk))
            V.gen[aid] = 1 + max(V.gen.get(int(x), 0) for x in (*key_ids, bid, oid))
        V.compose(aid, bid, oid, dk)
        out.append(SAME if aid == bid else aid)
    return tuple(out)


# ---------------------------------------------------------------- recall's metric per kind

def type_labels(kind, T_, type_key):
    """Each stored type's outcome, stated as card 031 states it for that type's tile alone: per changed place
    and codebook, unchanged, taken from the other changed place, or set to a code, over the tile's own tries.
    Tries of different tiles with the same statement are the same outcome."""
    if kind.moves:
        return [int(c) for c in kind.cat]
    labs = [None] * len(kind.w)
    for i in range(int(type_key.max()) + 1 if len(type_key) else 0):
        ts = [int(t) for t in np.flatnonzero(type_key == i)]
        for c, tt in CO.describe(kind, ts, T_).items():
            for t in tt:
                labs[t] = c
    return labs


def fit_lambda(X, Rk, fit=True, steps=1500, lr=0.02):
    """Leave-one-out over the kind's distinct keys (card 033): theta starts with every lambda equal, so
    that the median distance between keys is 1. Returns lambda and the leave-one-out log-likelihood at the
    start and after fitting (mean over keys)."""
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Rt = torch.as_tensor(Rk, dtype=torch.float64)
    n, L = Rt.shape
    D = torch.abs(Xt[:, None] - Xt[None])                           # (n, n, d)
    dsum = D.sum(-1)
    med = dsum[dsum > 0].median() if (dsum > 0).any() else torch.tensor(1.0, dtype=torch.float64)
    lam0 = float(1.0 / med.clamp(min=1e-6))
    th = torch.nn.Parameter(torch.full((X.shape[1],), float(np.log(np.expm1(lam0))), dtype=torch.float64))
    eye = torch.eye(n, dtype=torch.float64)

    def ll(theta):
        lam = torch.nn.functional.softplus(theta)
        k = torch.exp(-(D * lam).sum(-1)) * (1 - eye)
        P = (k @ Rt + 1.0 / L) / (k.sum(-1, keepdim=True) + 1.0)
        return (Rt * torch.log(P)).sum(-1).mean()

    with torch.no_grad():
        l0 = float(ll(th))
    if fit and n > 2:
        opt = torch.optim.Adam([th], lr=lr)
        for _ in range(steps):
            loss = -ll(th)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        l1 = float(ll(th))
        lam = torch.nn.functional.softplus(th).numpy()
    return lam, l0, l1


# ---------------------------------------------------------------- the model

def build_kinds(V0, V1, act, w, out):
    """Card 031's kinds, as its pooled_model builds them."""
    Kind = CO.Kind
    kinds = {}
    for a in MOVES:
        sel = np.flatnonzero(act == a)
        kinds[dl.ACTIONS[a]] = Kind(dl.ACTIONS[a], 1, None, None, V0[sel, FRONT], None, None, None, w[sel], cat=out[sel])
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
    return kinds


class RecallKind:
    """One kind's stored tries, grouped by key, with recall's metric."""

    def __init__(self, name, kind, T_, V, V0, w, log):
        self.name, self.kind, self.T, self.V, self.V0, self.w = name, kind, T_, V, V0, w
        keys = [tuple(int(v) for v in r) for r in kind.keys]
        self.ukeys = sorted(set(keys))
        self.key_index = {k: i for i, k in enumerate(self.ukeys)}
        self.type_key = np.array([self.key_index[k] for k in keys])
        self.nkey = np.bincount(self.type_key, kind.w, len(self.ukeys))
        self.X = np.stack([V.key(k) for k in self.ukeys]) if self.ukeys else np.zeros((0, 1))
        labs = type_labels(kind, T_, self.type_key)
        self.cat_of = labs
        vocab = sorted(set(labs), key=repr)
        li = {l: i for i, l in enumerate(vocab)}
        Rk = np.zeros((len(self.ukeys), max(len(vocab), 1)))
        for t, l in enumerate(labs):
            Rk[self.type_key[t], li[l]] += kind.w[t]
        Rk = Rk / Rk.sum(1, keepdims=True).clip(1e-12)
        if len(self.ukeys) >= 2:
            self.lam, l0, l1 = fit_lambda(self.X, Rk, CTX["fit"])
        else:
            self.lam, l0, l1 = np.ones(self.X.shape[1]), None, None
        self.report = {"keys": len(self.ukeys), "outcomes": len(vocab), "loo_start": None if l0 is None else round(l0, 4),
                       "loo_fitted": None if l1 is None else round(l1, 4)}
        self.cache = {}

    def votes(self, key_ids):
        """Per stored key: own (1) or recall's weight k; the query's total vote from other keys."""
        q = tuple(int(u) for u in key_ids)
        xq = self.V.key(q)
        k = np.exp(-(np.abs(self.X - xq) * self.lam).sum(1))
        own = self.key_index.get(q)
        if own is not None:
            k[own] = 0.0
        return k, own

    def entry(self, key_ids):
        """(types, info, P) or None: the weighted tries behind the entry, and card 031's info over them."""
        q = tuple(int(u) for u in key_ids)
        if q in self.cache:
            return self.cache[q]
        kind = self.kind
        k, own = self.votes(q)
        mult_key = np.where(self.nkey > 0, k / self.nkey.clip(1e-12), 0.0)
        if own is not None:
            mult_key[own] = 1.0
        keep_key = (k >= KMIN)
        if own is not None:
            keep_key[own] = True
        tidx = [t for t in range(len(kind.w)) if keep_key[self.type_key[t]]]
        vote = float(k.sum())
        res = None
        if tidx:
            mult = mult_key[self.type_key]
            wt_types = kind.w * mult
            if kind.moves:
                W = np.bincount(kind.cat[tidx], wt_types[tidx], 3)
                P = (W + PRIOR) / (W.sum() + 1.0)
                if P.max() >= 0.5:
                    res = (int(P.argmax()), None, float(P.max()))
            else:
                cats = self.group(tidx)
                wts = {c: float(wt_types[ts].sum()) for c, ts in cats.items()}
                Wtot = sum(wts.values())
                Pmax = max((v + PRIOR) / (Wtot + 1.0) for v in wts.values())
                if Pmax >= 0.5:
                    kept = [c for c, v in wts.items() if v / (Wtot + 1.0) >= EPS]
                    tidx = sorted(t for c in kept for t in cats[c])
                    cats = self.group(tidx)
                    wts = {c: float(wt_types[ts].sum()) for c, ts in cats.items()}
                    res = (tidx, self.info(tidx, cats, wts, mult), Pmax)
        self.cache[q] = res
        CTX["votes"][-1].setdefault(self.name, {})[q] = {"vote": round(vote, 3), "own": own is not None,
                                                         "entry": res is not None}
        return res

    def group(self, tidx):
        cats = {}
        for t in tidx:
            cats.setdefault(self.cat_of[t], []).append(t)
        return cats

    def info(self, tidx, cats, wts, mult):
        """Card 031's score_group info (categories, weights, default, rules), from weighted tries."""
        kind = self.kind
        info = {"cats": cats, "weights": wts, "rules": {}}
        if len(cats) > 1:
            nothing = [c for c in cats if not any(c[0])]
            default = nothing[0] if nothing else max(wts, key=wts.get)
            info["default"] = default
            rows = np.concatenate([kind.rows[t] for t in tidx])
            lab = np.concatenate([np.full(len(kind.rows[t]), i) for i, t in enumerate(tidx)])
            rmult = np.concatenate([np.full(len(kind.rows[t]), mult[t]) for t in tidx])
            cat_of = {t: c for c, ts in cats.items() for t in ts}
            order = list(cats)
            y_all = np.array([order.index(cat_of[tidx[i]]) for i in lab])
            wr = self.w[rows] * rmult
            for c in cats:
                if c == default:
                    continue
                y = (y_all == order.index(c)).astype(np.float64)
                d, inv = CO.rule_data(self.V0[rows], y, wr)
                rules, best, trace = CO.evidence_rules(d)
                info["rules"][c] = (d, rules, F.rule_rates(d, rules), None)
        return info


def recall_model(m, V0, V1, act, w, out, T_, log, mode="tree"):
    """Replaces card 031's pooled_model: the same write-back, with entries from recall."""
    t0 = time.monotonic()
    rep = {}
    assert [int(q) for q in m.change_places] == [FRONT, HELD], m.change_places
    if CTX.get("V") is None or CTX["V"].T is not T_:      # one set of tuples serves every world of a seed
        CTX["V"] = Vectors(T_)
    V = CTX["V"]
    CTX["votes"].append({})
    kinds = build_kinds(V0, V1, act, w, out)
    rk = {name: RecallKind(name, kind, T_, V, V0, w, log) for name, kind in kinds.items()}
    rep["recall"] = {name: r.report for name, r in rk.items()}
    CTX["rk"] = rk
    universe = set(int(v) for v in T_.lut[CO.VIEW_CODES])
    unseen = set(int(v) for v in T_.lut) - universe     # tile codes the view never shows: no entries
    done = set(unseen)
    holdable = set(int(v) for v in np.unique(V0[:, HELD])) | set(int(v) for v in np.unique(V1[:, HELD]))
    held_done = set()                                   # drop entries only for what can be in the hand
    m.move_out = {a: np.full(256, UNCHANGED, np.int64) for a in MOVES}
    draw, undraw = np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8)
    table = {}
    for sweep in range(6):
        deep = {u for u in range(len(T_.tup)) if V.gen.get(u, 0) > 1}   # made from a made tuple: no entries
        todo = set(range(len(T_.tup))) - done - deep
        holdable -= deep
        held_todo = holdable - held_done
        if not todo and not held_todo:
            break
        for u in sorted(todo):
            for a in MOVES:
                e = rk[dl.ACTIONS[a]].entry((u,))
                if e is not None:
                    m.move_out[a][u] = e[0]
            for name, arr in (("draw", draw), ("undraw", undraw)):
                e = rk[name].entry((u,))
                if e is None:
                    continue
                ts, info, _ = e
                c = max(info["cats"], key=lambda c: info["weights"][c])
                o = concretise(rk[name].kind, T_, c, info["cats"][c], (u,), V)
                arr[u] = u if o is None or o[0] == SAME else o[0]
        pairs = [(u, None) for u in sorted(todo)] + [(u, h) for u in range(len(T_.tup)) for h in sorted(holdable)
                                                       if (u in todo or h in held_todo) and u not in unseen and u not in deep]
        for u, h in pairs:
            for a in ((PICK, TOG) if h is None else (DROP,)):
                name = dl.ACTIONS[a]
                key_ids = (u,) if h is None else (u, h)
                e = rk[name].entry(key_ids)
                if e is None:
                    continue
                ts, info, _ = e
                kind = rk[name].kind
                ent = F.Entry()
                cats = info["cats"]
                conc = {c: concretise(kind, T_, c, cats[c], key_ids, V) for c in cats}
                holdable.update(o[1] for o in conc.values() if o is not None and o[1] != SAME)
                if len(cats) == 1:
                    ent.single = next(iter(conc.values()))
                    if ent.single is None:
                        continue
                else:
                    wt = info["weights"]
                    ent.default = conc[info["default"]]
                    opt = [conc[c] for c in sorted(cats, key=lambda c: -wt[c]) if conc[c] is not None]
                    if not opt:
                        continue
                    ent.optimistic = opt[0]
                    ent.multi = []
                    for c, (d, rules, rr, gain) in info["rules"].items():
                        rl, dr, _, _ = rr
                        ent.multi.append((conc[c], [(conds, rate) for conds, rate, _, _ in rl], dr))
                table[(a, u, h) if a == DROP else (a, u)] = ent
        done.update(todo)
        held_done.update(held_todo)
    ident = np.arange(256)
    for x, y in zip(ident[undraw != ident], undraw[undraw != ident]):
        if draw[y] == y:
            draw[y] = x
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
    rep["tuples_created_by_outcomes"] = [F.nm(u) for u in sorted(set(range(len(T_.tup))) - universe - unseen)]
    rep["tuples"] = len(T_.tup)
    rep["made_from_made"] = sum(1 for g in V.gen.values() if g > 1)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    CTX["model_reports"].append(rep)
    return rep


# ---------------------------------------------------------------- main

def main():
    args = sys.argv[1:]
    flag = lambda f: f in args
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    T.configure()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda msg: print(f"[{time.monotonic() - t00:6.0f}s] {msg}", flush=True)
    cache = pickle.loads(T.MEMORY.read_bytes())
    mem, tiles, pairs = cache["mem"], cache["tiles"], cache["pairs"]
    groups = T.use_groups(mem, dev)
    a, b = get("--seeds", "400-409").split("-")
    seeds = tuple(range(int(a), int(b) + 1))
    N.ENC = ENC

    if flag("--encoders"):
        for seed in seeds:
            e = N.encoder(MU, seed, tiles, pairs, groups, dev, log)
            log(f"encoder {seed}: {e['rep']}")
        return

    gate_file = ROOT / "runs" / "037_gate.json"
    if flag("--gate"):
        encs = {s: N.encoder(MU, s, tiles, pairs, groups, dev, log) for s in range(400, 410)}
        tot = None
        for s, e in encs.items():
            r, n, sh = N.loo(e["z"], e["books"], tiles)
            tot = r if tot is None else {k: tot[k] + r[k] for k in r}
        alpha = max(N.ALPHAS, key=lambda x: (tot[x], -x))
        names = {}
        for s, e in encs.items():
            ca, vec, rp = FR.fresh_codes(e["z"], e["books"], tiles, alpha)
            names[s] = FR.named_apart(ca, tiles)
        res = {"alpha": alpha, "loo_right": {str(k): v for k, v in tot.items()},
               "names": names, "seeds_named_apart": sum(v["apart"] for v in names.values())}
        res["names_pass"] = res["seeds_named_apart"] >= 8
        gate_file.write_text(json.dumps(res, indent=1) + "\n")
        log(f"gate: alpha {alpha}; named apart {res['seeds_named_apart']} of 10; loo {res['loo_right']}")
        return

    d = pickle.loads(T.DATA.read_bytes())
    data, tests = d["data"], d["tests"]
    ref = json.loads(T.REF.read_text())
    out = Path(get("--out", "runs/037_main.json"))
    res = {"note": "Card 037, tools/card037/recall_planner.py", "arms": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")

    if flag("--labels"):
        ca = CO.oracle_codes()
        o = CO.run_codes("arm 1", ca, True, data, tests, dev, log, ref)
        o["verdicts"] = T.verdicts(o)
        res["arms"]["1_labels"] = o
        log(f"arm 1 verdicts {o['verdicts']}")
        save()
        return

    if flag("--dev"):                   # shakedown on a spare seed: familiar worlds only, no test worlds
        CO.TESTS = {}
        seed = seeds[0]
        e = N.encoder(MU, seed, tiles, pairs, groups, dev, log)
        ca, vec, rp = FR.fresh_codes(e["z"], e["books"], tiles, float(get("--alpha", "6")))
        CTX.clear()
        CTX.update({"z": e["z"], "books": e["books"], "fresh": vec, "fit": not flag("--no-fit"), "votes": [],
                    "model_reports": []})
        CO.pooled_model = recall_model
        oc = CO.run_codes(f"dev seed {seed}", ca, True, data, tests, dev, log, ref)
        print(json.dumps({w: {"criterion_1": oc[w]["criterion_1"], "effects_lowest": oc[w]["heldout_effects"]["lowest"],
                              "errors_top": oc[w]["heldout_effects"]["errors_top"][:3],
                              "acting": oc[w]["acting"]["success"]} for w in CO.WORLDS}, indent=1, default=str))
        print(json.dumps([{"seconds": r["seconds"], "tuples": r["tuples"], "created": len(r["tuples_created_by_outcomes"]),
                           "made_from_made": r["made_from_made"]}
                          for r in CTX["model_reports"]]))
        return

    gate = json.loads(gate_file.read_text())
    alpha = gate["alpha"]
    arm = get("--arm", "2")
    runs = res["arms"][{"2": "2_recall", "3": "3_counted", "4": "4_recall_lambda_at_start"}[arm]] = {"alpha": alpha, "seeds": []}
    real_pooled = CO.pooled_model
    for seed in seeds:
        e = N.encoder(MU, seed, tiles, pairs, groups, dev, log)
        ca, vec, rp = FR.fresh_codes(e["z"], e["books"], tiles, alpha)
        CTX.clear()
        CTX.update({"z": e["z"], "books": e["books"], "fresh": vec, "fit": arm != "4", "votes": [],
                    "model_reports": []})
        CO.pooled_model = real_pooled if arm == "3" else recall_model
        try:
            oc = CO.run_codes(f"arm {arm} seed {seed}", ca, True, data, tests, dev, log, ref)
        except RuntimeError as ex:                      # the planner's 254 tuples: the seed fails every criterion
            runs["seeds"].append({"seed": seed, "error": str(ex), "tuples_made": len(CTX.get("made", {})),
                                  "verdicts": {"criterion_1": False, "criterion_2": False, "criterion_3": False}})
            log(f"arm {arm} seed {seed}: {ex}")
            save()
            continue
        finally:
            CO.pooled_model = real_pooled
        o = {"seed": seed, "encoder": e["rep"], "names": FR.named_apart(ca, tiles), "worlds": oc,
             "verdicts": T.verdicts(oc)}
        if arm != "3":
            o["recall_reports"] = [r["recall"] for r in CTX["model_reports"]]
            o["model_seconds"] = [r["seconds"] for r in CTX["model_reports"]]
            # the last world built is "both"; the new tiles' votes per kind (tuple ids from its lut)
            T_last = CO.Tuples(ca)
            nt = {nm: int(T_last.lut[c]) for nm, c in N.KP.new_tiles().items()}
            v0 = CTX["votes"][0] if CTX["votes"] else {}           # the key world, world (a)'s familiar one
            o["new_tile_votes"] = {name: {nm: v0.get(name, {}).get((u,)) for nm, u in nt.items()}
                                   for name in ("forward", "pickup", "toggle", "draw", "undraw")}
        if arm == "2":
            o["recall"] = T.recall_reports(e["z"], list(e["theta"]), mem)
        runs["seeds"].append(o)
        log(f"arm {arm} seed {seed}: verdicts {o['verdicts']}")
        save()
    runs["seeds_passing"] = {c: sum(s["verdicts"][c] for s in runs["seeds"]) for c in ("criterion_1", "criterion_2", "criterion_3")}
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done: {runs['seeds_passing']} -> {out}")


if __name__ == "__main__":
    main()
