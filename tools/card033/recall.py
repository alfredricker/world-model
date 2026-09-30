"""Card 033: relation codes.

Recall predicts an event's outcome from its own counts plus a vote of the other remembered events of the
same action, each weighted by how alike it is: k = exp(-sum_j lambda_j |x_j - x'_j|) (Shepard's law with
Nosofsky's attention weights). An event is (action, the tile in front, the held tile); its key x is the
two tiles' numbers in card 031's encoder (before the codebooks) and how far apart the two are within
each part (piece) of those numbers.

    P(c | q) = (n_qc + sum_i k_i r_ic + pi_c) / (n_q + sum_i k_i + 1)

What "alike" means (lambda, per world and action) is learned by predicting each memory from the others
(leave-one-out, as neighbourhood components analysis); the same term, replayed, trains the encoder. The
relation "fits" (leave-one-out recall says the door changes) becomes a code when card 010's evidence
prefers a pooled rule over it to one rule per door. Test: a yellow door, a colour never seen on keys or
doors, opened within two tries.

Steps:
  bin/prun python tools/card033/recall.py --collect                  experience -> runs/033_memory.npz
  bin/prun python tools/card033/recall.py --sweep MU                 gate: 10 encoders at weight MU
  bin/prun python tools/card033/recall.py --main --mu MU --seeds 100-104 --out runs/033_main_a.json
  bin/prun python tools/card033/recall.py --others --mu MU --out runs/033_others.json   (arms 1, 3, 4, 5)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# yellow keys and doors, appended to the renderer's object list before any tile table is built
from worldmodel.envs import keydoor_render as KR  # noqa: E402

YELLOW_OBJECTS = [("key", "yellow")] + [("door", "yellow", st) for st in range(3)]
for _o in YELLOW_OBJECTS:
    if _o not in KR.OBJ_INDEX:
        KR.OBJ_INDEX[_o] = len(KR.OBJECTS)
        KR.OBJECTS.append(_o)
KR.N_CODES = len(KR.OBJECTS) * 5

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card031"))
import codes as C  # noqa: E402  (appends purple after yellow; builds the tile tables)

F, Kd, dl, ld, closer, G = C.F, C.Kd, C.dl, C.ld, C.closer, C.G
APP0, TILES, NEW = C.APP0, C.TILES, C.NEW
FRONT, HELD, NV = C.FRONT, C.HELD, C.NV
PICK, DROP, TOG = C.PICK, C.DROP, C.TOG
ACTIONS = (PICK, DROP, TOG)
ANAME = {PICK: "pick up", DROP: "drop", TOG: "toggle"}
KINDS = ("nothing", "swapped", "in place", "other")
NOTHING, SWAPPED, IN_PLACE, OTHER = range(4)
WORLDS = C.WORLDS
C.TESTS = {}                                   # card 031's purple test worlds are not used here
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs" / "033_memory.pkl"
MU_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
LAST = {}                                      # the last encoder's codebook vectors
SEEDS = tuple(range(100, 110))
FLOOR = KR.code(None)
code = KR.code


# ---------------------------------------------------------------- experience and memory

def outcome_kinds(e0, e1):
    """The four kinds of change, from exact tiles at the front and held places (appendix A). As built:
    picking up and dropping are both a swap (an empty hand shows as floor), so "to the hand" and "to
    the front" are one kind."""
    f0, f1, h0, h1 = APP0[e0[:, FRONT]], APP0[e1[:, FRONT]], APP0[e0[:, HELD]], APP0[e1[:, HELD]]
    k = np.full(len(f0), OTHER)
    k[(f1 != f0) & (h1 == h0)] = IN_PLACE
    k[(f1 == h0) & (h1 == f0) & (f1 != f0)] = SWAPPED
    k[(f0 == f1) & (h0 == h1)] = NOTHING
    return k


def build_memory(D):
    """Per action: each distinct (front, held) appearance pair, one representative code each, weighted
    counts per kind. Also the toggle rows (for the evidence), with their memory index."""
    mem, rows = {}, None
    for a in ACTIONS:
        m = np.flatnonzero(D["act"] == a)
        e0, e1, w = D["ego0"][m], D["ego1"][m], D["w"][m]
        kinds = outcome_kinds(e0, e1)
        fc, hc = e0[:, FRONT].astype(np.int64), e0[:, HELD].astype(np.int64)
        key = APP0[fc].astype(np.int64) * 256 + APP0[hc]
        u, first, inv = np.unique(key, return_index=True, return_inverse=True)
        inv = inv.ravel()
        counts = np.zeros((len(u), len(KINDS)))
        np.add.at(counts, (inv, kinds), w)
        mem[a] = {"front": fc[first], "held": hc[first], "counts": counts}
        if a == TOG:
            rows = {"V0": APP0[e0].astype(np.uint8), "kind": kinds, "w": w, "mem": inv, "front_app": APP0[fc]}
    return mem, rows


def setup(episodes, heldout, n_test, res, save, log):
    """Card 031's familiar data (four worlds), without its purple test worlds."""
    data = {}
    C.patch()
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
                       "ego0": C.ego_codes(tr["c0"], tr["s0"]).astype(np.uint8),
                       "ego1": C.ego_codes(tr["c1"], tr["s1"]).astype(np.uint8),
                       "hact": htr["act"].astype(np.int64), "hterm1": htr["term1"].astype(bool),
                       "hego0": C.ego_codes(htr["c0"], htr["s0"]).astype(np.uint8),
                       "hego1": C.ego_codes(htr["c1"], htr["s1"]).astype(np.uint8),
                       "test": test, "shortest": sh["mean_steps"]}
        res.setdefault("data", {})[world] = {"stored_transitions": int(len(tr["act"])), "heldout": int(len(htr["act"])),
                                             "shortest_route": sh, "seconds": round(time.monotonic() - t0, 1)}
        log(f"{world}: {len(tr['act'])} stored transitions, {len(htr['act'])} held out; shortest {sh}")
    pool.close()
    worlds_seen = [{"ego0": data[w]["ego0"], "ego1": data[w]["ego1"], "act": data[w]["act"],
                    "view_changed": (APP0[data[w]["ego0"]][:, :NV] != APP0[data[w]["ego1"]][:, :NV]).any(1),
                    "term1": data[w]["term1"]} for w in WORLDS]
    tiles, pairs = C.tiles_and_pairs(worlds_seen)
    save()
    return data, tiles, pairs


def memory_all(data):
    mem, rows = {}, {}
    for w in WORLDS:
        mem[w], rows[w] = build_memory(data[w])
    return mem, rows


def memory_report(mem, rows):
    out = {}
    for w in WORLDS:
        r = out[w] = {ANAME[a]: len(mem[w][a]["counts"]) for a in ACTIONS}
        m = mem[w][TOG]
        doors = {}
        for i in range(len(m["counts"])):
            f, h = C.name_of_code(m["front"][i]), C.name_of_code(m["held"][i])
            if f.startswith("closed door"):
                doors[f"{f} / held {h}"] = {k: round(float(v)) for k, v in zip(KINDS, m["counts"][i]) if v > 0}
        r["toggles at closed doors (weighted)"] = doors
    return out


# ---------------------------------------------------------------- recall (torch)

PARTS = [slice(8 * k, 8 * k + 8) for k in range(4)]          # the encoder's four pieces


def keys_of(z, fcodes, hcodes, mode, parts=None):
    """An event's key: the two things' own numbers, and (mode "rel") how far apart they are within each
    part. As built: a distance, not the signed difference, so that a new value (a new colour) lands on
    the familiar scale: "same" is 0 whatever the value. Works for torch tensors and numpy arrays."""
    parts = PARTS if parts is None else parts
    zf, zh = z[fcodes], z[hcodes]
    if hasattr(zf, "detach"):
        import torch
        own = [zf, zh]
        rel = [torch.linalg.vector_norm(zf[:, s] - zh[:, s], dim=1, keepdim=True) for s in parts]
        return torch.cat(own + (rel if mode == "rel" else []), 1)
    rel = [np.linalg.norm(zf[..., s] - zh[..., s], axis=-1, keepdims=True) for s in parts]
    return np.concatenate([zf, zh] + (rel if mode == "rel" else []), -1)


def stack(z, groups, mode, parts=None):
    """All groups' keys, padded: X (G, N, D), rates R (G, N, kinds), mask (G, N)."""
    import torch
    xs = [keys_of(z, f, h, mode, parts) for f, h in zip(groups.f, groups.h)]
    G, N, D = len(xs), max(len(x) for x in xs), xs[0].shape[1]
    X = z.new_zeros((G, N, D))
    R = z.new_zeros((G, N, len(KINDS)))
    Mk = z.new_zeros((G, N))
    for g, (x, c) in enumerate(zip(xs, groups.c)):
        X[g, :len(x)] = x
        R[g, :len(x)] = c / c.sum(1, keepdim=True)
        Mk[g, :len(x)] = 1
    return X, R, Mk


def loo_loglik(X, R, Mk, lam):
    """Per group, the mean over memories (each once) of sum_c r_qc log P_-q(c | q): each memory predicted
    from the other memories' votes and the prior only. lam: (G, D). Returns (G,)."""
    import torch
    d = (torch.abs(X[:, :, None] - X[:, None]) * lam[:, None, None]).sum(-1)
    k = torch.exp(-d) * Mk[:, :, None] * Mk[:, None] * (1 - torch.eye(X.shape[1], device=X.device))
    P = (k @ R + 1.0 / len(KINDS)) / (k.sum(-1, keepdim=True) + 1.0)
    return ((R * torch.log(P)).sum(-1) * Mk).sum(1) / Mk.sum(1)


def init_theta(X, Mk):
    """As built: theta starts so that the median distance between two memories of a group is 1 (all
    lambda equal); with theta = 0 (lambda 0.69) every vote starts near zero and no gradient reaches
    lambda. Returns (G, D)."""
    import torch
    with torch.no_grad():
        out = []
        for g in range(len(X)):
            x = X[g, Mk[g] > 0]
            d = torch.abs(x[:, None] - x[None]).sum(-1)
            lam = 1.0 / d[d > 0].median().clamp(min=1e-6)
            out.append(torch.full((X.shape[2],), float(torch.log(torch.expm1(lam))), device=X.device))
        return torch.stack(out)


class Groups:
    """The memories as tensors, one group per (world, action)."""

    def __init__(self, mem, dev):
        import torch
        self.names, self.f, self.h, self.c = [], [], [], []
        for w in WORLDS:
            for a in ACTIONS:
                m = mem[w][a]
                self.names.append((w, a))
                self.f.append(torch.as_tensor(m["front"], device=dev))
                self.h.append(torch.as_tensor(m["held"], device=dev))
                self.c.append(torch.as_tensor(m["counts"], dtype=torch.float32, device=dev))


def fit_theta(z, groups, mode, dev, parts=None, steps=2000, lr=0.01):
    """Arms 1 and 4: learn lambda alone on fixed numbers."""
    import torch
    z = torch.as_tensor(z, dtype=torch.float32, device=dev)
    X, R, Mk = stack(z, groups, mode, parts)
    th = torch.nn.Parameter(init_theta(X, Mk))
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        loss = -loo_loglik(X, R, Mk, torch.nn.functional.softplus(th)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ll = loo_loglik(X, R, Mk, torch.nn.functional.softplus(th))
    return list(th.detach().cpu().numpy()), {f"{w} {ANAME[a]}": round(float(v), 4) for (w, a), v in zip(groups.names, ll)}


# ---------------------------------------------------------------- the encoder: card 031's, plus replay

def train_encoder(tile_codes, pair_codes, groups, mu, mode="rel", seed=0, dev="cuda", K=4, M=8, dim=8,
                  pair_w=0.1, updates=5000, reseed_new=False, extra=None, lam_penalty=None):
    """Card 031's train_codes as built (sliced VQ, rebuild, codebook and commitment terms, adaptive pair
    rule, re-seeding, "new"), plus mu x the replay term: minus the sum over (world, action) of the
    leave-one-out log-likelihood of the memories, with lambda learned alongside. mu = 0 is card 031's
    encoder. Returns codes, report, the continuous numbers z of every tile code, and theta."""
    import torch
    import torch.nn as nn
    fn = torch.nn.functional
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.manual_seed(seed)
    x_all = torch.as_tensor(TILES / 255.0, dtype=torch.float32, device=dev).permute(0, 3, 1, 2)
    enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                        nn.ReLU(), nn.Flatten(), nn.Linear(32 * 16, K * dim)).to(dev)
    dec = nn.Sequential(nn.Linear(K * dim, 32 * 16), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)),
                        nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()).to(dev)
    books = nn.Parameter(torch.randn(K, M, dim, device=dev))
    params = list(enc.parameters()) + list(dec.parameters()) + [books]
    tc = torch.as_tensor(tile_codes, device=dev)
    pa = torch.as_tensor([p[0] for p in pair_codes], device=dev, dtype=torch.long)
    pb = torch.as_tensor([p[1] for p in pair_codes], device=dev, dtype=torch.long)

    def pieces(x):
        return fn.normalize(enc(x).reshape(len(x), K, dim), dim=-1)

    def quant(z):
        e = fn.normalize(books, dim=-1)
        d = torch.cdist(z.transpose(0, 1), e)
        idx = d.argmin(2).T
        zq = e[torch.arange(K, device=dev)[None], idx]
        return zq, idx

    theta = []
    if mu > 0:
        with torch.no_grad():
            X0, _, M0 = stack(pieces(x_all).reshape(len(x_all), -1), groups, mode)
            theta = [nn.Parameter(init_theta(X0, M0))]
    opt = torch.optim.Adam(params + theta, lr=1e-3)

    t0 = time.monotonic()
    replay = torch.zeros(())
    for step in range(updates):
        x = x_all[tc]
        z = pieces(x)
        zq, _ = quant(z)
        st = z + (zq - z).detach()
        rebuild = fn.mse_loss(dec(st.reshape(len(x), -1)), x)
        vq = fn.mse_loss(zq, z.detach()) + 0.25 * fn.mse_loss(z, zq.detach())
        if len(pa) and pair_w > 0:
            za, zb = pieces(x_all[pa]), pieces(x_all[pb])
            d = (za - zb).norm(dim=-1)
            mid = (d.amax(1, keepdim=True) + d.amin(1, keepdim=True)).detach() / 2
            pair = (d * (d < mid)).sum(1).mean()
        else:
            pair = torch.zeros((), device=dev)
        loss = rebuild + vq + pair_w * pair
        if mu > 0:
            X, R, Mk = stack(pieces(x_all).reshape(len(x_all), -1), groups, mode)
            replay = -loo_loglik(X, R, Mk, fn.softplus(theta[0])).sum()
            loss = loss + mu * replay
            if lam_penalty is not None:     # card 035: a penalty on recall's weights, inside mu
                loss = loss + mu * lam_penalty(fn.softplus(theta[0]))
        if extra is not None:               # probes after card 033: a further term on every tile's pieces
            loss = loss + extra(pieces(x_all))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 249 and step < updates - 500:
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
                if reseed_new:              # diagnostic (gate): also re-seed where a training tile is "new"
                    e = fn.normalize(books, dim=-1)
                    for k in range(K):
                        used = sorted(set(it[:, k].tolist()))
                        unused = sorted(set(range(M)) - set(used))
                        if not unused or len(used) < 2:
                            continue
                        ek = e[k, used]
                        near, j = torch.cdist(zt[:, k], ek).min(1)
                        dd = torch.cdist(ek, ek) + torch.eye(len(used), device=dev) * 1e9
                        far = torch.nonzero(near > dd.min(1).values[j] / 2).flatten().tolist()
                        if far:
                            books[k, unused[0]] = zt[far[0], k]
    with torch.no_grad():
        e = fn.normalize(books, dim=-1)
        LAST["books"] = e.cpu().numpy()     # card 035 reads the codes again from these
        z_tr = pieces(x_all[tc])
        _, idx_tr = quant(z_tr)
        used = [sorted(set(idx_tr[:, k].tolist())) for k in range(K)]
        z_all = pieces(x_all)
        out = np.full((len(x_all), K), NEW, np.int64)
        for k in range(K):
            ek = e[k, used[k]]
            d = torch.cdist(z_all[:, k], ek)
            near, j = d.min(1)
            if len(used[k]) > 1:
                dd = torch.cdist(ek, ek) + torch.eye(len(used[k]), device=dev) * 1e9
                half = dd.min(1).values[j] / 2
            else:
                half = torch.full_like(near, 1.0)
            ok = (near <= half).cpu().numpy()
            out[ok, k] = np.array(used[k])[j.cpu().numpy()[ok]]
        rep = {"rebuild_mse": round(float(rebuild), 6), "pair_term": round(float(pair), 4),
               "replay_term": round(float(replay), 4), "codes_used": [len(u) for u in used],
               "seconds": round(time.monotonic() - t0, 1)}
    return out, rep, z_all.reshape(len(x_all), -1).cpu().numpy(), (list(theta[0].detach().cpu().numpy()) if theta else [])


def tiles_ok(codes_arr, tiles):
    t = [tuple(int(v) for v in codes_arr[c]) for c in tiles]
    return {"distinct": len(set(t)) == len(t), "none_new": all(NEW not in x for x in t),
            "collisions": sum(t.count(x) > 1 for x in set(t))}


# ---------------------------------------------------------------- recall (numpy, for predictions)

class Recall:
    """One (world, action) memory, with a key function and lambda; predictions and confidence."""

    def __init__(self, z, m, theta, mode, parts=None):
        self.z, self.mode, self.parts = z, mode, parts
        self.lam = np.log1p(np.exp(theta))
        self.f, self.h, self.c = list(m["front"]), list(m["held"]), [np.asarray(v, float) for v in m["counts"]]

    def key(self, f, h):
        return keys_of(self.z, np.array([f]), np.array([h]), self.mode, self.parts)[0]

    def find(self, f, h):
        """Memories are distinct appearance pairs: exact pixels, so equal numbers."""
        for i in range(len(self.f)):
            if APP0[self.f[i]] == APP0[f] and APP0[self.h[i]] == APP0[h]:
                return i
        return None

    def predict(self, f, h, loo=False, vote=True):
        """P(kind), own counts, votes' total weight. loo: leave the event's own memory out entirely."""
        q = self.find(f, h)
        own = np.zeros(len(KINDS)) if (q is None or loo) else self.c[q]
        num, den = own + 1.0 / len(KINDS), own.sum() + 1.0
        kw = 0.0
        if vote:
            xq = self.key(f, h)
            for i in range(len(self.f)):
                if i == q:
                    continue
                k = float(np.exp(-(self.lam * np.abs(xq - self.key(self.f[i], self.h[i]))).sum()))
                r = self.c[i] / self.c[i].sum()
                num, den, kw = num + k * r, den + k, kw + k
        return num / den, float(own.sum()), kw

    def add(self, f, h, kind, w=1.0):
        q = self.find(f, h)
        if q is None:
            self.f.append(f)
            self.h.append(h)
            self.c.append(np.zeros(len(KINDS)))
            q = len(self.f) - 1
        self.c[q][kind] += w


def right(P, true):
    """The true kind is strictly the most probable (a tie is wrong)."""
    return bool(P[true] > np.delete(P, true).max())


# ---------------------------------------------------------------- criterion 1: the evidence

def rule_data_fits(V0, y, w, fits):
    """Card 028's rule data (held = X, X in view), plus the atom "fits"."""
    held = V0[:, HELD]
    pres = np.zeros((len(V0), 256), bool)
    pres[np.arange(len(V0))[:, None], V0[:, :NV]] = True
    hv = np.unique(held)
    pv = np.flatnonzero(pres.any(0))
    cols = [held[:, None] == hv[None], pres[:, pv]]
    atoms = [(0, int(u)) for u in hv] + [(1, int(u)) for u in pv]
    if fits is not None:
        cols.append(fits[:, None])
        atoms.append((2, "fits"))
    X = np.concatenate(cols, 1)
    uniq, inv = np.unique(X, axis=0, return_inverse=True)
    inv = inv.ravel()
    d = type("Dat", (), {})()
    d.X, d.atoms = uniq, atoms
    d.n = np.bincount(inv, w, len(uniq))
    d.k = np.bincount(inv, w * y, len(uniq))
    return d


def atom_text(a):
    if a[0] == 2:
        return "fits(held, front)"
    return ("held = " if a[0] == 0 else "in view: ") + C.name_of_code(int(np.flatnonzero(APP0 == a[1])[0]))


def evidence(world, rows, mem_tog, rec):
    """Card 010's evidence for the toggles at fronts with several outcome kinds: per door, pooled, and
    pooled with "fits" (leave-one-out recall predicts "in place")."""
    counts = mem_tog["counts"]
    fronts = {}
    for i in range(len(counts)):
        fronts.setdefault(int(APP0[mem_tog["front"][i]]), np.zeros(len(KINDS)))
        fronts[int(APP0[mem_tog["front"][i]])] += counts[i]
    multi = sorted(u for u, c in fronts.items() if (c > 0).sum() > 1)
    sel = np.isin(rows["front_app"], multi)
    V0, kind, w, mi, fa = rows["V0"][sel], rows["kind"][sel], rows["w"][sel], rows["mem"][sel], rows["front_app"][sel]
    y = (kind == IN_PLACE).astype(np.float64)
    fits_mem = np.array([rec.predict(mem_tog["front"][i], mem_tog["held"][i], loo=True)[0][IN_PLACE] >= 0.5
                         for i in range(len(counts))])
    out = {"fronts": [C.name_of_code(int(np.flatnonzero(APP0 == u)[0])) for u in multi], "tries": int(len(y))}
    per = 0.0
    for u in multi:
        m = fa == u
        d = rule_data_fits(V0[m], y[m], w[m], None)
        rules, best, _ = C.evidence_rules(d)
        per += best
    d = rule_data_fits(V0, y, w, None)
    rp, bp, _ = C.evidence_rules(d)
    d = rule_data_fits(V0, y, w, fits_mem[mi])
    rf, bf, _ = C.evidence_rules(d)
    uses = any(len(d.atoms) - 1 in r for r in rf)
    scores = {"per door": round(float(per), 2), "pooled": round(float(bp), 2), "pooled with fits": round(float(bf), 2)}
    chosen = "pooled with fits" if bf > max(per, bp) else ("pooled" if bp > per else "per door")
    out.update({"evidence_nats": scores, "chosen": chosen, "fits_in_rules": bool(uses),
                "admitted": chosen == "pooled with fits" and bool(uses),
                "pooled_with_fits_rules": [[atom_text(d.atoms[j]) for j in r] for r in rf],
                "fits_true_for": [f"{C.name_of_code(mem_tog['front'][i])} / held {C.name_of_code(mem_tog['held'][i])}"
                                  for i in range(len(counts)) if fits_mem[i] and APP0[mem_tog['front'][i]] in multi]})
    return out


# ---------------------------------------------------------------- criterion 2b and criterion 3

def no_override(recs):
    """Every memory whose own tries all had one kind is predicted as that kind (own counts + votes)."""
    bad = []
    n = 0
    for (w, a), rec in recs.items():
        for i in range(len(rec.f)):
            c = rec.c[i]
            if (c > 0).sum() != 1:
                continue
            n += 1
            P, _, _ = rec.predict(rec.f[i], rec.h[i])
            if not right(P, int(np.argmax(c))):
                bad.append(f"{w}: {ANAME[a]} {C.name_of_code(rec.f[i])} / held {C.name_of_code(rec.h[i])} "
                           f"({round(float(c.sum()))} tries)")
    return {"memories_checked": n, "overridden": bad, "pass": not bad}


def new_colour_trial(rec, colour, rng, vote=True):
    """Criterion 3: the closed door of a new colour, four keys; tries in order of P(in place); each try
    stored at once. Then the 8 pairs."""
    import copy
    rec = copy.deepcopy(rec)
    door = code(("door", colour, 0))
    keys = {c: code(("key", c)) for c in (colour, "red", "green", "blue")}
    untried, tries, log = list(keys), 0, []
    first_conf = None
    while untried:
        ps = {}
        for c in untried:
            P, own, kw = rec.predict(door, keys[c], vote=vote)
            ps[c] = P[IN_PLACE]
            if first_conf is None and c == colour:
                first_conf = own + kw
        best = max(ps.values())
        tied = [c for c in untried if ps[c] == best]
        pick = tied[int(rng.integers(len(tied)))]
        tries += 1
        opened = pick == colour
        rec.add(door, keys[pick], IN_PLACE if opened else NOTHING)
        log.append({"key": pick, "p_in_place": round(float(ps[pick]), 3), "opened": opened})
        untried.remove(pick)
        if opened:
            break
    pairs = {}
    for c in list(keys) + ["nothing"]:
        h = FLOOR if c == "nothing" else keys[c]
        P, _, _ = rec.predict(door, h, vote=vote)
        pairs[f"{colour} door / held {c}"] = right(P, IN_PLACE if c == colour else NOTHING)
    for c in ("red", "green", "blue"):
        P, _, _ = rec.predict(code(("door", c, 0)), keys[colour], vote=vote)
        pairs[f"{c} door / held {colour} key"] = right(P, NOTHING)
    return {"tries": tries, "by_second_try": tries <= 2, "log": log, "pairs_right": pairs,
            "all_pairs_right": all(pairs.values()), "pass": tries <= 2 and all(pairs.values()),
            "confidence_first": round(float(first_conf), 3) if first_conf is not None else None}


def other_events(recs, colour, vote=True):
    """Reported: the new colour's other events, from the key world's memories."""
    key, opn = code(("key", colour)), code(("door", colour, 2))
    q = [("pick up the key", PICK, key, FLOOR, SWAPPED), ("pick it up holding a red key", PICK, key, code(("key", "red")), NOTHING),
         ("drop it", DROP, FLOOR, key, SWAPPED), ("toggle the open door (closes)", TOG, opn, FLOOR, IN_PLACE)]
    out = {}
    for nmk, a, f, h, true in q:
        P, own, kw = recs[("key", a)].predict(f, h, vote=vote)
        out[nmk] = {"right": right(P, true), "p_true": round(float(P[true]), 3), "confidence": round(own + kw, 3)}
    return out


def familiar_confidence(rec):
    """Confidence at familiar doors with their own keys: own counts + votes, and votes alone."""
    own, loo = [], []
    for c in ("red", "green", "blue"):
        P, o, kw = rec.predict(code(("door", c, 0)), code(("key", c)))
        own.append(o + kw)
        _, _, kw2 = rec.predict(code(("door", c, 0)), code(("key", c)), loo=True)
        loo.append(kw2)
    return {"with_own_counts": round(float(np.mean(own)), 2), "votes_alone": round(float(np.mean(loo)), 3)}


def analyse(z, theta, mem, rows, mode, seed, vote=True, parts=None):
    """Criteria 1, 2b and 3 and the reports, for one set of numbers."""
    recs = {(w, a): Recall(z, mem[w][a], theta[j], mode, parts) for j, (w, a) in
            enumerate([(w, a) for w in WORLDS for a in ACTIONS])}
    out = {"criterion_1": {w: evidence(w, rows[w], mem[w][TOG], recs[(w, TOG)]) for w in WORLDS}} if vote else {}
    if vote:
        c1 = out["criterion_1"]
        c1["pass"] = c1["key"]["admitted"] and not c1["switch"]["admitted"]
        out["criterion_2_no_override"] = no_override(recs)
    rng = np.random.default_rng(seed)
    out["criterion_3"] = new_colour_trial(recs[("key", TOG)], "yellow", rng, vote)
    out["purple_reported"] = new_colour_trial(recs[("key", TOG)], "purple", np.random.default_rng(seed), vote)
    if vote:
        out["other_events_yellow"] = other_events(recs, "yellow")
        out["other_events_purple"] = other_events(recs, "purple")
        out["confidence_familiar_doors"] = familiar_confidence(recs[("key", TOG)])
    return out


def label_numbers():
    """Arm 1: one-hot labels (object with state and agent; colour) for every tile code, and its two
    parts."""
    labels = [C.label_of(c) for c in range(len(TILES))]
    objs, cols = sorted({o for o, _ in labels}), sorted({c for _, c in labels})
    z = np.zeros((len(TILES), len(objs) + len(cols)), np.float32)
    for i, (o, c) in enumerate(labels):
        z[i, objs.index(o)] = 1
        z[i, len(objs) + cols.index(c)] = 1
    return z, [slice(0, len(objs)), slice(len(objs), len(objs) + len(cols))]


# ---------------------------------------------------------------- main

def configure():
    closer.MEASURE = "step"
    dl.MAX_GOALS = 10 ** 9
    dl.START_SHARE = 2.0
    dl.Exact = closer.Approach


def main():
    args = sys.argv[1:]
    opt = dict(zip(args[::2], args[1::2])) if len(args) % 2 == 0 else {}
    flag = lambda f: f in args
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    configure()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    assert KR.N_CODES <= 255
    episodes, n_test, heldout = int(get("--episodes", 5000)), int(get("--layouts", 500)), int(get("--heldout", 1000))

    if flag("--collect"):
        res = {"note": "Card 033: experience and memory."}
        out = ROOT / "runs" / "033_memory.json"
        save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")
        data, tiles, pairs = setup(episodes, heldout, n_test, res, save, log)
        mem, rows = memory_all(data)
        CACHE.write_bytes(pickle.dumps({"mem": mem, "rows": rows, "tiles": tiles, "pairs": pairs}))
        res["tiles"] = [C.name_of_code(c) for c in tiles]
        res["memory"] = memory_report(mem, rows)
        save()
        log(f"memory: {json.dumps(res['memory'], default=str)[:2000]}")
        return

    cache = pickle.loads(CACHE.read_bytes())
    mem, rows, tiles, pairs = cache["mem"], cache["rows"], cache["tiles"], cache["pairs"]
    groups = Groups(mem, dev)

    if flag("--sweep"):
        mu = float(get("--sweep", "0"))
        rn = flag("--reseed-new")
        res = {"mu": mu, "reseed_new": rn, "seeds": []}
        out = ROOT / "runs" / f"033_sweep{'_reseed_new' if rn else ''}_{mu}.json"
        for seed in SEEDS:
            ca, rep, z, th = train_encoder(tiles, pairs, groups, mu, seed=seed, dev=dev, reseed_new=rn)
            rep["new_tiles"] = [f"{C.name_of_code(c)} (codebook {k})" for c in tiles for k in range(ca.shape[1]) if ca[c, k] == NEW]
            ok = tiles_ok(ca, tiles)
            res["seeds"].append({"seed": seed, "encoder": rep, **ok})
            log(f"mu {mu} seed {seed}: {rep}; {ok}")
            out.write_text(json.dumps(res, indent=1) + "\n")
        res["qualifies"] = all(s["distinct"] and s["none_new"] for s in res["seeds"])
        out.write_text(json.dumps(res, indent=1) + "\n")
        log(f"mu {mu}: qualifies {res['qualifies']}")
        return

    mu = float(get("--mu", "0"))
    out = Path(get("--out", "runs/033_main.json"))
    res = {"note": "Card 033, tools/card033/recall.py", "mu": mu, "arms": {}}
    save = lambda: out.write_text(json.dumps(res, indent=1, default=str) + "\n")

    if flag("--others"):
        # arm 1: labels
        z1, parts1 = label_numbers()
        th1, ll1 = fit_theta(z1, groups, "rel", dev, parts1)
        res["arms"]["1_labels"] = {"loo_loglik": ll1, **analyse(z1, th1, mem, rows, "rel", SEEDS[0], parts=parts1)}
        log(f"arm 1: c1 {res['arms']['1_labels']['criterion_1']['pass']}, c3 {res['arms']['1_labels']['criterion_3']['pass']}")
        save()
        # arm 3: counting only
        res["arms"]["3_counting"] = {"seeds": []}
        for seed in SEEDS:
            o = analyse(z1, th1, mem, rows, "rel", seed, vote=False, parts=parts1)
            res["arms"]["3_counting"]["seeds"].append({"seed": seed, **o})
        res["arms"]["3_counting"]["criterion_3_seeds"] = sum(s["criterion_3"]["pass"] for s in res["arms"]["3_counting"]["seeds"])
        save()
        # arms 4 (no replay) and 5 (no differences)
        for arm, name in (("4", "4_no_replay"), ("5", "5_no_relation")):
            runs = res["arms"][name] = {"seeds": []}
            for seed in SEEDS:
                if arm == "4":
                    ca, rep, z, _ = train_encoder(tiles, pairs, groups, 0.0, seed=seed, dev=dev)
                    th, ll = fit_theta(z, groups, "rel", dev)
                    mode = "rel"
                else:
                    ca, rep, z, th = train_encoder(tiles, pairs, groups, mu, mode="own", seed=seed, dev=dev)
                    ll, mode = None, "own"
                o = analyse(z, th, mem, rows, mode, seed)
                runs["seeds"].append({"seed": seed, "encoder": rep, "tiles": tiles_ok(ca, tiles), "loo_loglik": ll, **o})
                log(f"arm {arm} seed {seed}: c1 {o['criterion_1']['pass']} c2b {o['criterion_2_no_override']['pass']} "
                    f"c3 {o['criterion_3']['pass']} (tries {o['criterion_3']['tries']}); purple {o['purple_reported']['pass']}")
                save()
            runs["seeds_passing"] = {k: sum(s[k]["pass"] for s in runs["seeds"]) for k in ("criterion_1", "criterion_3")}
            save()
        return

    # --main: arm 2 for the given seeds, with card 031's check (criterion 2a) against arm 3 in the same run
    a, b = get("--seeds", "100-109").split("-")
    seeds = range(int(a), int(b) + 1)
    data, _, _ = setup(episodes, heldout, n_test, res, save, log)
    o3 = C.run_codes("arm 3", APP0[:, None].astype(np.int64), False, data, {}, dev, log)
    ref = {w: o3[w]["acting"]["mean_steps_when_successful"] for w in WORLDS}
    res["arm3_reference"] = {w: o3[w]["acting"] for w in WORLDS}
    runs = res["arms"]["2_main"] = {"seeds": []}
    for seed in seeds:
        ca, rep, z, th = train_encoder(tiles, pairs, groups, mu, seed=seed, dev=dev)
        o = {"seed": seed, "encoder": rep, "tiles": tiles_ok(ca, tiles), "codes": C.codes_report(ca, tiles)}
        o.update(analyse(z, th, mem, rows, "rel", seed))
        oc = C.run_codes(f"arm 2 seed {seed}", ca, True, data, {}, dev, log, ref)
        o["criterion_2_codes"] = {w: oc[w]["criterion_1"] for w in WORLDS}
        o["criterion_2_codes"]["pass"] = all(oc[w]["criterion_1"]["pass"] for w in WORLDS)
        o["familiar_worlds"] = {w: {"heldout_effects_lowest": oc[w]["heldout_effects"]["lowest"],
                                    "acting": oc[w]["acting"]} for w in WORLDS}
        o["criterion_2"] = {"pass": o["criterion_2_codes"]["pass"] and o["criterion_2_no_override"]["pass"]}
        runs["seeds"].append(o)
        log(f"arm 2 seed {seed}: tiles {o['tiles']}; c1 {o['criterion_1']['pass']} "
            f"(key {o['criterion_1']['key']['chosen']}, switch {o['criterion_1']['switch']['chosen']}); "
            f"c2 {o['criterion_2']['pass']}; c3 {o['criterion_3']['pass']} (tries {o['criterion_3']['tries']}, "
            f"pairs {o['criterion_3']['all_pairs_right']}); purple {o['purple_reported']['pass']}")
        save()
    runs["seeds_passing"] = {k: sum(s[k]["pass"] for s in runs["seeds"]) for k in ("criterion_1", "criterion_2", "criterion_3")}
    res["seconds"] = round(time.monotonic() - t00, 1)
    save()
    log(f"done: {runs['seeds_passing']} -> {out}")


if __name__ == "__main__":
    main()
