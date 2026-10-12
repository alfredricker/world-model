"""Card 102.2: tokens of many tiles from common fate.

Each step's change from the view to the next view is explained by layers (Wang and Adelson 1994):
  - the background, shifted by the view's own motion (the shift that keeps most codes in place);
  - chains: tiles that shift by one step relative to the background. A chain starts where a tile appears that was
    beside it one step before, and follows the shift backward and forward until the ground: where a mover left, the
    ground shows again, and where it arrived, ground was covered. Ground is learned: the codes seen revealed and
    covered when exactly one tile moves (the agent's own steps, mostly). A tile that is not ground cannot simply
    vanish, so the inner tiles of a moved bar of one colour join its chain (the aperture problem);
  - tiles changed in place: what is left.
Each tile is tracked by identity through these explanations. Its fate in a step is one of: still, a shift (four),
changed in place. Two tracked tiles both present gain evidence of one object when they share a fate other than
still, log((1 - eps) / p), and lose it when exactly one of them moves or they move differently, log(eps / 2p(1-p)),
with p the share of a mover's steps in which it moves (training) and eps = 0.01. A group is the tiles linked by
positive evidence (connected components); every other tile is a group of one. Nothing about shape, colour or
adjacency is declared.

  bin/prun python tools/card102.2/layers.py world2|crafter     → runs/102.2/layers_<world>.json, groups_<world>.npz
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "102.2"
R, C = 7, 9                                            # the view's rows and columns of tiles (carried rows left out)
E4 = ((1, 0), (-1, 0), (0, 1), (0, -1))                # shifts (dx, dy); fate 1..4; fate 5 = changed in place
EPS = 0.01


def load(world, split):
    if world == "world2":
        z = np.load(ROOT / "runs" / "card102" / f"world2_{split}.npz")
        code = np.load(RUNS / f"world2_code_{split}.npy")[:, :63].reshape(-1, R, C).astype(np.int64)
        vec = np.load(RUNS / f"world2_vec_{split}.npy", mmap_mode="r")
        seg = z["ep"].astype(np.int64)
        return {"code": code, "vec": vec, "seg": seg, "act": z["act"], "sem": z["sem"], "obj": z["obj"]}
    ws = range(8) if split == "train" else (8, 9)
    out = {k: [] for k in ("code", "vec", "seg", "act", "sem", "pos")}
    base = 0
    for w in ws:
        z = np.load(ROOT / "runs" / "102.1" / f"curr_w{w}.npz")
        n = len(z["act"])
        # a segment: an episode's awake run (asleep, the view is dark)
        brk = np.r_[True, (z["ep"][1:] != z["ep"][:-1]) | z["sleep"][1:] | z["sleep"][:-1]]
        out["seg"].append(base + np.cumsum(brk))
        base = out["seg"][-1][-1] + 1
        out["code"].append(np.load(ROOT / "runs" / "102.1" / f"curr_code_w{w}.npy")[:, :63].reshape(n, R, C).astype(np.int64))
        out["vec"].append(np.load(ROOT / "runs" / "102.1" / f"curr_vec_w{w}.npy")[:, :63] if split == "test"
                          else np.zeros((n, 0), np.float16))
        out["act"].append(z["act"])
        out["sem"].append(z["sem"])
        out["pos"].append(z["pos"])
    d = {k: np.concatenate(v) for k, v in out.items()}
    sleep = np.concatenate([np.load(ROOT / "runs" / "102.1" / f"curr_w{w}.npz")["sleep"] for w in ws])
    d["seg"][sleep] = -1
    return d


def bg_shift(c0, c1):
    """The shift (dx, dy) of the view's content that keeps most codes: c1[y, x] = c0[y - dy, x - dx]."""
    best, bd = -1.0, (0, 0)
    for d in ((0, 0),) + E4:
        dx, dy = d
        a = c1[max(0, dy):R + min(0, dy), max(0, dx):C + min(0, dx)]
        b = c0[max(0, -dy):R + min(0, -dy), max(0, -dx):C + min(0, -dx)]
        s = (a == b).mean()
        if s > best + 1e-9:
            best, bd = s, d
    return bd


def aligned(c0, d, fill=-1):
    """c0 moved by the background's shift d: prev[y, x] = c0[y - dy, x - dx], fill where it left the view."""
    dx, dy = d
    out = np.full(c0.shape, fill, c0.dtype) if c0.ndim == 2 else np.zeros(c0.shape, c0.dtype)
    out[max(0, dy):R + min(0, dy), max(0, dx):C + min(0, dx)] = c0[max(0, -dy):R + min(0, -dy), max(0, -dx):C + min(0, -dx)]
    return out


def inview(y, x):
    return 0 <= y < R and 0 <= x < C


def learn_ground(d):
    """Ground: the codes revealed and covered when exactly one tile moves one step relative to the background. Of the
    two tiles that swap places in such a step, the mover is the one whose code is the more often seen to change
    (relative to the background) wherever it stands; the other is ground."""
    seen, chg, swaps, seg = {}, {}, [], d["seg"]
    for t in range(len(seg) - 1):
        if seg[t] < 0 or seg[t] != seg[t + 1]:
            continue
        c0, c1 = d["code"][t], d["code"][t + 1]
        bd = bg_shift(c0, c1)
        prev = aligned(c0, bd)
        ok = prev >= 0
        changed = (c1 != prev) & ok
        for k, v in zip(*np.unique(prev[ok], return_counts=True)):
            seen[int(k)] = seen.get(int(k), 0) + int(v)
        for k, v in zip(*np.unique(prev[changed], return_counts=True)):
            chg[int(k)] = chg.get(int(k), 0) + int(v)
        ch = np.argwhere(changed)
        if len(ch) != 2 or abs(ch[0] - ch[1]).sum() != 1:
            continue
        for a, b in ((tuple(ch[0]), tuple(ch[1])), (tuple(ch[1]), tuple(ch[0]))):
            if c1[b] == prev[a]:                          # a tile like the one at a now stands at b
                swaps.append((int(prev[a]), int(c1[a]), int(prev[b])))   # mover, revealed at a, covered at b
    rate = {k: chg.get(k, 0) / v for k, v in seen.items()}
    cnt, mover, n = {}, {}, 0
    for m, rev, cov in swaps:
        if rate[m] > max(rate.get(rev, 0), rate.get(cov, 0)):
            for k in (rev, cov):
                cnt[k] = cnt.get(k, 0) + 1
            mover[m] = mover.get(m, 0) + 1
            n += 1
    ground = sorted(k for k, v in cnt.items() if v >= 3)
    return ground, {"single-tile moves": n, "revealed or covered, by code": dict(sorted(cnt.items(), key=lambda kv: -kv[1])),
                    "moving, by code": dict(sorted(mover.items(), key=lambda kv: -kv[1])),
                    "share of a code's tiles that change, ground": {k: round(rate[k], 4) for k in ground}}


def explain(c0, c1, v0, v1, ground):
    """One step: the background's shift, the chains and the tiles changed in place.

    Returns bd, fate1 (R, C) per position of the next view (0 still, 1..4 arrived by that shift, 5 changed in place,
    6 new: revealed ground, appeared on ground or newly in view, 7 the tile here vanished and ground shows), src
    (R, C, 2) the position in the view one step before that each next tile came from (-1 where new)."""
    bd = bg_shift(c0, c1)
    prev = aligned(c0, bd)
    pv = aligned(v0, bd) if v0 is not None and v0.size else None
    changed = (c1 != prev) & (prev >= 0)
    isg = np.isin(c1, ground)
    pg = np.isin(prev, ground)

    def match(y, x, ys, xs):
        """The next view's tile at (y, x) is the tile at aligned (ys, xs) one step before, moved."""
        if not inview(ys, xs) or prev[ys, xs] < 0 or isg[y, x]:
            return False
        if c1[y, x] == prev[ys, xs]:
            return True
        if pv is None or prev[y, x] < 0:                  # by vectors: closer to it than to what stood here
            return False
        a = v1[y, x].astype(np.float32)
        return np.linalg.norm(a - pv[ys, xs]) < np.linalg.norm(a - pv[y, x])

    def chain(y, x, k):
        dx, dy = E4[k]
        mem = [(y, x)]
        ys, xs = y - dy, x - dx                           # backward: where the arrival came from
        while True:
            if isg[ys, xs]:                               # ground shows where the mover left
                break
            if match(ys, xs, ys - dy, xs - dx):
                mem.append((ys, xs))
                ys, xs = ys - dy, xs - dx
                continue
            break
        yf, xf = y, x                                     # forward: what stood where it arrived
        while True:
            if pg[yf, xf] or not inview(yf + dy, xf + dx):
                break
            if match(yf + dy, xf + dx, yf, xf):
                mem.append((yf + dy, xf + dx))
                yf, xf = yf + dy, xf + dx
                continue
            break
        tail = (ys, xs)
        return mem, tail

    fate = np.zeros((R, C), np.int64)
    src = np.stack(np.meshgrid(np.arange(R), np.arange(C), indexing="ij"), -1) - np.array([bd[1], bd[0]])
    new = np.argwhere(prev < 0)
    fate[new[:, 0], new[:, 1]] = 6
    src[new[:, 0], new[:, 1]] = -1
    todo = changed.copy()
    while todo.any():
        cands = []
        for y, x in np.argwhere(todo):
            for k, (dx, dy) in enumerate(E4):
                if match(y, x, y - dy, x - dx):
                    mem, tail = chain(y, x, k)
                    if inview(*tail) and not (isg[tail] or changed[tail]):
                        continue                          # the mover is still where it came from: not a move
                    gain = sum(todo[a, b] for a, b in mem) + (todo[tail] if inview(*tail) and isg[tail] else 0)
                    cands.append((gain, -len(mem), k, mem, tail))
        if not cands:
            break
        gain, _, k, mem, tail = max(cands, key=lambda c: (c[0], c[1]))
        if gain <= 0:
            break
        dx, dy = E4[k]
        for a, b in mem:
            fate[a, b] = 1 + k
            src[a, b] = (a - dy - bd[1], b - dx - bd[0])
            todo[a, b] = False
        if inview(*tail) and isg[tail] and fate[tail] == 0:
            fate[tail] = 6
            src[tail] = -1
            todo[tail] = False
    for y, x in np.argwhere(todo):
        if isg[y, x] and not pg[y, x]:                    # it vanished and ground shows: fate 7, a new ground tile
            fate[y, x] = 7
        elif pg[y, x] and not isg[y, x]:                  # it appeared on ground: a new tile
            fate[y, x] = 6
            src[y, x] = -1
        else:
            fate[y, x] = 5                                # changed in place: the same tile, now different
    # a tile is in one place: where two next positions claim one source, the mover keeps it and the other is new
    seen = {}
    for y, x in sorted(np.argwhere(src[..., 0] >= 0).tolist(), key=lambda q: not 1 <= fate[q[0], q[1]] <= 4):
        key = (src[y, x, 0], src[y, x, 1])
        if key in seen:
            fate[y, x], src[y, x] = 6, -1
        else:
            seen[key] = True
    return bd, fate, src


def analyse(d, ground, p_move=None):
    """Every segment: identities per state (n, R, C) and, per step, the fate of each position's tile one step on."""
    n = len(d["seg"])
    ident = np.full((n, R, C), -1, np.int64)
    fate0 = np.full((n, R, C), -1, np.int64)              # the fate of the tile at each position, in the step t → t+1
    bgs = np.zeros((n, 2), np.int64)
    nid = 0
    seg = d["seg"]
    has_v = d["vec"].shape[1] > 0 if d["vec"].ndim > 1 else False
    for t in range(n):
        if seg[t] < 0:
            continue
        if t == 0 or seg[t - 1] != seg[t]:
            ident[t] = np.arange(nid, nid + R * C).reshape(R, C)
            nid += R * C
        if t + 1 >= n or seg[t + 1] != seg[t]:
            continue
        v0 = np.asarray(d["vec"][t][:R * C]).reshape(R, C, -1) if has_v else None
        v1 = np.asarray(d["vec"][t + 1][:R * C]).reshape(R, C, -1) if has_v else None
        bd, f1, src = explain(d["code"][t], d["code"][t + 1], v0, v1, ground)
        bgs[t] = bd
        for y in range(R):
            for x in range(C):
                sy, sx = src[y, x]
                if f1[y, x] == 7:                         # the tile vanished; ground is a new tile
                    fate0[t, sy, sx] = 7
                if f1[y, x] in (6, 7) or sy < 0:
                    ident[t + 1, y, x] = nid
                    nid += 1
                else:
                    ident[t + 1, y, x] = ident[t, sy, sx]
                    fate0[t, sy, sx] = f1[y, x] if f1[y, x] in (1, 2, 3, 4, 5) else 0
    if p_move is None:
        f = fate0[np.isin(ident, np.unique(ident[fate0 >= 1]))]
        f = f[(f >= 0) & (f <= 5)]
        p_move = float((f >= 1).mean())
    return {"ident": ident, "fate0": fate0, "bg": bgs, "p_move": p_move, "seg": seg}


def evidence(A):
    """Per segment: the participants (tiles that ever moved or changed in place), their fates per step and the
    cumulative evidence of one object between every two of them (without the prior)."""
    p = A["p_move"]
    a, b = np.log((1 - EPS) / p), np.log(EPS / (2 * p * (1 - p)))
    seg, ident, fate0 = A["seg"], A["ident"], A["fate0"]
    out = []
    for s in np.unique(seg[seg >= 0]):
        ts = np.flatnonzero(seg == s)
        I, F = ident[ts], fate0[ts]
        ids = np.unique(I[(F >= 1) & (F <= 5)])
        if len(ids) < 2:
            continue
        T = len(ts)
        Fi = np.full((len(ids), T), -1, np.int64)        # each participant's fate per step (-1: absent or vanished)
        idx = {v: k for k, v in enumerate(ids)}
        for t in range(T):
            m = np.isin(I[t], ids) & (F[t] >= 0) & (F[t] <= 5)
            for v, f in zip(I[t][m], F[t][m]):
                Fi[idx[v], t] = f
        both = (Fi[:, None] >= 0) & (Fi[None] >= 0)
        co = both & (Fi[:, None] == Fi[None]) & (Fi[:, None] >= 1)
        split = both & (Fi[:, None] != Fi[None])
        out.append((ts, ids, np.cumsum(a * co + b * split, 2), both.any(2)))
    return out, (round(float(a), 3), round(float(b), 3))


def fit_prior(ev, iters=20):
    """The prior log-odds that two participants are one object: the share of pairs seen together that end their
    segment as one object, found as a fixed point (empirical Bayes) on the training play."""
    prior = 0.0
    for _ in range(iters):
        k = n = 0
        for ts, ids, E, seen in ev:
            iu = np.triu_indices(len(ids), 1)
            m = seen[iu]
            k += int(((E[:, :, -1][iu] + prior) > 0)[m].sum())
            n += int(m.sum())
        share = (k + 1) / (n + 2)                         # Laplace's rule: never exactly 0 or 1
        new = float(np.log(share / (1 - share)))
        if abs(new - prior) < 1e-3:
            break
        prior = new
    return prior, share


def groups(A, ev, prior):
    """Per state, each position's group: participants linked by positive evidence (with the prior) are one group."""
    group = A["ident"].copy()
    ident = A["ident"]
    for ts, ids, E, _ in ev:
        I = ident[ts]
        for t in range(1, len(ts)):
            pos = E[:, :, t - 1] + prior > 0
            np.fill_diagonal(pos, False)
            present = np.isin(ids, I[t])
            pos &= present[:, None] & present[None]
            if not pos.any():
                continue
            comp = np.arange(len(ids))                    # connected components by union-find
            for i, j in np.argwhere(np.triu(pos)):
                ri, rj = i, j
                while comp[ri] != ri:
                    ri = comp[ri]
                while comp[rj] != rj:
                    rj = comp[rj]
                if ri != rj:
                    comp[max(ri, rj)] = min(ri, rj)
            for k in range(len(ids)):
                r = k
                while comp[r] != r:
                    r = comp[r]
                comp[k] = r
            lut = dict(zip(ids, ids[comp]))
            g = group[ts[t]]
            for (y, x), v in np.ndenumerate(I[t]):
                if v in lut:
                    g[y, x] = lut[v]
    return group


# ------------------------------------------------------------------------------------------------ evaluation
def true_fates(d):
    """World 2's evaluator: each tile's fate one step on (0 still, 1..4 shift, 5 changed in place)."""
    obj, sem, seg = d["obj"].astype(np.int64), d["sem"].astype(np.int64), d["seg"]
    n = len(seg)
    F = np.full((n, R, C), -1, np.int64)
    for t in range(n - 1):
        if seg[t] != seg[t + 1]:
            continue
        F[t] = 0
        for o in np.unique(obj[t]):
            if o == 0:
                continue
            a = np.argwhere(obj[t] == o)
            b = np.argwhere(obj[t + 1] == o)
            if len(a) == len(b) and len(a):
                dyx = b.min(0) - a.min(0)
                if (dyx != 0).any() and set(map(tuple, a + dyx)) == set(map(tuple, b)):
                    F[t][a[:, 0], a[:, 1]] = 1 + E4.index((int(dyx[1]), int(dyx[0])))
                elif (sem[t][a[:, 0], a[:, 1]] != sem[t + 1][a[:, 0], a[:, 1]]).any():
                    F[t][a[:, 0], a[:, 1]] = 5
    return F


def colour_touch(code, ground):
    """The usual ARC baseline: tiles of one code that touch (4-neighbours) are one object; ground stays apart."""
    n = len(code)
    lab = np.full((n, R, C), -1, np.int64)
    for t in range(n):
        c, k = code[t], 0
        for y in range(R):
            for x in range(C):
                if lab[t, y, x] >= 0:
                    continue
                st = [(y, x)]
                lab[t, y, x] = k
                while st:
                    a, b = st.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = a + dy, b + dx
                        if inview(u, v) and lab[t, u, v] < 0 and c[u, v] == c[y, x] and c[y, x] not in ground:
                            lab[t, u, v] = k
                            st.append((u, v))
                k += 1
    return lab


def objects_exact(d, G, started_only=True):
    """Criterion 1: the share of (state, object of two or more tiles) whose tiles are exactly one group, counted
    from the state after the object first moved or changed (started_only) or before it (first sight)."""
    obj, seg = d["obj"].astype(np.int64), d["seg"]
    TF = true_fates(d)
    out = {"after": [0, 0], "before": [0, 0]}
    by_kind = {}
    started = {}
    for t in range(len(seg)):
        if t == 0 or seg[t] != seg[t - 1]:
            started = {}
        o = obj[t]
        for k in np.unique(o):
            if k == 0 or k == 99:
                continue
            m = o == k
            if m.sum() < 2:
                continue
            g = G[t][m]
            ok = (g == g[0]).all() and (G[t] == g[0]).sum() == m.sum()
            key = "after" if started.get(k) else "before"
            out[key][0] += ok
            out[key][1] += 1
            if key == "after":
                kind = "door" if k == 1 else ("shape, one colour" if len(np.unique(d["sem"][t][m])) == 1
                                              else "shape, several colours")
                by_kind.setdefault(kind, [0, 0])
                by_kind[kind][0] += ok
                by_kind[kind][1] += 1
        if t + 1 < len(seg) and seg[t + 1] == seg[t]:
            for k in np.unique(o[TF[t] >= 1]):
                started[k] = True
    r = {k: {"right": round(v[0] / max(1, v[1]), 4), "of": v[1]} for k, v in out.items()}
    r["after, by kind"] = {k: {"right": round(v[0] / max(1, v[1]), 4), "of": v[1]} for k, v in by_kind.items()}
    return r


def main():
    world = sys.argv[1]
    t0 = time.monotonic()
    tr, te = load(world, "train"), load(world, "test")
    if "--n" in sys.argv:                                 # a smoke test on the first n steps of each split
        k = int(sys.argv[sys.argv.index("--n") + 1])
        tr, te = ({key: v[:k] for key, v in x.items()} for x in (tr, te))
    ground, gstat = learn_ground(tr)
    print("ground codes", ground, gstat, f"({time.monotonic() - t0:.0f} s)", flush=True)
    A_tr = analyse(tr, ground)
    ev_tr, ab = evidence(A_tr)
    prior, share = fit_prior(ev_tr)
    A = analyse(te, ground, p_move=A_tr["p_move"])
    ev, _ = evidence(A)
    A["group"] = groups(A, ev, prior)
    print("analysed", f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep = {"ground codes": ground, "ground": gstat, "p (a mover's steps that move)": round(A_tr["p_move"], 4),
           "evidence for, against (nats)": ab, "prior log-odds of one object (training)": round(prior, 3),
           "share of pairs seen together that are one object (training)": round(share, 4)}
    if world == "world2":
        TF = true_fates(te)
        m = (TF >= 0) & (te["obj"] > 0)
        f = A["fate0"]
        movers = m & (TF >= 1)
        rep["gate: fates of moving or changing object tiles right"] = round(float((f[movers] == TF[movers]).mean()), 5)
        rep["gate: object tiles that stay, called still"] = round(float((f[m & (TF == 0)] == 0).mean()), 5)
        rep["moving or changing object tiles"] = int(movers.sum())
        rep["criterion 1: fate groups"] = objects_exact(te, A["group"])
        rep["criterion 1: colour and touching"] = objects_exact(te, colour_touch(te["code"], ground))
        np.savez_compressed(RUNS / "groups_world2_test.npz", group=A["group"], ident=A["ident"], fate0=A["fate0"])
        np.savez_compressed(RUNS / "groups_world2_train.npz", group=groups(A_tr, ev_tr, prior), ident=A_tr["ident"],
                            fate0=A_tr["fate0"])
    else:
        seg, G, I = te["seg"], A["group"], A["ident"]
        ok = seg >= 0
        multi = np.zeros(len(seg), np.int64)
        tiles_in = np.zeros(len(seg), np.int64)
        part = np.zeros(len(seg), np.int64)
        moved_ids = set(np.unique(I[:-1][(A["fate0"][:-1] >= 1) & (A["fate0"][:-1] <= 5)]).tolist())
        for t in np.flatnonzero(ok):
            g = G[t].ravel()
            u, c = np.unique(g, return_counts=True)
            multi[t] = (c >= 2).sum()
            tiles_in[t] = c[c >= 2].sum()
            part[t] = sum(1 for v in I[t].ravel() if v in moved_ids)
        # the view's shift against the agent's own move (evaluator), steps within a segment
        pos = te["pos"].astype(np.int64)
        st = np.flatnonzero(ok[:-1] & (seg[:-1] == seg[1:]))
        mv = pos[st + 1] - pos[st]
        rep["view shift = minus the agent's move (evaluator)"] = round(float(
            ((A["bg"][st] == -mv).all(1)).mean()), 4)
        rep["states"] = int(ok.sum())
        rep["criterion 3: states with a group of two or more tiles (all false in Crafter)"] = round(float((multi[ok] > 0).mean()), 4)
        rep["criterion 3: tiles in such groups / tracked tiles that ever moved or changed"] = round(
            float(tiles_in[ok].sum() / max(1, part[ok].sum())), 4)
        rep["tracked tiles that ever moved or changed, per state"] = round(float(part[ok].mean()), 2)
        f = A["fate0"][ok]
        rep["fates per state: moved, changed in place, vanished"] = [
            round(float(np.isin(f, k).sum() / ok.sum()), 3) for k in ((1, 2, 3, 4), (5,), (7,))]
        # how long false links last: runs of states in which an identity is in a group of two or more
        runs, cur = [], {}
        for t in np.flatnonzero(ok):
            g = G[t].ravel()
            u, c = np.unique(g, return_counts=True)
            inm = set(I[t].ravel()[np.isin(g, u[c >= 2])].tolist())
            for v in list(cur):
                if v not in inm:
                    runs.append(cur.pop(v))
            for v in inm:
                cur[v] = cur.get(v, 0) + 1
        runs += list(cur.values())
        rep["false links: states a tile stays in a group (median, 90th percentile)"] = (
            [float(np.median(runs)), float(np.quantile(runs, 0.9))] if runs else None)
        np.savez_compressed(RUNS / "groups_crafter_test.npz", group=A["group"], ident=A["ident"], fate0=A["fate0"])
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (RUNS / f"layers_{world}{'_smoke' if '--n' in sys.argv else ''}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
