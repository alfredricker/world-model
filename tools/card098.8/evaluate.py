"""Card 098.8's criteria 1 and 2, on one encoder's vectors (runs/098.8/<arm>/vec_w*.npy and codes.npz), or on card
098.3's (--arm 098.3: runs/098.5/vec_w*.npy and runs/098.3/codes_joint_blocked.npz).

Criterion 1 (workers 8-9): per class (cow, zombie, skeleton; scenery pooled), the share of night tokens (light below
0.3) whose code holds at least 1% of that class's day tokens (light above 0.7); purity by day and at night; also by
ring (distance from the view's centre) and codes per class.

Criterion 2: recall over "tokens like this" (a vote over stored tokens within the encoder's own identity radius, the
prior at one token's weight, as card 098.6's recall) predicts each creature token's next move (stay, one of four
steps, other). Memory: workers 0-6, every creature token with its next move, and scenery tokens (which stay) sampled
and weighted back. Queries: creatures of worker 7 (followed in training) and of workers 8-9 (never seen). The next
moves are the evaluator's: this is a probe of the space; the agent's own recall is criterion 3. Also: what the action
does to the front token (the class at the place faced, one step later), recalled over stored steps with the same
action within the radius, for every action that does not move the agent.

  bin/prun python tools/card098.8/evaluate.py --arm WMC     → runs/098.8/<arm>/eval.json, eval_lp.npz
  bin/prun python tools/card098.8/evaluate.py --compare WMC  → paired bootstrap against 098.3 (eval_compare.json)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PLAY = ROOT / "runs" / "098.5"
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CENTRE = 31
CREATURES = {14: "cow", 15: "zombie", 16: "skeleton", 17: "arrow"}
SCENERY = [c for c in range(1, 13)] + [18]
MOVES = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}       # else 5: "other"
NMOVE = 6
ALPHA = 1.0
RING = np.array([max(abs(k % 9 - 4), abs(k // 9 - 3)) for k in range(63)])


def paths(arm):
    if arm == "098.3":
        return (lambda w: PLAY / f"vec_w{w}.npy"), ROOT / "runs" / "098.3" / "codes_joint_blocked.npz", \
            ROOT / "runs" / "098.8" / "base_098.3"
    d = ROOT / "runs" / "098.8" / arm
    return (lambda w: d / f"vec_w{w}.npy"), d / "codes.npz", d


def codes_for(V, protos):
    P = torch.as_tensor(protos, device=dev, dtype=torch.float32)
    out = []
    for s in range(0, len(V), 200000):
        out.append(torch.cdist(torch.as_tensor(V[s:s + 200000], device=dev, dtype=torch.float32), P).argmin(1).cpu())
    return torch.cat(out).numpy()


def purity(code, label):
    """Share of tokens whose code's majority class is their own."""
    k = code.astype(np.int64) * 64 + label
    u, n = np.unique(k, return_counts=True)
    c, l = u // 64, u % 64
    best = {}
    for ci, li, ni in zip(c, l, n):
        if ni > best.get(ci, (0, -1))[0]:
            best[ci] = (ni, li)
    return float(sum(v[0] for v in best.values()) / len(code))


def day_codes_at_night(code, sem, day, night, classes):
    hit, tot = 0, 0
    for k in classes:
        dc = code[(sem == k) & day]
        nc = code[(sem == k) & night]
        if not len(nc):
            continue
        tot += len(nc)
        if len(dc):
            v, n = np.unique(dc, return_counts=True)
            hit += int(np.isin(nc, v[n >= 0.01 * len(dc)]).sum())
    return hit / max(tot, 1), tot


def criterion1(vec, protos):
    C, S, L, R = [], [], [], []
    for w in (8, 9):
        z = np.load(PLAY / f"play_w{w}.npz")
        V = np.load(vec(w)).astype(np.float32)
        C.append(codes_for(V.reshape(-1, V.shape[-1]), protos))
        S.append(z["sem"].reshape(-1))
        L.append(np.repeat(z["light"], 63))
        R.append(np.tile(RING, len(V)))
    code, sem, light, ring = map(np.concatenate, (C, S, L, R))
    ok = sem >= 0
    ok[np.tile(np.arange(63) == CENTRE, len(sem) // 63)] = False          # the agent's own place
    day, night = ok & (light > 0.7), ok & (light < 0.3)
    out = {"codes": int(len(protos))}
    for k, nm in list(CREATURES.items()) + [(None, "scenery")]:
        cl = SCENERY if k is None else [k]
        share, n = day_codes_at_night(code, sem, day, night, cl)
        out[f"night tokens in a code of their class's day tokens: {nm}"] = [round(share, 4), n]
        if k is None:
            out["scenery, by ring 1-4"] = [round(day_codes_at_night(code, sem, day, night & (ring == r), cl)[0], 4)
                                           for r in (1, 2, 3, 4)]
        else:
            out[f"{nm}, by ring 1-4"] = [round(day_codes_at_night(code, sem, day, night & (ring == r), cl)[0], 4)
                                         for r in (1, 2, 3, 4)]
    out["purity by day / at night"] = [round(purity(code[day], sem[day]), 4), round(purity(code[night], sem[night]), 4)]
    maj = defaultdict(int)
    for c in np.unique(code[ok]):
        v, n = np.unique(sem[ok][code[ok] == c], return_counts=True)
        maj[int(v[n.argmax()])] += 1
    out["codes per class, by majority"] = dict(sorted(maj.items()))
    return out


def next_moves(z):
    """Per (t, place) of a creature with a known next step: its class and next world move's index."""
    obj, pos, ep = z["obj"], z["pos"].astype(int), z["ep"]
    at = defaultdict(dict)
    for t, i, k, dx, dy, fx, fy in obj:
        at[int(t)][int(i)] = (int(k) + 13, (dy + 3) * 9 + (dx + 4), np.array([dx, dy]) + pos[t])
    T, P, K, M = [], [], [], []
    for t, objs in at.items():
        if t + 1 not in at or ep[t] != ep[t + 1]:
            continue
        for i, (k, p, wpos) in objs.items():
            if k in CREATURES and i in at[t + 1]:
                d = tuple((at[t + 1][i][2] - wpos).tolist())
                T.append(t), P.append(p), K.append(k), M.append(MOVES.get(d, 5))
    return np.array(T), np.array(P), np.array(K), np.array(M)


def vote(Q, Mv, Mw, Ml, radius, prior, nclass, Qa=None, Ma=None):
    """Recall: P(class) = (sum of weights within the radius per class + ALPHA * prior) / (sum + ALPHA). With actions,
    only stored items of the query's action count. Returns per-query probability vectors."""
    Mt = torch.as_tensor(Mv, device=dev, dtype=torch.float32)
    onehot = torch.zeros(len(Ml), nclass, device=dev)
    onehot[torch.arange(len(Ml)), torch.as_tensor(Ml, device=dev)] = torch.as_tensor(Mw, device=dev, dtype=torch.float32)
    pr = torch.as_tensor(prior, device=dev, dtype=torch.float32)
    out = []
    for s in range(0, len(Q), 1024):
        q = torch.as_tensor(Q[s:s + 1024], device=dev, dtype=torch.float32)
        m = (torch.cdist(q, Mt) < radius).float()
        if Qa is not None:
            m = m * (torch.as_tensor(Qa[s:s + 1024], device=dev)[:, None] == torch.as_tensor(Ma, device=dev)[None])
        cnt = m @ onehot
        p0 = pr[torch.as_tensor(Qa[s:s + 1024], device=dev)] if pr.dim() == 2 else pr[None]
        out.append(((cnt + ALPHA * p0) / (cnt.sum(1, keepdim=True) + ALPHA)).cpu().numpy())
    return np.concatenate(out)


def criterion2(vec, radius, rng):
    # memory: workers 0-6
    Mv, Mw, Ml = [], [], []
    Fv, Fa, Fl = [], [], []
    for w in range(7):
        z = np.load(PLAY / f"play_w{w}.npz")
        V = np.load(vec(w))
        T, P, K, M = next_moves(z)
        Mv.append(V[T, P]), Mw.append(np.ones(len(T))), Ml.append(M)
        sem = z["sem"].reshape(len(V), 63)
        n = 30000
        t = rng.integers(0, len(V) - 1, n)
        p = rng.integers(0, 63, n)
        s = np.isin(sem[t, p], SCENERY)
        rate = n / (len(V) * 63)
        Mv.append(V[t[s], p[s]]), Mw.append(np.full(int(s.sum()), 1 / rate)), Ml.append(np.zeros(int(s.sum()), int))
        f = front_steps(z)
        Fv.append(V[f[0], f[1]]), Fa.append(f[2]), Fl.append(f[3])
    Mv, Mw, Ml = np.concatenate(Mv), np.concatenate(Mw), np.concatenate(Ml)
    prior = np.bincount(Ml, weights=Mw, minlength=NMOVE) / Mw.sum()
    Fv, Fa, Fl = np.concatenate(Fv), np.concatenate(Fa), np.concatenate(Fl)
    fprior = np.stack([np.bincount(Fl[Fa == a], minlength=64)[:64] + 1e-3 for a in range(17)])
    fprior = fprior / fprior.sum(1, keepdims=True)
    out, lp = {"memory: stored tokens (creatures and sampled scenery), front steps": [int(len(Ml)), int(len(Fl))]}, {}
    for grp, ws in (("worker 7 (followed in training)", [7]), ("workers 8-9 (never seen)", [8, 9])):
        Q, K, L, D, E, QF, FA, FL, FE = [], [], [], [], [], [], [], [], []
        for w in ws:
            z = np.load(PLAY / f"play_w{w}.npz")
            V = np.load(vec(w))
            T, P, Kk, M = next_moves(z)
            Q.append(V[T, P]), K.append(Kk), L.append(M), D.append(z["light"][T]), E.append(z["ep"][T])
            f = front_steps(z)
            QF.append(V[f[0], f[1]]), FA.append(f[2]), FL.append(f[3]), FE.append(z["ep"][f[0]])
        Q, K, L, D, E = map(np.concatenate, (Q, K, L, D, E))
        pr = vote(Q, Mv, Mw, Ml, radius, prior, NMOVE)
        ll = np.log(pr[np.arange(len(L)), L])
        lp[grp] = {"ll": ll, "kind": K, "light": D, "ep": E}
        r = {}
        for tag, m in (("day", D > 0.7), ("night", D < 0.3)):
            r[tag] = {"all creatures": [round(float(ll[m].mean()), 4), int(m.sum())]}
            for k, nm in CREATURES.items():
                mk = m & (K == k)
                if mk.any():
                    r[tag][nm] = [round(float(ll[mk].mean()), 4), int(mk.sum())]
        QF, FA, FL, FE = map(np.concatenate, (QF, FA, FL, FE))
        pf = vote(QF, Fv, np.ones(len(Fl)), Fl, radius, fprior, 64, Qa=FA, Ma=Fa)
        fl = np.log(pf[np.arange(len(FL)), FL])
        lp[grp]["front ll"], lp[grp]["front ep"] = fl, FE
        r["front token after the action: mean log-likelihood, steps"] = [round(float(fl.mean()), 4), int(len(fl))]
        out[grp] = r
    return out, lp


def front_steps(z):
    """Awake steps whose action does not move the agent: (t, place faced, action, class at that place at t+1)."""
    act, sem, face, sleep, ep, done = z["act"], z["sem"].reshape(len(z["act"]), 63), z["face"], z["sleep"], z["ep"], \
        z["done"]
    t = np.flatnonzero((~np.isin(act, [1, 2, 3, 4])) & ~sleep & ~done & (np.r_[ep[1:] == ep[:-1], False]))
    p = (3 + face[t, 1]) * 9 + (4 + face[t, 0])
    lab = sem[t + 1, p].astype(np.int64)
    ok = lab >= 0
    return t[ok], p[ok], act[t[ok]].astype(np.int64), lab[ok]


def compare(arm, base="098.3", n=2000, seed=0):
    """Paired bootstrap over episodes: the arm's log-likelihood minus 098.3's, on the same queries."""
    rng = np.random.default_rng(seed)
    a = np.load(paths(arm)[2] / "eval_lp.npz", allow_pickle=True)["lp"].item()
    b = np.load(paths(base)[2] / "eval_lp.npz", allow_pickle=True)["lp"].item()
    out = {}
    for grp in a:
        for tag, key, ep, sel in (("moves, day", "ll", "ep", lambda g: g["light"] > 0.7),
                                  ("moves, night", "ll", "ep", lambda g: g["light"] < 0.3),
                                  ("front token", "front ll", "front ep", lambda g: np.ones(len(g["front ll"]), bool))):
            m = sel(a[grp])
            d, e = a[grp][key][m] - b[grp][key][m], a[grp][ep][m]
            eps = np.unique(e)
            idx = {x: np.flatnonzero(e == x) for x in eps}
            boots = []
            for _ in range(n):
                pick = rng.choice(eps, len(eps))
                ii = np.concatenate([idx[x] for x in pick])
                boots.append(d[ii].mean())
            boots = np.array(boots)
            out[f"{grp}: {tag}"] = {"mean difference (nats)": round(float(d.mean()), 4),
                                    "share of bootstrap means at or below 0 / at or above 0":
                                        [round(float((boots <= 0).mean()), 4), round(float((boots >= 0).mean()), 4)]}
    (paths(arm)[2] / "eval_compare.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main():
    if arg("--compare", None):
        return compare(arg("--compare", None))
    arm = arg("--arm", "WMC")
    vec, codes, out = paths(arm)
    out.mkdir(parents=True, exist_ok=True)
    c = np.load(codes)
    rng = np.random.default_rng(9888)
    rep = {"arm": arm, "radius": round(float(c["radius"]), 4), "criterion 1": criterion1(vec, c["protos"])}
    print(json.dumps(rep, indent=1), flush=True)
    rep["criterion 2"], lp = criterion2(vec, float(c["radius"]), rng)
    np.savez_compressed(out / "eval_lp.npz", lp=np.array(lp, dtype=object))
    (out / "eval.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep["criterion 2"], indent=1))


if __name__ == "__main__":
    main()
