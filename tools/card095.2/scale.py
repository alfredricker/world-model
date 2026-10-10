"""Card 095.2: recall at scale, offline, on card 095.1's arm C (arm B's vote, arm A as its prior).

Per world and action, against arm C's full vote (B reads every stored group; 095.1's saved predictions):
  gate    a head on arm A's token states, trained on training episodes (095.1's leave-one-try-out sample), says
          which tokens B could change A's answer for (|P_C - P_A| > 0.01, or a change A gets the wrong result for);
          the rest take A's answer. Its threshold passes 99.9% of the training positives.
  index   B's vote over the stored groups within radius R of the query (radius.py), exact within R.
  forget  stored groups with no change, whose neighbours (the group itself left out) predict a change below 0.02
          on at least as much weight as the group's own, are dropped; then every dropped group near a stored
          group whose prediction moved by more than 0.01 is restored. Changed instances are kept, merged when
          equal. Raw tries are kept only for surprises (against their group's majority) and a 2% sample.
Configurations: full, gate, index, forget, all. Scored on held-out episodes (095.1's metrics), tier 2's
counterfactual boxes, a one-try test, time per query (one CPU thread) and memory size (RAM and disk).

  bin/prun python tools/card095.2/scale.py tier2      → runs/095.2/scale_tier2.json
"""
import ast
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
sys.path.insert(0, str(ROOT / "tools" / "card095.2"))
import tok                                             # noqa: E402
import arm_a as AA                                     # noqa: E402
import arm_b as AB                                     # noqa: E402
import radius as RD                                    # noqa: E402

WORLD = sys.argv[1] if len(sys.argv) > 1 else None
OUT = ROOT / "runs" / "095.2"
R = 12.0
DELTA = 0.01
FORGET_P = 0.02
MOVED = 0.01
RAW_SHARE = 0.02
EPS = 1e-6
dev = AA.dev
BOXES = {"blue": 180, "red": 170, "grey": 190, "yellow": 185, "green": 175, "purple": 85}
CONFIGS = ("full", "gate", "index", "forget", "all")
log = lambda m: print(m, flush=True)


# ---------------------------------------------------------------- arm A

@torch.no_grad()
def a_states(net, arr_t, known, H, P, act, bs=4096):
    """Per try and slot: A's P(change), snapped result and token state h (the gate's input)."""
    K = arr_t[torch.as_tensor(known, device=dev)]
    pa, aa, hh = [], [], []
    for s in range(0, len(H), bs):
        Ht = torch.as_tensor(H[s:s + bs], device=dev)
        mask = Ht >= 0
        X = arr_t[Ht.clamp_min(0)] * mask[..., None]
        Pt = torch.as_tensor(P[s:s + bs], device=dev)
        at = torch.as_tensor(act[s:s + bs], device=dev)
        h = net.att(net.inp(X) + net.place(Pt) + net.act(at)[:, None], src_key_padding_mask=~mask)
        logit, after, _, _ = net(X, Pt, mask, at)
        snap = known[torch.cdist(after.reshape(-1, 32), K, p=1).argmin(1).cpu().numpy()].reshape(after.shape[:2])
        pa.append(torch.sigmoid(logit).cpu().numpy())
        aa.append(snap)
        hh.append(h.half().cpu().numpy())
    return np.concatenate(pa), np.concatenate(aa), np.concatenate(hh)


class Gate(torch.nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

    def forward(self, h):
        return self.f(h)[..., 0]


def train_gate(h, y, steps=800):
    torch.manual_seed(95)
    g = Gate().to(dev)
    opt = torch.optim.Adam(g.parameters(), lr=3e-3)
    ht = torch.as_tensor(np.asarray(h, np.float32), device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    pw = torch.as_tensor(max(1.0, (1 - y.mean()) / max(y.mean(), 1e-4)), device=dev)
    for _ in range(steps):
        l = torch.nn.functional.binary_cross_entropy_with_logits(g(ht), yt, pos_weight=pw)
        opt.zero_grad()
        l.backward()
        opt.step()
    sc = gate_scores(g, h)
    tau = float(np.quantile(sc[y > 0], 0.001)) if (y > 0).any() else float("inf")
    return g, tau


@torch.no_grad()
def gate_scores(g, h):
    return g(torch.as_tensor(np.asarray(h, np.float32), device=dev)).cpu().numpy()


# ---------------------------------------------------------------- arm B, rebuilt from 095.1's admitted conditions

def rebuild_arm(arr, Fs, ws, chs, trace):
    u, inv = np.unique(Fs, axis=0, return_inverse=True)
    inv = inv.ravel()
    b = lambda x: np.bincount(inv, weights=x, minlength=len(u))
    arm = AB.Arm(arr, u, b(ws * chs), b(ws), b(chs.astype(float)), b((~chs).astype(float)))
    arm.Vt, arm._gc = torch.as_tensor(arm.V, device=dev), {}
    arm.adm = [ast.literal_eval(x["cond"]) for x in trace]
    arm.lam = [x["lambda"] for x in trace]
    arm.G, arm.Gnc, arm.Gn, _, _ = arm.groups([arm.col(c) for c in arm.adm])
    return arm


def subset(arm, keep):
    k = copy.copy(arm)
    k.G, k.Gnc, k.Gn = arm.G[keep], arm.Gnc[keep], arm.Gn[keep]
    return k


def relation_result(V, q, Fi, ai, wk, known, arr):
    """card 095.1's rule (arm_b.results) for one query over the given changed instances and their weights."""
    top = np.argsort(-wk)[:AB.TOPK]
    top = top[wk[top] > 0]
    if not len(top):
        return -1
    after = arr[ai[top]]
    res = np.zeros(32, np.float32)
    for k in range(AB.NP):
        sl = slice(k * AB.PW, (k + 1) * AB.PW)
        a, own = after[:, sl], V[Fi[top, 0]][:, sl]
        score = {("keep",): wk[top][np.abs(a - own).sum(1) < 1e-5].sum()}
        for r in range(Fi.shape[1] - 2):
            cr = V[Fi[top, 2 + r]][:, sl]
            score[("copy", r)] = wk[top][(np.abs(a - cr).sum(1) < 1e-5) & (Fi[top, 2 + r] >= 0)].sum()
        best = {}
        for vals, kind in ((a - own, "shift"), (a, "set")):
            u, inv = np.unique(np.round(vals, 5), axis=0, return_inverse=True)
            sw = np.bincount(inv.ravel(), weights=wk[top], minlength=len(u))
            j = int(sw.argmax())
            score[(kind,)] = sw[j]
            best[kind] = u[j]
        way = max(score, key=score.get)
        if way[0] == "keep":
            res[sl] = V[q[0]][sl]
        elif way[0] == "copy":
            res[sl] = V[q[2 + way[1]]][sl]
        elif way[0] == "shift":
            res[sl] = V[q[0]][sl] + best["shift"]
        else:
            res[sl] = best["set"]
    return int(known[np.abs(arr[known] - res).sum(1).argmin()])


class Store:
    """Stored groups (counts) and changed instances, read through the radius index."""

    def __init__(self, arm, Fch, ach, wch):
        self.arm = arm
        self.rad = RD.Radius(arm, R)
        cols = self.rad.cols
        gi = np.array([self.rad.key.get(tuple(r), -1) for r in Fch[:, cols].tolist()], np.int64)
        assert (gi >= 0).all()                         # every changed instance lies in a kept group
        self.Fch, self.ach, self.wch = Fch, ach, wch
        self.by_group = {}
        for i, g in enumerate(gi.tolist()):
            self.by_group.setdefault(g, []).append(i)

    def vote(self, f):
        gi, d = self.rad.query(f)
        wgt = np.exp(-d)
        return float(wgt @ self.arm.Gnc[gi]), float(wgt @ self.arm.Gn[gi]), gi, wgt

    def result(self, f, gi, wgt, known, arr):
        idx, wk = [], []
        for g, wg in zip(gi.tolist(), wgt.tolist()):
            for i in self.by_group.get(g, ()):
                idx.append(i)
                wk.append(wg * self.wch[i])
        if not idx:
            return -1
        idx = np.array(idx)
        return relation_result(self.arm.V, f, self.Fch[idx], self.ach[idx], np.array(wk), known, arr)


def forget(arm):
    """Drop no-change groups their neighbours already predict; restore those whose loss moves a prediction."""
    Nc, N = arm.p_change(arm.G, totals=True)[1].T
    loo_N = N - arm.Gn                                 # the group's own kernel weight is 1
    drop = (arm.Gnc == 0) & (loo_N >= arm.Gn) & (Nc / np.maximum(loo_N, 1e-12) < FORGET_P)
    p0 = arm.p_change(arm.G)
    rad = RD.Radius(arm, R)
    restored = 0
    for _ in range(3):
        p1 = subset(arm, ~drop).p_change(arm.G)
        moved = np.flatnonzero(np.abs(p1 - p0) > MOVED)
        back = set()
        for g in moved.tolist():
            gi, _ = rad.query(arm.G[g])
            back.update(gi[drop[gi]].tolist())
        if not back:
            break
        drop[list(back)] = False
        restored += len(back)
    p1 = subset(arm, ~drop).p_change(arm.G)
    return drop, restored, float(np.abs(p1 - p0).max())


# ---------------------------------------------------------------- scoring (095.1's metrics)

def score(H, C, A, w, p, after):
    mask = H >= 0
    pc = np.clip(p, EPS, 1 - EPS)
    ll = (np.where(C, np.log(pc), np.log(1 - pc)) * mask).sum(1)
    right = (((p > 0.5) == C) | ~mask).all(1) & ((~C) | (after == A)).all(1)
    return {"change log-likelihood per try": round(float((ll * w).sum() / w.sum()), 6),
            "exact next state": round(float((right * w).sum() / w.sum()), 6)}


def size_of(path, arrays):
    np.savez_compressed(path, **arrays)
    return {"disk bytes": path.stat().st_size, "RAM bytes": int(sum(np.asarray(v).nbytes for v in arrays.values()))}


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    z = tok.load(WORLD)
    H, P, C, A = tok.build(z)
    tr, te = tok.split(z)
    near_n = int((np.abs(z["wh"][:169]).sum(1) <= tok.DMAX).sum()) - 1
    F, r, c = AB.fields(H, P, near_n)
    ch, af, w = C[r, c], A[r, c], z["w"][r].astype(np.float64)
    act, train = z["act"][r], tr[r]
    arr, known = z["arr"], np.unique(z["app"])
    arr_t = torch.as_tensor(arr, device=dev)
    alpha = json.loads((tok.RUNS / "c_alpha.json").read_text())[WORLD]
    b_rep = json.loads((tok.RUNS / f"b_{WORLD}.json").read_text())
    cz = np.load(tok.RUNS / f"c_{WORLD}.npz")
    bz = np.load(tok.RUNS / f"b_{WORLD}.npz")
    lo = np.load(tok.RUNS / f"b_loo_{WORLD}.npz")
    held = np.flatnonzero(~train)
    assert np.array_equal(cz["r"], r[held]) and np.array_equal(cz["c"], c[held])
    net = AA.StateNet().to(dev)
    net.load_state_dict(torch.load(tok.RUNS / "arm_a.pt")["net"])
    net.eval()
    pA, aA, hA = a_states(net, arr_t, known, H, P, z["act"])
    log(f"{WORLD}: arm A on {len(H)} tries ({round(time.monotonic() - t0, 1)} s)")
    rep = {"world": WORLD, "radius": R, "actions": {}, "timing": {}}
    pred = {k: np.zeros(len(F)) for k in CONFIGS}
    aft = {k: np.full(len(F), -1) for k in CONFIGS}
    raw_keep = np.zeros(len(H), bool)
    stores, gates = {}, {}
    for a in (3, 4, 5):
        al = alpha[str(a)]
        s = np.flatnonzero(train & (act == a))
        arm = rebuild_arm(arr, F[s], w[s], ch[s], b_rep[str(a)]["admitted"])
        assert len(arm.G) == b_rep[str(a)]["groups"]
        sc = s[ch[s]]
        q = np.flatnonzero(~train & (act == a))
        hp = np.searchsorted(held, q)
        uq, qinv = np.unique(F[q], axis=0, return_inverse=True)
        qinv = qinv.ravel()
        pa_q, aa_q = pA[r[q], c[q]], aA[r[q], c[q]]
        # the reference: 095.1's arm C (B's totals over every group); checked on a sample against the rebuilt arm
        chk = np.random.default_rng(0).choice(len(q), min(300, len(q)), replace=False)
        _, tchk = arm.p_change(F[q[chk]], totals=True)
        assert np.allclose(tchk, np.stack([bz["nc"][hp[chk]], bz["n"][hp[chk]]], 1), rtol=1e-5, atol=1e-8)
        Pf, Af = cz["p"][hp], cz["after"][hp]
        comb = lambda Nc, N: np.clip((Nc + al * pa_q) / (N + al), EPS, 1 - EPS)
        # gate, on training episodes
        m = lo["act"] == a
        lr_, lc_ = lo["r"][m], lo["c"][m]
        pa_l = pA[lr_, lc_]
        pc_l = (lo["nc"][m] + al * pa_l) / (lo["n"][m] + al)
        y = (np.abs(pc_l - pa_l) > DELTA) | (lo["ch"][m] & (aA[lr_, lc_] != A[lr_, lc_]))
        gate, tau = train_gate(hA[lr_, lc_], y.astype(np.float32))
        gates[a] = (gate, tau)
        passed = gate_scores(gate, hA[r[q], c[q]]) >= tau
        # index over every group (raw changed instances, as 095.1)
        st_i = Store(arm, F[sc], af[sc], w[sc])
        vi = [st_i.vote(f) for f in uq]
        Ni_c, Ni = np.array([v[0] for v in vi]), np.array([v[1] for v in vi])
        Pi = comb(Ni_c[qinv], Ni[qinv])
        _, tfull = arm.p_change(uq, totals=True)
        inside = float(Ni.sum() / tfull[:, 1].sum())
        log(f"   action {a}: index done ({round(time.monotonic() - t0, 1)} s)")
        # forgetting: dropped no-change groups; changed instances merged
        drop, restored, maxmove = forget(arm)
        kept = subset(arm, ~drop)
        ucm, cinv = np.unique(np.concatenate([F[sc], af[sc][:, None]], 1), axis=0, return_inverse=True)
        Fm, am = ucm[:, :-1], ucm[:, -1]
        wm = np.bincount(cinv.ravel(), weights=w[sc], minlength=len(ucm))
        st_k = Store(kept, Fm, am, wm)
        _, tk = kept.p_change(uq, totals=True)
        Pk = comb(tk[qinv, 0], tk[qinv, 1])
        vk = [st_k.vote(f) for f in uq]
        Pa = comb(np.array([v[0] for v in vk])[qinv], np.array([v[1] for v in vk])[qinv])
        log(f"   action {a}: forgetting done ({round(time.monotonic() - t0, 1)} s)")
        # results, for tokens that changed or that any configuration predicts to change
        need = np.unique(qinv[ch[q] | (pa_q > 0.5) | (Pf > 0.5) | (Pi > 0.5) | (Pk > 0.5) | (Pa > 0.5)])
        Ri = {i: st_i.result(uq[i], *vi[i][2:], known, arr) for i in need.tolist()}
        Ra = {i: st_k.result(uq[i], *vk[i][2:], known, arr) for i in need.tolist()}
        pick = lambda d: np.array([d.get(int(i), -1) for i in qinv])
        fb = lambda x: np.where(x >= 0, x, aa_q)           # no changed neighbours: A's result
        # forget alone keeps every changed instance, so its results are 095.1's
        P_ = {"full": Pf, "gate": np.where(passed, Pf, pa_q), "index": Pi, "forget": Pk,
              "all": np.where(passed, Pa, pa_q)}
        A_ = {"full": Af, "gate": np.where(passed, Af, aa_q), "index": fb(pick(Ri)), "forget": Af,
              "all": np.where(passed, fb(pick(Ra)), aa_q)}
        for k in CONFIGS:
            pred[k][q], aft[k][q] = P_[k], A_[k]
        # raw tries kept: surprises and a uniform sample
        gs = np.array([st_i.rad.key[tuple(x)] for x in F[s][:, st_i.rad.cols].tolist()])
        maj = (arm.Gnc / np.maximum(arm.Gn, 1e-12))[gs] > 0.5
        surprise = ch[s] != maj
        raw_keep[r[s[surprise]]] = True
        # one try: a failed try added at the state of 200 held-out tokens predicted to change
        oq = np.flatnonzero(Pf > 0.5)
        oq = oq[np.random.default_rng(1).choice(len(oq), min(200, len(oq)), replace=False)] if len(oq) else oq
        one = {}
        if len(oq):
            add = lambda Nc, N: (Nc + al * pa_q[oq]) / (N + 1 + al)
            pfull = add(bz["nc"][hp[oq]], bz["n"][hp[oq]])
            pidx = add(Ni_c[qinv[oq]], Ni[qinv[oq]])
            pall = add(np.array([vk[i][0] for i in qinv[oq]]), np.array([vk[i][1] for i in qinv[oq]]))
            one = {"tokens": int(len(oq)),
                   "full: mean P before / after": [round(float(Pf[oq].mean()), 4), round(float(pfull.mean()), 4)],
                   "all: mean P after": round(float(pall.mean()), 4),
                   "max |full - index| after": round(float(np.abs(pfull - pidx).max()), 5),
                   "max |full - all| after": round(float(np.abs(pfull - pall).max()), 5)}
        # time per token, one CPU thread: the full vote over every group against the index on kept groups
        torch.set_num_threads(1)
        tq = np.random.default_rng(0).choice(len(uq), min(300, len(uq)), replace=False)
        rad = st_i.rad
        tables = [(col, rad.vals[j], np.searchsorted(rad.vals[j], arm.G[:, col]), rad.conds[j])
                  for j, col in enumerate(rad.cols)]
        t1 = time.monotonic()
        for i in tq:
            f = uq[i]
            D = np.zeros(len(arm.G))
            for col, vals, pos, conds in tables:
                dv = np.zeros(len(vals))
                for cnd, wv in conds:
                    if cnd[0] == "place":
                        dv += wv * (vals != f[col])
                    else:
                        sl = slice(cnd[-1] * 8, cnd[-1] * 8 + 8)
                        dv += wv * np.abs(arm.V[vals][:, sl] - arm.V[int(f[col])][sl]).sum(1)
                D += dv[pos]
            K = np.exp(-D)
            _ = K @ arm.Gnc, K @ arm.Gn
        t_full = (time.monotonic() - t1) / len(tq)
        st_t = Store(kept, Fm, am, wm)                 # a fresh index: no distances cached from the runs above
        t2 = time.monotonic()
        for i in tq:
            st_t.vote(uq[i])
        t_idx = (time.monotonic() - t2) / len(tq)
        torch.set_num_threads(8)
        gp = float(passed.mean())
        rep["timing"][str(a)] = {"full vote, ms per token": round(1e3 * t_full, 4),
                                 "index on kept groups, ms per token": round(1e3 * t_idx, 4),
                                 "gate pass share": round(gp, 4),
                                 "all, ms per token (pass share x index)": round(1e3 * gp * t_idx, 5),
                                 "speed-up per token: index": round(t_full / max(t_idx, 1e-12), 1),
                                 "speed-up per token: all": round(t_full / max(gp * t_idx, 1e-12), 1)}
        stores[a] = (st_i, st_k)
        rep["actions"][str(a)] = {
            "alpha": al, "admitted": [str(x) for x in arm.adm], "stored groups": int(len(arm.G)),
            "groups after forgetting": int((~drop).sum()), "groups restored": restored,
            "largest move of a stored group's P after forgetting": round(maxmove, 5),
            "changed instances": int(len(sc)), "changed instances merged": int(len(Fm)),
            "distinct held-out queries": int(len(uq)), "gate threshold": round(tau, 4),
            "gate: training positives / instances": [int(y.sum()), int(len(y))],
            "gate: held-out tokens passed": round(gp, 4),
            "groups read per query: index (mean)": round(float(np.mean([len(v[2]) for v in vi])), 1),
            "groups read per query: all (mean, before the gate)": round(float(np.mean([len(v[2]) for v in vk])), 1),
            "kernel weight inside the radius": round(inside, 7),
            "surprising raw tries": int(surprise.sum()), "one try": one}
        log(f"{WORLD} action {a}: {json.dumps(rep['actions'][str(a)])}\n   timing {json.dumps(rep['timing'][str(a)])}"
            f" ({round(time.monotonic() - t0, 1)} s)")
    tt_ = np.flatnonzero(tr)                           # and a uniform sample of training tries
    raw_keep[tt_[np.random.default_rng(0).random(len(tt_)) < RAW_SHARE]] = True
    # held-out scores per configuration
    te_idx = np.flatnonzero(te)
    rows = np.searchsorted(te_idx, r[held])
    Hs, Cs, As, ws = H[te], C[te], A[te], z["w"][te]
    rep["held-out"] = {}
    for k in CONFIGS:
        pm, am_ = np.zeros(Hs.shape), np.full(Hs.shape, -1)
        pm[rows, c[held]] = pred[k][held]
        am_[rows, c[held]] = aft[k][held]
        rep["held-out"][k] = score(Hs, Cs, As, ws, pm, am_)
    log(f"{WORLD} held-out: {json.dumps(rep['held-out'])}")
    # counterfactual boxes (tier 2's pick ups, as 095.1's counterfactual.py)
    if WORLD == "tier2":
        fs = int(np.flatnonzero(P[0] == 71)[0])
        cand = np.flatnonzero(te & (z["act"] == 3) & (H[:, -1] == 0) & (H[:, fs] >= 0))
        Q = np.random.default_rng(0).choice(cand, min(300, len(cand)), replace=False)
        al = alpha["3"]
        st_i, st_k = stores[3]
        arm = st_i.arm
        gate, tau = gates[3]
        sc = np.flatnonzero(train & (act == 3) & ch)
        cf = {}
        for col, bx in BOXES.items():
            Hq = H[Q].copy()
            Hq[:, fs] = bx
            pq, aq, hq = a_states(net, arr_t, known, Hq, P[Q], np.full(len(Q), 3))
            pa, sa, ha = pq[:, -1], aq[:, -1], hq[:, -1]
            Fq, _, _ = AB.fields(Hq, P[Q], near_n)
            hf = Fq[Fq[:, 1] == tok.HAND]
            _, tb = arm.p_change(hf, totals=True)
            rb, _, _ = AB.results(arm, hf, F[sc], w[sc], af[sc], known, arr)
            passed = gate_scores(gate, ha) >= tau
            out = {}
            for k in CONFIGS:
                if k in ("index", "all"):
                    st = st_i if k == "index" else st_k
                    v = [st.vote(f) for f in hf]
                    Nc, N = np.array([x[0] for x in v]), np.array([x[1] for x in v])
                    res = np.array([st.result(f, *x[2:], known, arr) for f, x in zip(hf, v)])
                else:
                    tt = tb if k != "forget" else st_k.arm.p_change(hf, totals=True)[1]
                    Nc, N, res = tt[:, 0], tt[:, 1], rb
                p = (Nc + al * pa) / (N + al)
                res = np.where(res >= 0, res, sa)
                if k in ("gate", "all"):
                    p, res = np.where(passed, p, pa), np.where(passed, res, sa)
                out[k] = round(float(((res == bx) & (p > 0.5)).mean()), 4)
            out["gate passed"] = round(float(passed.mean()), 4)
            cf[col] = out
            log(f"counterfactual {col}: {out}")
        rep["counterfactual boxes: hand holds that box"] = cf
    # memory: interaction tries (095.1's states), the store before forgetting, the store after
    st = {k: z[k] for k in ("pos", "e0", "e1", "act", "w", "pres")}
    rep["memory"] = {"memory 066 file (every action, moves too), disk bytes":
                     (ROOT / "runs" / "066" / f"memory_tier{WORLD[-1]}.npz").stat().st_size,
                     "interaction tries (095.1's states)": size_of(OUT / f"_tries_{WORLD}.npz", st)}
    full_arrays, kept_arrays = {}, {}
    for a, (st_i, st_k) in stores.items():
        for s_, d in ((st_i, full_arrays), (st_k, kept_arrays)):
            d[f"g{a}"] = s_.arm.G[:, s_.rad.cols].astype(np.int32)
            d[f"nc{a}"], d[f"n{a}"] = s_.arm.Gnc.astype(np.float32), s_.arm.Gn.astype(np.float32)
            d[f"f{a}"], d[f"a{a}"] = s_.Fch.astype(np.int32), s_.ach.astype(np.int32)
            d[f"w{a}"] = s_.wch.astype(np.float32)
    raw = {f"raw_{k}": z[k][raw_keep] for k in st}
    rep["memory"]["store before forgetting (groups, changed instances)"] = size_of(OUT / f"_full_{WORLD}.npz", full_arrays)
    rep["memory"]["store after forgetting (+ raw tries kept)"] = size_of(OUT / f"memory_{WORLD}.npz",
                                                                          {**kept_arrays, **raw})
    rep["memory"]["raw tries kept / all"] = [int(raw_keep.sum()), int(len(H))]
    for f in (OUT / f"_tries_{WORLD}.npz", OUT / f"_full_{WORLD}.npz"):
        f.unlink()
    rep["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"scale_{WORLD}.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(f"{WORLD} saved ({rep['seconds']} s)")
