"""Card 086: version 19, recall for pick up, toggle and drop with card 084.3's level of cells over roles and relations.

  P = (N_own + beta P_1) / (|N_own| + beta);  P_l = (N_l + gamma P_l+1) / (|N_l| + gamma),  P_k+1 = version 18's
  neighbours; N_l the stored tries equal to the query on the first k - l + 1 admitted conditions.

Conditions (card 084.3): roles (what version 18's recall predicts the front or held tile does, empty-handed, from a
start situation, under the other actions: forward moves onto it, forward onto it ends the episode, a pick up changes
it, a toggle changes it) and cuts on card 070's |P (z_front - z_held)|_1 are admitted first, while each raises the
Dirichlet-multinomial evidence of the stored outcomes over the cells by more than log(number of candidates); then
identity and view tuples, for what they leave unexplained. The back-off drops the condition with most values first.
Evidence and gamma are on the category (whether and how the action changes something; card 084.3's diagnosis arm),
gamma by leaving whole (front, held) combinations out (card 084.4's unit). The tile a change produces is carried over
by card 038's ways, as in version 18. Online tries join their cells as well as their own situation; every episode
starts from the cells of stored memory.

  WM_GENERAL=1 <version 18's flags> bin/prun python tools/card069/run.py ...        (tools/card069/run.py installs it)
"""
import math
import os
import sys
import time

import numpy as np
import torch

STATS = {"general_predictions": 0}
REPORT = {}
STATE = {}
ROLE = {}
ROLE_NAMES = ["walk onto", "ends", "pick up changes it", "toggle changes it"]
MAX_CONDITIONS = 64


def hook(T):
    """Build the level right after setup: roles read version 18's recall, so the world must exist first."""
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        info["general"] = build(T, W, tier, log)
        return W, info

    T.setup = setup


# ---------------------------------------------------------------- candidates

def role(h):
    h = int(h)
    r = ROLE.get(h)
    if r is None:                                      # version 18's recall, empty-handed, from the start situation
        W, VP, TK = STATE["W"], STATE["VP"], STATE["TK"]
        r = ROLE[h] = [int(TK.free_of(h)), int(W.move_cat(VP.FWD, h) == VP.ENDED),
                       int(base_changes(VP.PICK, h)), int(base_changes(VP.TOG, h))]
    return r


def base_changes(a, h):
    """Whether version 18's recall predicts action a changes something on h, empty-handed (its own prediction,
    not this level's, so roles do not depend on the cells they define)."""
    kd = STATE["W"].kinds[a]
    p = STATE["base"][a](kd, [(int(h), STATE["empty"], STATE["ctx"])])[0]
    return int(p.argmax()) != 0 if p.max() >= 0.5 else False


def rel(f, h):
    Z = STATE["W"].S.arr
    return float(np.abs(STATE["P"] @ (Z[int(f)] - Z[int(h)])).sum())


class General:
    def __init__(self, a, kd):
        self.a, self.kd = a, kd
        CR = STATE["CR"]
        keys = list(kd.keys)
        self.L = kd.lc.shape[1]
        self.lc = kd.lc[:len(keys)].astype(np.float64)
        own_bit = {STATE["VP"].PICK: 2, STATE["VP"].TOG: 3}.get(a)
        bits = [b for b in range(4) if b != own_bit]
        vcand = [c for c in kd.cand if c[0] == "view"]
        cands, cols = [], []
        for side in (0, 1):
            for b in bits:
                cands.append(("role", side, b))
                cols.append(np.array([role(k[side])[b] for k in keys]))
        self.rel = np.array([rel(k[0], k[1]) for k in keys])
        u = np.unique(self.rel)
        for th in (u[1:] + u[:-1]) / 2:
            cands.append(("cut", float(th)))
            cols.append((self.rel > th).astype(np.int64))
        for side in (0, 1):
            cands.append(("id", side))
            cols.append(np.array([CR.code_id(kd.S, k[side]) for k in keys]))
        vf = np.stack([kd.view_feats(int(k[2])) for k in keys]) if vcand else np.zeros((len(keys), 0))
        for j, c in enumerate(vcand):
            cands.append(("view", j, c[1]))
            cols.append(vf[:, j].astype(np.int64))
        self.cands, self.cols = cands, cols
        self.M = self.cat_map(self.L)
        ev = self.lc @ self.M                          # evidence on the category (card 084.3's diagnosis arm)
        self.ev = ev[:, ev.sum(0) > 0]

    def cat_map(self, L):
        M = np.zeros((L, self.kd.ncat))
        M[np.arange(L), np.asarray(self.kd.lcat[:L])] = 1.0
        return M

    def value(self, c, q):
        if c[0] == "role":
            return role(q[c[1]])[c[2]]
        if c[0] == "cut":
            return int(rel(q[0], q[1]) > c[1])
        if c[0] == "id":
            return STATE["CR"].code_id(self.kd.S, q[c[1]])
        return int(self.kd.view_feats(int(q[2]))[c[1]])

    def name(self, c):
        if c[0] == "role":
            return f"{('front', 'held')[c[1]]} {ROLE_NAMES[c[2]]}"
        if c[0] == "cut":
            return f"rel:P > {c[1]:.3f}"
        if c[0] == "id":
            return f"{('front', 'held')[c[1]]} identity"
        return f"view {c[2]}"

    def evidence(self, cell):
        _, inv = np.unique(cell, return_inverse=True)
        C = np.zeros((inv.max() + 1, self.ev.shape[1]))
        np.add.at(C, inv, self.ev)
        Ct = torch.tensor(C)
        L = float(C.shape[1])
        return float((torch.lgamma(torch.tensor(L)) - torch.lgamma(L + Ct.sum(1))
                      + torch.lgamma(1.0 + Ct).sum(1)).sum())

    def admit(self):
        """Card 084.3: roles and cuts first, then every candidate; the back-off drops most values first."""
        cost = math.log(len(self.cands))
        cell = np.zeros(len(self.lc), np.int64)
        e0 = self.evidence(cell)
        self.adm, self.trace, used = [], [], set()
        stages = [("general", lambda c: c[0] in ("role", "cut")), ("all", lambda c: True)]
        for stage, ok in stages:
            while len(self.adm) < MAX_CONDITIONS:
                best = None
                for ci, col in enumerate(self.cols):
                    if ci in used or not ok(self.cands[ci]):
                        continue
                    _, nc = np.unique(np.stack([cell, col], 1), axis=0, return_inverse=True)
                    gain = self.evidence(nc.ravel()) - e0
                    if best is None or gain > best[0]:
                        best = (gain, ci, nc.ravel())
                if best is None or best[0] <= cost:
                    break
                gain, ci, cell = best
                e0 += gain
                used.add(ci)
                self.adm.append(ci)
                self.trace.append({"condition": self.name(self.cands[ci]), "gain_bits": round(gain / math.log(2), 1),
                                   "stage": stage})
        self.adm = sorted(self.adm, key=lambda ci: len(np.unique(self.cols[ci])))
        self.levels = [self.name(self.cands[ci]) for ci in self.adm]
        self.V = np.stack([self.cols[ci] for ci in self.adm], 1) if self.adm else np.zeros((len(self.lc), 0), np.int64)
        self.cells = []
        for m in range(1, len(self.adm) + 1):
            d = {}
            for i, row in enumerate(map(tuple, self.V[:, :m])):
                d[row] = d.get(row, 0) + self.lc[i]
            self.cells.append(d)
        self.base_cells = [{k: v.copy() for k, v in d.items()} for d in self.cells]

    def fit_gamma(self):
        """Card 084.4's fit (whole (front, held) combinations left out of their own situation, of every cell and of
        version 18's neighbours), scored on the category: a class naming the tile a change produces is carried over
        by card 038's ways and can never be predicted from other combinations (card 084.4)."""
        kd, L = self.kd, self.L
        keys = list(kd.keys)
        combo = [tuple(int(h) for h in k[:2]) for k in keys]
        wG, _ = kd.neighbours(keys)
        Gc = kd.ix["Gc"]
        Gc = np.hstack([Gc, np.zeros((len(Gc), L - Gc.shape[1]))]) if Gc.shape[1] < L else Gc[:, :L]
        bygc = {}
        for g, r in enumerate(kd.ix["rep"]):
            bygc.setdefault(tuple(int(h) for h in kd.keys[r][:2]), []).append(g)
        Nn = wG @ Gc
        for i, c in enumerate(combo):
            gs = bygc.get(c, [])
            if gs:
                Nn[i] -= wG[i, gs] @ Gc[gs]
        Nn = np.maximum(Nn, 0.0)
        s = (Nn + kd.a / L) / (Nn.sum(1, keepdims=True) + kd.a)
        M = self.M
        lev = []
        for m in range(len(self.adm)):
            own_c = {}
            rows = [tuple(r[:m + 1]) for r in self.V]
            for i, (r, c) in enumerate(zip(rows, combo)):
                own_c[(r, c)] = own_c.get((r, c), 0) + self.lc[i]
            lev.append(np.stack([self.cells[m][r] - own_c[(r, c)] for r, c in zip(rows, combo)]) @ M)
        sc, lcc = s @ M, self.lc @ M
        ii, jj = np.nonzero(lcc)
        w = lcc[ii, jj]

        def ll(gamma, levels=True):
            p = sc[ii, jj]
            if levels:
                for Nm in lev:
                    p = (Nm[ii, jj] + gamma * p) / (Nm[ii].sum(1) + gamma)
            return float((w * np.log(np.maximum(p, 1e-300))).sum())

        alphas = sys.modules["own"].ALPHAS
        vals = [ll(g) for g in alphas]
        j = int(np.argmax(vals))
        self.gamma = float(alphas[j])
        self.ll = {"combinations_out": round(vals[j], 2), "neighbours_only": round(ll(0.0, levels=False), 2),
                   "gamma_at_grid_top": bool(j == len(alphas) - 1)}

    def predict(self, qs):
        """Category probabilities, as version 18's predict."""
        kd = self.kd
        _, s = kd.neighbours(qs)
        L = s.shape[1]
        out = []
        for q, p in zip(qs, s):
            vals = [self.value(self.cands[ci], q) for ci in self.adm]
            for m, d in enumerate(self.cells):
                Nm = d.get(tuple(vals[:m + 1]))
                if Nm is not None:
                    Nm = np.pad(Nm, (0, L - len(Nm)))[:L]
                    p = (Nm + self.gamma * p) / (Nm.sum() + self.gamma)
            No = kd.own_counts(q)[:L]
            out.append((No + kd.beta * p) / (No.sum() + kd.beta))
        STATS["general_predictions"] += len(qs)
        return np.stack(out) @ self.cat_map(L)

    def joined(self, key, delta):
        """An online try joins its cells."""
        vals = [self.value(self.cands[ci], key) for ci in self.adm]
        for m, d in enumerate(self.cells):
            k = tuple(vals[:m + 1])
            c = d.get(k)
            c = np.zeros(len(delta)) if c is None else np.pad(c, (0, len(delta) - len(c)))
            d[k] = c + delta

    def reset(self):
        self.cells = [{k: v.copy() for k, v in d.items()} for d in self.base_cells]


def install_kind(g):
    kd = g.kd
    add0, reset0 = kd.add, kd.reset

    def add(key, cat, after=None, w=1.0):
        key = tuple(int(h) for h in key)
        i = kd.index.get(key)
        old = None if i is None else kd.lc[i].copy()
        add0(key, cat, after, w)
        new = kd.lc[kd.index[key]]
        g.joined(key, new - (np.pad(old, (0, len(new) - len(old))) if old is not None else 0.0))

    def reset():
        reset0()
        g.reset()
        ROLE.clear()

    kd.add, kd.reset, kd.predict = add, reset, g.predict


def build(T, W, tier, log):
    t0 = time.monotonic()
    VP = T.VP
    STATE.update(W=W, VP=VP, TK=T.TK, CR=sys.modules["code_recall"],
                 P=torch.load(os.environ.get("WM_REL_ENCODER", "runs/070/encoder.pt"))["P"].double().numpy())
    pl = T.S7.Plan047(W)                               # the start situation roles are predicted from (card 084)
    env = T.make(tier)
    env.reset(seed=T.SEED_TEST + (999 if tier == 1 else 1000 * tier))
    codes = T.now_codes(env)
    _, st, _, _ = pl.observe(None, T.PV.crop(T.Kd.APP[codes], codes))
    STATE.update(ctx=pl.ctx_id(st[0], st[1]), empty=int(pl.facts[st[0]][VP.HELD]))
    acts = [("pick up", VP.PICK), ("drop", VP.DROP), ("toggle", VP.TOG)]
    STATE["base"] = {a: type(W.kinds[a]).predict for _, a in acts}
    STATE["general"] = {}
    for an, a in acts:
        kd = W.kinds[a]
        t1 = time.monotonic()
        g = General(a, kd)
        g.admit()
        g.fit_gamma()
        STATE["general"][a] = g
        REPORT[an] = {"stored_keys": len(kd.keys), "stored_tries": float(kd.lc.sum()), "candidates": len(g.cands),
                      "admitted": g.trace, "levels_general_first": g.levels, "gamma": g.gamma, "beta": kd.beta,
                      "identity_admitted": any(t["condition"].endswith("identity") for t in g.trace),
                      "ll": g.ll, "seconds": round(time.monotonic() - t1, 1)}
        log(f"card 086 {an}: admitted {[t['condition'] for t in g.trace]} gamma {g.gamma:.4g} "
            f"({REPORT[an]['seconds']} s)")
    for _, a in acts:                                  # after every level is built: roles read version 18's own
        install_kind(STATE["general"][a])
        W.kinds[a].forget()
    W.clear_lazy()
    ROLE.clear()
    REPORT["seconds"] = round(time.monotonic() - t0, 1)
    return dict(REPORT)
