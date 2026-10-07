"""Card 084: conditions over roles and relations, a level of recall between own tries and neighbours (report only).

  WM_REL_ENCODER=runs/070/encoder.pt bin/prun python tools/card084/lifted.py red     # one colour fold
  ... lifted.py none                                                                # the gate: nothing removed
  ... lifted.py tier2                                                               # tier 2's hand tables
  Card 084.1: WM_EVIDENCE=key WM_TAG=_084.1 (each stored key's counts scaled to one in the evidence).
  Card 084.2: WM_ORDER=specific WM_TAG=_084.2 (the back-off drops the condition with most values first).
  Card 084.3: WM_STAGES=1 WM_ORDER=specific WM_TAG=_084.3 (roles and cuts admitted before identity and view tuples).
  Card 084.4: also WM_GAMMA=combination, WM_TAG=_084.4 (gamma fitted by leaving whole (front, held) combinations out).

For pick up, toggle and drop, conditions are chosen greedily from candidates that read no tile's vector directly:
  roles     what version 18's recall predicts the front or held tile does, empty-handed, under the other actions
            (card 077's bits: forward moves onto it; forward onto it ends the episode; a pick up changes it; a toggle
            changes it), the action's own bit left out;
  cut       card 070's |P (z_front - z_held)|_1 below or above a midpoint between consecutive stored values;
  identity  the front's and the held tile's code tuples, and a code tuple in view (card 049's candidates).
A condition is admitted while it raises the evidence (Dirichlet-multinomial marginal likelihood of the stored outcome
counts over the cells, one pseudo-count per class) by more than log(number of candidates). Prediction:
  P = (N_own + beta P_1) / (|N_own| + beta);  P_l = (N_l + gamma P_l+1) / (|N_l| + gamma),  P_k+1 = version 18's
  neighbours; N_l the stored tries equal to the query on the first k - l + 1 admitted conditions.
beta is version 18's; gamma is fitted by leaving one try out, as beta was (card 050).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
TIER = 2 if MODE == "tier2" else 1
if TIER == 2:
    sys.argv = ["run.py", "tiers", "--tier", "2", "--online", "0", "--memory", "066"]
else:
    sys.argv = ["run.py", "decoy", "--fold", MODE if MODE != "none" else "red", "--hold", "opening", "--online", "0",
                "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

T, R = RUN.T, RUN.R
VP, TK, SP = T.VP, T.TK, sys.modules["slot_planner"]
CR, OWN = sys.modules["code_recall"], sys.modules["own"]
t00 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)

if TIER == 1:                                          # card 079's colour folds: the fold hue's openings removed
    T.ENVS[1] = R.register()
    T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
    R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
    _memory = T.memory

    def memory(tier, pool=None):
        D, stats = _memory(tier, pool)
        if MODE == "none":
            return D, stats
        f = D["ego0"][:, T.FRONT].astype(np.int64) // 5 * 5
        h = D["ego0"][:, T.HELD].astype(np.int64) // 5 * 5
        drop = (D["act"] == VP.TOG) & (f == T.KR.code(("door", MODE, 0))) & (h == T.KR.code(("key", MODE)))
        inter = np.isin(D["act"], T.INTER)
        keep_t = ~drop[inter]
        D = {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}
        return D, {**stats, "opening_tries_removed": int(drop.sum()), "rows": int(len(D["act"]))}

    T.memory = memory
t_setup = time.monotonic()
W, info = T.setup(TIER, log)
t_setup = time.monotonic() - t_setup
app = lambda o: int(T.Kd.APP[T.KR.code(o)])
pl = T.S7.Plan047(W)
env = T.make(TIER)
env.reset(seed=T.SEED_TEST + (999 if TIER == 1 else 1000 * TIER))
codes = T.now_codes(env)
_, st, _, _ = pl.observe(None, T.PV.crop(T.Kd.APP[codes], codes))
ctx, empty = pl.ctx_id(st[0], st[1]), int(pl.facts[st[0]][VP.HELD])
P = torch.load(os.environ.get("WM_REL_ENCODER", "runs/070/encoder.pt"))["P"].double().numpy()
Z = W.S.arr
EVIDENCE = os.environ.get("WM_EVIDENCE", "try")      # card 084.1: "key"
TAG = os.environ.get("WM_TAG", "")
ORDER = os.environ.get("WM_ORDER", "admission")       # card 084.2: "specific"
STAGES = os.environ.get("WM_STAGES") == "1"            # card 084.3: conditions without constants admitted first
GAMMA_FIT = os.environ.get("WM_GAMMA", "try")          # card 084.4: "combination"
ACTS = [("pick up", VP.PICK), ("drop", VP.DROP), ("toggle", VP.TOG)]
OWN_BIT = {VP.PICK: 2, VP.TOG: 3, VP.DROP: None}

# ---------------------------------------------------------------- the tables (the evaluator's truth)
NAME = {}
for h in range(len(Z)):                                # the evaluator's names, to build the tables only
    try:
        NAME.setdefault(W.judge.name(h), h)
    except Exception:                                  # noqa: BLE001
        pass
if TIER == 1:                                          # cards 079 and 082's tables
    names = {**{f"door {c}": app(("door", c, 0)) for c in T.HUES}, **{f"key {c}": app(("key", c)) for c in T.HUES},
             **{nm: NAME[nm] for nm in ("wall", "floor") if nm in NAME}, "nothing": empty}
    fronts = [n for n in names if n != "nothing"]
    TABLES = {
        "toggle": [(fn, hn, fn.startswith("door") and hn.startswith("key") and fn[5:] == hn[4:])
                   for fn in [f"door {c}" for c in T.HUES] + ["wall", "floor"]
                   for hn in [f"key {c}" for c in T.HUES] + ["nothing"]],
        "pick up": [(fn, hn, fn.startswith("key") and hn == "nothing") for fn in fronts for hn in ("nothing", "key red")],
        "drop": [(fn, hn, fn == "floor" and hn != "nothing") for fn in fronts for hn in ("nothing", "key red", "key blue")],
    }
else:                                                  # tier 2: the layout's own box, ball, key and locked door
    objs = {}
    g = env.unwrapped.grid
    for i in range(g.width):
        for j in range(g.height):
            o = g.get(i, j)
            if o is not None and o.type in ("box", "ball", "key", "door"):
                objs[o.type] = (o.type, o.color, 0) if o.type == "door" else (o.type, o.color)
    names = {**{f"{o[0]} {o[1]}": app(o) for o in objs.values()}, "nothing": empty}
    door, box, ball, key = (f"{objs[t][0]} {objs[t][1]}" for t in ("door", "box", "ball", "key"))
    TABLES = {
        "pick up": [(fn, hn, hn == "nothing") for fn in (box, ball, key) for hn in ("nothing", ball, key)],
        "toggle": [(door, hn, hn == key) for hn in ("nothing", ball, key)],
    }


# ---------------------------------------------------------------- candidates
ROLE = {}


def role(h):
    h = int(h)
    if h not in ROLE:                                  # version 18's recall, empty-handed, from the start situation
        ROLE[h] = [int(TK.free_of(h)), int(W.move_cat(VP.FWD, h) == VP.ENDED),
                   int(W.outcome(VP.PICK, h, empty, ctx)[0] != 0), int(W.outcome(VP.TOG, h, empty, ctx)[0] != 0)]
    return ROLE[h]


ROLE_NAMES = ["walk onto", "ends", "pick up changes it", "toggle changes it"]
rel = lambda f, h: float(np.abs(P @ (Z[int(f)] - Z[int(h)])).sum())


class Lifted:
    def __init__(self, a, kd, keep=None, by="class"):
        self.a, self.kd, self.by = a, kd, by
        n = len(kd.keys)
        self.idx = np.arange(n) if keep is None else np.asarray(keep)
        keys = [kd.keys[i] for i in self.idx]
        self.lc = kd.lc[self.idx].astype(np.float64)
        self.bits = [b for b in range(4) if b != OWN_BIT[a]]
        vcand = [c for c in kd.cand if c[0] == "view"]
        cands, cols = [], []
        for side in (0, 1):
            for b in self.bits:
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
        self.active = self.lc.sum(0) > 0
        ev = self.lc
        if EVIDENCE == "key":                          # card 084.1: each stored situation counts once
            ev = ev / np.maximum(ev.sum(1, keepdims=True), 1e-300)
        if by == "category":                           # diagnosis: evidence over whether it changed, not the result
            M = np.zeros((self.lc.shape[1], kd.ncat))
            M[np.arange(self.lc.shape[1]), kd.lcat[:self.lc.shape[1]]] = 1.0
            self.ev = (ev @ M)[:, (self.lc @ M).sum(0) > 0]
        else:
            self.ev = ev[:, self.active]

    def value(self, c, q):
        """Candidate c's value for a query key (front, held, sid)."""
        if c[0] == "role":
            return role(q[c[1]])[c[2]]
        if c[0] == "cut":
            return int(rel(q[0], q[1]) > c[1])
        if c[0] == "id":
            return CR.code_id(self.kd.S, q[c[1]])
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

    def admit(self, max_conditions=64):                # 12 until card 084.3, where toggle reached it
        cost = math.log(len(self.cands))
        cell = np.zeros(len(self.lc), np.int64)
        e0 = self.evidence(cell)
        self.adm, self.trace, used = [], [], set()
        # card 084.3: conditions that name no particular tile (roles, the relation's cut) first; then every candidate,
        # identity among them, for what they leave unexplained
        stages = [("general", lambda c: c[0] in ("role", "cut")), ("all", lambda c: True)] if STAGES \
            else [("all", lambda c: True)]
        for stage, ok in stages:
            while len(self.adm) < max_conditions:
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
        self.cost_bits = cost / math.log(2)
        if ORDER == "specific":                        # card 084.2: the condition with most values is dropped first
            self.adm = sorted(self.adm, key=lambda ci: len(np.unique(self.cols[ci])))
        self.levels = [self.name(self.cands[ci]) for ci in self.adm]
        # cells per prefix of the admitted conditions: values -> summed counts
        V = np.stack([self.cols[ci] for ci in self.adm], 1) if self.adm else np.zeros((len(self.lc), 0), np.int64)
        self.V = V
        self.cells = []
        for m in range(1, len(self.adm) + 1):
            d = {}
            for i, row in enumerate(map(tuple, V[:, :m])):
                d[row] = d.get(row, 0) + self.lc[i]
            self.cells.append(d)

    def level_counts(self, vals):
        L = self.kd.lc.shape[1]
        out = []
        for m, d in enumerate(self.cells, 1):
            c = d.get(tuple(vals[:m]))
            c = np.zeros(L) if c is None else np.pad(c, (0, L - len(c)))
            out.append(c)
        return out                                     # coarsest first

    def fit_gamma(self):
        """gamma by leaving one try out over the stored keys used here, as card 050's beta."""
        kd = self.kd
        keys = [kd.keys[i] for i in self.idx]
        _, s = kd.neighbours(keys)
        Nown = np.stack([kd.own_counts(k) for k in keys])[:, :self.lc.shape[1]]
        lev = [np.stack([self.cells[m][tuple(r[:m + 1])] for r in self.V]) for m in range(len(self.adm))]
        ii, jj = np.nonzero(self.lc)
        w = self.lc[ii, jj]

        def ll(gamma, levels=True):
            p = s[ii, jj]
            if levels:
                for Nm in lev:
                    p = (Nm[ii, jj] - 1.0 + gamma * p) / (Nm[ii].sum(1) - 1.0 + gamma)
            p = (Nown[ii, jj] - 1.0 + kd.beta * p) / (Nown[ii].sum(1) - 1.0 + kd.beta)
            return float((w * np.log(np.maximum(p, 1e-300))).sum())

        vals = [ll(g) for g in OWN.ALPHAS]
        j = int(np.argmax(vals))
        self.gamma = float(OWN.ALPHAS[j])
        self.ll = {"lifted": round(vals[j], 2), "version_18": round(ll(0.0, levels=False), 2)}
        self.gamma_try = self.gamma
        if GAMMA_FIT == "combination":
            self.fit_gamma_combination()

    def fit_gamma_combination(self):
        """Card 084.4: gamma by leaving whole (front, held) combinations out (card 071's unit). Each stored try is
        predicted with every try of its combination removed from its own situation (then empty), from every cell and
        from version 18's neighbours; tries weighted as stored, as for beta."""
        kd = self.kd
        keys = [kd.keys[i] for i in self.idx]
        L = self.lc.shape[1]
        combo = [tuple(int(h) for h in k[:2]) for k in keys]
        wG, _ = kd.neighbours(keys)                    # keys x groups
        Gc = kd.ix["Gc"]
        Gc = np.hstack([Gc, np.zeros((len(Gc), L - Gc.shape[1]))]) if Gc.shape[1] < L else Gc[:, :L]
        gcombo = [tuple(int(h) for h in kd.keys[r][:2]) for r in kd.ix["rep"]]
        bygc = {}
        for g, c in enumerate(gcombo):
            bygc.setdefault(c, []).append(g)
        Nn = wG @ Gc
        for i, c in enumerate(combo):                  # the combination's own groups leave the neighbours
            gs = bygc.get(c, [])
            if gs:
                Nn[i] -= wG[i, gs] @ Gc[gs]
        Nn = np.maximum(Nn, 0.0)
        s = (Nn + kd.a / L) / (Nn.sum(1, keepdims=True) + kd.a)
        lev = []
        for m in range(len(self.adm)):                 # each cell's counts less the combination's own in that cell
            own_c = {}
            rows = [tuple(r[:m + 1]) for r in self.V]
            for i, (r, c) in enumerate(zip(rows, combo)):
                own_c[(r, c)] = own_c.get((r, c), 0) + self.lc[i]
            lev.append(np.stack([self.cells[m][r] - own_c[(r, c)] for r, c in zip(rows, combo)]))
        ii, jj = np.nonzero(self.lc)
        w = self.lc[ii, jj]

        def ll(gamma, levels=True):
            p = s[ii, jj]
            if levels:
                for Nm in lev:
                    p = (Nm[ii, jj] + gamma * p) / (Nm[ii].sum(1) + gamma)
            return float((w * np.log(np.maximum(p, 1e-300))).sum())

        vals = [ll(g) for g in OWN.ALPHAS]
        j = int(np.argmax(vals))
        self.gamma = float(OWN.ALPHAS[j])
        self.ll["combination_out"] = {"lifted": round(vals[j], 2), "neighbours_only": round(ll(0.0, levels=False), 2),
                                      "at_try_gamma": round(ll(self.gamma_try), 2)}
        if os.environ.get("WM_GAMMA_DIAG") == "1":     # diagnosis: which held-out combinations favour a large gamma
            def per_try(gamma):
                p = s[ii, jj]
                for Nm in lev:
                    p = (Nm[ii, jj] + gamma * p) / (Nm[ii].sum(1) + gamma)
                return w * np.log(np.maximum(p, 1e-300))
            lo, hi = per_try(OWN.ALPHAS[0]), per_try(OWN.ALPHAS[-1])
            by = {}
            for t, i in enumerate(ii):
                d = by.setdefault(combo[i], [0.0, 0.0, 0.0])
                d[0] += lo[t]
                d[1] += hi[t]
                d[2] += w[t]
            nm = lambda c: W.judge.name(c[0]) + " | " + W.judge.name(c[1])
            top = sorted(by.items(), key=lambda kv: kv[1][0] - kv[1][1])[:12]
            # gamma if every combination weighed the same (mean log-likelihood per try within it)
            cid = {c: n for n, c in enumerate(by)}
            ci_t = np.array([cid[combo[i]] for i in ii])
            tot = np.zeros(len(by))
            np.add.at(tot, ci_t, w)

            def ll_eq(gamma):
                v = per_try(gamma)
                acc = np.zeros(len(by))
                np.add.at(acc, ci_t, v)
                return float((acc / tot).sum())
            ve = [ll_eq(g) for g in OWN.ALPHAS]
            self.ll["diagnosis"] = {
                "favour_large_gamma": [(nm(c), round(d[2]), round(d[0] - d[1], 1)) for c, d in top],
                "combinations": len(by), "gamma_each_combination_equal": float(OWN.ALPHAS[int(np.argmax(ve))]),
                "combinations_favouring_small": int(sum(d[0] > d[1] for d in by.values()))}

    def predict(self, q, levels=True):
        kd = self.kd
        _, s = kd.neighbours([q])
        p = s[0]
        L = len(p)
        vals = [self.value(self.cands[ci], q) for ci in self.adm]
        for Nm in (self.level_counts(vals) if levels else []):
            Nm = Nm[:L]
            p = (Nm + self.gamma * p) / (Nm.sum() + self.gamma)
        No = kd.own_counts(q)[:L]
        p = (No + kd.beta * p) / (No.sum() + kd.beta)
        M = np.zeros((L, kd.ncat))
        M[np.arange(L), kd.lcat[:L]] = 1.0
        pc = p @ M
        return (int(pc.argmax()) if pc.max() >= 0.5 else 0), pc, vals


# ---------------------------------------------------------------- run
res = {"note": "Card 084, tools/card084/lifted.py", "mode": MODE, "evidence": EVIDENCE, "order": ORDER, "stages": STAGES, "gamma_fit": GAMMA_FIT, "setup_seconds": round(t_setup, 1),
       "memory": info.get("memory"), "actions": {}}
ARMS = ["class", "category"]                           # "class" as declared; "category" a diagnosis
LF = {}
for arm, an, a in [(arm, an, a) for arm in ARMS for an, a in ACTS]:
    if an not in TABLES:
        continue
    kd = W.kinds[a]
    t0 = time.monotonic()
    lf = Lifted(a, kd, by=arm)
    lf.admit()
    lf.fit_gamma()
    t_admit = time.monotonic() - t0
    rows = []
    tq = time.monotonic()
    for fn, hn, truth in TABLES[an]:
        q = (names[fn], names[hn], ctx)
        c, pc, vals = lf.predict(q)
        v18 = int(W.outcome(a, q[0], q[1], ctx)[0])
        rows.append({"v18_formula": lf.predict(q, levels=False)[0], "front": fn, "held": hn, "truth": bool(truth), "lifted": c, "v18": v18,
                     "p_change": round(float(pc[1:].sum()), 4), "own": float(kd.own_counts(q).sum()),
                     "rel": round(rel(q[0], q[1]), 3), "values": [int(v) for v in vals]})
    per_q = (time.monotonic() - tq) / len(rows)
    right = lambda k: sum((r[k] != 0) == r["truth"] for r in rows)
    out = {"stored_keys": len(kd.keys), "stored_tries": float(kd.lc.sum()), "candidates": len(lf.cands),
           "cost_bits": round(lf.cost_bits, 2), "admitted": lf.trace, "levels_general_first": lf.levels,
           "gamma_leave_one_try_out": lf.gamma_try, "gamma": lf.gamma, "beta": kd.beta,
           "loo_try_ll": lf.ll, "admit_seconds": round(t_admit, 1), "seconds_per_query": round(per_q, 5),
           "lifted_right": right("lifted"), "v18_right": right("v18"), "of": len(rows),
           "v18_formula_agrees": sum(r["v18_formula"] == r["v18"] for r in rows),
           "lifted_wrong": [(r["front"], r["held"]) for r in rows if (r["lifted"] != 0) != r["truth"]],
           "v18_wrong": [(r["front"], r["held"]) for r in rows if (r["v18"] != 0) != r["truth"]], "cells": rows}
    if an == "toggle" and TIER == 1 and lf.adm:        # the cell the left-out pair reaches once identity is dropped
        hue = MODE if MODE != "none" else "red"
        q = (names[f"door {hue}"], names[f"key {hue}"], ctx)
        m = sum(len(np.unique(lf.cols[ci])) <= 2 for ci in lf.adm)
        vals = [lf.value(lf.cands[ci], q) for ci in lf.adm][:m]
        chg = np.array(kd.lcat[:lf.lc.shape[1]]) != 0
        hold = {}
        for i, k in enumerate(kd.keys):
            if m and tuple(lf.V[i, :m]) == tuple(vals):
                nm = W.judge.name(int(k[0])) + " | " + W.judge.name(int(k[1]))
                h = hold.setdefault(nm, [0.0, 0.0])
                h[0] += float(lf.lc[i, chg].sum())
                h[1] += float(lf.lc[i, ~chg].sum())
        top = sorted(hold.items(), key=lambda kv: -sum(kv[1]))
        near = []                                      # stored keys near each admitted cut
        for ci in lf.adm:
            c = lf.cands[ci]
            if c[0] == "cut":
                for i, k in enumerate(kd.keys):
                    if abs(lf.rel[i] - c[1]) < 0.5:
                        near.append((round(float(lf.rel[i]), 3), W.judge.name(int(k[0])) + " | " + W.judge.name(int(k[1])),
                                     float(lf.lc[i, chg].sum()), float(lf.lc[i, ~chg].sum())))
        out["general_cell"] = {"pair": f"door {hue} | key {hue}", "conditions": lf.levels[:m], "values": vals,
                               "changed": sum(v[0] for v in hold.values()),
                               "unchanged": sum(v[1] for v in hold.values()), "holds": top[:15],
                               "near_cut": sorted(set(near))[:40]}
    if an == "toggle" and TIER == 1:
        door = {c: names[f"door {c}"] for c in T.HUES}
        out["closed_door_toggle_tries"] = float(sum(kd.lc[i].sum() for i, k in enumerate(kd.keys)
                                                    if k[0] in {app(("door", c, 1)) for c in T.HUES}))
        out["keys_at_fold_door"] = int(sum(1 for k in kd.keys if MODE != "none" and k[0] == door.get(MODE)))
        out["relation_own_pairs"] = {c: round(rel(door[c], names[f"key {c}"]), 3) for c in T.HUES}
        changed = np.array(kd.lcat[:lf.lc.shape[1]]) != 0
        dk = [i for i, k in enumerate(kd.keys) if k[0] in door.values() and k[1] in {names[f"key {c}"] for c in T.HUES}]
        opened = [lf.rel[i] for i in dk if lf.lc[i, changed].sum() > 0]
        failed = [lf.rel[i] for i in dk if lf.lc[i, ~changed].sum() > 0]
        out["stored_door_key_relation"] = {"opened_max": round(max(opened, default=float("nan")), 3),
                                           "failed_min": round(min(failed, default=float("nan")), 3),
                                           "opened_keys": len(opened), "failed_keys": len(failed)}
        LF[arm] = lf
        if MODE == "none":                             # P19's curve: admission on memory with 1, 2, 3, 5 colours
            curve = {}
            col = {app(("door", c, s)): c for c in T.HUES for s in range(3)} | {names[f"key {c}"]: c for c in T.HUES}
            for nc in (1, 2, 3, 5):
                hs = set(T.HUES[:nc])
                inside = lambda h: col.get(int(h)) is None or col[int(h)] in hs     # uncoloured tiles stay
                keep = [i for i, k in enumerate(kd.keys) if inside(k[0]) and inside(k[1])]
                sub = Lifted(a, kd, keep, by=arm)
                sub.admit()
                curve[nc] = {"keys": len(keep), "admitted": [t["condition"] for t in sub.trace]}
            out["p19_curve"] = curve
    res["actions"][f"{an} | {arm}"] = out
    log(f"{an} | {arm}: admitted {lf.trace} gamma {lf.gamma:.4g} lifted {out['lifted_right']}/{len(rows)} "
        f"v18 {out['v18_right']}/{len(rows)} wrong {out['lifted_wrong']}")
if TIER == 1 and MODE != "none":                       # criterion 3: one failed try of the left-out pair, after all else
    kd = W.kinds[VP.TOG]
    door = {c: names[f"door {c}"] for c in T.HUES}
    q = (door[MODE], names[f"key {MODE}"], ctx)
    before = {arm: lf.predict(q)[0] for arm, lf in LF.items()}
    i = kd.index.get(q)
    old = np.zeros(kd.lc.shape[1]) if i is None else kd.lc[i].copy()
    kd.add(q, 0, after=(q[0], q[1]))
    delta = kd.lc[kd.index[q]] - np.pad(old, (0, kd.lc.shape[1] - len(old)))
    for arm, lf in LF.items():
        vals = [lf.value(lf.cands[ci], q) for ci in lf.adm]
        for m, d in enumerate(lf.cells, 1):            # the try joins its cells as well as its own situation
            c = d.get(tuple(vals[:m]), np.zeros(len(delta)))
            c = np.pad(c, (0, len(delta) - len(c)))
            d[tuple(vals[:m])] = c + delta
        res["actions"][f"toggle | {arm}"]["fold_pair_after_one_failed_try"] = {"before": before[arm],
                                                                               "after": lf.predict(q)[0]}
if TIER == 1 and MODE != "none":
    res["fold_pair_opens"] = {}
    for arm in ARMS:
        t = res["actions"][f"toggle | {arm}"]
        pair = [r for r in t["cells"] if r["front"] == f"door {MODE}" and r["held"] == f"key {MODE}"][0]
        res["fold_pair_opens"][arm] = {"lifted": pair["lifted"] != 0, "v18": pair["v18"] != 0}
out_p = ROOT / "runs" / "084" / f"{MODE}{TAG}.json"
out_p.parent.mkdir(parents=True, exist_ok=True)
out_p.write_text(json.dumps(res, indent=1, default=str) + "\n")
log(json.dumps({k: v for k, v in res.items() if k not in ("actions", "memory")}))
