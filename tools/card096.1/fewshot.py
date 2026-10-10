"""Card 096.1: how many occurrences of an event arm C needs, and where recall alone falls short (offline).

An event is one (action, token in front, token in the hand) of the stored interaction tries; its outcome is what its
tries did (does the front change, does the hand change, and into what). In arm C's store (095.3: front and hand only)
an event's tries are one group at distance 0 from its own query, so "memory saw it n times" is computed exactly: the
vote without that group, plus n tries of it (its mean try weight, its own outcome). For n = 0, 1, 3, 10, 30, 100:
  B alone:  (Nc0 + n w q) / (N0 + n w)                  (no evidence at all: 0.5, counted wrong)
  A alone:  arm A's P on the two tokens
  C:        (Nc0 + n w q + alpha P_A) / (N0 + n w + alpha), alpha as fitted (095.1)
What a changed token becomes: B's relation over the remaining changed neighbours (the event's own instances scaled to
n), else A's result. Right: the predicted side of 0.5 matches the event's majority outcome for both front and hand,
and every predicted changed token is the event's. Also per action, the alpha that would maximise the n = 0
likelihood (leave-one-event-out), to see whether the prior is weighted for the few-occurrence regime.

  bin/prun python tools/card096.1/fewshot.py      → runs/096.1/fewshot.json
"""
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for c in ("card095.1", "card095.2", "card095.3"):
    sys.path.insert(0, str(ROOT / "tools" / c))
import tok                                             # noqa: E402
import arm_a as AA                                     # noqa: E402
import scale as SC                                     # noqa: E402

OUT = ROOT / "runs" / "096.1"
FRONT, HAND = 71, 169
NS = (0, 1, 3, 10, 30, 100)
ACTS = {3: "pick up", 4: "drop", 5: "toggle"}
GRID = np.logspace(-3, 4, 29)
EPS = 1e-6


def fields(u, h):
    f = np.full((2, 15), -1, np.int64)
    f[:, 2], f[:, 14] = u, h
    f[0, :2] = (u, FRONT)
    f[1, :2] = (h, HAND)
    return f


def events(tier):
    """{(a, u, h): [weight, tries, front changed weight, hand changed weight, Counter of (front after, hand after)]}"""
    z = tok.load(f"tier{tier}")
    H, P, C, A = tok.build(z)
    fi = np.argmax(P == FRONT, 1)
    ok = (P[np.arange(len(P)), fi] == FRONT)
    r = np.arange(len(P))
    u, h = np.where(ok, H[r, fi], -1), H[:, -1]
    cf, ch = np.where(ok, C[r, fi], False), C[:, -1]
    af, ah = np.where(ok, A[r, fi], -1), A[:, -1]
    act, w = z["act"], z["w"].astype(np.float64)
    ev = {}
    for i in np.flatnonzero(np.isin(act, list(ACTS)) & (u >= 0) & (h >= 0)).tolist():
        k = (int(act[i]), int(u[i]), int(h[i]))
        e = ev.get(k)
        if e is None:
            e = ev[k] = [0.0, 0, 0.0, 0.0, defaultdict(float)]
        e[0] += w[i]
        e[1] += 1
        e[2] += w[i] * cf[i]
        e[3] += w[i] * ch[i]
        if cf[i] or ch[i]:
            e[4][(int(af[i]) if cf[i] else -1, int(ah[i]) if ch[i] else -1)] += w[i]
    known = np.unique(z["app"][z["app"] >= 0])
    return ev, known, z["arr"]


@torch.no_grad()
def arm_a(net, arr, known, a, u, h):
    X = torch.as_tensor(arr[[u, h]].astype(np.float32))[None]
    logit, after, _, _ = net(X, torch.tensor([[FRONT, HAND]]), torch.ones(1, 2, dtype=torch.bool), torch.tensor([a]))
    snap = known[np.abs(arr[known][None] - after[0].numpy()[:, None]).sum(-1).argmin(1)]
    return torch.sigmoid(logit[0]).numpy().astype(np.float64), snap


def result_scaled(st, f, gi, wgt, exact, scale, known, arr):
    """Store.result with the event's own groups' instances scaled (0: left out)."""
    idx, wk = [], []
    for g, wg in zip(gi.tolist(), wgt.tolist()):
        s = scale if g in exact else 1.0
        if s == 0:
            continue
        for i in st.by_group.get(g, ()):
            idx.append(i)
            wk.append(wg * st.wch[i] * s)
    if not idx:
        return -1
    idx = np.array(idx)
    return SC.relation_result(st.arm.V, f, st.Fch[idx], st.ach[idx], np.array(wk), known, arr)


def run(tier, net, log):
    path = os.environ.get("WM_STORE", str(ROOT / "runs" / "095.3" / "store_tier{tier}.pkl")).format(tier=tier)
    data = pickle.loads(Path(path).read_bytes())
    ev, known, arr = events(tier)
    arr = arr.astype(np.float32)
    V = np.concatenate([arr, np.zeros((1, arr.shape[1]), np.float32)])
    for st in data["stores"].values():
        st.arm.V = V
    names = {}
    p = ROOT / "runs" / "095.1" / f"names_tier{tier}.json"
    if p.exists():
        names = {int(k): v for k, v in json.loads(p.read_text()).items()}
    nm = lambda x: names.get(int(x), str(int(x)))
    out = {}
    for a, an in ACTS.items():
        st, al = data["stores"][a], data["alpha"][str(a)]
        rows = []
        for (a2, u, h), (W, cnt, wf, wh, outs) in ev.items():
            if a2 != a:
                continue
            q = np.array([wf / W, wh / W])
            maj = q > 0.5
            mo = max(outs, key=outs.get) if outs else (-1, -1)
            pa, snap = arm_a(net, arr, known, a, u, h)
            f2 = fields(u, h)
            per = []
            for i in range(2):
                Nc, N, gi, wgt = st.vote(f2[i])
                ex = {int(g) for g, wg in zip(gi.tolist(), wgt.tolist()) if wg >= 1 - 1e-9}
                Gn = sum(float(st.arm.Gn[g]) for g in ex)
                Gnc = sum(float(st.arm.Gnc[g]) for g in ex)
                per.append((Nc - Gnc, N - Gn, gi, wgt, ex, Gnc))
            wbar = W / cnt
            row = {"a": a, "u": u, "h": h, "tries": cnt, "q": q.round(3).tolist(), "per_n": {}}
            for n in NS:
                res = {}
                for arm in ("B", "A", "C"):
                    P, after = np.zeros(2), [-1, -1]
                    for i in range(2):
                        Nc0, N0, gi, wgt, ex, Gnc = per[i]
                        add_n, add_c = n * wbar, n * wbar * q[i]
                        if arm == "A":
                            P[i] = pa[i]
                        elif arm == "B":
                            P[i] = (Nc0 + add_c) / (N0 + add_n) if N0 + add_n > 0 else 0.5
                        else:
                            P[i] = (Nc0 + add_c + al * pa[i]) / (N0 + add_n + al)
                        if P[i] > 0.5:
                            x = -1
                            if arm != "A":
                                scale = (add_c / Gnc) if Gnc > 0 else 0.0
                                x = result_scaled(st, f2[i], gi, wgt, ex, scale, known, arr)
                            after[i] = int(x) if x >= 0 else int(snap[i])
                    side = ((P > 0.5) == maj).all() and not (np.abs(P - 0.5) < 1e-12).any()
                    res_ok = all(not maj[i] or after[i] == mo[i] for i in range(2))
                    ll = float(np.sum(q * np.log(np.clip(P, EPS, 1)) + (1 - q) * np.log(np.clip(1 - P, EPS, 1))))
                    res[arm] = {"right": bool(side and res_ok), "side": bool(side), "ll": ll,
                                "P": P.round(4).tolist(), "after": after}
                row["per_n"][n] = res
            row["n0_alpha_parts"] = [(per[i][0], per[i][1], float(pa[i]), float(q[i])) for i in range(2)]
            rows.append(row)
        # the alpha that maximises the zero-occurrence likelihood (leave-one-event-out over this action's events)
        best = None
        for g in GRID:
            ll = 0.0
            for r in rows:
                for Nc0, N0, pa_i, q_i in r["n0_alpha_parts"]:
                    P = (Nc0 + g * pa_i) / (N0 + g)
                    ll += q_i * np.log(max(P, EPS)) + (1 - q_i) * np.log(max(1 - P, EPS))
            if best is None or ll > best[1]:
                best = (float(g), ll)
        summ = {"events": len(rows),
                "right_share": {n: {arm: round(float(np.mean([r["per_n"][n][arm]["right"] for r in rows])), 4)
                                    for arm in ("B", "A", "C")} for n in NS},
                "ll_mean": {n: {arm: round(float(np.mean([r["per_n"][n][arm]["ll"] for r in rows])), 4)
                                for arm in ("B", "A", "C")} for n in NS},
                "alpha_fitted": al, "alpha_best_at_n0": round(best[0], 4),
                "wrong_at_n0_C": [f"{nm(r['u'])} | {nm(r['h'])} (q {r['q']}, P {r['per_n'][0]['C']['P']}, after "
                                  f"{[nm(x) if x >= 0 else None for x in r['per_n'][0]['C']['after']]})"
                                  for r in rows if not r["per_n"][0]["C"]["right"]][:40],
                "wrong_at_n1_C": [f"{nm(r['u'])} | {nm(r['h'])} (q {r['q']}, P {r['per_n'][1]['C']['P']})"
                                  for r in rows if not r["per_n"][1]["C"]["right"]][:20]}
        out[an] = summ
        log(f"tier {tier} {an}: {len(rows)} events; right at n = {NS}: "
            f"{[(n, summ['right_share'][n]) for n in NS]}; alpha {al} (best at n=0: {best[0]})")
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    net = AA.StateNet()
    net.load_state_dict(torch.load(tok.RUNS / "arm_a.pt", map_location="cpu")["net"])
    net.eval()
    torch.set_num_threads(4)
    rep = {f"tier{t}": run(t, net, log) for t in (int(x) for x in (sys.argv[1:] or ["2", "3"]))}
    (OUT / "fewshot.json").write_text(json.dumps(rep, indent=1) + "\n")
