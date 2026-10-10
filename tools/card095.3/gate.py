"""Card 095.3's feasibility gate (offline, on 095.1's held-out tries): arm C with only the token in front and the
hand known. Arm B's conditions on other context places are left out of the distance (the store regrouped on the
rest: marginalised); arm A sees only the front's and the hand's tokens. Every other token is predicted unchanged.
Compared with 095.1's arm C (full context): exact next state, change log-likelihood, and tier 2's counterfactual
boxes.

  bin/prun python tools/card095.3/gate.py      → runs/095.3/gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
sys.path.insert(0, str(ROOT / "tools" / "card095.2"))
import tok                                             # noqa: E402
import arm_a as AA                                     # noqa: E402
import arm_b as AB                                     # noqa: E402
import scale as SC                                     # noqa: E402

OUT = ROOT / "runs" / "095.3"
FRONT, HAND = 71, 169
KNOWN_COLS = (0, 1, 2, 14)                             # own token, place, context 0 (the front), context 12 (the hand)
BOXES = SC.BOXES
dev = AA.dev
log = lambda m: print(m, flush=True)


def reduced(H, P):
    """Only the front's and the hand's tokens kept."""
    keep = (P == FRONT) | (P == HAND)
    return np.where(keep, H, -1)


def marginal(arm):
    """Arm B with conditions only on the known columns (the same λ), regrouped."""
    keep = [i for i, c in enumerate(arm.adm) if arm.col(c) in KNOWN_COLS]
    arm.adm = [arm.adm[i] for i in keep]
    arm.lam = [arm.lam[i] for i in keep]
    arm.G, arm.Gnc, arm.Gn, _, _ = arm.groups([arm.col(c) for c in arm.adm])
    return arm


def mask_fields(F):
    F = F.copy()
    F[:, [j for j in range(F.shape[1]) if j not in KNOWN_COLS]] = -1
    return F


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    net = AA.StateNet().to(dev)
    net.load_state_dict(torch.load(tok.RUNS / "arm_a.pt")["net"])
    net.eval()
    rep = {}
    for world in ("tier1", "tier2", "tier3"):
        z = tok.load(world)
        H, P, C, A = tok.build(z)
        tr, te = tok.split(z)
        near_n = int((np.abs(z["wh"][:169]).sum(1) <= tok.DMAX).sum()) - 1
        F, r, c = AB.fields(H, P, near_n)
        ch, af, w = C[r, c], A[r, c], z["w"][r].astype(np.float64)
        act, train = z["act"][r], tr[r]
        arr, known = z["arr"], np.unique(z["app"])
        arr_t = torch.as_tensor(arr, device=dev)
        alpha = json.loads((tok.RUNS / "c_alpha.json").read_text())[world]
        b_rep = json.loads((tok.RUNS / f"b_{world}.json").read_text())
        cz = np.load(tok.RUNS / f"c_{world}.npz")
        held = np.flatnonzero(~train)
        Hr = reduced(H, P)
        pA, aA, _ = SC.a_states(net, arr_t, known, Hr, P, z["act"])
        # 095.1's arm C on held-out tries, every token (the reference)
        p_full, a_full = np.zeros(len(F)), np.full(len(F), -1)
        p_full[held], a_full[held] = cz["p"], cz["after"]
        p_red, a_red = np.zeros(len(F)), np.full(len(F), -1)
        Fm = mask_fields(F)
        adm = {}
        for a in (3, 4, 5):
            al = alpha[str(a)]
            s = np.flatnonzero(train & (act == a))
            arm = marginal(SC.rebuild_arm(arr, F[s], w[s], ch[s], b_rep[str(a)]["admitted"]))
            adm[str(a)] = [str(x) for x in arm.adm]
            q = np.flatnonzero(~train & (act == a) & ((P[r, c] == FRONT) | (P[r, c] == HAND)))
            uq, qinv = np.unique(Fm[q], axis=0, return_inverse=True)
            qinv = qinv.ravel()
            _, t = arm.p_change(uq, totals=True)
            pa = pA[r[q], c[q]]
            p_red[q] = (t[qinv, 0] + al * pa) / (t[qinv, 1] + al)
            need = np.unique(qinv[(p_red[q] > 0.5) | ch[q]])
            sc = s[ch[s]]
            res = np.full(len(uq), -1)
            if len(need):
                res[need] = AB.results(arm, uq[need], F[sc], w[sc], af[sc], known, arr)[0]
            a_red[q] = np.where(res[qinv] >= 0, res[qinv], aA[r[q], c[q]])
            log(f"{world} action {a}: admitted kept {adm[str(a)]}, groups {len(arm.G)}")
        te_idx = np.flatnonzero(te)
        rows = np.searchsorted(te_idx, r[held])
        Hs, Cs, As, ws = H[te], C[te], A[te], z["w"][te]
        out = {"admitted (known columns)": adm}
        for name, (p, af_) in (("full context (095.1)", (p_full, a_full)), ("front and hand only", (p_red, a_red))):
            pm, am = np.zeros(Hs.shape), np.full(Hs.shape, -1)
            pm[rows, c[held]] = p[held]
            am[rows, c[held]] = af_[held]
            out[name] = SC.score(Hs, Cs, As, ws, pm, am)
        log(f"{world}: {json.dumps(out)}")
        if world == "tier2":                           # counterfactual boxes, as 095.1
            fs = int(np.flatnonzero(P[0] == FRONT)[0])
            cand = np.flatnonzero(te & (z["act"] == 3) & (H[:, -1] == 0) & (H[:, fs] >= 0))
            Q = np.random.default_rng(0).choice(cand, min(300, len(cand)), replace=False)
            s = np.flatnonzero(train & (act == 3))
            arm = marginal(SC.rebuild_arm(arr, F[s], w[s], ch[s], b_rep["3"]["admitted"]))
            sc = s[ch[s]]
            al = alpha["3"]
            cf = {}
            for col, bx in BOXES.items():
                Hq = H[Q].copy()
                Hq[:, fs] = bx
                pq, aq, _ = SC.a_states(net, arr_t, known, reduced(Hq, P[Q]), P[Q], np.full(len(Q), 3))
                Fq, _, _ = AB.fields(Hq, P[Q], near_n)
                hf = mask_fields(Fq[Fq[:, 1] == HAND])
                _, tb = arm.p_change(hf, totals=True)
                rb = AB.results(arm, hf, F[sc], w[sc], af[sc], known, arr)[0]
                p = (tb[:, 0] + al * pq[:, -1]) / (tb[:, 1] + al)
                res = np.where(rb >= 0, rb, aq[:, -1])
                cf[col] = round(float(((res == bx) & (p > 0.5)).mean()), 4)
            out["counterfactual boxes: hand holds that box"] = cf
            log(f"counterfactual: {cf}")
        rep[world] = out
    (OUT / "gate.json").write_text(json.dumps(rep, indent=1) + "\n")
