"""Card 095.1: version 20's recall, arm A and arm B on the same held-out tries (every tenth episode).

Per tier, weighted by memory's try weights:
- **change log-likelihood:** per try, summed over the state's tokens, of which tokens changed (version 20 predicts
  the front and the hand from its categories and every other token unchanged, with certainty);
- **exact next state:** the share of tries where every token's predicted change (p > 0.5) is right and every
  changed token's predicted result is the token observed;
- **roles:** mean predicted change at places other than the front and the hand, where none happened; the hand's
  result on a pick up copied from the front (arm A: copy attention on the front's slot above 0.5; arm B: the
  chosen place);
- **box check (tier 2):** pick ups of a box with an empty hand that put it in the hand: the predicted hand after
  is the box in front.

  bin/prun python tools/card095.1/evaluate.py      → runs/095.1/results.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
import tok                                             # noqa: E402

FRONT, HAND = 71, 169
EPS = 1e-6
ACT = {3: "pick up", 4: "drop", 5: "toggle"}


def ll_of(p, C, mask):
    p = np.clip(p, EPS, 1 - EPS)
    return (np.where(C, np.log(p), np.log(1 - p)) * mask).sum(1)


def tier_report(world):
    z = tok.load(world)
    H, P, C, A = tok.build(z)
    tr, te = tok.split(z)
    idx = np.flatnonzero(te)
    H, P, C, A = H[idx], P[idx], C[idx], A[idx]
    w, act = z["w"][idx], z["act"][idx]
    mask = H >= 0
    fslot, hslot = (P == FRONT) & mask, (P == HAND) & mask
    preds = {}
    # version 20
    vf = tok.RUNS / f"v20_{world}.npz"
    v = np.load(vf) if vf.exists() else None           # tier 3: not run (setup without the held-out episodes > 1 h)
    if v is not None:
        assert np.array_equal(v["idx"], idx)
    p = np.zeros(H.shape)
    af = np.full(H.shape, -1)
    if v is not None:
        p[fslot] = np.repeat(v["pf"], fslot.sum(1))
        p[hslot] = np.repeat(v["ph"], hslot.sum(1))
        af[fslot] = np.repeat(v["af"], fslot.sum(1))
        af[hslot] = np.repeat(v["ah"], hslot.sum(1))
    if v is not None:
        preds["version 20"] = (p, af >= 0, af)
    # arm A
    a = np.load(tok.RUNS / f"a_{world}.npz")
    assert np.array_equal(a["idx"], idx)
    preds["arm A (network)"] = (a["p"], a["p"] > 0.5, a["after"])
    # arm B
    b = np.load(tok.RUNS / f"b_{world}.npz")
    pos = np.searchsorted(idx, b["r"])
    assert np.array_equal(idx[pos], b["r"])
    pb = np.zeros(H.shape)
    ab = np.full(H.shape, -1)
    pb[pos, b["c"]] = b["p"]
    ab[pos, b["c"]] = b["after"]
    preds["arm B (exemplars)"] = (pb, pb > 0.5, ab)
    cf = tok.RUNS / f"c_{world}.npz"
    if cf.exists():                                    # arm C: B's vote with A as its prior
        cc = np.load(cf)
        pos = np.searchsorted(idx, cc["r"])
        pc_, ac_ = np.zeros(H.shape), np.full(H.shape, -1)
        pc_[pos, cc["c"]] = cc["p"]
        ac_[pos, cc["c"]] = cc["after"]
        preds["arm C (B with A as prior)"] = (pc_, pc_ > 0.5, ac_)
    names = {}
    nf = tok.RUNS / f"names_{world}.json"
    if nf.exists():
        names = {int(k): v for k, v in json.loads(nf.read_text()).items()}
    rep = {"held-out tries": int(len(idx))}
    for name, (p, ch, after) in preds.items():
        r = {}
        ll = ll_of(p, C, mask)
        right = ((ch == C) | ~mask).all(1) & ((~C) | (after == A)).all(1)
        r["change log-likelihood per try"] = round(float((ll * w).sum() / w.sum()), 5)
        r["exact next state"] = round(float((right * w).sum() / w.sum()), 5)
        r["by action"] = {}
        for x, nm in ACT.items():
            s = act == x
            r["by action"][nm] = {"change log-likelihood": round(float((ll[s] * w[s]).sum() / w[s].sum()), 5),
                                  "exact": round(float((right[s] * w[s]).sum() / w[s].sum()), 5)}
        other = mask & ~fslot & ~hslot & ~C
        r["mean predicted change elsewhere"] = round(float(p[other].mean()), 6)
        pick = (act == 3) & (C & hslot).any(1)
        hand_after = np.where(hslot, after, -1).max(1)
        true_after = np.where(hslot, A, -1).max(1)
        front_tok = np.where(fslot, H, -1).max(1)
        r["pick up: hand after is the token in front"] = round(float((hand_after[pick] == front_tok[pick]).mean()), 4)
        if names:
            box = pick & np.array([names.get(int(h), "").startswith("box") for h in front_tok])
            r["box check: tries"] = int(box.sum())
            r["box check: hand after is the box in front"] = round(float((hand_after[box] == front_tok[box]).mean()), 4) \
                if box.any() else None
            wrong = box & (hand_after != front_tok)
            r["box check: wrong predictions"] = sorted({f"{names.get(int(f), f)} -> {names.get(int(h), h)}"
                                                        for f, h in zip(front_tok[wrong], hand_after[wrong])})[:6]
        if name == "arm A (network)":
            pf = a["ptr_front"][hslot]
            pk = (act[:, None] == 3) & hslot & C
            r["pick up: hand copies the front (attention > 0.5)"] = round(float((a["ptr_front"][pk] > 0.5).mean()), 4)
        rep[name] = r
    return rep


if __name__ == "__main__":
    out = {w: tier_report(w) for w in ("tier1", "tier2", "tier3")}
    for w in ("tier1", "tier2", "tier3"):
        b = json.loads((tok.RUNS / f"b_{w}.json").read_text())
        out[w]["arm B (exemplars)"]["admitted"] = {ACT[int(a)]: [x["cond"] for x in b[a]["admitted"]] for a in "345"}
        out[w]["arm B (exemplars)"]["context places"] = b["context_places"]
        out[w]["arm B (exemplars)"]["pick up: hand's copy source (context index: count)"] = b["3"]["copy_source_on_hand_tokens"]
        if (tok.RUNS / "c_alpha.json").exists():
            out[w]["arm C (B with A as prior)"]["alpha per action"] = json.loads((tok.RUNS / "c_alpha.json").read_text())[w]
    (tok.RUNS / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
