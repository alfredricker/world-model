"""Card 095.1's baseline: version 20's recall on the held-out tries, with the held-out episodes' interaction tries
left out of its memory (moves keep every row; they do not enter pick up, toggle or drop).

Per held-out try: P(front changes), P(hand changes) from the category probabilities (recall's predict: own tries
first, card 091.1's router as the prior), and the predicted tokens after (W.outcome: the most likely category and
its result). Every other token is predicted unchanged, with certainty (version 20's two slots).

  <version 20's flags> bin/prun python tools/card095.1/baseline.py tier2     → runs/095.1/v20_tier2.npz
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORLD = sys.argv[1]
TIER = int(WORLD[-1])
sys.argv = ["run.py", "tiers", "--tier", str(TIER), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
sys.path.insert(0, str(ROOT / "tools" / "card095.1"))
import run as RUN                                      # noqa: E402
import tok                                             # noqa: E402

T = RUN.T
VP = T.VP
Z = tok.load(WORLD)
TRAIN, TEST = tok.split(Z)
_memory = T.memory


def memory(tier, pool=None):
    D, stats = _memory(tier, pool)
    inter = np.flatnonzero(np.isin(D["act"], T.INTER))
    assert len(inter) == len(TEST)
    drop = np.zeros(len(D["act"]), bool)
    drop[inter[TEST]] = True
    keep_t = ~TEST
    return {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}, stats


T.memory = memory

if __name__ == "__main__":
    t0 = time.monotonic()
    W, info = T.setup(TIER, lambda m: None)
    APP = np.asarray(T.Kd.APP, np.int64)
    e0 = Z["e0"].astype(np.int64)
    pres = Z["pres"]
    idx = np.flatnonzero(TEST)
    ctx = {}
    qs = []
    for i in idx:
        key = pres[i].tobytes()
        c = ctx.get(key)
        if c is None:
            c = ctx[key] = int(W.ctx_of(APP[np.flatnonzero(pres[i])]))
        qs.append((int(APP[e0[i, T.FRONT]]), int(APP[e0[i, T.HELD]]), c))
    # the same contexts as memory's own keys (checked on the training tries' keys)
    chk = [int(W.ctx_of(APP[np.flatnonzero(pres[i])])) for i in np.flatnonzero(TRAIN)[:300]]
    known = {int(k[2]) for a in T.INTER for k in W.kinds[a].keys}
    print("context ids of training tries found among memory's keys:", round(np.mean([c in known for c in chk]), 3),
          flush=True)
    act = Z["act"][idx]
    pf, ph = np.zeros(len(idx)), np.zeros(len(idx))
    af, ah = np.full(len(idx), -1), np.full(len(idx), -1)
    for a in T.INTER:
        sel = np.flatnonzero(act == a)
        kd = W.kinds[a]
        uq = list(dict.fromkeys(qs[j] for j in sel))
        P = {}
        for s in range(0, len(uq), 512):
            B = uq[s:s + 512]
            for q, p in zip(B, kd.predict(B)):
                P[q] = p
        for j in sel:
            p = P[qs[j]]
            pf[j], ph[j] = p[1] + p[3], p[2] + p[3]
            c, (fa, ha) = W.outcome(a, *qs[j])
            af[j] = -1 if fa is None else fa
            ah[j] = -1 if ha is None else ha
        print("action", a, "held-out tries", len(sel), "distinct queries", len(uq), round(time.monotonic() - t0, 1),
              "s", flush=True)
    np.savez_compressed(tok.RUNS / f"v20_{WORLD}.npz", idx=idx, pf=pf, ph=ph, af=af, ah=ah)
    print(WORLD, "saved", round(time.monotonic() - t0, 1), "s", flush=True)
