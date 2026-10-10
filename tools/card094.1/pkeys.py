"""Card 094.1: the router's stored keys with positions, per world.

Memory (card 066) keeps each stored try's full 13 x 13 window (ego0, ego1) and the *set* of tokens in the agent's
believed view (pres), not their places. The play is replayed here with the same seeds and sampling, and card 062's
belief replay is run with each believed place kept; the replayed sets must equal the stored ones (checked). Then
per pick up, toggle and drop:

  key = (front token, held token, the attended tokens of the believed view with their place, nearest first, <= 6)

attended as card 094 (memory or card 087 says some action changes it, or it ends the episode), the front's own place
and the agent's aside; keys equal in all three are merged, with their outcome counts (bit 0 the front changed, bit 1
the hand changed; weighted as memory weights the try). Forward's keys are card 091's (front tile only).

  <version 20's flags> bin/prun python tools/card094.1/pkeys.py tier2     → runs/094.1/pkeys_tier2.npz
  (decoy: card 070's decoy world, under tier 1)
"""
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORLD = sys.argv[1]
TIER = int(WORLD[-1]) if WORLD.startswith("tier") else 1
sys.argv = ["run.py", "tiers", "--tier", str(TIER), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

T = RUN.T
VP = T.VP
if WORLD == "decoy":                                   # card 070's decoy world (tools/card070/why_fold.py)
    T.ENVS[1] = RUN.R.register()
    T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
    RUN.R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
sys.path.insert(0, str(ROOT / "tools" / "card094"))
import files as FILES                                  # noqa: E402
FILES.STATE["VP"] = VP
BL = T.BL
VMAX = 6
OUT = ROOT / "runs" / "094.1"
SIDE_ROWS = []


def believe_pos(ego, act, rows):
    """card 062's believe(), keeping each wanted row's believed codes per place (NV) as well."""
    vis = BL.M61.visible(ego)
    seen = np.where(vis, ego, -1)
    B = np.full((BL.SIDE, BL.SIDE), -1, np.int64)
    Rm, t = np.eye(2, dtype=np.int64), np.zeros(2, np.int64)
    want = {int(r): i for i, r in enumerate(rows)}
    pres = np.zeros((len(rows), BL.NCODES), bool)
    stale = np.zeros(len(rows), np.int64)
    pos = np.zeros((len(rows), BL.NV), np.int16)
    for s in range(len(act) + 1):
        q = np.flatnonzero(vis[s, :BL.NV])
        wf = (BL.WH[q] - t) @ Rm
        v = ego[s, q].copy()
        c = q == BL.CENTRE
        v[c] = (v[c] // 5) * 5
        B[wf[:, 0] + BL.OFF, wf[:, 1] + BL.OFF] = v
        i = want.get(s)
        if i is not None:
            wa = (BL.WH - t) @ Rm
            bv = B[wa[:, 0] + BL.OFF, wa[:, 1] + BL.OFF]
            shown = np.where(bv >= 0, bv, BL.WALL)
            keep = np.ones(BL.NV, bool)
            keep[BL.CENTRE] = False
            pres[i, shown[keep]] = True
            known = (bv >= 0) & keep
            stale[i] = int((bv[known] != ego[s, :BL.NV][known]).sum())
            pos[i] = np.where(bv >= 0, bv, -1)          # -1: never seen (shown as the unseen appearance)
        if s == len(act):
            break
        a = int(act[s])
        if a in (BL.LEFT, BL.RIGHT) or (a == BL.FWD and (seen[s, :BL.NV] != seen[s + 1, :BL.NV]).any()):
            Ma, ba = BL.T[a]
            Rm, t = Ma @ Rm, Ma @ t + ba
    SIDE_ROWS.append(pos)
    return pres, stale


def job(j):
    SIDE_ROWS.clear()
    D, stats = T._play(j)
    return D, stats, np.concatenate(SIDE_ROWS) if SIDE_ROWS else np.zeros((0, BL.NV), np.int16)


def replay(tier):
    BL.moves_from_card061()
    BL.believe = believe_pos
    env = T.make(tier)
    n_ep = T.MEMORY_STEPS // env.max_steps
    seeds = T.SEED_MEMORY + 100_000 * tier + np.arange(n_ep)
    jobs = [(tier, s.tolist(), env.max_steps) for s in np.array_split(seeds, 80)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(20) as pool:
        parts = pool.map(job, jobs)
    D = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    return D, np.concatenate([p[2] for p in parts])


def build(W, D, POS):
    A = W.S.arr.astype(np.float32)
    APP = T.Kd.APP
    CR = sys.modules["code_recall"]
    WH = BL.WH
    order = np.lexsort((np.arange(len(WH)), np.abs(WH).sum(1)))      # nearest first (steps from the agent)
    order = [p for p in order.tolist() if p not in (BL.CENTRE, T.FRONT)]
    inter = np.flatnonzero(np.isin(D["act"], T.INTER))
    att = {}
    out = {}
    for a in (VP.PICK, VP.DROP, VP.TOG):
        acc = defaultdict(lambda: np.zeros(4))
        rows = np.flatnonzero(D["act"][inter] == a)
        for r in rows:
            g = inter[r]
            e0, e1 = D["ego0"][g].astype(np.int64), D["ego1"][g].astype(np.int64)
            fh, hh = int(APP[e0[T.FRONT]]), int(APP[e0[T.HELD]])
            cat = int(e1[T.FRONT] // 5 != e0[T.FRONT] // 5) | (int(e1[T.HELD] // 5 != e0[T.HELD] // 5) << 1)
            toks = []
            for p in order:
                c = int(POS[r, p])
                if c < 0:
                    continue
                h = int(APP[c])
                ok = att.get(h)
                if ok is None:
                    ok = att[h] = bool(FILES.static(W, h))
                if ok:
                    toks.append((h, p))
                    if len(toks) == VMAX:
                        break
            acc[(fh, hh, tuple(toks))][cat] += float(D["w"][g])
        keys = list(acc)
        n = len(keys)
        V = np.zeros((n, VMAX, A.shape[1]), np.float32)
        vm = np.zeros((n, VMAX), bool)
        P = np.zeros((n, VMAX), np.int64)
        Vh = np.full((n, VMAX), -1, np.int64)
        for i, (_, _, toks) in enumerate(keys):
            for j, (h, p) in enumerate(toks):
                V[i, j], vm[i, j], P[i, j], Vh[i, j] = A[h], True, p, h
        fh = np.array([k[0] for k in keys])
        hh = np.array([k[1] for k in keys])
        out[f"{a}_F"], out[f"{a}_H"], out[f"{a}_V"], out[f"{a}_vm"], out[f"{a}_P"] = A[fh], A[hh], V, vm, P
        out[f"{a}_C"] = np.stack([acc[k] for k in keys])
        out[f"{a}_fh"], out[f"{a}_hh"], out[f"{a}_Vh"] = fh, hh, Vh
        out[f"{a}_fid"] = np.array([CR.code_id(W.S, int(x)) for x in fh])
        out[f"{a}_hid"] = np.array([CR.code_id(W.S, int(x)) for x in hh])
        kd = W.kinds[a]
        print(f"action {a}: tries {len(rows)} keys {n} (set form: {len(kd.keys)}); outcome totals "
              f"{out[f'{a}_C'].sum(0).round(1).tolist()}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.monotonic()
    W, info = T.setup(TIER, lambda m: None)
    z = np.load(T.OUT / f"memory_tier{TIER}.npz")
    D, POS = replay(TIER)
    same = {k: bool(np.array_equal(D[k], z[k])) for k in ("act", "ego0", "pres")}
    print("replay equals stored memory:", same, "rows", len(D["act"]), "tries", len(POS), flush=True)
    assert all(same.values()), same
    out = build(W, D, POS)
    k91 = np.load(ROOT / "runs" / "091" / f"keys_{WORLD}.npz")
    for k in ("2_F", "2_C", "2_fid"):
        out[k] = k91[k]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"pkeys_{WORLD}.npz", **out)
    print(WORLD, "saved", round(time.monotonic() - t0, 1), "s", flush=True)
