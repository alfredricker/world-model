"""Card 061: the move model and undraw learned from the 7 x 7 view with occlusion.

Version 12 acts on MiniGrid's 7 x 7 view with occlusion, but its starting memory was learned from play seen through
the 13 x 13 window that holds the whole map (card 057's declared exception). Here the play is seen as the agent
would see it (the same crop and occlusion as acting), and two parts of memory are learned from that alone:

  moves:  card 028's counted correspondences, counting only rows where both places were observed; a place's source
          is kept only where it predicts the place exactly in at least MIN_ROWS rows and shows more than one
          appearance there (a pair that is always wall says nothing about where a place went). Card 044's least squares then gives each move's transformation of where, which
          holds for every where, seen or not.
  undraw: how a tile looks once the agent steps off it. The full view read it behind the agent after a forward
          step; the 7 x 7 view has nothing behind. Here it is read from the same forward steps the other way round:
          the agent on X at the centre after, and X ahead before.

What a place with no token shows (card 044's "unseen" appearance) is the commonest appearance a place without a
source shows when it comes into view. Pick up, toggle and drop keep what was in view in the full window (declared
exception, card 062's subject). The full-view fit is computed alongside, for the report only.

  bin/prun python tools/card061/moves.py tools/card057/partial.py runs/054/b_m0.5_399.pt --arm B --occlude ...
"""
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card057"))
sys.path.insert(0, str(TOOLS / "card053"))
sys.path.insert(0, str(TOOLS / "card051"))
sys.path.insert(0, str(TOOLS / "card049"))
import planner_check as PC                             # noqa: E402
import walk2                                           # noqa: E402

MV = walk2.CM.S7.MV
VP, F, TK = MV.VP, MV.F, MV.TK
NV, NPL, W13, CENTRE, FRONT, HELD, BEHIND = VP.NV, VP.NPL, VP.W13, VP.CENTRE, VP.FRONT, VP.HELD, VP.BEHIND
MOVES, FWD, INTER = VP.MOVES, VP.FWD, VP.INTER
ENDED, CHANGED, UNCHANGED = VP.ENDED, VP.CHANGED, VP.UNCHANGED
R = (W13 - 1) // 2
KR = PC.KR
SENT = 255                                             # a hidden place (never an appearance's handle)
MIN_ROWS = 50
REPORT = {}


def _blocks():
    b = np.zeros(256, bool)
    for c in range(len(KR.OBJECTS) * 5):
        o = KR.OBJECTS[c // 5]
        b[c] = o == "wall" or (isinstance(o, tuple) and o[0] == "door" and o[2] != 2)
    return b


BLK = _blocks()


def visible(codes):
    """MiniGrid's process_vis on the 7 x 7 view, for many egocentric views at once (renderer codes, N x NPL):
    the places observed, N x NPL (the held row always)."""
    codes = np.asarray(codes)
    N = len(codes)
    E = codes[:, :NV].reshape(N, W13, W13)[:, R - 6:R + 1, R - 3:R + 4]     # [n, j, i], j = 6 at the agent
    blk = BLK[E]
    m = np.zeros((N, 7, 7), bool)                                            # [n, i, j]
    m[:, 3, 6] = True
    for j in reversed(range(7)):
        for i in range(6):
            go = m[:, i, j] & ~blk[:, j, i]
            m[:, i + 1, j] |= go
            if j > 0:
                m[:, i + 1, j - 1] |= go
                m[:, i, j - 1] |= go
        for i in reversed(range(1, 7)):
            go = m[:, i, j] & ~blk[:, j, i]
            m[:, i - 1, j] |= go
            if j > 0:
                m[:, i - 1, j - 1] |= go
                m[:, i, j - 1] |= go
    vv = np.zeros((N, W13, W13), bool)
    vv[:, R - 6:R + 1, R - 3:R + 4] = m.transpose(0, 2, 1)
    vis = np.ones((N, NPL), bool)
    vis[:, :NV] = vv.reshape(N, NV)
    return vis


def fit_map_partial(V0, V1, vis0, vis1, dev):
    """Card 028's counted map over observed pairs only: for each place after, the place before whose appearance
    predicts it exactly wherever both were observed (at least MIN_ROWS rows, p showing more than one appearance
    there); -1 if none. Also the commonest observed appearance of each place without a source."""
    import torch
    apps = np.unique(np.concatenate([V0[vis0], V1[vis1]]))
    K = len(apps)
    lut = np.full(256, K, np.int64)
    lut[apps] = np.arange(K)
    a0 = np.where(vis0, lut[V0], K)
    a1 = np.where(vis1, lut[V1], K)
    N, P = V0.shape
    C = torch.zeros(P * K, P * K, device=dev)
    for s in range(0, N, 2048):
        o0 = torch.nn.functional.one_hot(torch.as_tensor(a0[s:s + 2048], device=dev), K + 1)[..., :K].float()
        o1 = torch.nn.functional.one_hot(torch.as_tensor(a1[s:s + 2048], device=dev), K + 1)[..., :K].float()
        C += o0.reshape(len(o0), P * K).T @ o1.reshape(len(o1), P * K)
    C4 = C.reshape(P, K, P, K)
    best = C4.amax(3).sum(1)                           # [p, q]
    npq = C4.sum((1, 3))                               # [p, q]: rows where both were observed
    varies = (C4.sum(3) > 0).sum(1) > 1               # [p, q]: p shows more than one appearance where both were observed
    exact = (best == npq) & (npq >= MIN_ROWS) & varies
    score = torch.where(exact, npq, torch.full_like(npq, -1.0))
    src = score.argmax(0).cpu().numpy().astype(np.int16)
    rows_q = score.amax(0).cpu().numpy()
    src[rows_q < 0] = -1
    src = trimmed(src, rows_q)
    cnt = np.zeros((P, K + 1))
    for q in range(P):
        cnt[q] = np.bincount(a1[:, q], minlength=K + 1)
    ent = apps[cnt[:, :K].argmax(1)].astype(np.uint8)
    seen_q = cnt[:, :K].sum(1) > 0
    return src, ent, seen_q, cnt[:, :K], apps


def trimmed(src, rows_q):
    """A move sends every where by one transformation (card 044), so the correspondences must agree on one: fit
    it by least squares weighted by the rows behind each correspondence, drop the worst-fitting one, and repeat
    until all that remain fit within half a place."""
    WH = TK.WH
    q = np.flatnonzero(src[:NV] >= 0)
    q = q[src[q] < NV]
    while len(q) >= 3:
        X = np.concatenate([WH[src[q]], np.ones((len(q), 1))], 1).astype(np.float64)
        sw = np.sqrt(rows_q[q])[:, None]
        sol = np.linalg.lstsq(X * sw, WH[q].astype(np.float64) * sw, rcond=None)[0]
        err = np.abs(X @ sol - WH[q]).max(1)
        if err.max() < 0.5:
            break
        q = np.delete(q, err.argmax())
    out = np.full_like(src, -1)
    out[q] = src[q]
    out[NV:] = src[NV:]
    return out


def world_init(self, S, parts, D, dev, log):
    """Card 038's World.__init__ with the moves and undraw from the 7 x 7 view (card 061)."""
    t0 = time.monotonic()
    self.S, self.parts, self.judge = S, parts, VP.Judge(S)
    self.keep = np.ones(NV, bool)
    self.keep[CENTRE] = False
    self.ctxc, self.openc, self.openedc = {}, {}, {}
    lut = S.hof.astype(np.uint8)
    assert lut.max() < SENT
    V0, V1 = lut[D["ego0"]], lut[D["ego1"]]
    vis0, vis1 = visible(D["ego0"]), visible(D["ego1"])
    act, w, term = D["act"], D["w"], D["term1"]
    P0, P1 = np.where(vis0, V0, SENT), np.where(vis1, V1, SENT)            # what the agent sees
    vc = (P0[:, :NV] != P1[:, :NV]).any(1)
    vc_full = (V0[:, :NV] != V1[:, :NV]).any(1)
    out = np.where(term, ENDED, np.where(vc, CHANGED, UNCHANGED))
    out_full = np.where(term, ENDED, np.where(vc_full, CHANGED, UNCHANGED))
    rep = {"rows": int(len(act)), "observed_share": round(float(vis0[:, :NV].mean()), 4),
           "move_outcome_differs_from_full_view": {VP.KNAME[a]: int((out != out_full)[act == a].sum()) for a in MOVES}}
    m = F.Model()
    m_full = F.Model()
    rng = np.random.default_rng(0)
    m.src, m.ent, m_full.src, m_full.ent = {}, {}, {}, {}
    for a in MOVES:
        rows = np.flatnonzero((act == a) & (out != UNCHANGED))
        if len(rows) > 40000:
            rows = np.sort(rng.choice(rows, 40000, replace=False))
        src, ent, seen_q, cnt, apps = fit_map_partial(V0[rows], V1[rows], vis0[rows], vis1[rows], dev)
        if os.environ.get("DUMP61") and a == MOVES[0]:
            np.savez_compressed(os.environ["DUMP61"], V0=V0[rows], V1=V1[rows], vis0=vis0[rows], vis1=vis1[rows],
                                src=src, c0=D["ego0"][rows], c1=D["ego1"][rows])
        src[NV:] = np.arange(NV, NPL)                  # the held row moves with the agent (card 044: fixed beside the hand)
        m.src[a], m.ent[a] = src, ent
        REPORT.setdefault("sources_per_move", {})[VP.KNAME[a]] = int((src[:NV] >= 0).sum())
        rows_f = np.flatnonzero((act == a) & (out_full != UNCHANGED))
        if len(rows_f) > 40000:
            rows_f = np.sort(np.random.default_rng(0).choice(rows_f, 40000, replace=False))
        sf, ef, _, _, _, _ = F.fit_map(V0[rows_f], V1[rows_f], dev)
        m_full.src[a], m_full.ent[a] = sf, ef
    T, trep = TK.fit_moves(m)
    Tf, _ = TK.fit_moves(m_full)
    rep["moves"] = trep
    print("card 061 fit", trep, REPORT.get("sources_per_move"), flush=True)
    rep["same_transformations_as_full_view"] = all(np.array_equal(T[a][0], Tf[a][0]) and np.array_equal(T[a][1], Tf[a][1])
                                                   for a in MOVES)
    rep["correspondences_used"] = REPORT.get("sources_per_move")
    # what a where with no token shows: the commonest appearance of places entering the view without a source
    allc = None
    for a in MOVES:
        rows = np.flatnonzero((act == a) & (out != UNCHANGED))
        q = np.flatnonzero(m.src[a][:NV] < 0)
        v = P1[rows][:, q].ravel()
        v = v[v != SENT]
        c = np.bincount(v, minlength=256)
        allc = c if allc is None else allc + c
    unseen = int(allc.argmax())
    for a in MOVES:                                    # build_tokens takes the commonest entering appearance
        e = np.asarray(m.ent[a]).copy()
        e[:NV][np.asarray(m.src[a])[:NV] < 0] = unseen
        m.ent[a] = e
    m.change_places = np.array([FRONT, HELD])
    F.build_poses(m)
    uf, nf = np.unique(np.concatenate([np.asarray(m_full.ent[a])[:NV][np.asarray(m_full.src[a])[:NV] < 0]
                                       for a in MOVES]), return_counts=True)
    unseen_full = int(uf[nf.argmax()])
    rep["unseen_appearance"] = {"partial": unseen, "full_view": unseen_full, "same": unseen == unseen_full}
    assert all(h == HELD for h in m.hidx), "the held place moves with the pose"
    self.M = m
    # what is in view, per stored try: the full window (declared exception: card 062)
    sel = np.flatnonzero(np.isin(act, INTER))
    pres = np.zeros((len(sel), S.nobs), bool)
    pres[np.arange(len(sel))[:, None], V0[sel, :NV][:, self.keep]] = True
    up, pinv = np.unique(pres, axis=0, return_inverse=True)
    pc = np.array([self.ctx_of(np.flatnonzero(r)) for r in up], np.int64)
    cid = np.zeros(len(act), np.int64)
    cid[sel] = pc[pinv.ravel()]
    assert not (P0[np.isin(act, MOVES), FRONT] == SENT).any(), "the place ahead is always observed"
    k = {}
    for a in MOVES:
        s = np.flatnonzero(act == a)
        k[a] = VP.Kind(VP.KNAME[a], S, parts, 1, 3, V0[s, FRONT][:, None], out[s], w[s])
    fw = np.flatnonzero((act == FWD) & (out != UNCHANGED))
    lamf = k[FWD].lam
    k["draw"] = VP.Kind("draw", S, parts, 1, 1, V0[fw, FRONT][:, None], np.zeros(len(fw), np.int64), w[fw],
                        after=V1[fw, CENTRE][:, None], other=V0[fw, CENTRE], lam=lamf)
    # undraw from what is observed: the agent on X at the centre after a forward step, X ahead before it
    ent_rows = fw
    k["undraw"] = VP.Kind("undraw", S, parts, 1, 1, V1[ent_rows, CENTRE][:, None], np.zeros(len(ent_rows), np.int64),
                          w[ent_rows], after=V0[ent_rows, FRONT][:, None], other=V0[ent_rows, FRONT], lam=lamf)
    fw_full = np.flatnonzero((act == FWD) & (out_full != UNCHANGED))
    und_full = VP.Kind("undraw", S, parts, 1, 1, V0[fw_full, CENTRE][:, None], np.zeros(len(fw_full), np.int64),
                       w[fw_full], after=V1[fw_full, BEHIND][:, None], other=V0[fw_full, FRONT], lam=lamf)
    for a in INTER:
        s = np.flatnonzero(act == a)
        b, af = V0[s][:, [FRONT, HELD]].astype(np.int64), V1[s][:, [FRONT, HELD]].astype(np.int64)
        cat = (af[:, 0] != b[:, 0]).astype(np.int64) + 2 * (af[:, 1] != b[:, 1])
        k[a] = VP.Kind(VP.KNAME[a], S, parts, 3, 4, np.concatenate([b, cid[s][:, None]], 1), cat, w[s], after=af)
    self.kinds = k
    m.move_out = {a: VP.Lazy(lambda h, a=a: self.move_cat(a, h)) for a in MOVES}
    m.free = VP.Lazy(lambda h: self.move_cat(FWD, h) == CHANGED)
    m.draw = VP.Lazy(lambda h: self.kinds["draw"].look((h,)))
    m.undraw = VP.Lazy(lambda h: self.kinds["undraw"].look((h,)))
    hs = np.unique(np.concatenate([V0[:, CENTRE], V1[:, CENTRE]]))
    agree = [int(m.undraw[int(h)]) == int(und_full.look((int(h),))) for h in hs]
    rep["undraw_same_as_full_view"] = {"centre_appearances": int(len(hs)), "same": int(sum(agree)),
                                       "differ": [[int(h), int(m.undraw[int(h)]), int(und_full.look((int(h),)))]
                                                  for h, ok in zip(hs, agree) if not ok],
                                       "stepped_on_from_front": int(len(np.unique(V1[ent_rows, CENTRE]))),
                                       "differ_tiles": [[str(KR.OBJECTS[int(c) // 5]) + f"/{int(c) % 5}"
                                                         for c in np.flatnonzero(lut[:len(KR.OBJECTS) * 5] == h)[:2]]
                                                        for h, ok in zip(hs, agree) if not ok],
                                       "full_view_result_is_a_tile": [bool((lut == int(und_full.look((int(h),)))).any())
                                                                      for h, ok in zip(hs, agree) if not ok],
                                       "partial_result_tiles": [[str(KR.OBJECTS[int(c) // 5]) + f"/{int(c) % 5}"
                                                                 for c in np.flatnonzero(lut[:len(KR.OBJECTS) * 5] == int(m.undraw[int(h)]))[:2]]
                                                                for h, ok in zip(hs, agree) if not ok]}
    self.seconds = round(time.monotonic() - t0, 1)
    self.report = {"kinds": {kd.name: kd.report for kd in k.values()}, "poses": int(len(m.A)),
                   "pose_route_conflicts": m.pose_conflicts, "views_in_memory": int(len(up)),
                   "view_vectors_in_memory": int(len(set(pc.tolist()))), "seconds": self.seconds,
                   "partial_memory": rep}
    log(f"  card 061: same transformations {rep['same_transformations_as_full_view']}, unseen "
        f"{rep['unseen_appearance']}, undraw same {rep['undraw_same_as_full_view']}, outcomes differing "
        f"{rep['move_outcome_differs_from_full_view']}")
    print("card 061", rep, flush=True)


def main():
    TK._world_init = world_init                        # card 044's SlotWorld.__init__ calls it
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
