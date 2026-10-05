"""Card 062: what was in view for each stored try, as the agent believed it under the 7 x 7 view.

Version 12's memory stores each pick up, toggle and drop with what was in view in the 13 x 13 window that holds
the whole map (card 057's declared exception; card 061 removed it for the moves and undraw). Here the stored play
is replayed episode by episode as the agent saw it (MiniGrid's 7 x 7 view with occlusion), and each stored try
takes what was in view in the agent's belief at that step: the tiles it has seen in this episode, placed by its own
motion (card 044's transformations, as card 061 learned them from the small view; a forward step happened when the
view changed), drawn around the agent; a place never seen shows the unseen appearance (wall), as in acting.

The replay regenerates card 033's collection (the same seeds, chunks and sampling); every stored row's action, and
every stored try's view, is checked against the stored play before it is used.

  bin/prun python tools/card062/believed.py --collect            # writes runs/062/believed_<world>.npz
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS / "card061"))
import moves as M61                                    # noqa: E402

VP, F, TK, KR = M61.VP, M61.F, M61.TK, M61.KR
NV, NPL, CENTRE = M61.NV, M61.NPL, M61.CENTRE
INTER = VP.INTER
LEFT, RIGHT, FWD = VP.MOVES
sys.path.insert(0, str(TOOLS / "card031"))
import codes as C                                      # noqa: E402

dl, ld = C.dl, C.ld
WH = TK.WH[:NV].astype(np.int64)
WALL = KR.code("wall")
NCODES = len(KR.OBJECTS) * 5
OFF, SIDE = 20, 41
OUT = ROOT / "runs" / "062"
WORLDS = ("key", "switch", "either", "both")
EPISODES, SEED = 5000, 11                              # card 033's setup (card 034's data)
T = {}                                                 # the moves' transformations, from card 061


def moves_from_card061(path=ROOT / "runs" / "061" / "fam_399.json"):
    d = json.loads(Path(path).read_text())
    mv = d["key"]["model"]["partial_memory"]["moves"]
    for a, name in ((LEFT, "left"), (RIGHT, "right"), (FWD, "forward")):
        T[a] = (np.array(mv[name]["M"], np.int64), np.array(mv[name]["b"], np.int64))


def believe(ego, act, rows):
    """One episode: ego (T+1, NPL) renderer codes, act (T,), rows: the stored steps whose believed view is wanted.
    Returns per row the codes present in the believed view (centre aside) and the believed tiles that disagree
    with the true view (stale beliefs)."""
    vis = M61.visible(ego)
    seen = np.where(vis, ego, -1)
    B = np.full((SIDE, SIDE), -1, np.int64)
    Rm, t = np.eye(2, dtype=np.int64), np.zeros(2, np.int64)
    want = {int(r): i for i, r in enumerate(rows)}
    pres = np.zeros((len(rows), NCODES), bool)
    stale = np.zeros(len(rows), np.int64)
    for s in range(len(act) + 1):
        q = np.flatnonzero(vis[s, :NV])
        wf = (WH[q] - t) @ Rm                         # first-sight wheres: Rm^T (where - t), as row vectors
        v = ego[s, q].copy()
        c = q == CENTRE
        v[c] = (v[c] // 5) * 5                          # the tile under the agent (card 061's undraw)
        B[wf[:, 0] + OFF, wf[:, 1] + OFF] = v
        i = want.get(s)
        if i is not None:
            wa = (WH - t) @ Rm
            bv = B[wa[:, 0] + OFF, wa[:, 1] + OFF]
            shown = np.where(bv >= 0, bv, WALL)
            keep = np.ones(NV, bool)
            keep[CENTRE] = False
            pres[i, shown[keep]] = True
            known = (bv >= 0) & keep
            stale[i] = int((bv[known] != ego[s, :NV][known]).sum())
        if s == len(act):
            break
        a = int(act[s])
        if a in (LEFT, RIGHT) or (a == FWD and (seen[s, :NV] != seen[s + 1, :NV]).any()):
            Ma, ba = T[a]
            Rm, t = Ma @ Rm, Ma @ t + ba
    return pres, stale


def chunk(job):
    """dl._collect_chunk's rows (the same seeds and sampling), with each stored try's believed view."""
    rule, n, seed, play, p_uniform = job
    rng = np.random.default_rng(seed + 7)
    data = ld.collect(rule, n, seed, play_starts=play)
    acts, inter_ego, pres_all, stale_all = [], [], [], []
    for ep in data:
        lay, S, A = ep["layout"], ep["states"], np.array(ep["actions"], np.int8)
        codes = dl.encode_logic_states(lay, S)
        st = np.array([dl.to_arr(s) for s in S], np.int8)
        Tn = len(A)
        obj = codes // 5
        changed = (obj[1:] != obj[:-1]).reshape(Tn, -1).any(1)
        term = np.array([(s[0], s[1]) == lay.goal for s in S[1:]])
        forced = changed | term
        i = np.flatnonzero(forced | (rng.random(Tn) < p_uniform))
        acts.append(A[i])
        ego = C.ego_codes(codes, st)
        r = i[np.isin(A[i], INTER)]
        pres, stale = believe(ego, A, r)
        inter_ego.append(ego[r].astype(np.uint8))
        pres_all.append(pres)
        stale_all.append(stale)
    cat = lambda v, k=0: np.concatenate(v) if v else np.zeros((0, k))
    return cat(acts), cat(inter_ego, NPL), cat(pres_all, NCODES), cat(stale_all)


def collect(world, pool):
    C.patch()
    n_chunks = (EPISODES + dl.CHUNK - 1) // dl.CHUNK
    jobs = [(world, min(dl.CHUNK, EPISODES - c * dl.CHUNK), SEED * 100003 + c, 0.0, 1 / 8) for c in range(n_chunks)]
    parts = pool.map(chunk, jobs)
    act = np.concatenate([p[0] for p in parts]).astype(np.int64)
    ego = np.concatenate([p[1] for p in parts])
    pres = np.concatenate([p[2] for p in parts])
    stale = np.concatenate([p[3] for p in parts])
    return act, ego, pres, stale


def main():
    import pickle
    moves_from_card061()
    OUT.mkdir(parents=True, exist_ok=True)
    stored = pickle.loads(VP.T.DATA.read_bytes())["data"]
    pool = F.pool20()
    rep = {}
    for world in WORLDS:
        t0 = time.monotonic()
        act, ego, pres, stale = collect(world, pool)
        D = stored[world]
        same_act = len(act) == len(D["act"]) and bool((act == D["act"]).all())
        sel = np.flatnonzero(np.isin(D["act"], INTER))
        same_ego = same_act and bool((ego == D["ego0"][sel]).all())
        np.savez_compressed(OUT / f"believed_{world}.npz", act=act, pres=pres, stale=stale)
        rep[world] = {"rows": int(len(act)), "tries": int(len(sel)), "same_actions": same_act,
                      "same_views": same_ego, "tries_with_stale_belief": int((stale > 0).sum()),
                      "codes_in_view_mean_believed": round(float(pres.sum(1).mean()), 2),
                      "seconds": round(time.monotonic() - t0, 1)}
        print(world, rep[world], flush=True)
    pool.close()
    (OUT / "believed.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if "--collect" in sys.argv:
        main()
