"""Card 044's gate: can the state be a set of tokens, each a what (the tile) and a where (its offset from the
agent, or the hand), without losing anything the grid and the pose table carry?

The grid state is card 028's: one appearance per place in the first view's frame, plus a pose (which first-frame
place each view place shows, composed from the learned move maps). A token's where is its view place read as
an offset in the agent's frame (x to the right, y ahead; the place ahead is (0, 1)); the held place is the hand.
The view's grid is a declared prior, as in card 041. Nothing here acts; appearances are the exact tiles (the
encoder's vectors are one per tile, so the check does not depend on an encoder seed).

  1. moves: each learned move map, read in where coordinates, is one rigid transformation (a quarter turn
     and a shift); its entering places are exactly the wheres whose source lies outside the view.
  2. poses: at every pose, the first-frame place each view place shows is the one the agent's place and
     heading give by the same rigid geometry. If so, the pose table holds nothing the wheres do not.
  3. stepping: on held-out transitions whose view changed, the tokens before, moved by the transformation,
     give the view after at every place that did not enter (the agent's own place and the one it leaves aside:
     the agent is drawn there, as now).
  4. size: tokens per view, against the distinct appearances card 039's set keeps, for the pick up, toggle
     and drop tries recall stores.

  bin/prun python tools/card044/token_check.py --out runs/044_token_check.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card038"))
import vector_planner as VP                                    # noqa: E402

F, T = VP.F, VP.T
NV, NPL, W13, CENTRE, FRONT, HELD = VP.NV, VP.NPL, VP.W13, VP.CENTRE, VP.FRONT, VP.HELD
MOVES, INTER, WORLDS = VP.MOVES, VP.INTER, VP.WORLDS
R = (W13 - 1) // 2
BEHIND = CENTRE + W13
ROT = [np.array(m) for m in (((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)))]


def where_of(i):
    """A view place as an offset in the agent's frame: x to the right, y ahead."""
    r, c = divmod(int(i), W13)
    return np.array([c - R, R - r])


def place_of(w):
    x, y = int(w[0]), int(w[1])
    return (R - y) * W13 + (x + R) if abs(x) <= R and abs(y) <= R else -1


WH = np.stack([where_of(i) for i in range(NV)])


def rigid(src):
    """The quarter turn k and shift t that send the most sources' wheres onto their places' wheres."""
    q = np.flatnonzero(src[:NV] >= 0)
    q = q[src[q] < NV]
    a, b = WH[src[q]], WH[q]
    best = None
    for k in range(4):
        d = b - a @ ROT[k].T
        u, n = np.unique(d, axis=0, return_counts=True)
        t = u[n.argmax()]
        agree = int((d == t).all(1).sum())
        if best is None or agree > best[2]:
            best = (k, t, agree)
    k, t, agree = best
    return k, t, agree, len(q)


def affine(src):
    """Card 041's form: where after = M where before + b, by least squares over the map's correspondences."""
    q = np.flatnonzero(src[:NV] >= 0)
    q = q[src[q] < NV]
    X = np.concatenate([WH[src[q]], np.ones((len(q), 1))], 1)
    sol = np.linalg.lstsq(X, WH[q], rcond=None)[0]
    return sol[:2].T, sol[2], float(np.abs(X @ sol - WH[q]).max())


def check_move(a, src, k, t):
    """Places whose learned source disagrees with the transformation; entering places that should not enter."""
    Rinv = ROT[k].T                                            # rotations are orthogonal
    bad_src, bad_ent = [], []
    for q in range(NV):
        pre = Rinv @ (WH[q] - t)
        p = place_of(pre)
        if src[q] >= 0 and src[q] != p:
            bad_src.append(q)
        if src[q] < 0 and p >= 0:
            bad_ent.append(q)
    held_row = [q for q in range(NV, NPL) if src[q] >= 0 and src[q] != q]
    return bad_src, bad_ent, held_row


def check_poses(m):
    """At every pose, the where of each first-frame place the view shows, from the agent's place and heading."""
    DIRS = np.array(F.DIRS)                                    # (column, row) steps in turning order
    mism, total, ent_inside = 0, 0, 0
    for p in range(len(m.A)):
        A = m.A[p]
        c, h = int(A[CENTRE]), int(m.pd[p])
        ahead, right = DIRS[h], DIRS[(h + 1) % 4]
        for i in range(NV):
            if A[i] >= 0:
                off = np.array([A[i] % W13 - c % W13, A[i] // W13 - c // W13])
                w = np.array([off @ right, off @ ahead])
                total += 1
                mism += int((w != WH[i]).any())
            else:                                              # entering: outside the first frame
                col, row = c % W13 + WH[i] @ np.array([right[0], ahead[0]]), \
                    c // W13 + WH[i] @ np.array([right[1], ahead[1]])
                ent_inside += int(0 <= col < W13 and 0 <= row < W13)
    return {"poses": int(len(m.A)), "view_places": int(len(m.A) * NV), "places_checked": total,
            "places_where_disagrees": mism,
            "entering_places_inside_the_first_frame": ent_inside,
            "held_place_fixed": bool(all(int(A[HELD]) == HELD for A in m.A))}


def step_check(V0, V1, act, term, maps):
    """Held-out moves whose view changed: tokens moved by the transformation against the view after."""
    out = {}
    for a in MOVES:
        k, t = maps[a]
        rows = np.flatnonzero((act == a) & ~term & (V0 != V1).any(1))
        Rinv = ROT[k].T
        pre = np.array([place_of(Rinv @ (WH[q] - t)) for q in range(NV)])
        skip = {CENTRE}
        if a == VP.FWD:
            skip.add(BEHIND)                                   # the agent is drawn where it stands and undrawn behind
        q = np.array([i for i in range(NV) if pre[i] >= 0 and i not in skip])
        same = V1[np.ix_(rows, q)] == V0[np.ix_(rows, pre[q])]
        out[VP.KNAME[a]] = {"transitions": int(len(rows)), "places_per_transition": int(len(q)),
                            "exact_share": round(float(same.mean()), 6),
                            "transitions_all_exact": round(float(same.all(1).mean()), 6),
                            "held_unchanged": round(float((V1[rows, HELD] == V0[rows, HELD]).mean()), 6)}
    return out


def view_check(m, layouts, free):
    """At the start of each layout, the view from every placement as card 028's pose table gives it and as the
    tokens give it (tools/card044/tokens.py), at the places other than the agent's own. Also for the
    placements standing on a tile forward has moved the agent onto (free), the only ones walking uses."""
    import copy
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tokens as TK
    mt = copy.copy(m)
    TK.build_tokens(mt)
    by = {(c, d): q for q, (c, d) in enumerate(zip(m.cidx, m.pd))}
    pairs = [(p, by[(c, d)]) for p, (c, d) in enumerate(zip(mt.cidx, mt.pd)) if (c, d) in by]
    keep = np.ones(NV, bool)
    keep[CENTRE] = False
    n = {k: np.zeros(4, np.int64) for k in ("all", "standing")}      # views, places, places differing, sets differing
    app, F.Kd.APP = F.Kd.APP, np.arange(256)             # views as the data's tile codes
    Vs = [F.see(lay, VP.ld.start_state(lay)).astype(np.int64) for lay in layouts]
    F.Kd.APP = app
    for V in Vs:
        ft = np.full(TK.NT, TK.ABSENT, np.int64)
        ft[:NPL] = V
        for p, q in pairs:
            A = mt.A[p][:NV]
            vt = np.where(A >= 0, ft[np.maximum(A, 0)], TK.ABSENT)
            vt = np.where(vt == TK.ABSENT, mt.unseen, vt)
            Ag = m.A[q][:NV]
            vg = np.where(Ag >= 0, V[np.maximum(Ag, 0)], m.Ent[q][:NV])
            row = np.array([1, keep.sum(), (vt != vg)[keep].sum(), set(vt[keep].tolist()) != set(vg[keep].tolist())])
            n["all"] += row
            if int(V[mt.cidx[p]]) in free:
                n["standing"] += row
    return {"layouts": len(layouts), "placements_matched": len(pairs), "placements": int(len(mt.A)),
            "poses": int(len(m.A)),
            **{k: dict(zip(("views", "view_places", "view_places_differing", "views_whose_set_differs"),
                           v.tolist())) for k, v in n.items()}}


def size_check(V0, act):
    rows = np.flatnonzero(np.isin(act, INTER))
    keep = np.ones(NV, bool)
    keep[CENTRE] = False
    X = V0[rows][:, :NV][:, keep]
    distinct = np.array([len(np.unique(x)) for x in X])
    tok = X.shape[1]
    sets = {frozenset(np.unique(x).tolist()) for x in X}
    configs = {x.tobytes() for x in X}
    return {"tries": int(len(rows)), "tokens_per_view": int(tok), "plus_hand": 1,
            "distinct_appearances_per_view_mean": round(float(distinct.mean()), 2),
            "distinct_appearances_per_view_max": int(distinct.max()),
            "distinct_views_as_sets_of_appearances": len(sets),
            "distinct_views_as_token_sets": len(configs),
            "matching_cost_per_pair_ratio": round(float(tok ** 2 / (distinct ** 2).mean()), 1)}


def main():
    args = sys.argv[1:]
    out = Path(next((args[i + 1] for i in range(len(args) - 1) if args[i] == "--out"), "runs/044_token_check.json"))
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.monotonic()
    log = lambda msg: print(f"[{time.monotonic() - t0:6.0f}s] {msg}", flush=True)
    T.configure()
    d = pickle.loads(T.DATA.read_bytes())
    res = {"note": "Card 044's gate, tools/card044/token_check.py", "worlds": {}}
    rng = np.random.default_rng(0)
    for world in WORLDS:
        D = d["data"][world]
        V0, V1, act, term = D["ego0"], D["ego1"], D["act"], D["term1"].astype(bool)
        vc = (V0[:, :NV] != V1[:, :NV]).any(1)
        m = F.Model()
        m.src, m.ent = {}, {}
        r = res["worlds"][world] = {"moves": {}}
        maps = {}
        for a in MOVES:
            rows = np.flatnonzero((act == a) & vc & ~term)
            if len(rows) > 40000:
                rows = np.sort(rng.choice(rows, 40000, replace=False))
            src, ent, _, share, const, _ = F.fit_map(V0[rows], V1[rows], dev)
            m.src[a], m.ent[a] = src, ent
            k, t, agree, n = rigid(src)
            bad_src, bad_ent, held_row = check_move(a, src, k, t)
            maps[a] = (k, t)
            Ma, ba, resid = affine(src)
            r["moves"][VP.KNAME[a]] = {"affine_M": np.round(Ma, 6).tolist(), "affine_b": np.round(ba, 6).tolist(),
                                       "affine_max_residual": round(resid, 9),
                                       "affine_equals_rigid": bool(np.allclose(Ma, ROT[k]) and np.allclose(ba, t)),
                                       "quarter_turns": k, "shift": t.tolist(), "places_agreeing": agree,
                                       "places_with_a_source": n, "sources_disagreeing": len(bad_src),
                                       "entering_but_source_in_view": len(bad_ent),
                                       "held_row_moved": len(held_row)}
        m.change_places = np.array([FRONT, HELD])
        F.build_poses(m)
        r["poses"] = check_poses(m)
        r["poses"]["route_conflicts"] = int(m.pose_conflicts)
        fw = (act == VP.FWD) & vc & ~term
        r["views_at_start"] = view_check(m, D["test"][:100], set(np.unique(V0[fw, FRONT]).tolist()))
        r["stepping_heldout"] = step_check(D["hego0"], D["hego1"], D["hact"], D["hterm1"].astype(bool), maps)
        r["size"] = size_check(V0, act)
        log(f"{world}: moves {r['moves']}; poses {r['poses']}; stepping {r['stepping_heldout']}; size {r['size']}")
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f"written {out}")


if __name__ == "__main__":
    main()
