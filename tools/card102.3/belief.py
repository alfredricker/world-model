"""Card 102.3: short-term memory. Every token tracked in view is kept once out of view, its place spreading as its
code's tokens have been seen to move, its existence falling as they vanish and wherever it was predicted in view and
not seen.

Tracks (per step, the tokens in view with an identity, a code and a place in the frame of the segment's start):
  own    card 102.2's layers on the codes: identities, ground, and the view placed by its own shift;
  truth  the evaluator's objects, identities and places (the upper bound).
Learned from the training tracks, per code: P(this step's move | last step's move) over still and the four shifts
(counts, add one half); the rate of vanishing in view; the density of tokens new to the view per place and step;
Q(b | a), the share of a tracked token's life it looks like code b, over steps it looks like a (a cow on sand and on
grass are two codes of one token).
A code expected to move less than once in 50 steps (rate x 50 < 0.5) is kept in the map, exactly as last seen.

Belief, each step: every file's distribution over (place, last move) is moved by its code's motion; a token new to
the view takes the file with the largest existence x predicted mass at its place x Q(its code | the file's code), if
that beats its code's density of new tokens (greedy, one file each); every other file's mass in view is removed with detection
probability 0.95 and its existence updated as IPDA does; existence then falls by the code's rate of vanishing.
At most K files are kept (the most recently in view).

Forms: learned (as above); still ("stays as last seen", version 27: every code's motion is "still"); rule098 (as
learned, each file dropped 20 steps after it was last in view: card 098.6's rule).

  bin/prun python tools/card102.3/belief.py world3|crafter [--K k] [--forms a,b]   → runs/102.3/belief_<world>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card102.2"))
import layers as LY                                    # noqa: E402

RUNS = ROOT / "runs" / "102.3"
R, C = 7, 9
AR_, AC = 3, 4                                         # the agent's row and column in the view
MOV = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))       # state 0 still, 1..4 card 102.2's shifts (dx, dy)
WIN = 20                                               # a file's window: places within 20 of where it was last seen
HORIZON, PD, EPS = 50, 0.95, 0.01
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


# ------------------------------------------------------------------------------------------------ data
def load(world, split):
    if world == "world3":
        z = np.load(ROOT / "runs" / "card102" / f"world3_{split}.npz")
        n = len(z["act"])
        d = {"code": np.load(RUNS / f"world3_code_{split}.npy")[:, :63].reshape(n, R, C).astype(np.int64), "sem": z["sem"],
             "vec": np.load(RUNS / f"world3_vec_{split}.npy", mmap_mode="r")[:, :63],
             "seg": z["ep"].astype(np.int64), "light": z["light"], "pos": z["pos"].astype(np.int64)}
        o = z["obj"]                                     # step, id, kind, dx, dy, x, y
        d["obj"] = np.stack([o[:, 0], o[:, 1], o[:, 2], o[:, 3], o[:, 4]], 1).astype(np.int64)
        d["movers"], d["static"], d["floor"] = (5, 6), (7,), (0, 1, 2)   # ball, cat; box; floor (evaluator)
        return d
    ws = range(8) if split == "train" else (8, 9)
    out = {k: [] for k in ("code", "vec", "seg", "light", "pos", "obj", "sem")}
    base, t0 = 0, 0
    for w in ws:
        z = np.load(ROOT / "runs" / "098.5" / f"play_w{w}.npz")
        n = len(z["act"])
        sleep = z["sleep"]
        brk = np.r_[True, (z["ep"][1:] != z["ep"][:-1]) | sleep[1:] | sleep[:-1]]
        seg = base + np.cumsum(brk)
        seg[sleep] = -1
        base = seg.max() + 1
        out["seg"].append(seg)
        out["code"].append(np.load(ROOT / "runs" / "098.5" / f"codes_w{w}.npy")[:, :63].reshape(n, R, C).astype(np.int64))
        out["vec"].append(np.load(ROOT / "runs" / "098.5" / f"vec_w{w}.npy"))
        out["light"].append(z["light"])
        out["sem"].append(z["sem"])
        out["pos"].append(z["pos"].astype(np.int64))
        o = z["obj"].astype(np.int64)                    # step, slot, kind, dx, dy, facing
        o = o[o[:, 2] > 0]                               # the player is not an object here
        out["obj"].append(np.stack([o[:, 0] + t0, o[:, 1] + 1000000 * w, o[:, 2], o[:, 3], o[:, 4]], 1))
        t0 += n
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["movers"], d["static"], d["floor"] = (1, 2, 3), (5,), (2, 4, 5)   # cow, zombie, skeleton; plant; grass, path, sand
    return d


def own_tracks(d, ground):
    """Card 102.2's layers: per step, the tokens in view that are not ground (identity, code, place)."""
    A = LY.analyse(d, ground, p_move=0.1)
    seg, n = d["seg"], len(d["seg"])
    origin = np.zeros((n, 2), np.int64)                  # the view's top-left place in the segment's frame (x, y)
    for t in range(1, n):
        if seg[t] >= 0 and seg[t] == seg[t - 1]:
            origin[t] = origin[t - 1] - A["bg"][t - 1]
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    steps = []
    for t in range(n):
        if seg[t] < 0:
            steps.append(None)
            continue
        m = ~np.isin(d["code"][t], ground)
        steps.append({"id": A["ident"][t][m], "code": d["code"][t][m],
                      "x": origin[t, 0] + cc[m], "y": origin[t, 1] + rr[m], "vr": rr[m], "vc": cc[m]})
    return steps, origin


def truth_tracks(d):
    """The evaluator's objects in view, with the view placed at the agent's true position."""
    seg, n = d["seg"], len(d["seg"])
    origin = d["pos"] - np.array([AC, AR_])
    o = d["obj"]
    order = np.argsort(o[:, 0], kind="stable")
    o = o[order]
    starts = np.searchsorted(o[:, 0], np.arange(n + 1))
    steps = []
    for t in range(n):
        if seg[t] < 0:
            steps.append(None)
            continue
        r = o[starts[t]:starts[t + 1]]
        steps.append({"id": r[:, 1], "code": r[:, 2], "x": origin[t, 0] + AC + r[:, 3], "y": origin[t, 1] + AR_ + r[:, 4],
                      "vr": AR_ + r[:, 4], "vc": AC + r[:, 3]})
    return steps, origin


# ------------------------------------------------------------------------------------------------ learning
def learn(steps, seg, origin):
    """Per code: move counts given the last move, the rate of vanishing in view, new tokens per place and step."""
    trans, vanish, seen, new = {}, {}, {}, {}
    worn = {}                                            # id -> {code: steps}: the codes one tracked token wore
    last = {}                                            # id -> (t, x, y, move state)
    nsteps = 0
    for t, s in enumerate(steps):
        if s is None:
            last = {}
            continue
        if t == 0 or seg[t - 1] != seg[t]:
            last = {}
        nsteps += 1
        cur = {}
        for i, k, x, y in zip(s["id"], s["code"], s["x"], s["y"]):
            i, k, x, y = int(i), int(k), int(x), int(y)
            st = None
            if i in last and last[i][0] == t - 1:
                dx, dy = x - last[i][1], y - last[i][2]
                if (dx, dy) in MOV:
                    st = MOV.index((dx, dy))
                    p = last[i][3]
                    if p is not None:
                        trans.setdefault(k, np.zeros((5, 5)))[p, st] += 1
            elif i not in last:
                new[k] = new.get(k, 0) + 1
            cur[i] = (t, x, y, st, k)
            w = worn.setdefault((int(seg[t]), i), {})
            w[k] = w.get(k, 0) + 1
        if t + 1 < len(steps) and steps[t + 1] is not None and seg[t + 1] == seg[t]:
            nxt = set(steps[t + 1]["id"].tolist())
            ox, oy = origin[t + 1]
            for i, (_, x, y, _, k) in cur.items():
                if ox + 1 <= x < ox + C - 1 and oy + 1 <= y < oy + R - 1:   # its place and every neighbour still in view
                    seen[k] = seen.get(k, 0) + 1
                    vanish[k] = vanish.get(k, 0) + (i not in nxt)
        last = {i: v[:4] for i, v in cur.items()}
    codes = sorted(set(trans) | set(seen) | set(new))
    # Q(b | a): given a step where a token looks like code a, the share of its tracked life it looks like b
    Q = {}
    for w in worn.values():
        n = sum(w.values())
        for a, na in w.items():
            q = Q.setdefault(a, {})
            for b, nb in w.items():
                q[b] = q.get(b, 0.0) + na * nb / n
    Q = {a: {b: v / sum(q.values()) for b, v in q.items()} for a, q in Q.items()}
    M = {}
    for k in codes:
        cnt = trans.get(k, np.zeros((5, 5)))
        T = (cnt + 0.5) / (cnt + 0.5).sum(1, keepdims=True)
        moves = cnt[:, 1:].sum() / max(cnt.sum(), 1)
        M[k] = {"T": T, "move rate": float(moves), "steps": int(cnt.sum()),
                "static": bool(moves * HORIZON < 0.5),
                "vanish": float((vanish.get(k, 0) + 0.5) / (seen.get(k, 0) + 1)),
                "new density": float((new.get(k, 0) + 0.5) / (nsteps * R * C)), "Q": Q.get(k, {k: 1.0})}
    return M


# ------------------------------------------------------------------------------------------------ belief
def propagate(B, T):
    """B (5, 2W+1, 2W+1): mass over (last move, place). One step of the code's motion; mass leaving the window is
    lost (the token is far)."""
    S = np.einsum("es,exy->sxy", T, B)
    out = np.zeros_like(B)
    n = B.shape[1]
    for s, (dx, dy) in enumerate(MOV):
        out[s, max(0, dy):n + min(0, dy), max(0, dx):n + min(0, dx)] = S[s, max(0, -dy):n + min(0, -dy),
                                                                        max(0, -dx):n + min(0, -dx)]
    return out


def view_slice(f, ox, oy):
    """The part of a file's window inside the view: window rows/columns and view rows/columns."""
    x0, y0 = f["cx"] - WIN, f["cy"] - WIN
    xa, xb = max(ox, x0), min(ox + C, x0 + 2 * WIN + 1)
    ya, yb = max(oy, y0), min(oy + R, y0 + 2 * WIN + 1)
    if xa >= xb or ya >= yb:
        return None
    return (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0)), (slice(ya - oy, yb - oy), slice(xa - ox, xb - ox))


def in_view_map(f, ox, oy):
    """The file's mass over the view's places (R, C)."""
    out = np.zeros((R, C))
    vs = view_slice(f, ox, oy)
    if vs is not None:
        (wr, wc), (vr, vc) = vs
        out[vr, vc] = f["B"][:, wr, wc].sum(0)
    return out


def run(steps, seg, origin, M, form, K, truth_at, light, movers, static_kinds, kind_of, sem, floor_kinds):
    """One pass over the test tracks. Returns per return event of a true object: its kind, time away, the prediction
    of its place (log-likelihood, inside the 90% region) and whether its identity was kept."""
    files, mapped = {}, {}
    nf = 0
    events = []
    last_in = {}                                          # true id -> last step in view
    prev = None
    t_belief = 0.0
    n_belief = 0
    for t, s in enumerate(steps):
        if s is None or t == 0 or seg[t - 1] != seg[t]:
            files, mapped, last_in, prev = {}, {}, {}, None
            if s is None:
                continue
        ox, oy = origin[t]
        ids = s["id"].tolist()
        tb = time.perf_counter()
        # 1. tokens no longer in view become files (or, for static codes, entries in the map)
        if prev is not None:
            now = set(ids)
            for i, (k, x, y, st, tid, sm) in prev.items():
                if i in now:
                    continue
                m = M.get(k)
                if m is None:
                    continue
                if m["static"] or form == "still":
                    if m["static"]:
                        mapped[(x, y)] = (k, tid, sm)
                        continue
                B = np.zeros((5, 2 * WIN + 1, 2 * WIN + 1))
                B[st if st is not None else 0, WIN, WIN] = 1.0
                files[nf] = {"code": k, "cx": x, "cy": y, "B": B, "r": 1.0, "last": t - 1, "tid": tid}
                nf += 1
        # 2. move every file by its code's motion
        for f in files.values():
            T = np.eye(5)[[0] * 5] if form == "still" else M[f["code"]]["T"]
            f["B"] = propagate(f["B"], T)
        # 3. tokens new to the view take the best file of their code, or are new
        new = [j for j, i in enumerate(ids) if prev is None or i not in prev]
        pred = {fid: in_view_map(f, ox, oy) for fid, f in files.items()}
        cand = []
        for j in new:
            k, vr, vc = int(s["code"][j]), int(s["vr"][j]), int(s["vc"][j])
            if k not in M:
                continue
            for fid, f in files.items():
                q = M[f["code"]]["Q"].get(k, 0.0)        # a token like the file's seen looking like this one
                if q > 0:
                    sc = f["r"] * pred[fid][vr, vc] / max(f["B"].sum(), 1e-12) * q
                    if sc > M[k]["new density"]:
                        cand.append((sc, j, fid))
        matched, taken, used = {}, set(), set()
        for sc, j, fid in sorted(cand, reverse=True):
            if j in taken or fid in used:
                continue
            matched[j] = fid
            taken.add(j)
            used.add(fid)
        t_belief += time.perf_counter() - tb
        # score returns of true objects (before the files that matched are removed)
        for tid, vr, vc in truth_at(t):
            if tid in last_in and last_in[tid] < t - 1 and t - 1 - last_in[tid] <= HORIZON:
                away = t - 1 - last_in[tid]
                kind = kind_of[tid]
                mine = [q for q, f in files.items() if f["tid"] == tid]
                fid = max(mine, key=lambda q: files[q]["last"]) if mine else None   # its most recent file
                ev = {"kind": kind, "away": away, "day": bool(light[t] >= 0.3), "has file": fid is not None}
                if kind in movers:
                    p = pred[fid] / pred[fid].sum() if fid is not None and pred[fid].sum() > 1e-12 \
                        else np.full((R, C), 1.0 / (R * C))
                    ev["log-likelihood"] = float(np.log((1 - EPS) * p[vr, vc] + EPS / (R * C)))
                    # the 90% region, its boundary place counted in proportion so that it holds exactly 90%
                    order = np.sort(p.ravel())[::-1]
                    cut = order[min(int(np.searchsorted(np.cumsum(order), 0.9)), len(order) - 1)]
                    above, at = p[p > cut].sum(), p[p == cut].sum()
                    ev["inside 90%"] = 1.0 if p[vr, vc] > cut else (
                        float(np.clip((0.9 - above) / at, 0, 1)) if p[vr, vc] == cut else 0.0)
                    j = next((q for q, i in enumerate(ids) if int(s["vr"][q]) == vr and int(s["vc"][q]) == vc), None)
                    ev["identity kept"] = bool(j is not None and j in matched and files[matched[j]]["tid"] == tid)
                    if fid is not None:                   # why identity was or was not kept (reported)
                        ev["token new"] = bool(j is not None and j in new)
                        ev["same code"] = bool(j is not None and int(s["code"][j]) == files[fid]["code"])
                        ev["score over new density"] = float(files[fid]["r"] * pred[fid][vr, vc] / max(files[fid]["B"].sum(), 1e-12)
                                                             * (M[files[fid]["code"]]["Q"].get(int(s["code"][j]), 0.0) if j is not None else 0)
                                                             / M[files[fid]["code"]]["new density"])
                        ev["taken by another file"] = bool(j is not None and j in matched and files[matched[j]]["tid"] != tid)
                events.append(ev)
        tb = time.perf_counter()
        for j, fid in matched.items():
            del files[fid]
        # 4. misses: a file's mass in view, not seen there, is removed (IPDA)
        for fid in list(files):
            f = files[fid]
            tot = f["B"].sum()
            if tot <= 1e-12:
                del files[fid]
                continue
            vs = view_slice(f, ox, oy)
            if vs is not None:
                (wr, wc), _ = vs
                q = f["B"][:, wr, wc].sum() / tot
                f["r"] = f["r"] * (1 - PD * q) / max(1 - f["r"] * PD * q, 1e-12)
                f["B"][:, wr, wc] *= 1 - PD
            f["r"] *= 1 - M[f["code"]]["vanish"]
            if f["r"] < 1e-4 or (form == "rule098" and t - f["last"] > 20):
                del files[fid]
        if len(files) > K:                                # capacity: the most recently in view
            for fid in sorted(files, key=lambda q: files[q]["last"])[:len(files) - K]:
                del files[fid]
        # places kept in the map that are back in view: is the evaluator's kind there the one kept? Then what stands
        # there now replaces the map
        for x in range(ox, ox + C):
            for y in range(oy, oy + R):
                e = mapped.pop((x, y), None)
                now = int(sem[t][y - oy, x - ox])
                if e is not None and now not in floor_kinds:     # floor there now: it vanished (existence's matter)
                    events.append({"kind": -1, "day": bool(light[t] >= 0.3), "map place right": bool(now == e[2])})
        t_belief += time.perf_counter() - tb
        n_belief += 1
        # the tokens in view now (with their last move) for the next step
        cur = {}
        for j, i in enumerate(ids):
            i, k, x, y = int(i), int(s["code"][j]), int(s["x"][j]), int(s["y"][j])
            st = None
            if prev is not None and i in prev:
                dx, dy = x - prev[i][1], y - prev[i][2]
                st = MOV.index((dx, dy)) if (dx, dy) in MOV else None
            elif j in matched:
                st = None
            tid = truth_cell(t, int(s["vr"][j]), int(s["vc"][j]))
            cur[i] = (k, x, y, st, tid, int(sem[t][int(s["vr"][j]), int(s["vc"][j])]))
        prev = cur
        for tid, vr, vc in truth_at(t):
            last_in[tid] = t
    return events, t_belief / max(n_belief, 1)


def summarise(events, movers, static_kinds, day_only):
    ev = [e for e in events if (e["day"] or not day_only)]
    mv = [e for e in ev if e["kind"] in movers]
    st = [e for e in ev if e["kind"] == -1]
    with_file = [e for e in mv if e["has file"]]
    out = {"returns of movers within 50 steps": len(mv),
           "with a file at the return": round(len(with_file) / max(len(mv), 1), 4),
           "log-likelihood per return (all)": round(float(np.mean([e["log-likelihood"] for e in mv])), 4) if mv else None,
           "inside the 90% region (returns with a file)": round(float(np.mean([e["inside 90%"] for e in with_file])), 4)
           if with_file else None,
           "identity kept (all returns)": round(float(np.mean([e["identity kept"] for e in mv])), 4) if mv else None,
           "map places right when back in view": round(float(np.mean([e["map place right"] for e in st])), 4)
           if st else None, "map places back in view": len(st)}
    by = {}
    for lo, hi in ((1, 5), (6, 20), (21, 50)):
        b = [e for e in mv if lo <= e["away"] <= hi]
        if b:
            by[f"{lo}-{hi} steps away"] = {"returns": len(b), "log-likelihood": round(float(np.mean([e["log-likelihood"] for e in b])), 3),
                                           "identity kept": round(float(np.mean([e["identity kept"] for e in b])), 3)}
    out["by time away"] = by
    kinds = {}
    for e in mv:
        kinds.setdefault(e["kind"], []).append(e)
    out["by kind"] = {int(k): {"returns": len(v), "log-likelihood": round(float(np.mean([e["log-likelihood"] for e in v])), 3),
                               "identity kept": round(float(np.mean([e["identity kept"] for e in v])), 3)}
                      for k, v in kinds.items()}
    return out


truth_cell = None


def main():
    global truth_cell
    world = sys.argv[1]
    forms = arg("--forms", "learned,still,rule098,truth").split(",")
    Ks = [int(k) for k in arg("--K", "1000000").split(",")]
    t0 = time.monotonic()
    tr, te = load(world, "train"), load(world, "test")
    if "--n" in sys.argv:
        k = int(arg("--n", "0"))
        te = {key: (v[:k] if key not in ("obj", "movers", "static", "floor") else v) for key, v in te.items()}
        te["obj"] = te["obj"][te["obj"][:, 0] < k]
    ground, _ = LY.learn_ground(tr)
    print("ground", ground, f"({time.monotonic() - t0:.0f} s)", flush=True)
    # the evaluator's objects per step, for scoring
    o = te["obj"]
    at = {}
    for t, i, k, dx, dy in o:
        at.setdefault(int(t), []).append((int(i), AR_ + int(dy), AC + int(dx)))
    cell = {(t, vr, vc): i for t, v in at.items() for i, vr, vc in v}
    kind_of = {int(i): int(k) for _, i, k, _, _ in o}
    truth_at = lambda t: at.get(t, [])
    truth_cell = lambda t, vr, vc: cell.get((t, vr, vc))
    rep = {"world": world, "ground codes": ground}
    tracks = {}
    for src in ("own", "truth"):
        if src == "own" and not any(f != "truth" for f in forms):
            continue
        if src == "truth" and "truth" not in forms:
            continue
        f_tr = own_tracks(tr, ground) if src == "own" else truth_tracks(tr)
        f_te = own_tracks(te, ground) if src == "own" else truth_tracks(te)
        M = learn(f_tr[0], tr["seg"], f_tr[1])
        tracks[src] = (f_te, M)
        rep[f"learned per code ({src} tracks)"] = {str(k): {"move rate": round(m["move rate"], 4), "static": m["static"],
                                                            "vanish per step": round(m["vanish"], 4), "steps": m["steps"]}
                                                   for k, m in M.items() if m["steps"] > 0}
        print(src, "tracks", f"({time.monotonic() - t0:.0f} s)", flush=True)
    day_only = world == "crafter"
    for form in forms:
        (steps, origin), M = tracks["truth" if form == "truth" else "own"]
        for K in Ks:
            ev, sec = run(steps, te["seg"], origin, M, "learned" if form == "truth" else form, K, truth_at,
                          te["light"], te["movers"], te["static"], kind_of, te["sem"], te["floor"])
            key = form if K >= 1000000 else f"{form}, K = {K}"
            rep[key] = summarise(ev, te["movers"], te["static"], day_only)
            rep[key]["belief seconds per step"] = round(sec, 6)
            np.save(RUNS / f"events_{world}_{key.replace(', ', '_').replace(' = ', '')}.npy", np.array(ev, dtype=object))
            print(key, json.dumps({k: v for k, v in rep[key].items() if not isinstance(v, dict)}),
                  f"({time.monotonic() - t0:.0f} s)", flush=True)
    rep["seconds"] = round(time.monotonic() - t0, 1)
    tag = arg("--tag", "")
    (RUNS / f"belief_{world}{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
