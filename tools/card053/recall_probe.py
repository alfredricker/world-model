"""Card 053, step 2a: version 10's recall (card 049's admission, card 050's own tries first, card 051's index) on
a frozen encoder from card 052's harness, over the generator's tries (no tint, pixel noise, play starts at 0.5,
stratified by kind of try, as step T's probe).

Codes are read as the agent reads them (card 035): a piece takes its nearest used entry if it lies within 6 x
that entry's radius (the farthest of the entry's memory pieces, at least their median distance), else it is
"new". Memory is keyed by code tuples: a tile's handle is the mean vector of the memory tiles with its tuple; a
tile with a "new" piece, or a tuple no memory tile has, keeps its own vector. Fresh codes (card 036) are not
made. Every try has the same view (one constant token whose codes are "new"), so only the front and held tiles'
parts and the relations between them can be admitted.

Criterion 1: the two colour cases at a locked door. Criterion 2: the probe rendered with a second noise draw
(the same objects) gets the same prediction. Reported: accuracy per kind of try, the admitted conditions, alpha,
and, when the weights hold a transition model, its own predictions on the same probe (by the encoder's codes).

  bin/prun python tools/card053/recall_probe.py runs/053/s1_T0_399.pt [--interaction transition] [--gates 0]
      [--out runs/053/s2a_T0_399.json]
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as fn

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card052"))
sys.path.insert(0, str(TOOLS / "card051"))
import effect as EF                                    # noqa: E402
import predflip as PF                                  # noqa: E402
import index as IX                                     # noqa: E402

DR, GN = EF.DR, EF.DR.GN
CR, SP = IX.CR, IX.CD.SP
VP, NV = SP.VP, CR.NV
ACTS = EF.ACTS
NAMES = {3: "pick up", 4: "drop", 5: "toggle"}


class Tagged(np.ndarray):
    """A rendered tile that remembers the object drawn, so the probe can be drawn again with fresh noise."""


def tagged_tile(self, obj):
    t = (GN.tile(obj, self.tint, self.rng) * 255).astype(np.uint8).view(Tagged)
    t.obj = obj
    return t


def load(path, interaction, gates):
    ck = torch.load(path)
    EF.INTERACTION = interaction
    EF.GATES = gates
    DR.Encoder.EMA = 0.99 if interaction == "transition" else 0.0
    enc = EF.Encoder(399)
    enc.enc.load_state_dict(ck["enc"])
    with torch.no_grad():
        enc.books.copy_(ck["books"].to(enc.dev))
        if ck.get("trans") is not None:
            enc.trans.load_state_dict(ck["trans"])
        if ck.get("gates") is not None and hasattr(enc, "gates"):
            enc.gates.copy_(ck["gates"].to(enc.dev))
    return enc


@torch.no_grad()
def vectors(enc, tiles):
    out = []
    for i in range(0, len(tiles), 2048):
        out.append(enc.pieces(enc.x(np.stack(tiles[i:i + 2048]))).reshape(-1, EF.K * EF.DIM).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


class Keys:
    """Handles for tiles: one per code tuple seen in memory (its memory tiles' mean vector); otherwise the tile's
    own vector."""

    def __init__(self, enc, mem_tiles):
        self.enc = enc
        self.S = VP.Store(np.zeros((1, EF.K * EF.DIM)))   # handle 0: the constant view token
        books = fn.normalize(enc.books.detach(), dim=-1).cpu().numpy().astype(np.float64)
        z = vectors(enc, mem_tiles)
        zz = z.reshape(len(z), EF.K, -1)
        used, rad = [], []
        for k in range(EF.K):
            u, r = NV.code_radii(zz[:, k], books[k])
            used.append(np.asarray(u))
            rad.append(np.asarray(r))
        CR.CC.clear()
        CR.CC.update({"z": np.zeros((1, EF.K * EF.DIM)), "books": books, "ca": np.full((1, EF.K), NV.NEW),
                      "fresh": {}, "used": used, "rad": rad, "cut": {k: 0.0 for k in range(EF.K)}, "hcodes": {},
                      "ids": {}})
        tups = [CR.code_of_vec(v) for v in z]
        groups = defaultdict(list)
        for v, t in zip(z, tups):
            if NV.NEW not in t:
                groups[t].append(v)
        self.mean = {t: self.S.add(np.mean(vs, 0)) for t, vs in groups.items()}
        for t, h in self.mean.items():             # the handle reads as its tuple, whatever its mean's nearest entry
            CR.CC["hcodes"][int(h)] = CR.CC["ids"].setdefault(t, len(CR.CC["ids"]))
        self.view = SP.set_of([0])
        self.new_pieces = int(sum(NV.NEW in t for t in tups))

    def handles(self, tiles):
        z = vectors(self.enc, tiles)
        out = []
        for v in z:
            t = CR.code_of_vec(v)
            h = self.mean.get(t)
            out.append(int(h) if h is not None else int(self.S.add(v)))
        return out


def tries_of(store):
    """(action, kind, front, held, front after, held after, the evaluator's outcome) per try. The outcome is the
    simulator's: bit 0 the front object changed, bit 1 the held object changed (2026-10-05: scoring by code
    tuples counted a change the codes cannot see as "unchanged", and so as right)."""
    out = []
    for a, lst in store.items():
        for f, h, o, ids, f1, h1 in lst:
            out.append((a, PF.kind_of(a, ids), f, h, f1, h1, int(o)))
    return out


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    path = args[0]
    out = Path(get("--out", str(path).replace(".pt", "_recall.json")))
    t0 = time.monotonic()
    GN.TINT = 0.0
    EF.START_SHARE = 0.5
    PF.COLLECT_STEPS = int(get("--collect-steps", "150000"))
    EF.Stream._tile = tagged_tile
    DR.Stream._tile = tagged_tile
    mem, probe = PF.collect_strat(399 + 555)
    M, P = tries_of(mem), tries_of(probe)
    enc = load(path, get("--interaction", "transition"), get("--gates", "0") == "1")
    keys = Keys(enc, [t[i] for t in M for i in (2, 3, 4, 5)])
    hm = {i: keys.handles([t[i] for t in M]) for i in (2, 3, 4, 5)}
    hp = {i: keys.handles([t[i] for t in P]) for i in (2, 3, 4, 5)}
    rng = np.random.default_rng(7)
    redraw = lambda tl: [(GN.tile(t.obj, np.zeros(3), rng) * 255).astype(np.uint8) for t in tl]
    hp2 = {i: keys.handles(redraw([t[i] for t in P])) for i in (2, 3)}
    print({"memory_tries": len(M), "probe_tries": len(P), "handles": keys.S.n, "memory_tuples": len(keys.mean),
           "memory_pieces_new": keys.new_pieces, "seconds": round(time.monotonic() - t0, 1)}, flush=True)
    res = {"weights": str(path), "memory_tries": len(M), "probe_tries": len(P), "memory_tuples": len(keys.mean),
           "memory_tiles_with_a_new_piece": keys.new_pieces, "actions": {}, "kinds": {}}
    per_kind, own_seen = defaultdict(list), defaultdict(list)
    same = []
    for a in ACTS:
        im = [i for i, t in enumerate(M) if t[0] == a]
        ip = [i for i, t in enumerate(P) if t[0] == a]
        k = np.array([(hm[2][i], hm[3][i], keys.view) for i in im], np.int64)
        af = np.array([(hm[4][i], hm[5][i]) for i in im], np.int64)
        cat = (af[:, 0] != k[:, 0]).astype(np.int64) + 2 * (af[:, 1] != k[:, 1])
        ta = time.monotonic()
        kd = IX.IndexKind(NAMES[a], keys.S, [slice(8 * j, 8 * j + 8) for j in range(4)], 3, 4, k, cat,
                          np.ones(len(k)), after=af)
        fit_s = time.monotonic() - ta
        qs = [(hp[2][i], hp[3][i], keys.view) for i in ip]
        qs2 = [(hp2[2][i], hp2[3][i], keys.view) for i in ip]
        pred, pred2 = kd.cat_of(qs), kd.cat_of(qs2)
        truth = [P[i][6] for i in ip]                  # the simulator's outcome, not the codes'
        res.setdefault("memory_categories_agree_with_simulator", {})[NAMES[a]] = round(
            float(np.mean([c == M[i][6] for c, i in zip(cat.tolist(), im)])), 4)
        seen = set(zip(kd.fid.tolist(), kd.hid.tolist()))
        for i, p, p2, tr, q in zip(ip, pred, pred2, truth, qs):
            per_kind[P[i][1]].append(p == tr)
            own_seen[P[i][1]].append((CR.code_id(keys.S, q[0]), CR.code_id(keys.S, q[1])) in seen)
            same.append(p == p2)
        res["actions"][NAMES[a]] = {"memory_keys": len(kd.keys), "fit_seconds": round(fit_s, 1),
                                    "admitted": kd.report["conditions"]["admitted"],
                                    "alpha": float(np.asarray(getattr(kd, "beta", np.nan)).ravel()[0]),
                                    "accuracy": round(float(np.mean([p == t for p, t in zip(pred, truth)])), 4),
                                    "unknown": round(float(np.mean([p is None for p in pred])), 4)}
        print(NAMES[a], res["actions"][NAMES[a]], flush=True)
    for name, q in PF.COLOUR.items():
        v = per_kind.get(q, [])
        res[f"colour: {name}"] = {"tries": len(v), "accuracy": round(float(np.mean(v)), 4) if v else None,
                                  "own_situation_in_memory": round(float(np.mean(own_seen[q])), 4) if v else None}
    res["kinds"] = {str(q): round(float(np.mean(v)), 4) for q, v in sorted(per_kind.items(), key=str)}
    res["own_situation_in_memory_all"] = round(float(np.mean([x for v in own_seen.values() for x in v])), 4)
    res["accuracy_all"] = round(float(np.mean([x for v in per_kind.values() for x in v])), 4)
    res["same_prediction_two_noise_draws"] = round(float(np.mean(same)), 4)
    if hasattr(enc, "trans"):                          # reported: the transition model's own answer, by its codes
        tm = {}
        with torch.no_grad():
            for ai, a in enumerate(ACTS):
                ip = [i for i, t in enumerate(P) if t[0] == a]
                zf = enc.pieces(enc.x(np.stack([P[i][2] for i in ip])))
                zh = enc.pieces(enc.x(np.stack([P[i][3] for i in ip])))
                pf, ph = enc.predict_after(ai, zf, zh)
                chf = (enc.quant(pf)[1] != enc.quant(zf)[1]).any(1).long()
                chh = (enc.quant(ph)[1] != enc.quant(zh)[1]).any(1).long()
                ok = (chf + 2 * chh).cpu() == torch.as_tensor([P[i][6] for i in ip])
                for i, o in zip(ip, ok.cpu().numpy()):
                    tm.setdefault(P[i][1], []).append(bool(o))
        res["transition_model"] = {name: round(float(np.mean(tm.get(q, [np.nan]))), 4) for name, q in PF.COLOUR.items()}
        res["transition_model"]["all"] = round(float(np.mean([x for v in tm.values() for x in v])), 4)
    res["seconds"] = round(time.monotonic() - t0, 1)
    print({k: v for k, v in res.items() if k not in ("kinds",)}, flush=True)
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
