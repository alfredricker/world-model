"""Card 170, gate part (a) and criterion 1: does SceneWNet's predicted next state match the reading after the touch?

  effects.py gate            fit seed 0 on 320 training episodes, score the 80 held out -> runs/170/effects_gate.json
  effects.py test 0 1 2      frozen runs/168 checkpoints on the 300 test episodes       -> runs/170/effects_test.json

No simulator truth is read. "After" is the encoder's own reading of the next pre-touch frame; thresholds are the planner's.
"""
import json, sys
import numpy as np
import torch
import model as M

OUT = M.ROOT / "runs/170"
LEARN, TEST = ["m400", "m600", "m800", "tail"], ["t500", "t700", "val"]


def bal(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    return round(50.0 * (pred[y].mean() + (~pred[~y]).mean()), 1)


def pct(m):
    return round(100.0 * float(np.mean(m)), 1) if len(m) else None


def score(net, meta, Z_ref, Z, tc, ep):
    """Z_ref: training frames, for the remembered ends of each slide (written-effects upper bound only)."""
    S, mov = net.n_slots, meta["mov_si"]
    kinds, cuts = meta["slot_kind"], {int(k): float(v) for k, v in meta["slot_cuts"].items()}
    _, Val_ref, _, _ = M.encode_all(net, Z_ref)
    H, Val, V, A = M.encode_all(net, Z)
    s = np.where(ep[:-1] == ep[1:])[0]
    u = M.make_u_act(tc, meta)
    out = {k: [] for k in ("p_chg", "val_next", "vis_next", "aim_next")}
    with torch.no_grad():
        for i in range(0, len(s), 256):
            b = s[i:i + 256]
            eo = net.encode(torch.from_numpy(Z[b]).to(M.DEV))
            to = net.forward_transition(eo["h"], eo["vis_logit"], eo["aim"], torch.from_numpy(u[b]).to(M.DEV))
            out["p_chg"].append(to["p_chg"].cpu().numpy()); out["val_next"].append(to["val_next"].cpu().numpy())
            out["vis_next"].append(torch.sigmoid(to["vis_next_logit"]).cpu().numpy()); out["aim_next"].append(to["aim_next"].cpu().numpy())
    p_chg, val_next, vis_next, aim_next = (np.concatenate(out[k], 0) for k in ("p_chg", "val_next", "vis_next", "aim_next"))

    v0, v1 = Val[s], Val[s + 1]
    vis0, vis1 = V[s] >= 0.5, V[s + 1] >= 0.5
    a0, a1 = A[s], A[s + 1]
    grp = meta["group"][np.linalg.norm(tc[s, None, :3] - meta["cen"][None], axis=-1).argmin(1)]
    touched = np.stack([grp == meta["slots"][k] for k in range(S)], 1)

    def same_slot(k, va, vb):
        return ((va > 0.0) == (vb > 0.0)) if kinds[k] == "lock" else (np.abs(va - vb) <= cuts[k])

    def same_mov(visa, aa, visb, ab):
        near = np.linalg.norm(aa - ab, axis=1) * 0.1 <= cuts[mov]
        return (visa == visb) & (~visb | near)

    # three predictions of the state after the touch
    chg = p_chg >= 0.5
    preds = {}
    mv = np.where(chg, val_next, v0)
    mvis = np.where(chg[:, mov], vis_next >= 0.5, vis0)
    maim = np.where((chg[:, mov] & mvis)[:, None], aim_next, a0)
    preds["model"] = (mv, mvis, maim)
    preds["no change"] = (v0, vis0, a0)
    wv = v0.copy()
    for k in range(S):
        if kinds[k] == "lock":
            wv[touched[:, k], k] = np.where(v0[touched[:, k], k] > 0.0, -2.5, 2.5)
        elif kinds[k] == "slide":
            lo, hi = np.quantile(Val_ref[:, k], [0.03, 0.97])
            wv[touched[:, k], k] = np.where(v1[touched[:, k], k] < v0[touched[:, k], k], lo, hi)   # direction given
    waim = np.where(touched[:, mov][:, None], tc[s, 3:6], a0)
    preds["written (og5)"] = (wv, vis0, waim)

    read_chg = np.stack([~same_slot(k, v0[:, k], v1[:, k]) if k != mov else ~same_mov(vis0, a0, vis1, a1) for k in range(S)], 1)
    locks = [k for k in range(S) if kinds[k] == "lock"]
    slides = [k for k in range(S) if kinds[k] == "slide"]
    zc = float(meta["z_cut"])
    table_move = touched[:, mov] & read_chg[:, mov] & vis0 & vis1 & (a0[:, 2] <= zc) & (a1[:, 2] <= zc)
    put_in = touched[:, mov] & vis0 & vis1 & (a0[:, 2] <= zc) & (a1[:, 2] > zc)
    # pulls that carry the slide across the middle of its remembered range: the kind the planner imagines and replays.
    # (How far a recorded pull goes is not in the pre-touch action, so short pulls cannot be predicted to the threshold.)
    mid = {k: float(np.mean(np.quantile(Val_ref[:, k], [0.03, 0.97]))) for k in slides}
    across = {k: touched[:, k] & read_chg[:, k] & ((v0[:, k] - mid[k]) * (v1[:, k] - mid[k]) < 0) for k in slides}
    cont = meta["cont_si"]
    in_drawer = touched[:, cont] & ((vis0 & (a0[:, 2] > zc)) | ~vis0)

    res = {"n": dict(transitions=int(len(s)),
                     press_flips=int(sum((touched[:, k] & read_chg[:, k]).sum() for k in locks)),
                     pulls_that_move=int(sum((touched[:, k] & read_chg[:, k]).sum() for k in slides)),
                     pulls_across_middle=int(sum(across[k].sum() for k in slides)),
                     table_moves=int(table_move.sum()), put_ins=int(put_in.sum()),
                     drawer_touches_cube_inside=int(in_drawer.sum()), of_which_hidden_after=int((in_drawer & ~vis1).sum()))}
    for name, (pv, pvis, paim) in preds.items():
        ok = np.stack([same_slot(k, pv[:, k], v1[:, k]) if k != mov else same_mov(pvis, paim, vis1, a1) for k in range(S)], 1)
        kept = np.stack([same_slot(k, pv[:, k], v0[:, k]) if k != mov else same_mov(pvis, paim, vis0, a0) for k in range(S)], 1)
        res[name] = dict(
            press_flip=pct(np.concatenate([ok[touched[:, k] & read_chg[:, k], k] for k in locks])),
            pull_moves=pct(np.concatenate([ok[touched[:, k] & read_chg[:, k], k] for k in slides])),
            pull_moves_per_slide={int(k): pct(ok[touched[:, k] & read_chg[:, k], k]) for k in slides},
            pull_across_middle=pct(np.concatenate([ok[across[k], k] for k in slides])),
            pull_across_middle_per_slide={int(k): pct(ok[across[k], k]) for k in slides},
            table_move=pct(ok[table_move, mov]),
            put_in=pct(ok[put_in, mov]),
            hidden_after_drawer_touch_bal=bal(~pvis[in_drawer], ~vis1[in_drawer]),
            unchanged_kept=pct(kept[~read_chg]),
            whole_state=pct(ok.all(1)),
            whole_state_when_something_changed=pct(ok.all(1)[read_chg.any(1)]),
        )
    # how far off are the model's misses? (diagnostic)
    res["model_error"] = dict(
        slide_abs_err_on_moved_pulls={int(k): [round(float(x), 3) for x in np.quantile(np.abs(mv[:, k] - v1[:, k])[touched[:, k] & read_chg[:, k]], [0.5, 0.8, 0.95])] for k in slides},
        slide_cut={int(k): cuts[k] for k in slides},
        table_move_cm=[round(float(x), 2) for x in np.quantile((np.linalg.norm(maim - a1, axis=1) * 10.0)[table_move], [0.5, 0.8, 0.95])],
        mov_cut_cm=round(cuts[mov] * 100.0, 2),
        p_chg_own_slot_on_table_moves=pct(chg[table_move, mov]),
        pulls_across_middle_predicted_to_move={int(k): pct(chg[across[k], k]) for k in slides},
        match_given_predicted_to_move={int(k): pct(same_slot(k, mv[:, k], v1[:, k])[across[k] & chg[:, k]]) for k in slides},
        end_reading_spread_on_pulls_across_middle={int(k): {d: [round(float(x), 3) for x in np.quantile(v1[across[k] & ((v1[:, k] < v0[:, k]) == (d == "lo")), k], [0.1, 0.5, 0.9])] for d in ("lo", "hi")} for k in slides},
        predicted_end_spread={int(k): {d: [round(float(x), 3) for x in np.quantile(mv[across[k] & chg[:, k] & ((v1[:, k] < v0[:, k]) == (d == "lo")), k], [0.1, 0.5, 0.9])] for d in ("lo", "hi")} for k in slides},
    )
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "gate":
        Z, tc, ep = M.load_split(LEARN)
        tr = (ep % 100) < 80
        net, meta = M.train_model(Z[tr], tc[tr], ep[tr], seed=0, steps=3500)
        res = score(net, meta, Z[tr], Z[~tr], tc[~tr], ep[~tr])
        json.dump(res, open(OUT / "effects_gate.json", "w"), indent=1)
    else:
        Zr, _, _ = M.load_split(LEARN)
        Z, tc, ep = M.load_split(TEST)
        res = {}
        for sd in sys.argv[2:]:
            ck = torch.load(M.ROOT / f"runs/168/scene_wnet_{sd}.pt", map_location=M.DEV, weights_only=False)
            net = M.SceneWNet(n_slots=ck["n_slots"], mov_si=ck["mov_si"]).to(M.DEV)
            net.load_state_dict(ck["state_dict"]); net.eval()
            res[sd] = score(net, ck["meta"], Zr, Z, tc, ep)
        json.dump(res, open(OUT / "effects_test.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
