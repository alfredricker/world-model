"""Card 168: Feasibility gate (on held-out LEARN fold ONLY) and frozen 3-seed evaluation on TEST (t500, t700, val).

Usage:
  ev.py gate                                -> fits seed 0 on 320 LEARN episodes, scores on 80 held-out LEARN episodes + action-only baseline
  ev.py fit <seed>                          -> fits SceneWNet seed on all 400 LEARN episodes -> runs/168/scene_wnet_<seed>.pt
  ev.py eval <seed0> <seed1> <seed2>        -> scores frozen checkpoints on the 300 TEST episodes -> experiments/168-.../results.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import common as C141
import model as M

ROOT = M.ROOT
OUT = ROOT / "runs/168"
LEARN_SETS = ["m400", "m600", "m800", "tail"]
TEST_SETS = ["t500", "t700", "val"]


def slot_differs(H, Val, meta, k, a, b):
    kd = meta["slot_kind"][k]
    cut = meta["slot_cuts"][k]
    if kd == "slide":
        return np.abs(Val[a, k] - Val[b, k]) > cut
    if kd == "lock":
        return (Val[a, k] > 0.0) != (Val[b, k] > 0.0)
    return np.linalg.norm(H[a, k] - H[b, k], axis=1) > cut


def bookkeeping_map(net, meta, Z_ref, t_ref):
    """Bookkeeping ONLY (exactly as tools/card141/current.py lines 83-91), fitted strictly on LEARN reference frames."""
    H, Val, _, _ = M.encode_all(net, Z_ref)
    a, b = C141.pairs(t_ref, seed=0)
    mov_si = meta["mov_si"]
    non_mov = [k for k in range(net.n_slots) if k != mov_si]
    pred_diff = {k: slot_differs(H, Val, meta, k, a, b) for k in non_mov}
    var_map = {}
    for vname, y_true in [
        ("drawer", np.abs(t_ref["drawer"][a] - t_ref["drawer"][b]) >= 0.01),
        ("window", np.abs(t_ref["window"][a] - t_ref["window"][b]) >= 0.01),
        ("lock0", t_ref["locks"][a, 0] != t_ref["locks"][b, 0]),
        ("lock1", t_ref["locks"][a, 1] != t_ref["locks"][b, 1]),
    ]:
        var_map[vname] = max(non_mov, key=lambda k: C141.bal(pred_diff[k], y_true) or 0.0)
    return var_map


def score_representation(net, meta, var_map, Z_ev, t_ev):
    """Evaluate SceneWNet single-frame encoder on Card 141's exact pair cells + cube position & visibility metrics."""
    H, Val, V, A = M.encode_all(net, Z_ev)
    mov_si = meta["mov_si"]
    cuts = meta["slot_cuts"]
    aim_world = A * 0.1 + M.CEN
    vis = V >= 0.5

    a, b = C141.pairs(t_ev, seed=0)
    differs = {}
    va, vb = vis[a], vis[b]
    d_aim = np.linalg.norm(aim_world[a] - aim_world[b], axis=1)
    differs["cube"] = np.where(va & vb, d_aim > cuts[mov_si], va != vb)

    for vname in ("drawer", "window", "lock0", "lock1"):
        differs[vname] = slot_differs(H, Val, meta, var_map[vname], a, b)

    pairs = C141.score_pairs(t_ev, a, b, differs)
    err_cm = np.linalg.norm(aim_world[:, :2] - t_ev["cube"][:, :2], axis=1) * 100.0
    cube_val = {}
    for w in ("table", "open", "hidden"):
        m = t_ev["where"] == w
        cube_val[w] = dict(
            n=int(m.sum()),
            median_cm=round(float(np.median(err_cm[m])), 2) if m.any() else None,
            within_2cm=round(float((err_cm[m] <= 2.0).mean()), 3) if m.any() else None,
            read_in_view=round(float(vis[m].mean()), 3) if m.any() else None,
        )
    non_14 = [k for k in pairs if "1-4" not in k]
    mean_non14 = round(float(np.mean([pairs[k]["bal"] for k in non_14])), 2)
    return dict(
        pairs=pairs,
        mean_non14_bal=mean_non14,
        cube_value=cube_val,
        in_view_bal=C141.bal(vis, t_ev["where"] != "hidden"),
    )


def score_consequences(net, meta, var_map, Z_ev, tc_ev, ep_ev, t_ev):
    """Evaluate SceneWNet on held-out touch consequences, counterfactual lock flips, and drawer-hides-cube transitions."""
    same = ep_ev[:-1] == ep_ev[1:]
    s_idx = np.where(same)[0]
    u_act_ev = M.make_u_act(tc_ev, meta)

    p_chg_all, vis_next_all = [], []
    with torch.no_grad():
        for i in range(0, len(s_idx), 256):
            bi = s_idx[i:i + 256]
            zt = torch.from_numpy(Z_ev[bi]).to(M.DEV)
            ut = torch.from_numpy(u_act_ev[bi]).to(M.DEV)
            eo = net.encode(zt)
            to = net.forward_transition(eo["h"], eo["vis_logit"], eo["aim"], ut)
            p_chg_all.append(to["p_chg"].cpu().numpy())
            vis_next_all.append(torch.sigmoid(to["vis_next_logit"]).cpu().numpy())
    p_chg = np.concatenate(p_chg_all, 0)
    vis_next = np.concatenate(vis_next_all, 0)

    # 1. Slide-pull consequence accuracy (Criterion 2a)
    pl = np.linalg.norm(tc_ev[s_idx, None, :3] - meta["cen"][None], axis=-1).argmin(1)
    grp = meta["group"][pl]
    disp = np.linalg.norm(tc_ev[s_idx, 3:6] - tc_ev[s_idx, :3], axis=1)

    dr_si = var_map["drawer"]; wi_si = var_map["window"]
    dr_u = meta["slots"][dr_si]; wi_u = meta["slots"][wi_si]
    l0_flip = t_ev["locks"][s_idx + 1, 0] != t_ev["locks"][s_idx, 0]
    l1_flip = t_ev["locks"][s_idx + 1, 1] != t_ev["locks"][s_idx, 1]
    mixed = l0_flip | l1_flip

    dr_true_mv = np.abs(t_ev["drawer"][s_idx + 1] - t_ev["drawer"][s_idx]) > 0.01
    wi_true_mv = np.abs(t_ev["window"][s_idx + 1] - t_ev["window"][s_idx]) > 0.01

    dr_pull = (grp == dr_u) & (~mixed) & (disp > 0.10)
    wi_pull = (grp == wi_u) & (~mixed) & (disp > 0.10)
    all_pull_true = np.concatenate([dr_true_mv[dr_pull], wi_true_mv[wi_pull]])
    all_pull_pred = np.concatenate([p_chg[dr_pull, dr_si] >= 0.5, p_chg[wi_pull, wi_si] >= 0.5])

    dr_pull_bal = C141.bal(p_chg[dr_pull, dr_si] >= 0.5, dr_true_mv[dr_pull])
    wi_pull_bal = C141.bal(p_chg[wi_pull, wi_si] >= 0.5, wi_true_mv[wi_pull])
    pull_bal = C141.bal(all_pull_pred, all_pull_true)

    # 2. Counterfactual lock sensitivity (Criterion 2b):
    # Replace the lock slot's embedding h[:, c_si] with the opposite-state lock prototype while keeping u_act IDENTICAL!
    # Also report 2-step imagined-press counterfactual (press cause button -> then pull slide).
    cf_swap_flips, cf_press_flips = [], []
    for slide_si, pull_mask, true_mv, lk_col in [
        (dr_si, dr_pull, dr_true_mv, 0),
        (wi_si, wi_pull, wi_true_mv, 1),
    ]:
        c_si = meta["cause_of"].get(slide_si, var_map[f"lock{lk_col}"])
        protos = meta["cause_protos"].get(c_si)
        press_u = meta["cause_press_u"].get(c_si)
        if protos is None:
            continue
        unlocked_rows = np.where(pull_mask & true_mv & (t_ev["locks"][s_idx, lk_col] == 1))[0]
        if not len(unlocked_rows):
            continue
        bi = s_idx[unlocked_rows]
        with torch.no_grad():
            zt = torch.from_numpy(Z_ev[bi]).to(M.DEV)
            ut = torch.from_numpy(u_act_ev[bi]).to(M.DEV)
            eo = net.encode(zt)
            p_orig = net.forward_transition(eo["h"], eo["vis_logit"], eo["aim"], ut)["p_chg"][:, slide_si]
            ok_orig = p_orig >= 0.5
            if ok_orig.any():
                # (a) Direct lock-slot prototype swap (Criterion 2b exact definition)
                h_cf = eo["h"].clone()
                pt = torch.from_numpy(protos).to(M.DEV)  # [2, D]: pt[0]=disabled, pt[1]=enabled
                d0 = (h_cf[:, c_si] - pt[0]).norm(dim=-1)
                d1 = (h_cf[:, c_si] - pt[1]).norm(dim=-1)
                opp = torch.where((d0 < d1)[:, None], pt[1][None], pt[0][None])
                h_cf[:, c_si] = opp
                p_swapped = net.forward_transition(h_cf, eo["vis_logit"], eo["aim"], ut)["p_chg"][:, slide_si]
                cf_swap_flips.append((p_swapped[ok_orig] < 0.5).cpu().numpy())

                # (b) 2-step imagined button press via forward_transition -> then pull slide
                if press_u is not None:
                    up = torch.from_numpy(press_u)[None].expand(len(bi), -1).to(M.DEV)
                    t1 = net.forward_transition(eo["h"], eo["vis_logit"], eo["aim"], up)
                    p_after_press = net.forward_transition(t1["h_next"], t1["vis_next_logit"], t1["aim_next"], ut)["p_chg"][:, slide_si]
                    cf_press_flips.append((p_after_press[ok_orig] < 0.5).cpu().numpy())

    cf_swap_rate = round(100.0 * float(np.concatenate(cf_swap_flips).mean()), 2) if cf_swap_flips else 0.0
    cf_press_rate = round(100.0 * float(np.concatenate(cf_press_flips).mean()), 2) if cf_press_flips else 0.0

    # 3. Cross-variable occlusion consequence (Criterion 3):
    in_open = t_ev["where"][s_idx] == "open"
    becomes_hidden = t_ev["where"][s_idx + 1] == "hidden"
    pred_hidden = vis_next < 0.5
    hide_all_bal = C141.bal(pred_hidden[in_open], becomes_hidden[in_open])
    in_open_dr_touch = in_open & (grp == dr_u)
    hide_dr_bal = C141.bal(pred_hidden[in_open_dr_touch], becomes_hidden[in_open_dr_touch])

    return dict(
        pull_bal=pull_bal,
        drawer_pull_bal=dr_pull_bal,
        window_pull_bal=wi_pull_bal,
        n_pulls=[int(dr_pull.sum()), int(wi_pull.sum())],
        counterfactual_lock_flip_pct=cf_swap_rate,
        counterfactual_imagined_press_flip_pct=cf_press_rate,
        drawer_hides_cube_bal=hide_all_bal,
        drawer_touch_hides_cube_bal=hide_dr_bal,
        n_open_transitions=[int((in_open & becomes_hidden).sum()), int((in_open & ~becomes_hidden).sum())],
    )


def fit_action_only_baseline(tc_tr, ep_tr, t_tr, tc_ev, ep_ev, t_ev, meta):
    """Pre-touch action-only baseline (u_act [8] without scene slots): scored on slide pulls and counterfactual lock flip."""
    ua_tr = M.make_u_act(tc_tr, meta)
    ua_ev = M.make_u_act(tc_ev, meta)
    s_tr = np.where(ep_tr[:-1] == ep_tr[1:])[0]
    s_ev = np.where(ep_ev[:-1] == ep_ev[1:])[0]

    pl_tr = np.linalg.norm(tc_tr[s_tr, None, :3] - meta["cen"][None], axis=-1).argmin(1)
    disp_tr = np.linalg.norm(tc_tr[s_tr, 3:6] - tc_tr[s_tr, :3], axis=1)
    slide_us = [meta["slots"][s] for s in meta["eff_sis"]]
    pull_tr = np.isin(meta["group"][pl_tr], slide_us) & (disp_tr > 0.10)

    y_tr = ((np.abs(t_tr["drawer"][s_tr + 1] - t_tr["drawer"][s_tr]) > 0.01) |
            (np.abs(t_tr["window"][s_tr + 1] - t_tr["window"][s_tr]) > 0.01))[pull_tr].astype(np.float32)
    X_tr = torch.from_numpy(ua_tr[s_tr][pull_tr]).to(M.DEV)
    Y_tr = torch.from_numpy(y_tr).to(M.DEV)

    mlp = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)).to(M.DEV)
    opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3)
    for _ in range(800):
        lg = mlp(X_tr).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(lg, Y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    pl_ev = np.linalg.norm(tc_ev[s_ev, None, :3] - meta["cen"][None], axis=-1).argmin(1)
    disp_ev = np.linalg.norm(tc_ev[s_ev, 3:6] - tc_ev[s_ev, :3], axis=1)
    l_flip = (t_ev["locks"][s_ev + 1, 0] != t_ev["locks"][s_ev, 0]) | (t_ev["locks"][s_ev + 1, 1] != t_ev["locks"][s_ev, 1])
    pull_ev = np.isin(meta["group"][pl_ev], slide_us) & (~l_flip) & (disp_ev > 0.10)
    y_ev = ((np.abs(t_ev["drawer"][s_ev + 1] - t_ev["drawer"][s_ev]) > 0.01) |
            (np.abs(t_ev["window"][s_ev + 1] - t_ev["window"][s_ev]) > 0.01))[pull_ev]
    with torch.no_grad():
        p_ev = torch.sigmoid(mlp(torch.from_numpy(ua_ev[s_ev][pull_ev]).to(M.DEV)).squeeze(-1)).cpu().numpy() >= 0.5
    real_bal = C141.bal(p_ev, y_ev)
    cf_bal = round(100.0 - real_bal, 1)
    return dict(real_pull_bal=real_bal, counterfactual_pull_bal=cf_bal, counterfactual_flip_pct=0.0)


def subset_truth(t_full, mask):
    idx = np.where(mask)[0]
    return dict(
        cube=t_full["cube"][idx], drawer=t_full["drawer"][idx], window=t_full["window"][idx],
        locks=t_full["locks"][idx], set=t_full["set"][idx], row=np.arange(len(idx)),
        ep=t_full["ep"][idx], where=t_full["where"][idx],
    )


def run_gate():
    """Gate strictly on LEARN (320 train episodes / 80 held-out LEARN episodes — zero TEST exposure)."""
    OUT.mkdir(parents=True, exist_ok=True)
    Z_all, tc_all, ep_all = M.load_split(LEARN_SETS)
    t_all = C141.truth(LEARN_SETS)
    ep_mod = ep_all % 100
    tr_m = ep_mod < 80
    va_m = ep_mod >= 80

    t0 = time.time()
    net, meta = M.train_model(Z_all[tr_m], tc_all[tr_m], ep_all[tr_m], seed=0, steps=3500)
    fit_sec = round(time.time() - t0, 1)

    t_tr = subset_truth(t_all, tr_m)
    t_va = subset_truth(t_all, va_m)
    var_map = bookkeeping_map(net, meta, Z_all[tr_m], t_tr)
    rep = score_representation(net, meta, var_map, Z_all[va_m], t_va)
    con = score_consequences(net, meta, var_map, Z_all[va_m], tc_all[va_m], ep_all[va_m], t_va)
    base_ao = fit_action_only_baseline(tc_all[tr_m], ep_all[tr_m], t_tr, tc_all[va_m], ep_all[va_m], t_va, meta)

    gate_res = dict(fit_sec=fit_sec, var_map=var_map, slot_kind=meta["slot_kind"], slot_cuts=meta["slot_cuts"],
                    representation=rep, consequences=con, action_only_baseline=base_ao)
    json.dump(gate_res, open(OUT / "gate.json", "w"), indent=1)
    print(json.dumps(gate_res, indent=1), flush=True)


def run_fit(seed):
    OUT.mkdir(parents=True, exist_ok=True)
    Z_tr, tc_tr, ep_tr = M.load_split(LEARN_SETS)
    t_tr = C141.truth(LEARN_SETS)
    t0 = time.time()
    net, meta = M.train_model(Z_tr, tc_tr, ep_tr, seed=seed, steps=3500)
    fit_sec = round(time.time() - t0, 1)
    var_map = bookkeeping_map(net, meta, Z_tr, t_tr)
    ck = dict(
        seed=seed, fit_sec=fit_sec, n_slots=net.n_slots, mov_si=net.mov_si,
        state_dict=net.state_dict(), meta=meta, var_map=var_map,
    )
    torch.save(ck, OUT / f"scene_wnet_{seed}.pt")
    print(f"Saved runs/168/scene_wnet_{seed}.pt in {fit_sec}s | var_map={var_map}", flush=True)


def run_eval(seeds):
    Z_te, tc_te, ep_te = M.load_split(TEST_SETS)
    t_te = C141.truth(TEST_SETS)
    Z_tr, tc_tr, ep_tr = M.load_split(LEARN_SETS)
    t_tr = C141.truth(LEARN_SETS)
    bed141 = json.load(open(ROOT / "runs/141/bed.json"))

    per_seed = {}
    meta0 = None
    for s in seeds:
        ck = torch.load(OUT / f"scene_wnet_{s}.pt", map_location=M.DEV, weights_only=False)
        if meta0 is None:
            meta0 = ck["meta"]
        net = M.SceneWNet(n_slots=ck["n_slots"], mov_si=ck["mov_si"]).to(M.DEV)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        rep = score_representation(net, ck["meta"], ck["var_map"], Z_te, t_te)
        con = score_consequences(net, ck["meta"], ck["var_map"], Z_te, tc_te, ep_te, t_te)
        per_seed[str(s)] = dict(fit_sec=ck["fit_sec"], var_map=ck["var_map"], representation=rep, consequences=con)
        print(f"=== SEED {s} ===", flush=True)
        print("  Pairs:", {k: v["bal"] for k, v in rep["pairs"].items()}, "mean_non14:", rep["mean_non14_bal"], flush=True)
        print("  Cube cm:", {k: v["median_cm"] for k, v in rep["cube_value"].items()}, "in_view_bal:", rep["in_view_bal"], flush=True)
        print("  Conseq:", con, flush=True)

    base_ao = fit_action_only_baseline(tc_tr, ep_tr, t_tr, tc_te, ep_te, t_te, meta0)

    cells = list(per_seed[str(seeds[0])]["representation"]["pairs"].keys())
    mean_pairs = {c: round(float(np.mean([per_seed[str(s)]["representation"]["pairs"][c]["bal"] for s in seeds])), 2) for c in cells}
    mean_cube_cm = {
        w: round(float(np.mean([per_seed[str(s)]["representation"]["cube_value"][w]["median_cm"] for s in seeds])), 2)
        for w in ("table", "open", "hidden")
    }
    mean_con = {}
    for k in ("pull_bal", "drawer_pull_bal", "window_pull_bal", "counterfactual_lock_flip_pct",
              "counterfactual_imagined_press_flip_pct", "drawer_hides_cube_bal", "drawer_touch_hides_cube_bal"):
        mean_con[k] = round(float(np.mean([per_seed[str(s)]["consequences"][k] for s in seeds])), 2)

    summary = dict(
        card="168",
        seeds=seeds,
        mean_pairs=mean_pairs,
        mean_non14_bal=round(float(np.mean([per_seed[str(s)]["representation"]["mean_non14_bal"] for s in seeds])), 2),
        mean_cube_median_cm=mean_cube_cm,
        mean_in_view_bal=round(float(np.mean([per_seed[str(s)]["representation"]["in_view_bal"] for s in seeds])), 2),
        mean_consequences=mean_con,
        og4_baseline=dict(
            pairs={c: bed141["current"]["pairs"][c]["bal"] for c in cells},
            mean_non14_bal=round(float(np.mean([bed141["current"]["pairs"][c]["bal"] for c in cells if "1-4" not in c])), 2),
            cube_median_cm={w: bed141["current"]["cube_value"][w]["median_cm"] for w in ("table", "open", "hidden")},
        ),
        probe_upper_bound=dict(
            pairs={c: bed141["probe"]["pairs"][c]["bal"] for c in cells},
            cube_median_cm={w: bed141["probe"]["cube_value"][w]["median_cm"] for w in ("table", "open")},
        ),
        action_only_baseline=base_ao,
        per_seed=per_seed,
    )
    out_file = ROOT / "experiments/168-consequence-trained-scene-world-model/results.json"
    json.dump(summary, open(out_file, "w"), indent=2)
    print("\n=== 3-SEED MEAN SUMMARY ===")
    print("  mean_pairs:", mean_pairs)
    print("  mean_non14_bal:", summary["mean_non14_bal"], "vs og4:", summary["og4_baseline"]["mean_non14_bal"])
    print("  mean_cube_cm:", mean_cube_cm, "vs og4:", summary["og4_baseline"]["cube_median_cm"])
    print("  mean_consequences:", mean_con)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gate":
        run_gate()
    elif cmd == "fit":
        run_fit(int(sys.argv[2]))
    elif cmd == "eval":
        run_eval([int(x) for x in sys.argv[2:]])
