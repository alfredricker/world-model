"""Card 171, gate part (a) and criteria 1 and 3.

  effects171.py gate <arm>     arm in base | chains | rng | both: fit seed 0 on 320 training episodes, score the 80
                               held out -> runs/171/gate_a_<arm>.json   (base is card 168's training, unchanged)
  effects171.py fit <seed> [arm]   this card's model (or one arm) on all 400 training episodes -> runs/171/scene_wnet_<seed>.pt
  effects171.py test 0 1 2     frozen checkpoints on the 300 test episodes -> runs/171/test.json
  effects171.py summarize      the four gate files against the card's bars -> runs/171/gate_a.json

No simulator truth is read by the gate. "After" is the encoder's own reading of the frame after a touch; thresholds
are the planner's. The test mode also repeats card 168's representation measures, which score against truth.
"""
import json, sys, time
import numpy as np
import torch
import model171 as M1
M = M1.M                     # card 168's model.py
import effects as E170       # card 170: one-touch scoring

OUT = M.ROOT / "runs/171"
LEARN, TEST = ["m400", "m600", "m800", "tail"], ["t500", "t700", "val"]
ARMS = dict(base=(False, False), chains=(True, False), rng=(False, True), both=(True, True))
pct, bal = E170.pct, E170.bal


def score(net, meta, Z_ref, tc_ref, ep_ref, Z, tc, ep, use_rng):
    """Measures 1-4 of the card's gate (a) for one model on one set of episodes."""
    S, mov = net.n_slots, meta["mov_si"]
    kinds, cuts = meta["slot_kind"], {int(k): float(v) for k, v in meta["slot_cuts"].items()}
    slides = [k for k in range(S) if kinds[k] == "slide"]
    zc, cont = float(meta["z_cut"]), meta["cont_si"]
    res = {"one_touch": E170.score(net, meta, Z_ref, Z, tc, ep)}

    H_ref, Val_ref, _, _ = M.encode_all(net, Z_ref)
    bank_h, bank_val = torch.from_numpy(H_ref).to(M.DEV), torch.from_numpy(Val_ref).to(M.DEV)
    H, Val, V, A = M.encode_all(net, Z)
    vis_r = V >= 0.5
    u = torch.from_numpy(M.make_u_act(tc, meta)).to(M.DEV)
    Zt = torch.from_numpy(Z).to(M.DEV)
    same = ep[:-1] == ep[1:]
    grp = meta["group"][np.linalg.norm(tc[:, None, :3] - meta["cen"][None], axis=-1).argmin(1)]

    @torch.no_grad()
    def rollout(starts, L, teacher=False):
        """Imagine L consecutive touches from the reading of frame `starts`; return the imagined state as numpy.
        teacher=True (diagnostic): every touch is predicted from the real reading before it, and only the predicted
        changes are carried along, so errors add up but no prediction is made from an imagined state."""
        out = dict(val=[], lo=[], hi=[], vis=[], aim=[])
        for i in range(0, len(starts), 256):
            c = torch.as_tensor(starts[i:i + 256], dtype=torch.long, device=M.DEV)
            eo = net.encode(Zt[c])
            st = dict(h=eo["h"], vis_logit=eo["vis_logit"], aim=eo["aim"])
            val, c_vis, c_aim = eo["val"].clone(), eo["vis_logit"].clone(), eo["aim"].clone()
            lo = torch.full_like(val, float("nan")); hi = torch.full_like(val, float("nan"))
            for j in range(L):
                tr = net.forward_transition(st["h"], st["vis_logit"], st["aim"], u[c + j])
                chg = tr["p_chg"] >= 0.5
                chg_nm = chg.clone(); chg_nm[:, mov] = False
                val = torch.where(chg_nm, tr["val_next"], val)
                q_lo = tr["q_lo"] if (use_rng and "q_lo" in tr) else tr["val_next"]
                q_hi = tr["q_hi"] if (use_rng and "q_hi" in tr) else tr["val_next"]
                lo = torch.where(chg_nm, q_lo, lo); hi = torch.where(chg_nm, q_hi, hi)
                c_aim = torch.where((chg[:, mov] & (tr["vis_next_logit"] >= 0.0))[:, None], tr["aim_next"], c_aim)
                c_vis = torch.where(chg[:, mov], tr["vis_next_logit"], c_vis)
                if teacher and j < L - 1:
                    eo = net.encode(Zt[c + j + 1])
                    st = dict(h=eo["h"], vis_logit=eo["vis_logit"], aim=eo["aim"])
                else:
                    st = M1.imagine_next(net, st, tr, bank_h, bank_val)
            out["val"].append(val.cpu().numpy()); out["lo"].append(lo.cpu().numpy()); out["hi"].append(hi.cpu().numpy())
            out["vis"].append((c_vis >= 0.0).cpu().numpy()); out["aim"].append(c_aim.cpu().numpy())
        return {k: np.concatenate(v, 0) for k, v in out.items()}

    def matches(im, end):
        """Per slot: is the imagined state the same as the reading of frame `end`, by the planner's tests?"""
        ok = np.zeros((len(end), S), bool)
        for k in range(S):
            if k == mov:
                near = np.linalg.norm(im["aim"] - A[end], axis=1) * 0.1 <= cuts[mov]
                ok[:, k] = (im["vis"] == vis_r[end]) & (~vis_r[end] | near)
            elif kinds[k] == "lock":
                ok[:, k] = (im["val"][:, k] > 0.0) == (Val[end, k] > 0.0)
            else:
                v = Val[end, k]
                in_rng = (v >= im["lo"][:, k] - cuts[k]) & (v <= im["hi"][:, k] + cuts[k])   # False where no range (nan)
                ok[:, k] = (np.abs(im["val"][:, k] - v) <= cuts[k]) | in_rng
        return ok

    # --- measure 2: a pull's read end lies in the predicted range ---
    s = np.where(same)[0]
    one = rollout(s, 1)
    same_ref = ep_ref[:-1] == ep_ref[1:]
    grp_ref = meta["group"][np.linalg.norm(tc_ref[:, None, :3] - meta["cen"][None], axis=-1).argmin(1)]
    hit, hit_mem, w_model, w_mem, n_pull = [], [], [], [], 0
    for k in slides:
        mvd = (grp[s] == meta["slots"][k]) & (np.abs(Val[s + 1, k] - Val[s, k]) > cuts[k])
        r = np.where(same_ref & (grp_ref[:-1] == meta["slots"][k]))[0]
        dref = Val_ref[r + 1, k] - Val_ref[r, k]
        mem = {d: np.quantile(Val_ref[r + 1, k][(dref < -cuts[k]) if d == "lo" else (dref > cuts[k])], [M1.Q_LO, M1.Q_HI]) for d in ("lo", "hi")}
        v1 = Val[s + 1, k][mvd]
        down = v1 < Val[s, k][mvd]
        m_lo = np.where(down, mem["lo"][0], mem["hi"][0]); m_hi = np.where(down, mem["lo"][1], mem["hi"][1])
        p_lo, p_hi = one["lo"][mvd, k], one["hi"][mvd, k]            # nan where the model predicted no move
        hit.append((v1 >= p_lo - cuts[k]) & (v1 <= p_hi + cuts[k]))
        hit_mem.append((v1 >= m_lo - cuts[k]) & (v1 <= m_hi + cuts[k]))
        pm = ~np.isnan(p_lo)
        w_model.append((p_hi - p_lo)[pm]); w_mem.append((m_hi - m_lo)[pm]); n_pull += int(mvd.sum())
    wm, wr = float(np.mean(np.concatenate(w_model))), float(np.mean(np.concatenate(w_mem)))
    res["pull_end_in_range"] = dict(
        n=n_pull, model=pct(np.concatenate(hit)), remembered_range=pct(np.concatenate(hit_mem)),
        per_slide={int(k): pct(h) for k, h in zip(slides, hit)},
        mean_width_model=round(wm, 3), mean_width_remembered=round(wr, 3), width_ratio=round(wm / max(wr, 1e-6), 2),
    )

    # --- measure 3: whole state after three imagined touches, against the reading after the third ---
    n_t = len(same)
    ok3 = same[:n_t - 2] & same[1:n_t - 1] & same[2:]
    c3 = np.where(ok3)[0]
    m3 = matches(rollout(c3, 3), c3 + 3)
    m1 = matches(rollout(c3 + 2, 1), c3 + 3)                       # upper bound: the third touch from the real reading
    mt = matches(rollout(c3, 3, teacher=True), c3 + 3)             # diagnostic: errors add up, no imagined inputs
    eo0 = dict(val=Val[c3], lo=np.full((len(c3), S), np.nan), hi=np.full((len(c3), S), np.nan), vis=vis_r[c3], aim=A[c3])
    m0 = matches(eo0, c3 + 3)                                      # nothing changes over the three touches
    res["three_touches"] = dict(
        n=int(len(c3)), whole_state=pct(m3.all(1)), last_touch_from_real_reading=pct(m1.all(1)), nothing_changes=pct(m0.all(1)),
        each_touch_from_real_reading=pct(mt.all(1)),
        per_slot={f"{k}:{kinds[k]}": pct(m3[:, k]) for k in range(S)},
        per_slot_each_touch_from_real={f"{k}:{kinds[k]}": pct(mt[:, k]) for k in range(S)},
        per_slot_last_touch_from_real={f"{k}:{kinds[k]}": pct(m1[:, k]) for k in range(S)},
    )

    # --- measure 4: cube in view or not after a drawer touch, the cube put in by an earlier imagined touch ---
    put = same & (grp[:-1] == meta["slots"][mov]) & vis_r[1:] & (A[1:, 2] > zc) & ~(A[:-1, 2] > zc)
    dr = np.where(same & (grp[:-1] == meta["slots"][cont]))[0]
    pred, real, pred1 = [], [], []
    for L in (2, 3):
        e = dr[dr - (L - 1) >= 0]
        st0 = e - (L - 1)
        keep = put[st0] & np.all([same[st0 + j] for j in range(L)], 0)
        if L == 3:
            keep &= ~put[e - 1]                                     # those are counted under L == 2
        e, st0 = e[keep], st0[keep]
        if len(e):
            pred.append(rollout(st0, L)["vis"]); pred1.append(rollout(e, 1)["vis"]); real.append(vis_r[e + 1])
    if pred:
        pred, pred1, real = np.concatenate(pred), np.concatenate(pred1), np.concatenate(real)
        res["hidden_after_imagined_put_in"] = dict(
            n=int(len(real)), hidden_after=int((~real).sum()), bal=bal(~pred, ~real), acc=pct(pred == real),
            bal_last_touch_from_real=bal(~pred1, ~real), acc_last_touch_from_real=pct(pred1 == real),
        )
    else:
        res["hidden_after_imagined_put_in"] = dict(n=0, hidden_after=0)
    return res


def load(path):
    ck = torch.load(path, map_location=M.DEV, weights_only=False)
    cls = M1.SceneWNet171 if "card171" in ck["meta"] else M.SceneWNet
    net = cls(n_slots=ck["n_slots"], mov_si=ck["mov_si"]).to(M.DEV)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "gate":
        arm = sys.argv[2]
        chains, rng = ARMS[arm]
        Z, tc, ep = M.load_split(LEARN)
        tr = (ep % 100) < 80
        t0 = time.time()
        if arm == "base":
            net, meta = M.train_model(Z[tr], tc[tr], ep[tr], seed=0, steps=3500)
        else:
            net, meta = M1.train_model(Z[tr], tc[tr], ep[tr], seed=0, steps=3500, chains=chains, rng_head=rng)
        res = dict(arm=arm, fit_sec=round(time.time() - t0, 1), slot_kind=meta["slot_kind"])
        res.update(score(net, meta, Z[tr], tc[tr], ep[tr], Z[~tr], tc[~tr], ep[~tr], use_rng=rng))
        tag = "diag" if len(sys.argv) > 3 and sys.argv[3] == "diag" else "gate_a"   # diag: a refit with one more number reported
        json.dump(res, open(OUT / f"{tag}_{arm}.json", "w"), indent=1, default=float)
        brief(arm, res)
    elif cmd == "fit":
        import ev as EV168
        import common as C141
        seed = int(sys.argv[2])
        Z, tc, ep = M.load_split(LEARN)
        t0 = time.time()
        arm = sys.argv[3] if len(sys.argv) > 3 else "both"
        chains, rng = ARMS[arm]
        net, meta = M1.train_model(Z, tc, ep, seed=seed, steps=3500, chains=chains, rng_head=rng)
        fit_sec = round(time.time() - t0, 1)
        var_map = EV168.bookkeeping_map(net, meta, Z, C141.truth(LEARN))      # bookkeeping for truth-scored reports only
        torch.save(dict(seed=seed, fit_sec=fit_sec, n_slots=net.n_slots, mov_si=net.mov_si, state_dict=net.state_dict(),
                        meta=meta, var_map=var_map), OUT / (f"scene_wnet_{seed}.pt" if arm == "both" else f"scene_wnet_{arm}_{seed}.pt"))
        print(f"saved runs/171 {arm} seed {seed} in {fit_sec}s kinds={meta['slot_kind']} cause_of={meta['cause_of']} var_map={var_map}", flush=True)
    elif cmd == "test":
        import ev as EV168
        import common as C141
        Zr, tcr, epr = M.load_split(LEARN)
        Z, tc, ep = M.load_split(TEST)
        t_te = C141.truth(TEST)
        res = {}
        for sd in sys.argv[2:]:
            row = {}
            for name, path, use_rng in (("card171", OUT / f"scene_wnet_{sd}.pt", True), ("card168", M.ROOT / f"runs/168/scene_wnet_{sd}.pt", False)):
                net, ck = load(path)
                r = score(net, ck["meta"], Zr, tcr, epr, Z, tc, ep, use_rng=use_rng)
                rep = EV168.score_representation(net, ck["meta"], ck["var_map"], Z, t_te)
                con = EV168.score_consequences(net, ck["meta"], ck["var_map"], Z, tc, ep, t_te)
                r["card168_measures"] = dict(mean_non14_bal=rep["mean_non14_bal"], pull_bal=con["pull_bal"],
                                             drawer_touch_hides_cube_bal=con["drawer_touch_hides_cube_bal"])
                row[name] = r
                brief(f"{name} seed {sd}", r)
            res[sd] = row
        json.dump(res, open(OUT / "test.json", "w"), indent=1)
    elif cmd == "summarize":
        g = {a: json.load(open(OUT / f"gate_a_{a}.json")) for a in ARMS if (OUT / f"gate_a_{a}.json").exists()}
        for a, r in g.items():
            brief(a, r)
        if "base" in g and "both" in g:
            b, m = g["base"], g["both"]
            base3, ub = b["three_touches"]["whole_state"], m["three_touches"]["last_touch_from_real_reading"]
            gap = round(ub - base3, 1)
            bar3 = round(base3 + 0.5 * gap, 1)
            o, o170 = m["one_touch"]["model"], dict(press_flip=98.6, table_move=94.2, unchanged_kept=99.3)
            h = m["hidden_after_imagined_put_in"]
            s = dict(
                m1_one_touch={k: dict(model=o[k], card170=v, ok=bool(o[k] >= v - 2.0)) for k, v in o170.items()},
                m2_pull_end=dict(model=m["pull_end_in_range"]["model"], remembered=m["pull_end_in_range"]["remembered_range"],
                                 width_ratio=m["pull_end_in_range"]["width_ratio"],
                                 ok=bool(m["pull_end_in_range"]["model"] >= 85.0 and m["pull_end_in_range"]["width_ratio"] <= 1.25)),
                m3_three_touches=dict(model=m["three_touches"]["whole_state"], baseline=base3, upper_bound=ub, gap=gap, bar=bar3,
                                      judged=bool(gap >= 4.0), ok=bool(m["three_touches"]["whole_state"] >= bar3)),
                m4_hidden=dict(model=h.get("bal"), baseline=b["hidden_after_imagined_put_in"].get("bal"), n=h["n"], hidden=h["hidden_after"],
                               judged=bool(h["hidden_after"] >= 20), ok=bool((h.get("bal") or 0.0) >= 80.0)),
            )
            json.dump(dict(summary=s, arms=g), open(OUT / "gate_a.json", "w"), indent=1)
            print(json.dumps(s, indent=1))


def brief(name, r):
    o, p, t, h = r["one_touch"]["model"], r["pull_end_in_range"], r["three_touches"], r["hidden_after_imagined_put_in"]
    print(f"[{name}] one touch: press {o['press_flip']} pull(point) {o['pull_moves']} table {o['table_move']} unchanged {o['unchanged_kept']} "
          f"hidden(1) {o['hidden_after_drawer_touch_bal']} whole {o['whole_state']} | pull end in range {p['model']} (remembered {p['remembered_range']}, "
          f"width x{p['width_ratio']}) | 3 touches whole {t['whole_state']} (last from real {t['last_touch_from_real_reading']}, each from real {t.get('each_touch_from_real_reading')}, nothing changes {t['nothing_changes']}; "
          f"n={t['n']}) per slot {t['per_slot']} | hidden after imagined put-in: {h}", flush=True)


if __name__ == "__main__":
    main()
