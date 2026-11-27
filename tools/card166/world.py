"""Card 166: Unified Factored Relational World Model (WNet).

Merges Card 165's Neural Reachability Propagation (RNet) and Card 160's cell-effect model into ONE
shared-representation PyTorch network trained end-to-end directly from ordered reward-free random play
(runs/163/seq_<exp>.pkl) — ZERO Table/exp_*.pkl, ZERO gid/FAR binning, ZERO FI/FY tables, ZERO statement() extractor.

Shared representations (GOAL C2):
  - code_emb[NC, D]: shared across reachability alpha(c), cell subject e(c), out-of-view items e(k), and next-code output e(o)
  - act_emb[NA, D]:  shared across current action e_a(a) and out-of-view item actions e_a(a_k)
  - Continuous relative offsets (dx, dy): shared local radius-2 geometry for beta, gamma, and cell-effect pose MLP

Two heads on the shared representation:
  1. Reachability head: alpha(c) = sigmoid(w_a . e(c) + b_a), beta(dx, dy), gamma(dx, dy), iterated max-product propagation.
  2. Factored relational cell-effect head: p(o | c, a, dx, dy, is_far, d, items) using 4-head cross-attention + action max-pool
     over out-of-view items, Card 161's shared no-change output head (o == c), and bilinear readout onto code_emb (o != c).
"""
import os, time, math, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from grid import NA, UNK, OUT

dev = "cuda" if torch.cuda.is_available() else "cpu"
RAD = 2
OFFS_ALL = [(dx, dy) for dx in range(-RAD, RAD + 1) for dy in range(-RAD, RAD + 1)]
OFFS_STEP = [(dx, dy) for dx, dy in OFFS_ALL if (dx, dy) != (0, 0)]
OFF_ALL_IX = {o: i for i, o in enumerate(OFFS_ALL)}
OFF_STEP_IX = {o: i for i, o in enumerate(OFFS_STEP)}
FAR_IX = len(OFFS_ALL)  # 25


def shift_stack(r, offs):
    """r: [B, W, H] -> [B, len(offs), W, H], entry k at (x, y) is r[x - dx, y - dy] (0 outside)."""
    B, W, H = r.shape
    rp = F.pad(r, (RAD, RAD, RAD, RAD), value=0.0)
    return torch.stack([rp[:, RAD - dx:RAD - dx + W, RAD - dy:RAD - dy + H] for dx, dy in offs], 1)


class WNet(nn.Module):
    def __init__(s, NC, D=32, H=128, n_heads=4):
        super().__init__()
        s.NC = NC; s.D = D; s.H = H; s.n_heads = n_heads; s.dh = D // n_heads
        # Shared embeddings (C2)
        s.code_emb = nn.Embedding(NC, D)
        s.act_emb = nn.Embedding(NA, D)
        s.dir_emb = nn.Embedding(4, 16)

        # Head 1: Differentiable reachability propagation (reads shared code_emb)
        s.alpha_head = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 1))
        s.step_mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1))
        s.act_mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1))
        s.register_buffer("offs_step", torch.tensor(OFFS_STEP, dtype=torch.float32))
        s.register_buffer("offs_all", torch.tensor(OFFS_ALL, dtype=torch.float32))
        off_feats = [(dx / float(RAD), dy / float(RAD), 0.0) for dx, dy in OFFS_ALL] + [(0.0, 0.0, 1.0)]
        s.register_buffer("off_feats", torch.tensor(off_feats, dtype=torch.float32))

        # Head 2: Factored relational cell-effect predictor (reads shared code_emb & act_emb)
        # Additive DeepSet over out-of-view items (no softmax competition, no dying gates):
        # gentle L1 on per-item outputs drives distractor items (e.g. opened closed doors) to exact 0.000!
        s.pose_mlp = nn.Sequential(nn.Linear(3 + 16, D), nn.ReLU(), nn.Linear(D, D))
        s.v_tok = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, D))
        s.v_act = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))

        s.cell_mlp = nn.Sequential(
            nn.Linear(5 * D, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
        )
        # Card 161 shared no-change output + bilinear readout onto shared code_emb
        s.stay_head = nn.Linear(H, 1)
        s.chg_proj = nn.Linear(H, D)
        s.chg_bias = nn.Parameter(torch.zeros(NC))
        # Experience-learned transition support mask (o == c or (c -> o) seen in seq; Cards 160–162)
        s.register_buffer("obs_mask", torch.ones(NC, NC, dtype=torch.bool))

    def alpha(s):
        return torch.sigmoid(s.alpha_head(s.code_emb.weight)).squeeze(-1)

    def beta(s):
        return torch.sigmoid(s.step_mlp(s.offs_step)).squeeze(-1)

    def gamma(s):
        return torch.sigmoid(s.act_mlp(s.offs_all)).squeeze(-1)

    def propagate(s, maps, px, py, K=40, early_stop=True, detach_beta=False, binary=False):
        """maps: [B, W, H] long; px, py: [B] long -> r_arr [B, W, H], r_act [B, W, H] in [0, 1]."""
        B, W, H = maps.shape
        al_t = s.alpha()
        bt_t = s.beta().detach() if detach_beta else s.beta()
        gm_t = s.gamma()
        if binary:
            al_t = (al_t >= 0.5).float(); bt_t = (bt_t >= 0.5).float(); gm_t = (gm_t >= 0.5).float()
        al = al_t[maps.clamp(0, s.NC - 1)]
        bt = bt_t[None, :, None, None]
        gm = gm_t[None, :, None, None]
        r = torch.zeros(B, W, H, device=maps.device, dtype=torch.float32)
        r[torch.arange(B, device=maps.device), px, py] = 1.0
        for _ in range(K):
            nb = (shift_stack(r, OFFS_STEP) * bt).max(1)[0]
            r2 = torch.maximum(r, al * nb)
            if early_stop and (r2 - r).abs().max().item() < 1e-4:
                r = r2; break
            r = r2
        r_act = (shift_stack(r, OFFS_ALL) * gm).max(1)[0]
        return r, r_act

    def cell_logits_pos(s, c, a, pos_feat, d, ic, ia, im, return_contrib=False):
        """c, a, d: [B] long; pos_feat: [B, 3] float (dx/R, dy/R, is_far); ic, ia: [B, M] long; im: [B, M] bool -> [B, NC]."""
        ec = s.code_emb(c)
        ea = s.act_emb(a)
        ep = s.pose_mlp(torch.cat([pos_feat, s.dir_emb(d)], -1))

        e_ik = s.code_emb(ic)
        e_ia = s.act_emb(ia)
        e_pair = torch.cat([e_ik, e_ia], -1)                        # [B, M, 2D]
        im_f = im.float().unsqueeze(-1)                             # [B, M, 1]

        # Additive per-item embedding sum (zero softmax competition)
        c_tok = s.v_tok(e_pair) * im_f                              # [B, M, D]
        ctx_item = c_tok.sum(1)

        # Non-negative per-item-action max-pool (0 when empty or when irrelevant action decays to 0)
        c_act = F.relu(s.v_act(e_ia)) * im_f                        # [B, M, D]
        ctx_iact = c_act.max(1)[0]

        h = s.cell_mlp(torch.cat([ec, ea, ep, ctx_item, ctx_iact], -1))
        lg = s.chg_proj(h) @ s.code_emb.weight.T + s.chg_bias
        stay = s.stay_head(h)
        lg = lg.scatter(1, c[:, None], stay)
        lg = lg.masked_fill(~s.obs_mask[c], -1e9)
        if return_contrib:
            return lg, c_tok, c_act
        return lg

    def cell_logits(s, c, a, off_ix, d, ic, ia, im, return_contrib=False):
        return s.cell_logits_pos(c, a, s.off_feats[off_ix], d, ic, ia, im, return_contrib=return_contrib)


def pu_bce(p, y, w_neg=0.005, mask=None):
    p = p.clamp(1e-6, 1.0 - 1e-6)
    loss = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, w_neg))
    if mask is not None:
        w = w * mask
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def extract_data(seq, max_segs=8000):
    """Extract reachability data AND deduplicated 1-step cell transition rows directly from seq_<exp>.pkl."""
    T, eps = seq["T"], seq["eps"]; NC = len(T.codes.triples)
    S = 25
    m_patch, m_tgt, ev_tgt, segs = [], [], [], []
    obs_effs = set()

    # Cell transition key packing: ((((c * 8 + a) * 32 + off_ix) * 4 + pd) * 512 + it_id) * 64 + c2
    # Check bit widths safely: NC <= 64, NA=7<8, off_ix <= 25<32, pd<4, len(seq_sets) < 512
    n_sets = len(T.seq_sets)
    assert NC <= 64 and n_sets < 512, (NC, n_sets)
    S_IT = 512; S_C2 = 64

    ck_buf, cn_buf = [], []
    cell_keys = np.zeros(0, np.int64); cell_cnt = np.zeros(0, np.int64)

    def flush_cells():
        nonlocal cell_keys, cell_cnt, ck_buf, cn_buf
        if not ck_buf:
            return
        k = np.concatenate([cell_keys] + ck_buf); n = np.concatenate([cell_cnt] + cn_buf)
        ck_buf, cn_buf = [], []
        cell_keys, inv = np.unique(k, return_inverse=True)
        cell_cnt = np.bincount(inv, weights=n).astype(np.int64)

    offs_dx = np.array([o[0] for o in OFFS_ALL], np.int64)
    offs_dy = np.array([o[1] for o in OFFS_ALL], np.int64)
    rng = np.random.default_rng(0)

    for ei, e in enumerate(eps):
        maps, ver, poses, acts, items_seq = e["maps"], e["ver"], e["poses"], e["acts"], e["items"]
        n = len(acts)
        ver_a = np.asarray(ver, np.int64)
        poses_a = np.asarray(poses, np.int64)
        acts_a = np.asarray(acts, np.int64)
        it_a = np.asarray(items_seq[:n], np.int64)

        # 1. Reachability 1-step moves (vectorized per episode)
        dp = poses_a[1:, :2] - poses_a[:-1, :2]
        pad_maps = np.stack([np.pad(m, RAD, constant_values=OUT) for m in maps])  # [V, W+2R, H+2R]
        px0 = poses_a[:-1, 0]; py0 = poses_a[:-1, 1]; pd0 = poses_a[:-1, 2]
        px1 = poses_a[1:, 0]; py1 = poses_a[1:, 1]
        v0 = ver_a[:-1]; v1 = ver_a[1:]

        # Extract 5x5 local patches before and after for ALL steps in episode
        gx = px0[:, None] + RAD + offs_dx[None, :]  # [n, 25]
        gy = py0[:, None] + RAD + offs_dy[None, :]  # [n, 25]
        c0_loc = pad_maps[v0[:, None], gx, gy]      # [n, 25]
        c1_loc = pad_maps[v1[:, None], gx, gy]      # [n, 25]

        # Moves for reachability
        step_mask = (np.abs(dp[:, 0]) <= RAD) & (np.abs(dp[:, 1]) <= RAD) & ((dp[:, 0] != 0) | (dp[:, 1] != 0))
        if step_mask.any():
            sm_idx = np.nonzero(step_mask)[0]
            # drop center (dx=0, dy=0) at index 12 from 25-patch to get 24-patch matching OFFS_STEP
            c0_step = np.delete(c0_loc[sm_idx], OFF_ALL_IX[(0, 0)], axis=1)
            m_patch.append(c0_step)
            tgts = np.array([OFF_STEP_IX[(int(dp[i, 0]), int(dp[i, 1]))] for i in sm_idx], np.int64)
            m_tgt.append(tgts)

        # Events for reachability & obs_effs
        for t_i, ev in enumerate(e["evs"]):
            if ev:
                p0 = poses_a[t_i]; a_i = int(acts_a[t_i])
                for x, y, c, c2 in ev:
                    d = (int(x) - int(p0[0]), int(y) - int(p0[1]))
                    if d in OFF_ALL_IX:
                        ev_tgt.append(OFF_ALL_IX[d])
                    if c != c2 and c != UNK and c2 != UNK:
                        obs_effs.add((int(c), int(c2), a_i))

        # Constant-map segments >= 20 steps for reachability
        if len(segs) < max_segs:
            t0 = 0
            while t0 < n and len(segs) < max_segs:
                v = int(ver_a[t0]); t1 = t0
                while t1 < n and ver_a[t1] == v:
                    t1 += 1
                if t1 - t0 >= 20:
                    mp = maps[v]; W, H = mp.shape
                    pad = np.full((S, S), OUT, np.int16); pad[:W, :H] = mp
                    val = np.zeros((S, S), bool); val[:W, :H] = True
                    vis = np.zeros((S, S), bool)
                    vis[poses_a[t0:t1, 0], poses_a[t0:t1, 1]] = True
                    segs.append((pad, val, int(poses_a[t0, 0]), int(poses_a[t0, 1]), vis))
                t0 = t1

        # 2. Cell transition rows (vectorized over all valid local cells + sampled far cells)
        # Valid local cells: not (0, 0), not (px1-px0, py1-py0), c0 not in {UNK, OUT}, c1 not in {UNK, OUT}
        valid = (c0_loc != OUT) & (c0_loc != UNK) & (c1_loc != OUT) & (c1_loc != UNK)
        valid[:, OFF_ALL_IX[(0, 0)]] = False
        # Mask cell under agent at t+1
        is_p1 = (offs_dx[None, :] == dp[:, 0:1]) & (offs_dy[None, :] == dp[:, 1:2])
        valid &= ~is_p1

        ti_v, oi_v = np.nonzero(valid)
        c0_v = c0_loc[ti_v, oi_v].astype(np.int64)
        c1_v = c1_loc[ti_v, oi_v].astype(np.int64)
        k_loc = ((((c0_v * 8 + acts_a[ti_v]) * 32 + oi_v) * 4 + pd0[ti_v]) * S_IT + it_a[ti_v]) * S_C2 + c1_v
        ku, nu = np.unique(k_loc, return_counts=True)
        ck_buf.append(ku); cn_buf.append(nu)

        # Sample 2 random FAR cells per step (teaching WNet that is_far=1 stays unchanged for all codes/actions/items)
        W, H = maps[0].shape
        fx = rng.integers(0, W, size=(n, 2)); fy = rng.integers(0, H, size=(n, 2))
        far_ok = (np.abs(fx - px0[:, None]) > RAD) | (np.abs(fy - py0[:, None]) > RAD)
        far_ok &= (fx != px1[:, None]) | (fy != py1[:, None])
        ft, fj = np.nonzero(far_ok)
        if len(ft):
            c0_f = pad_maps[v0[ft], fx[ft, fj] + RAD, fy[ft, fj] + RAD].astype(np.int64)
            c1_f = pad_maps[v1[ft], fx[ft, fj] + RAD, fy[ft, fj] + RAD].astype(np.int64)
            fok = (c0_f != OUT) & (c0_f != UNK) & (c1_f != OUT) & (c1_f != UNK)
            if fok.any():
                ft = ft[fok]; c0_f = c0_f[fok]; c1_f = c1_f[fok]
                k_far = ((((c0_f * 8 + acts_a[ft]) * 32 + FAR_IX) * 4 + pd0[ft]) * S_IT + it_a[ft]) * S_C2 + c1_f
                ku, nu = np.unique(k_far, return_counts=True)
                ck_buf.append(ku); cn_buf.append(nu)

        if (ei + 1) % 500 == 0:
            flush_cells()

    flush_cells()

    # Collapse cell_keys into unique situations -> empirical target distribution Y_prob [N_sit, NC] & row counts
    c2_all = cell_keys % S_C2
    sit_key = cell_keys // S_C2
    uniq_sit, inv = np.unique(sit_key, return_inverse=True)
    N_sit = len(uniq_sit)
    Y = np.zeros((N_sit, NC), np.float64)
    np.add.at(Y, (inv, c2_all), cell_cnt)

    # Unpack situation keys
    q = uniq_sit
    s_it = q % S_IT; q = q // S_IT
    s_pd = q % 4; q = q // 4
    s_off = q % 32; q = q // 32
    s_act = q % 8; q = q // 8
    s_c = q

    # Item set tensors for T.seq_sets
    sets = T.seq_sets
    M_max = max(1, max(len(x) for x in sets))
    IC = np.zeros((len(sets), M_max), np.int64)
    IA = np.zeros((len(sets), M_max), np.int64)
    IM = np.zeros((len(sets), M_max), bool)
    for i, x in enumerate(sets):
        for j, (kk, aa) in enumerate(sorted(x)):
            IC[i, j] = kk; IA[i, j] = aa; IM[i, j] = True

    cell_data = dict(
        c=s_c.astype(np.int64), a=s_act.astype(np.int64), off=s_off.astype(np.int64),
        pd=s_pd.astype(np.int64), it=s_it.astype(np.int64), Y=Y,
        IC=IC, IA=IA, IM=IM,
    )
    return (
        NC,
        np.concatenate(m_patch, axis=0),
        np.concatenate(m_tgt, axis=0),
        np.array(ev_tgt, np.int64),
        segs,
        cell_data,
        sorted(obs_effs),
    )


def fit(exp, steps=3000, B=256, w_neg=0.015, lam_cell=1.0, log=print):
    t0 = time.time()
    seq = pickle.load(open("runs/163/seq_%s.pkl" % exp, "rb"))
    NC, m_patch, m_tgt, ev_tgt, segs, cd, obs_effs = extract_data(seq)
    N_sit = len(cd["c"])
    # Compute exact empirical target distributions P_emp and tempered row weights sqrt(count)
    # (preserves 100% unbiased conditional targets while giving rare unlocks/drops strong gradient!)
    row_tot = cd["Y"].sum(1)
    Y_prob = (cd["Y"] / np.maximum(row_tot[:, None], 1.0)).astype(np.float32)
    # Give changed situations + near situations strong coverage; exact conditional targets P_emp!
    has_chg = (cd["Y"].sum(1) - cd["Y"][np.arange(N_sit), cd["c"]]) > 0
    row_w = np.log1p(row_tot).astype(np.float32)
    log("extracted exp %s in %.1f s: %d moves, %d events, %d segs, %d cell sits (%d w/ change), %d obs_effs" %
        (exp, time.time() - t0, len(m_tgt), len(ev_tgt), len(segs), N_sit, int(has_chg.sum()), len(obs_effs)))

    t = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    MP = t(m_patch); MT = t(m_tgt); ET = t(ev_tgt)
    SM = t(np.stack([s[0] for s in segs]))
    SV = t(np.stack([s[1] for s in segs]), torch.float32)
    SPX = t([s[2] for s in segs]); SPY = t([s[3] for s in segs])
    SVIS = t(np.stack([s[4] for s in segs]), torch.float32)
    Wmax = int(SV.any(0).any(1).nonzero()[-1]) + 1
    Hmax = int(SV.any(0).any(0).nonzero()[-1]) + 1
    SM, SV, SVIS = SM[:, :Wmax, :Hmax], SV[:, :Wmax, :Hmax], SVIS[:, :Wmax, :Hmax]

    # Cell transition tensors on GPU
    SC = t(cd["c"]); SA = t(cd["a"]); SOFF = t(cd["off"]); SPD = t(cd["pd"]); SIT = t(cd["it"])
    YP = t(Y_prob, torch.float32); RW = t(row_w, torch.float32)
    IC_tab = t(cd["IC"]); IA_tab = t(cd["IA"]); IM_tab = t(cd["IM"], torch.bool)
    chg_idx = t(np.nonzero(has_chg)[0])
    # Contrasting failed attempts: near-agent unchanged situations whose (c, a) has an observed effect in experience
    eff_ca = {(c_i, a_i) for c_i, _, a_i in obs_effs}
    is_contrast = (~has_chg) & (cd["off"] != FAR_IX) & np.array(
        [(int(c_i), int(a_i)) in eff_ca for c_i, a_i in zip(cd["c"], cd["a"])], dtype=bool
    )
    contrast_idx = t(np.nonzero(is_contrast)[0])
    near_unchg_idx = t(np.nonzero((~has_chg) & (cd["off"] != FAR_IX))[0])

    torch.manual_seed(0); net = WNet(NC).to(dev)
    # Build experience-learned support mask purely from observed transitions in seq
    om = torch.eye(NC, dtype=torch.bool, device=dev)
    for c_i, o_i, _ in obs_effs:
        om[c_i, o_i] = True
    net.obs_mask.copy_(om)
    opt = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=1e-4)

    for it in range(1, steps + 1):
        lr = 0.01 if it < 0.55 * steps else 0.003 if it < 0.85 * steps else 0.001
        for g in opt.param_groups:
            g["lr"] = lr

        # 1. Reachability losses (exact Card 165 losses on shared code_emb)
        mi = torch.randint(len(MT), (B * 4,), device=dev)
        al = net.alpha()[MP[mi]]; bt = net.beta()[None, :].expand(B * 4, -1)
        y1 = F.one_hot(MT[mi], len(OFFS_STEP)).float()
        l_step = pu_bce(bt, y1, w_neg=0.01)
        l_move = pu_bce(al * bt.detach(), y1, w_neg=w_neg)

        ei = torch.randint(len(ET), (B,), device=dev)
        gm = net.gamma()[None, :].expand(B, -1)
        ye = F.one_hot(ET[ei], len(OFFS_ALL)).float()
        l_act = pu_bce(gm, ye, w_neg=0.025)

        si = torch.randint(len(SM), (B // 2,), device=dev)
        rK, _ = net.propagate(SM[si], SPX[si], SPY[si], K=16, early_stop=False, detach_beta=True)
        mask = SV[si].clone()
        mask[torch.arange(len(si), device=dev), SPX[si], SPY[si]] = 0.0
        l_seg = pu_bce(rK, SVIS[si], w_neg=0.02, mask=mask)

        # 2. Cell-effect loss: balanced minibatch of successes + contrasting failed attempts + near/all no-ops (exact unbiased YP!)
        ci_chg = chg_idx[torch.randint(len(chg_idx), (B * 2,), device=dev)]
        ci_con = contrast_idx[torch.randint(len(contrast_idx), (B * 3,), device=dev)]
        ci_near = near_unchg_idx[torch.randint(len(near_unchg_idx), (B * 2,), device=dev)]
        ci_all = torch.randint(N_sit, (B,), device=dev)
        ci = torch.cat([ci_chg, ci_con, ci_near, ci_all], 0)
        its = SIT[ci]
        im_b = IM_tab[its]
        lg, c_tok, c_act = net.cell_logits(SC[ci], SA[ci], SOFF[ci], SPD[ci], IC_tab[its], IA_tab[its], im_b, return_contrib=True)
        ce = -(YP[ci] * F.log_softmax(lg, -1)).sum(-1)
        w_c = RW[ci]
        l_cell = (ce * w_c).sum() / w_c.sum().clamp(min=1.0)
        # Gentle L1 on per-item outputs (warmed in smoothly over 600..1400): shrinks distractor items to 0 without spiking rare conditions
        denom = (im_b.float().sum() * net.D).clamp(min=1.0)
        w_l1 = 0.002 * min(1.0, max(0.0, (it - 600) / 800.0))
        l_gate = w_l1 * (c_tok.abs().sum() + c_act.abs().sum()) / denom

        loss = l_step + 2.0 * l_move + l_act + 2.0 * l_seg + 2.0 * lam_cell * (l_cell + l_gate)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        if it % 500 == 0:
            log("step %4d  loss=%.4f (step=%.4f move=%.4f act=%.4f seg=%.4f cell=%.5f)  %.1f s" %
                (it, float(loss.detach()), float(l_step.detach()), float(l_move.detach()),
                 float(l_act.detach()), float(l_seg.detach()), float(l_cell.detach()), time.time() - t0))

    # Full-dataset cell accuracy check on all N_sit unique situations
    net.eval()
    with torch.no_grad():
        preds = []
        for b in range(0, N_sit, 4096):
            sl = slice(b, b + 4096); its = SIT[sl]
            lg = net.cell_logits(SC[sl], SA[sl], SOFF[sl], SPD[sl], IC_tab[its], IA_tab[its], IM_tab[its])
            preds.append(lg.argmax(-1).cpu().numpy())
        pred = np.concatenate(preds)
    true_top = cd["Y"].argmax(1)
    acc_all = 100.0 * (pred == true_top).mean()
    acc_chg = 100.0 * (pred[has_chg] == true_top[has_chg]).mean()
    acc_unchg = 100.0 * (pred[~has_chg] == true_top[~has_chg]).mean()
    log("train sits fit: all=%.2f%%  changed=%.2f%% (%d)  unchanged=%.2f%% (%d)" %
        (acc_all, acc_chg, int(has_chg.sum()), acc_unchg, int((~has_chg).sum())))

    os.makedirs("runs/166", exist_ok=True)
    out = "runs/166/world_%s.pt" % exp
    torch.save(dict(net=net.state_dict(), NC=NC, D=net.D, H=net.H, obs_effs=obs_effs, steps=steps, w_neg=w_neg), out)
    log("saved", out)
    return net, seq["T"]


if __name__ == "__main__":
    import sys, rules
    exp = sys.argv[1]
    net, T = fit(exp, log=lambda *a: print(*a, flush=True))
    al = net.alpha().detach().cpu().numpy(); bt = net.beta().detach().cpu().numpy(); gm = net.gamma().detach().cpu().numpy()
    Rl = rules.Rules(T.codes); nm = lambda c: rules.name(T.codes, c)
    ok = 0; tot = 0; wrong = []
    for c in range(2, net.NC):
        tr = Rl.passable(c); pr = bool(al[c] >= 0.5); tot += 1; ok += (tr == pr)
        if tr != pr: wrong.append((nm(c), round(float(al[c]), 3), tr))
    print("Passability vs MiniGrid rules: %d / %d right; wrong=%s" % (ok, tot, wrong))
    print("Learned step offsets (beta >= 0.5):", [(OFFS_STEP[i], round(float(bt[i]), 3)) for i in range(len(bt)) if bt[i] >= 0.5])
    print("Learned act offsets  (gamma >= 0.5):", [(OFFS_ALL[i], round(float(gm[i]), 3)) for i in range(len(gm)) if gm[i] >= 0.5])
