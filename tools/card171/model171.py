"""Card 171: SceneWNet's transition head trained on chains of its own imagined states, with a pull's end as a range.

Card 168's model (tools/card168/model.py) with two additions, each behind a switch so the gate can tell them apart:

  chains  a training example is three consecutive touches of one episode. The first is predicted from the frame's
          reading, as in card 168. The state after it is built as card 170's search builds it (imagine_next below):
          predicted value, vector recalled from memory for that value. Touches two and three are predicted from that
          imagined state and compared with the reading of the real frame after each.
  rng     for a slide, two more outputs: the 10th and 90th percentile of the reading after the touch (pinball loss).

The encoder, the data, discover_slots and its targets are card 168's, unchanged.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, importlib.util
from pathlib import Path
if "card168_model" in sys.modules:      # card 169's planner loads card 168's model.py under this name
    M = sys.modules["card168_model"]
else:
    _spec = importlib.util.spec_from_file_location("card168_model", Path(__file__).resolve().parents[1] / "card168/model.py")
    M = importlib.util.module_from_spec(_spec); sys.modules["card168_model"] = M; _spec.loader.exec_module(M)

DEV = M.DEV
Q_LO, Q_HI = 0.10, 0.90
CHAIN = 3            # touches per chain
N_RECALL = 16        # remembered vectors averaged for an imagined value (as card 169's snap_slide_h)
CHAIN_FROM, CHAIN_FULL = 1000, 1500
BANK_EVERY = 250


class SceneWNet171(M.SceneWNet):
    def __init__(self, n_slots=5, mov_si=1, **kw):
        super().__init__(n_slots=n_slots, mov_si=mov_si, **kw)
        d_hid = self.chg_head.in_features
        self.q_head = nn.Linear(d_hid, 2)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -2.0)

    def forward_transition(self, h, vis_logit, aim, u_act):
        """Card 168's head, plus q_lo / q_hi [B, S]: the range the slot's value is predicted to end in."""
        B, S, D = h.shape
        val = self.slot_val(h)
        p_vis_mov = torch.sigmoid(vis_logit)
        p_vis = torch.ones(B, S, 1, device=h.device, dtype=h.dtype)
        p_vis[:, self.mov_si, 0] = p_vis_mov
        u_slot = torch.zeros(B, S, 3, device=h.device, dtype=h.dtype)
        u_slot[:, self.mov_si] = aim
        ids = torch.arange(S, device=h.device)
        se = self.slot_emb(ids)[None].expand(B, S, -1)

        ctx_in = torch.cat([se, h, val.unsqueeze(-1), p_vis, u_slot], dim=-1)
        m = self.ctx_tok(ctx_in)
        c_other = m.sum(dim=1, keepdim=True) - m

        ae = self.act_enc(u_act)[:, None].expand(B, S, -1)
        tr_in = torch.cat([se, h, val.unsqueeze(-1), p_vis, u_slot, ae, c_other], dim=-1)
        tr_h = self.trans_mlp(tr_in)

        chg_logit = self.chg_head(tr_h).squeeze(-1)
        p_chg = torch.sigmoid(chg_logit).unsqueeze(-1)
        dh = self.dh_head(tr_h)
        h_next = F.normalize(h + p_chg * dh, dim=-1)
        val_next = self.slot_val(h_next)

        q = F.softplus(self.q_head(tr_h))                                        # [B, S, 2] distances below / above
        q_lo = val_next.detach() - q[..., 0]
        q_hi = val_next.detach() + q[..., 1]

        mov_h = h_next[:, self.mov_si]
        mov_tr = tr_h[:, self.mov_si]
        vis_next_logit = self.next_vis_head(torch.cat([mov_h, mov_tr, val_next, aim, p_vis_mov[:, None]], dim=-1)).squeeze(-1)
        du = self.next_aim_head(torch.cat([mov_h, mov_tr, aim], dim=-1))
        aim_next = aim + p_chg[:, self.mov_si] * du
        return dict(
            chg_logit=chg_logit, p_chg=p_chg.squeeze(-1), h_next=h_next, val_next=val_next,
            vis_next_logit=vis_next_logit, aim_next=aim_next, ctx_m=m, q_lo=q_lo, q_hi=q_hi,
        )


def recall_h(bank_h, bank_val, k, v):
    """Mean of the N_RECALL remembered vectors of slot k whose value is nearest v [B] -> [B, D], unit length."""
    idx = torch.topk((bank_val[None, :, k] - v[:, None]).abs(), N_RECALL, dim=1, largest=False).indices   # [B, n]
    return F.normalize(bank_h[idx, k].mean(1), dim=-1)


@torch.no_grad()
def imagine_next(net, st, tr, bank_h, bank_val):
    """The state after an imagined touch, built as card 170's search builds it.

    st: dict(h [B,S,D], vis_logit [B], aim [B,3]); tr: forward_transition's output for the touch.
    A slot predicted changed takes the vector recalled for its predicted value; the cube takes its predicted
    in-view logit, and its predicted position when predicted in view. Everything else is kept exactly.
    """
    mov = net.mov_si
    chg = tr["p_chg"] >= 0.5
    h = st["h"].clone()
    for k in range(h.shape[1]):
        if k == mov:
            continue
        rows = torch.where(chg[:, k])[0]
        if len(rows):
            h[rows, k] = recall_h(bank_h, bank_val, k, tr["val_next"][rows, k])
    vis = torch.where(chg[:, mov], tr["vis_next_logit"], st["vis_logit"])
    aim = torch.where((chg[:, mov] & (tr["vis_next_logit"] >= 0.0))[:, None], tr["aim_next"], st["aim"])
    return dict(h=h, vis_logit=vis, aim=aim)


def pinball(d, tau):
    return torch.maximum(tau * d, (tau - 1.0) * d)


def train_model(Z_tr, tc_tr, ep_tr, seed=0, steps=3500, batch_size=128, lr=1.5e-3, chains=True, rng_head=True, log=print):
    """Card 168's train_model with the chain terms and the range terms added. Returns (net, meta)."""
    t0 = time.time()
    meta, tgt = M.discover_slots(Z_tr, tc_tr, ep_tr)
    n_slots = len(meta["slots"])
    mov_si = meta["mov_si"]
    u_act_tr = M.make_u_act(tc_tr, meta)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    net = SceneWNet171(n_slots=n_slots, mov_si=mov_si).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 1.5, total_steps=steps)

    Zt = torch.from_numpy(Z_tr).to(DEV)
    uat = torch.from_numpy(u_act_tr).to(DEV)
    val_yt = torch.from_numpy(tgt["val_y"]).to(DEV)
    val_mt = torch.from_numpy(tgt["val_m"]).to(DEV)
    aim_yt = torch.from_numpy(tgt["aim_y"]).to(DEV)
    aim_mt = torch.from_numpy(tgt["aim_m"]).to(DEV)
    vis_yt = torch.from_numpy(tgt["vis_y"]).to(DEV)
    vis_mt = torch.from_numpy(tgt["vis_m"]).to(DEV)
    chg_yt = torch.from_numpy(tgt["slot_chg"]).to(DEV)
    chg_mt = torch.from_numpy(tgt["slot_has_chg"]).to(DEV)

    same = tgt["same"]
    same_idx = np.where(same)[0]
    aim_idx = np.where(tgt["aim_m"])[0]
    aim_elev_idx = np.where(tgt["aim_m"] & (tgt["aim_y"][:, 2] > meta["z_cut"]))[0]
    vis_neg_idx = np.where(tgt["vis_m"] & (tgt["vis_y"] < 0.5))[0]
    vis_pos_idx = np.where(tgt["vis_m"] & (tgt["vis_y"] > 0.5))[0]

    pl_g = meta["group"][tgt["pl"][:-1]]
    slide_us = [meta["slots"][s] for s in meta["eff_sis"]]
    disp = np.linalg.norm(tc_tr[:-1, 3:6] - tc_tr[:-1, :3], axis=1)
    moved_any = tgt["slot_chg"][:, meta["eff_sis"]].max(1) > 0.5
    slide_pos_try = np.where(same & np.isin(pl_g, slide_us) & (disp > 0.10) & moved_any)[0]
    slide_neg_try = np.where(same & np.isin(pl_g, slide_us) & (disp > 0.10) & ~moved_any)[0]
    hide_trans = np.where(same & tgt["vis_m"][:-1] & (tgt["vis_y"][:-1] > 0.5) & tgt["vis_m"][1:] & (tgt["vis_y"][1:] < 0.5))[0]
    stay_open_trans = np.where(same & tgt["aim_m"][:-1] & (tgt["aim_y"][:-1, 2] > meta["z_cut"]) & tgt["vis_m"][1:] & (tgt["vis_y"][1:] > 0.5))[0]

    # chains: starts c with touches c, c+1, c+2 in one episode (frames c .. c+3)
    n_t = len(same)
    ok = np.zeros(n_t, bool)
    ok[:n_t - (CHAIN - 1)] = same[:n_t - (CHAIN - 1)]
    for j in range(1, CHAIN):
        ok[:n_t - (CHAIN - 1)] &= same[j:n_t - (CHAIN - 1) + j]
    chain_idx = np.where(ok)[0]

    def chain_pool(trans):
        """Chain starts whose second or third touch is one of `trans`."""
        c = np.concatenate([trans - j for j in range(1, CHAIN)])
        c = c[c >= 0]
        return c[ok[c]]
    pool_pos, pool_neg = chain_pool(slide_pos_try), chain_pool(slide_neg_try)
    pool_hide, pool_open = chain_pool(hide_trans), chain_pool(stay_open_trans)

    log(f"Slots={meta['slots']} kinds={meta['slot_kind']} cause_of={meta['cause_of']} cont={meta['cont_si']} | "
        f"vis(+/-)={len(vis_pos_idx)}/{len(vis_neg_idx)} pulls(+/-)={len(slide_pos_try)}/{len(slide_neg_try)} hide={len(hide_trans)} | "
        f"chains={len(chain_idx)} (pull+ {len(pool_pos)}, pull- {len(pool_neg)}, hide {len(pool_hide)}, open {len(pool_open)}) "
        f"chains={chains} rng={rng_head}")

    slide_sis = [k for k, kd in enumerate(meta["slot_kind"]) if kd == "slide"]
    lock_sis = [k for k, kd in enumerate(meta["slot_kind"]) if kd == "lock"]
    zero = torch.tensor(0.0, device=DEV)

    def step_losses(tr_out, b_from, enc_next, neg_all, temp):
        """Losses of one predicted touch (frame b_from -> b_from + 1), given the encoder's reading of the frame after."""
        b_to = b_from + 1
        n = len(b_from)
        l_val = zero
        for s_k in slide_sis:
            m1 = val_mt[b_to, s_k]
            if m1.any():
                l_val = l_val + F.smooth_l1_loss(tr_out["val_next"][m1, s_k], val_yt[b_to, s_k][m1], beta=0.05)
        for l_k in lock_sis:
            m1 = val_mt[b_to, l_k]
            if m1.any():
                l_val = l_val + F.binary_cross_entropy_with_logits(tr_out["val_next"][m1, l_k], val_yt[b_to, l_k][m1])
        cm = chg_mt[b_from]
        l_chg = F.binary_cross_entropy_with_logits(tr_out["chg_logit"][cm], chg_yt[b_from][cm])
        l_nce = zero
        for k in range(n_slots):
            keys = torch.cat([enc_next["h"][:, k].detach(), neg_all[:, k]], dim=0)
            l_nce = l_nce + F.cross_entropy(temp * (tr_out["h_next"][:, k] @ keys.T), torch.arange(n, device=DEV)) / n_slots
        vm1 = vis_mt[b_to]
        vis_tgt = torch.where(vm1, vis_yt[b_to], torch.sigmoid(enc_next["vis_logit"].detach()))
        l_vis = F.binary_cross_entropy_with_logits(tr_out["vis_next_logit"], vis_tgt)
        am1 = aim_mt[b_to] & (vis_tgt > 0.5)
        l_aim = F.smooth_l1_loss(tr_out["aim_next"][am1], aim_yt[b_to][am1], beta=0.04) if am1.any() else zero
        l_q = zero
        if rng_head:
            for s_k in slide_sis:
                mq = chg_mt[b_from, s_k] & (chg_yt[b_from, s_k] > 0.5)
                if mq.any():
                    y = enc_next["val"][mq, s_k].detach()
                    l_q = l_q + (pinball(y - tr_out["q_lo"][mq, s_k], Q_LO) + pinball(y - tr_out["q_hi"][mq, s_k], Q_HI)).mean()
        return l_val, l_chg, l_nce, l_vis, l_aim, l_q

    bank_h = bank_val = None
    for st in range(1, steps + 1):
        parts = [rng.choice(same_idx, 64, replace=False)]
        if len(slide_pos_try) > 0 and len(slide_neg_try) > 0:
            parts.append(rng.choice(slide_pos_try, 24, replace=True))
            parts.append(rng.choice(slide_neg_try, 24, replace=True))
        else:
            parts.append(rng.choice(same_idx, 48, replace=False))
        if len(hide_trans) > 0 and len(stay_open_trans) > 0:
            parts.append(rng.choice(hide_trans, 8, replace=True))
            parts.append(rng.choice(stay_open_trans, 8, replace=True))
        else:
            parts.append(rng.choice(same_idx, 16, replace=False))
        b0 = torch.as_tensor(np.concatenate(parts), dtype=torch.long, device=DEV)
        b1 = b0 + 1

        enc0 = net.encode(Zt[b0])
        enc1 = net.encode(Zt[b1])
        tr_out = net.forward_transition(enc0["h"], enc0["vis_logit"], enc0["aim"], uat[b0])

        # 1. reading the cube's position and whether it is in view (card 168)
        if len(aim_elev_idx) > 0:
            ba = torch.as_tensor(np.concatenate([rng.choice(aim_idx, 48, replace=False), rng.choice(aim_elev_idx, 24, replace=True)]), dtype=torch.long, device=DEV)
        else:
            ba = torch.as_tensor(rng.choice(aim_idx, 64, replace=False), dtype=torch.long, device=DEV)
        loss_aim = F.smooth_l1_loss(net.encode(Zt[ba])["aim"], aim_yt[ba], beta=0.04)
        if len(vis_neg_idx) > 0:
            bv = torch.as_tensor(np.concatenate([rng.choice(vis_pos_idx, 48, replace=False), rng.choice(vis_neg_idx, 32, replace=True)]), dtype=torch.long, device=DEV)
            loss_vis = F.binary_cross_entropy_with_logits(net.encode(Zt[bv])["vis_logit"], vis_yt[bv])
        else:
            loss_vis = zero

        # 2. reading each slot's value on both frames (card 168)
        loss_val = zero
        for s_k in slide_sis:
            for enc, b in ((enc0, b0), (enc1, b1)):
                m = val_mt[b, s_k]
                if m.any():
                    loss_val = loss_val + F.smooth_l1_loss(enc["val"][m, s_k], val_yt[b, s_k][m], beta=0.05)
        for l_k in lock_sis:
            for enc, b in ((enc0, b0), (enc1, b1)):
                m = val_mt[b, l_k]
                if m.any():
                    loss_val = loss_val + F.binary_cross_entropy_with_logits(enc["val"][m, l_k], val_yt[b, l_k][m])
        unchg = (chg_yt[b0] < 0.5) & chg_mt[b0]
        dh_true = (enc1["h"] - enc0["h"]).norm(dim=-1)
        loss_inv = (dh_true[unchg] ** 2).mean() if unchg.any() else zero

        # 3. the touch's predicted consequences, one step from a real reading (card 168, plus the range)
        temp = net.logit_scale.exp().clamp(1.0, 30.0)
        br = torch.randint(0, len(Zt), (batch_size,), device=DEV)
        with torch.no_grad():
            neg_all = net.encode(Zt[br])["h"]
        l_val1, l_chg1, l_nce1, l_vis1, l_aim1, l_q1 = step_losses(tr_out, b0, enc1, neg_all, temp)

        w_l1 = 0.0015 * min(1.0, max(0.0, (st - 500) / 1000.0))
        loss_l1 = w_l1 * tr_out["ctx_m"].abs().mean()
        loss = (
            2.5 * loss_aim + 1.5 * loss_vis + 1.5 * loss_val + 1.0 * l_val1
            + 2.0 * l_chg1 + 0.5 * l_nce1 + 3.0 * loss_inv
            + 2.0 * l_vis1 + 1.0 * l_aim1 + 1.0 * l_q1 + loss_l1
        )

        # 4. chains: touches two and three, each predicted from the state imagined after the touch before
        loss_chain = zero
        if chains and st > CHAIN_FROM:
            if bank_h is None or (st - CHAIN_FROM - 1) % BANK_EVERY == 0:
                net.eval()
                H_b, Val_b, _, _ = M.encode_all(net, Z_tr)
                net.train()
                bank_h, bank_val = torch.from_numpy(H_b).to(DEV), torch.from_numpy(Val_b).to(DEV)
            cp = [rng.choice(chain_idx, 48, replace=False)]
            for pool, n in ((pool_pos, 16), (pool_neg, 16), (pool_hide, 8), (pool_open, 8)):
                cp.append(rng.choice(pool if len(pool) else chain_idx, n, replace=True))
            c0 = torch.as_tensor(np.concatenate(cp), dtype=torch.long, device=DEV)
            with torch.no_grad():
                e0 = net.encode(Zt[c0])
                s_im = dict(h=e0["h"], vis_logit=e0["vis_logit"], aim=e0["aim"])
                s_im = imagine_next(net, s_im, net.forward_transition(s_im["h"], s_im["vis_logit"], s_im["aim"], uat[c0]), bank_h, bank_val)
            w_c = min(1.0, (st - CHAIN_FROM) / float(CHAIN_FULL - CHAIN_FROM))
            for j in range(1, CHAIN):
                bj = c0 + j
                with torch.no_grad():
                    enc_n = net.encode(Zt[bj + 1])
                tr_j = net.forward_transition(s_im["h"], s_im["vis_logit"], s_im["aim"], uat[bj])
                lv, lc, ln, lvi, la, lq = step_losses(tr_j, bj, enc_n, neg_all, temp)
                loss_chain = loss_chain + (1.0 * lv + 2.0 * lc + 0.5 * ln + 2.0 * lvi + 1.0 * la + 1.0 * lq) / (CHAIN - 1)
                if j < CHAIN - 1:
                    s_im = imagine_next(net, s_im, tr_j, bank_h, bank_val)
            loss = loss + w_c * loss_chain

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        sched.step()

        if st % 700 == 0 or st == steps:
            log(f"seed={seed} step={st}/{steps} loss={loss.item():.3f} (aim={loss_aim.item():.4f} vis={loss_vis.item():.3f} val={loss_val.item():.3f} "
                f"chg={l_chg1.item():.3f} nvis={l_vis1.item():.3f} q={float(l_q1):.3f} chain={float(loss_chain):.3f}) {time.time()-t0:.1f}s")

    # thresholds and prototypes: card 168's, unchanged
    net.eval()
    H_tr, Val_tr, V_tr, A_tr = M.encode_all(net, Z_tr)
    s_idx = np.where(same)[0]
    slot_cuts = {}
    for k in range(n_slots):
        kd = meta["slot_kind"][k]
        if kd == "mov":
            tab_m = tgt["aim_m"] & (tgt["aim_y"][:, 2] <= meta["z_cut"])
            err_m = np.linalg.norm((A_tr[tab_m] - tgt["aim_y"][tab_m]) * 0.1, axis=1)
            slot_cuts[k] = float(np.clip(2.0 * np.quantile(err_m, 0.90), 0.018, 0.026))
        elif kd == "slide":
            unchg_k = s_idx[tgt["slot_chg"][s_idx, k] < 0.5]
            d_u = np.abs(Val_tr[unchg_k + 1, k] - Val_tr[unchg_k, k])
            slot_cuts[k] = float(max(np.quantile(d_u, 0.94) * 1.35, 0.18))
        elif kd == "lock":
            slot_cuts[k] = 0.0
        else:
            unchg_k = s_idx[tgt["slot_chg"][s_idx, k] < 0.5]
            d_u = np.linalg.norm(H_tr[unchg_k + 1, k] - H_tr[unchg_k, k], axis=1)
            slot_cuts[k] = float(np.quantile(d_u, 0.92) * 1.3)

    cause_protos, cause_press_u = {}, {}
    pl_all = tgt["pl"]
    for eff_si, c_si in meta["cause_of"].items():
        m_c = tgt["val_m"][:, c_si]
        st_b = tgt["val_y"][:, c_si] > 0.5
        p0 = H_tr[m_c & (~st_b), c_si].mean(0); p0 = p0 / (np.linalg.norm(p0) + 1e-8)
        p1 = H_tr[m_c & st_b, c_si].mean(0); p1 = p1 / (np.linalg.norm(p1) + 1e-8)
        cause_protos[c_si] = np.stack([p0, p1], 0).astype(np.float32)
        c_touches = np.where(meta["group"][pl_all[:-1]] == meta["slots"][c_si])[0]
        cause_press_u[c_si] = np.median(u_act_tr[c_touches], axis=0).astype(np.float32)

    meta["slot_cuts"] = slot_cuts
    meta["cause_protos"] = cause_protos
    meta["cause_press_u"] = cause_press_u
    meta["card171"] = dict(chains=bool(chains), rng=bool(rng_head), q=[Q_LO, Q_HI])
    return net, meta
