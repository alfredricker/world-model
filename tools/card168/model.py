"""Card 168: SceneWNet — consequence-trained shared-slot scene world model for OGBench visual-scene.

Trains ONE shared-representation neural network purely from frozen DINOv2 PCA-64 patch features (runs/132/z_*.npy)
and pre-touch proprioceptive action descriptors — zero simulator truth labels, zero object names, and zero
reconstruction decoders (Card 145 rule).

Self-supervised consequence & persistence signals (extracted strictly on training episodes in discover_slots):
  1. Unsupervised slots: K-means (K=10) on contact_xyz merged by bidirectional contact<->release links (>=15%) -> active slots.
  2. Movable slot (mov_si): pre-touch contact_xyz + post-touch release_xyz propagated across unchanged stretches;
     calibrated presence (vis=1 when grasped/released in view, vis=0 when inside container after container closes).
  3. Effect/slide slots (eff_sis): 1-D controllable factor via SVD(Z_after - Z_before) on changed touches of that slot
     (ACF principle: isolates pure slide motion orthogonal to lock colour & arm), median-propagated across unoccluded
     frames of each unchanged segment so single-frame attention learns to read past arm occlusion.
  4. Cause/lock slots (cause_of): Fisher contrast between succeeded vs blocked slide attempts + 2-means state,
     majority-propagated across unoccluded frames between presses of the cause slot.
  5. Pre-touch action u_act [8]: (contact_xyz, contact_open, is_mov * target_release_xyz, is_mov) — zero post-touch
     outcome leak on fixture pulls, forcing the additive DeepSet-L1 transition head to read the lock slot!
"""
import math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import common as C141

ROOT = Path(__file__).resolve().parents[2]
CEN = np.array([0.425, 0.0, 0.0], dtype=np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(sets):
    """Load z [N, 1024, 64] fp32, touch [N, 9] fp32, episode [N] int64 for a list of set names."""
    zs, tcs, eps = [], [], []
    for si, nm in enumerate(sets):
        ev = np.load(ROOT / f"runs/132/ev_{nm}.npz")
        zs.append(np.load(ROOT / f"runs/132/z_{nm}.npy").astype(np.float32))
        tcs.append(ev["touch"].astype(np.float32))
        eps.append(si * 1000 + ev["episode"].astype(np.int64))
    return np.concatenate(zs, 0), np.concatenate(tcs, 0), np.concatenate(eps, 0)


def make_u_act(tc, meta):
    """Build 8-D pre-touch action vector with ZERO post-touch outcome leak on fixture/slide touches:
    [contact_xyz (3), contact_open (1), is_mov * target_release_xyz (3), is_mov (1)]."""
    pl = np.linalg.norm(tc[:, None, :3] - meta["cen"][None], axis=-1).argmin(1)
    is_mov = (meta["group"][pl] == meta["slots"][meta["mov_si"]]).astype(np.float32)[:, None]
    return np.concatenate([tc[:, :3], tc[:, 6:7], is_mov * tc[:, 3:6], is_mov], axis=1).astype(np.float32)


def discover_slots(Z, tc, ep):
    """100% unsupervised slot discovery and consequence/persistence target extraction on training episodes ONLY."""
    km = KMeans(10, n_init=10, random_state=0).fit(tc[:, :3])
    pl = km.labels_
    cen = km.cluster_centers_.astype(np.float32)
    same = ep[:-1] == ep[1:]
    disp = np.linalg.norm(tc[:, 3:6] - tc[:, :3], axis=1)
    mv = disp > 0.3
    ca = pl[mv]
    cb = np.linalg.norm(tc[mv, None, 3:6] - cen[None], axis=-1).argmin(1)
    T = np.zeros((10, 10), np.float64)
    np.add.at(T, (ca, cb), 1)
    S = T / np.maximum(T.sum(1, keepdims=True), 1)
    link = (S >= 0.15) & (S.T >= 0.15)
    g = np.arange(10)
    for _ in range(10):
        for a in range(10):
            for b in range(10):
                if link[a, b]:
                    g[a] = g[b] = min(g[a], g[b])

    min_touches = max(120, int(0.02 * len(tc)))
    slots = [int(u) for u in np.unique(g) if (g[pl] == u).sum() >= min_touches]
    n_slots = len(slots)
    mov_u = max(slots, key=lambda u: int((g == u).sum()))
    mov_si = slots.index(mov_u)

    D = np.zeros((len(tc) - 1, 1024), np.float32)
    D[same] = ((Z[1:][same] - Z[:-1][same]) ** 2).mean(-1)
    base = D[same].mean(0) + 1e-3

    slot_chg = np.zeros((len(tc) - 1, n_slots), np.float32)
    slot_has_chg = np.zeros((len(tc) - 1, n_slots), bool)
    val_y = np.zeros((len(tc), n_slots), np.float32)
    val_m = np.zeros((len(tc), n_slots), bool)
    slot_kind = ["other"] * n_slots
    slot_kind[mov_si] = "mov"

    # 1. Movable slot change
    m_mov = (g[pl[:-1]] == mov_u) & same
    slot_chg[same, mov_si] = (m_mov[same] & (disp[:-1][same] > 0.25)).astype(np.float32)
    slot_has_chg[same, mov_si] = True

    # 2. Effect / slide slots (ACF: 1-D SVD of Z_after - Z_before on changed touches + unoccluded segment median)
    eff_sis = []
    cause_of = {}
    for si, u in enumerate(slots):
        if si == mov_si:
            continue
        m = (g[pl[:-1]] == u) & same
        rat = D[m].mean(0) / base
        top48 = np.sort(np.argsort(rat)[-48:])
        if rat[top48].mean() < 3.8:
            continue
        eff_sis.append(si)
        slot_kind[si] = "slide"
        sc = np.log(D[:, top48].mean(1) + 1e-5)
        cut_chg = C141.otsu(sc[m])
        m_chg = m & (sc > cut_chg)

        dX = (Z[1:][m_chg][:, top48] - Z[:-1][m_chg][:, top48]).reshape(int(m_chg.sum()), -1)
        _, _, Vt = np.linalg.svd(dX, full_matrices=False)
        v1 = Vt[0]
        unocc = g[pl] != u
        mu = Z[unocc][:, top48].reshape(int(unocc.sum()), -1).mean(0)
        s_raw = (Z[:, top48].reshape(len(Z), -1) - mu) @ v1
        s_raw = (s_raw - s_raw[unocc].mean()) / (s_raw[unocc].std() + 1e-6)

        prop = s_raw.copy()
        pmsk = np.zeros(len(Z), bool)
        for e_id in np.unique(ep):
            idx = np.where(ep == e_id)[0]
            s0 = 0
            for k_i in range(len(idx)):
                if k_i == len(idx) - 1 or g[pl[idx[k_i]]] == u:
                    seg = idx[s0:k_i + 1]
                    clean = seg[g[pl[seg]] != u]
                    if len(clean) > 0:
                        prop[seg] = float(np.median(s_raw[clean]))
                        pmsk[seg] = True
                    s0 = k_i + 1
        val_y[:, si] = prop
        val_m[:, si] = pmsk
        # Clean slide change label: touch at u that changes unoccluded segment coordinate (> 0.14 std) or visual Otsu
        both_m = same & pmsk[:-1] & pmsk[1:]
        slot_chg[same, si] = ((g[pl[:-1]][same] == u) & (
            np.where(both_m[same], np.abs(prop[1:][same] - prop[:-1][same]) > 0.14, sc[same] > cut_chg)
        )).astype(np.float32)
        slot_has_chg[same, si] = True

        # 3. Fisher contrast on unoccluded slide attempts -> cause/lock slot + clean segment-propagated binary state!
        att = m & (disp[:-1] > 0.10)
        ya = slot_chg[:-1][att[:len(slot_chg)]] > 0.5 if False else (slot_chg[att, si] > 0.5)
        if min(int(ya.sum()), int((~ya).sum())) >= 20:
            X0, X1 = Z[:-1][att][~ya], Z[:-1][att][ya]
            f = ((X1.mean(0) - X0.mean(0)) ** 2).sum(-1) / (X1.var(0).sum(-1) + X0.var(0).sum(-1) + 1e-3)
            cond_p = np.argsort(f)[-12:]
            km2 = KMeans(2, n_init=10, random_state=0).fit(Z[:-1][m][:, cond_p].reshape(int(m.sum()), -1))
            st_raw = km2.predict(Z[:, cond_p].reshape(len(Z), -1))
            # Orient st_raw so 1 = enabled (higher slide change rate!)
            r0 = float(ya[st_raw[:-1][att] == 0].mean()) if (st_raw[:-1][att] == 0).any() else 0.0
            r1 = float(ya[st_raw[:-1][att] == 1].mean()) if (st_raw[:-1][att] == 1).any() else 0.0
            if r0 > r1:
                st_raw = 1 - st_raw
            chg_st = st_raw[1:][same] != st_raw[:-1][same]
            cands = [c for c in range(n_slots) if c not in (si, mov_si)]
            best_c = max(cands, key=lambda c: float(chg_st[g[pl[:-1]][same] == slots[c]].mean()) if (g[pl[:-1]][same] == slots[c]).any() else -1.0)
            best_rate = float(chg_st[g[pl[:-1]][same] == slots[best_c]].mean())
            if best_rate >= 0.45:
                cause_of[si] = best_c
                slot_kind[best_c] = "lock"
                c_u = slots[best_c]
                lst = np.zeros(len(Z), np.float32)
                lmsk = np.zeros(len(Z), bool)
                for e_id in np.unique(ep):
                    idx = np.where(ep == e_id)[0]
                    s0 = 0
                    for k_i in range(len(idx)):
                        if k_i == len(idx) - 1 or g[pl[idx[k_i]]] == c_u:
                            seg = idx[s0:k_i + 1]
                            clean = seg[g[pl[seg]] != c_u]
                            if len(clean) > 0:
                                lst[seg] = float(np.mean(st_raw[clean]) >= 0.5)
                                lmsk[seg] = True
                            s0 = k_i + 1
                val_y[:, best_c] = lst
                val_m[:, best_c] = lmsk
                both_l = same & lmsk[:-1] & lmsk[1:]
                slot_chg[same, best_c] = (both_l[same] & (np.abs(lst[1:][same] - lst[:-1][same]) > 0.5)).astype(np.float32)
                slot_has_chg[same & both_l, best_c] = True

    # 4. Movable aim + container occlusion tracking using the discovered container's 1-D slide coordinate!
    zc = tc[g[pl] == mov_u, 2]
    z_med = float(np.median(zc))
    z_mad = float(np.median(np.abs(zc - z_med)))
    z_cut = z_med + 6.0 * max(z_mad, 0.02)

    put_m = (g[pl[:-1]] == mov_u) & same & (disp[:-1] > 0.25) & (tc[:-1, 5] > z_cut)
    put_xy = np.median(tc[:-1][put_m, 3:5], 0) if put_m.any() else np.zeros(2, np.float32)
    cont_si = min(eff_sis, key=lambda si: np.linalg.norm(cen[g == slots[si], :2].mean(0) - put_xy)) if eff_sis else None

    # Open vs closed values of cont_si along its 1-D coordinate val_y[:, cont_si]:
    # Post-put frames (where cube was just dropped into open container) give exact s_open;
    # the opposite tail (5th/95th percentile on val_m[:, cont_si]) gives exact s_closed!
    is_cont_shut = np.zeros(len(Z), bool)
    s_open, s_closed = 0.0, 0.0
    if cont_si is not None and put_m.any():
        s_vals = val_y[:, cont_si]
        s_open = float(np.median(s_vals[np.where(put_m)[0] + 1]))
        q_lo, q_hi = float(np.quantile(s_vals[val_m[:, cont_si]], 0.05)), float(np.quantile(s_vals[val_m[:, cont_si]], 0.95))
        s_closed = q_lo if abs(s_open - q_lo) > abs(s_open - q_hi) else q_hi
        span = max(abs(s_open - s_closed), 1e-3)
        # Shut (< ~3 cm out of 16 cm = within 20% of s_closed!)
        is_cont_shut = np.abs(s_vals - s_closed) < 0.22 * span

    aim_y = np.zeros((len(Z), 3), np.float32)
    aim_m = np.zeros(len(Z), bool)
    vis_y = np.zeros(len(Z), np.float32)
    vis_m = np.zeros(len(Z), bool)

    pre_mov = (g[pl] == mov_u) & (disp > 0.25)
    aim_y[pre_mov] = tc[pre_mov, :3]
    aim_m[pre_mov] = True
    vis_y[pre_mov] = 1.0
    vis_m[pre_mov] = True

    for e_id in np.unique(ep):
        idx = np.where(ep == e_id)[0]
        inside_cont = False
        cur_pos = None
        for k_i, idx_i in enumerate(idx):
            # Before touch idx_i:
            if pre_mov[idx_i]:
                inside_cont = bool(tc[idx_i, 2] > z_cut)
                cur_pos = tc[idx_i, :3]
            elif inside_cont:
                if is_cont_shut[idx_i]:
                    vis_y[idx_i] = 0.0; vis_m[idx_i] = True
                elif cont_si is not None and abs(val_y[idx_i, cont_si] - s_open) < 0.35 * abs(s_open - s_closed):
                    vis_y[idx_i] = 1.0; vis_m[idx_i] = True
                    if cur_pos is not None and not aim_m[idx_i]:
                        aim_y[idx_i] = cur_pos; aim_m[idx_i] = True
            elif cur_pos is not None:
                # Propagate stationary table position across non-movable touches!
                if not aim_m[idx_i]:
                    aim_y[idx_i] = cur_pos; aim_m[idx_i] = True
                vis_y[idx_i] = 1.0; vis_m[idx_i] = True

            # Across touch idx_i -> idx_{k_i+1}:
            if pre_mov[idx_i]:
                inside_cont = bool(tc[idx_i, 5] > z_cut)
                cur_pos = tc[idx_i, 3:6]
                if k_i + 1 < len(idx):
                    nxt_i = idx[k_i + 1]
                    if not aim_m[nxt_i]:
                        aim_y[nxt_i] = cur_pos; aim_m[nxt_i] = True
                    vis_y[nxt_i] = 1.0; vis_m[nxt_i] = True
            elif inside_cont and cont_si is not None and k_i < len(idx) - 1 and slot_chg[idx_i, cont_si] > 0.5:
                # Container moved while movable object was inside -> invalidate static cur_pos & mark movable changed!
                cur_pos = None
                slot_chg[idx_i, mov_si] = 1.0

    meta = dict(
        cen=cen, group=g, slots=slots, slot_kind=slot_kind, mov_si=mov_si,
        eff_sis=eff_sis, cause_of=cause_of, cont_si=cont_si, z_cut=z_cut,
        s_open=s_open, s_closed=s_closed,
    )
    targets = dict(
        pl=pl, same=same, slot_chg=slot_chg, slot_has_chg=slot_has_chg,
        val_y=val_y, val_m=val_m, aim_y=aim_y, aim_m=aim_m, vis_y=vis_y, vis_m=vis_m,
    )
    return meta, targets


class SceneWNet(nn.Module):
    """Unified shared-slot encoder + additive DeepSet-L1 relational touch transition model (GOAL C2)."""
    def __init__(self, n_slots=5, mov_si=1, d_in=64, d_hid=128, d_slot=64):
        super().__init__()
        self.n_slots = n_slots
        self.mov_si = mov_si
        self.d_slot = d_slot

        g = torch.linspace(-1.0, 1.0, 32)
        grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(1024, 2)
        self.register_buffer("grid", grid)

        self.patch_trunk = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid),
            nn.GELU(),
        )
        self.pos_proj = nn.Linear(2, d_hid, bias=False)
        self.trunk2 = nn.Sequential(
            nn.LayerNorm(d_hid),
            nn.Linear(d_hid, d_hid),
            nn.GELU(),
        )

        # Per-slot spatial attention & shared slot latent h[k]
        self.slot_attn = nn.Linear(d_hid, n_slots)
        self.slot_proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_hid + 2), nn.Linear(d_hid + 2, d_hid), nn.GELU(), nn.Linear(d_hid, d_slot))
            for _ in range(n_slots)
        ])
        # Shared linear readout of each slot's controllable scalar state from h[k] (works identically on h and h_next!)
        self.val_w = nn.Parameter(torch.randn(n_slots, d_slot) * 0.05)
        self.val_b = nn.Parameter(torch.zeros(n_slots))

        # Movable presence logit & 3D spatial aim
        self.vis_head = nn.Sequential(nn.Linear(d_slot + 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.where_map = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        self.patch_shift = nn.Linear(d_hid, 3)
        nn.init.zeros_(self.patch_shift.weight)
        nn.init.zeros_(self.patch_shift.bias)

        # Factored Relational Touch-Consequence Head (8-D pre-touch u_act + additive DeepSet-L1 over context slots)
        self.slot_emb = nn.Embedding(n_slots, 32)
        self.act_enc = nn.Sequential(nn.Linear(8, d_hid), nn.GELU(), nn.Linear(d_hid, d_slot))
        self.ctx_tok = nn.Sequential(
            nn.Linear(32 + d_slot + 5, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_slot),
        )
        self.trans_mlp = nn.Sequential(
            nn.Linear(32 + d_slot + 5 + d_slot + d_slot, d_hid * 2),
            nn.GELU(),
            nn.LayerNorm(d_hid * 2),
            nn.Linear(d_hid * 2, d_hid),
            nn.GELU(),
        )
        self.chg_head = nn.Linear(d_hid, 1)
        self.dh_head = nn.Linear(d_hid, d_slot)
        self.next_vis_head = nn.Sequential(nn.Linear(d_slot + d_hid + n_slots + 4, 64), nn.GELU(), nn.Linear(64, 1))
        self.next_aim_head = nn.Linear(d_slot + d_hid + 3, 3)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def slot_val(self, h):
        """Read scalar controllable state [B, S] directly from slot embeddings h [B, S, D]."""
        return (h * self.val_w[None]).sum(-1) + self.val_b[None]

    def encode(self, z):
        """Single-frame encoder: z [B, 1024, 64] -> dict(h, val, vis_logit, aim, max_sc)."""
        p = self.trunk2(self.patch_trunk(z) + self.pos_proj(self.grid)[None])   # [B, 1024, H]
        sc = self.slot_attn(p)                                                   # [B, 1024, S]
        w = torch.softmax(sc, dim=1)                                             # [B, 1024, S]
        max_sc = sc.max(dim=1).values                                            # [B, S]
        mean_sc = sc.mean(dim=1)                                                 # [B, S]
        pooled = torch.einsum("bps,bph->bsh", w, p)                              # [B, S, H]
        ctr = torch.einsum("bps,pd->bsd", w, self.grid)                          # [B, S, 2]

        hs = [F.normalize(self.slot_proj[k](torch.cat([pooled[:, k], ctr[:, k]], dim=-1)), dim=-1)
              for k in range(self.n_slots)]
        h = torch.stack(hs, dim=1)                                               # [B, S, D]
        val = self.slot_val(h)                                                   # [B, S]

        wm = w[:, :, self.mov_si]
        pts = self.where_map(self.grid)[None] + 0.45 * torch.tanh(self.patch_shift(p))
        aim = torch.einsum("bp,bpd->bd", wm, pts)                                # [B, 3]

        vis_in = torch.cat([h[:, self.mov_si], (max_sc[:, self.mov_si] - mean_sc[:, self.mov_si])[:, None], max_sc[:, self.mov_si:self.mov_si + 1] / 10.0], dim=-1)
        vis_logit = self.vis_head(vis_in).squeeze(-1)                            # [B]
        return dict(h=h, val=val, vis_logit=vis_logit, aim=aim, max_sc=max_sc)

    def forward_transition(self, h, vis_logit, aim, u_act):
        """Predict post-touch changes & next slot states from current (h, vis_logit, aim) and pre-touch u_act [B, 8]."""
        B, S, D = h.shape
        val = self.slot_val(h)                                                   # [B, S]
        p_vis_mov = torch.sigmoid(vis_logit)                                     # [B]
        p_vis = torch.ones(B, S, 1, device=h.device, dtype=h.dtype)
        p_vis[:, self.mov_si, 0] = p_vis_mov
        u_slot = torch.zeros(B, S, 3, device=h.device, dtype=h.dtype)
        u_slot[:, self.mov_si] = aim
        ids = torch.arange(S, device=h.device)
        se = self.slot_emb(ids)[None].expand(B, S, -1)

        # Additive DeepSet-L1 relational context over OTHER slots j != k
        ctx_in = torch.cat([se, h, val.unsqueeze(-1), p_vis, u_slot], dim=-1)    # [B, S, 32+D+5]
        m = self.ctx_tok(ctx_in)                                                 # [B, S, D]
        c_other = m.sum(dim=1, keepdim=True) - m                                 # [B, S, D]

        ae = self.act_enc(u_act)[:, None].expand(B, S, -1)                       # [B, S, D]
        tr_in = torch.cat([se, h, val.unsqueeze(-1), p_vis, u_slot, ae, c_other], dim=-1)
        tr_h = self.trans_mlp(tr_in)                                             # [B, S, H]

        chg_logit = self.chg_head(tr_h).squeeze(-1)                              # [B, S]
        p_chg = torch.sigmoid(chg_logit).unsqueeze(-1)                           # [B, S, 1]
        dh = self.dh_head(tr_h)                                                  # [B, S, D]
        h_next = F.normalize(h + p_chg * dh, dim=-1)                             # [B, S, D]
        val_next = self.slot_val(h_next)                                         # [B, S]

        mov_h = h_next[:, self.mov_si]
        mov_tr = tr_h[:, self.mov_si]
        vis_next_logit = self.next_vis_head(torch.cat([mov_h, mov_tr, val_next, aim, p_vis_mov[:, None]], dim=-1)).squeeze(-1)
        du = self.next_aim_head(torch.cat([mov_h, mov_tr, aim], dim=-1))
        aim_next = aim + p_chg[:, self.mov_si] * du
        return dict(
            chg_logit=chg_logit, p_chg=p_chg.squeeze(-1), h_next=h_next,
            val_next=val_next, vis_next_logit=vis_next_logit, aim_next=aim_next, ctx_m=m,
        )


def train_model(Z_tr, tc_tr, ep_tr, seed=0, steps=3500, batch_size=128, lr=1.5e-3, log=print):
    """Fit SceneWNet strictly on training split; calibrate thresholds strictly on training split."""
    t0 = time.time()
    meta, tgt = discover_slots(Z_tr, tc_tr, ep_tr)
    n_slots = len(meta["slots"])
    mov_si = meta["mov_si"]
    u_act_tr = make_u_act(tc_tr, meta)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    net = SceneWNet(n_slots=n_slots, mov_si=mov_si).to(DEV)
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

    same_idx = np.where(tgt["same"])[0]
    aim_idx = np.where(tgt["aim_m"])[0]
    aim_elev_idx = np.where(tgt["aim_m"] & (tgt["aim_y"][:, 2] > meta["z_cut"]))[0]
    vis_neg_idx = np.where(tgt["vis_m"] & (tgt["vis_y"] < 0.5))[0]
    vis_pos_idx = np.where(tgt["vis_m"] & (tgt["vis_y"] > 0.5))[0]

    # Contrast-balanced pools for slide pulls (unlocked vs locked!) and container-hides-cube transitions!
    pl_g = meta["group"][tgt["pl"][:-1]]
    slide_us = [meta["slots"][s] for s in meta["eff_sis"]]
    disp = np.linalg.norm(tc_tr[:-1, 3:6] - tc_tr[:-1, :3], axis=1)
    slide_pos_try = np.where(tgt["same"] & np.isin(pl_g, slide_us) & (disp > 0.10) & (tgt["slot_chg"][:, meta["eff_sis"]].max(1) > 0.5))[0]
    slide_neg_try = np.where(tgt["same"] & np.isin(pl_g, slide_us) & (disp > 0.10) & (tgt["slot_chg"][:, meta["eff_sis"]].max(1) < 0.5))[0]
    hide_trans = np.where(tgt["same"] & tgt["vis_m"][:-1] & (tgt["vis_y"][:-1] > 0.5) & tgt["vis_m"][1:] & (tgt["vis_y"][1:] < 0.5))[0]
    stay_open_trans = np.where(tgt["same"] & tgt["aim_m"][:-1] & (tgt["aim_y"][:-1, 2] > meta["z_cut"]) & tgt["vis_m"][1:] & (tgt["vis_y"][1:] > 0.5))[0]

    log(f"Slots={meta['slots']} kinds={meta['slot_kind']} cause_of={meta['cause_of']} cont={meta['cont_si']} | vis(+/-)={len(vis_pos_idx)}/{len(vis_neg_idx)} pulls(+/-)={len(slide_pos_try)}/{len(slide_neg_try)} hide={len(hide_trans)}")

    slide_sis = [k for k, kd in enumerate(meta["slot_kind"]) if kd == "slide"]
    lock_sis = [k for k, kd in enumerate(meta["slot_kind"]) if kd == "lock"]

    for st in range(1, steps + 1):
        # Contrast-balanced transition batch: 64 general + 24 unlocked pulls + 24 locked pulls + 16 hide/open-drawer transitions
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
        b_tr = np.concatenate(parts)
        b0 = torch.as_tensor(b_tr, dtype=torch.long, device=DEV)
        b1 = b0 + 1

        enc0 = net.encode(Zt[b0])
        enc1 = net.encode(Zt[b1])

        # IMPORTANT: Prevent slide_si's own embedding from leaking lock colour into its transition input so that
        # slide_si's precondition is read cleanly through the relational DeepSet context C_other (cause_si)!
        # (We detach nothing; both encode and forward_transition share the exact same h!)
        tr_out = net.forward_transition(enc0["h"], enc0["vis_logit"], enc0["aim"], uat[b0])

        # 1. Perceptual aim (table + oversampled open-drawer releases) & calibrated presence loss
        if len(aim_elev_idx) > 0:
            ba = torch.as_tensor(np.concatenate([
                rng.choice(aim_idx, 48, replace=False),
                rng.choice(aim_elev_idx, 24, replace=True),
            ]), dtype=torch.long, device=DEV)
        else:
            ba = torch.as_tensor(rng.choice(aim_idx, 64, replace=False), dtype=torch.long, device=DEV)
        enc_a = net.encode(Zt[ba])
        loss_aim = F.smooth_l1_loss(enc_a["aim"], aim_yt[ba], beta=0.04)

        if len(vis_neg_idx) > 0:
            bv = torch.as_tensor(np.concatenate([
                rng.choice(vis_pos_idx, 48, replace=False),
                rng.choice(vis_neg_idx, 32, replace=True),
            ]), dtype=torch.long, device=DEV)
            enc_v = net.encode(Zt[bv])
            loss_vis = F.binary_cross_entropy_with_logits(enc_v["vis_logit"], vis_yt[bv])
        else:
            loss_vis = torch.tensor(0.0, device=DEV)

        # 2. Controllable-factor readout loss on enc0, enc1, AND predicted h_next!
        loss_val = torch.tensor(0.0, device=DEV)
        loss_next_val = torch.tensor(0.0, device=DEV)
        for s_k in slide_sis:
            m0 = val_mt[b0, s_k]; m1 = val_mt[b1, s_k]
            if m0.any():
                loss_val = loss_val + F.smooth_l1_loss(enc0["val"][m0, s_k], val_yt[b0, s_k][m0], beta=0.05)
            if m1.any():
                loss_val = loss_val + F.smooth_l1_loss(enc1["val"][m1, s_k], val_yt[b1, s_k][m1], beta=0.05)
                loss_next_val = loss_next_val + F.smooth_l1_loss(tr_out["val_next"][m1, s_k], val_yt[b1, s_k][m1], beta=0.05)
        for l_k in lock_sis:
            m0 = val_mt[b0, l_k]; m1 = val_mt[b1, l_k]
            if m0.any():
                loss_val = loss_val + F.binary_cross_entropy_with_logits(enc0["val"][m0, l_k], val_yt[b0, l_k][m0])
            if m1.any():
                loss_val = loss_val + F.binary_cross_entropy_with_logits(enc1["val"][m1, l_k], val_yt[b1, l_k][m1])
                loss_next_val = loss_next_val + F.binary_cross_entropy_with_logits(tr_out["val_next"][m1, l_k], val_yt[b1, l_k][m1])

        # Also adversarial/decorrelation regularizer so a slide slot's embedding h[:, slide_si] stays pure slide state
        # (invariant to lock flips when the slide itself did NOT move -> forces lock precondition through cause_si in C_other!)
        unchg = (chg_yt[b0] < 0.5) & chg_mt[b0]
        dh_true = (enc1["h"] - enc0["h"]).norm(dim=-1)
        loss_inv = (dh_true[unchg] ** 2).mean() if unchg.any() else torch.tensor(0.0, device=DEV)

        # 3. Slot change BCE + InfoNCE post-touch prediction
        cm = chg_mt[b0]
        loss_chg = F.binary_cross_entropy_with_logits(tr_out["chg_logit"][cm], chg_yt[b0][cm])

        temp = net.logit_scale.exp().clamp(1.0, 30.0)
        loss_nce = torch.tensor(0.0, device=DEV)
        br = torch.randint(0, len(Zt), (batch_size,), device=DEV)
        with torch.no_grad():
            neg_all = net.encode(Zt[br])["h"]
        for k in range(n_slots):
            q = tr_out["h_next"][:, k]
            tgt_k = enc1["h"][:, k].detach()
            keys = torch.cat([tgt_k, neg_all[:, k]], dim=0)
            loss_nce = loss_nce + F.cross_entropy(temp * (q @ keys.T), torch.arange(batch_size, device=DEV)) / n_slots

        # 4. Post-touch movable presence & aim prediction (combines self-supervised vis_yt + encoder self-distillation)
        vm1 = vis_mt[b1]
        vis_tgt1 = torch.where(vm1, vis_yt[b1], torch.sigmoid(enc1["vis_logit"].detach()))
        loss_next_vis = F.binary_cross_entropy_with_logits(tr_out["vis_next_logit"], vis_tgt1)
        am1 = aim_mt[b1] & (vis_tgt1 > 0.5)
        loss_next_aim = F.smooth_l1_loss(tr_out["aim_next"][am1], aim_yt[b1][am1], beta=0.04) if am1.any() else torch.tensor(0.0, device=DEV)

        w_l1 = 0.0015 * min(1.0, max(0.0, (st - 500) / 1000.0))
        loss_l1 = w_l1 * tr_out["ctx_m"].abs().mean()

        loss = (
            2.5 * loss_aim + 1.5 * loss_vis + 1.5 * loss_val + 1.0 * loss_next_val
            + 2.0 * loss_chg + 0.5 * loss_nce + 3.0 * loss_inv
            + 2.0 * loss_next_vis + 1.0 * loss_next_aim + loss_l1
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        sched.step()

        if st % 700 == 0 or st == steps:
            log(f"seed={seed} step={st}/{steps} loss={loss.item():.3f} (aim={loss_aim.item():.4f} vis={loss_vis.item():.3f} val={loss_val.item():.3f} chg={loss_chg.item():.3f} nvis={loss_next_vis.item():.3f}) {time.time()-t0:.1f}s")

    net.eval()
    H_tr, Val_tr, V_tr, A_tr = encode_all(net, Z_tr)
    s_idx = np.where(tgt["same"])[0]
    slot_cuts = {}
    for k in range(n_slots):
        kd = meta["slot_kind"][k]
        if kd == "mov":
            # 2x 90th percentile of self-supervised table contact/release error, clamped to [1.8, 2.6] cm
            tab_m = tgt["aim_m"] & (tgt["aim_y"][:, 2] <= meta["z_cut"])
            err_m = np.linalg.norm((A_tr[tab_m] - tgt["aim_y"][tab_m]) * 0.1, axis=1)
            slot_cuts[k] = float(np.clip(2.0 * np.quantile(err_m, 0.90), 0.018, 0.026))
        elif kd == "slide":
            # Calibrate threshold on 1-D controllable state |val[b] - val[a]| using unchanged vs changed segments!
            unchg_k = s_idx[tgt["slot_chg"][s_idx, k] < 0.5]
            d_u = np.abs(Val_tr[unchg_k + 1, k] - Val_tr[unchg_k, k])
            slot_cuts[k] = float(max(np.quantile(d_u, 0.94) * 1.35, 0.18))
        elif kd == "lock":
            slot_cuts[k] = 0.0  # logit sign (val > 0 vs val <= 0)
        else:
            unchg_k = s_idx[tgt["slot_chg"][s_idx, k] < 0.5]
            d_u = np.linalg.norm(H_tr[unchg_k + 1, k] - H_tr[unchg_k, k], axis=1)
            slot_cuts[k] = float(np.quantile(d_u, 0.92) * 1.3)

    # Cause slot prototypes AND representative button-press u_act for counterfactual / BFS queries
    cause_protos = {}
    cause_press_u = {}
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
    return net, meta


def encode_all(net, Z_np, batch_size=256):
    H_l, Val_l, V_l, A_l = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(Z_np), batch_size):
            zt = torch.from_numpy(Z_np[i:i + batch_size]).to(DEV)
            o = net.encode(zt)
            H_l.append(o["h"].cpu().numpy())
            Val_l.append(o["val"].cpu().numpy())
            V_l.append(torch.sigmoid(o["vis_logit"]).cpu().numpy())
            A_l.append(o["aim"].cpu().numpy())
    return np.concatenate(H_l, 0), np.concatenate(Val_l, 0), np.concatenate(V_l, 0), np.concatenate(A_l, 0)
