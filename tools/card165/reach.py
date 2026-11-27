"""Card 165: Differentiable Neural Reachability Propagation (RNet).

Learns binary spatial reachability from pure reward-free random play (seq_<exp>.pkl) with ZERO Dijkstra,
ZERO move-action labels, ZERO headings, and ZERO hardcoded cardinal offsets:
  - alpha(c) in (0, 1): learned per-code traversability (what can be stepped onto)
  - beta(dx, dy) in (0, 1): continuous MLP over relative offset (dx, dy) (which single-step moves exist)
  - gamma(dx, dy) in (0, 1): continuous MLP over relative offset (dx, dy) (from which offsets an event can be brought about)

Forward pass iterates differentiable max-product message passing:
  r^(t+1)[x, y] = max(r^(t)[x, y], alpha(map[x, y]) * max_{dx, dy} beta(dx, dy) * r^(t)[x - dx, y - dy])
  r_act[x, y]   = max_{dx, dy} gamma(dx, dy) * r^(K)[x - dx, y - dy]

Trained end-to-end with positive-unlabeled (PU) weighted BCE on:
  1. 1-step pose transitions (pose_t -> pose_{t+1}): discovers beta(dx, dy) and alpha(c) cleanly;
  2. multi-step constant-map segment visitations (K=16 iterations): reinforces alpha(c) across multi-room barriers;
  3. 1-step observed event offsets (X - pose_t): discovers gamma(dx, dy).
"""
import os, time, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from grid import UNK, OUT

dev = "cuda" if torch.cuda.is_available() else "cpu"
RAD = 2
OFFS_ALL = [(dx, dy) for dx in range(-RAD, RAD + 1) for dy in range(-RAD, RAD + 1)]
OFFS_STEP = [(dx, dy) for dx, dy in OFFS_ALL if (dx, dy) != (0, 0)]


def shift_stack(r, offs):
    """r: [B, W, H] -> [B, len(offs), W, H], where entry k at (x, y) is r[x - dx, y - dy] (0 outside)."""
    B, W, H = r.shape
    rp = F.pad(r, (RAD, RAD, RAD, RAD), value=0.0)
    return torch.stack([rp[:, RAD - dx:RAD - dx + W, RAD - dy:RAD - dy + H] for dx, dy in offs], 1)


class RNet(nn.Module):
    def __init__(s, NC):
        super().__init__(); s.NC = NC
        s.pass_logit = nn.Parameter(torch.zeros(NC))
        s.step_mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        s.act_mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        s.register_buffer("offs_step", torch.tensor(OFFS_STEP, dtype=torch.float32))
        s.register_buffer("offs_all", torch.tensor(OFFS_ALL, dtype=torch.float32))

    def alpha(s):
        return torch.sigmoid(s.pass_logit)

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


def pu_bce(p, y, w_neg=0.005, mask=None):
    """Positive-unlabeled weighted BCE on probabilities p in [0, 1]."""
    p = p.clamp(1e-6, 1.0 - 1e-6)
    loss = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, w_neg))
    if mask is not None:
        w = w * mask
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def extract_data(seq, max_segs=8000):
    """Extract 1-step moves, constant-map segments (>=20 steps), and 1-step event offsets from seq_<exp>.pkl."""
    T, eps = seq["T"], seq["eps"]; NC = len(T.codes.triples)
    S = 25  # padded grid size (training maps are <= 11x6, inference works at any size!)
    # 1. 1-step moves: (5x5 local code patch around pose_t, target offset index in OFFS_STEP)
    # 2. 1-step events: (target offset index in OFFS_ALL)
    # 3. multi-step constant-map segments: (map [S, S], valid [S, S], start (px, py), visited [S, S])
    m_patch, m_tgt = [], []
    ev_tgt = []
    segs = []
    off_step_ix = {o: i for i, o in enumerate(OFFS_STEP)}
    off_all_ix = {o: i for i, o in enumerate(OFFS_ALL)}
    for e in eps:
        maps, ver, poses, acts = e["maps"], e["ver"], e["poses"], e["acts"]
        n = len(acts)
        # 1-step moves (subsample if huge; keep all rare-code moves!)
        for t in range(n):
            p0, p1 = poses[t], poses[t + 1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            if (dx, dy) in off_step_ix:
                mp = maps[ver[t]]; W, H = mp.shape
                patch = np.full(len(OFFS_STEP), OUT, np.int64)
                for j, (ox, oy) in enumerate(OFFS_STEP):
                    nx, ny = p0[0] + ox, p0[1] + oy
                    if 0 <= nx < W and 0 <= ny < H:
                        patch[j] = int(mp[nx, ny])
                m_patch.append(patch); m_tgt.append(off_step_ix[(dx, dy)])
        # 1-step events
        for t, ev in enumerate(e["evs"]):
            if ev:
                p0 = poses[t]
                for x, y, c, c2 in ev:
                    d = (x - p0[0], y - p0[1])
                    if d in off_all_ix:
                        ev_tgt.append(off_all_ix[d])
        # constant-map segments >= 20 steps
        if len(segs) < max_segs:
            t0 = 0
            while t0 < n and len(segs) < max_segs:
                v = ver[t0]; t1 = t0
                while t1 < n and ver[t1] == v:
                    t1 += 1
                if t1 - t0 >= 20:
                    mp = maps[v]; W, H = mp.shape
                    pad = np.full((S, S), OUT, np.int16); pad[:W, :H] = mp
                    val = np.zeros((S, S), bool); val[:W, :H] = True
                    vis = np.zeros((S, S), bool)
                    for tt in range(t0, t1):
                        vis[poses[tt][0], poses[tt][1]] = True
                    segs.append((pad, val, poses[t0][0], poses[t0][1], vis))
                t0 = t1
    return NC, np.array(m_patch, np.int64), np.array(m_tgt, np.int64), np.array(ev_tgt, np.int64), segs


def fit(exp, steps=1500, B=256, w_neg=0.004, log=print):
    t0 = time.time()
    seq = pickle.load(open("runs/163/seq_%s.pkl" % exp, "rb"))
    NC, m_patch, m_tgt, ev_tgt, segs = extract_data(seq)
    log("extracted exp %s in %.1f s: %d moves, %d events, %d segs (>=20 steps)" %
        (exp, time.time() - t0, len(m_tgt), len(ev_tgt), len(segs)))
    t = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    MP = t(m_patch); MT = t(m_tgt); ET = t(ev_tgt)
    SM = t(np.stack([s[0] for s in segs]))
    SV = t(np.stack([s[1] for s in segs]), torch.float32)
    SPX = t([s[2] for s in segs]); SPY = t([s[3] for s in segs])
    SVIS = t(np.stack([s[4] for s in segs]), torch.float32)
    # Trim padded canvas to max W, H present in training segments for fast training
    Wmax = int(SV.any(0).any(1).nonzero()[-1]) + 1
    Hmax = int(SV.any(0).any(0).nonzero()[-1]) + 1
    SM, SV, SVIS = SM[:, :Wmax, :Hmax], SV[:, :Wmax, :Hmax], SVIS[:, :Wmax, :Hmax]

    torch.manual_seed(0); net = RNet(NC).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    for it in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = 0.02 if it < 0.6 * steps else 0.005
        # 1. 1-step move loss: teaches beta(dx, dy) from observed 1-step pose deltas AND alpha(c) from stepped-on codes
        mi = torch.randint(len(MT), (B * 4,), device=dev)
        al = net.alpha()[MP[mi]]; bt = net.beta()[None, :].expand(B * 4, -1)
        y1 = F.one_hot(MT[mi], len(OFFS_STEP)).float()
        l_step = pu_bce(bt, y1, w_neg=0.01)
        l_move = pu_bce(al * bt.detach(), y1, w_neg=w_neg)
        # 2. 1-step event offset loss
        ei = torch.randint(len(ET), (B,), device=dev)
        gm = net.gamma()[None, :].expand(B, -1)
        ye = F.one_hot(ET[ei], len(OFFS_ALL)).float()
        l_act = pu_bce(gm, ye, w_neg=0.01)
        # 3. Multi-step segment propagation loss (K=16 iterations; teaches alpha across multi-room barriers)
        si = torch.randint(len(SM), (B // 2,), device=dev)
        rK, _ = net.propagate(SM[si], SPX[si], SPY[si], K=16, early_stop=False, detach_beta=True)
        mask = SV[si].clone()
        mask[torch.arange(len(si), device=dev), SPX[si], SPY[si]] = 0.0
        l_seg = pu_bce(rK, SVIS[si], w_neg=0.01, mask=mask)
        loss = l_step + l_move + l_act + l_seg
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 300 == 0:
            log("step %4d  loss=%.4f (step=%.4f move=%.4f act=%.4f seg=%.4f)  %.1f s" %
                (it, float(loss.detach()), float(l_step.detach()), float(l_move.detach()), float(l_act.detach()), float(l_seg.detach()), time.time() - t0))
    os.makedirs("runs/165", exist_ok=True)
    out = "runs/165/reach_%s.pt" % exp
    torch.save(dict(net=net.state_dict(), NC=NC, w_neg=w_neg, steps=steps), out)
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
