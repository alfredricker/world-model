"""Card 164: fixed single-event System 1 prediction q(situation, action; event).

Two fixes over card 163's q.py:
  1. An event specifies the achieving action: (cell, next_code, action) for a token change, or (cell, ARR, NA) for arrival.
     This separates picking up a box from destroying it by toggle (both turn the box cell to empty).
  2. Within-segment bootstrapping (SINGLE=True): q bootstraps across steps where no other token changes and returns 0 when
     a different event occurs.  q(s, a; E) is therefore a clean single-event skill and reachability predictor:
     high (~GAMMA**d) when E is directly achievable from s without intermediate events, and ~0 when blocked or precondition-unmet.

Trained solely on reward-free random play kept in order (card 163's seq_<exp>.pkl, equal to card 160's 700,000 steps).
"""
import os, time, math, pickle, numpy as np, torch, torch.nn as nn
from grid import NA, UNK, OUT

dev = "cuda" if torch.cuda.is_available() else "cpu"
D, HEADS, LAYERS, FF, GAMMA, S = 64, 4, 2, 256, 0.95, 25
FREQ = (2, 4, 8, 16)
ARR_ACT = NA


def background(maps):
    """the commonest code is left out of the token list (a cell with no token)"""
    n = np.bincount(np.concatenate([m.ravel() for m in maps]).astype(np.int64)); n[[UNK, OUT]] = 0
    return int(n.argmax())


def tokens_of(m, bg):
    xs, ys = np.nonzero((m != bg) & (m != OUT) & (m != UNK))
    return xs, ys, m[xs, ys]


class Pack:
    """the experience as tensors on the device, with (x, y, next_code, action) events and segment boundaries"""
    def __init__(s, seq):
        T, eps = seq["T"], seq["eps"]; s.NC = NC = len(T.codes.triples); s.ARR = NC
        maps, WH, st, tr, ev, ep_end = [], [], [], [], [], []
        s.bg = background([m for e in eps for m in e["maps"]])
        known_set = [{(NC, ARR_ACT)} for _ in range(NC)]
        for e in eps:
            v0 = len(maps); s0 = len(st); n = len(e["acts"]); first = {}
            for t, v in enumerate(e["ver"]):
                first.setdefault(v, t)
            for i, m in enumerate(e["maps"]):
                a = np.full((S, S), OUT, np.int16); a[:m.shape[0], :m.shape[1]] = m; maps.append(a); WH.append(m.shape)
                if i:
                    old = e["maps"][i - 1]; step_i = first[i] - 1; act_i = int(e["acts"][step_i])
                    for x, y in np.argwhere((old != m) & (old != UNK)):
                        c0, c1 = int(old[x, y]), int(m[x, y])
                        ev.append((s0 + step_i, int(x), int(y), c1, act_i))
                        if c0 < NC:
                            known_set[c0].add((c1, act_i))
            st += [(v0 + e["ver"][t], *e["poses"][t], e["items"][t]) for t in range(n + 1)]; ep_end += [s0 + n] * (n + 1)
            tr += [(s0 + t, e["acts"][t], 0 if t < n - 1 else 1 if (e["ended"] and e["succ"]) else 2) for t in range(n)]
        V = len(maps); toks = [tokens_of(m, s.bg) for m in maps]; L = max(len(t[0]) for t in toks)
        TX = np.zeros((V, L), np.int64); TY = TX.copy(); TC = TX.copy(); TM = np.zeros((V, L), bool)
        for i, (xs, ys, cs) in enumerate(toks):
            k = len(xs); TX[i, :k] = xs; TY[i, :k] = ys; TC[i, :k] = cs; TM[i, :k] = True
        sets = T.seq_sets; M = max(1, max(len(x) for x in sets))
        IC = np.zeros((len(sets), M), np.int64); IA = IC.copy(); IM = np.zeros((len(sets), M), bool)
        for i, x in enumerate(sets):
            for j, (k, a) in enumerate(sorted(x)):
                IC[i, j] = k; IA[i, j] = a; IM[i, j] = True
        Kmax = max(len(ks) for ks in known_set)
        KO = np.full((NC, Kmax), NC, np.int64); KA = np.full((NC, Kmax), ARR_ACT, np.int64); KC = np.zeros(NC, np.int64)
        for c, ks in enumerate(known_set):
            ks_sorted = sorted(ks); KC[c] = len(ks_sorted)
            for j, (oo, aa) in enumerate(ks_sorted):
                KO[c, j] = oo; KA[c, j] = aa
        t = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        st = np.array(st); tr = np.array(tr); ev = np.array(ev)
        s.MAP = t(np.stack(maps), torch.int16); s.W = t([w for w, h in WH]); s.H = t([h for w, h in WH])
        s.TX, s.TY, s.TC, s.TM = t(TX), t(TY), t(TC), t(TM, torch.bool)
        s.IC, s.IA, s.IM = t(IC), t(IA), t(IM, torch.bool)
        s.ver, s.px, s.py, s.pd, s.it = (t(st[:, i]) for i in range(5)); s.ep_end = t(ep_end)
        s.tr_s, s.tr_a, s.tr_k = (t(tr[:, i]) for i in range(3))
        s.ev_s, s.ev_x, s.ev_y, s.ev_o, s.ev_a = (t(ev[:, i]) for i in range(5))
        s.fe = t(np.searchsorted(ev[:, 0], np.arange(len(st)))); s.le = t(np.searchsorted(ev[:, 0], np.array(ep_end)))
        ev_tr_np = np.searchsorted(tr[:, 0], ev[:, 0])
        assert (tr[ev_tr_np, 0] == ev[:, 0]).all()
        s.ev_tr = t(ev_tr_np)
        has_ev = np.zeros(len(tr), bool); has_ev[ev_tr_np] = True; s.has_ev = t(has_ev, torch.bool)
        # end of the current event-free segment for every state: the step of the next event, or ep_end if none
        fe_np = np.searchsorted(ev[:, 0], np.arange(len(st))); le_np = np.searchsorted(ev[:, 0], np.array(ep_end))
        seg_end = np.where(fe_np < le_np, ev[np.minimum(fe_np, len(ev) - 1), 0], np.array(ep_end))
        s.seg_end = t(seg_end)
        s.KO, s.KA, s.KC = t(KO), t(KA), t(KC); s.known_set = [sorted(ks) for ks in known_set]
        # non-wall (interactive or goal) tokens per map: codes with >1 known event or not the wall code (commonest non-bg token)
        tok_counts = np.bincount(TC[TM].ravel(), minlength=NC)
        wall_code = int(tok_counts.argmax()) if tok_counts.sum() > 0 else -1
        is_inter = (TC != wall_code) & TM
        IL = max(1, int(is_inter.sum(1).max()))
        IX = np.zeros((V, IL), np.int64); IY = IX.copy(); IC_m = IX.copy(); IM_m = np.zeros((V, IL), bool)
        for i in range(V):
            idx = np.nonzero(is_inter[i])[0]; k = len(idx)
            if k:
                IX[i, :k] = TX[i, idx]; IY[i, :k] = TY[i, idx]; IC_m[i, :k] = TC[i, idx]; IM_m[i, :k] = True
        s.IX, s.IY, s.IC_m, s.IM_m = t(IX), t(IY), t(IC_m), t(IM_m, torch.bool)
        s.sizes = dict(codes=NC, maps=V, tokens_most=L, inter_most=IL, states=len(st), transitions=len(tr), events=len(ev), background=s.bg, wall=wall_code)

    def code_at(s, ver, x, y):
        return s.MAP[ver, x, y].long()

    def sample(s, B, mode="full"):
        """transitions and, for each, an event (gx, gy, o, ea) to ask about"""
        single = mode in ("single", "reach")
        geo = lambda p: (torch.log(torch.rand(B, device=dev).clamp(min=1e-9)) / math.log(1 - p)).long()
        ti = torch.randint(len(s.tr_s), (B,), device=dev)
        ne = B // 7 if mode == "full" else B // 5
        ne_pos = (ne * 3) // 4 if mode != "reach" else (ne // 2)
        e0 = torch.randint(len(s.ev_s), (ne,), device=dev); ti[:ne] = s.ev_tr[e0]
        st = s.tr_s[ti]; ver = s.ver[st]; u = torch.rand(B, device=dev)
        # default random query
        if mode == "reach":
            # sample interactive (non-wall) tokens 55%, empty/any cell 35%, any token 10% so blocked objects get plenty of negatives
            icnt = s.IM_m[ver].sum(1)
            islot = torch.minimum((torch.rand(B, device=dev) * icnt.clamp(min=1).float()).long(), (icnt - 1).clamp(min=0))
            cnt = s.TM[ver].sum(1)
            slot = torch.minimum((torch.rand(B, device=dev) * cnt.clamp(min=1).float()).long(), (cnt - 1).clamp(min=0))
            rx = torch.minimum((torch.rand(B, device=dev) * s.W[ver].float()).long(), s.W[ver] - 1)
            ry = torch.minimum((torch.rand(B, device=dev) * s.H[ver].float()).long(), s.H[ver] - 1)
            r = torch.rand(B, device=dev)
            use_i = (r < 0.55) & (icnt > 0); use_t = (~use_i) & (r < 0.65) & (cnt > 0)
            gx = torch.where(use_i, s.IX[ver, islot], torch.where(use_t, s.TX[ver, slot], rx))
            gy = torch.where(use_i, s.IY[ver, islot], torch.where(use_t, s.TY[ver, slot], ry))
        else:
            cnt = s.TM[ver].sum(1)
            slot = torch.minimum((torch.rand(B, device=dev) * cnt.clamp(min=1).float()).long(), (cnt - 1).clamp(min=0))
            tok = (torch.rand(B, device=dev) < 0.6) & (cnt > 0)
            rx = torch.minimum((torch.rand(B, device=dev) * s.W[ver].float()).long(), s.W[ver] - 1)
            ry = torch.minimum((torch.rand(B, device=dev) * s.H[ver].float()).long(), s.H[ver] - 1)
            gx = torch.where(tok, s.TX[ver, slot], rx)
            gy = torch.where(tok, s.TY[ver, slot], ry)
        c0 = s.code_at(ver, gx, gy).clamp(0, s.NC - 1)
        ki = torch.minimum((torch.rand(B, device=dev) * s.KC[c0].float()).long(), s.KC[c0] - 1)
        o = s.KO[c0, ki]; ea = s.KA[c0, ki]
        # hindsight event
        j = s.fe[st] if single else (s.fe[st] + geo(0.5))
        he_p = 0.40 if mode == "full" else 0.30
        he = (u < he_p) & (j < s.le[st]); j = j.clamp(max=len(s.ev_s) - 1)
        he[:ne_pos] = True; j[:ne_pos] = e0[:ne_pos]
        if mode != "reach":
            he[ne_pos:ne] = True; j[ne_pos:ne] = e0[ne_pos:ne]
        gx = torch.where(he, s.ev_x[j], gx); gy = torch.where(he, s.ev_y[j], gy)
        o = torch.where(he, s.ev_o[j], o); ea = torch.where(he, s.ev_a[j], ea)
        # wrong-action / other-object negatives on event transitions (exact y=0 under single/reach!)
        if ne > ne_pos:
            if mode == "reach":
                # half swapped action on e0's cell, half keep the random interactive token (exact y=0 at segment end!)
                mid = ne_pos + (ne - ne_pos) // 2
                sl = slice(ne_pos, mid)
                gx[sl] = s.ev_x[e0[sl]]; gy[sl] = s.ev_y[e0[sl]]
                c_alt = s.code_at(ver[sl], gx[sl], gy[sl]).clamp(0, s.NC - 1)
                k_alt = torch.minimum((torch.rand(mid - ne_pos, device=dev) * s.KC[c_alt].float()).long(), s.KC[c_alt] - 1)
                o[sl] = s.KO[c_alt, k_alt]; ea[sl] = s.KA[c_alt, k_alt]
            else:
                sl = slice(ne_pos, ne); c_alt = s.code_at(ver[sl], gx[sl], gy[sl]).clamp(0, s.NC - 1)
                k_alt = torch.minimum((torch.rand(ne - ne_pos, device=dev) * s.KC[c_alt].float()).long(), s.KC[c_alt] - 1)
                o[sl] = s.KO[c_alt, k_alt]; ea[sl] = s.KA[c_alt, k_alt]
        # hindsight arrival
        end = s.seg_end[st] if single else s.ep_end[st]
        sj = torch.minimum(st + 1 + geo(0.15), end)
        ha_p = 0.60 if mode == "full" else 0.55
        ha = ~he & (u < ha_p) & (sj > st)
        if mode == "reach":
            ha[ne_pos:ne] = False  # keep exact-zero interactive negatives on event transitions!
        gx = torch.where(ha, s.px[sj], gx); gy = torch.where(ha, s.py[sj], gy)
        o = torch.where(ha, torch.full_like(o, s.ARR), o); ea = torch.where(ha, torch.full_like(ea, ARR_ACT), ea)
        return ti, st, s.tr_a[ti], s.tr_k[ti], gx, gy, o, ea

    def hit(s, st, a, gx, gy, o, ea):
        c0 = s.code_at(s.ver[st], gx, gy); c1 = s.code_at(s.ver[st + 1], gx, gy)
        on0 = (s.px[st] == gx) & (s.py[st] == gy); on1 = (s.px[st + 1] == gx) & (s.py[st + 1] == gy)
        return torch.where(o == s.ARR, on1 & ~on0, (c1 == o) & (c0 != o) & (c0 != UNK) & (a == ea))

    def rows(s, st, gx, gy, o, ea):
        ver = s.ver[st]; L = int(s.TM[ver].sum(1).max().clamp(min=1)); it = s.it[st]; px, py = s.px[st], s.py[st]
        return dict(pd=s.pd[st], ac=s.code_at(ver, gx, gy), adx=gx - px, ady=gy - py, o=o, ea=ea,
                    tc=s.TC[ver, :L], tdx=s.TX[ver, :L] - px[:, None], tdy=s.TY[ver, :L] - py[:, None], tm=s.TM[ver, :L],
                    ic=s.IC[it], ia=s.IA[it], im=s.IM[it])


class Net(nn.Module):
    def __init__(s, NC):
        super().__init__(); s.NC = NC
        s.what = nn.Embedding(NC + 1, D); s.want = nn.Embedding(NC + 1, D); s.ev_act = nn.Embedding(NA + 1, D)
        s.head = nn.Embedding(4, D); s.gone = nn.Embedding(NA, D); s.kind = nn.Embedding(3, D)
        s.where = nn.Sequential(nn.Linear(2 + 4 * len(FREQ), D), nn.ReLU(), nn.Linear(D, D))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True, norm_first=True), LAYERS, enable_nested_tensor=False)
        s.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, NA))

    def wh(s, dx, dy):
        dx = dx.float(); dy = dy.float(); f = [dx / 8, dy / 8]
        for k in FREQ:
            f += [torch.sin(math.pi * dx / k), torch.cos(math.pi * dx / k), torch.sin(math.pi * dy / k), torch.cos(math.pi * dy / k)]
        return s.where(torch.stack(f, -1))

    def forward(s, pd, ac, adx, ady, o, ea, tc, tdx, tdy, tm, ic, ia, im):
        """logits of q for each action"""
        B = len(pd); k = s.kind.weight
        q_tok = (s.what(ac) + s.wh(adx, ady) + s.want(o) + s.ev_act(ea))[:, None]
        x = torch.cat([s.head(pd)[:, None], q_tok, s.what(tc) + s.wh(tdx, tdy) + k[0], s.what(ic) + s.gone(ia) + k[1]], 1)
        pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x.device), ~tm, ~im], 1)
        return s.out(s.enc(x, src_key_padding_mask=pad)[:, 0])


def fit(seq_path, out, steps=40000, B=1024, budget=520, mode="full", log=print):
    """resumable: stops at the time budget and continues from the saved state on the next call"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    single = mode in ("single", "reach")
    t0 = time.time(); P = Pack(pickle.load(open(seq_path, "rb"))); torch.manual_seed(0)
    net = Net(P.NC).to(dev); tgt = Net(P.NC).to(dev); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.0); it0 = 0; hist = []
    if os.path.exists(out):
        ck = torch.load(out, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); tgt.load_state_dict(ck["tgt"]); opt.load_state_dict(ck["opt"]); it0 = ck["it"]; hist = ck["hist"]
    log("packed in %.0f s" % (time.time() - t0), P.sizes, "mode=%s" % mode, "from step", it0); acc = [0.0, 0.0, 0.0, 0]
    for it in range(it0 + 1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = 5e-4 * min(1.0, it / 500) * (1.0 if it < 0.7 * steps else 0.3)
        ti, st, a, kind, gx, gy, o, ea = P.sample(B, mode=mode); hit = P.hit(st, a, gx, gy, o, ea)
        with torch.no_grad():
            R1 = P.rows(st + 1, gx, gy, o, ea); nxt = torch.sigmoid(tgt(**R1)).max(1)[0]
            cont = (kind == 0) & (~P.has_ev[ti]) if single else (kind == 0)
            y = torch.where(hit, torch.ones_like(nxt), torch.where(cont, GAMMA * nxt, torch.zeros_like(nxt)))
            w = (hit | (kind != 2)).float()
        lg = net(**P.rows(st, gx, gy, o, ea)).gather(1, a[:, None])[:, 0]
        loss = (nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for p, q in zip(tgt.parameters(), net.parameters()):
                p.lerp_(q, 0.005)
            acc[0] += float(loss); acc[1] += float(y.mean()); acc[2] += float((torch.sigmoid(lg)[hit]).mean()) if hit.any() else 0.0; acc[3] += 1
        if it % 1000 == 0:
            hist.append((it, round(acc[0] / acc[3], 4), round(acc[1] / acc[3], 4), round(acc[2] / acc[3], 3)))
            log("step", *hist[-1], "(loss, mean target, q at the event step)  %.0f s" % (time.time() - t0)); acc = [0.0, 0.0, 0.0, 0]
        if it % 10000 == 0 and it < steps:
            torch.save(dict(net=net.state_dict(), tgt=tgt.state_dict(), opt=opt.state_dict(), it=it, hist=hist,
                            NC=P.NC, bg=P.bg, known_set=P.known_set, mode=mode, single=single, steps=steps), out)
        if it == steps or time.time() - t0 > budget:
            break
    torch.save(dict(net=net.state_dict(), tgt=tgt.state_dict(), opt=opt.state_dict(), it=it, hist=hist,
                    NC=P.NC, bg=P.bg, known_set=P.known_set, mode=mode, single=single, steps=steps), out)
    log("saved at step", it, "of", steps)
    return it >= steps


if __name__ == "__main__":
    import sys
    exp = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40000; budget = int(sys.argv[4]) if len(sys.argv) > 4 else 520
    src = "runs/163/seq_%s.pkl" % exp
    done = fit(src, "runs/164/q_%s_%s.pt" % (mode, exp), steps, budget=budget, mode=mode, log=lambda *a: print(*a, flush=True))
    print("DONE" if done else "MORE")
