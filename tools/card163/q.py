"""Card 163: one prediction to act by.

q(situation, action; event) in [0, 1]: 1 if the action brings the event now, GAMMA times the best prediction one step later
otherwise.  An event is a token changing to a given code, or the agent arriving on a token.  The network is card 162's form
made general: every token is what it is (one table of code embeddings) and where it lies (a function of its offset vector
from the agent), combined by attention.  No rotation is built in.

Learned from play kept in order (seq.py), by the rule that learns a tick: the target at the step where the event happens is
1; at any other step it is bootstrapped from the network's own next prediction.  Every step teaches about any token on the
map: the event asked about is one that came later in the episode (hindsight), or any token and known outcome.
"""
import os, time, math, pickle, numpy as np, torch, torch.nn as nn
from grid import NA, UNK, OUT

dev = "cuda" if torch.cuda.is_available() else "cpu"
D, HEADS, LAYERS, FF, GAMMA, S = 64, 4, 2, 256, 0.95, 25
FREQ = (2, 4, 8, 16)


def background(maps):
    """the commonest code is left out of the token list (a cell with no token)"""
    n = np.bincount(np.concatenate([m.ravel() for m in maps]).astype(np.int64)); n[[UNK, OUT]] = 0
    return int(n.argmax())


def tokens_of(m, bg):
    xs, ys = np.nonzero((m != bg) & (m != OUT) & (m != UNK))
    return xs, ys, m[xs, ys]


class Pack:
    """the experience as tensors on the device"""
    def __init__(s, seq):
        T, eps = seq["T"], seq["eps"]; s.NC = NC = len(T.codes.triples); s.ARR = NC; maps, WH, st, tr, ev, ep_end = [], [], [], [], [], []
        s.bg = background([m for e in eps for m in e["maps"]]); known = np.zeros((NC, NC + 1), bool); known[:, NC] = True
        for e in eps:
            v0 = len(maps); s0 = len(st); n = len(e["acts"]); first = {}
            for t, v in enumerate(e["ver"]):
                first.setdefault(v, t)
            for i, m in enumerate(e["maps"]):
                a = np.full((S, S), OUT, np.int16); a[:m.shape[0], :m.shape[1]] = m; maps.append(a); WH.append(m.shape)
                if i:
                    old = e["maps"][i - 1]
                    for x, y in np.argwhere((old != m) & (old != UNK)):
                        ev.append((s0 + first[i] - 1, x, y, m[x, y])); known[old[x, y], m[x, y]] = True
            st += [(v0 + e["ver"][t], *e["poses"][t], e["items"][t]) for t in range(n + 1)]; ep_end += [s0 + n] * (n + 1)
            tr += [(s0 + t, e["acts"][t], 0 if t < n - 1 else 1 if (e["ended"] and e["succ"]) else 2) for t in range(n)]
        V = len(maps); toks = [tokens_of(m, s.bg) for m in maps]; L = max(len(t[0]) for t in toks); TX = np.zeros((V, L), np.int64); TY = TX.copy(); TC = TX.copy(); TM = np.zeros((V, L), bool)
        for i, (xs, ys, cs) in enumerate(toks):
            k = len(xs); TX[i, :k] = xs; TY[i, :k] = ys; TC[i, :k] = cs; TM[i, :k] = True
        sets = T.seq_sets; M = max(1, max(len(x) for x in sets)); IC = np.zeros((len(sets), M), np.int64); IA = IC.copy(); IM = np.zeros((len(sets), M), bool)
        for i, x in enumerate(sets):
            for j, (k, a) in enumerate(sorted(x)):
                IC[i, j] = k; IA[i, j] = a; IM[i, j] = True
        t = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), dtype=dt, device=dev); st = np.array(st); tr = np.array(tr); ev = np.array(ev)
        s.MAP = t(np.stack(maps), torch.int16); s.W = t([w for w, h in WH]); s.H = t([h for w, h in WH]); s.TX, s.TY, s.TC, s.TM = t(TX), t(TY), t(TC), t(TM, torch.bool)
        s.IC, s.IA, s.IM = t(IC), t(IA), t(IM, torch.bool); s.ver, s.px, s.py, s.pd, s.it = (t(st[:, i]) for i in range(5)); s.ep_end = t(ep_end)
        s.tr_s, s.tr_a, s.tr_k = (t(tr[:, i]) for i in range(3)); s.ev_s, s.ev_x, s.ev_y, s.ev_o = (t(ev[:, i]) for i in range(4))
        s.fe = t(np.searchsorted(ev[:, 0], np.arange(len(st)))); s.le = t(np.searchsorted(ev[:, 0], np.array(ep_end))); s.known = t(known, torch.float32)
        s.ev_tr = t(np.searchsorted(tr[:, 0], ev[:, 0]))               # the transition of each event (transitions are in state order, one per state but the last of an episode)
        assert (tr[np.searchsorted(tr[:, 0], ev[:, 0]), 0] == ev[:, 0]).all()
        s.sizes = dict(codes=NC, maps=V, tokens_most=L, states=len(st), transitions=len(tr), events=len(ev), background=s.bg)

    def code_at(s, ver, x, y):
        return s.MAP[ver, x, y].long()

    def sample(s, B):
        """transitions and, for each, an event to ask about"""
        geo = lambda p: (torch.log(torch.rand(B, device=dev).clamp(min=1e-9)) / math.log(1 - p)).long()
        ti = torch.randint(len(s.tr_s), (B,), device=dev); ne = B // 7; e0 = torch.randint(len(s.ev_s), (ne,), device=dev); ti[:ne] = s.ev_tr[e0]
        st = s.tr_s[ti]; ver = s.ver[st]; u = torch.rand(B, device=dev)
        cnt = s.TM[ver].sum(1); slot = (torch.rand(B, device=dev) * cnt.clamp(min=1)).long(); tok = (torch.rand(B, device=dev) < 0.6) & (cnt > 0)       # any token, or any cell
        gx = torch.where(tok, s.TX[ver, slot], (torch.rand(B, device=dev) * s.W[ver]).long()); gy = torch.where(tok, s.TY[ver, slot], (torch.rand(B, device=dev) * s.H[ver]).long())
        o = torch.multinomial(s.known[s.code_at(ver, gx, gy).clamp(max=s.NC - 1)], 1)[:, 0]
        j = s.fe[st] + geo(0.5); he = (u < 0.4) & (j < s.le[st]); j = j.clamp(max=len(s.ev_s) - 1); he[:ne] = True; j[:ne] = e0                           # an event that came later
        gx = torch.where(he, s.ev_x[j], gx); gy = torch.where(he, s.ev_y[j], gy); o = torch.where(he, s.ev_o[j], o)
        sj = torch.minimum(st + 1 + geo(0.15), s.ep_end[st]); ha = ~he & (u < 0.6)                                                                         # a place the agent stood on later
        gx = torch.where(ha, s.px[sj], gx); gy = torch.where(ha, s.py[sj], gy); o = torch.where(ha, torch.full_like(o, s.ARR), o)
        return st, s.tr_a[ti], s.tr_k[ti], gx, gy, o

    def hit(s, st, gx, gy, o):
        c0 = s.code_at(s.ver[st], gx, gy); c1 = s.code_at(s.ver[st + 1], gx, gy); on0 = (s.px[st] == gx) & (s.py[st] == gy); on1 = (s.px[st + 1] == gx) & (s.py[st + 1] == gy)
        return torch.where(o == s.ARR, on1 & ~on0, (c1 == o) & (c0 != o) & (c0 != UNK))

    def rows(s, st, gx, gy, o):
        ver = s.ver[st]; L = int(s.TM[ver].sum(1).max().clamp(min=1)); it = s.it[st]; px, py = s.px[st], s.py[st]
        return dict(pd=s.pd[st], ac=s.code_at(ver, gx, gy), adx=gx - px, ady=gy - py, o=o, tc=s.TC[ver, :L], tdx=s.TX[ver, :L] - px[:, None], tdy=s.TY[ver, :L] - py[:, None], tm=s.TM[ver, :L],
                    ic=s.IC[it], ia=s.IA[it], im=s.IM[it])


class Net(nn.Module):
    def __init__(s, NC):
        super().__init__(); s.NC = NC
        s.what = nn.Embedding(NC + 1, D); s.want = nn.Embedding(NC + 1, D); s.head = nn.Embedding(4, D); s.gone = nn.Embedding(NA, D); s.kind = nn.Embedding(3, D)
        s.where = nn.Sequential(nn.Linear(2 + 4 * len(FREQ), D), nn.ReLU(), nn.Linear(D, D))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True, norm_first=True), LAYERS, enable_nested_tensor=False)
        s.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, NA))

    def wh(s, dx, dy):
        dx = dx.float(); dy = dy.float(); f = [dx / 8, dy / 8]
        for k in FREQ:
            f += [torch.sin(math.pi * dx / k), torch.cos(math.pi * dx / k), torch.sin(math.pi * dy / k), torch.cos(math.pi * dy / k)]
        return s.where(torch.stack(f, -1))

    def forward(s, pd, ac, adx, ady, o, tc, tdx, tdy, tm, ic, ia, im):
        """logits of q for each action"""
        B = len(pd); k = s.kind.weight
        x = torch.cat([s.head(pd)[:, None], (s.what(ac) + s.wh(adx, ady) + s.want(o))[:, None], s.what(tc) + s.wh(tdx, tdy) + k[0], s.what(ic) + s.gone(ia) + k[1]], 1)
        pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x.device), ~tm, ~im], 1)
        return s.out(s.enc(x, src_key_padding_mask=pad)[:, 0])


DOUBLE = False      # "+dq" in the stage name: the next step's best action is chosen by the fitted network and valued by the slow copy (double Q-learning, van Hasselt 2010),
                    # so that noise in the predictions is not taken for a better future


def fit(seq_path, out, steps=40000, B=1024, budget=520, log=print):
    """resumable: stops at the time budget and continues from the saved state on the next call"""
    t0 = time.time(); P = Pack(pickle.load(open(seq_path, "rb"))); torch.manual_seed(0); net = Net(P.NC).to(dev); tgt = Net(P.NC).to(dev); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.0); it0 = 0; hist = []
    if os.path.exists(out):
        ck = torch.load(out, map_location=dev, weights_only=False); net.load_state_dict(ck["net"]); tgt.load_state_dict(ck["tgt"]); opt.load_state_dict(ck["opt"]); it0 = ck["it"]; hist = ck["hist"]
    log("packed in %.0f s" % (time.time() - t0), P.sizes, "from step", it0); acc = [0.0, 0.0, 0.0, 0]
    for it in range(it0 + 1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = 5e-4 * min(1.0, it / 500) * (1.0 if it < 0.7 * steps else 0.3)
        st, a, kind, gx, gy, o = P.sample(B); hit = P.hit(st, gx, gy, o)
        with torch.no_grad():
            R1 = P.rows(st + 1, gx, gy, o); qt = torch.sigmoid(tgt(**R1)); nxt = qt.gather(1, net(**R1).argmax(1, keepdim=True))[:, 0] if DOUBLE else qt.max(1)[0]; y = torch.where(hit, torch.ones_like(nxt), torch.where(kind == 0, GAMMA * nxt, torch.zeros_like(nxt))); w = (hit | (kind != 2)).float()
        lg = net(**P.rows(st, gx, gy, o)).gather(1, a[:, None])[:, 0]; loss = (nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for p, q in zip(tgt.parameters(), net.parameters()):
                p.lerp_(q, 0.005)
            acc[0] += float(loss); acc[1] += float(y.mean()); acc[2] += float((torch.sigmoid(lg)[hit]).mean()) if hit.any() else 0.0; acc[3] += 1
        if it % 1000 == 0:
            hist.append((it, round(acc[0] / acc[3], 4), round(acc[1] / acc[3], 4), round(acc[2] / acc[3], 3))); log("step", *hist[-1], "(loss, mean target, q at the event step)  %.0f s" % (time.time() - t0)); acc = [0.0, 0.0, 0.0, 0]
        if it == steps or time.time() - t0 > budget:
            break
    torch.save(dict(net=net.state_dict(), tgt=tgt.state_dict(), opt=opt.state_dict(), it=it, hist=hist, NC=P.NC, bg=P.bg, known=P.known.cpu().numpy() > 0, steps=steps), out)
    log("saved at step", it, "of", steps)
    return it >= steps


if __name__ == "__main__":
    import sys
    exp, stage = sys.argv[1], sys.argv[2]; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40000; budget = int(sys.argv[4]) if len(sys.argv) > 4 else 520
    DOUBLE = "+dq" in stage; base = stage.split("+")[0]; src = "runs/163/seq_%s.pkl" % exp if base.startswith("R") else "runs/163/seq_%s_%s.pkl" % (exp, base)
    done = fit(src, "runs/163/q_%s_%s.pt" % (stage, exp), steps, budget=budget, log=lambda *a: print(*a, flush=True)); print("DONE" if done else "MORE")
