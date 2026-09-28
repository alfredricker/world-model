"""Card 013 feasibility gate: walking as a chain of short place conditions.

Key world (play starts as card 012 run 5). Two ways with exact ready frames
(A0): 1 = forward onto the goal square, 2 = toggle the door. Link k of a way
learns "A_{k-1} can be reached within j walking steps", j = 1..H (fixed-horizon
reachability, grounded on A_{k-1}); A_k = A_{k-1} or within H of it. Walking
from a frame uses the smallest k with A_k on: the move with the highest
sum over j of link k's values. Trained through a copy of the pretrained
encoder (cross-entropy, uniform walking batch + half from frames where the
way's condition is on, as in card 012).

  (a) upper bound: every link grounded on exact A_{k-1}, trained together
  (b) learned chain: link k grounded on learned A_{k-1}, level by level
  (c) baseline: card 012's flat walk value (discount 0.8) to A0

Score (held-out episodes): share of chosen moves that reduce the exact
walking distance to A0 (random moves: ~45%), by distance; AUC of learned A_k
against exact "within k*H".

Run: PYTHONPATH=tools/card012:tools/card013 bin/prun python -c "import bench_chain; bench_chain.main()"
"""
import copy, json, multiprocessing as mp, sys, time
import numpy as np
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld
from worldmodel.envs.keydoor_render import render
import bench_reach as br

H, K = 4, 8
WAYS = (1, 2)
PLAY = 1 / 3


def _dist(job):
    layouts, ep, arrays = job
    outs = [np.full((len(ep), 2), -1, np.int16) for _ in arrays]
    cur = ex = None
    for i in range(len(ep)):
        if ep[i] != cur:
            cur = ep[i]; ex = dl.Exact(layouts[cur], br.PARENT, br.ACTION)
        for o, st in zip(outs, arrays):
            s = dl.to_tup(st[i])
            for j, n in enumerate(WAYS):
                if ex.holds(n, s) and not ex.holds(br.PARENT[n], s):
                    o[i, j] = ex.distances(n, s).get(s[:3], -1)
    return outs


def distances(pool, layouts, ep, arrays, jobs=160):
    """Exact walking steps to each way's ready states (0 = ready; -1 = unreachable or goal already on)."""
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    sel = starts[np.linspace(0, len(starts) - 1, min(jobs, len(starts))).astype(int)]
    b = sorted(set(sel.tolist()) | {len(ep)})
    tasks = [({int(e): layouts[e] for e in np.unique(ep[x:y])}, ep[x:y], [a[x:y] for a in arrays]) for x, y in zip(b, b[1:])]
    parts = pool.map(_dist, tasks)
    return [np.concatenate([p[k] for p in parts]) for k in range(len(arrays))]


def Head(out, width=256):
    """Two MLPs, the smaller estimate used (card 012's MinHead with any output size)."""
    torch, nn, F = dl._torch()

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.nets = nn.ModuleList(nn.Sequential(nn.Linear(256, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                                    nn.Linear(width, out)) for _ in range(2))
            with torch.no_grad():
                for net in self.nets:
                    net[-1].bias.fill_(-4.0)

        def forward(self, z):
            return torch.minimum(self.nets[0](z), self.nets[1](z))

        def both(self, z):
            return self.nets[0](z), self.nets[1](z)

    return _Head()


class Net:
    """Encoder copy + a head with n slots x H horizons x 3 moves (min of two, as card 012)."""

    def __init__(self, enc, n, dev):
        torch, nn, F = dl._torch()
        self.torch, self.F, self.n, self.dev = torch, F, n, dev
        self.net = nn.ModuleDict({"enc": copy.deepcopy(enc), "head": Head(n * H * 3)}).to(dev)
        self.tgt = copy.deepcopy(self.net).requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=3e-4, fused=True)
        self.out = n * H * 3

    def q(self, img, target=False, both=False):
        m = self.tgt if target else self.net
        z = m["enc"](img)
        if both:
            return [x.view(-1, self.n, H, 3) for x in m["head"].both(z)]
        return m["head"](z).view(-1, self.n, H, 3)


def train_links(net, tiles, c0, c1, act, term1, ready0, ready1, active, rows, focus, updates, log, gen, tag, bs=1024):
    """ready0/ready1: (M, n) bool, 'in the link's target set' on s0 / s1. active: slot mask."""
    torch, F = net.torch, net.F
    n = net.n
    act_m = torch.zeros(n, device=net.dev); act_m[active] = 1.0
    P, TP = list(net.net.parameters()), list(net.tgt.parameters())
    k_f = (bs // 2) // max(len(focus), 1)
    t0 = time.monotonic()
    for u in range(updates):
        r = torch.cat([rows[torch.randint(len(rows), (bs - k_f * len(focus),), device=net.dev, generator=gen)]]
                      + [f[torch.randint(len(f), (k_f,), device=net.dev, generator=gen)] for f in focus])
        img0 = render(c0[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
        img1 = render(c1[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
        a = act[r]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qs = net.q(img0, both=True)
            with torch.no_grad():
                q1 = net.q(img1, target=True)
        with torch.no_grad():
            v1 = torch.sigmoid(q1.float()).max(3).values                       # (B, n, H)
            boot = torch.cat([torch.zeros_like(v1[:, :, :1]), v1[:, :, :-1]], 2)  # horizon j from j-1
            y = torch.where(ready1[r][:, :, None], 1.0, torch.where(term1[r][:, None, None], 0.0, boot))
            mk = ((~ready0[r]).float() * act_m)[:, :, None]
        loss = 0.0
        for q in qs:
            qa = q.float().gather(3, a.view(-1, 1, 1, 1).expand(-1, n, H, 1)).squeeze(3)
            ce = F.binary_cross_entropy_with_logits(qa, y, reduction="none")
            loss = loss + ((ce * mk).sum((0, 2)) / mk.sum((0, 2)).clamp_min(1.0)).sum() / 2
        net.opt.zero_grad(set_to_none=True); loss.backward(); net.opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(TP, P, 0.005)
        if u % 5000 == 0 or u == updates - 1:
            log(f"  {tag} {u}: loss {float(loss):.4f} {(u + 1) / (time.monotonic() - t0):.0f} upd/s")


def read(net, tiles, codes, slots=None, chunk=8192):
    """Sigmoid values (N, n or len(slots), H, 3)."""
    torch = net.torch
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(codes), chunk):
            img = render(codes[s:s + chunk].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
            v = torch.sigmoid(net.q(img).float())
            out.append(v if slots is None else v[:, slots])
    return torch.cat(out)


def main():
    updates = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    link_updates = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    pool = mp.get_context("fork").Pool(20)
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:5.0f}s] {m}", flush=True)
    br.WORLD = "key"
    layouts, tr, seq, st0 = dl.collect(pool, "key", 30000, 11, PLAY)
    tl, _, tseq, _ = dl.collect(pool, "key", 300, 999, 0.0, seq_episodes=300)
    D0, D1 = distances(pool, layouts, tr["ep"], [tr["s0"], tr["s1"]])
    (Dt,) = distances(pool, tl, tseq["ep"], [tseq["st"]])
    log(f"rows {len(tr['act'])}; test frames {len(Dt)}; reachable share way1/way2 {(D0 >= 0).mean(0).round(4).tolist()}; "
        f"max distance {int(D0.max())}; test frames per distance band (way1) "
        f"{[int(((Dt[:, 0] > (k - 1) * H) & (Dt[:, 0] <= k * H)).sum()) for k in range(1, K + 1)]}")
    torch, nn, F = dl._torch()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    trainer = dl.Trainer(dl.make_model(), tr, dev)
    L0 = torch.zeros(len(tr["act"]), dl.K_MAX, dtype=torch.bool, device=dev)
    L1 = L0.clone(); L1[:, 0] = trainer.term1
    trainer.train(L0, L1, [0], 30000, log)
    tiles, c0, c1, act, term1 = trainer.tiles, trainer.c0, trainer.c1, trainer.act, trainer.term1
    rows = torch.nonzero(act <= 2).squeeze(1)
    d0, d1 = T(D0.astype(np.int32)), T(D1.astype(np.int32))
    focus = [torch.nonzero((act <= 2) & (d0[:, w] >= 0)).squeeze(1) for w in range(2)]
    codes_t = T(tseq["codes"])
    gen = torch.Generator(device=dev).manual_seed(0)

    # scoring set: held-out frames that can reach a ready state, not ready yet; exact next-step distances
    rs = np.random.default_rng(1)
    score_sets = []
    for w, n in enumerate(WAYS):
        idx = np.flatnonzero(Dt[:, w] >= 1)
        idx = rs.choice(idx, min(4000, len(idx)), replace=False)
        good = np.zeros((len(idx), 3), bool)
        for q_, i in enumerate(idx):
            e = tseq["ep"][i]; st = dl.to_tup(tseq["st"][i]); ex = dl.Exact(tl[e], br.PARENT, br.ACTION)
            for mv in range(3):
                s1 = ld.step(tl[e], st, mv)[0]
                d = ex.distances(n, s1).get(s1[:3]) if s1[3:] == st[3:] else None
                good[q_, mv] = d is not None and d < Dt[i, w]
        score_sets.append((idx, good))

    def score(pick_fn, name):
        out = {}
        for w in range(2):
            idx, good = score_sets[w]
            pick = pick_fn(w, idx)
            ok = pick >= 0
            hit = np.zeros(len(idx), bool)
            hit[ok] = good[np.arange(len(idx))[ok], pick[ok]]
            dist = Dt[idx, w]
            out[f"way{w + 1}"] = {"moves_closer": round(float(hit.mean()), 3), "no_link_on": round(float((~ok).mean()), 3),
                                  "by_distance": {f"{a}-{b}": round(float(hit[(dist >= a) & (dist <= b)].mean()), 3)
                                                  for a, b in ((1, 4), (5, 8), (9, 12), (13, 16), (17, 40))
                                                  if ((dist >= a) & (dist <= b)).any()},
                                  "random": round(float(good.mean()), 3)}
        log(f"RESULT {name}: {json.dumps(out)}")
        return out

    res = {}
    # (c) baseline: flat walk value (card 012), discount 0.8, to A0
    flat = Net(trainer.model.enc, 2, dev)
    gam = 0.8
    Pf, TPf = list(flat.net.parameters()), list(flat.tgt.parameters())
    r0f, r1f = d0 == 0, d1 == 0
    k_f = 512 // 2
    t0 = time.monotonic()
    for u in range(updates):
        r = torch.cat([rows[torch.randint(len(rows), (512,), device=dev, generator=gen)]]
                      + [f[torch.randint(len(f), (k_f,), device=dev, generator=gen)] for f in focus])
        img0 = render(c0[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
        img1 = render(c1[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
        a = act[r]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qs = flat.q(img0, both=True)
            with torch.no_grad():
                q1 = flat.q(img1, target=True)
        with torch.no_grad():
            v1 = torch.sigmoid(q1[:, :, 0].float()).max(2).values
            y = torch.where(r1f[r], 1.0, torch.where(term1[r][:, None], 0.0, gam * v1))
            mk = (~r0f[r]).float()
        loss = 0.0
        for q in qs:
            qa = q[:, :, 0].float().gather(2, a.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(2)
            loss = loss + ((F.binary_cross_entropy_with_logits(qa, y, reduction="none") * mk).sum(0) / mk.sum(0).clamp_min(1)).sum() / 2
        flat.opt.zero_grad(set_to_none=True); loss.backward(); flat.opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(TPf, Pf, 0.005)
    log(f"  flat trained {time.monotonic() - t0:.0f}s")
    Vf = read(flat, tiles, codes_t)[:, :, 0].cpu().numpy()          # (N, 2, 3)
    res["c_flat_walk"] = score(lambda w, idx: Vf[idx, w].argmax(1), "(c) flat walk, discount 0.8")
    del flat

    # (a) upper bound: all links on exact A_{k-1}
    n = 2 * K
    slot = lambda w, k: w * K + (k - 1)
    ready0 = torch.zeros(len(act), n, dtype=torch.bool, device=dev); ready1 = ready0.clone()
    for w in range(2):
        for k in range(1, K + 1):
            ready0[:, slot(w, k)] = (d0[:, w] >= 0) & (d0[:, w] <= (k - 1) * H)
            ready1[:, slot(w, k)] = (d1[:, w] >= 0) & (d1[:, w] <= (k - 1) * H)
    up = Net(trainer.model.enc, n, dev)
    train_links(up, tiles, c0, c1, act, term1, ready0, ready1, list(range(n)), rows, focus, updates, log, gen, "upper")
    V = read(up, tiles, codes_t).cpu().numpy()                        # (N, n, H, 3)

    def pick_exact_k(w, idx):
        k = np.clip(np.ceil(Dt[idx, w] / H).astype(int), 1, K)
        return V[idx, w * K + k - 1].sum(1).argmax(1)
    res["a_upper_bound"] = score(pick_exact_k, "(a) exact place conditions, learned short walks")
    del up

    # (b) learned chain, level by level
    lc = Net(trainer.model.enc, n, dev)
    A0 = [d0 == 0, d1 == 0]                                           # A_0 exact (ready frames; card 012 learns these well)
    prev0, prev1 = A0[0].clone(), A0[1].clone()                       # (M, 2): learned A_{k-1} on s0, s1
    rd0 = torch.zeros(len(act), n, dtype=torch.bool, device=dev); rd1 = rd0.clone()
    Z = {}
    auc = {}
    Vt_by_k = {}
    for k in range(1, K + 1):
        for w in range(2):
            rd0[:, slot(w, k)] = prev0[:, w]; rd1[:, slot(w, k)] = prev1[:, w]
        active = [slot(w, j) for w in range(2) for j in range(1, k + 1)]
        train_links(lc, tiles, c0, c1, act, term1, rd0, rd1, active, rows, focus, link_updates, log, gen, f"chain k={k}")
        # learned A_k on training rows (s0, s1) and on test frames
        for which, codes in ((0, c0), (1, c1)):
            Vk = read(lc, tiles, codes, slots=[slot(0, k), slot(1, k)])[:, :, H - 1].max(2).values > 0.5   # (M, 2)
            if which == 0:
                prev0 = prev0 | Vk
            else:
                prev1 = prev1 | Vk
        Vt = read(lc, tiles, codes_t).cpu().numpy()
        Vt_by_k[k] = Vt
        for w in range(2):
            y = (Dt[:, w] >= 0) & (Dt[:, w] <= k * H)
            m = Dt[:, w] != 0
            auc[f"way{w + 1}_A{k}"] = round(dl.auc(Vt[m, slot(w, k), H - 1].max(1), y[m]), 4)
    log(f"learned A_k AUC vs exact 'within k*H': {json.dumps(auc)}")
    Vt = Vt_by_k[K]

    def pick_learned_k(w, idx):
        on = np.zeros((len(idx), K + 1), bool)
        on[:, 0] = Dt[idx, w] == 0
        for k in range(1, K + 1):
            on[:, k] = on[:, k - 1] | (Vt[idx, slot(w, k), H - 1].max(1) > 0.5)
        has = on[:, 1:].any(1)
        k = np.where(has, on[:, 1:].argmax(1) + 1, 1)
        pick = Vt[idx, w * K + k - 1].sum(1).argmax(1)
        return np.where(has, pick, -1)
    res["b_learned_chain"] = score(pick_learned_k, "(b) learned place conditions, learned short walks")
    res["b_auc"] = auc
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
