"""Card 014: local-step walking (a value iteration network on the encoder's map).

Attached to card 012 run 5's saved network (conditions, ready frames); the
walking module has its own trainable copy of the encoder's convolutions
(run 5's network stays frozen so its conditions stay valid while acting).

Stages (argv[1]):
  gate  - (a) upper bound: supervised on exact walking distances, agent's
              true (facing, cell) given; (b) learned (TD, learned readout):
              readout check + walking on held-out frames
  main  - criterion 1 (walking, every way), 2 (unseen wall column vs a flat
          value trained the same way), 3 (acting on 500 new layouts)

Run: PYTHONPATH=tools/card012:tools/card013:tools/card014 bin/prun python -c \
       "import sys; sys.argv=['x','gate']; import bench_vin; bench_vin.main()"
"""
import copy, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld
from worldmodel.envs.keydoor_render import render, encode_logic_states

RUN = Path("runs/012run5_key")
K = next((int(a[2:]) for a in sys.argv if a.startswith("k=")), 32)   # recurrence steps (longest walk in the data: 23); argv k=24 (card 016)
NF = 32           # map feature channels fed to the step
NG = 16           # whole-frame summary channels fed to the step
PLAY = 1 / 3
HELD_WALL = 5
LOGIT = "logit" in sys.argv


# ---------------------------------------------------------------- exact distances

def _dist(job):
    layouts, parent, action, nodes, ep, arrays = job
    outs = [np.full((len(ep), len(nodes)), -1, np.int16) for _ in arrays]
    cur = ex = None
    for i in range(len(ep)):
        if ep[i] != cur:
            cur = ep[i]; ex = dl.Exact(layouts[cur], parent, action)
        for o, st in zip(outs, arrays):
            s = dl.to_tup(st[i])
            for j, n in enumerate(nodes):
                if ex.holds(n, s) and not ex.holds(parent[n], s):
                    o[i, j] = ex.distances(n, s).get(s[:3], -1)
    return outs


def distances(pool, layouts, tree, nodes, ep, arrays, jobs=160):
    """Exact walking steps to each node's ready states (0 = ready, -1 = unreachable or goal on)."""
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    sel = starts[np.linspace(0, len(starts) - 1, min(jobs, len(starts))).astype(int)]
    b = sorted(set(sel.tolist()) | {len(ep)})
    tasks = [({int(e): layouts[e] for e in np.unique(ep[x:y])}, list(tree.parent), list(tree.action), list(nodes),
               ep[x:y], [a[x:y] for a in arrays]) for x, y in zip(b, b[1:])]
    parts = pool.map(_dist, tasks)
    return [np.concatenate([p[k] for p in parts]) for k in range(len(arrays))]


def _dmaps(job):
    """Exact walking distance to node n's ready states from every (facing, cell): (len, nn, 4, 9, 8), -1 = none."""
    layouts, parent, action, nodes, items = job
    out = np.full((len(items), len(nodes), 4, 9, 8), -1, np.int8)
    succ = np.full((len(items), len(nodes), 3, 4, 9, 8), -1, np.int8)
    cache = {}
    for i, (e, st) in enumerate(items):
        lay = layouts[e]
        ex = cache.get(e) or cache.setdefault(e, dl.Exact(lay, parent, action))
        s = dl.to_tup(st)
        rest = s[3:]
        comp, members, _ = ex.component(rest)
        _, _, edges = ex.component(rest)
        for j, n in enumerate(nodes):
            for mem in members:
                D = ex.distances(n, (*mem[0], *rest))
                for (x, y, d), v in D.items():
                    out[i, j, d, y, x] = min(v, 127)
                for (x, y, d) in mem:
                    for a, t in edges[(x, y, d)]:          # moves into the goal square are absent: -1
                        if t in D:
                            succ[i, j, a, d, y, x] = min(D[t], 127)
    return out, succ


def _succ(job):
    """For scoring: exact distance after each move (3), per frame and node."""
    layouts, parent, action, items = job
    out = []
    ex_cache = {}
    for e, st, n in items:
        lay = layouts[e]
        ex = ex_cache.get(e) or ex_cache.setdefault(e, dl.Exact(lay, parent, action))
        s = dl.to_tup(st)
        row = []
        for mv in range(3):
            s1 = ld.step(lay, s, mv)[0]
            row.append(ex.distances(n, s1).get(s1[:3], -1) if s1[3:] == s[3:] else -1)
        out.append(row)
    return np.array(out, np.int16).reshape(-1, 3)


# ---------------------------------------------------------------- the module

def make_vin(trunk, n_ways, H=9, W=8, fixed=None):
    """H, W: map size in tiles. fixed: (4, H, W) one-hot readout used when no position is given (card 016)."""
    torch, nn, F = dl._torch()

    class VIN(nn.Module):
        """Target map T (ready here), one local step applied K times, agent readout A."""

        def __init__(self):
            super().__init__()
            self.trunk = copy.deepcopy(trunk)                       # conv layers of the encoder: (B, 64, 9, 8)
            self.feat = nn.Conv2d(64, NF, 1)
            # what each cell knows can be global (e.g. the held item, drawn in one corner tile);
            # how "within k steps" spreads stays local
            self.glob = nn.Sequential(nn.Flatten(), nn.Linear(64 * H * W, 64), nn.ReLU())
            self.glob_step = nn.Linear(64, NG)
            self.target = nn.Conv2d(64 + 64, 4 * n_ways, 3, padding=1)
            self.step = nn.Sequential(nn.Conv2d(4 + NF + NG, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 12, 1))
            self.attn = nn.Conv2d(64, 4, 3, padding=1)
            self.register_buffer("fixed", None if fixed is None else fixed[None].float())
            with torch.no_grad():
                self.target.bias.fill_(-4.0)
                self.step[-1].bias.fill_(-4.0)

        def forward(self, img, way, given=None, full=False):
            """img (B,3,72,64); way (B,) index; given: (B,4,9,8) one-hot agent entry or None (learned).
            Returns readouts: T (B,), A map (B,4,9,8), Q (B,K,3) sigmoid, V (B,K+1) sigmoid."""
            B = img.shape[0]
            m = self.trunk(img)
            g = self.glob(m)
            gm = g[:, :, None, None].expand(-1, -1, H, W)
            f = torch.cat([F.relu(self.feat(m)), self.glob_step(g)[:, :, None, None].expand(-1, -1, H, W)], 1)
            Tl = self.target(torch.cat([m, gm], 1)).float().reshape(B, -1, 4, H, W)[torch.arange(B, device=img.device), way]
            if LOGIT:
                # value-iteration style: the recurrence stays in logits (max is piecewise linear, so
                # gradients survive 32 steps); probabilities only at the readout
                A = given if given is not None else self.fixed.expand(B, -1, -1, -1) if self.fixed is not None \
                    else torch.softmax(self.attn(m).float().reshape(B, -1), 1).reshape(B, 4, H, W)
                V, Qr, Vr, maps, qmaps = Tl, [], [torch.sigmoid((A * Tl).sum((1, 2, 3)))], [Tl], []
                for _ in range(K):
                    ql = self.step(torch.cat([torch.sigmoid(V).to(f.dtype), f], 1)).float().reshape(B, 3, 4, H, W)
                    Qr.append(torch.sigmoid((A[:, None] * ql).sum((2, 3, 4))))
                    V = torch.maximum(Tl, ql.max(1).values)
                    Vr.append(torch.sigmoid((A * V).sum((1, 2, 3))))
                    maps.append(V); qmaps.append(ql)
                if full:
                    return torch.stack(maps, 1), torch.stack(qmaps, 1), A   # (B,K+1,4,9,8), (B,K,3,4,9,8) logits; (B,4,9,8)
                return Vr[0], A, torch.stack(Qr, 1), torch.stack(Vr, 1)
            T = torch.sigmoid(Tl)
            A = given if given is not None else self.fixed.expand(B, -1, -1, -1) if self.fixed is not None \
                    else torch.softmax(self.attn(m).float().reshape(B, -1), 1).reshape(B, 4, H, W)
            V, Qr, Vr = T, [], [(A * T).sum((1, 2, 3))]
            for _ in range(K):
                q = torch.sigmoid(self.step(torch.cat([V.to(f.dtype), f], 1)).float()).reshape(B, 3, 4, H, W)
                Qr.append((A[:, None] * q).sum((2, 3, 4)))
                V = torch.maximum(T, q.max(1).values)
                Vr.append((A * V).sum((1, 2, 3)))
            return Vr[0], A, torch.stack(Qr, 1), torch.stack(Vr, 1)

    return VIN()


def agent_onehot(st, dev, torch):
    """(B,4,9,8) one-hot of the agent's (facing, row=y, col=x) -- the upper bound's given position."""
    st = torch.as_tensor(st, device=dev).long()
    g = torch.zeros(len(st), 4, 9, 8, device=dev)
    g[torch.arange(len(st), device=dev), st[:, 2], st[:, 1], st[:, 0]] = 1.0
    return g


# ---------------------------------------------------------------- setup shared by the stages

class Setup:
    def __init__(self, log):
        torch, nn, F = dl._torch()
        self.torch, self.F, self.log = torch, F, log
        self.pool = mp.get_context("fork").Pool(20)
        self.layouts, self.tr, _, _ = dl.collect(self.pool, "key", 30000, 11, PLAY)
        self.tl, _, self.tseq, _ = dl.collect(self.pool, "key", 600, 999, 0.0, seq_episodes=600)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dev = dev = torch.device("cuda")
        ck = torch.load(RUN / "nets.pt", weights_only=False)
        res = json.loads((RUN / "result.json").read_text())
        model = dl.make_model(); model.load_state_dict(ck["model"]); model.eval()
        self.trainer = dl.Trainer(model, self.tr, dev)
        self.tree = dl.Tree.from_report(res["learned"]["tree"])
        self.way_index = {int(k): v for k, v in ck["way_index"].items()}
        self.lt = dl.LearnedTree(self.tree, self.trainer, model.val, self.way_index)
        pers = res["learned"]["persistence"]
        self.persist = [float(pers.get("/".join(self.tree.path(n)), np.nan)) if n else np.nan for n in range(len(self.tree.parent))]
        self.ways = sorted(self.way_index)
        log(f"ways {[ '/'.join(self.tree.path(c)) for c in self.ways]}")
        # run 5's own readiness and conditions on every training row (s0, s1)
        M = len(self.tr["act"])
        nw = len(self.ways)
        self.ready0 = torch.zeros(M, nw, dtype=torch.bool, device=dev); self.ready1 = self.ready0.clone()
        self.cond0 = torch.zeros(M, nw, dtype=torch.bool, device=dev)
        for which, codes in ((0, self.trainer.c0), (1, self.trainer.c1)):
            for s in range(0, M, 262144):
                Z = self.trainer.encode(codes[s:s + 262144])
                root = self.trainer.term1[s:s + 262144] if which == 1 else torch.zeros(len(Z), dtype=torch.bool, device=dev)
                out = self.lt.evaluate(Z, root)
                L, P = out["L"], out["P"]
                for j, c in enumerate(self.ways):
                    g, a = self.tree.parent[c], self.tree.action[c]
                    r = (P[:, g, a] > 0.5) & ~L[:, g]
                    (self.ready0 if which == 0 else self.ready1)[s:s + 262144, j] = r
                    if which == 0:
                        self.cond0[s:s + 262144, j] = L[:, c] & ~L[:, g]
        self.walking = (self.trainer.act <= 2) & ~self.trainer.term1
        log(f"ready frames per way {self.ready0.sum(0).tolist()}; condition-on walking rows "
            f"{(self.cond0 & self.walking[:, None]).sum(0).tolist()}")
        wall = np.array([l.wall_x for l in self.layouts])
        self.row_wall = torch.from_numpy(wall[self.tr["ep"]]).to(dev)
        # held-out frames: exact distances per way
        self.Dt, = distances(self.pool, self.tl, self.tree, self.ways, self.tseq["ep"], [self.tseq["st"]])
        self.test_wall = np.array([l.wall_x for l in self.tl])[self.tseq["ep"]]
        log(f"held-out frames {len(self.Dt)}; reachable per way {(self.Dt >= 1).sum(0).tolist()}")

    def score_set(self, w, idx):
        """Exact distance after each move for held-out frames idx of way w: (len, 3)."""
        c = self.ways[w]
        chunks = np.array_split(np.arange(len(idx)), 40)
        tasks = []
        for ch in chunks:
            eps = self.tseq["ep"][idx[ch]]
            tasks.append(({int(e): self.tl[e] for e in np.unique(eps)}, list(self.tree.parent), list(self.tree.action),
                          [(int(e), self.tseq["st"][i], c) for e, i in zip(eps, idx[ch])]))
        return np.concatenate(self.pool.map(_succ, tasks))


def images(codes, tiles):
    return render(codes.long(), tiles).permute(0, 3, 1, 2).float() / 255.0


# ---------------------------------------------------------------- training

def train(S, vin, updates, mode, allowed=None, d0=None, d1=None, tag="", bs=512):
    """mode 'learned': TD at the learned readout; 'upper': supervised on exact distances with the
    agent's position given (d0, d1: (M, nw) exact distances). allowed: row mask (criterion 2)."""
    torch, F = S.torch, S.F
    dev, tr = S.dev, S.trainer
    nw = len(S.ways)
    tgt = copy.deepcopy(vin).requires_grad_(False)
    opt = torch.optim.Adam(vin.parameters(), lr=3e-4)
    walk = S.walking if allowed is None else S.walking & allowed
    sets = []
    for j in range(nw):
        u = torch.nonzero(walk).squeeze(1)
        f = torch.nonzero(walk & S.cond0[:, j]).squeeze(1)
        r = torch.nonzero(walk & S.ready1[:, j] & ~S.ready0[:, j]).squeeze(1)
        sets.append([x if len(x) else u for x in (u, f, r)])
    per = bs // nw
    shares = (per // 4, per // 2, per - per // 4 - per // 2)
    gen = torch.Generator(device=dev).manual_seed(0)
    P, TP = list(vin.parameters()), list(tgt.parameters())
    kk = torch.arange(1, K + 1, device=dev)
    t0 = time.monotonic()
    for u in range(updates):
        rows, way, part = [], [], []
        for j in range(nw):
            for p, (x, n) in enumerate(zip(sets[j], shares)):
                rows.append(x[torch.randint(len(x), (n,), device=dev, generator=gen)])
                way.append(torch.full((n,), j, device=dev)); part.append(torch.full((n,), p, device=dev))
        r, way, part = torch.cat(rows), torch.cat(way), torch.cat(part)
        img0, img1 = images(tr.c0[r], tr.tiles), images(tr.c1[r], tr.tiles)
        a = tr.act[r].long()
        rd0, rd1 = S.ready0[r, way], S.ready1[r, way]
        if mode == "upper":
            st0 = torch.as_tensor(S.tr["s0"], device=dev)[r] if not hasattr(S, "_s0") else S._s0[r]
            T0, _, Q, V = vin(img0, way, given=agent_onehot(st0, dev, torch))
            e0, e1 = d0[r, way].long(), d1[r, way].long()
            yT = (e0 == 0).float()
            yQ = ((e1 >= 0)[:, None] & (e1[:, None] <= kk[None] - 1)).float()             # after the move: within k-1
            yV = ((e0 >= 0)[:, None] & (e0[:, None] <= torch.arange(K + 1, device=dev)[None])).float()
            qa = Q.gather(2, a.view(-1, 1, 1).expand(-1, K, 1)).squeeze(2)
            loss = F.binary_cross_entropy(T0.clamp(1e-5, 1 - 1e-5), yT) \
                + F.binary_cross_entropy(qa.clamp(1e-5, 1 - 1e-5), yQ) + F.binary_cross_entropy(V.clamp(1e-5, 1 - 1e-5), yV)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                T0, _, Q, _ = vin(img0, way)
                with torch.no_grad():
                    _, _, _, V1 = tgt(img1, way)
            # target map: "ready here", on the uniform and focus rows (not the over-sampled ready rows)
            mT = (part < 2).float()
            lT = (F.binary_cross_entropy(T0.clamp(1e-5, 1 - 1e-5), rd0.float(), reduction="none") * mT).sum() / mT.sum()
            # fixed-horizon TD at the readout: after move a, a ready state within k-1 steps
            y = torch.where(rd1[:, None], 1.0, torch.where(tr.term1[r][:, None], 0.0, V1[:, :K].float()))
            qa = Q.gather(2, a.view(-1, 1, 1).expand(-1, K, 1)).squeeze(2)
            mQ = (~rd0).float()[:, None]
            lQ = (F.binary_cross_entropy(qa.clamp(1e-5, 1 - 1e-5), y, reduction="none") * mQ).sum() / (mQ.sum() * K).clamp_min(1)
            loss = lT + lQ
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(TP, P, 0.005)
        if u % 2000 == 0 or u == updates - 1:
            S.log(f"  {tag} {u}: loss {float(loss.detach()):.4f} {(u + 1) / (time.monotonic() - t0):.1f} upd/s")
    vin.eval()


def read(S, vin, idx, way_j, given=False, chunk=2048):
    """Held-out frames idx, way j: move choice (argmax over moves of summed Q), reach V_K, attention argmax."""
    torch = S.torch
    codes = torch.as_tensor(S.tseq["codes"][idx], device=S.dev)
    moves, reach, amax = [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            img = images(codes[s:s + chunk], S.trainer.tiles)
            w = torch.full((len(img),), way_j, device=S.dev)
            g = agent_onehot(S.tseq["st"][idx[s:s + chunk]], S.dev, torch) if given else None
            _, A, Q, V = vin(img, w, given=g)
            moves.append(Q.sum(1).argmax(1)); reach.append(V[:, -1]); amax.append(A.view(len(img), -1).argmax(1))
    return torch.cat(moves).cpu().numpy(), torch.cat(reach).cpu().numpy(), torch.cat(amax).cpu().numpy()


def walk_score(S, vin, frames, given=False, tag=""):
    """frames: {way j: held-out indices}. Share of chosen moves that bring the agent closer."""
    out = {}
    for j, idx in frames.items():
        if not len(idx):
            continue
        succ = S.score_set(j, idx)
        mv, _, _ = read(S, vin, idx, j, given)
        d = S.Dt[idx, j]
        ok = (succ[np.arange(len(idx)), mv] >= 0) & (succ[np.arange(len(idx)), mv] < d)
        rnd = ((succ >= 0) & (succ < d[:, None])).mean()
        out["/".join(S.tree.path(S.ways[j]))] = {"frames": int(len(idx)), "moves_closer": round(float(ok.mean()), 4),
                                                 "random": round(float(rnd), 3),
                                                 "by_distance": {f"{a}-{b}": round(float(ok[(d >= a) & (d <= b)].mean()), 3)
                                                                 for a, b in ((1, 4), (5, 8), (9, 12), (13, 40)) if ((d >= a) & (d <= b)).any()}}
    S.log(f"RESULT walking {tag}: {json.dumps(out)}")
    return out


def pick_frames(S, n=3000, wall=None, seed=1, not_wall=None):
    rs = np.random.default_rng(seed)
    fr = {}
    for j in range(len(S.ways)):
        m = S.Dt[:, j] >= 1
        if wall is not None:
            m &= S.test_wall == wall
        if not_wall is not None:
            m &= S.test_wall != not_wall
        idx = np.flatnonzero(m)
        fr[j] = np.sort(rs.choice(idx, min(n, len(idx)), replace=False)) if len(idx) else idx
    return fr


# ---------------------------------------------------------------- stages

def gate(S, updates):
    torch = S.torch
    res = {}
    main = [j for j, c in enumerate(S.ways) if S.tree.path(c) in (("forward",), ("forward", "toggle"))]
    frames = {j: f for j, f in pick_frames(S).items() if j in main}
    # (a) upper bound: exact distances, agent's entry given
    d0, d1 = distances(S.pool, S.layouts, S.tree, S.ways, S.tr["ep"], [S.tr["s0"], S.tr["s1"]])
    S._s0 = torch.as_tensor(S.tr["s0"], device=S.dev)
    D0, D1 = torch.as_tensor(d0, device=S.dev), torch.as_tensor(d1, device=S.dev)
    up = make_vin(S.trainer.model.enc[:6], len(S.ways)).to(S.dev)
    train(S, up, updates, "upper", d0=D0, d1=D1, tag="upper")
    res["a_upper_bound"] = walk_score(S, up, frames, given=True, tag="(a) upper bound")
    del up
    # (b) learned: readout check and walking
    lv = make_vin(S.trainer.model.enc[:6], len(S.ways)).to(S.dev)
    train(S, lv, updates, "learned", tag="learned")
    idx = np.sort(np.random.default_rng(2).choice(len(S.Dt), 5000, replace=False))
    _, _, am = read(S, lv, idx, main[0])
    st = S.tseq["st"][idx]
    true = st[:, 2].astype(int) * 72 + st[:, 1].astype(int) * 8 + st[:, 0].astype(int)
    res["readout_finds_agent"] = round(float((am == true).mean()), 4)
    S.log(f"RESULT readout finds the agent: {res['readout_finds_agent']}")
    res["b_learned"] = walk_score(S, lv, frames, tag="(b) learned")
    torch.save(lv.state_dict(), "runs/014_vin_learned.pt")
    return res


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "gate"
    updates = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    S = Setup(log)
    res = {"gate": gate, "debug": debug, "readout": readout_check, "main": main_stage}[stage](S, updates)
    out = Path(f"runs/014_{stage}{'_logit' if LOGIT else ''}{'_fullmap' if 'fullmap' in sys.argv else ''}.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"done -> {out}")


def train_fullmap(S, vin, updates, main_, n_frames=60000, bs=256, learn_attn=False, relabel=None):
    """Upper bound with exact "within k steps" labels at every (facing, cell) of the map.
    learn_attn: the agent's position is not given; the attention is trained only by the per-move
    values read out through it (exact labels at the agent's own entry), never told where the agent is."""
    torch, F = S.torch, S.F
    rs = np.random.default_rng(5)
    rows = np.sort(rs.choice(np.flatnonzero(S.walking.cpu().numpy()), n_frames, replace=False))
    chunks = np.array_split(rows, 160)
    nodes = [S.ways[j] for j in main_]
    tasks = [({int(e): S.layouts[e] for e in np.unique(S.tr["ep"][ch])}, list(S.tree.parent), list(S.tree.action), nodes,
              [(int(S.tr["ep"][i]), S.tr["s0"][i]) for i in ch]) for ch in chunks]
    t0 = time.monotonic()
    parts = S.pool.map(_dmaps, tasks)
    dm = torch.as_tensor(np.concatenate([p[0] for p in parts]), device=S.dev)          # (N, 2, 4, 9, 8)
    sm = torch.as_tensor(np.concatenate([p[1] for p in parts]), device=S.dev)          # (N, 2, 3, 4, 9, 8)
    S.log(f"  distance maps {tuple(dm.shape)} in {time.monotonic() - t0:.0f}s")
    if relabel is not None:      # card 016: the maps in the walking module's own frame
        dm, sm = relabel(dm, sm, S.tr["s0"][rows])
    rows_t = torch.as_tensor(rows, device=S.dev)
    ks = torch.arange(K + 1, device=S.dev).view(1, -1, 1, 1, 1)
    opt = torch.optim.Adam(vin.parameters(), lr=3e-4)
    gen = torch.Generator(device=S.dev).manual_seed(0)
    t0 = time.monotonic()
    for u in range(updates):
        i = torch.randint(len(rows), (bs,), device=S.dev, generator=gen)
        wj = torch.randint(len(main_), (bs,), device=S.dev, generator=gen)
        way = torch.as_tensor(main_, device=S.dev)[wj]
        img = images(S.trainer.c0[rows_t[i]], S.trainer.tiles)
        g = agent_onehot(S.tr["s0"][rows[i.cpu().numpy()]], S.dev, torch)
        maps, qmaps, A = vin(img, way, given=None if learn_attn else g, full=True)
        d = dm[i, wj].long()[:, None]                                          # (B, 1, 4, 9, 8)
        y = ((d >= 0) & (d <= ks)).float()
        sd = sm[i, wj].long()[:, None]                                         # (B, 1, 3, 4, 9, 8)
        yq = ((sd >= 0) & (sd <= (ks[:, 1:] - 1)[..., None])).float()               # after the move: within k-1
        loss = F.binary_cross_entropy_with_logits(maps, y) + F.binary_cross_entropy_with_logits(qmaps, yq)
        if learn_attn:
            # the readout: per-move logits weighted by the attention; labels at the agent's entry
            # (g is used only to pick the label, as TD targets at the agent would supply it)
            ra = (A[:, None, None] * qmaps).sum((3, 4, 5))                     # (B, K, 3)
            ya = (yq * g[:, None, None]).sum((3, 4, 5))
            loss = loss + F.binary_cross_entropy_with_logits(ra, ya)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if u % 2000 == 0 or u == updates - 1:
            S.log(f"  fullmap {u}: loss {float(loss.detach()):.4f} {(u + 1) / (time.monotonic() - t0):.1f} upd/s")
    vin.eval()
    return rows, dm, sm


def probe_objects(S, trunk, tag, updates=3000):
    """Linear read-out (1x1 conv) of each tile's object class from a frozen map; per-class accuracy, held-out."""
    torch, nn, F = dl._torch()
    from worldmodel.envs.keydoor_render import OBJECTS
    names = ["empty", "wall", "goal", "door closed", "door open", "key", "switch off", "switch on", "vase"]
    def cls(o):
        if o is None: return 0
        if o == "wall": return 1
        if o == "goal": return 2
        if o[0] == "door": return 4 if o[2] == 2 else 3
        if o[0] == "key": return 5
        if o == ("ball", "grey"): return 6
        if o == ("ball", "yellow"): return 7
        return 8
    lut = torch.as_tensor([cls(o) for o in OBJECTS], device=S.dev)
    trunk = copy.deepcopy(trunk).eval().requires_grad_(False)
    head = nn.Conv2d(64, len(names), 1).to(S.dev)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    gen = torch.Generator(device=S.dev).manual_seed(0)
    M = len(S.trainer.c0)
    for u in range(updates):
        r = torch.randint(M, (256,), device=S.dev, generator=gen)
        codes = S.trainer.c0[r].long()
        with torch.no_grad():
            m = trunk(images(codes, S.trainer.tiles))
        y = lut[codes // 5]
        loss = F.cross_entropy(head(m.float()), y)
        opt.zero_grad(); loss.backward(); opt.step()
    codes = torch.as_tensor(S.tseq["codes"][:20000], device=S.dev).long()
    with torch.no_grad():
        pred = torch.cat([head(trunk(images(codes[s:s + 2048], S.trainer.tiles)).float()).argmax(1) for s in range(0, len(codes), 2048)])
    y = lut[codes // 5]
    acc = {names[c]: round(float((pred[y == c] == c).float().mean()), 4) for c in range(len(names)) if (y == c).any()}
    S.log(f"RESULT object read-out from the map ({tag}): {json.dumps(acc)}")
    return acc


def debug(S, updates):
    """Upper bound only, with where-is-the-error diagnostics on held-out frames.
    argv 'fullmap': supervised at every cell of the map, not only at the agent."""
    torch = S.torch
    main_ = [j for j, c in enumerate(S.ways) if S.tree.path(c) in (("forward",), ("forward", "toggle"))]
    up = make_vin(S.trainer.model.enc[:6], len(S.ways)).to(S.dev)
    res0 = {}
    if "fullmap" in sys.argv:
        res0["objects_run5_encoder"] = probe_objects(S, S.trainer.model.enc[:6], "run 5 encoder, frozen")
        train_fullmap(S, up, updates, main_)
        res0["objects_after_walk_training"] = probe_objects(S, up.trunk, "walking module's map after training")
    else:
        d0, d1 = distances(S.pool, S.layouts, S.tree, S.ways, S.tr["ep"], [S.tr["s0"], S.tr["s1"]])
        S._s0 = torch.as_tensor(S.tr["s0"], device=S.dev)
        train(S, up, updates, "upper", d0=torch.as_tensor(d0, device=S.dev), d1=torch.as_tensor(d1, device=S.dev), tag="upper")
    res = {}
    for j in main_:
        idx = np.sort(np.random.default_rng(3).choice(np.flatnonzero(S.Dt[:, j] >= 0), 3000, replace=False))
        codes = torch.as_tensor(S.tseq["codes"][idx], device=S.dev)
        with torch.no_grad():
            T0, _, Q, V = up(images(codes, S.trainer.tiles), torch.full((len(idx),), j, device=S.dev),
                             given=agent_onehot(S.tseq["st"][idx], S.dev, torch))
        d = S.Dt[idx, j]
        V = V.cpu().numpy()
        r = {"ready_here_acc": round(float(((T0.cpu().numpy() > 0.5) == (d == 0)).mean()), 4)}
        for k in (1, 4, 8, 12, 16, 24):
            y = d <= k
            r[f"V{k}_acc"] = round(float(((V[:, k] > 0.5) == y).mean()), 4)
        est = np.where((V > 0.5).any(1), (V > 0.5).argmax(1), 99)
        r["distance_exact_share"] = round(float((est == d).mean()), 4)
        r["distance_mean_abs_err"] = round(float(np.abs(est - d).mean()), 3)
        res["/".join(S.tree.path(S.ways[j]))] = r
    S.log(f"RESULT debug values: {json.dumps(res)}")
    res["walking"] = walk_score(S, up, {j: f for j, f in pick_frames(S).items() if j in main_}, given=True, tag="upper (debug)")
    res.update(res0)
    return res


def readout_check(S, updates):
    """Gate, readout check: recurrence supervised at every cell (as the upper bound), the attention
    learned from the readout alone. Share of held-out frames where it peaks at the agent's true
    (facing, cell), and walking through it."""
    main_ = [j for j, c in enumerate(S.ways) if S.tree.path(c) in (("forward",), ("forward", "toggle"))]
    lv = make_vin(S.trainer.model.enc[:6], len(S.ways)).to(S.dev)
    train_fullmap(S, lv, updates, main_, learn_attn=True)
    res = {}
    idx = np.sort(np.random.default_rng(2).choice(len(S.Dt), 5000, replace=False))
    st = S.tseq["st"][idx]
    true = st[:, 2].astype(int) * 72 + st[:, 1].astype(int) * 8 + st[:, 0].astype(int)
    for name, sel in (("wall_column_seen", S.test_wall[idx] != HELD_WALL), ("wall_column_unseen", S.test_wall[idx] == HELD_WALL)):
        _, _, am = read(S, lv, idx[sel], main_[0])
        res[f"readout_finds_agent_{name}"] = round(float((am == true[sel]).mean()), 4)
        cell_ok = (am % 72) == (true[sel] % 72)
        res[f"readout_right_cell_{name}"] = round(float(cell_ok.mean()), 4)
    S.log(f"RESULT readout finds the agent: {json.dumps(res)}")
    frames = {j: f for j, f in pick_frames(S).items() if j in main_}
    res["walking_learned_readout"] = walk_score(S, lv, frames, tag="full map, learned readout")
    res["walking_given_position"] = walk_score(S, lv, frames, given=True, tag="full map, position given (same module)")
    return res


def flat_score(S, frames, tag=""):
    """Run 5's own walking values (a flat value per way on the flattened encoder), same frames."""
    torch = S.torch
    out = {}
    for j, idx in frames.items():
        if not len(idx):
            continue
        succ = S.score_set(j, idx)
        codes = torch.as_tensor(S.tseq["codes"][idx], device=S.dev)
        with torch.no_grad():
            o = S.lt.evaluate(S.trainer.encode(codes), torch.zeros(len(idx), dtype=torch.bool, device=S.dev))
        mv = o["Q"][:, S.way_index[S.ways[j]]].float().argmax(1).cpu().numpy()
        d = S.Dt[idx, j]
        ok = (succ[np.arange(len(idx)), mv] >= 0) & (succ[np.arange(len(idx)), mv] < d)
        out["/".join(S.tree.path(S.ways[j]))] = {"frames": int(len(idx)), "moves_closer": round(float(ok.mean()), 4)}
    S.log(f"RESULT walking {tag}: {json.dumps(out)}")
    return out


def act_vin(S, vin, layouts, budget=200, eps=0.05, seed=0, view=None):
    """Criterion 3: run 5's learned conditions choose the way and fire its action (as dl.act,
    chooser and executor learned); walking moves come from the new module instead of run 5's values."""
    torch = S.torch
    rng = np.random.default_rng(seed)
    tl, lt, wi = S.tree, S.lt, S.way_index
    jof = {c: j for j, c in enumerate(S.ways)}
    states = [ld.start_state(l) for l in layouts]
    done = np.zeros(len(layouts), bool)
    steps = np.full(len(layouts), budget)
    for step_i in range(budget):
        live = np.flatnonzero(~done)
        if not len(live):
            break
        pairs = [(layouts[i], states[i]) for i in live]
        o = dl.learned_on_states(lt, pairs, S.trainer)
        L, P, W, R = [o[k].float().cpu().numpy() for k in ("L", "P", "W", "R")]
        acts, need = [None] * len(live), []
        for j, i in enumerate(live):
            c = dl.choose(tl, lambda n: bool(L[j, n]), lambda n: R[j, wi[n]], lambda n: W[j, wi[n]], S.persist)
            if c is None:
                continue
            g, aw = tl.parent[c], tl.action[c]
            if P[j, g, aw] > 0.5 and not L[j, g]:
                acts[j] = aw
            elif rng.random() < eps:
                acts[j] = int(rng.choice(dl.MOVES))
            else:
                need.append((j, jof[c]))
        if need:
            codes = torch.as_tensor(dl.encode_pairs([pairs[j] for j, _ in need]), device=S.dev)
            if view is not None:     # card 016: the walking module's own frame of the same states
                codes = view(codes, np.array([pairs[j][1][:3] for j, _ in need]))
            with torch.no_grad():
                _, _, Q, _ = vin(images(codes, S.trainer.tiles), torch.as_tensor([w for _, w in need], device=S.dev))
            for (j, _), a in zip(need, Q.sum(1).argmax(1).cpu().numpy()):
                acts[j] = int(a)
        for j, i in enumerate(live):
            a = acts[j] if acts[j] is not None else int(rng.integers(len(dl.ACTIONS)))
            states[i], end = ld.step(layouts[i], states[i], a)
            if end:
                done[i] = True
                steps[i] = step_i + 1
    return {"success": round(float(done.mean()), 4),
            "mean_steps_when_successful": round(float(steps[done].mean()), 1) if done.any() else None}


def main_stage(S, updates):
    """Criteria 1-3, all learned: fixed-horizon TD at the learned readout (no exact distances, no
    given position), trained without layouts whose wall is in column 5."""
    torch = S.torch
    res = {}
    lv = make_vin(S.trainer.model.enc[:6], len(S.ways)).to(S.dev)
    train(S, lv, updates, "learned", allowed=S.row_wall != HELD_WALL, tag="learned")
    torch.save(lv.state_dict(), "runs/014_vin_main.pt")
    idx = np.sort(np.random.default_rng(2).choice(len(S.Dt), 5000, replace=False))
    st = S.tseq["st"][idx]
    true = st[:, 2].astype(int) * 72 + st[:, 1].astype(int) * 8 + st[:, 0].astype(int)
    _, _, am = read(S, lv, idx, S.ways.index(next(c for c in S.ways if S.tree.path(c) == ("forward",))))
    res["readout_finds_agent"] = round(float((am == true).mean()), 4)
    S.log(f"RESULT readout finds the agent: {res['readout_finds_agent']}")
    seen, unseen = pick_frames(S, not_wall=HELD_WALL), pick_frames(S, wall=HELD_WALL)
    res["criterion_1_walking"] = walk_score(S, lv, seen, tag="criterion 1, every way, wall column seen")
    res["criterion_2_unseen_column"] = walk_score(S, lv, unseen, tag="criterion 2, wall column 5 (never trained on)")
    res["criterion_2_flat_run5"] = flat_score(S, unseen, tag="run 5's flat values, wall column 5 (trained with it)")
    lay_rng = np.random.default_rng(777)
    layouts = [ld.make_layout(8, "key", lay_rng) for _ in range(500)]
    res["criterion_3_acting"] = act_vin(S, lv, layouts)
    S.log(f"RESULT criterion 3 acting: {json.dumps(res['criterion_3_acting'])}")
    return res


if __name__ == "__main__":
    main()
