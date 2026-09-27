"""Bench: reach trained through the encoder (joint), key world, goal-square-only
encoder as the start, exact ready frames. Compare with frozen-encoder reach."""
import copy, json, multiprocessing as mp, sys, time
import numpy as np
from worldmodel import discover_logic as dl
from worldmodel.envs.keydoor_render import render
import bench_reach as br
from worldmodel.envs import logicdoor as ld


def main():
    world = sys.argv[1] if len(sys.argv) > 1 else "key"
    updates = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    br.WORLD = world
    pool = mp.get_context("fork").Pool(20)
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:5.0f}s] {m}", flush=True)
    layouts, tr, seq, st0 = dl.collect(pool, world, 30000, 11, 0.0)
    tl, _, tseq, _ = dl.collect(pool, world, 300, 999, 0.0, seq_episodes=300)
    lab0, lab1 = br.labels(pool, layouts, tr["ep"], [tr["s0"], tr["s1"]])
    (labt,) = br.labels(pool, tl, tseq["ep"], [tseq["st"]])
    torch, nn, F = dl._torch()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    T = lambda x: torch.from_numpy(x).to(dev)
    trainer = dl.Trainer(dl.make_model(), tr, dev)
    L0 = torch.zeros(len(tr["act"]), dl.K_MAX, dtype=torch.bool, device=dev)
    L1 = L0.clone(); L1[:, 0] = trainer.term1
    if not any(x.startswith("load=") for x in sys.argv):
        trainer.train(L0, L1, [0], 30000, log)
    r0, r1 = T(lab0[:, 1:, 1]), T(lab1[:, 1:, 1])
    n = 2
    # walking-move quality set: held-out frames that can reach a ready state (not already ready)
    rs = np.random.default_rng(1)
    qual = []
    for j, nd in enumerate((1, 2)):
        idx = np.flatnonzero(labt[:, nd, 0] & ~labt[:, br.PARENT[nd], 0] & ~labt[:, nd, 1])
        idx = rs.choice(idx, min(3000, len(idx)), replace=False)
        good = np.zeros((len(idx), 3), bool)
        for q, i in enumerate(idx):
            e = tseq["ep"][i]; st = dl.to_tup(tseq["st"][i]); ex = dl.Exact(tl[e], br.PARENT, br.ACTION)
            d0 = ex.distances(nd, st).get(st[:3])
            for mv in range(3):
                s1 = ld.step(tl[e], st, mv)[0]
                d1 = ex.distances(nd, s1).get(s1[:3]) if s1[3:] == st[3:] else None
                good[q, mv] = d1 is not None and d0 is not None and d1 < d0
        qual.append((idx, good))
    if any(x.startswith("load=") for x in sys.argv):   # score a saved run's network (its node ids for forward, forward/toggle)
        run = [x[5:] for x in sys.argv if x.startswith("load=")][0]
        ck = torch.load(run + "/nets.pt", weights_only=False)
        trainer.model.load_state_dict(ck["model"]); trainer.model.eval()
        tree = dl.Tree.from_report(ck["tree"])
        ids = {tree.path(c): c for c in range(1, len(tree.parent))}
        Z = trainer.encode(tseq["codes"])
        W, R, Q = dl.values_of(trainer.model.val, Z, dl.VAL_SLOTS)
        R, Q = R.float().cpu().numpy(), Q.float().cpu().numpy()
        out = {}
        for j, (nd, path) in enumerate(((1, ("forward",)), (2, ("forward", "toggle")))):
            c = ids[path]
            m = ~labt[:, br.PARENT[nd], 0] & ~labt[:, nd, 1]
            y = labt[m, nd, 0]; p = R[m, c] > 0.5
            idx, good = qual[j]
            pick = Q[idx, c].argmax(1)
            out["/".join(path)] = {"recall": round(float(p[y].mean()), 4), "false_on": round(float(p[~y].mean()), 4),
                                   "walk_move_closer": round(float(good[np.arange(len(idx)), pick].mean()), 3)}
            if nd == 1:
                ii = np.flatnonzero(m)[y]
                side = np.array([tseq["st"][i][0] > tl[tseq["ep"][i]].wall_x for i in ii])
                atdoor = np.array([tseq["st"][i][0] == tl[tseq["ep"][i]].wall_x for i in ii])
                dd = br.dist_labels(tl, {"ep": tseq["ep"][ii], "st": tseq["st"][ii]})[:, 0]
                pp = p[y]
                out["forward"]["recall_left_room"] = round(float(pp[~side & ~atdoor].mean()), 3)
                out["forward"]["recall_in_doorway"] = round(float(pp[atdoor].mean()), 3) if atdoor.any() else None
                out["forward"]["recall_right_room"] = round(float(pp[side].mean()), 3)
                out["forward"]["share_left_door_right"] = [round(float((~side & ~atdoor).mean()), 3), round(float(atdoor.mean()), 3), round(float(side.mean()), 3)]
                out["forward"]["recall_by_distance"] = {f"{lo}-{hi}": round(float(pp[(dd >= lo) & (dd <= hi)].mean()), 3) for lo, hi in ((1, 4), (5, 9), (10, 14), (15, 40)) if ((dd >= lo) & (dd <= hi)).any()}
        log(f"RESULT saved {run}: {json.dumps(out)}")
        return
    if "pipeline" in sys.argv:   # the pipeline's own trainer, slots 1 and 2 = the bench ways
        trainer.rd0[:, 1:3] = r0; trainer.rd1[:, 1:3] = r1
        walking = trainer.act <= 2
        focus = [torch.nonzero(walking & T(lab0[:, nd, 0]) & ~T(lab0[:, br.PARENT[nd], 0])).squeeze(1) for nd in (1, 2)]
        trainer.train(L0, L1, [0], updates, log, ways=[1, 2], focus=focus)
        codes = T(tseq["codes"])
        Z = trainer.encode(tseq["codes"])
        W, R, Q = dl.values_of(trainer.model.val, Z, dl.VAL_SLOTS)
        R, Q = R.float().cpu().numpy(), Q.float().cpu().numpy()
        out = {}
        for j, nd in enumerate((1, 2)):
            m = ~labt[:, br.PARENT[nd], 0] & ~labt[:, nd, 1]
            y = labt[m, nd, 0]; p = R[m, nd] > 0.5
            idx, good = qual[j]
            pick = Q[idx, nd].argmax(1)
            out[f"way{nd}"] = {"recall": round(float(p[y].mean()), 4), "false_on": round(float(p[~y].mean()), 4),
                               "walk_move_closer": round(float(good[np.arange(len(idx)), pick].mean()), 3)}
        log(f"RESULT pipeline trainer: {json.dumps(out)}")
        return
    for gw in [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["0.95"])]:
        run_gamma(gw, trainer, tr, tseq, labt, lab0, r0, r1, n, qual, updates, bs, dev, T, log)


def run_gamma(gw, trainer, tr, tseq, labt, lab0, r0, r1, n, qual, updates, bs, dev, T, log):
    torch, nn, F = dl._torch()
    enc = copy.deepcopy(trainer.model.enc)
    head = dl.MinHead(n).to(dev)
    net = nn.ModuleDict({"enc": enc, "head": head})
    tgt = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, fused=True)
    tiles, c0, c1, act, term1 = trainer.tiles, trainer.c0, trainer.c1, trainer.act, trainer.term1
    gam = torch.tensor([gw, dl.GAMMA_REACH], device=dev).view(1, 1, 2)

    def step(r):
        img0 = render(c0[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
        img1 = render(c1[r].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
        a = act[r]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qs = net["head"].both(net["enc"](img0))
            with torch.no_grad():
                q1 = tgt["head"](tgt["enc"](img1))
        q1 = torch.sigmoid(q1.float()).view(-1, n, 2, 3).max(3).values
        y = torch.where(r1[r][:, :, None], 1.0, torch.where(term1[r][:, None, None], 0.0, gam * q1))
        mk = (~r0[r]).float()[:, :, None]
        loss = 0.0
        for q in qs:
            q = q.float().view(-1, n, 2, 3)
            qa = q.gather(3, a.view(-1, 1, 1, 1).expand(-1, n, 2, 1)).squeeze(3)
            loss = loss + (F.binary_cross_entropy_with_logits(qa, y, reduction="none") * mk).sum() / mk.sum().clamp_min(1.0) / 2
        loss.backward()
        return loss.detach()
    step_c = torch.compile(step)
    rows = torch.nonzero(act <= 2).squeeze(1)
    focus = [torch.nonzero((act <= 2) & T(lab0[:, nd, 0]) & ~T(lab0[:, br.PARENT[nd], 0])).squeeze(1) for nd in (1, 2)] if "focus" in sys.argv else []
    # over-sample rows whose next frame is ready (rare), importance weights omitted: bench only
    gen = torch.Generator(device=dev).manual_seed(0)
    P, TP = list(net.parameters()), list(tgt.parameters())
    t0 = time.monotonic()
    for u in range(updates):
        if focus:
            k = bs // 2 // len(focus)
            r = torch.cat([rows[torch.randint(len(rows), (bs - k * len(focus),), device=dev, generator=gen)]]
                          + [f[torch.randint(len(f), (k,), device=dev, generator=gen)] for f in focus])
        else:
            r = rows[torch.randint(len(rows), (bs,), device=dev, generator=gen)]
        opt.zero_grad(set_to_none=True)
        loss = step_c(r)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(TP, P, 0.005)
        if u % 5000 == 0 or u == updates - 1:
            log(f"  joint {u}: loss {float(loss):.5f} {(u + 1) / (time.monotonic() - t0):.0f} upd/s")
    codes = T(tseq["codes"])
    R, WQ = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(codes), 8192):
            img = render(codes[s:s + 8192].long(), tiles).permute(0, 3, 1, 2).float() / 255.0
            q = torch.sigmoid(net["head"](net["enc"](img)).float()).view(-1, n, 2, 3)
            R.append(q[:, :, 1].max(2).values); WQ.append(q[:, :, 0])
    R = torch.cat(R).cpu().numpy(); WQ = torch.cat(WQ).cpu().numpy()
    out = {}
    for j, nd in enumerate((1, 2)):
        m = ~labt[:, br.PARENT[nd], 0] & ~labt[:, nd, 1]
        y = labt[m, nd, 0]; p = R[m, j] > 0.5
        out[f"way{nd}"] = {"share_reachable": round(float(y.mean()), 3), "accuracy": round(float((p == y).mean()), 4),
                           "recall": round(float(p[y].mean()), 4), "false_on": round(float(p[~y].mean()), 4),
                           "auc": round(dl.auc(R[m, j], y), 4)}
    for j, (idx, good) in enumerate(qual):
        pick = WQ[idx, j].argmax(1)
        out[f"way{j + 1}"]["walk_move_closer"] = round(float(good[np.arange(len(idx)), pick].mean()), 3)
        out[f"way{j + 1}"]["random_move_closer"] = round(float(good.mean()), 3)
    log(f"RESULT joint gamma_walk {gw} batch {bs}: {json.dumps(out)}")
    return
    # frozen refit on the jointly trained encoder
    def enc_all(c):
        c = T(c); Z = torch.empty(len(c), 256, dtype=torch.half, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(c), 8192):
                Z[s:s + 8192] = net["enc"](render(c[s:s + 8192].long(), tiles).permute(0, 3, 1, 2).float() / 255.0).half()
        return Z
    Z0, Z1, Zt = enc_all(tr["c0"]), enc_all(tr["c1"]), enc_all(tseq["codes"])
    read = br.train("double", Z0, Z1, act, term1, r0, r1, dev, 15000, log)
    R = read(Zt).cpu().numpy()[:, :, -1]
    for j, nd in enumerate((1, 2)):
        m = ~labt[:, br.PARENT[nd], 0] & ~labt[:, nd, 1]
        y = labt[m, nd, 0]; p = R[m, j] > 0.5
        out[f"way{nd}"] = {"recall": round(float(p[y].mean()), 4), "false_on": round(float(p[~y].mean()), 4), "auc": round(dl.auc(R[m, j], y), 4)}
    log(f"RESULT frozen refit after joint: {json.dumps(out)}")


if __name__ == "__main__":
    main()
