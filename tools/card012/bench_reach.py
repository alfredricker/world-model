"""Run: PYTHONPATH=tools/card012 bin/prun python -c "import sys; sys.argv=[...]; import <module>; <module>.main()".
Bench: learned reach ("a ready state can be reached by walking") vs exact.

Switch world. Encoder: card 012's trainer pretrained on the goal-square signal.
Ready frames: exact (simulator), so only the reach learner is tested.
Ways: 1 = forward into goal square (condition ~ door open),
      2 = toggle that makes way 1's condition come on (condition ~ switch reachable & off).
Learners: plain (1 net, gamma .99), double (clipped min of 2, gamma .99),
          fh (fixed-horizon, De Asis et al. 2020: R_h from R_{h-1}, no discount).
Scored on held-out episodes (all frames, off-parent): accuracy, recall, false-on rate, AUC.
"""
import copy, json, multiprocessing as mp, sys, time
import numpy as np
from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld

H = 32
WORLD = "key" if "key" in sys.argv else "switch"
PARENT, ACTION = [-1, 0, 1], [-1, ld.FORWARD, ld.TOGGLE]


def _labels(job):
    layouts, ep, arrays = job
    out = [np.zeros((len(ep), 3, 2), bool) for _ in arrays]   # [:, n, 0]=holds, [:, n, 1]=ready(way n)
    cur = ex = None
    for i in range(len(ep)):
        if ep[i] != cur:
            cur = ep[i]; ex = dl.Exact(layouts[cur], PARENT, ACTION)
        for o, st in zip(out, arrays):
            s = dl.to_tup(st[i])
            for n in range(3):
                o[i, n, 0] = ex.holds(n, s)
                if n:
                    o[i, n, 1] = ex.ready(PARENT[n], ACTION[n], s)
    return out


def labels(pool, layouts, ep, arrays, jobs=160):
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    sel = starts[np.linspace(0, len(starts) - 1, min(jobs, len(starts))).astype(int)]
    b = sorted(set(sel.tolist()) | {len(ep)})
    tasks = [({int(e): layouts[e] for e in np.unique(ep[x:y])}, ep[x:y], [a[x:y] for a in arrays]) for x, y in zip(b, b[1:])]
    parts = pool.map(_labels, tasks)
    return [np.concatenate([p[k] for p in parts]) for k in range(len(arrays))]


def dist_labels(layouts, seq):
    out = np.full((len(seq["ep"]), 2), -1)
    cur = ex = None
    for i, (e, st) in enumerate(zip(seq["ep"], seq["st"])):
        if e != cur:
            cur = e; ex = dl.Exact(layouts[e], PARENT, ACTION)
        s = dl.to_tup(st)
        for j, n in enumerate((1, 2)):
            if ex.holds(n, s) and not ex.holds(PARENT[n], s):
                out[i, j] = ex.distances(n, s).get(s[:3], -1)   # walking steps to a ready state
    return out


def train(mode, Z0, Z1, act, term1, ready0, ready1, dev, updates, log, batch=4096, lr=3e-4):
    torch, nn, F = dl._torch()
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(0)
    n = ready0.shape[1]
    K = H if mode == "fh" else 1
    nets = 1 if mode == "plain" else 2 if mode == "double" else 1

    def mk():
        m = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n * K * 3))
        with torch.no_grad():
            m[-1].bias.fill_(-4.0)
        return m
    head = nn.ModuleList(mk() for _ in range(nets)).to(dev)
    target = copy.deepcopy(head).requires_grad_(False)
    opt = torch.optim.Adam(head.parameters(), lr=lr, fused=True)
    rows = torch.nonzero(act <= 2).squeeze(1)
    P, TP = list(head.parameters()), list(target.parameters())
    t0 = time.monotonic()
    for u in range(updates):
        r = rows[torch.randint(len(rows), (batch,), device=dev, generator=gen)]
        a = act[r]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qs = [h(Z0[r]) for h in head]
            with torch.no_grad():
                t1 = [h(Z1[r]) for h in target]
        with torch.no_grad():
            q1 = torch.stack([torch.sigmoid(t.float()) for t in t1]).min(0).values.view(-1, n, K, 3).max(3).values
            if mode == "fh":   # horizon h bootstraps from h-1; horizon 1: only an immediate ready state counts
                boot = torch.cat([torch.zeros_like(q1[:, :, :1]), q1[:, :, :-1]], 2)
            else:
                boot = 0.99 * q1
            y = torch.where(ready1[r][:, :, None], 1.0, torch.where(term1[r][:, None, None], 0.0, boot))
        m = (~ready0[r]).float()[:, :, None]
        loss = 0.0
        for q in qs:
            q = torch.sigmoid(q.float()).view(-1, n, K, 3)
            qa = q.gather(3, a.view(-1, 1, 1, 1).expand(-1, n, K, 1)).squeeze(3)
            loss = loss + ((qa - y) ** 2 * m).sum() / m.sum().clamp_min(1) / K / len(qs)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(TP, P, 0.01)
        if u % 10000 == 0 or u == updates - 1:
            log(f"  {mode} {u}: loss {float(loss):.5f} {(u + 1) / (time.monotonic() - t0):.0f} upd/s")

    def read(Z):
        with torch.no_grad():
            v = torch.stack([torch.sigmoid(h(Z.float())) for h in head]).min(0).values.view(-1, n, K, 3).max(3).values
        return v   # (N, n, K)
    return read


def main():
    updates = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    pool = mp.get_context("fork").Pool(20)
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:5.0f}s] {m}", flush=True)
    layouts, tr, seq, st0 = dl.collect(pool, WORLD, 30000, 11, 0.0)
    tl, _, tseq, _ = dl.collect(pool, WORLD, 300, 999, 0.0, seq_episodes=300)
    log(f"data {len(tr['act'])} rows; test frames {len(tseq['act'])}")
    lab0, lab1 = labels(pool, layouts, tr["ep"], [tr["s0"], tr["s1"]])
    (labt,) = labels(pool, tl, tseq["ep"], [tseq["st"]])
    log("labels done")
    torch, _, _ = dl._torch()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    T = lambda x: torch.from_numpy(x).to(dev)
    trainer = dl.Trainer(dl.make_model(), tr, dev)
    L0 = torch.zeros(len(tr["act"]), dl.K_MAX, dtype=torch.bool, device=dev)
    L1 = L0.clone(); L1[:, 0] = trainer.term1
    trainer.train(L0, L1, [0], 30000, log)
    # as in the procedure: fine-tune on the level-1 goal (here its exact labels) before level-2 values
    L0[:, 1], L1[:, 1] = T(lab0[:, 1, 0]), T(lab1[:, 1, 0])
    if "nofinetune" not in sys.argv:
        trainer.train(L0, L1, [0, 1], 12000, log)
    Z0, Z1, Zt = trainer.encode(tr["c0"]), trainer.encode(tr["c1"]), trainer.encode(tseq["codes"])
    r0, r1 = T(lab0[:, 1:, 1]), T(lab1[:, 1:, 1])
    if "learned_ready" in sys.argv:   # ready = the agent's own achievement head > 0.5 (goal off), as in run_learned
        P0, P1 = trainer.achieve(Z0), trainer.achieve(Z1)
        h0, h1 = T(lab0[:, :, 0]), T(lab1[:, :, 0])
        l0 = torch.stack([(P0[:, 0, ld.FORWARD] > 0.5) & ~h0[:, 0], (P0[:, 1, ld.TOGGLE] > 0.5) & ~h0[:, 1]], 1)
        l1 = torch.stack([(P1[:, 0, ld.FORWARD] > 0.5) & ~h1[:, 0], (P1[:, 1, ld.TOGGLE] > 0.5) & ~h1[:, 1]], 1)
        for j in range(2):
            e, g = r0[:, j], l0[:, j]
            log(f"learned ready way{j + 1}: exact {int(e.sum())} learned {int(g.sum())} both {int((e & g).sum())}")
        r0, r1 = l0, l1
        del P0, P1
    res = {}
    labt_d = dist_labels(tl, tseq)
    for mode, updates in [(m, u) for u in ((15000,) if "learned_ready" in sys.argv else (15000, 30000)) for m in ("plain", "double", "fh")]:
        t0 = time.monotonic()
        read = train(mode, Z0, Z1, trainer.act, trainer.term1, r0, r1, dev, updates, log)
        V = read(Zt).cpu().numpy()                     # (N, 2, K)
        R = V[:, :, -1]
        out = {"seconds": round(time.monotonic() - t0, 1)}
        for j, n in enumerate((1, 2)):
            m = ~labt[:, PARENT[n], 0] & ~labt[:, n, 1]   # parent off, not already ready
            y = labt[m, n, 0]
            p = R[m, j] > 0.5
            d = {"frames": int(m.sum()), "share_reachable": round(float(y.mean()), 3),
                 "accuracy": round(float((p == y).mean()), 4), "recall": round(float(p[y].mean()), 4),
                 "false_on": round(float(p[~y].mean()), 4), "auc": round(dl.auc(R[m, j], y), 4),
                 "reach_unreachable_median_p90": [round(float(np.median(R[m, j][~y])), 3), round(float(np.quantile(R[m, j][~y], .9)), 3)]}
            if mode == "fh":   # distance estimate: first horizon on vs exact BFS distance
                on = V[m, j, :] > 0.5
                est = np.where(on.any(1), on.argmax(1) + 1, -1)
                dd = labt_d[m, j]
                d["distance_mean_abs_error_reachable"] = round(float(np.abs(est - dd)[y & (est > 0)].mean()), 2)
                d["recall_by_distance"] = {f"{a}-{b}": round(float(p[y & (dd >= a) & (dd <= b)].mean()), 3) for a, b in ((0, 4), (5, 9), (10, 14), (15, 40)) if (y & (dd >= a) & (dd <= b)).any()}
                d["est_distance_on_reachable_median"] = float(np.median(est[y & (est > 0)])) if (y & (est > 0)).any() else None
            out[f"way{n}"] = d
        res[f"{mode}_{updates}"] = out
        log(f"RESULT {mode} {updates}: {json.dumps(out)}")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
