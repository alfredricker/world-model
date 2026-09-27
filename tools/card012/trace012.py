"""Card 012 diagnostic: run the learned procedure (run 2 settings) on one world
and, after each depth, compare every new learned condition with its exact
meaning (the same tree evaluated by the simulator), on
  - training rows (what the next fine-tune learns from), before and after the walking rule;
  - 300 held-out episodes (computed the way the pipeline computes its own frames);
and at the end with the final value refit (what acting and criterion 1 use).
"""
import argparse, json, multiprocessing as mp, sys, time
import numpy as np
from worldmodel import discover_logic as dl

TRAIN_EPS = 3000    # training rows scored: those of the first 3000 episodes
TEST_STRIDE = 3     # every 3rd frame of the held-out episodes


def _job(job):
    layouts, parent, action, ep, states, nodes = job
    H = np.zeros((len(ep), len(nodes)), bool)
    Rd = np.zeros((len(ep), len(nodes)), bool)
    cur = ex = None
    for i in range(len(ep)):
        if ep[i] != cur:
            cur = ep[i]; ex = dl.Exact(layouts[cur], parent, action)
        s = dl.to_tup(states[i])
        for k, n in enumerate(nodes):
            H[i, k] = ex.holds(n, s)
            Rd[i, k] = ex.ready(parent[n], action[n], s)
    return H, Rd


def exact(pool, layouts, tree, ep, states, nodes, jobs=120):
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    sel = starts[np.linspace(0, len(starts) - 1, min(jobs, len(starts))).astype(int)]
    b = sorted(set(sel.tolist()) | {len(ep)})
    tasks = [({int(e): layouts[e] for e in np.unique(ep[x:y])}, list(tree.parent), list(tree.action), ep[x:y], states[x:y], nodes)
             for x, y in zip(b, b[1:])]
    parts = pool.map(_job, tasks)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def score(learned, ex, parent_off):
    m = parent_off
    y, p = ex[m], learned[m]
    return {"frames": int(m.sum()), "exact_on": round(float(y.mean()), 4), "learned_on": round(float(p.mean()), 4),
            "recall": round(float(p[y].mean()), 3) if y.any() else None,
            "false_on": round(float(p[~y].mean()), 4) if (~y).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="key")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-depth", type=int, default=99)
    a = ap.parse_args()
    pool = mp.get_context("fork").Pool(20)
    t00 = time.monotonic()
    out = {"world": a.world, "stages": []}
    logf = open(a.out + ".log", "w")

    def log(m):
        m = f"[{time.monotonic() - t00:5.0f}s] {m}"
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    play = 1 / 3 if a.world == "both" else 0.0
    layouts, tr, seq, st0 = dl.collect(pool, a.world, 30000, 11, play)
    tl, _, tseq, _ = dl.collect(pool, a.world, 300, 999, play, seq_episodes=300)
    ti = np.arange(0, len(tseq["ep"]), TEST_STRIDE)
    t_ep, t_st, t_codes = tseq["ep"][ti], tseq["st"][ti], tseq["codes"][ti]
    rows = np.flatnonzero(tr["ep"] < TRAIN_EPS)
    log(f"train rows scored {len(rows)}, test frames {len(ti)}")

    torch, _, _ = dl._torch()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    trainer = dl.Trainer(dl.make_model(), tr, dev)
    root_t = torch.tensor([(s[0], s[1]) == tl[e].goal for e, s in zip(t_ep, t_st)], device=dev)
    Lt = torch.zeros(len(ti), dl.K_MAX, dtype=torch.bool, device=dev)
    Lt[:, 0] = root_t
    cache = {}   # node -> (train exact s0, test exact holds, test exact ready)

    def ex_of(tree, nodes):
        need = [n for n in nodes if n not in cache]
        if need:
            h0, _ = exact(pool, layouts, tree, tr["ep"][rows], tr["s0"][rows], need)
            ht, rt = exact(pool, tl, tree, t_ep, t_st, need)
            for k, n in enumerate(need):
                cache[n] = (h0[:, k], ht[:, k], rt[:, k])

    def trace(stage, tree, nodes, head, L0, L1):
        t0 = time.monotonic()
        ex_of(tree, [0] + list(nodes))
        L0r = L0[torch.from_numpy(rows).to(dev)].cpu().numpy()
        rec = {"stage": stage, "nodes": {}}
        if stage == "before_walking_rule":      # extend held-out labels exactly as the pipeline does for its own frames
            Zt = trainer.encode(t_codes)
            P = trainer.achieve(Zt)
            _, R, _ = dl.values_of(head, Zt, dl.VAL_SLOTS, with_q=False)
            for j, c in enumerate(nodes):
                g, act = tree.parent[c], tree.action[c]
                ready = (P[:, g, act] > 0.5) & ~Lt[:, g]
                Lt[:, c] = Lt[:, g] | ready | (R[:, c] > 0.5)
                rd = ready.cpu().numpy()
                cache[c] = cache[c] + (rd,)
        if stage == "final":
            lt = dl.LearnedTree(tree, trainer, head, {c: c for c in nodes})
            Lf = lt.evaluate(trainer.encode(t_codes), root_t)["L"].cpu().numpy()
        Ltn = Lt.cpu().numpy()
        for c in nodes:
            g = tree.parent[c]
            e0, et, er = cache[c][:3]
            eg0, egt = cache[g][0] if g else None, cache[g][1]
            d = {"path": "/".join(tree.path(c)), "status": tree.status[c]}
            # training rows: rows where the exact parent is off
            off0 = ~eg0 if g else np.ones(len(rows), bool)
            d["train_rows"] = score(L0r[:, c], e0, off0)
            offt = ~egt
            if stage == "final":
                d["heldout_final"] = score(Lf[:, c], et, offt)
            else:
                d["heldout"] = score(Ltn[:, c], et, offt)
                if len(cache[c]) > 3:
                    d["ready_heldout"] = score(cache[c][3], er, offt)
            rec["nodes"][c] = d
            log(f"{stage} {d['path']:<28} " + json.dumps({k: v for k, v in d.items() if k not in ('path',)}))
        out["stages"].append(rec)
        log(f"  trace {stage}: {time.monotonic() - t0:.0f}s")
        json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
        if stage == "after_walking_rule" and tree.depth[nodes[0]] - 1 >= a.max_depth:
            raise SystemExit("stopped at max depth")

    args = argparse.Namespace(pretrain=30000, finetune=12000, value_updates=15000)
    tree = dl.Tree()
    lt, persist, timings = dl.run_learned(tree, trainer, tr, seq, st0, layouts, args, log, trace=trace)
    out["tree"] = tree.report()
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
