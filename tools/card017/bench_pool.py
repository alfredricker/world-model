"""Card 017: card 016's egocentric walking with the whole-frame summary taken as an average over
cells (what is in the frame, not where).

Stages (argv[1]): probe (the gate: can a linear read-out tell the held item from run 5's encoder
map averaged over cells?), main (criteria 1-3, learned), maps (the trained module's maps at every
cell against the exact ones).

Run: PYTHONPATH=tools/card012:tools/card013:tools/card014:tools/card016:tools/card017 bin/prun python -c \
       "import sys; sys.argv=['x','main','30000','logit','k=24']; import bench_pool; bench_pool.main()"
"""
import json, sys, time
from pathlib import Path
import numpy as np
import bench_vin as bv
import bench_ego as be

be.CARD = "017"
be.new_vin = lambda S: bv.make_vin(S.trainer.model.enc[:6], len(S.ways), be.H, be.W, fixed=S.fixed,
                                   summary="mean").to(S.dev)


def probe(S, log, pool=lambda m: m.mean((2, 3)), tag="averaged over cells", n_train=30000, n_test=10000, epochs=30):
    """Linear read-out of the held item (the code in the held row) from run 5's encoder map, averaged over
    cells; training rows to fit, held-out frames to score."""
    torch, F = S.torch, S.F
    trunk = S.trainer.model.enc[:6].eval()
    rng = np.random.default_rng(4)
    tr_rows = torch.as_tensor(np.sort(rng.choice(len(S.trainer.c0), n_train, replace=False)), device=S.dev)
    te_rows = np.sort(rng.choice(len(S.tseq["codes"]), n_test, replace=False))

    def feats(codes):
        out = []
        with torch.no_grad():
            for s in range(0, len(codes), 1024):
                out.append(pool(trunk(bv.images(codes[s:s + 1024], S.trainer.tiles)).float()))
        return torch.cat(out)

    c_tr = S.trainer.c0[tr_rows]
    c_te = torch.as_tensor(S.tseq["codes"][te_rows], device=S.dev)
    x_tr, x_te = feats(c_tr), feats(c_te)
    held_tr, held_te = c_tr[:, be.H - 1, 0].long(), c_te[:, be.H - 1, 0].long()
    classes = torch.unique(torch.cat([held_tr, held_te]))
    y_tr, y_te = torch.searchsorted(classes, held_tr), torch.searchsorted(classes, held_te)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    lin = torch.nn.Linear(x_tr.shape[1], len(classes)).to(S.dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(epochs):
        for s in torch.randperm(n_train, device=S.dev).split(1024):
            loss = F.cross_entropy(lin(x_tr[s]), y_tr[s])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = lin(x_te).argmax(1)
    res = {"held_item_right": round(float((pred == y_te).float().mean()), 4),
           "per_item": {str(int(classes[k])): {"frames": int((y_te == k).sum()),
                                               "right": round(float((pred[y_te == k] == k).float().mean()), 4)}
                        for k in range(len(classes)) if (y_te == k).any()}}
    log(f"RESULT probe, held item from the map {tag}: {json.dumps(res)}")
    return res


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "probe"
    updates = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30000
    if stage != "probe":
        return be.main()
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    S = bv.Setup(log)
    be.to_ego(S)
    res = {"averaged": probe(S, log)}
    # for comparison, not the gate: the held tile alone, the average plus the held tile, the old flattening
    res["held_tile"] = probe(S, log, lambda m: m[:, :, be.H - 1, 0], "at the held tile only")
    res["average_plus_held_tile"] = probe(S, log, lambda m: S.torch.cat([m[:, :, :be.H - 1].mean((2, 3)), m[:, :, be.H - 1, 0]], 1),
                                          "averaged over the room, plus the held tile")
    res["flattened"] = probe(S, log, lambda m: m.flatten(1), "flattened (cards 014, 016)")
    out = Path(f"runs/017_probe.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
