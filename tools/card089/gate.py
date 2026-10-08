"""Card 089's gate, per pooling arm: card 054's margins at the end of training; card 087's property ensemble retrained
on the new vectors and scored on part (i); card 088's leftover and card 069's probe.

  bin/prun python tools/card089/gate.py meanmax       writes runs/089/gate_meanmax.json and ensemble_meanmax.pt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
POOL = sys.argv[1]
sys.path.insert(0, str(ROOT / "tools" / "card088"))
import leftover as L                                   # noqa: E402  (card 069's probe, card 087's props; tiers first)
sys.path.insert(0, str(ROOT / "tools" / "card089"))
import encoder as E89                                  # noqa: E402
E89.install(POOL)
PR, G69 = L.PR, L.G69
OUT = ROOT / "runs" / "089"
ENC = OUT / f"encoder_{POOL}.pt"
PR.ENCODER = ENC
PR._ENC.clear()


def margins():
    """The last checkpoint's margins and effects, against card 054's (runs/054/b_m0.5_399.json)."""
    keys = ("pair_margin_last", "visibility_last", "effect", "loo_loglik_per_action")
    new = json.loads((OUT / f"encoder_{POOL}.json").read_text())["rows"][-1]
    ref = json.loads((ROOT / "runs" / "054" / "b_m0.5_399.json").read_text())["rows"][-1]
    return {k: {"this": new.get(k), "card 054": ref.get(k)} for k in keys} | {
        "colour_from_part_vector": new.get("colour_from_part_vector"), "kind_from_part_vector": new.get("kind_from_part_vector")}


def main():
    rep = {"note": "Card 089 gate, tools/card089/gate.py", "pool": POOL, "training (last checkpoint)": margins()}
    rng = np.random.default_rng(870)                   # card 087's draws
    train = PR.play(20000, rng)
    Ztr = PR.encode(train["tiles"])
    named, ntiles = PR.named_set()
    Zn = PR.encode(ntiles)
    nets = PR.fit(Ztr, train)
    right, spread, _ = PR.score(nets, Zn, named)
    rep["(i) MiniGrid's tiles, nine colours"] = {k: f"{int(v.sum())} of {len(v)}" for k, v in right.items()}
    torch.save([n.state_dict() for n in nets], OUT / f"ensemble_{POOL}.pt")
    # card 088's leftover on these vectors
    D = PR.play(20000, np.random.default_rng(880))
    Z = PR.encode(D["tiles"])
    W, Pp, Tp = PR.predict(nets, Z)
    profile = np.array([f"{w}{int(p > 0.5)}{int(t > 0.5)}" for w, p, t in zip(W.mean(0).argmax(1), Pp.mean(0), Tp.mean(0))])
    G = L.sensitivity(nets, Z)
    C = L.within(Z, profile)
    mu, V = L.leftover(C, G)
    d = np.diag(G)
    rep["share of what the properties read (G's trace), per part"] = [round(float(d[8 * k:8 * k + 8].sum() / d.sum()), 4)
                                                                       for k in range(4)]
    Z6 = PR.encode(G69.tiles(G69.SIX))
    Z9 = PR.encode(G69.tiles(G69.SIX + G69.UNSEEN))
    for r in L.RANKS:
        rep[f"leftover, rank {r}"] = L.probe(V[:, :r].T, Z6, Z9)
        rep[f"leftover, rank {r}, weight on the made-of parts"] = round(float((V[16:, :r] ** 2).sum() / (V[:, :r] ** 2).sum()), 4)
    col = np.isin(D["kind"], PR.COLOURED)
    H = np.c_[D["hue"][col] / 255.0, np.ones(col.sum())]
    for name, sl in (("arrangement parts", slice(0, 16)), ("made-of parts", slice(16, 32))):
        Y = Z[col][:, sl]
        beta, *_ = np.linalg.lstsq(H, Y, rcond=None)
        rep[f"hue's share of the variance, {name} (coloured kinds)"] = round(float(1 - (Y - H @ beta).var(0).sum() / Y.var(0).sum()), 4)
    ang = {}
    for kind in ("key", "locked"):
        m = D["kind"] == kind
        _, Vk = L.leftover(L.within(Z[m], profile[m]), G)
        ang[kind] = np.linalg.qr(Vk[:, :2])[0]
    rep["principal angles (degrees), keys' and locked doors' top-2 leftover"] = [
        round(float(np.degrees(np.arccos(np.clip(s, -1, 1)))), 1) for s in np.linalg.svd(ang["key"].T @ ang["locked"])[1]]
    np.savez(OUT / f"P_{POOL}.npz", V=V, mu=mu, G=G, C=C)
    (OUT / f"gate_{POOL}.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
