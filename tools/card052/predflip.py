"""Card 052, step 2c: drift measured by recall's predictions (the card's second rate), on step 2b's plain arm.

Fixed tries of pick up, drop and toggle, rendered once with nuisance from separate episodes: a memory set (up to
2,000 with a change and 2,000 without, per action) and a probe set (up to 200 and 200). At every checkpoint:
  - recall by codes: memory keyed by (action, front code tuple, held code tuple) under that checkpoint's codes; a
    probe try is predicted as its key's most frequent outcome, or "unknown" (-1) if no memory try has its key;
  - recall by vectors: step 2b's leave-one-out recall, the memory tries as voters, lambda as learned;
  - per recall: the prediction flip rate (the share of probe tries whose prediction differs from the previous
    checkpoint's) and accuracy ("unknown" is wrong).
Computed once: the same recall keyed by the evaluator's identities (upper bound) and each action's most frequent
outcome (trivial baseline).

  bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --mu 0.1 --restart 100000000 \
      --out runs/052/predflip_399.json
Step 2d: --starts 0.5 (play starts in the stream, effect.py) --strat 1 (tries stratified by kind of try).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from minigrid.core.constants import IDX_TO_OBJECT

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effect as EF                                    # noqa: E402

fn = torch.nn.functional
COLLECT_STEPS = 400000


def collect(seed, n_mem=2000, n_probe=200, steps=400000):
    """Per action: memory and probe tries (front pixels, held pixels, outcome, identities), from one stream of
    separate episodes; each try goes to the probe set first until it is full."""
    s = EF.Stream(seed)
    mem = {a: ([], []) for a in EF.ACTS}
    probe = {a: ([], []) for a in EF.ACTS}
    for _ in range(steps):
        s.step()
        for a, f, h, o, ids in s.tries:
            c = int(o > 0)
            if len(probe[a][c]) < n_probe:
                probe[a][c].append((f, h, o, ids))
            elif len(mem[a][c]) < n_mem:
                mem[a][c].append((f, h, o, ids))
        if all(len(mem[a][c]) >= n_mem for a in EF.ACTS for c in (0, 1)):
            break
    flat = lambda d: {a: d[a][0] + d[a][1] for a in EF.ACTS}
    return flat(mem), flat(probe)


def kind_of(a, ids):
    """Step 2d's kind of try (evaluator side, for stratifying and reporting): the action, the front tile's kind
    (and door state), the held tile's kind and whether its colour matches the front's."""
    k0, c0 = ids
    front = "none" if k0 is None else IDX_TO_OBJECT[k0[0]] + (f"/{k0[2]}" if IDX_TO_OBJECT[k0[0]] == "door" else "")
    held = "none" if c0 is None else IDX_TO_OBJECT[c0[0]] + ("" if k0 is None else ("/same" if k0[1] == c0[1] else "/other"))
    return (a, front, held)


COLOUR = {"locked, same key": (5, "door/2", "key/same"), "locked, other key": (5, "door/2", "key/other")}


def collect_strat(seed, n_mem=300, n_probe=40, steps=None):
    """Step 2d: per kind of try, up to n_probe probe tries first, then up to n_mem memory tries."""
    s = EF.Stream(seed)
    mem, probe = defaultdict(list), defaultdict(list)
    for _ in range(steps or COLLECT_STEPS):
        s.step()
        for a, f, h, o, ids in s.tries:
            q = kind_of(a, ids)
            if len(probe[q]) < n_probe:
                probe[q].append((f, h, o, ids))
            elif len(mem[q]) < n_mem:
                mem[q].append((f, h, o, ids))
    flat = lambda d: {a: [t for q in sorted(d, key=str) if q[0] == a for t in d[q]] for a in EF.ACTS}
    return flat(mem), flat(probe)


def table(mem, keyf):
    out = {}
    for a in EF.ACTS:
        cnt = defaultdict(Counter)
        for i, t in enumerate(mem[a]):
            cnt[keyf(a, i)][t[2]] += 1
        out[a] = {k: c.most_common(1)[0][0] for k, c in cnt.items()}
    return out


def code_tuples(enc, tries):
    c, _ = enc.codes(np.stack([t[0] for t in tries] + [t[1] for t in tries]))
    n = len(tries)
    return [tuple(r) for r in c[:n].tolist()], [tuple(r) for r in c[n:].tolist()]


class Check:
    def __init__(self, seed, strat=False):
        self.mem, self.probe = (collect_strat if strat else collect)(seed + 555)
        self.kinds = np.array([str(kind_of(a, t[3])) for a in EF.ACTS for t in self.probe[a]])
        self.prev = {"codes": None, "vectors": None}
        truth = [t[2] for a in EF.ACTS for t in self.probe[a]]
        self.truth = np.array(truth)
        ids = table(self.mem, lambda a, i: self.mem[a][i][3])
        pred = [ids[a].get(t[3], -1) for a in EF.ACTS for t in self.probe[a]]
        major = {a: Counter(t[2] for t in self.mem[a]).most_common(1)[0][0] for a in EF.ACTS}
        self.fixed = {
            "identity_keyed_accuracy": round(float(np.mean(np.array(pred) == self.truth)), 4),
            "trivial_accuracy": round(float(np.mean(np.array([major[a] for a in EF.ACTS for _ in self.probe[a]]) == self.truth)), 4),
            "memory_tries": {int(a): [sum(t[2] > 0 for t in self.mem[a]), sum(t[2] == 0 for t in self.mem[a])] for a in EF.ACTS},
            "colour_probe_tries": {n: int((self.kinds == str(q)).sum()) for n, q in COLOUR.items()},
            "colour_memory_tries": {n: sum(kind_of(a, t[3]) == q for a in EF.ACTS for t in self.mem[a]) for n, q in COLOUR.items()},
            "probe_tries": {int(a): [sum(t[2] > 0 for t in self.probe[a]), sum(t[2] == 0 for t in self.probe[a])] for a in EF.ACTS}}
        print(self.fixed, flush=True)

    @torch.no_grad()
    def vectors(self, enc):
        pred = []
        for ai, a in enumerate(EF.ACTS):
            m, p = self.mem[a], self.probe[a]
            Xm = torch.cat([enc.keys([t[0] for t in m[i:i + 1000]], [t[1] for t in m[i:i + 1000]]) for i in range(0, len(m), 1000)])
            Xp = enc.keys([t[0] for t in p], [t[1] for t in p])
            R = fn.one_hot(torch.as_tensor([t[2] for t in m], device=enc.dev), EF.NK).float()
            lam = fn.softplus(enc.theta[ai])
            for j in range(0, len(Xp), 100):          # in chunks of probe tries (GPU memory)
                k = torch.exp(-(torch.abs(Xp[j:j + 100, None] - Xm[None]) * lam).sum(-1))
                P = (k @ R + 1.0 / EF.NK) / (k.sum(-1, keepdim=True) + 1.0)
                pred += P.argmax(1).cpu().numpy().tolist()
        return np.array(pred)

    def codes(self, enc):
        keys_m = {a: code_tuples(enc, self.mem[a]) for a in EF.ACTS}
        tab = table(self.mem, lambda a, i: (keys_m[a][0][i], keys_m[a][1][i]))
        pred = []
        for a in EF.ACTS:
            f, h = code_tuples(enc, self.probe[a])
            pred += [tab[a].get(k, -1) for k in zip(f, h)]
        return np.array(pred)

    def __call__(self, enc):
        out = dict(self.fixed) if self.prev["codes"] is None else {}
        for name, pred in (("codes", self.codes(enc)), ("vectors", self.vectors(enc))):
            prev = self.prev[name]
            out[f"pred_flip_{name}"] = None if prev is None else round(float(np.mean(pred != prev)), 4)
            out[f"accuracy_{name}"] = round(float(np.mean(pred == self.truth)), 4)
            if name == "codes":
                out["unknown_codes"] = round(float(np.mean(pred == -1)), 4)
            for n, q in COLOUR.items():                 # the colour tries
                m = self.kinds == str(q)
                if m.any():
                    out[f"accuracy_{name}: {n}"] = round(float(np.mean(pred[m] == self.truth[m])), 4)
                    if prev is not None:
                        out[f"pred_flip_{name}: {n}"] = round(float(np.mean(pred[m] != prev[m])), 4)
            out[f"accuracy_per_kind_{name}"] = {k: round(float(np.mean(pred[self.kinds == k] == self.truth[self.kinds == k])), 3)
                                                for k in sorted(set(self.kinds))}
            self.prev[name] = pred
        return out


class ColourCheck:
    """Step 2e (report only, evaluator labels): can a linear classifier read colour (and kind) from the encoder?
    On step 2a's probe tiles that have a colour (keys, balls, boxes, doors, switches): multinomial logistic
    regression, 300 Adam steps, fitted on alternate tiles and tested on the rest; from the whole vector, each part's
    vector and each part's code (one-hot)."""

    KINDS = ("key", "ball", "box", "door", "switch")

    def __init__(self, seed):
        tiles, ids = EF.DR.probe_set(seed)
        keep = [i for i, q in enumerate(ids) if q is not None and IDX_TO_OBJECT[q[0]] in self.KINDS]
        self.tiles = tiles[keep]
        self.colour = np.array([ids[i][1] for i in keep])
        self.kind = np.array([self.KINDS.index(IDX_TO_OBJECT[ids[i][0]]) for i in keep])
        self.fit_ix, self.test_ix = np.arange(len(keep))[0::2], np.arange(len(keep))[1::2]

    def classify(self, X, y, dev):
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        X = (X - X[self.fit_ix].mean(0)) / (X[self.fit_ix].std(0) + 1e-6)
        _, yy = np.unique(y, return_inverse=True)
        yt = torch.as_tensor(yy, device=dev)
        W = torch.zeros(X.shape[1], int(yy.max()) + 1, device=dev, requires_grad=True)
        b = torch.zeros(int(yy.max()) + 1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.05)
        fi = torch.as_tensor(self.fit_ix, device=dev)
        for _ in range(300):
            loss = fn.cross_entropy(X[fi] @ W + b, yt[fi]) + 1e-4 * (W ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        ti = torch.as_tensor(self.test_ix, device=dev)
        with torch.no_grad():
            return round(float(((X[ti] @ W + b).argmax(1) == yt[ti]).float().mean()), 3)

    def __call__(self, enc):
        with torch.no_grad():
            z = torch.cat([enc.pieces(enc.x(self.tiles[i:i + 1000])) for i in range(0, len(self.tiles), 1000)])
            _, idx = enc.quant(z)
        z, idx = z.cpu().numpy(), idx.cpu().numpy()
        out = {}
        for name, y in (("colour", self.colour), ("kind", self.kind)):
            with torch.enable_grad():
                out[f"{name}_from_vector"] = self.classify(z.reshape(len(z), -1), y, enc.dev)
                out[f"{name}_from_part_vector"] = [self.classify(z[:, k], y, enc.dev) for k in range(EF.K)]
                out[f"{name}_from_part_code"] = [self.classify(np.eye(EF.M)[idx[:, k]], y, enc.dev) for k in range(EF.K)]
            out[f"{name}_chance"] = round(float(np.bincount(np.unique(y, return_inverse=True)[1][self.test_ix]).max() / len(self.test_ix)), 3)
        return out


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    if get("--colours", None):                        # step 2f: the old setting (TRAIN is shared by reference)
        EF.DR.GN.TRAIN[:] = get("--colours", None).split(",")
        EF.STARTS[:] = [st for st in EF.STARTS if st[2] in EF.DR.GN.TRAIN]
    EF.DR.GN.TINT = float(get("--tint", str(EF.DR.GN.TINT)))
    EF.DR.GN.NOISE = float(get("--noise", str(EF.DR.GN.NOISE)))
    global COLLECT_STEPS
    COLLECT_STEPS = int(get("--collect-steps", "400000"))
    EF.START_SHARE = float(get("--starts", "0"))      # the tries come from the same stream as training
    chk = Check(int(get("--seed", "399")), strat=get("--strat", "0") == "1")
    EF.CHECK = chk
    if get("--colour", "0") == "1":                   # step 2e: the colour check as well
        cc = ColourCheck(int(get("--seed", "399")))
        EF.CHECK = lambda enc: {**chk(enc), **cc(enc)}
    EF.main()


if __name__ == "__main__":
    main()
