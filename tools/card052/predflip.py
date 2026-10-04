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


def collect_strat(seed, n_mem=300, n_probe=40, steps=400000):
    """Step 2d: per kind of try, up to n_probe probe tries first, then up to n_mem memory tries."""
    s = EF.Stream(seed)
    mem, probe = defaultdict(list), defaultdict(list)
    for _ in range(steps):
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
            k = torch.exp(-(torch.abs(Xp[:, None] - Xm[None]) * lam).sum(-1))
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


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    EF.START_SHARE = float(get("--starts", "0"))      # the tries come from the same stream as training
    chk = Check(int(get("--seed", "399")), strat=get("--strat", "0") == "1")
    EF.CHECK = chk
    EF.main()


if __name__ == "__main__":
    main()
