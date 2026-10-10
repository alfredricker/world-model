"""Card 098.1: Crafter interactions predicted by recall over card 098.2's tokens (offline).

Codes for every frame (card 098.2's encoder and prototypes). The place acted on: for each code of the agent's own
tile, the neighbouring place whose code most often changes after "do". Events: interaction steps (do, sleep, place,
make); key: action, front code, the codes of the twelve non-vital inventory slots; outcome: the front code after and
those slots' codes after. Recall: the most common outcome among stored events (workers 0-7) with the same key, else
the same action and front code, else "nothing changes". Judged on workers 8-9. Upper bound: the same recall on the
evaluator's true state (front class, true counts).

  bin/prun python tools/card098.1/interact.py      → runs/098.1/results.json
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card098"))
import encoder as EN                                   # noqa: E402

OUT = ROOT / "runs" / "098.1"
dev = EN.dev
CENTRE = 31
NB = {22: (0, -1), 40: (0, 1), 30: (-1, 0), 32: (1, 0)}   # neighbour place -> (dx, dy) facing
ACTS = {5: "do", 6: "sleep", 7: "place stone", 8: "place table", 9: "place furnace", 10: "place plant",
        11: "make wood pickaxe", 12: "make stone pickaxe", 13: "make iron pickaxe", 14: "make wood sword",
        15: "make stone sword", 16: "make iron sword"}
NONVITAL = slice(4, 16)
RELATIVE = "--relative" in sys.argv                    # revision: outcomes as changes per part


@torch.no_grad()
def all_codes(T, enc, protos):
    P = torch.as_tensor(protos, device=dev)
    out = np.zeros(T.shape[:2], np.int16)
    for s in range(0, len(T), 2000):
        E = torch.as_tensor(EN.embed(enc, T[s:s + 2000]), device=dev)
        out[s:s + 2000] = torch.cdist(E.reshape(-1, E.shape[-1]), P).argmin(1).reshape(len(E), 81).cpu().numpy()
    return out


def recall(train, test):
    """train/test: lists of (key, key1, outcome). Most common outcome at the full key, else at key1, else None
    (nothing changes). Returns per test event (prediction, stored count at the level used)."""
    full, part = defaultdict(Counter), defaultdict(Counter)
    for k, k1, o in train:
        full[k][o] += 1
        part[k1][o] += 1
    res = []
    for k, k1, o in test:
        c = full.get(k) or part.get(k1)
        res.append((c.most_common(1)[0][0] if c else None, sum(c.values()) if c else 0))
    return res


def recall_parts(train, test):
    """Revision: outcomes as changes per part. Event: (action, own code, front code, slot codes (12), front after,
    slots after (12)). Per part: does it change (majority among stored events with the same action, own code, front
    code and slot codes; else the same action, own code and front code; else no), and into what: the front's most
    common code after among those matched events that changed it; a slot's most common code after among all stored
    events of the same action in which the same slot showing the same code changed (its learned successor)."""
    full, part, succ = defaultdict(list), defaultdict(list), defaultdict(Counter)
    for e in train:
        a, own, f, sl, fa, sa = e
        full[(a, own, f, sl)].append(e)
        part[(a, own, f)].append(e)
        for k, (b, c) in enumerate(zip(sl, sa)):
            if b != c:
                succ[(a, k, b)][c] += 1                 # a slot is a token with a where: its position
    res = []
    for a, own, f, sl, fa, sa in test:
        m = full.get((a, own, f, sl)) or part.get((a, own, f)) or []
        n = len(m)
        if not m:
            res.append(((f, sl), 0))
            continue
        fch = [e[4] for e in m if e[4] != e[2]]
        pf = Counter(fch).most_common(1)[0][0] if len(fch) * 2 > n else f
        ps = []
        for k, b in enumerate(sl):
            chg = sum(e[5][k] != e[3][k] for e in m) * 2 > n
            ps.append(succ[(a, k, b)].most_common(1)[0][0] if chg and succ.get((a, k, b)) else b)
        res.append(((pf, tuple(ps)), n))
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(EN.OUT / "play.npz")
    T = EN.tiles(z["obs"])
    enc = EN.Enc().to(dev)
    TOK = Path(os.environ.get("WM98_TOKENS", str(ROOT / "runs" / "098.2" / "encoder_joint.pt")))
    enc.load_state_dict(torch.load(TOK))
    enc.eval()
    C = all_codes(T, enc, np.load(TOK.parent / TOK.name.replace("encoder", "codes").replace(".pt", ".npz"))["protos"])
    act, ep, done, light, face, inv = z["act"], z["ep"], z["done"], z["light"], z["face"], z["inv"]
    sem = z["sem"].reshape(len(act), 63)
    ok = np.zeros(len(act), bool)
    ok[:-1] = (ep[:-1] == ep[1:]) & ~done[:-1]
    test = ep // 100000 >= 8
    rep = {}
    # the place acted on, learned from "do"
    do = np.flatnonzero(ok & (act == 5) & ~test)
    tally = defaultdict(Counter)
    for t in do:
        ch = [q for q in NB if C[t, q] != C[t + 1, q]]
        if len(ch) == 1:
            tally[int(C[t, CENTRE])][ch[0]] += 1
    front_of = {c: n.most_common(1)[0][0] for c, n in tally.items()}
    common = Counter(q for n in tally.values() for q in n.elements()).most_common(1)[0][0]
    F = lambda t: front_of.get(int(C[t, CENTRE]), common)
    true_front = lambda t: (3 + int(face[t][1])) * 9 + 4 + int(face[t][0])
    dt = np.flatnonzero(ok & (act == 5) & test)
    rep["front learned: agrees with the true heading (judged do steps)"] = round(
        float(np.mean([F(t) == true_front(t) for t in dt])), 4)
    rep["codes of the agent's own tile with a learned front"] = len(front_of)
    # events
    ev = np.flatnonzero(ok & np.isin(act, list(ACTS)))
    def tok(t):
        f = F(t)
        key = (int(act[t]), int(C[t, f]), tuple(C[t, 63 + 4:63 + 16].tolist()))
        out = (int(C[t + 1, f]), tuple(C[t + 1, 63 + 4:63 + 16].tolist()))
        return key, key[:2], out
    def true(t):
        f = true_front(t)
        key = (int(act[t]), int(sem[t, f]), tuple(inv[t, NONVITAL].tolist()))
        out = (int(sem[t + 1, f]), tuple(inv[t + 1, NONVITAL].tolist()))
        return key, key[:2], out
    tr, te = ev[~test[ev]], ev[test[ev]]
    rows = {}
    if RELATIVE:
        def tok_r(t):
            f = F(t)
            return (int(act[t]), int(C[t, CENTRE]), int(C[t, f]), tuple(C[t, 63 + 4:63 + 16].tolist()),
                    int(C[t + 1, f]), tuple(C[t + 1, 63 + 4:63 + 16].tolist()))
        def true_r(t):
            f = true_front(t)
            return (int(act[t]), int(face[t][0] * 3 + face[t][1]), int(sem[t, f]), tuple(inv[t, NONVITAL].tolist()),
                    int(sem[t + 1, f]), tuple(inv[t + 1, NONVITAL].tolist()))
        for name, fn_ in (("tokens", tok_r), ("true state (upper bound)", true_r)):
            a, b = [fn_(t) for t in tr], [fn_(t) for t in te]
            pred = recall_parts(a, b)
            right = np.array([p == (e[4], e[5]) for e, (p, n) in zip(b, pred)])
            changed = np.array([(e[4], e[5]) != (e[2], e[3]) for e in b])
            rows[name] = (right, changed, np.array([n for _, n in pred]))
    for name, fn_ in (() if RELATIVE else (("tokens", tok), ("true state (upper bound)", true))):
        a, b = [fn_(t) for t in tr], [fn_(t) for t in te]
        pred = recall(a, b)
        right = np.array([(p if p is not None else (k[1], k[2])) == o
                          for (k, k1, o), (p, n) in zip(b, pred)])
        changed = np.array([o != (k[1], k[2]) for k, k1, o in b])
        n = np.array([m for _, m in pred])
        rows[name] = (right, changed, n)
    right, changed, n = rows["tokens"]
    ub_right, ub_changed, _ = rows["true state (upper bound)"]
    night = light[te] < 0.3
    o = z["obs"][te][:, :49].astype(np.float32)
    asleep = (o.max(-1) - o.min(-1)).mean((1, 2)) < 8   # the frame is grey: the agent is asleep
    rep["right, events that change something, awake / asleep: tokens"] = [
        round(float(right[changed & ~asleep].mean()), 4) if (changed & ~asleep).any() else None,
        round(float(right[changed & asleep].mean()), 4) if (changed & asleep).any() else None]
    rep["right, events that change something, awake: true state"] = round(
        float(ub_right[ub_changed & ~asleep].mean()), 4) if (ub_changed & ~asleep).any() else None
    acts_te = act[te]
    rep["events: stored / judged"] = [int(len(tr)), int(len(te))]
    rep["judged events that change something (tokens / true state)"] = [int(changed.sum()), int(ub_changed.sum())]
    rep["right, events that change something: tokens"] = round(float(right[changed].mean()), 4)
    rep["right, events that change something: true state (upper bound)"] = round(float(ub_right[ub_changed].mean()), 4)
    rep["right, events that change something, at night: tokens"] = round(float(right[changed & night].mean()), 4) \
        if (changed & night).any() else None
    rep["right, events that change something, key stored 1-3 times: tokens"] = round(
        float(right[changed & (n >= 1) & (n <= 3)].mean()), 4) if (changed & (n >= 1) & (n <= 3)).any() else None
    rep["right, events that change nothing: tokens / baseline 'nothing changes'"] = [
        round(float(right[~changed].mean()), 4), 1.0]
    rep["baseline 'nothing changes' on events that change something"] = 0.0
    rep["by action (tokens: changing events right / n; true state: right / n)"] = {
        ACTS[a]: [f"{int(right[changed & (acts_te == a)].sum())}/{int((changed & (acts_te == a)).sum())}",
                  f"{int(ub_right[ub_changed & (acts_te == a)].sum())}/{int((ub_changed & (acts_te == a)).sum())}"]
        for a in ACTS if ((acts_te == a) & (changed | ub_changed)).any()}
    rep["right by stored count (changing events)"] = {
        lab: round(float(right[changed & m].mean()), 4) if (changed & m).any() else None
        for lab, m in (("0", n == 0), ("1-3", (n >= 1) & (n <= 3)), ("4-30", (n >= 4) & (n <= 30)), ("over 30", n > 30))}
    # what the misses are (front class before -> after by the evaluator, action)
    miss = Counter()
    for t, r, c in zip(te, right, changed):
        if c and not r:
            f = true_front(t)
            miss[f"{ACTS[int(act[t])]}: {int(sem[t, f])} -> {int(sem[t + 1, f])}"] += 1
    rep["misses (action: front class before -> after, evaluator's classes)"] = miss.most_common(12)
    rep["tokens"] = str(TOK)
    (OUT / os.environ.get("WM98_OUT", "results.json")).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
