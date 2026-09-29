"""Card 027: kinds from what actions do in front.

The agent sees card 016's egocentric view (the room centred on it, facing
up, plus a place showing what it holds). "In front" and "held" are fixed
places in that view. From its own transitions it reads what each action did
to the thing in front (before/after pixels), groups tile appearances into
kinds by those effects, and gives each way of card 026's tree the kind its
action succeeded on. Acting keeps card 026's depth-first search and exact
conditions; only a non-move way's target changes: the tiles of the way's
kind that the way's route depends on (swap the tile for floor; if the way
can no longer achieve its parent by moving closer, it is a target).

Stages, per world (key, switch):
  gate: effects from pixels against the simulator's events; the upper bound
        (card 026's simulator targets); exact kinds (stochastic bisimulation
        over appearances) and acting with them.
  main (only if every gate check passes): a learned tile encoder (4-bit
        code), its purity, the ways' kinds, acting (main arm), and the
        comparison arms (kind alone; contrast alone).

Run: bin/prun python tools/card027/kinds.py [--episodes 5000] [--layouts 500] [--updates 20000]
     [--sampling outcome|uniform] [--network recognise|effects] [--out runs/027_kinds.json]
"""
import dataclasses
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card023"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card026"))
import closer  # noqa: E402
import dfs as D  # noqa: E402

from worldmodel import discover_logic as dl  # noqa: E402
from worldmodel.envs import logicdoor as ld  # noqa: E402
from worldmodel.envs.keydoor_render import code, encode_logic_states, tile_images  # noqa: E402

BUDGET = 200
R = 6                                   # card 016's view radius: 13 x 13 tiles plus the held row
FRONT, HELD = (R - 1, R), (2 * R + 1, 0)
UP = 3
WALL = code("wall")
TILES = tile_images()                                              # (codes, 8, 8, 3)
_, APP = np.unique(TILES.reshape(len(TILES), -1), axis=0, return_inverse=True)
APP = APP.ravel()                                                  # appearance (distinct pixel tile) of each code
FWD, PICK, DROP, TOG = ld.FORWARD, ld.PICKUP, ld.DROP, ld.TOGGLE
ACTS = (FWD, PICK, DROP, TOG)                                      # the actions that act on the thing in front

KIND_OF_CODE = {code(None): "floor", code("wall"): "wall", code("goal"): "goal",
                code(("ball", "grey")): "switch", code(("ball", "yellow")): "switch", code(("box", "purple")): "vase"}
for _c in ("red", "green", "blue"):
    KIND_OF_CODE[code(("door", _c, 0))] = "closed door"
    KIND_OF_CODE[code(("door", _c, 2))] = "open door"
    KIND_OF_CODE[code(("key", _c))] = "key"
KIND_OF_APP = {int(APP[c]): k for c, k in KIND_OF_CODE.items()}     # evaluator only
NAME_OF_APP = {}
for _c in KIND_OF_CODE:
    NAME_OF_APP[int(APP[_c])] = {code(None): "floor", code("wall"): "wall", code("goal"): "goal",
                                 code(("ball", "grey")): "switch off", code(("ball", "yellow")): "switch on",
                                 code(("box", "purple")): "vase"}.get(_c)
for _c in ("red", "green", "blue"):
    NAME_OF_APP[int(APP[code(("door", _c, 0))])] = f"closed door {_c}"
    NAME_OF_APP[int(APP[code(("door", _c, 2))])] = f"open door {_c}"
    NAME_OF_APP[int(APP[code(("key", _c))])] = f"key {_c}"
EXPECTED_WAY_KIND = {PICK: {"key"}, TOG: {"closed door", "vase", "switch"}, DROP: {"floor"}, FWD: {"goal"}}


def pool20():
    return mp.get_context("fork").Pool(20)


# ---------------------------------------------------------------- the egocentric view

_r, _c = np.meshgrid(np.arange(2 * R + 1), np.arange(2 * R + 1), indexing="ij")
_F, _RT = R - _r, _c - R                                            # steps ahead, steps to the right
_DX = np.stack([_F * ld.DIR_VEC[d][0] + _RT * ld.DIR_VEC[(d + 1) % 4][0] for d in range(4)])
_DY = np.stack([_F * ld.DIR_VEC[d][1] + _RT * ld.DIR_VEC[(d + 1) % 4][1] for d in range(4)])


def ego(codes, st):
    """Top-down codes (B, 9, 8) and states (B, >=3) -> egocentric codes (B, 14, 13), as card 016."""
    x, y, d = st[:, 0].astype(np.int64), st[:, 1].astype(np.int64), st[:, 2].astype(np.int64)
    X, Y = x[:, None, None] + _DX[d], y[:, None, None] + _DY[d]
    ok = (X >= 0) & (X < 8) & (Y >= 0) & (Y < 8)
    b = np.arange(len(codes))[:, None, None]
    e = np.where(ok, codes[b, Y.clip(0, 7), X.clip(0, 7)], WALL).astype(np.int64)
    e[:, R, R] = (e[:, R, R] // 5) * 5 + UP + 1                      # the agent, drawn facing up
    held = np.zeros((len(codes), 1, 2 * R + 1), np.int64)
    held[:, 0, 0] = codes[:, 8, 0]
    return np.concatenate([e, held], 1)


def effects(tr, chunk=50000):
    """What each stored transition did, read from the egocentric pixels: appearances at the fixed
    front and held places before and after, and whether the view changed."""
    n = len(tr["act"])
    out = {k: np.zeros(n, np.int64) for k in ("f0", "f1", "h0", "h1")}
    out["view_changed"] = np.zeros(n, bool)
    out["front_topdown_ok"] = np.zeros(n, bool)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        s0, s1 = tr["s0"][sl], tr["s1"][sl]
        e0, e1 = APP[ego(tr["c0"][sl], s0)], APP[ego(tr["c1"][sl], s1)]
        out["f0"][sl], out["f1"][sl] = e0[:, FRONT[0], FRONT[1]], e1[:, FRONT[0], FRONT[1]]
        out["h0"][sl], out["h1"][sl] = e0[:, HELD[0], HELD[1]], e1[:, HELD[0], HELD[1]]
        out["view_changed"][sl] = (e0[:, :2 * R + 1] != e1[:, :2 * R + 1]).reshape(len(s0), -1).any(1)
        # consistency: the fixed front place shows the top-down tile in front of the agent
        fx = s0[:, 0] + np.array([v[0] for v in ld.DIR_VEC])[s0[:, 2]]
        fy = s0[:, 1] + np.array([v[1] for v in ld.DIR_VEC])[s0[:, 2]]
        top = APP[tr["c0"][sl][np.arange(len(s0)), fy, fx]]
        out["front_topdown_ok"][sl] = top == out["f0"][sl]
    act = tr["act"].astype(np.int64)
    out["fc"] = out["f0"] != out["f1"]
    out["hc"] = out["h0"] != out["h1"]
    goal = tr["term1"].astype(bool)
    out["fwd"] = np.where(goal, 2, np.where(out["view_changed"], 0, 1))     # moved / blocked / goal reached
    out["act"] = act
    return out


def effects_check(tr, E):
    """Effects from pixels against the simulator's events (evaluator)."""
    act, s0, s1 = E["act"], tr["s0"].astype(np.int64), tr["s1"].astype(np.int64)
    goal = tr["term1"].astype(bool)
    f = act == FWD
    moved = (s1[:, :2] != s0[:, :2]).any(1)
    inter = np.isin(act, (PICK, DROP, TOG))
    sim_front = (s1[:, 3:] != s0[:, 3:]).any(1)
    sim_held = s1[:, 3] != s0[:, 3]
    return {"forward_moved_agree": float(((E["fwd"] == 0) == moved)[f & ~goal].mean()),
            "front_changed_agree": float((E["fc"] == sim_front)[inter].mean()),
            "held_changed_agree": float((E["hc"] == sim_held)[inter].mean()),
            "front_place_is_tile_in_front": float(E["front_topdown_ok"].mean())}


# ---------------------------------------------------------------- exact kinds: stochastic bisimulation

def _logev(counts, alpha=1.0):
    """Dirichlet-multinomial log evidence of a count vector (uniform prior)."""
    k, n = len(counts), sum(counts)
    return math.lgamma(k * alpha) - math.lgamma(k * alpha + n) + sum(math.lgamma(alpha + c) - math.lgamma(alpha)
                                                                   for c in counts)


def _cluster(members, counts):
    """Agglomerative: merge groups while pooling their outcome counts has higher evidence."""
    cats = sorted({o for u in members for o in counts[u]}, key=str)
    vec = {u: [counts[u][o] for o in cats] for u in members}
    groups = [([u], vec[u]) for u in members]
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i][1], groups[j][1]
                gain = _logev([x + y for x, y in zip(a, b)]) - _logev(a) - _logev(b)
                if gain >= 0 and (best is None or gain > best[0]):
                    best = (gain, i, j)
        if best is None:
            break
        _, i, j = best
        merged = (groups[i][0] + groups[j][0], [x + y for x, y in zip(groups[i][1], groups[j][1])])
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [merged]
    return [sorted(g[0]) for g in groups]


def exact_kinds(E, log):
    """Coarsest grouping of front appearances in which every action has the same effect and turns the
    thing into things of the same kinds. Raw stored counts: storage keeps changes at a fixed rate
    independent of the appearance, so equal outcome distributions stay equal (weights only rescale)."""
    apps = sorted(set(E["f0"].tolist()))
    raw = defaultdict(Counter)
    for u, a, fwd, fc, f1, hc, h1 in zip(E["f0"], E["act"], E["fwd"], E["fc"], E["f1"], E["hc"], E["h1"]):
        if a in ACTS:
            raw[(int(u), int(a))][(int(fwd),) if a == FWD else (bool(fc), int(f1) if fc else -1,
                                                              bool(hc), int(h1) if hc else -1)] += 1
    block = {u: 0 for u in apps}
    for it in range(20):
        def outcome(o, a):
            if a == FWD:
                return o
            fc, f1, hc, h1 = o
            return (fc, block.get(f1, -2) if fc else -1, hc, block.get(h1, -2) if hc else -1)
        new, nid = {}, 0
        for b in sorted(set(block.values())):
            members = [u for u in apps if block[u] == b]
            labels = {u: [] for u in members}
            for a in ACTS:
                counts = {u: Counter() for u in members}
                for u in members:
                    for o, c in raw[(u, a)].items():
                        counts[u][outcome(o, a)] += c
                for k, g in enumerate(_cluster(members, counts)):
                    for u in g:
                        labels[u].append(k)
            ids = {}
            for u in members:
                key = tuple(labels[u])
                if key not in ids:
                    ids[key], nid = nid, nid + 1
                new[u] = ids[key]
        if sorted(Counter(new.values()).values()) == sorted(Counter(block.values()).values()) and \
                len(set(new.values())) == len(set(block.values())):
            block = new
            break
        block = new
    groups = defaultdict(list)
    for u in apps:
        groups[block[u]].append(u)
    log(f"  exact kinds after {it + 1} rounds: " + "; ".join(
        "{" + ", ".join(NAME_OF_APP.get(u, str(u)) for u in g) + "}" for g in groups.values()))
    return {u: block[u] for u in apps}


def partition_check(kind_of):
    """Does the grouping equal the expected kinds (evaluator)? Also: no group mixes two kinds."""
    groups = defaultdict(set)
    for u, k in kind_of.items():
        groups[k].add(u)
    mixed = [sorted(NAME_OF_APP[u] for u in g) for g in groups.values() if len({KIND_OF_APP[u] for u in g}) > 1]
    exp = defaultdict(set)
    for u in kind_of:
        exp[KIND_OF_APP[u]].add(u)
    equal = sorted(map(sorted, groups.values())) == sorted(map(sorted, exp.values()))
    merged = {k: len({kind_of[u] for u in us}) for k, us in exp.items()}
    return {"equals_expected": equal, "groups_mixing_kinds": mixed, "codes_per_expected_kind": merged,
            "groups": [sorted(NAME_OF_APP[u] for u in g) for g in groups.values()]}


# ---------------------------------------------------------------- learned kinds: a tile encoder

def outcome_classes(E, keep):
    """Each training row's outcome: the action and what it did (forward: moved, blocked, goal;
    otherwise which of the front and held places changed)."""
    a, fwd, fc, hc = E["act"][keep], E["fwd"][keep], E["fc"][keep].astype(int), E["hc"][keep].astype(int)
    name = {FWD: "forward", PICK: "pickup", DROP: "drop", TOG: "toggle"}
    lab = np.where(a == FWD, np.char.add("forward ", np.array(["moved", "blocked", "goal"])[fwd]),
                   np.char.add(np.char.add(np.vectorize(name.get)(a), " front "),
                               np.char.add(np.array(["same", "changed"])[fc],
                                           np.char.add(", held ", np.array(["same", "changed"])[hc]))))
    return lab


def train_encoder(E, log, updates=20000, batch=1024, seed=0, sampling="outcome"):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    keep = np.isin(E["act"], ACTS)
    rep = np.zeros(APP.max() + 1, np.int64)                        # one code per appearance, for its pixels
    for c in range(len(APP) - 1, -1, -1):
        rep[APP[c]] = c
    pix = torch.as_tensor(TILES[rep].reshape(len(rep), -1) / 255.0, dtype=torch.float32, device=dev)
    col = lambda k: torch.as_tensor(E[k][keep], device=dev)
    f0, f1, h1 = col("f0"), col("f1"), col("h1")
    act = torch.as_tensor(np.searchsorted(np.array(ACTS), E["act"][keep]), device=dev)
    fwd, fc, hc = col("fwd"), col("fc").float(), col("hc").float()
    enc = nn.Sequential(nn.Linear(pix.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4)).to(dev)
    head = nn.Sequential(nn.Linear(4 + 4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3 + 2 + 4 + 4)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    bce = nn.functional.binary_cross_entropy_with_logits

    def bits(x):
        p = torch.sigmoid(enc(x))
        return (p > 0.5).float() + p - p.detach()                  # straight-through, as DeepSym

    n = len(f0)
    lab = outcome_classes(E, keep)
    names, cls, counts = np.unique(lab, return_inverse=True, return_counts=True)
    classes = {str(k): int(v) for k, v in zip(names, counts)}
    log(f"  outcomes in the training rows: {classes}; sampling {sampling}")
    prob = torch.as_tensor(1.0 / (len(names) * counts[cls.ravel()]), dtype=torch.float32, device=dev)
    t0 = time.monotonic()
    for step in range(updates):
        if sampling == "outcome":                                  # every outcome equally likely (second revision)
            i = torch.multinomial(prob, batch, replacement=True)
        else:
            i = torch.randint(0, n, (batch,), device=dev)
        b = bits(pix[f0[i]])
        a1 = nn.functional.one_hot(act[i], 4).float()
        out = head(torch.cat([b, a1], 1))
        isf = act[i] == 0
        loss_f = nn.functional.cross_entropy(out[isf, :3], fwd[i][isf]) if isf.any() else out.sum() * 0
        ii = ~isf
        loss_c = bce(out[ii, 3], fc[i][ii]) + bce(out[ii, 4], hc[i][ii]) if ii.any() else out.sum() * 0
        with torch.no_grad():
            tf, th = (torch.sigmoid(enc(pix[f1[i]])) > 0.5).float(), (torch.sigmoid(enc(pix[h1[i]])) > 0.5).float()
        mf, mh = ii & (fc[i] > 0), ii & (hc[i] > 0)
        loss_b = (bce(out[mf, 5:9], tf[mf]) if mf.any() else out.sum() * 0) + \
                 (bce(out[mh, 9:13], th[mh]) if mh.any() else out.sum() * 0)
        loss = loss_f + loss_c + loss_b
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5000 == 0 or step == updates - 1:
            log(f"  encoder step {step}: forward {loss_f.item():.4f}, changed {loss_c.item():.4f}, "
                f"becomes {loss_b.item():.4f}")
    with torch.no_grad():
        codes = (torch.sigmoid(enc(pix)) > 0.5).long().cpu().numpy()
    code_of_app = codes @ (1 << np.arange(4))                      # 0..15 per appearance
    return code_of_app.astype(np.int64), round(time.monotonic() - t0, 1), dev, classes


def train_recogniser(kind_of, updates=2000, seed=0, exclude=None):
    """Third revision: a network trained to give each appearance the kind counting assigned it.

    kind_of: appearance -> counted kind. Training set: the distinct appearances (one fact each, equally
    weighted), without `exclude` if given. Returns (kind per appearance for every appearance, last loss)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rep = np.zeros(APP.max() + 1, np.int64)
    for c in range(len(APP) - 1, -1, -1):
        rep[APP[c]] = c
    pix = torch.as_tensor(TILES[rep].reshape(len(rep), -1) / 255.0, dtype=torch.float32)
    kinds = sorted(set(kind_of.values()))
    kidx = {k: j for j, k in enumerate(kinds)}
    apps = sorted(u for u in kind_of if u != exclude)
    x = pix[apps]
    y = torch.as_tensor([kidx[kind_of[u]] for u in apps])
    net = nn.Sequential(nn.Linear(pix.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, len(kinds)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(updates):
        loss = nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(pix).argmax(1).numpy()
    return np.array([kinds[j] for j in pred], np.int64), float(loss.item())


def kind_name(kind_of, k):
    """Evaluator's name for a counted kind (report only)."""
    return "/".join(sorted({KIND_OF_APP.get(int(u), "?") for u, v in kind_of.items() if v == k})) or "?"


def leave_one_out(kind_of, seeds=(0, 1, 2)):
    """Reported, not a criterion: each appearance whose counted kind has another member, withheld from
    training, and the kind the network then gives it."""
    size = Counter(kind_of.values())
    out = {}
    for u in sorted(kind_of):
        if size[kind_of[u]] < 2:
            continue
        preds = [int(train_recogniser(kind_of, seed=s, exclude=u)[0][u]) for s in seeds]
        out[NAME_OF_APP[int(u)]] = {"correct_seeds": sum(p == kind_of[u] for p in preds), "of": len(seeds),
                                    "given": [kind_name(kind_of, p) for p in preds]}
    return out


# ---------------------------------------------------------------- a way's kind

def thing_ways(tree):
    """Non-move ways: pick up, drop, toggle, and the step onto the goal square (forward under goal 0)."""
    return [n for n in range(1, len(tree.parent)) if tree.accepted(n)
            and (tree.action[n] in (PICK, DROP, TOG) or (tree.action[n] == FWD and tree.parent[n] == 0))]


def way_kinds(tree, ways, L0, L1, tr, E, kind_of):
    """The kinds in front (and held, for drop) at each way's successes in experience.

    code_pure_share (criterion 2, revised): the share of successes whose front code belongs only to
    appearances of kinds expected for the action (for drop, also a held code belonging only to keys)."""
    out = {}
    act, w = E["act"], tr["w0"].astype(np.float64)
    code_kinds = defaultdict(set)
    for u, k in kind_of.items():
        code_kinds[k].add(KIND_OF_APP.get(int(u), "?"))
    for n in ways:
        g, a = tree.parent[n], tree.action[n]
        m = (act == a) & ~L0[:, g] & L1[:, g]
        if not m.any():
            out[n] = {"successes": 0}
            continue
        ws = w[m]
        fk = Counter()
        for u, x in zip(E["f0"][m], ws):
            fk[kind_of.get(int(u), -1)] += x
        tot = sum(fk.values())
        codes = sorted(k for k, v in fk.items() if v / tot >= 0.01)
        ek = Counter()
        for u, x in zip(E["f0"][m], ws):
            ek[KIND_OF_APP.get(int(u), "?")] += x
        top, topw = ek.most_common(1)[0]
        hk = Counter()
        for u, x in zip(E["h0"][m], ws):
            hk[KIND_OF_APP.get(int(u), "?")] += x
        ok = np.array([code_kinds.get(kind_of.get(int(u), -1), {"?"}) <= EXPECTED_WAY_KIND[a] for u in E["f0"][m]])
        if a == DROP:
            ok &= np.array([code_kinds.get(kind_of.get(int(u), -1), {"?"}) <= {"key"} for u in E["h0"][m]])
        out[n] = {"path": "/".join(tree.path(n)), "successes": int(m.sum()), "codes": codes,
                  "expected_kind_share": round(topw / tot, 4), "kind": top,
                  "kinds": {k: round(v / tot, 4) for k, v in ek.most_common() if v / tot >= 0.01},
                  "expected_share": round(sum(v for k, v in ek.items() if k in EXPECTED_WAY_KIND[a]) / tot, 4),
                  "code_pure_share": round(float(ws[ok].sum() / tot), 4),
                  "as_expected": top in EXPECTED_WAY_KIND[a],
                  "held_kind": hk.most_common(1)[0][0] if a == DROP else None}
    return out


# ---------------------------------------------------------------- acting with kind targets

def swap_for_floor(lay, s, pos):
    """The state (and layout) with the thing at pos replaced by floor; None if nothing to swap."""
    x, y, d, carry, key, dist, door, sw, vase = s
    if key == pos:
        return lay, (x, y, d, carry, (0, 0), dist, door, sw, vase)
    if dist == pos:
        return lay, (x, y, d, carry, key, (0, 0), door, sw, vase)
    if pos == lay.vase and not vase:
        return lay, (x, y, d, carry, key, dist, door, sw, 1)
    if pos == lay.door:
        return lay, (x, y, d, carry, key, dist, 1, sw, vase)            # a door swapped for floor: open
    if pos == lay.switch:
        return dataclasses.replace(lay, switch=(0, 0)), (x, y, d, carry, key, dist, door, 0, vase)
    if pos == lay.goal:
        return dataclasses.replace(lay, goal=(0, 0)), s
    return None


def target_poses(lay, s, tiles):
    """Poses facing one of the tiles, standing on a free tile that is not the goal (as card 023)."""
    poses = set()
    for tx, ty in tiles:
        for d in range(4):
            px, py = tx - ld.DIR_VEC[d][0], ty - ld.DIR_VEC[d][1]
            if (px, py) == lay.goal:
                continue
            c = ld.cell(lay, s, (px, py))
            if c == "empty" or (c == "door" and s[6]):
                poses.add((px, py, d))
    return poses


def move_closer(ex, s, poses, keep=None, refused=None):
    """The first move bringing the agent closer to one of the poses. keep (revision): a way whose
    condition must still hold after the move; moves that would end it are refused and counted."""
    if not poses:
        return None
    rset = (poses, None)
    here = closer.closeness(ex, s, rset)
    for a in closer.MOVE_ORDER:
        t = ld.step(ex.lay, s, a)[0]
        if (t[0], t[1]) == ex.lay.goal and t[:2] != s[:2]:
            continue
        if closer.closeness(ex, t, rset) < here:
            if keep is not None and not ex.holds(keep, t):
                if refused is not None:
                    refused[0] += 1
                continue
            return a
    return None


def _act_job(job):
    idx, layouts, parent, action, depth, kids, expanded, things, arm, K, rules = job
    code_of_app, way_codes, floor_app, wall_app = K
    out = []
    for i, lay in zip(idx, layouts):
        rng = np.random.default_rng(1000 + int(i))
        ex = closer.Approach(lay, parent, action)
        swapped = {}
        s = ld.start_state(lay)
        failed = set()                                  # (way, tile, pose) that did not work this episode
        refused = [0]
        rec = {"done": False, "steps": BUDGET, "random_no_way": 0, "random_no_move": 0, "move_way_steps": 0,
               "dep_checks": 0, "moves_with_candidates": 0, "first_key": None, "picked_other": False,
               "failed_acts": 0, "ways_used": Counter()}
        for t in range(BUDGET):
            c, need = D.dfs(kids, depth, expanded, lambda n: ex.holds(n, s), 0, [0])
            a = None
            if c is None:
                rec["random_no_way"] += 1
            elif c not in things:
                rec["move_way_steps"] += 1
                g, aw = parent[c], action[c]
                a = aw if ex.ready(g, aw, s) else closer.closer_move(ex, c, s)
                if a is None:
                    rec["random_no_move"] += 1
            else:
                aw = action[c]
                apps = APP[encode_logic_states(lay, [s])[0][:lay.size]]            # every tile in view
                cand = set()
                for y in range(lay.size):
                    for x in range(lay.size):
                        if (x, y) == (s[0], s[1]):
                            continue                                            # the agent's own place
                        u = apps[y, x]
                        if arm == "contrast" and u != floor_app and u != wall_app:
                            cand.add((x, y))
                        elif arm != "contrast" and code_of_app[u] in way_codes[c]:
                            cand.add((x, y))
                targets = cand
                if arm != "kind_alone" and len(cand) > 1:
                    rec["moves_with_candidates"] += 1
                    dep = set()
                    for x, y in cand:
                        if apps[y, x] == floor_app:
                            continue                                            # swapping floor for floor
                        sw = swap_for_floor(lay, s, (x, y))
                        if sw is None:
                            continue
                        lay2, s2 = sw
                        if lay2 is lay:
                            ex2 = ex
                        else:
                            ex2 = swapped.setdefault(lay2, closer.Approach(lay2, parent, action))
                        rec["dep_checks"] += 1
                        if not closer.approach(ex2, c, s2):
                            dep.add((x, y))
                    targets = dep or cand
                poses = target_poses(lay, s, targets)
                if rules:
                    poses = {q for q in poses if (c, (q[0] + ld.DIR_VEC[q[2]][0], q[1] + ld.DIR_VEC[q[2]][1]), q)
                             not in failed}
                if s[:3] in poses:
                    a = aw
                    rec["ways_used"][c] += 1
                else:
                    a = move_closer(ex, s, poses, c if rules else None, refused)
                if a is None:
                    rec["random_no_move"] += 1
            if a is None:
                a = int(rng.integers(len(dl.ACTIONS)))
            s2, end = ld.step(lay, s, a)
            if rules and c in things and a == action[c] and s[:3] in poses and not end \
                    and not ex.holds(parent[c], s2):
                failed.add((c, ld.front(s), s[:3]))       # acting here did not turn the parent true
                rec["failed_acts"] += 1
            if s2[3] and not s[3] and rec["first_key"] is None:
                rec["first_key"] = "matching" if s2[3] == 1 else "other"
            if s2[3] == 2:
                rec["picked_other"] = True
            s = s2
            if end:
                rec["done"], rec["steps"] = True, t + 1
                break
        rec["moves_refused"] = refused[0]
        out.append(rec)
    return out


def act_arm(pool, test, tree, expanded, arm, K, rules=True):
    kids = {g: tree.children(g) for g in expanded}
    things = set(thing_ways(tree))
    chunks = [c.tolist() for c in np.array_split(np.arange(len(test)), 40) if len(c)]
    parts = pool.map(_act_job, [(c, [test[i] for i in c], tree.parent, tree.action, tree.depth, kids,
                                 set(expanded), things, arm, K, rules) for c in chunks])
    recs = [r for p in parts for r in p]
    done = np.array([r["done"] for r in recs])
    steps = np.array([r["steps"] for r in recs])
    rnd = sum(r["random_no_way"] + r["random_no_move"] for r in recs)
    res = {"success": round(float(done.mean()), 4),
           "mean_steps_when_successful": round(float(steps[done].mean()), 1) if done.any() else None,
           "random_steps": int(rnd), "moves": int(steps.sum()), "random_share": round(rnd / max(int(steps.sum()), 1), 4),
           "move_way_steps": int(sum(r["move_way_steps"] for r in recs)),
           "dependence_checks_per_move": round(sum(r["dep_checks"] for r in recs) / max(int(steps.sum()), 1), 3),
           "failed_acts": int(sum(r["failed_acts"] for r in recs)),
           "layouts_with_failed_acts": int(sum(r["failed_acts"] > 0 for r in recs)),
           "moves_refused": int(sum(r["moves_refused"] for r in recs))}
    used = Counter()
    for r in recs:
        used.update(r["ways_used"])
    res["ways_used"] = {"/".join(tree.path(n)): k for n, k in sorted(used.items())}
    fk = [r["first_key"] for r in recs]
    if any(fk):
        mm = np.array([f == "matching" for f in fk])
        res["first_key_matching_success"] = round(float(done[mm].mean()), 4) if mm.any() else None
        res["layouts_first_key_matching"] = int(mm.sum())
        fail = ~done
        res["failing_layouts"] = int(fail.sum())
        res["failing_that_picked_other_key"] = int(sum(r["picked_other"] for r, f in zip(recs, fail) if f))
    return res


# ---------------------------------------------------------------- main

def grow_tree(pool, test, layouts, tr, seq, st0, log):
    """Card 026's discovery on demand; the last round's play is the upper bound (simulator targets)."""
    tree, expanded = dl.Tree(), set()
    tree.status[0] = "not expanded"
    rounds = 0
    while True:
        recs = D.play(pool, test, tree, expanded)
        paused = Counter(r["pause"] for r in recs if r["pause"] is not None)
        log(f"  round {rounds}: {sum(paused.values())} layouts paused, goals needed "
            f"{ {'/'.join(tree.path(g)) or '(goal square)': k for g, k in paused.items()} }")
        if not paused or rounds >= 40:
            return tree, expanded, recs, rounds + 1
        D.expand_goals(pool, tree, sorted(paused), expanded, layouts, tr, seq, st0, lambda m: None)
        rounds += 1


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    out = Path(args.get("--out", "runs/027_kinds.json"))
    episodes, n_test = int(args.get("--episodes", 5000)), int(args.get("--layouts", 500))
    updates = int(args.get("--updates", 20000))
    sampling = args.get("--sampling", "outcome")                   # "uniform" reproduces the second run
    network = args.get("--network", "recognise")                   # "effects" reproduces the second and third runs
    closer.MEASURE = "step"
    dl.MAX_GOALS = 10 ** 9
    dl.START_SHARE = 2.0
    dl.Exact = closer.Approach                       # the workers read the module's Exact
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    res = {"note": "Card 027, tools/card027/kinds.py; exact conditions and walking, kinds from pixels.", "worlds": {}}
    W = {}
    for world in ("key", "switch"):
        log(f"=== {world} world: gate")
        r = res["worlds"][world] = {}
        pool = pool20()
        layouts, tr, seq, st0 = dl.collect(pool, world, episodes, 11, 0.0, seq_episodes=300)
        rng = np.random.default_rng(777)
        test = [ld.make_layout(8, world, rng) for _ in range(n_test)]
        E = effects(tr)
        r["effects_check"] = effects_check(tr, E)
        log(f"  effects from pixels against the simulator: {r['effects_check']}")
        if world == "key":
            rep = json.loads(Path("runs/026_dfs.json").read_text())["tree"]
            tree = dl.Tree.from_report(rep)
            expanded = {t["id"] for t in rep if t["status"] == "expanded" or t["status"].startswith("leaf: no action")}
            recs, rounds = D.play(pool, test, tree, expanded), "rebuilt from card 026"
        else:
            tree, expanded, recs, rounds = grow_tree(pool, test, layouts, tr, seq, st0, log)
        ub = D.summary(recs, tree)
        r["tree_nodes"], r["goals_expanded"], r["rounds"] = len(tree.parent), len(expanded), rounds
        r["tree"] = [(t["path"], t["status"]) for t in tree.report()]
        r["upper_bound"] = {k: ub[k] for k in ("success", "mean_steps_when_successful", "random_steps", "moves",
                                               "random_share", "ways_used_steps")}
        log(f"  upper bound (simulator targets): {json.dumps(r['upper_bound'])}")
        exact = exact_kinds(E, log)
        r["exact_kinds"] = partition_check(exact)
        ways = thing_ways(tree)
        parents = sorted({tree.parent[n] for n in ways})
        L0, L1 = dl.exact_labels(pool, layouts, tree.parent, tree.action, tr["ep"], [tr["s0"], tr["s1"]], parents)
        wk = way_kinds(tree, ways, L0, L1, tr, E, exact)
        r["way_kinds_exact"] = {v["path"]: {k: v[k] for k in ("successes", "kinds", "expected_share", "code_pure_share")}
                                for v in wk.values() if v.get("successes")}
        wc = np.bincount(E["f0"], weights=tr["w0"].astype(np.float64))
        floor_app, wall_app = [int(u) for u in np.argsort(-wc)[:2]]           # the two most common appearances
        n_app = int(APP.max()) + 1
        code_exact = np.full(n_app, -1, np.int64)
        for u, k in exact.items():
            code_exact[u] = k
        K = (code_exact, {n: set(v.get("codes", [])) for n, v in wk.items()}, floor_app, wall_app)
        r["arm_exact_kinds"] = act_arm(pool, test, tree, expanded, "main", K)
        r["arm_exact_kinds_no_rules"] = act_arm(pool, test, tree, expanded, "main", K, rules=False)
        log(f"  exact kinds: {json.dumps(r['exact_kinds'])}")
        log(f"  ways' kinds (exact): {json.dumps(r['way_kinds_exact'])}")
        log(f"  acting with exact kinds: {json.dumps(r['arm_exact_kinds'])}")
        log(f"  acting with exact kinds, first run's rule (no revision rules): {json.dumps(r['arm_exact_kinds_no_rules'])}")
        pool.close()
        ec = r["effects_check"]
        r["gate"] = {"effects_agree": all(v == 1.0 for v in ec.values()),
                     "upper_bound_98": ub["success"] >= 0.98,
                     "exact_kinds_equal_expected": r["exact_kinds"]["equals_expected"],
                     "exact_kinds_acting_98": r["arm_exact_kinds"]["success"] >= 0.98}
        log(f"  GATE {world}: {r['gate']}")
        W[world] = (layouts, tr, test, E, tree, expanded, ways, L0, L1, floor_app, wall_app, n_app, exact)
        out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    res["gate_passed"] = all(all(res["worlds"][w]["gate"].values()) for w in W)
    log(f"GATE passed: {res['gate_passed']}")
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    if not res["gate_passed"]:
        log("gate failed: main run not started")
        return

    for world, (layouts, tr, test, E, tree, expanded, ways, L0, L1, floor_app, wall_app, n_app, exact) in W.items():
        log(f"=== {world} world: main")
        r = res["worlds"][world]
        if network == "recognise":                                 # third revision
            t1 = time.monotonic()
            code_of_app, last = train_recogniser(exact)
            secs, dev, classes = round(time.monotonic() - t1, 1), "cpu", {"final_loss": round(last, 6)}
            r["leave_one_out"] = leave_one_out(exact)
            log(f"  leave one out (reported): {json.dumps(r['leave_one_out'])}")
        else:
            code_of_app, secs, dev, classes = train_encoder(E, log, updates=updates, sampling=sampling)
        r["encoder"] = {"network": network, "seconds": secs, "device": dev, "sampling": sampling, "outcome_rows": classes,
                        "code_of_appearance": {NAME_OF_APP[u]: int(code_of_app[u]) for u in sorted(set(E["f0"].tolist()))}}
        learned = {u: int(code_of_app[u]) for u in sorted(set(E["f0"].tolist()))}
        r["criterion_1_learned_kinds"] = partition_check(learned)
        log(f"  learned codes: {json.dumps(r['encoder']['code_of_appearance'])}")
        log(f"  RESULT criterion 1 (learned kinds): {json.dumps(r['criterion_1_learned_kinds'])}")
        wk = way_kinds(tree, ways, L0, L1, tr, E, learned)
        pool = pool20()
        K = (code_of_app, {n: set(v.get("codes", [])) for n, v in wk.items()}, floor_app, wall_app)
        for arm in ("main", "kind_alone", "contrast"):
            r[f"arm_{arm}"] = act_arm(pool, test, tree, expanded, arm, K)
            log(f"  RESULT acting, {arm}: {json.dumps(r[f'arm_{arm}'])}")
        pool.close()
        r["way_kinds_learned"] = {v["path"]: {k: v[k] for k in ("successes", "codes", "kinds", "expected_share",
                                                                  "code_pure_share", "held_kind")}
                                  for v in wk.values() if v.get("successes")}
        log(f"  RESULT ways' kinds, learned: {json.dumps(r['way_kinds_learned'])}")
        used = r["arm_main"]["ways_used"]
        shares = {w: r["way_kinds_learned"].get(w, {}).get("code_pure_share", 0.0) for w in used}
        r["criterion_2"] = {"ways_used_by_main": len(used), "lowest_code_pure_share": min(shares.values()) if shares else None,
                            "failing_ways": {w: s for w, s in shares.items() if s < 0.99},
                            "pass": all(s >= 0.99 for s in shares.values())}
        m, ub = r["arm_main"], r["upper_bound"]
        r["criterion_3"] = {"success_98": m["success"] >= 0.98, "random_share_1pct": m["random_share"] <= 0.01,
                            "steps_within_1_5x": (m["mean_steps_when_successful"] or 1e9)
                            <= 1.5 * ub["mean_steps_when_successful"]}
        log(f"  RESULT criterion 2: {json.dumps(r['criterion_2'])}")
        log(f"  RESULT criterion 3: {json.dumps(r['criterion_3'])}")
        out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    res["seconds"] = round(time.monotonic() - t00, 1)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
