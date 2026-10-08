"""Card 087: properties as learned action effects. Property play, the ensemble, and the gate's parts (i) and the
upper bound.

Property play: a tile of one kind MiniGrid's tiers draw (floor, wall, goal, open, closed or locked door, key, ball,
box), coloured kinds in a fresh hue (card 070's sampler: uniform in RGB, at least 60 from MiniGrid's six and the three
held-out hues), stands in front of the agent in a small room. The agent moves forward once, picks up once with an
empty hand, and toggles once with each of four holdings (nothing, a key of the tile's hue, a key of another fresh
hue, a ball), each from the same start; the outcome is what the next state shows (the agent's place, the held tile,
the front tile), as memory reads it (tools/card066/tiers.py, `grid_codes`). The tile is drawn as card 052 draws it
(MiniGrid's drawing at 8 pixels, noise 4/255) and encoded by the agent's encoder (card 070's, frozen).

Properties of a tile (one per instance): forward = moved / blocked / ended; pick up = taken (empty hand); toggle = it
changed under at least one holding (it can change in some situation).

  bin/prun python tools/card087/props.py --check                   section 4: runs/087/datacheck.json
  bin/prun python tools/card087/props.py --gate                    upper bound, part (i), P19 curve, box left out
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card070"))
import relplay as RPL                                  # noqa: E402  (card 052's generator, the hue sampler)
GN = RPL.GN
sys.path.insert(0, str(ROOT / "tools" / "card053"))
import recall_probe as RP                              # noqa: E402  (card 052's encoder classes)
import torch                                           # noqa: E402
from minigrid.core import constants as K               # noqa: E402
from minigrid.core.grid import Grid                    # noqa: E402
from minigrid.core.mission import MissionSpace         # noqa: E402
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Wall  # noqa: E402
from minigrid.minigrid_env import MiniGridEnv          # noqa: E402

OUT = ROOT / "runs" / "087"
ENCODER = ROOT / "runs" / "070" / "encoder.pt"
KINDS = ("floor", "wall", "goal", "open", "closed", "locked", "key", "ball", "box")
COLOURED = ("open", "closed", "locked", "key", "ball", "box")
SLOTS = ("fresh_a", "fresh_b")                          # the tile's hue, another key's hue
for s in SLOTS:
    if s not in K.COLOR_TO_IDX:
        K.COLORS[s] = np.array([0, 0, 0])
        K.COLOR_TO_IDX[s] = len(K.COLOR_TO_IDX)
        K.IDX_TO_COLOR[K.COLOR_TO_IDX[s]] = s
FWD, PICK, TOG = 2, 3, 5                               # MiniGrid's actions
HOLD = ("nothing", "key same hue", "key other hue", "ball")
NAMED = ("red", "green", "blue", "purple", "yellow", "grey", "pink", "brown", "teal")   # gate (i); the last three
                                                                                       # never in MiniGrid's tiers


def make_obj(kind, colour):
    return {"floor": lambda: None, "wall": lambda: Wall(), "goal": lambda: Goal(),
            "open": lambda: Door(colour, is_open=True), "closed": lambda: Door(colour),
            "locked": lambda: Door(colour, is_locked=True), "key": lambda: Key(colour),
            "ball": lambda: Ball(colour), "box": lambda: Box(colour)}[kind]()


class Room(MiniGridEnv):
    """3 x 4: walls around, the agent at (1, 2) facing up, the tile at (1, 1)."""

    def __init__(self):
        super().__init__(mission_space=MissionSpace(mission_func=lambda: ""), width=3, height=4, max_steps=10,
                         see_through_walls=True)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.agent_pos, self.agent_dir = (1, 2), 3


ROOM = Room()
ROOM.reset(seed=0)


def outcome(kind, colour, held, action):
    """One try from the start: what the next state shows. Returns (moved, ended, held changed, front changed)."""
    u = ROOM
    g = Grid(3, 4)
    g.wall_rect(0, 0, 3, 4)
    g.set(1, 1, make_obj(kind, colour))
    u.grid, u.agent_pos, u.agent_dir, u.step_count = g, (1, 2), 3, 0
    u.carrying = held
    before_front = g.get(1, 1)
    before_front = None if before_front is None else before_front.encode()
    before_held = None if held is None else held.encode()
    _, _, term, _, _ = u.step(action)
    after_front = u.grid.get(1, 1)
    after_front = None if after_front is None else after_front.encode()
    after_held = None if u.carrying is None else u.carrying.encode()
    moved = tuple(u.agent_pos) != (1, 2)
    return moved, bool(term), after_held != before_held, after_front != before_front


def held_obj(h, ):
    return {"nothing": None, "key same hue": Key(SLOTS[0]), "key other hue": Key(SLOTS[1]),
            "ball": Ball(SLOTS[1])}[h]


def labels(kind):
    """The three properties of a tile of this kind (its hue in slot a; another key's hue in slot b)."""
    moved, ended, _, _ = outcome(kind, SLOTS[0], None, FWD)
    _, _, took, _ = outcome(kind, SLOTS[0], None, PICK)
    tog = [outcome(kind, SLOTS[0], held_obj(h), TOG)[3] for h in HOLD]
    walk = 2 if ended else (0 if moved else 1)         # 0 moved onto it, 1 blocked, 2 ended
    return walk, int(took), int(any(tog)), tog


def draw(kind, rgb, rng, noise=True):
    """The tile as card 052 draws it, in hue rgb (coloured kinds); uint8 (8, 8, 3)."""
    K.COLORS[SLOTS[0]] = np.asarray(rgb, np.int64)
    img = GN.clean_tile(make_obj(kind, SLOTS[0])).astype(np.float64) / 255.0
    if noise:
        img = img + rng.normal(0, GN.NOISE, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def draw_named(kind, colour):
    """MiniGrid's own tile in a named colour, as the agent's catalogue draws it (no noise)."""
    return GN.clean_tile(make_obj(kind, colour))


def play(n, rng, kinds=KINDS):
    """n tile instances: kind, hue, tile, labels. The labels are computed per instance from its own tries."""
    lab = {k: labels(k) for k in kinds}              # tries are the same for every hue of a kind; checked in --check
    ks = rng.choice(len(kinds), n)
    hues = RPL.hues(n, rng)
    tiles = np.stack([draw(kinds[k], hues[i], rng) for i, k in enumerate(ks)])
    y = np.array([lab[kinds[k]][:3] for k in ks], np.int64)
    return {"kind": np.array([kinds[k] for k in ks]), "hue": hues, "tiles": tiles, "walk": y[:, 0], "pick": y[:, 1],
            "tog": y[:, 2]}


_ENC = {}


def encode(tiles):
    if "enc" not in _ENC:
        _ENC["enc"] = RP.load(str(ENCODER), "transition", False)
    return RP.vectors(_ENC["enc"], list(tiles))


# ---------------------------------------------------------------- the ensemble

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                     torch.nn.ReLU(), torch.nn.Linear(64, 5))

    def forward(self, z):
        o = self.f(z)
        return o[:, :3], o[:, 3], o[:, 4]              # forward (3 outcomes), pick up, toggle


def fit(Z, D, members=5, steps=3000, lr=3e-3, seed=0):
    """Each member from its own initialisation and batch order; cross-entropy on every property."""
    nets = []
    Zt = torch.as_tensor(Z, dtype=torch.float32, device="cuda")
    yw = torch.as_tensor(D["walk"], device="cuda")
    yp = torch.as_tensor(D["pick"], dtype=torch.float32, device="cuda")
    yt = torch.as_tensor(D["tog"], dtype=torch.float32, device="cuda")
    for m in range(members):
        torch.manual_seed(seed * 100 + m)
        net = Net().cuda()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        g = torch.Generator(device="cuda").manual_seed(seed * 100 + m)
        for t in range(steps):
            i = torch.randint(0, len(Zt), (min(512, len(Zt)),), device="cuda", generator=g)
            w, p, q = net(Zt[i])
            loss = (torch.nn.functional.cross_entropy(w, yw[i])
                    + torch.nn.functional.binary_cross_entropy_with_logits(p, yp[i])
                    + torch.nn.functional.binary_cross_entropy_with_logits(q, yt[i]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        nets.append(net.eval())
    return nets


@torch.no_grad()
def predict(nets, Z):
    """Per member: P(forward outcome) (n, 3), P(pick up takes it), P(toggle can change it)."""
    Zt = torch.as_tensor(Z, dtype=torch.float32, device="cuda")
    W, P, T = [], [], []
    for net in nets:
        w, p, q = net(Zt)
        W.append(torch.softmax(w, -1).cpu().numpy()), P.append(torch.sigmoid(p).cpu().numpy())
        T.append(torch.sigmoid(q).cpu().numpy())
    return np.stack(W), np.stack(P), np.stack(T)


def score(nets, Z, D):
    W, P, T = predict(nets, Z)
    w, p, t = W.mean(0), P.mean(0), T.mean(0)
    right = {"walk": w.argmax(1) == D["walk"], "pick": (p > 0.5) == (D["pick"] == 1), "tog": (t > 0.5) == (D["tog"] == 1)}
    spread = {"walk": W.std(0).max(1), "pick": P.std(0), "tog": T.std(0)}
    return right, spread, {"walk": w, "pick": p, "tog": t}


def ece(prob, y, bins=10):
    """Expected calibration error of a binary probability."""
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (prob >= lo) & (prob < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(e)


# ---------------------------------------------------------------- section 4 and the gate

def check():
    rng = np.random.default_rng(87)
    rep = {"labels per kind": {}}
    for k in KINDS:
        walk, pick, tog, per = labels(k)
        rep["labels per kind"][k] = {"forward": ["moved", "blocked", "ended"][walk], "pick up takes it": bool(pick),
                                     "toggle can change it": bool(tog), "toggle per holding": dict(zip(HOLD, per))}
    # the labels do not depend on the hue: recomputed with 200 random hue pairs per kind against the slot labels
    agree = 0
    for _ in range(200):
        k = KINDS[rng.integers(len(KINDS))]
        a, b = RPL.hues(1, rng)[0], RPL.hues(1, rng)[0]
        K.COLORS[SLOTS[0]], K.COLORS[SLOTS[1]] = a.astype(np.int64), b.astype(np.int64)
        agree += labels(k)[:3] == rep_lab(rep, k)
    D = play(20000, rng)
    rep["hue labels agree"] = f"{agree} of 200"
    rep["instances"] = len(D["kind"])
    rep["positives"] = {"moved": int((D["walk"] == 0).sum()), "ended": int((D["walk"] == 2).sum()),
                        "pick": int(D["pick"].sum()), "tog": int(D["tog"].sum())}
    rep["nearest test hue"] = round(float(np.linalg.norm(D["hue"][:, None] - RPL.TEST[None], axis=-1).min()), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datacheck.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def rep_lab(rep, k):
    r = rep["labels per kind"][k]
    return (["moved", "blocked", "ended"].index(r["forward"]), int(r["pick up takes it"]), int(r["toggle can change it"]))


def named_set():
    """Gate (i): every kind in MiniGrid's six colours and the three held-out ones; floor, wall and goal once."""
    rows = []
    for k in KINDS:
        for c in (NAMED if k in COLOURED else (None,)):
            walk, pick, tog, _ = labels(k)
            rows.append((k, c, draw_named(k, c), walk, pick, tog))
    D = {"kind": np.array([r[0] for r in rows]), "colour": [r[1] for r in rows],
         "walk": np.array([r[3] for r in rows]), "pick": np.array([r[4] for r in rows]),
         "tog": np.array([r[5] for r in rows])}
    return D, np.stack([r[2] for r in rows])


def gate():
    t0 = time.monotonic()
    rng = np.random.default_rng(870)
    train = play(20000, rng)
    test = play(4000, np.random.default_rng(871))
    Ztr, Zte = encode(train["tiles"]), encode(test["tiles"])
    named, ntiles = named_set()
    Zn = encode(ntiles)
    rep = {"note": "Card 087 gate: property play on card 070's encoder", "train": len(Ztr), "test": len(Zte)}
    nets = fit(Ztr, train)
    right, spread, prob = score(nets, Zte, test)
    rep["upper bound (fresh hues, held-out instances)"] = {k: round(float(v.mean()), 4) for k, v in right.items()}
    rep["calibration (ECE, fresh hues)"] = {"pick": round(ece(prob["pick"], test["pick"]), 4),
                                            "tog": round(ece(prob["tog"], test["tog"]), 4)}
    right, spread, prob = score(nets, Zn, named)
    rep["gate (i): MiniGrid's tiles, nine colours"] = {k: f"{int(v.sum())} of {len(v)}" for k, v in right.items()}
    rep["gate (i) wrong"] = [(named["kind"][i], named["colour"][i], prop, round(float(prob[prop][i] if prop != "walk"
                              else prob["walk"][i].max()), 3))
                             for prop, v in right.items() for i in np.flatnonzero(~v)]
    rep["gate (i) spread, largest"] = {k: round(float(v.max()), 4) for k, v in spread.items()}
    # majority baselines, per property on the named set
    rep["majority baseline on (i)"] = {"walk": f"{int(np.bincount(named['walk']).max())} of {len(named['walk'])}",
                                       "pick": f"{int(max(named['pick'].sum(), (1 - named['pick']).sum()))} of {len(named['pick'])}",
                                       "tog": f"{int(max(named['tog'].sum(), (1 - named['tog']).sum()))} of {len(named['tog'])}"}
    torch.save([n.state_dict() for n in nets], OUT / "ensemble.pt")
    np.savez(OUT / "named_vectors.npz", z=Zn, kind=named["kind"], colour=np.array([str(c) for c in named["colour"]]))
    # P19: accuracy on (i) against the number of positives per property in training
    curve = []
    for n in (30, 100, 300, 1000, 3000, 10000):
        sub = {k: v[:n] for k, v in train.items()}
        r, _, _ = score(fit(Ztr[:n], sub, steps=1500, seed=n), Zn, named)
        curve.append({"instances": n, "positives": {"moved": int((sub["walk"] == 0).sum()), "pick": int(sub["pick"].sum()),
                                                    "tog": int(sub["tog"].sum())},
                      "right on (i)": {k: f"{int(v.sum())} of {len(v)}" for k, v in r.items()}})
        print(curve[-1], flush=True)
    rep["P19 curve"] = curve
    # P5: the box left out of property play; its spread against the kinds trained on
    keep = train["kind"] != "box"
    nb = fit(Ztr[keep], {k: v[keep] for k, v in train.items()}, seed=7)
    _, sp, pr = score(nb, Zn, named)
    box = named["kind"] == "box"
    rep["box left out (report)"] = {
        "spread on boxes, mean": {k: round(float(v[box].mean()), 4) for k, v in sp.items()},
        "spread on the other kinds, mean": {k: round(float(v[~box].mean()), 4) for k, v in sp.items()},
        "P(pick up takes it) on boxes": [round(float(x), 3) for x in pr["pick"][box]],
        "P(toggle can change it) on boxes": [round(float(x), 3) for x in pr["tog"][box]]}
    rep["seconds"] = round(time.monotonic() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "P19 curve"}, indent=1, default=str))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    if "--gate" in sys.argv:
        gate()
