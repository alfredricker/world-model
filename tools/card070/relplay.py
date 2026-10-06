"""Card 070: relation play, its data check, and step A (P alone on card 054's frozen vectors).

Relation play: the agent stands at a locked door of a fresh hue and toggles, holding the door's key (1/2), a key of
another fresh hue (1/4), a ball or box of the door's hue (1/8), or nothing (1/8). The outcome is read from pixels:
the front tile after the toggle differs from before beyond card 053's noise threshold. Hues are uniform in RGB, none
within MIN_D of MiniGrid's six or the three held-out hues, none darker than DARK in every channel, and another key's
hue at least MIN_D from the door's.

Tiles are drawn as card 052's generator draws them (clean_tile, then noise 4/255, no tint). MiniGrid fills each
object's shapes with its colour, so a tile is linear in that colour: every tile is made from three renders per kind
(the colour's red, green and blue at 255), within 1.2 levels of a direct render (checked before the run).

  bin/prun python tools/card070/relplay.py --check                   section 4, runs/070/datacheck.json
  bin/prun python tools/card070/relplay.py --step-a                  writes runs/070/a_frozen.pt (card 054 + P)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card052"))
import generator as GN                                 # noqa: E402
from minigrid.core import constants as K               # noqa: E402
from minigrid.core.world_object import Ball, Box, Door, Key   # noqa: E402

MIN_D, DARK = 60.0, 60.0
VIS_THRESH = 0.025965073529411763                      # card 054's run (runs/054/b_m0.5_399.json)
TEST = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (112, 39, 195), (255, 255, 0), (100, 100, 100),
                 GN.NEW_HUES["pink"], GN.NEW_HUES["brown"], GN.NEW_HUES["teal"]], np.float64)
assert all(np.array_equal(TEST[i], K.COLORS[c]) for i, c in enumerate(("red", "green", "blue", "purple", "yellow", "grey")))
SLOT = "fresh"
if SLOT not in K.COLOR_TO_IDX:
    K.COLORS[SLOT] = np.array([0, 0, 0])
    K.COLOR_TO_IDX[SLOT] = len(K.COLOR_TO_IDX)
    K.IDX_TO_COLOR[K.COLOR_TO_IDX[SLOT]] = SLOT
KINDS = {"locked": lambda: Door(SLOT, is_open=False, is_locked=True), "open": lambda: Door(SLOT, is_open=True),
         "key": lambda: Key(SLOT), "ball": lambda: Ball(SLOT), "box": lambda: Box(SLOT)}


def _basis():
    out = {}
    for k, make in KINDS.items():
        planes = []
        for i in range(3):
            K.COLORS[SLOT] = (255 * np.eye(3)[i]).astype(np.int64)
            planes.append(GN.clean_tile(make()).astype(np.float64) / 255.0)
        out[k] = np.stack(planes)                      # (3, 8, 8, 3): the tile per unit of red, green, blue
    K.COLORS[SLOT] = np.array([0, 0, 0])
    return out


BASIS = _basis()
EMPTY = GN.clean_tile(None).astype(np.float64) / 255.0


def hues(n, rng, away=None):
    """n hues, uniform in RGB, MIN_D from the nine test hues (and from `away`, row by row), not too dark."""
    out = np.zeros((n, 3))
    todo = np.arange(n)
    while len(todo):
        c = rng.uniform(0, 255, (len(todo), 3))
        ok = (c.max(1) >= DARK) & (np.linalg.norm(c[:, None] - TEST[None], axis=-1).min(1) >= MIN_D)
        if away is not None:
            ok &= np.linalg.norm(c - away[todo], axis=1) >= MIN_D
        out[todo[ok]] = c[ok]
        todo = todo[~ok]
    return out


def draw(kind, c, rng):
    """Tiles of one kind in colours c (n, 3), with card 052's noise; uint8 like the stream's tiles."""
    img = np.einsum("nc,chwk->nhwk", c / 255.0, BASIS[kind])
    img = np.clip(img + rng.normal(0, GN.NOISE, img.shape), 0, 1)
    return (img * 255).astype(np.uint8)


def batch(n, rng):
    """n tries: (front before, held, front after, opened by the simulator's rule, changed by pixels)."""
    door = hues(n, rng)
    u = rng.random(n)
    held_kind = np.where(u < 0.5, "key", np.where(u < 0.75, "other", np.where(u < 0.8125, "ball",
                         np.where(u < 0.875, "box", "nothing"))))
    other = hues(n, rng, away=door)
    before = draw("locked", door, rng)
    held = np.empty_like(before)
    for k, c in (("key", door), ("other", other), ("ball", door), ("box", door)):
        m = held_kind == k
        if m.any():
            held[m] = draw("key" if k == "other" else k, c[m], rng)
    m = held_kind == "nothing"
    if m.any():
        img = np.clip(EMPTY[None] + rng.normal(0, GN.NOISE, (int(m.sum()),) + EMPTY.shape), 0, 1)
        held[m] = (img * 255).astype(np.uint8)
    opened = held_kind == "key"                        # MiniGrid's Door.toggle: a key of the door's colour
    after = draw("locked", door, rng)
    if opened.any():
        after[opened] = draw("open", door[opened], rng)
    changed = np.abs(after.astype(np.float64) - before.astype(np.float64)).mean((1, 2, 3)) / 255 > VIS_THRESH
    return before, held, after, opened, changed, held_kind, door, other


def toggle_rule_check(rng, n=200):
    """MiniGrid's own Door.toggle on the same situations agrees with `opened` (the colour slot names the hue)."""
    agree = 0
    for _ in range(n):
        kind = rng.choice(["key", "other", "ball", "nothing"])
        d = Door(SLOT, is_open=False, is_locked=True)
        carrying = {"key": Key(SLOT), "other": Key("red"), "ball": Ball(SLOT), "nothing": None}[kind]
        env = type("E", (), {"carrying": carrying})()
        agree += bool(d.toggle(env, (0, 0))) == (kind == "key")
    return agree, n


def check():
    rng = np.random.default_rng(70)
    rows = [batch(512, rng) for _ in range(10)]
    opened = np.concatenate([r[3] for r in rows])
    changed = np.concatenate([r[4] for r in rows])
    door = np.concatenate([r[6] for r in rows])
    kinds = np.concatenate([r[5] for r in rows])
    other = np.concatenate([r[7] for r in rows])[kinds == "other"]
    a, n = toggle_rule_check(rng)
    out = {"tries": int(len(opened)), "opened_share": round(float(opened.mean()), 4),
           "pixels_agree_with_simulator": f"{int((opened == changed).sum())}/{len(opened)}",
           "door_hue_nearest_test_hue": round(float(np.linalg.norm(door[:, None] - TEST[None], axis=-1).min()), 1),
           "other_key_hue_nearest_test_hue": round(float(np.linalg.norm(other[:, None] - TEST[None], axis=-1).min()), 1),
           "held": {k: int((kinds == k).sum()) for k in ("key", "other", "ball", "box", "nothing")},
           "minigrid_toggle_rule_agrees": f"{a}/{n}"}
    print(json.dumps(out, indent=1))
    (ROOT / "runs" / "070").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "070" / "datacheck.json").write_text(json.dumps(out, indent=1) + "\n")


def step_a(updates=3000, r=8, lr=0.01):
    """P (r x 32), a and b on card 054's frozen vectors: P(opened) = sigmoid(a - b * ||P z_front - P z_held||_1)."""
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "card053"))
    import recall_probe as RP                          # noqa: E402
    src = ROOT / "runs" / "054" / "b_m0.5_399.pt"
    enc = RP.load(str(src), "transition", False)
    rng = np.random.default_rng(71)
    g = torch.Generator().manual_seed(70)
    P = torch.nn.Parameter((torch.randn(r, 32, generator=g) / np.sqrt(32)).double().cuda())
    ab = torch.nn.Parameter(torch.tensor([0.0, 0.5413], dtype=torch.float64, device="cuda"))
    opt = torch.optim.Adam([P, ab], lr=lr)
    log = []
    for t in range(updates):
        f, h, _, _, y, *_ = batch(512, rng)
        with torch.no_grad():
            zf = enc.pieces(enc.x(f)).reshape(len(f), -1).double()
            zh = enc.pieces(enc.x(h)).reshape(len(h), -1).double()
        rel = ((zf - zh) @ P.T).abs().sum(1)
        logit = ab[0] - torch.nn.functional.softplus(ab[1]) * rel
        yt = torch.as_tensor(y, dtype=torch.float64, device="cuda")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (t + 1) % 500 == 0:
            acc = float(((logit > 0) == (yt > 0)).double().mean())
            log.append({"update": t + 1, "loss": round(float(loss), 4), "batch_accuracy": round(acc, 4)})
            print(log[-1], flush=True)
    ck = torch.load(src)
    ck["P"] = P.detach().float().cpu()
    (ROOT / "runs" / "070" / "a_frozen_ab.json").write_text(json.dumps({"ab": ab.detach().cpu().tolist()}) + "\n")
    torch.save(ck, ROOT / "runs" / "070" / "a_frozen.pt")
    (ROOT / "runs" / "070" / "a_frozen.json").write_text(json.dumps({"note": "Card 070 step A", "log": log}, indent=1) + "\n")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    if "--step-a" in sys.argv:
        step_a()
