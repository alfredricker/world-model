"""Card 090: card 054's encoder and recipe, trained on a stream whose colours are slots with fresh hues each episode.

Nine slot names (s0..s8) replace the nine training colours; MiniGrid draws a colour by its name's RGB, so each
episode redraws every slot's RGB with card 070's sampler (uniform in RGB, at least 60 from MiniGrid's six and the three
held-out hues). Pairings and rules go by name, so a key still opens the door of its slot. No relation term.

  bin/prun python tools/card090/train.py --out runs/090/encoder.json        (writes runs/090/encoder.pt)
  bin/prun python tools/card090/train.py --check                           section 4: runs/090/datacheck.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card052"))
sys.path.insert(0, str(ROOT / "tools" / "card070"))
import effect as EF                                    # noqa: E402
import relplay as RPL                                  # noqa: E402  (the hue sampler)
from minigrid.core import constants as K               # noqa: E402

GN = EF.DR.GN
SLOTS = [f"s{i}" for i in range(9)]
for s in SLOTS:
    if s not in K.COLOR_TO_IDX:
        K.COLORS[s] = np.array([128, 128, 128])
        K.COLOR_TO_IDX[s] = len(K.COLOR_TO_IDX)
        K.IDX_TO_COLOR[K.COLOR_TO_IDX[s]] = s
        K.COLOR_NAMES.append(s)
RNG = np.random.default_rng(90)
COMBOS = sorted({(ev, k) for ev, k, _ in EF.STARTS})
GN.TRAIN[:] = SLOTS                                    # the stream, its play starts and the probe set use slots
EF.STARTS[:] = [(ev, k, c) for c in SLOTS for ev, k in COMBOS]
HUES_DRAWN = []


def redraw():
    h = RPL.hues(len(SLOTS), RNG)
    for s, c in zip(SLOTS, h):
        K.COLORS[s] = np.round(c).astype(np.int64)
    HUES_DRAWN.append(h)


_episode = GN.Generator.episode


def episode(self):
    redraw()
    return _episode(self)


GN.Generator.episode = episode


def check(n=1000):
    """Section 4: kinds per slot and hues per kind; no hue near a test hue; a key opens the door of its slot."""
    s = EF.Stream(90)
    per = Counter()
    eps = 0
    while eps < n:
        s.new_episode()
        eps += 1
        g = s.env.grid
        for o in g.grid:
            if o is not None and o.color in SLOTS and o.type in ("key", "ball", "box", "door"):
                per[(o.type, o.color)] += 1
    H = np.concatenate(HUES_DRAWN)
    rep = {"episodes": n, "kinds x slots seen": len(per), "of": 4 * len(SLOTS),
           "min count per kind and slot": min(per.values()),
           "nearest test hue": round(float(np.linalg.norm(H[:, None] - RPL.TEST[None], axis=-1).min()), 1),
           "distinct hues": int(len(np.unique(np.round(H), axis=0)))}
    # the rule by slot: MiniGrid's Door.toggle with a key of the door's slot and of another slot
    from minigrid.core.world_object import Door, Key
    ok = 0
    for i in range(200):
        a, b = SLOTS[i % 9], SLOTS[(i + 1 + i // 9) % 9]
        d = Door(a, is_locked=True)

        class E:
            carrying = Key(a if i % 2 == 0 else b)
        d.toggle(E(), (0, 0))
        ok += d.is_open == (i % 2 == 0 or a == b)
    rep["rule by slot holds"] = f"{ok} of 200"
    out = ROOT / "runs" / "090"
    out.mkdir(parents=True, exist_ok=True)
    (out / "datacheck.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check" in args:
        check()
        sys.exit()
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    out = get("--out", str(ROOT / "runs" / "090" / "encoder.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    import predflip as PF                              # noqa: E402
    sys.argv = [sys.argv[0]] + ("--seed 399 --mu 0.1 --starts 0.5 --strat 1 --tint 0 --collect-steps 100000 "
                                "--anchor transition --interaction transition --ema 0.99 --restart 100000000 "
                                "--vis-w 1 --gates 0 --pair-m 0.5").split() + [
        "--out", out, "--checkpoints", get("--checkpoints", "12"), "--updates", get("--updates", "2000")]
    PF.main()
