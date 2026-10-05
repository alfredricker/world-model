"""Card 069, step (a): card 054's encoder trained further with the relation term.

Card 054's recipe unchanged (runs/054/b.sh: transitions, the visibility margin, the pair margin m = 0.5, card 052's
stream with play starts at 0.5, no tint), started from its saved encoder (seed 399), plus one term. The relation
between the tile in front and the held tile is the L1 distance between their vectors under one learned projection P
(r x 32, r = 8), shared by both tiles:

  rel = || P z_front - P z_held ||_1

A toggle's outcome (the front tile changed) is predicted from each tile on its own and from the relation, which is
the only place the two tiles meet (the relational bottleneck):

  logit = a + u . z_front + v . z_held - g(z_front) * b * rel,      g(z_front) = sigmoid(s . z_front + c)

u, v and s read one tile each, so no pair of tiles can be fitted as a table; g lets the relation count only where the
front tile makes it matter (a closed door opens whatever is held). Trained on card 052's 512 toggle tries per update
(half changed) made while the hand holds something (by pixels: card 052's toggle draw is mostly toggles at walls
with an empty hand), weight REL_W = 1, beside the recipe's terms.

  bin/prun python tools/card069/train.py --checkpoints 12 --out runs/069/encoder.json     (writes encoder.pt)
"""
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card052"))
import effect as EF                                    # noqa: E402

fn = torch.nn.functional
R = 8
TOG = 5
HELD_THRESH = 0.05                                     # mean absolute pixel difference; noise is 4/255
INIT = ROOT / "runs" / "054" / "b_m0.5_399.pt"
RECIPE = ("--seed 399 --mu 0.1 --starts 0.5 --strat 1 --tint 0 --anchor transition "
          "--interaction transition --ema 0.99 --restart 100000000 --vis-w 1 --gates 0 --pair-m 0.5").split()

_init = EF.Encoder.__init__
_extra = EF.Encoder.extra


def __init__(self, seed, dev="cuda"):
    _init(self, seed, dev)
    ck = torch.load(INIT)
    self.enc.load_state_dict(ck["enc"])
    self.dec.load_state_dict(ck["dec"])
    self.trans.load_state_dict(ck["trans"])
    with torch.no_grad():
        self.books.copy_(ck["books"].to(dev))
        self.ema_w = fn.normalize(self.books.detach().clone(), dim=-1)
        self.target.load_state_dict(self.enc.state_dict())
    n = EF.K * EF.DIM
    g = torch.Generator().manual_seed(seed)
    self.P = nn.Parameter((torch.randn(R, n, generator=g) / math.sqrt(n)).to(dev))
    self.u, self.v, self.s = (nn.Linear(n, 1).to(dev) for _ in range(3))
    self.ab = nn.Parameter(torch.tensor([0.0, 0.5413], device=dev))          # a = 0, b = softplus(.) = 1
    self.opt.add_param_group({"params": [self.P, self.ab] + [p for m in (self.u, self.v, self.s) for p in m.parameters()]})


def rel(self, zf, zh):
    return ((zf - zh) @ self.P.T).abs().sum(1)


REL_BUF = (deque(maxlen=10000), deque(maxlen=10000))   # toggles with something in the hand: not changed, changed
REL_N = 512
RNG = np.random.default_rng(69)
EMPTY = {}
_step = EF.Stream.step


def step(self):
    """Card 052's step; toggles tried while the hand holds something (the held tile is not the empty hand's tile,
    by pixels, beyond the noise) also go to the relation's own buffer: among all toggles they are rare."""
    out = _step(self)
    if "tile" not in EMPTY:
        EMPTY["tile"] = EF.DR.GN.clean_tile(None).astype(np.float64)
    for t in self.tries:
        a, f, h, o = t[:4]
        if a == TOG and np.abs(h.astype(np.float64) - EMPTY["tile"]).mean() / 255 > HELD_THRESH:
            REL_BUF[int(o & 1)].append((f, h, o))
    return out


def relation(self, _tries):
    """The relation's tries: half changed, from the buffer above (the run loop's own toggle draw is not used)."""
    ch, no = REL_BUF[1], REL_BUF[0]
    if len(ch) < 16 or len(no) < 16:
        return None
    nc = min(REL_N // 2, len(ch))
    tries = [ch[i] for i in RNG.choice(len(ch), size=nc, replace=False)] + \
            [no[i] for i in RNG.choice(len(no), size=min(REL_N - nc, len(no)), replace=False)]
    n = len(tries)
    zf = self.pieces(self.x([t[0] for t in tries])).reshape(n, -1)
    zh = self.pieces(self.x([t[1] for t in tries])).reshape(n, -1)
    y = torch.tensor([float(t[2] & 1) for t in tries], device=self.dev)
    gate = torch.sigmoid(self.s(zf)).squeeze(1)
    logit = self.ab[0] + self.u(zf).squeeze(1) + self.v(zh).squeeze(1) - gate * fn.softplus(self.ab[1]) * rel(self, zf, zh)
    self.rel_used = n
    return fn.binary_cross_entropy_with_logits(logit, y)


def extra(self):
    out = _extra(self)
    tries = getattr(self, "_rel", None)
    if EF.REL_W > 0 and tries:
        r = self.relation(tries)
        if r is not None:
            self.last_relation = float(r.detach())
            self.rel_buf = (len(REL_BUF[0]), len(REL_BUF[1]))
            out = out + EF.REL_W * r
    return out


EF.Encoder.__init__ = __init__
EF.Encoder.relation = relation
EF.Stream.step = step
EF.Encoder.extra = extra

_save = torch.save


def save(obj, path, *a, **k):
    """The run loop saves card 054's fields; the relation's parameters are kept with them."""
    enc = save.enc
    if isinstance(obj, dict) and "enc" in obj and enc is not None:
        obj = {**obj, "P": enc.P.detach().cpu(), "ab": enc.ab.detach().cpu(),
               "u": enc.u.state_dict(), "v": enc.v.state_dict(), "s": enc.s.state_dict()}
    return _save(obj, path, *a, **k)


save.enc = None
torch.save = save
_init2 = EF.Encoder.__init__


def __init2__(self, *a, **k):
    _init2(self, *a, **k)
    save.enc = self


EF.Encoder.__init__ = __init2__


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    last = str(get("--out", str(ROOT / "runs" / "069" / "encoder.json"))).replace(".json", "_last.pt")

    def check(enc):
        """Each checkpoint also saved (a run stopped early keeps its last checkpoint)."""
        torch.save({"enc": enc.enc.state_dict(), "dec": enc.dec.state_dict(), "books": enc.books.detach().cpu(),
                    "theta": None, "trans": enc.trans.state_dict(), "gates": None}, last)
        return {"relation_buffer_not_changed_changed": [len(REL_BUF[0]), len(REL_BUF[1])]}

    EF.CHECK = check
    sys.argv = [sys.argv[0]] + RECIPE + ["--rel", get("--rel", "1"), "--checkpoints", get("--checkpoints", "12"),
                                         "--updates", get("--updates", "2000"),
                                         "--out", get("--out", str(ROOT / "runs" / "069" / "encoder.json"))]
    EF.main()


if __name__ == "__main__":
    main()
