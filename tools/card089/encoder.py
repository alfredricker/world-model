"""Card 089: the encoder with an arrangement path and a "made of" path.

  parts 0, 1 (16 numbers): card 054's path, convolutions over the tile (how it is arranged)
  parts 2, 3 (16 numbers): each pixel alone (two 1 x 1 layers), pooled over the 64 pixels (mean and max, or max only),
                           then linear (what it is made of; where a pixel is cannot be seen)

install(pool) replaces the encoder network wherever card 052's Encoder is built (training, and loading through card
053's recall_probe.load), so every later reader gets this architecture; the checkpoint records it ("arch").

  bin/prun python tools/card089/train.py [--pool meanmax|max]     card 054's recipe, from scratch
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card052"))
import effect as EF                                    # noqa: E402

DR = EF.DR
STATE = {"pool": None}


class MadeOf(nn.Module):
    def __init__(self, pool="meanmax"):
        super().__init__()
        self.pool = pool
        self.arr = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                 nn.ReLU(), nn.Flatten(), nn.Linear(32 * 16, 16))
        self.px = nn.Sequential(nn.Conv2d(3, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 1), nn.ReLU())
        self.mat = nn.Linear(64 if pool == "meanmax" else 32, 16)

    def forward(self, x):
        h = self.px(x).flatten(2)                        # (n, 32, 64): each pixel's own features
        p = h.amax(2) if self.pool == "max" else torch.cat([h.mean(2), h.amax(2)], 1)
        return torch.cat([self.arr(x), self.mat(p)], 1)


def install(pool="meanmax"):
    if STATE["pool"] is not None:
        assert STATE["pool"] == pool
        return
    STATE["pool"] = pool
    init0 = DR.Encoder.__init__

    def __init__(self, seed, dev="cuda"):
        init0(self, seed, dev)
        torch.manual_seed(seed)
        self.enc = MadeOf(pool).to(dev)
        self.opt.param_groups[0]["params"] = list(self.enc.parameters()) + list(self.dec.parameters()) + [self.books]
        self.opt.state.clear()

    DR.Encoder.__init__ = __init__
