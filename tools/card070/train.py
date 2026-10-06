"""Card 070, step B: card 054's encoder trained further with the relation as the only path, on relation play.

Card 054's recipe and stream unchanged (runs/054/b.sh), from its saved encoder, plus one term per update on 512 fresh
tries of relation play (relplay.py):

  P(the front tile changed) = sigmoid(a - b * ||P z_front - P z_held||_1)

with P (8 x 32), a and b one set for every try (no term reads one tile alone), starting from step A's.

  bin/prun python tools/card070/train.py --checkpoints 12 --out runs/070/encoder.json      (writes encoder.pt)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card052"))
sys.path.insert(0, str(ROOT / "tools" / "card070"))
import effect as EF                                    # noqa: E402
import relplay as RPL                                  # noqa: E402

fn = torch.nn.functional
INIT = ROOT / "runs" / "054" / "b_m0.5_399.pt"
STEP_A = ROOT / "runs" / "070" / "a_frozen.pt"
RECIPE = ("--seed 399 --mu 0.1 --starts 0.5 --strat 1 --tint 0 --anchor transition "
          "--interaction transition --ema 0.99 --restart 100000000 --vis-w 1 --gates 0 --pair-m 0.5").split()
REL_N = 512
RNG = np.random.default_rng(72)
_init = EF.Encoder.__init__
_extra = EF.Encoder.extra
ENC = {}


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
    a = torch.load(STEP_A)
    self.P = nn.Parameter(a["P"].float().to(dev))
    lg = __import__("json").loads((ROOT / "runs" / "070" / "a_frozen_ab.json").read_text())
    self.ab = nn.Parameter(torch.tensor(lg["ab"], dtype=torch.float32, device=dev))
    self.opt.add_param_group({"params": [self.P, self.ab]})
    ENC["enc"] = self


def relation(self):
    f, h, _, _, y, *_ = RPL.batch(REL_N, RNG)
    n = len(f)
    zf = self.pieces(self.x(f)).reshape(n, -1)
    zh = self.pieces(self.x(h)).reshape(n, -1)
    rel = ((zf - zh) @ self.P.T).abs().sum(1)
    logit = self.ab[0] - fn.softplus(self.ab[1]) * rel
    yt = torch.as_tensor(y, dtype=torch.float32, device=self.dev)
    self.last_rel_acc = float(((logit > 0) == (yt > 0)).float().mean())
    return fn.binary_cross_entropy_with_logits(logit, yt)


def extra(self):
    r = self.relation()
    self.last_relation = float(r.detach())
    self.rel_used = REL_N
    return _extra(self) + EF.REL_W * r


EF.Encoder.__init__ = __init__
EF.Encoder.relation = relation
EF.Encoder.extra = extra
_save = torch.save


def save(obj, path, *a, **k):
    if isinstance(obj, dict) and "enc" in obj and "enc" in ENC:
        obj = {**obj, "P": ENC["enc"].P.detach().cpu(), "ab": ENC["enc"].ab.detach().cpu()}
    return _save(obj, path, *a, **k)


torch.save = save


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    out = get("--out", str(ROOT / "runs" / "070" / "encoder.json"))
    last = out.replace(".json", "_last.pt")

    def check(enc):
        torch.save({"enc": enc.enc.state_dict(), "dec": enc.dec.state_dict(), "books": enc.books.detach().cpu(),
                    "theta": None, "trans": enc.trans.state_dict(), "gates": None}, last)
        return {"relation_batch_accuracy_last": round(getattr(enc, "last_rel_acc", float("nan")), 4)}

    EF.CHECK = check
    sys.argv = [sys.argv[0]] + RECIPE + ["--rel", "1", "--checkpoints", get("--checkpoints", "12"),
                                         "--updates", get("--updates", "2000"), "--out", out]
    EF.main()


if __name__ == "__main__":
    main()
