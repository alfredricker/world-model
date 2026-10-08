"""Card 089: the made-of encoder trained with card 054's recipe (runs/054/b.sh, pair margin 0.5, seed 399), from
scratch, no relation term.

  bin/prun python tools/card089/train.py --pool meanmax --out runs/089/encoder_meanmax.json   (writes .pt beside it)
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card089"))
import encoder as E89                                  # noqa: E402

args = sys.argv[1:]
get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
POOL = get("--pool", "meanmax")
OUT = get("--out", str(ROOT / "runs" / "089" / f"encoder_{POOL}.json"))
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
E89.install(POOL)
_save = torch.save


def save(obj, path, *a, **k):
    if isinstance(obj, dict) and "enc" in obj:
        obj = {**obj, "arch": f"made_of:{POOL}"}
    return _save(obj, path, *a, **k)


torch.save = save
sys.path.insert(0, str(ROOT / "tools" / "card052"))
import predflip as PF                                  # noqa: E402

sys.argv = [sys.argv[0]] + ("--seed 399 --mu 0.1 --starts 0.5 --strat 1 --colour 1 "
                            "--tint 0 --collect-steps 100000 --anchor transition --interaction transition --ema 0.99 "
                            "--restart 100000000 --vis-w 1 --gates 0 --pair-m 0.5").split() + ["--out", OUT, "--checkpoints", get("--checkpoints", "12"), "--updates", get("--updates", "2000")]
PF.main()
