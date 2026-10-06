"""Card 073's criterion 1: the decoy world with the key known (every opening in memory), one fold's door hue, 100
episodes; version 16 (WM_TRYING=1), with card 073's order when WM_ORDER=1. Seeds as card 072's folds.

  WM_TRYING=1 WM_ORDER=1 WM_REL_ENCODER=runs/070/encoder.pt bin/prun python tools/card073/known.py red runs/073/known_red.json
"""
import json
import sys
import time
from pathlib import Path

hue, OUT = sys.argv[1], Path(sys.argv[2])
TRACE = "--trace" in sys.argv
SEEDS = next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == "--seeds"), None)
sys.argv = ["run.py", "decoy", "--online", "0", "--memory", "066"] + (["--trace"] if TRACE else [])
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import numpy as np                                     # noqa: E402
import run as RUN                                      # noqa: E402

T, R = RUN.T, RUN.R
t00 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
others = [c for c in T.HUES if c != hue]
T.ENVS[1] = R.register()
T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
W, info = T.setup(1, log)
R.DECOY.update(door=[hue], decoy=others)
seeds = [int(s) for s in SEEDS.split(",")] if SEEDS else (T.SEED_TEST + 1000 + np.arange(100)).tolist()
if len(seeds) == 1:
    recs = [RUN.episode((1, seeds[0]))]
else:
    pool = T.F.pool20()
    recs = sorted(pool.imap_unordered(RUN.episode, [(1, int(s)) for s in seeds]), key=lambda r: r["seed"])
    pool.close()
ok = [r for r in recs if r["success"]]
keys = [k for k in (RUN.ORDER.STATS if RUN.ORDER else {})] + [k for k in (RUN.CONFLICTS.STATS if RUN.CONFLICTS else {})] + [k for k in (RUN.TIES.STATS if RUN.TIES else {})]
out = {"note": f"card 073: fold {hue}'s door hue, every opening in memory (key known)", "order": RUN.ORDER is not None,
       "conflicts": RUN.CONFLICTS is not None, "ties": RUN.TIES and RUN.TIES.MODE, "split": bool(RUN.TIES and RUN.TIES.SPLIT),
       "trying": RUN.TRY is not None, "episodes": len(recs), "success": round(len(ok) / len(recs), 4),
       "failed_seeds": [r["seed"] for r in recs if not r["success"]],
       "mean_steps_when_successful": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else None,
       "random_share": round(sum(r["random"] for r in recs) / sum(r["steps"] for r in recs), 4),
       "seconds_per_step": round(sum(r["seconds"] for r in recs) / sum(r["steps"] for r in recs), 4),
       "order_counts": {k: int(sum(r.get(k, 0) for r in recs)) for k in keys}, "per_episode": recs}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
log(json.dumps({k: v for k, v in out.items() if k != "per_episode"}))
