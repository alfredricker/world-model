"""Card 072's control (report only): the decoy episodes of fold red with every opening in memory, no trying."""
import json, sys, time
from pathlib import Path
hue = sys.argv[1]
sys.argv = ["run.py", "decoy", "--online", "0", "--memory", "066"]
sys.path.insert(0, "tools/card069")
import numpy as np
import run as RUN
T, R = RUN.T, RUN.R
t00 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
others = [c for c in T.HUES if c != hue]
T.ENVS[1] = R.register()
T.OUT = Path("runs/070/decoy_memory066")
R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
W, info = T.setup(1, log)
R.DECOY.update(door=[hue], decoy=others)
seeds = T.SEED_TEST + 1000 + np.arange(100)
pool = T.F.pool20()
recs = sorted(pool.imap_unordered(RUN.episode, [(1, int(s)) for s in seeds]), key=lambda r: r["seed"])
pool.close()
fails = [r["seed"] for r in recs if not r["success"]]
ok = [r for r in recs if r["success"]]
out = {"note": f"card 072 control: fold {hue} episodes, every opening in memory, no trying (card 070's configuration)",
       "success": len(ok) / 100, "failed_seeds": fails, "mean_steps_when_successful": round(float(np.mean([r["steps"] for r in ok])), 1),
       "random_share": round(sum(r["random"] for r in recs) / sum(r["steps"] for r in recs), 4)}
Path(f"runs/072/norefit/control_{hue}_known.json").write_text(json.dumps(out, indent=1) + "\n")
log(json.dumps(out))
