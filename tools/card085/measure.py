"""Card 085's measurement: one tier's setup (memory and recall built, no episodes), its time and peak memory, on
memory cut to its first fraction of stored rows (episodes are stored in order; the try rows are cut to match).

  WM_SAMPLED=1 <version 18's flags> bin/prun python tools/card085/measure.py --tier 3 --fraction 0.25 \
      --out runs/085/tier3_0.25.json
  (--fits FILE: also save every fitted weight, alpha and beta, for the gate's exactness check;
   --eval FILE: also each action's fitted weights evaluated exactly, for the gate's sampled-against-exact check)
"""
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d=None: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
TIER, FRAC, OUT, FITS = arg("--tier"), float(arg("--fraction", "1")), arg("--out"), arg("--fits")
EVAL = arg("--eval")
sys.argv = ["run.py", "tiers", "--tier", TIER, "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                       # noqa: E402  (version 18's runner and its flags)
import numpy as np                                      # noqa: E402

T = RUN.T
_memory = T.memory


def memory(tier, pool=None):
    D, stats = _memory(tier, pool)
    if FRAC < 1:
        n = int(round(FRAC * len(D["act"])))
        nt = int(np.isin(D["act"][:n], T.INTER).sum())
        D = {k: (v[:nt] if k in ("pres", "stale") else v[:n]) for k, v in D.items()}
        stats = dict(stats, fraction=FRAC, rows=n, tries=nt)
    return D, stats


T.memory = memory
t0 = time.monotonic()
log = lambda m: print(f"[{time.monotonic() - t0:6.0f}s] {str(m)[:400]}", flush=True)
rep = {"tier": int(TIER), "fraction": FRAC}
try:
    W, info = T.setup(int(TIER), log)
    rep["setup_seconds"] = round(time.monotonic() - t0, 1)
    rep["kinds"] = {str(a): {"keys": len(kd.keys), "sets": len(getattr(kd, "ulist", []))} for a, kd in W.kinds.items()}
except BaseException as e:                              # noqa: BLE001  (a failure is a result here)
    rep["failed"] = repr(e)[:400]
    import traceback
    rep["traceback"] = traceback.format_exc()[-3000:]
    print(rep["traceback"], flush=True)
    rep["failed_after_seconds"] = round(time.monotonic() - t0, 1)
    W = None
rep["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
try:
    import torch
    rep["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
except Exception:                                       # noqa: BLE001
    pass
SM = sys.modules.get("sampled")
if SM is not None:
    rep["fits"] = SM.STATS
log(json.dumps(rep))
if OUT:
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(rep, indent=1) + "\n")
if FITS and W is not None:
    arrs = {}
    for a, kd in W.kinds.items():
        for f in ("lam", "lam1", "lam2", "alpha", "beta"):
            v = getattr(kd, f, None)
            if v is not None:
                arrs[f"{a}_{f}"] = np.asarray(v, np.float64)
    np.savez(FITS, **arrs)
if EVAL and W is not None:
    sys.path.insert(0, str(ROOT / "tools" / "card085"))
    import sampled as SM2                               # noqa: E402  (exact_eval only; installs nothing)
    arrs = {}
    for a, kd in W.kinds.items():
        if hasattr(kd, "lam2") and hasattr(kd, "ulist") and len(kd.keys) > 2:
            r = SM2.exact_eval(kd)
            log(f"exact eval {a}: " + json.dumps({k: v for k, v in r.items() if k != "category"}))
            arrs[f"{a}_category"] = r.pop("category")
            for k, v in r.items():
                arrs[f"{a}_{k}"] = np.asarray(v)
    np.savez(EVAL, **arrs)
