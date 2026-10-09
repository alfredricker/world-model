"""Run chosen seeds of a tier with the flags in the environment (card 092's gate): one process, episodes in turn.

  <flags> bin/prun python tools/card092/seeds.py 2 out.json 1002008 1002012 ...
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tier, out, seeds = int(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
sys.argv = ["run.py", "tiers", "--tier", str(tier), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

T = RUN.T
T.TRACE = os.environ.get("WM_TRACE") == "1"
if os.environ.get("WM_PV_DEBUG") == "1":
    T.PV.DEBUG = True
    _explore = T.PV.explore

    def _dbg_explore(pl, st):
        n0 = T.PV.STATS.get("open_to_see", 0)
        a = _explore(pl, st)
        print(f"   explore -> {a} via open_to_see {T.PV.STATS.get('open_to_see', 0) > n0} look_j {getattr(pl, 'look_j', None)}",
              flush=True)
        return a

    T.PV.explore = _dbg_explore
T.CAP_SECONDS = float(os.environ.get("WM_CAP", T.CAP_SECONDS))
W0, _ = T.setup(tier, lambda m: print(m, flush=True))
if os.environ.get("WM_TIMING") == "1":                 # seconds per part of a step, printed per episode
    import time
    TIMES = {}

    def _timed(obj, name, key):
        f = getattr(obj, name)

        def g(*a, **k):
            t0 = time.monotonic()
            try:
                return f(*a, **k)
            finally:
                TIMES[key] = TIMES.get(key, 0.0) + time.monotonic() - t0
        setattr(obj, name, g)
    _timed(type(W0), "learn_try", "learn_try")
    for cls in type(W0).__mro__:
        pass
    PL = T.S7.Plan047
    for nm in ("choose", "observe", "step", "view"):
        _timed(PL, nm, nm)
    _timed(T.PV, "fallback", "fallback")
recs = []
for s in seeds:
    r = RUN.episode((tier, s))
    recs.append(r)
    print(f"seed {s}: success {r['success']} steps {r['steps']} random {r['random']} explore {r['explore']} "
          f"reveal {r.get('reveal_acts')} {r.get('reveal_open')} {r.get('reveal_none')} {r['seconds']}s", flush=True)
    if os.environ.get("WM_TIMING") == "1":
        print("   seconds by part:", {k: round(v, 1) for k, v in TIMES.items()}, flush=True)
        TIMES.clear()
    out.write_text(json.dumps({"tier": tier, "per_episode": recs}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
