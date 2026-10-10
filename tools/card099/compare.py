"""Card 099: version 25's episodes against version 24's, seed by seed (every count the runner keeps per episode,
times aside). Writes runs/099/compare.json. With arguments, any two runs' files (card 100 uses it):

  python3 tools/card099/compare.py
  python3 tools/card099/compare.py <before: dir/prefix> <after: dir/prefix> <out.json>   e.g. 099/lean 100/a
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [("tier 1", "095.5/cond_tier1.json", "099/lean_tier1.json"),
         ("tier 2", "095.5/cond_tier2.json", "099/lean_tier2.json"),
         ("tier 3", "095.5/cond_tier3.json", "099/lean_tier3.json"),
         ("fresh tier 3", "095.5/fresh_cond_tier3.json", "099/fresh_lean_tier3.json")]
if len(sys.argv) == 4:
    a0, b0 = sys.argv[1].split("/"), sys.argv[2].split("/")
    PAIRS = [(f"tier {t}", f"{a0[0]}/{a0[1]}_tier{t}.json", f"{b0[0]}/{b0[1]}_tier{t}.json") for t in (1, 2, 3)]
    PAIRS.append(("fresh tier 3", f"{a0[0]}/fresh_{a0[1]}_tier3.json", f"{b0[0]}/fresh_{b0[1]}_tier3.json"))
OUTF = ROOT / "runs" / (sys.argv[3] if len(sys.argv) == 4 else "099/compare.json")
TIMES = ("seconds", "_seconds")
REPORT = {"rel_weight_end": "card 069's report on rel:P's weight, a removed part",
          "sc_new": "misses of arm C's cache, which persists across a worker's episodes (scheduling)",
          "props_prior_tiles": "misses of card 087's cache, which persists across a worker's episodes (scheduling)"}


def norm(x):
    """Card 074's weighings with card 042's code ids dropped (numbered in first-seen order, a report label)."""
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        if len(x) == 2 and isinstance(x[0], int) and isinstance(x[1], list) and len(x[1]) > 8:
            return ["id", x[1]]
        return [norm(v) for v in x]
    return x


def summary(d):
    return {"success": f"{d['successes']}/{d['episodes']}", "steps": d["mean_steps_when_successful"],
            "s_per_step": d["seconds_per_step"], "model_seconds": d["setup"].get("model_seconds")}


out = {}
for name, a, b in PAIRS:
    pa, pb = ROOT / "runs" / a, ROOT / "runs" / b
    if not pb.exists():
        continue
    A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
    ea = {r["seed"]: r for r in A["per_episode"]}
    eb = {r["seed"]: r for r in B["per_episode"]}
    same, differ = 0, []
    for s in sorted(ea):
        ra, rb = ea[s], eb.get(s)
        if rb is None:
            differ.append({"seed": s, "missing": True})
            continue
        keys = [k for k in ra if k in rb and not k.endswith(TIMES) and k not in REPORT]
        d = {k: [ra[k], rb[k]] for k in keys if (norm(ra[k]) if k == "weighings" else ra[k])
             != (norm(rb[k]) if k == "weighings" else rb[k])}
        if d:
            differ.append({"seed": s, **d})
        else:
            same += 1
    out[name] = {"not compared": REPORT, "weighings": "compared with code ids dropped",
                 "version 24": summary(A), "version 25": summary(B), "episodes the same": f"{same}/{len(ea)}",
                 "only in one": sorted(set(k for r in A["per_episode"] for k in r) ^
                                       set(k for r in B["per_episode"] for k in r)),
                 "differ": differ}
OUTF.write_text(json.dumps(out, indent=1) + "\n")
for name, r in out.items():
    print(name, r["version 24"], r["version 25"], "same", r["episodes the same"])
    for d in r["differ"][:12]:
        print("   ", json.dumps(d)[:300])
