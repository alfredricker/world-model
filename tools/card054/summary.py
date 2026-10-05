"""Card 054: one line per evaluated encoder (runs/054/eval.sh outputs): the gap, recall, the planner, conditions.

  python tools/card054/summary.py NAME [NAME ...]
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "runs" / "054"
for n in sys.argv[1:]:
    out = [n]
    g = R / f"gap_{n}.txt"
    if g.exists():
        t = g.read_text().splitlines()
        out.append("gap: " + (t[0][:90] if t else "?") + " | " + next((x for x in t if "every part" in x), "?"))
    r = R / f"recall_{n}.json"
    if r.exists():
        r = json.loads(r.read_text())
        out.append(f"recall: same key {r['colour: locked, same key']['accuracy']}, other key {r['colour: locked, other key']['accuracy']}, "
                   f"all {r['accuracy_all']}, noise agreement {r['same_prediction_two_noise_draws']}, memory categories "
                   f"{r.get('memory_categories_agree_with_simulator')}, tuples {r['memory_tuples']}, identity {r.get('identity')}, "
                   f"transition model {r.get('transition_model')}; toggle admits {r['actions']['toggle']['admitted']}")
    p = R / f"planner_{n}.json"
    if p.exists():
        p = json.loads(p.read_text())
        for w in ("key", "switch", "either", "both"):
            if w in p:
                x = p[w]
                adm = {k: v["conditions"]["admitted"] for k, v in x["model"]["kinds"].items()
                       if isinstance(v, dict) and v.get("conditions")}
                out.append(f"planner {w}: success {x['acting']['success']}, steps {x['acting'].get('mean_steps_when_successful')}, "
                           f"effects lowest {x['heldout_effects']['lowest']}, pass {x['criterion_1']['pass']}; toggle admits {adm.get('toggle')}")
    print("\n  ".join(out))
