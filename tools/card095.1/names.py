"""Card 095.1 (report only): the name of every token handle in a world, for the evaluation's box check.

  <version 20's flags> bin/prun python tools/card095.1/names.py 2     → runs/095.1/names_tier2.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIER = sys.argv[1]
sys.argv = ["run.py", "tiers", "--tier", TIER, "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

W, info = RUN.T.setup(int(TIER), lambda m: None)
RT = sys.modules["routed"]
names = {int(h): str(RT._nm(W, h)) for h in range(len(W.S.arr))}
(ROOT / "runs" / "095.1" / f"names_tier{TIER}.json").write_text(json.dumps(names) + "\n")
print(len(names), "names")
