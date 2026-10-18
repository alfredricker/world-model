"""Card 111.2: fresh world 1 test-room episodes (card 102's generator and random policy, a new seed).

  bin/prun python tools/card111.2/fresh_world1.py → runs/111.2/world1_test_fresh.npz (no frames: scored on kinds)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card102"))
import worlds as WD                                       # noqa: E402

z = WD.play(WD.World1, 2000, 100, 1112, True)
z.pop("obs")
np.savez_compressed(ROOT / "runs" / "111.2" / "world1_test_fresh.npz", **z)
sem = z["sem"].reshape(len(z["act"]), -1)
same = z["ep"][1:] == z["ep"][:-1]
print("steps", len(z["act"]), "lamps lit / put out", int((((sem[1:] == WD.LIT) & (sem[:-1] == WD.LAMP)).any(1) & same).sum()),
      int((((sem[1:] == WD.LAMP) & (sem[:-1] == WD.LIT)).any(1) & same).sum()))
