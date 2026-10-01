"""Card 040's data: the four familiar worlds with keys and doors in four colours (red, green, blue, grey). Yellow and
purple stay unseen in training and keep their tile codes (90-129); grey's tiles get codes 130-149. Nothing in src/
changes: grey is appended to the renderer's object list before any tile table is built, and the layouts draw from
four colours. Every path that card 034's data, card 038's encoders and card 040's parts write is redirected to
runs/040_*, so nothing earlier is overwritten. Grey is also the walls' and the switch-off ball's colour.

  bin/prun python tools/card040/four_colours.py --collect
  bin/prun python tools/card040/four_colours.py --encoders --seeds 399-404
  bin/prun python tools/card040/four_colours.py --check --arm B --tokens vector --score dist --bound \\
      --holdout front --roles tokens --worlds key,either,both --layouts 10 --out runs/040_check4_B.json
"""
import sys
from pathlib import Path

from worldmodel.envs import keydoor_render as KR
from worldmodel.envs import logicdoor as LD

for _col in ("yellow", "purple", "grey"):          # yellow and purple in the order cards 033 and 031 add them
    for _o in [("key", _col)] + [("door", _col, st) for st in range(3)]:
        if _o not in KR.OBJ_INDEX:
            KR.OBJ_INDEX[_o] = len(KR.OBJECTS)
            KR.OBJECTS.append(_o)
KR.N_CODES = len(KR.OBJECTS) * 5
LD.COLOURS = ("red", "green", "blue", "grey")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card040"))
import attention_planner as AP                                  # noqa: E402

VP, T = AP.VP, AP.VP.T
T.SUF = "_040four"
T.DATA = ROOT / "runs" / "040_data.pkl"
T.MEMORY = ROOT / "runs" / "040_memory.pkl"
T.REF = ROOT / "runs" / "040_ref.json"
VP.ENC_B = ROOT / "runs" / "040_encB"
VP.ENC_A = ROOT / "runs" / "040_encA_unused"
AP.PARTS_DIR = ROOT / "runs" / "040_parts4"
assert KR.OBJ_INDEX[("key", "yellow")] == 18 and KR.OBJ_INDEX[("key", "purple")] == 22, "test colours moved"
assert LD.COLOURS[-1] == "grey" and KR.N_CODES == 150


def main():
    if "--collect" in sys.argv:
        T.main()
    elif "--encoders" in sys.argv:
        VP.main()
    else:
        AP.main()


if __name__ == "__main__":
    main()
