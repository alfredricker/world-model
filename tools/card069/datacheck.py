"""Card 069, section 4: toggle tries in card 054's training stream (card 052's generator, play starts at 0.5, no
tint), counted by what the evaluator says is in front and in the hand (report only; training never reads these).

  bin/prun python tools/card069/datacheck.py --steps 200000
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "card052"))
import effect as EF                                    # noqa: E402
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR   # noqa: E402

TOG = 5


def kind(k):
    if k is None:
        return "nothing"
    o = IDX_TO_OBJECT[k[0]]
    if o == "door":
        return "door " + ("open", "closed", "locked")[k[2]]
    return o


def main():
    args = sys.argv[1:]
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 200000
    EF.START_SHARE = 0.5
    EF.DR.GN.TINT = 0.0
    s = EF.Stream(399)
    c = Counter()
    t0 = time.monotonic()
    for _ in range(steps):
        s.step()
        for t in s.tries:
            a, f, h, o = t[:4]
            if a != TOG:
                continue
            k0, c0 = t[4]
            fk, hk = kind(k0), kind(c0)
            c["toggles"] += 1
            if fk == "door locked":
                if hk == "key":
                    same = IDX_TO_COLOR[k0[1]] == IDX_TO_COLOR[c0[1]]
                    c[f"locked door, {'matching' if same else 'other'} key, {'opened' if o & 1 else 'not opened'}"] += 1
                else:
                    c[f"locked door, {hk} held, {'opened' if o & 1 else 'not opened'}"] += 1
            elif fk == "door closed":
                c[f"closed door, {'opened' if o & 1 else 'not opened'}"] += 1
    out = {"steps": steps, "seconds": round(time.monotonic() - t0, 1), **dict(sorted(c.items()))}
    print(json.dumps(out, indent=1))
    (ROOT / "runs" / "069").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "069" / "datacheck.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
