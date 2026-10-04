"""Card 051, change 2 (the readable half only): the rules recall holds, printed.

For pick up, toggle and drop: every own situation in stored memory (step 1's index: the front and held tiles'
code tuples and the admitted view values), with its tries and its most frequent outcome, named by the evaluator
(names are for the reader only; the agent never sees them). Nothing in acting changes.

  bin/prun python tools/card051/rules.py --seed 400 --out runs/051/rules_400.md
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1 as S1                                     # noqa: E402

VP = S1.VP
OUT = {0: "nothing changes", 1: "the front tile changes", 2: "the hand changes", 3: "front and hand change"}


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    seed = int(get("--seed", "400"))
    W = S1.setup("index", seed)
    nm = lambda h: W.judge.name(int(h))
    lines = [f"# Rules held by recall (seed {seed}, the key world's stored memory)", ""]
    for a in (S1.PICK, S1.TOG, S1.DROP):
        kd = W.kinds[a]
        adm = kd.report["conditions"]["admitted"]
        lines += [f"## {VP.KNAME[a]}", "", f"Conditions read: {', '.join(adm)}.", "",
                  "| Front | Held | View conditions | Tries | Outcome (share) | Becomes |", "|---|---|---|---|---|---|"]
        rows = defaultdict(lambda: [None, None, 0, np.zeros(kd.lc.shape[1])])
        oof = kd.ix["oof"]
        inv = {v[1]: k for k, v in kd.ix["o"].items()}
        for i, key in enumerate(kd.keys):
            r = rows[oof[i]]
            if r[0] is None:
                r[0], r[1] = key, inv[oof[i]][2]
            r[2] += 1
            r[3] = r[3] + kd.lc[i]
        vnames = [kd.cname(ci) for ci in kd.adm if kd.cand[ci][0] == "view"]
        for o, (key, vb, n, cnt) in sorted(rows.items(), key=lambda x: (nm(x[1][0][0]), nm(x[1][0][1]))):
            j = int(cnt.argmax())
            cat = kd.lcat[j]
            aft = kd.lafter[j]
            becomes = ", ".join(f"{('front', 'hand')[p]}: {nm(h)}" for p, h in enumerate(aft) if h is not None) or "-"
            view = ", ".join(f"{v}={'yes' if b else 'no'}" for v, b in zip(vnames, vb)) or "-"
            lines.append(f"| {nm(key[0])} | {nm(key[1])} | {view} | {int(cnt.sum())} | {OUT[cat]} "
                         f"({cnt[j] / cnt.sum():.2f}) | {becomes} |")
        lines.append("")
    Path(get("--out", f"runs/051/rules_{seed}.md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
