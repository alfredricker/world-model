"""Card 127: where a run of plan.py went wrong, from its log and the simulator's truth (scoring only).
Per slide place: how often the variable is said to differ when the slide is and is not at the goal; whether the
lock is read right (now, goal frame); of the pulls made when the slide was unlocked and away from the goal, how
many had a remembered start like the live one and a remembered end like the goal, and how many worked.

  bin/prun python tools/card127/diag.py runs/127/plan_dev30.json
"""
import json
import sys
from pathlib import Path

import numpy as np

EP = 1001
ROOT = Path(__file__).resolve().parents[2]
q = np.load(Path.home() / '.ogbench/data/scene-play-v0.npz')['qpos'][400 * EP:500 * EP]
k = np.load(ROOT / 'runs/127/know.npz')
st, en = k['st'], k['en']
r = json.load(open(sys.argv[1]))
slide = {'0': ('window', 24, 1), '8': ('window', 24, 1), '1': ('drawer', 23, 0), '5': ('drawer', 23, 0)}
for c, (s, qi, li) in slide.items():
    same, diff, now, goal, rows = [], [], [], [], []
    for ep in r['log']:
        g = ep['goal']
        for j, t in enumerate(ep['touches']):
            tr = t['truth']
            away = abs(tr[s] - g[s]) > 0.04
            (diff if away else same).append(t['differs'][c] > 1)
            a, b, e = t['state'][c]
            now.append((a == e) == (tr['locks'][li] == 1)); goal.append((b == e) == (g['locks'][li] == 1))
            if t['act'] and t['act'][0] == 'pull' and str(t['act'][1]) == c and away and tr['locks'][li] == 1:
                i = t['act'][2]
                nxt = ep['touches'][j + 1]['truth'] if j + 1 < len(ep['touches']) else ep['end']
                rows.append((abs(q[st[i] - 2, qi] - tr[s]) < 0.03, abs(q[st[i + 1] - 2, qi] - g[s]) < 0.04, abs(nxt[s] - g[s]) <= 0.04))
    v = np.array(rows).reshape(-1, 3)
    f = lambda x: round(float(np.mean(x)), 2) if len(x) else None
    print(f'place {c} ({s}): says differs, slide at goal {f(same)} (n {len(same)}), away {f(diff)} (n {len(diff)}); '
          f'lock read right now {f(now)}, goal {f(goal)}; needed pulls {len(v)}: start matched {f(v[:, 0])}, '
          f'end matched {f(v[:, 1])}, worked {f(v[:, 2])}')
print('success', r['success'], 'gave up', r['gave_up'])
