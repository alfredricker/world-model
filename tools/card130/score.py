"""Card 130: how often a touch worked, by the agent's own measure, from a plan log.

A move worked if the next reading no longer shows its variable differing from the goal (differs <= 1); a move that
ends the episode in success worked; a move cut off by the step limit is not counted. Slides are the appearance
variables (window: places 0 and 8, drawer: places 1 and 5); the cube is group 3. "Opening the drawer" is a move at
a drawer place when the goal has the drawer open (truth is used only to split the table, not by the agent).

  python tools/card130/score.py runs/130/plan_task3.json runs/129.2/plan_places_task3.json
"""
import json
import sys

KIND = {0: 'window', 8: 'window', 1: 'drawer', 5: 'drawer', 3: 'cube'}
for f in sys.argv[1:]:
    r = json.load(open(f))
    n = {}
    for ep in r['log']:
        ts = ep['touches']
        for a, b in zip(ts, ts[1:] + [None]):
            if not a['act'] or a['act'][0] != 'move':
                continue
            g = a['act'][1]
            if b is None:
                if not ep['success']:
                    continue
                ok = True
            else:
                ok = b['differs'][str(g)] <= 1
            keys = [KIND.get(g, str(g))]
            if keys[0] != 'cube':
                keys.append('slides')
            if keys[0] == 'drawer' and ep['goal']['drawer'] < -0.08:
                keys.append('open drawer')
            for k in keys:
                s = n.setdefault(k, [0, 0])
                s[0] += ok; s[1] += 1
    print(f, 'success', r['success'], {k: f'{s}/{t} = {s / t:.0%}' for k, (s, t) in sorted(n.items())})
