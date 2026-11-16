import sys, json, numpy as np
sys.path.insert(0, 'tools/card134.1')
import factored as F
from factored import E, B, C
mem, test = C.load(C.MEM), C.load(C.TEST)
tgm, chm, _ = C.targets(mem); tgt, cht, _ = C.targets(test)
lk, _ = B.lock_of(mem, tgm, chm)
Xm, Ym, mm = E.events(mem, tgm, chm, lk); Xt, Yt, mt = E.events(test, tgt, cht, lk); Ym = Ym.numpy()
r, tg = F.records(Xm['effects'], False); rq, tq = F.records(Xt['effects'], False)
out = {}
for n, draws in ((400, 1), (100, 3)):
    rows = []
    for s in range(draws):
        keep = np.where(~mm['mixed'] & np.isin(mm['ep'], np.random.default_rng(s).choice(400, n, replace=False)))[0]
        sub = {k: v[keep] for k, v in r.items()}
        sel = F.select(sub, tg[keep], Ym[keep], mm['ep'][keep])
        rows.append(E.score(*F.answer(sel, sub, tg[keep], Ym[keep], rq, tq, mt['slide']), mt, None))
        if s == 0: print(n, {f'{a}->{b}': v for (a, b), v in sel.items() if (a, b) in ((1, 1), (2, 2), (1, 0))})
    out[n] = {k: round(float(np.mean([x[k] for x in rows])), 3) for k in rows[0]}
    print(n, json.dumps(out[n]))
json.dump(out, open('runs/134.1/noflag.json', 'w'), indent=1)
