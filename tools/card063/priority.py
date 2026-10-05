"""Card 063: the encoder's transition model replays the tries it gets wrong (prioritized replay, Schaul et al. 2016).

Card 054's encoder learns from its transition model: per update and per action, 128 tries from the last 20,000, half
with a change and half without, drawn uniformly. A rare consequence (toggling a locked door with a key of another
colour does nothing) is a sliver of the no-change half, and on three of four seeds the transition model got it right
only 70-85%. Here, within each half, half of the tries are drawn uniformly and half in proportion to (the try's last
transition error + EPS) ** ALPHA; a new try enters at the largest priority seen. The error is the transition model's
own (no label): after each update, every drawn try's priority becomes its squared error in that update. Nothing else
changes (card 054's recipe at m = 0.5).

  bin/prun python tools/card063/priority.py tools/card052/predflip.py --seed 399 ... --pair-m 0.5 --out runs/063/b_399.json
"""
import runpy
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "card052"))
import effect as EF                                    # noqa: E402

ALPHA, EPS, SHARE = 0.6, 1e-3, 0.5
LAST = {"draw": None}
STATS = {"updates": 0, "written": 0}


class PDeque(deque):
    """A deque with a parallel deque of priorities; a new item enters at the largest priority seen."""

    def __init__(self, iterable=(), maxlen=None):
        super().__init__(iterable, maxlen)
        self.pr = deque([1.0] * len(self), maxlen=maxlen)
        self.top = 1.0

    def append(self, x):
        super().append(x)
        self.pr.append(self.top)

    def extend(self, xs):
        for x in xs:
            self.append(x)


def draw(buf, rng):
    """Per action: PER tries, half with a change and half without (as card 052); within each half, SHARE drawn by
    priority, the rest uniformly."""
    out, picks = [], []
    for a in EF.ACTS:
        ch, no = buf[a][1], buf[a][0]
        nc = min(EF.PER // 2, len(ch))
        sel = []
        for k, d, n in ((1, ch, nc), (0, no, EF.PER - nc)):
            if n == 0:
                continue
            nu = n - int(n * SHARE)
            iu = rng.choice(len(d), size=nu, replace=False)
            p = np.asarray(d.pr, np.float64) ** ALPHA
            p[iu] = 0.0
            p /= p.sum()
            ip = rng.choice(len(d), size=n - nu, replace=False, p=p)
            sel += [(k, int(i)) for i in np.concatenate([iu, ip])]
        out.append([buf[a][k][i] for k, i in sel])
        picks.append(sel)
    LAST["draw"] = (buf, picks)
    return out


def transition(self, tries):
    """effect.py's transition term, also keeping each try's error."""
    total, parts, per = 0.0, [], []
    for ai, tr in enumerate(tries):
        zf, zh = self.pieces(self.x([t[0] for t in tr])), self.pieces(self.x([t[1] for t in tr]))
        pf, ph = self.predict_after(ai, zf, zh, sample=True)
        with torch.no_grad():
            tf, th = self.tpieces(self.x([t[3] for t in tr])), self.tpieces(self.x([t[4] for t in tr]))
        lv = ((pf - tf) ** 2).sum((-1, -2)) + ((ph - th) ** 2).sum((-1, -2))
        l = lv.mean()
        per.append(lv.detach().cpu().numpy())
        total = total + l
        parts.append(float(l.detach()))
        if EF.VIS_W > 0:
            total = total + EF.VIS_W * self.visibility(tr, zf, zh)
    if EF.GATES and self.sp_w is not None:
        w = torch.as_tensor(self.sp_w, dtype=torch.float32, device=self.dev).reshape(-1, 1)
        total = total + (w * self.gate_values()).sum()
    self.last_trans = parts
    self.last_per_try = per
    return total


_update = EF.Encoder.update


def update(self, *a, **k):
    r = _update(self, *a, **k)
    d, per = LAST["draw"], getattr(self, "last_per_try", None)
    if d is not None and per is not None:
        buf, picks = d
        for ai, act in enumerate(EF.ACTS):
            for (kk, i), v in zip(picks[ai], per[ai]):
                q = buf[act][kk]
                q.pr[i] = float(v) + EPS
                q.top = max(q.top, q.pr[i])
                STATS["written"] += 1
        STATS["updates"] += 1
    LAST["draw"], self.last_per_try = None, None
    return r


def main():
    EF.deque = PDeque
    EF.draw = draw
    EF.Encoder.transition = transition
    EF.Encoder.update = update
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")
    print({"card063_replay": STATS}, flush=True)


if __name__ == "__main__":
    main()
