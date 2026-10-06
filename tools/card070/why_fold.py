"""Card 070, report only: why recall predicts that the fold hue's key does not open its door. Builds the decoy
world's World for one fold (every hue in memory, less the hue's opening tries) and prints, for the query (locked door
of the hue, key of the hue), the admitted conditions' weights and the stored groups that weigh most.

  WM_REL_ENCODER=runs/070/encoder.pt bin/prun python tools/card070/why_fold.py decoy --fold red --hold opening --memory 066
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card069"))
import run as RUN                                       # noqa: E402  (installs card 069's relation into the harness)

T, R = RUN.T, RUN.R
import numpy as np                                     # noqa: E402

RUN.main_setup_only = True
hue = RUN.arg("--fold")
others = [c for c in T.HUES if c != hue]
T.ENVS[1] = R.register()
T.OUT = RUN.ROOT / "runs" / "070" / f"decoy_memory{RUN.arg('--memory', '066')}"
R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
_memory = T.memory


def memory(tier, pool=None):
    D, stats = _memory(tier, pool)
    f = D["ego0"][:, T.FRONT].astype(np.int64) // 5 * 5
    h = D["ego0"][:, T.HELD].astype(np.int64) // 5 * 5
    drop = (D["act"] == T.VP.TOG) & (f == T.KR.code(("door", hue, 0))) & (h == T.KR.code(("key", hue)))
    keep_t = ~drop[np.isin(D["act"], T.INTER)]
    return {k: (v[keep_t] if k in ("pres", "stale") else v[~drop]) for k, v in D.items()}, stats


T.memory = memory
W, info = T.setup(1, lambda m: None)
VP, KR, Kd = T.VP, T.KR, T.Kd
hd = lambda o: int(Kd.APP[KR.code(o)])
kd = W.kinds[VP.TOG]
print("toggle admitted", [kd.cname(c) for c in kd.adm])
print("weights per unit of distance (lambda / scale):", {kd.cname(c): round(float(l), 3) for c, l in zip(kd.adm, kd.lamc)})
env = RUN._make(1)
env.reset(seed=T.SEED_TEST + 999)
pl = T.S7.Plan047(W)
codes = T.now_codes(env)
facts, st, _, _ = pl.observe(None, T.PV.crop(Kd.APP[codes], codes))
ctx = pl.ctx_id(st[0], st[1])
for key_hue in (hue, others[0]):
    q = (hd(("door", hue, 0)), hd(("key", key_hue)), int(ctx))
    Fq, Fg = kd.feats([q]), kd.group_feats()
    parts = {kd.cname(c): l * kd.cand_dist(c, Fq, Fg)[0] for c, l in zip(kd.adm, kd.lamc)}
    d = sum(parts.values())
    w = np.exp(-d)
    Gc = kd.ix["Gc"]
    top = np.argsort(-w)[:8]
    print(f"\nquery: locked {hue} door, {key_hue} key held; predicted category {kd.cat_of([q])[0]}; "
          f"rel:P of the query {float(Fq[2][0, 4]):.2f}")
    print("  weight, front tile, held tile, rel:P, stored outcomes (counts by class), distance by condition")
    for g in top:
        i = kd.ix["rep"][g]
        k = kd.keys[i]
        print(f"  {w[g]:.3g}  {W.judge.name(int(k[0]))[:18]:18s} {W.judge.name(int(k[1]))[:14]:14s} {float(Fg[2][g, 4]):5.2f}  "
              f"{np.round(Gc[g], 1).tolist()}  " + ", ".join(f"{n} {float(v[g]):.2f}" for n, v in parts.items()))
    opened = Gc[:, 1:].sum(1) > 0
    print(f"  total weight on groups with an opening: {w[opened].sum():.3g}; on groups without: {w[~opened].sum():.3g}")
