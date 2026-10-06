"""Card 077: what each tile does, by the agent's own recall in its world (version 18's memory), empty-handed, from a
start situation of that world: forward moves onto it; forward onto it ends the episode; a pick up changes it; a
toggle changes it. For every handle card 076's weighings of that world use.

  WM_REL_ENCODER=runs/070/encoder.pt WM_TRYING=1 WM_CONFLICTS=1 WM_TIES=1 WM_SPLIT=1 WM_CACHE=1 \
    bin/prun python tools/card077/affordances.py tier2
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
FILES = {"tier1": ["train_tier1", "tier1"], "tier2": ["train_tier2", "train_tier2b", "tier2"],
         "decoy": ["train_decoy"] + [f"{k}_{h}" for k in ("known", "decoy")
                                     for h in ("red", "green", "blue", "purple", "yellow", "grey")]}
tier = 2 if MODE == "tier2" else 1
sys.argv = ["run.py", "decoy" if MODE == "decoy" else "tiers", "--tier", str(tier), "--online", "0", "--memory", "066"]
sys.path.insert(0, str(ROOT / "tools" / "card069"))
import run as RUN                                      # noqa: E402

T, R = RUN.T, RUN.R
VP, TK = T.VP, T.TK
if MODE == "decoy":                                    # as card 073's known.py: every hue in memory
    T.ENVS[1] = R.register()
    T.OUT = ROOT / "runs" / "070" / "decoy_memory066"
    R.DECOY.update(door=list(T.HUES), decoy=list(T.HUES))
W, _ = T.setup(tier, print)
pl = T.S7.Plan047(W)
env = T.make(tier)
env.reset(seed=T.SEED_TEST + 1000 * tier)
codes = T.now_codes(env)
facts, st, _, _ = pl.observe(None, T.PV.crop(T.Kd.APP[codes], codes))
empty, ctx = int(pl.facts[st[0]][VP.HELD]), pl.ctx_id(st[0], st[1])
handles = set()
for f in FILES[MODE]:
    p = ROOT / "runs" / "076" / f"{f}.json"
    if p.exists():
        for ep in json.load(open(p))["per_episode"]:
            handles |= {int(h) for h in ep.get("handles", {})}
out = {}
for h in sorted(handles):
    out[h] = [int(TK.free_of(h)), int(W.move_cat(VP.FWD, h) == VP.ENDED),
              int(W.outcome(VP.PICK, h, empty, ctx)[0] != 0), int(W.outcome(VP.TOG, h, empty, ctx)[0] != 0)]
names = {h: VP.WORLD.judge.name(h) for h in out}         # the evaluator's names, for the report only
res = {"note": "card 077: [walk onto, ends, pick up changes it, toggle changes it], empty-handed", "mode": MODE,
       "empty_hand": empty, "affordances": out, "names": names}
(ROOT / "runs" / "077").mkdir(parents=True, exist_ok=True)
(ROOT / "runs" / "077" / f"affordances_{MODE}.json").write_text(json.dumps(res, indent=1) + "\n")
for h in out:
    print(h, names[h], out[h])
