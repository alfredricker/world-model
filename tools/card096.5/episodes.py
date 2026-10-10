"""Card 096.5: each episode stored as the sequence of condition changes that mattered (WM_EPISODES=1), and how soon
the goal is read from stored episodes (offline, --analyse).

In the agent: every real interaction try (pick up, toggle, drop) is recorded with its step, the tokens in front and
in the hand before and after, and arm C's surprise (minus the log probability arm C gave the outcome category before
the try). The episode's record goes into its per-episode result ("events").

Offline: a condition change is a recorded try that changed the token in front or the hand. For each held-out episode
and each of its condition changes, the steps still to the goal are predicted from the stored episodes (every other
episode of the same tier): the median, over stored episodes that made the same change (the same action, tokens before
and after), of their steps from that change to their goal. Baselines: the median over stored episodes of the steps
remaining after their k-th change (the same count of changes done), and after the same elapsed step.

  WM_EPISODES=1 ... tools/card069/run.py tiers ...                       (recording)
  bin/prun python tools/card096.5/episodes.py --analyse a.json b.json   → runs/096.5/howsoon.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATS = {"ep_tries": 0, "ep_changes": 0}
STATE = {"rec": None}


def hook(T):
    setup0 = T.setup

    def setup(tier, log):
        W, info = setup0(tier, log)
        VP = T.VP
        World = type(W)
        learn0 = World.learn_try
        episode0 = T.episode

        def learn_try(self, V, a, V2, end):
            rec = STATE["rec"]
            if rec is not None:
                rec["t"] += 1
            if rec is not None and a in VP.INTER:
                f0, h0, f1, h1 = int(V[VP.FRONT]), int(V[VP.HELD]), int(V2[VP.FRONT]), int(V2[VP.HELD])
                cat = int(f1 != f0) + 2 * int(h1 != h0)
                p = self.kinds[a].predict([(f0, h0, self.ctx_of_view(V))])[0]
                nm = self.judge.name
                rec["events"].append({"t": rec["t"], "a": VP.KNAME.get(a, str(a)), "front": nm(f0), "hand": nm(h0),
                                      "front_after": nm(f1), "hand_after": nm(h1), "cat": cat,
                                      "surprise": round(float(-np.log(max(p[cat], 1e-9))), 4)})
                STATS["ep_tries"] += 1
                STATS["ep_changes"] += int(cat > 0)
            return learn0(self, V, a, V2, end)

        STATE["episode0"] = episode0
        World.learn_try = learn_try
        T.episode = episode
        log("card 096.5: episodes recorded as condition changes")
        return W, info

    T.setup = setup


def episode(job):
    """The harness's episode with its record attached (module level: the worker pool pickles it by name)."""
    STATE["rec"] = {"t": 0, "events": []}
    try:
        r = STATE["episode0"](job)
    finally:
        rec, STATE["rec"] = STATE["rec"], None
    r["events"] = rec["events"]
    return r


# ---------------------------------------------------------------- offline: how soon, from stored episodes

def changes(r):
    return [e for e in r["events"] if e["cat"] > 0]


def key(e):
    return (e["a"], e["front"], e["hand"], e["front_after"], e["hand_after"])


def analyse(paths):
    eps = []
    for p in paths:
        eps += [r for r in json.loads(Path(p).read_text())["per_episode"] if r["success"]]
    out = {"episodes": len(eps)}
    err = defaultdict(list)
    cover = []
    for i, r in enumerate(eps):
        other = [x for j, x in enumerate(eps) if j != i]
        by_key, by_k, by_t = defaultdict(list), defaultdict(list), []
        for x in other:
            for k, e in enumerate(changes(x)):
                by_key[key(e)].append(x["steps"] - e["t"])
                by_k[k].append(x["steps"] - e["t"])
                by_t.append((e["t"], x["steps"] - e["t"]))
        bt = np.array(by_t, np.float64)
        for k, e in enumerate(changes(r)):
            truth = r["steps"] - e["t"]
            got = by_key.get(key(e))
            cover.append(got is not None)
            if got is None:
                continue
            near_t = bt[np.argsort(np.abs(bt[:, 0] - e["t"]))[:max(5, len(got))], 1]
            err["same change"].append(abs(np.median(got) - truth))
            err["same count of changes done"].append(abs(np.median(by_k.get(k, [np.nan])) - truth))
            err["same elapsed step"].append(abs(np.median(near_t) - truth))
            err["truth"].append(truth)
    out["change points"] = len(cover)
    out["covered by a stored same change"] = round(float(np.mean(cover)), 3)
    out["median remaining steps (truth)"] = float(np.median(err.pop("truth")))
    out["median absolute error"] = {k: round(float(np.nanmedian(v)), 1) for k, v in err.items()}
    out["mean absolute error"] = {k: round(float(np.nanmean(v)), 1) for k, v in err.items()}
    a, b = np.array(err["same change"]), np.array(err["same elapsed step"])
    out["same change better / worse than same elapsed step"] = [int((a < b).sum()), int((a > b).sum())]
    s = [e["surprise"] for r in eps for e in r["events"]]
    sc = [e["surprise"] for r in eps for e in changes(r)]
    out["surprise"] = {"tries": len(s), "changes": len(sc),
                       "share of changes with surprise > 0.1": round(float(np.mean(np.array(sc) > 0.1)), 3),
                       "share of tries with surprise > 0.1 that are changes": round(
                           float(np.mean([e["cat"] > 0 for r in eps for e in r["events"] if e["surprise"] > 0.1])), 3)}
    hi = defaultdict(int)
    for r in eps:
        for e in r["events"]:
            if e["surprise"] > 0.1:
                hi[f"{e['a']} {e['front']} | {e['hand']} -> {e['front_after']} | {e['hand_after']}"] += 1
    out["surprising tries (most common)"] = sorted(hi.items(), key=lambda kv: -kv[1])[:12]
    return out


if __name__ == "__main__" and "--analyse" in sys.argv:
    paths = [a for a in sys.argv[1:] if a != "--analyse"]
    rep = analyse(paths)
    (ROOT / "runs" / "096.5").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "096.5" / "howsoon.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
