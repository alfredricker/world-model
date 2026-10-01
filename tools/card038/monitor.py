"""Card 038: progress of the main runs, from each run's progress file, saved results and log.

  python3 tools/card038/monitor.py                       arms B and A (runs/038_armB.json, runs/038_armA.json)
  python3 tools/card038/monitor.py runs/038_armB.json    one run
  python3 tools/card038/monitor.py --watch 60            refresh every 60 seconds

The run rewrites <out>.progress.json after every batch of layouts and <out> after every seed. The card's
pass rule is each criterion in at least 8 of 10 seeds, for yellow and for purple.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ["runs/038_armB.json", "runs/038_armA.json"]
WORLDS = ("key", "switch", "either", "both")
COLOURS = ("yellow", "purple")


def load(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def mins(sec):
    return f"{sec / 60:.0f} min" if sec >= 90 else f"{sec:.0f} s"


def alive(pid):
    return pid is not None and Path(f"/proc/{pid}").exists()


def last_log_line(p):
    try:
        lines = [x for x in p.read_text(errors="replace").splitlines() if x.strip() and "Warning" not in x]
    except OSError:
        return None, False
    err = any(x.startswith("Traceback") for x in lines)
    return (lines[-1] if lines else None), err


def seed_row(s):
    """One seed's line: criteria, then the numbers behind them."""
    w, v = s["worlds"], s["verdicts"]
    yn = lambda b: "pass" if b else "FAIL"
    eff = min(w[x]["heldout_effects"]["lowest"] for x in WORLDS if x in w)
    fam = " ".join(f"{w[x]['acting']['success']:.2f}" for x in WORLDS if x in w)
    ratio = max(w[x]["acting"]["steps_ratio_to_card029"] for x in WORLDS if x in w)
    ta = [w["key"].get(f"test_a_{c}") for c in COLOURS]
    tb = [w["switch"].get(f"test_b_{c}") for c in COLOURS]
    cases = "/".join(f"{t['criterion_2']['lowest_case']:.2f}" if t and t["criterion_2"]["lowest_case"] is not None
                     else "-" for t in ta)
    goal_a = "/".join(f"{t['acting']['success']:.2f}" if t else "-" for t in ta)
    steps_a = "/".join(f"{t['acting']['steps_ratio_to_shortest']:.2f}" if t else "-" for t in ta)
    goal_b = "/".join(f"{t['acting']['success']:.2f}" if t else "-" for t in tb)
    return (f"  {s['seed']}  {yn(v['criterion_1']):4} {yn(v['criterion_2']):4} {yn(v['criterion_3']):4}  "
            f"{eff:.4f}  {fam}  {ratio:.2f}   {cases:9}  {goal_a:9}  {steps_a:9}  {goal_b}")


HEADER = ("  seed  c1   c2   c3    effects  goal: key switch either both  steps  new cases  (a) goal   (a) steps  (b) goal\n"
          "                          lowest                                 x029   yel/pur    yel/pur    yel/pur    yel/pur")


def shown(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def report(out):
    out = ROOT / out if not Path(out).is_absolute() else Path(out)
    prog, res = load(out.with_suffix(".progress.json")), load(out)
    line, err = last_log_line(out.with_suffix(".log"))
    lines = []
    if prog is None and res is None:
        return [f"{shown(out)}: not started"]
    arm = (prog or res).get("arm")
    now = time.time()
    if prog is None:
        state = "finished" if res and "seeds_passing" in res else "no progress file (started before monitoring?)"
    elif prog.get("finished"):
        state = f"finished after {mins(prog['updated'] - prog['started'])}"
    elif alive(prog.get("pid")):
        state = f"running, pid {prog['pid']}, {mins(now - prog['started'])} elapsed"
    else:
        state = f"STOPPED: process {prog.get('pid')} is gone; last update {mins(now - prog['updated'])} ago"
    if err:
        state += "; the log has a Traceback"
    lines.append(f"arm {arm} ({shown(out)}): {state}")
    if prog and not prog.get("finished"):
        seeds = prog.get("seeds", [])
        k = seeds.index(prog["seed"]) + 1 if prog.get("seed") in seeds else "?"
        where = f"seed {prog.get('seed')} ({k} of {len(seeds)}), world {prog.get('world')}, {prog.get('stage')}"
        if prog.get("layouts_total"):
            where += f": {prog['layouts_done']} of {prog['layouts_total']} layouts"
            if prog.get("goal_so_far") is not None and prog["layouts_done"]:
                where += f", goal reached in {prog['goal_so_far']}"
        lines.append(f"  now: {where} (updated {mins(now - prog['updated'])} ago)")
        done, secs = prog.get("seeds_done", []), prog.get("seed_seconds", [])
        if secs:
            mean = sum(secs) / len(secs)
            left = max(0.0, mean * (len(seeds) - len(done)) - (now - prog.get("seed_started", now)))
            lines.append(f"  seeds done {len(done)} of {len(seeds)}, {mins(min(secs))}–{mins(max(secs))} each; "
                         f"about {mins(left)} left for this arm")
        else:
            lines.append(f"  seeds done 0 of {len(seeds)}; current seed running for "
                         f"{mins(now - prog.get('seed_started', now))}")
    if res and res.get("seeds"):
        lines.append(HEADER)
        lines += [seed_row(s) for s in res["seeds"]]
        n = len(res["seeds"])
        tot = {c: sum(s["verdicts"][c] for s in res["seeds"]) for c in ("criterion_1", "criterion_2", "criterion_3")}
        lines.append(f"  passing so far: criterion 1 in {tot['criterion_1']} of {n}, criterion 2 in "
                     f"{tot['criterion_2']} of {n}, criterion 3 in {tot['criterion_3']} of {n} (the card needs 8 of 10)")
    if line and not (prog and prog.get("finished")):
        lines.append(f"  log: {line.strip()[:160]}")
    return lines


def main():
    args = sys.argv[1:]
    watch = None
    if "--watch" in args:
        i = args.index("--watch")
        watch = float(args[i + 1])
        del args[i:i + 2]
    outs = args or DEFAULT
    while True:
        text = [f"Card 038 main runs, {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        for o in outs:
            text += report(o)
        if watch:
            print("\033[2J\033[H", end="")
        print("\n".join(text), flush=True)
        if not watch:
            return
        time.sleep(watch)


if __name__ == "__main__":
    main()
