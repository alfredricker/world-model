"""Version 27 on CHARTER's tier 3, MiniGrid-ObstructedMaze-Full-v1, as a video (a demo, not an experiment).

Nothing is rerun: card 101's tier 3 run (runs/101/c_tier3.json, version 27 as runs/101/run.sh runs it with NOPLAN=1)
kept every episode's actions, and this script replays chosen episodes in MiniGrid from their seeds and actions. The
replay is checked against the stored outcome. The captions name what each action did in the simulator (the
evaluator's view); the agent's own plan was not logged in that run.

  bin/prun python tools/demos/obstructed_maze.py --out demos/v27_obstructed_maze.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
arg = lambda k, d=None: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
RUN = ROOT / "runs" / "101" / "c_tier3.json"
FRESH = ROOT / "runs" / "101" / "fresh_c_tier3.json"
ACT = {0: "turn left", 1: "turn right", 2: "forward", 3: "pick up", 4: "drop", 5: "toggle"}
WID, HEI, FPS, TS = 1280, 720, 8, 40
BG = (28, 28, 34)
X0, Y0 = 20, 40


def ts(sec):
    return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"


def name(o):
    if o is None:
        return "nothing"
    if o.type == "door":
        return f"{o.color} door ({'locked' if o.is_locked else 'open' if o.is_open else 'closed'})"
    return f"{o.color} {o.type}"


def event(a, front0, held0, front1, held1):
    """What the action did, read from the simulator before and after."""
    if a == 3 and held1 is not None and held0 is None:
        return f"picked up the {name(held1)}"
    if a == 4 and held0 is not None and held1 is None:
        return f"put down the {name(held0)}"
    if a == 5 and front0 is not None:
        if front0[0] == "box" and front1 != front0:
            return "opened the box: " + (f"a {front1[1]} {front1[0]} was inside" if front1 else "it was empty")
        if front0[0] == "door" and front1 != front0:
            return f"{'unlocked and opened' if front0[2] == 'locked' else 'opened'} the {front0[1]} door"
    if a in (3, 4, 5):
        return "no effect"
    return None


def snap(o):
    if o is None:
        return None
    return (o.type, o.color, ("locked" if o.is_locked else "open" if o.is_open else "closed") if o.type == "door" else "")


def main():
    import gymnasium as gym
    import minigrid  # noqa: F401
    res = json.loads(RUN.read_text())
    recs = {r["seed"]: r for r in res["per_episode"]}
    fresh = json.loads(FRESH.read_text())
    order = sorted(recs.values(), key=lambda r: r["steps"])
    pick = [order[len(order) // 6], order[len(order) // 2], order[-6]]      # short, median and long episodes
    env = gym.make(res["env"]).unwrapped
    out = ROOT / arg("--out", "demos/v27_obstructed_maze.mp4")
    ass = [f"[Script Info]\nScriptType: v4.00+\nPlayResX: {WID}\nPlayResY: {HEI}\nWrapStyle: 0\n\n[V4+ Styles]\n"
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding\n"
           "Style: D,DejaVu Sans,21,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
    frames, t = [], [0.0]

    def canvas(img=None):
        c = np.empty((HEI, WID, 3), np.uint8)
        c[:] = BG
        if img is not None:
            c[Y0:Y0 + img.shape[0], X0:X0 + img.shape[1]] = img
        return c

    def show(img, sec, text, ml=None, mv=Y0):
        n = max(1, round(sec * FPS))
        frames.append((img, n))
        ml = ml if ml is not None else X0 + env.width * TS + 30
        ass.append(f"Dialogue: 0,{ts(t[0])},{ts(t[0] + n / FPS)},D,,{ml},20,{mv},,{text}\n")
        t[0] += n / FPS

    show(canvas(), 9, "{\\fs40}Version 27 on ObstructedMaze-Full{\\fs23}\\N\\N"
         "A learned world model (architecture version 27) acting in MiniGrid's hardest\\Ntier of this project. "
         "Nine rooms; locked doors, each with a key of its colour hidden\\Nin a box somewhere; balls blocking the "
         "doors. The task: pick up the blue ball.\\N\\N"
         "It sees only the 7 x 7 window ahead (the lit area); walls and closed doors hide\\Nwhat lies behind them. "
         "Its memory is random play; it plans backward from the goal\\Nover conditions learned from what its "
         "actions did, and when it has no plan it acts\\Nwhere its model knows the outcome least (opening an "
         "unknown box, say).\\N\\N"
         f"Card 101's evaluation: {res['successes']} of {res['episodes']} layouts solved, "
         f"{res['mean_steps_when_successful']} steps on average;\\N"
         f"{fresh['successes']} of {fresh['episodes']} on fresh layouts. Memory is reset in every layout.\\N\\N"
         "Below: three of the 30 layouts (short, median and long), replayed from the run's\\Nrecorded actions.",
         ml=80, mv=60)
    for k, r in enumerate(pick):
        env.reset(seed=r["seed"])
        title = (f"{{\\fs28}}Layout {k + 1} of {len(pick)}{{\\fs21}}\\NSeed {r['seed']}; "
                 f"mission: {env.mission}")
        show(canvas(env.get_frame(highlight=True, tile_size=TS)), 3,
             title + "\\N\\NThe agent (red triangle) starts knowing nothing\\nabout this layout.".replace("\\n", "\\N"))
        log = []
        te = rew = False
        for i, a in enumerate(r["actions"]):
            img = env.get_frame(highlight=True, tile_size=TS)
            f0, h0 = snap(env.grid.get(*env.front_pos)), env.carrying
            _, rew, te, tr, _ = env.step(a)
            ev = event(a, f0, h0, snap(env.grid.get(*env.front_pos)), env.carrying)
            if ev:
                log = (log + [f"step {i + 1}: {ev}"])[-9:]
            evs = "\\N".join(("{\\c&H00D7FF&}" + x + "{\\c&HFFFFFF&}") if j == len(log) - 1 and ev else x
                             for j, x in enumerate(log))
            show(canvas(img), 0.6 if ev else 0.125,
                 f"{title}\\N\\N{{\\fs25}}Step {i + 1}: {ACT[a]}{{\\fs21}}\\N"
                 f"holding {name(h0)}\\N\\NWhat its pick ups, drops and toggles did:\\N{evs}")
            if te or tr:
                break
        assert te and rew > 0 and i + 1 == r["steps"], ("replay differs from the run", r["seed"])
        show(canvas(env.get_frame(highlight=True, tile_size=TS)), 3.5,
             f"{title}\\N\\N{{\\fs30}}{{\\c&H64DC50&}}Blue ball picked up{{\\c&HFFFFFF&}} in {r['steps']} steps"
             f"{{\\fs21}}\\NMiniGrid return {rew:.3f}")

    sub = out.with_suffix(".ass")
    sub.write_text("".join(ass))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WID}x{HEI}",
           "-r", str(FPS), "-i", "-", "-vf", f"subtitles={sub}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-r", "25", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for img, n in frames:
        b = img.tobytes()
        for _ in range(n):
            p.stdin.write(b)
    p.stdin.close()
    p.wait()
    sub.unlink()
    print("wrote", out, f"{t[0]:.1f}s", [(r["seed"], r["steps"]) for r in pick], flush=True)


if __name__ == "__main__":
    main()
