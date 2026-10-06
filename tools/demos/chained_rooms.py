"""Version 17 in chained rooms, as a video (a demo, not an experiment).

The world (evaluator side): card 048's chained rooms rebuilt in MiniGrid. A 13 x 8 room split by walls at x = 4 and
x = 8 into three rooms of 3 x 6, each wall with a locked door of its own colour (two of card 048's red, green and blue,
drawn per layout). The agent starts in the middle room with a grey ball and a purple box (card 048's switch and vase; they open
nothing); the goal square is in an
outer room, left or right at random.
  two doors: the goal room's key lies in the other outer room, behind the other door, whose key is in the middle
             room. Pick up key A, open door A, put key A down, fetch key B, open door B, reach the goal.
  one door:  both keys in the middle room.

The agent is version 17 as CHARTER's tiers run it (card 069's runner with WM_TRYING=1, WM_ORDER=1, card 070's
encoder, --online 0): memory from random play in this world (card 066's collection, about 3.2 million steps,
both variants), memory reset for every layout, the token lattice 14 (card 066's prior: it covers the map from
any start).

  WM_TRYING=1 WM_ORDER=1 WM_REL_ENCODER=runs/070/encoder.pt bin/prun python tools/demos/chained_rooms.py eval --n 30
  bin/prun python tools/demos/chained_rooms.py render --out demos/v17_chained_rooms.mp4

eval writes runs/demo_chained/eval.json (every episode's actions and the plan behind each); render replays the chosen
episodes in MiniGrid from their seeds and actions, so drawing again does not rerun the agent.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODE = sys.argv[1]
arg = lambda k, d=None: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)
OUT = ROOT / "runs" / "demo_chained"
EVAL = OUT / "eval.json"
WD, HT, WALLS = 13, 8, (4, 8)
MAX_STEPS = 640                                        # card 034's episode length, for play and for tests
SEED_TEST = 2_000_000
ACT = {0: "turn left", 1: "turn right", 2: "forward", 3: "pick up", 4: "drop", 5: "toggle"}

import numpy as np                                     # noqa: E402


def register(variant=None):
    """MiniGrid's version of card 048's chained rooms; variant None draws one of the two per layout."""
    import gymnasium as gym
    from minigrid.core.grid import Grid
    from minigrid.core.mission import MissionSpace
    from minigrid.core.world_object import Ball, Box, Door, Goal, Key
    from minigrid.minigrid_env import MiniGridEnv

    hues = ("red", "green", "blue", "purple", "yellow", "grey")

    class ChainedRooms(MiniGridEnv):
        def __init__(self, variant=None, **kw):
            self.variant = variant
            super().__init__(mission_space=MissionSpace(mission_func=lambda: "reach the goal"), width=WD, height=HT,
                             max_steps=MAX_STEPS, **kw)

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            for x in WALLS:
                self.grid.vert_wall(x, 0)
            two = self.variant == "two_doors" if self.variant else self._rand_bool()
            self.layout_variant = "two_doors" if two else "one_door"
            cols = self._rand_subset(hues[:3], 2)        # card 048's three colours
            side = self._rand_int(0, 2)                # 0: the goal is in the left room, 1: the right
            goal_wall, other_wall = (WALLS[0], WALLS[1]) if side == 0 else (WALLS[1], WALLS[0])
            self.put_obj(Door(cols[0], is_locked=True), goal_wall, self._rand_int(1, height - 1))
            self.put_obj(Door(cols[1], is_locked=True), other_wall, self._rand_int(1, height - 1))
            rooms = {"left": (1, 1), "middle": (WALLS[0] + 1, 1), "right": (WALLS[1] + 1, 1)}
            goal_room, other_room = ("left", "right") if side == 0 else ("right", "left")
            self.place_obj(Goal(), top=rooms[goal_room], size=(3, height - 2))
            self.place_obj(Key(cols[1]), top=rooms["middle"], size=(3, height - 2))
            self.place_obj(Key(cols[0]), top=rooms[other_room if two else "middle"], size=(3, height - 2))
            self.place_obj(Ball("grey"), top=rooms["middle"], size=(3, height - 2))
            self.place_obj(Box("purple"), top=rooms["middle"], size=(3, height - 2))
            self.place_agent(top=rooms["middle"], size=(3, height - 2))
            self.mission = "reach the goal"

    name = f"WM-ChainedRooms{'-' + variant if variant else ''}-v0"
    if name not in gym.registry:
        gym.register(id=name, entry_point=ChainedRooms, kwargs={"variant": variant})
    return name


# ---------------------------------------------------------------- eval: version 17 as card 069's runner installs it

def setup_agent():
    os.environ["WM_LATTICE"] = "14"
    sys.argv += ["--tier", "1"]
    for c in ("card066", "card067", "card069", "card072", "card073"):
        sys.path.insert(0, str(ROOT / "tools" / c))
    import tiers as T
    import propagate as WALK
    import relation as R
    T.INSTALL += [WALK.install, R.install]
    sys.modules["index"].HOLD["by"] = "try"
    R.ONLINE["on"] = False
    assert os.environ.get("WM_TRYING") == "1" and os.environ.get("WM_ORDER") == "1", "version 17 needs both"
    import trying as TRY
    import order as ORDER
    T.INSTALL += [TRY.install, ORDER.install]
    T.ENVS[1] = register()                             # memory: random play over both variants
    T.OUT = OUT
    return T


def describe(T, c):
    """A condition of the plan in plain words (MiniGrid's names for the tiles)."""
    J = T.VP.WORLD.judge
    nm = lambda h: J.name(int(h)) if h is not None else "?"
    verb = {2: "step onto", 3: "pick up", 4: "drop onto", 5: "toggle"}
    try:
        if c[0] == "end":
            return "reach the goal (the episode ends)"
        if c[0] == "has":
            return f"hold the {nm(c[1])}"
        if c[0] == "face":
            return f"face the {nm(c[2])}, to {verb.get(c[1], T.VP.KNAME.get(c[1], c[1]))} it"
        if c[0] == "walk":
            if isinstance(c[1], (int, np.integer)):
                return "clear the way: a tile on the route must become walkable"
            return f"walk ({c[1]})"
        if c[0] == "part":
            act = verb.get(c[2], T.VP.KNAME.get(c[2], c[2]))
            if c[1] == T.VP.HELDP:
                return f"hold what makes '{act} the {nm(c[3])}' work"
            return f"change what is in view so that '{act} the {nm(c[3])}' works"
        if c[0] == "act":
            return f"do it: {ACT.get(c[1], c[1])}"
        return T.name(c)
    except Exception:
        return str(c)[:60]


def episode(job):
    """Card 066's episode, keeping the actions and the plan behind each step."""
    T = sys.modules["tiers"]
    PV, VP, F, Kd = T.PV, T.VP, T.F, T.Kd
    tier, seed = job
    W = VP.WORLD
    W.reset()
    pl = (T.PLANNER["cls"] or T.S7.Plan047)(W)
    env = T.make(tier)
    env.reset(seed=int(seed))
    rng = np.random.default_rng(seed)
    codes = T.now_codes(env)
    V = PV.crop(Kd.APP[codes], codes)
    facts, st, _, _ = pl.observe(None, V)
    rec = {"seed": int(seed), "variant": env.layout_variant, "random": 0, "explore": 0, "steps_log": []}
    te = False
    rew = 0.0
    t0 = time.monotonic()
    rec["timed_out"] = False
    signal.signal(signal.SIGALRM, T._out_of_time)
    signal.setitimer(signal.ITIMER_REAL, T.CAP_SECONDS)
    t = 0
    try:
        for t in range(env.max_steps):
            r0, e0 = rec["random"], rec["explore"]
            res = pl.choose(st)
            a = res.action if res is not None else PV.fallback(pl, st, rng, rec)
            why = ([describe(T, c) for c in res.trace[:8]] if res is not None
                   else ["no chain of conditions: " + ("look at never-seen places" if rec["explore"] > e0
                                                       else "a random action" if rec["random"] > r0 else "fallback")])
            rec["steps_log"].append({"a": int(a), "why": why})
            Vb = pl.view(st)
            pred = pl.step(st, a)
            _, rew, te, tr, _ = env.step(a)
            codes = T.now_codes(env)
            V2 = PV.crop(Kd.APP[codes], codes)
            facts, st2, miss, _ = pl.observe(facts, V2, te, prefer=pred[1], cands=PV.moved_to(st, a))
            W.learn_try(Vb, a, pl.view(st2), te)
            pl.forget()
            st = st2
            if te or tr:
                break
    except T.OutOfTime:
        rec["timed_out"], te = True, False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    rec.update({"success": bool(te and rew > 0), "steps": t + 1, "seconds": round(time.monotonic() - t0, 2)})
    return rec


def shortest(env):
    """The evaluator's fewest steps to the goal, by breadth-first search over (place, heading, held key, where the
    keys, ball and box lie, door states) with MiniGrid's rules; the ball and box may be carried aside, not
    toggled."""
    from collections import deque
    u = env.unwrapped
    wall, doors, keys, goal = set(), [], [], None
    for x in range(u.width):
        for y in range(u.height):
            o = u.grid.get(x, y)
            if o is None:
                continue
            if o.type == "wall":
                wall.add((x, y))
            elif o.type == "door":
                doors.append(((x, y), o.color))
            elif o.type in ("key", "ball", "box"):
                keys.append(((x, y), (o.type, o.color)))
            elif o.type == "goal":
                goal = (x, y)
    dpos = {p: i for i, (p, _) in enumerate(doors)}
    vec = ((1, 0), (0, 1), (-1, 0), (0, -1))
    s0 = (*u.agent_pos, u.agent_dir, None, frozenset(keys), (0,) * len(doors))
    seen, q = {s0}, deque([(s0, 0)])
    while q:
        (x, y, d, c, ks, ds), n = q.popleft()
        f = (x + vec[d][0], y + vec[d][1])
        kat = {p: col for p, col in ks}
        nxt = [(x, y, (d - 1) % 4, c, ks, ds), (x, y, (d + 1) % 4, c, ks, ds)]
        if f == goal:
            return n + 1
        if f not in wall and f not in kat and (f not in dpos or ds[dpos[f]] == 2):
            nxt.append((*f, d, c, ks, ds))
        if c is None and f in kat:
            nxt.append((x, y, d, kat[f], ks - {(f, kat[f])}, ds))
        if c is not None and f not in wall and f not in kat and f not in dpos:
            nxt.append((x, y, d, None, ks | {(f, c)}, ds))
        if f in dpos:
            i = dpos[f]
            st = 2 if ds[i] == 1 or (ds[i] == 0 and c == ("key", doors[i][1])) else 1 if ds[i] == 2 else 0
            nxt.append((x, y, d, c, ks, ds[:i] + (st,) + ds[i + 1:]))
        for s2 in nxt:
            if s2 not in seen:
                seen.add(s2)
                q.append((s2, n + 1))
    return None


def routes():
    """Fills in the shortest route of every episode in eval.json and the summary."""
    import gymnasium as gym
    import minigrid  # noqa: F401
    res = json.loads(EVAL.read_text())
    env = gym.make(register()).unwrapped
    for r in res["per_episode"]:
        env.reset(seed=r["seed"])
        r["shortest"] = shortest(env)
    recs = res["per_episode"]
    for v in ("one_door", "two_doors"):
        rs = [r for r in recs if r["variant"] == v]
        ok = [r for r in rs if r["success"]]
        res["summary"][v] = {"episodes": len(rs), "successes": len(ok),
                             "steps_over_shortest": round(float(np.mean([r["steps"] / r["shortest"] for r in ok])), 2)
                             if ok else None}
    EVAL.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res["summary"]), flush=True)


def evaluate():
    t00 = time.monotonic()
    log = lambda m: print(f"[{time.monotonic() - t00:6.0f}s] {m}", flush=True)
    T = setup_agent()
    W, info = T.setup(1, log)
    n = int(arg("--n", "30"))
    seeds = SEED_TEST + np.arange(n)
    pool = T.F.pool20()
    recs = []
    for r in pool.imap_unordered(episode, [(1, int(s)) for s in seeds]):
        recs.append(r)
        log(f"seed {r['seed']} {r['variant']}: success {r['success']} steps {r['steps']} random {r['random']} "
            f"explore {r['explore']} {r['seconds']}s ({len(recs)}/{n})")
    pool.close()
    recs.sort(key=lambda r: r["seed"])
    res = {"note": "tools/demos/chained_rooms.py eval: version 17 in MiniGrid's chained rooms", "summary": {},
           "setup": info, "per_episode": recs}
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL.write_text(json.dumps(res, indent=1, default=str) + "\n")
    log("episodes written")
    routes()


# ---------------------------------------------------------------- render

WID, HEI, FPS, TS = 1280, 720, 4, 40
BG = (28, 28, 34)


def ts(sec):
    return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"


def esc(s):
    return s.replace("{", "(").replace("}", ")")


def render():
    import gymnasium as gym
    import minigrid  # noqa: F401
    res = json.loads(EVAL.read_text())
    recs = res["per_episode"]
    s = {v: {"episodes": sum(r["variant"] == v for r in recs),
             "successes": sum(r["variant"] == v and r["success"] for r in recs)} for v in ("one_door", "two_doors")}
    pick = [r for r in recs if r["variant"] == "two_doors" and r["success"]][:3]
    pick += [r for r in recs if r["variant"] == "one_door" and r["success"]][:1]
    env = gym.make(register()).unwrapped
    out = Path(arg("--out", "demos/v17_chained_rooms.mp4"))
    ass = [f"[Script Info]\nScriptType: v4.00+\nPlayResX: {WID}\nPlayResY: {HEI}\nWrapStyle: 0\n\n[V4+ Styles]\n"
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding\n"
           "Style: D,DejaVu Sans,21,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
    frames, t = [], [0.0]
    X0, Y0 = 30, 60

    def canvas(img=None):
        c = np.empty((HEI, WID, 3), np.uint8)
        c[:] = BG
        if img is not None:
            c[Y0:Y0 + img.shape[0], X0:X0 + img.shape[1]] = img
        return c

    def show(img, sec, text, ml=X0 + WD * TS + 30, mv=Y0):
        n = max(1, round(sec * FPS))
        frames.append((img, n))
        ass.append(f"Dialogue: 0,{ts(t[0])},{ts(t[0] + n / FPS)},D,,{ml},30,{mv},,{text}\n")
        t[0] += n / FPS

    def rate(v):
        x = s[v]
        return f"{x['successes']} of {x['episodes']}"

    show(canvas(), 7, "{\\fs40}Version 17 in chained rooms{\\fs23}\\N\\N"
         "A learned world model (architecture version 17) acting in MiniGrid.\\N"
         "It sees only MiniGrid's 7 x 7 view ahead of it (the lit area), and walls\\Nand closed doors hide what is "
         "behind them. Its memory comes from random play;\\Nit plans backward from the goal over conditions it "
         "learned (what must be true first).\\N\\N"
         "Three rooms, two locked doors of different colours. With two doors, the goal\\Nroom's key is behind the "
         "other door: fetch key A, open door A, put it down,\\Nfetch key B, open door B, reach the goal.\\N\\N"
         f"Layouts it has never seen (memory reset each time):\\N   two doors: {rate('two_doors')}\\N"
         f"   one door: {rate('one_door')}", ml=80, mv=80)
    for k, r in enumerate(pick):
        env.reset(seed=r["seed"])
        title = (f"{{\\fs28}}Layout {k + 1} of {len(pick)}: {r['variant'].replace('_', ' ')}{{\\fs21}}\\N"
                 f"Seed {r['seed']}" + (f"; shortest route {r['shortest']} steps" if r.get("shortest") else ""))
        img = env.get_frame(highlight=True, tile_size=TS)
        show(canvas(img), 3, title + "\\N\\NThe agent (red triangle) starts in the middle room, knowing nothing\\N"
                                     "about this layout.")
        for i, stp in enumerate(r["steps_log"]):
            fast = i >= 40 and not r["success"]        # a failure: 40 steps as usual, then ten steps per frame
            if fast and i % 10:
                env.step(stp["a"])
                continue
            img = env.get_frame(highlight=True, tile_size=TS)
            why = "\\N".join(("    " * min(j, 1)) + ("<- " if j else "") + esc(x) for j, x in enumerate(stp["why"]))
            held = env.carrying
            held = f"{held.color} {held.type}" if held is not None else "nothing"
            show(canvas(img), 0.25 if fast else 0.5,
                 f"{title}\\N\\N{{\\fs25}}Step {i + 1}: {ACT[stp['a']]}{{\\fs21}}   (holding {held})"
                 + ("   {\\c&H00D7FF&}fast-forward, 10 steps per frame{\\c&HFFFFFF&}" if fast else "")
                 + f"\\N\\NWhy (each line is needed for the one above):\\N{why}")
            env.step(stp["a"])
        img = env.get_frame(highlight=True, tile_size=TS)
        end = ("{\\c&H64DC50&}Goal reached{\\c&HFFFFFF&}" if r["success"]
               else "{\\c&H3C3CF0&}Goal not reached{\\c&HFFFFFF&}")
        show(canvas(img), 3, f"{title}\\N\\N{{\\fs30}}{end} in {r['steps']} steps{{\\fs21}}"
                             + ("\\N(out of time)" if r.get("timed_out") else ""))

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
    print("wrote", out, f"{t[0]:.1f}s", [(r["seed"], r["variant"], r["success"]) for r in pick], flush=True)


if __name__ == "__main__":
    {"eval": evaluate, "routes": routes, "render": render}[MODE]()
