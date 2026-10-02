"""Card 045's agent in unseen rooms, as a video (a demo, not an experiment).

Seed 400's encoder (arm A), architecture 7 with card 045's walking, learning online as in the runs. Each step
shows the plan behind the next move: the conditions worked backward from the goal, the route System 1's approach
predicts (cyan), its waypoints (white), where it will stand (green), the thing it is heading for (yellow) and a
tile that has become a condition (red). The simulator draws the room; the agent sees only its own view.

  nice -n 19 bin/prun python tools/card045/demo.py --seed 400 --out runs/045_demo_unseen.mp4 [--rerun]

The episodes are kept next to the video (.episodes.pkl), so drawing again does not rerun the agent; --rerun does.
"""
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import movement as MV                                  # noqa: E402
from minigrid.core.grid import Grid                    # noqa: E402
from worldmodel.envs import keydoor_render as KR      # noqa: E402

VP, F, CS, ld = MV.VP, MV.F, MV.CS, MV.ld
EPISODES = (("key", "room6"), ("switch", "room7"), ("either", "mirror8"), ("both", "room7"), ("key", "mirror8"),
            ("both", "room6"))
RULES = {"key": "the door opens with the key of its colour", "switch": "the door opens once the switch is on",
         "either": "the door opens with its key or the switch", "both": "the door needs its key and the switch on"}
ROOMS = {"room6": "a 6 x 6 room (training used only 8 x 8)", "room7": "a 7 x 7 room (training used only 8 x 8)",
         "mirror8": "a mirrored 8 x 8 room (start room on the right)"}
ACT = {0: "turn left", 1: "turn right", 2: "forward", 3: "pick up", 4: "drop", 5: "toggle"}
VERB = {2: "step onto it", 3: "pick it up", 4: "drop onto it", 5: "toggle it"}
TS, WID, HEI, FPS, X0, Y0 = 72, 1280, 720, 5, 40, 40
BG, CYAN, WHITE, GREEN, YELLOW, RED = (28, 28, 34), (0, 200, 255), (255, 255, 255), (80, 220, 100), (255, 215, 0), (240, 60, 60)
_tiles = {}


def tile(c):
    if c not in _tiles:
        a = c % 5 - 1
        _tiles[c] = Grid.render_tile(KR._minigrid_obj(KR.OBJECTS[c // 5]), agent_dir=None if a < 0 else a,
                                     highlight=False, tile_size=TS)
    return _tiles[c]


def origin(lay):
    off = (8 - lay.size) * TS // 2
    return X0 + off, Y0 + off


def at(lay, cell):
    ox, oy = origin(lay)
    return ox + int(cell[0]) * TS, oy + int(cell[1]) * TS


def border(img, xy, colour, t, inset=0):
    x, y = xy[0] + inset, xy[1] + inset
    n = TS - 2 * inset
    img[y:y + t, x:x + n] = colour
    img[y + n - t:y + n, x:x + n] = colour
    img[y:y + n, x:x + t] = colour
    img[y:y + n, x + n - t:x + n] = colour


def frame(lay, s, marks=None):
    img = np.empty((HEI, WID, 3), np.uint8)
    img[:] = BG
    g = KR.encode_logic_states(lay, [s])[0]
    n = lay.size
    for y in range(n):
        for x in range(n):
            px, py = at(lay, (x, y))
            img[py:py + TS, px:px + TS] = tile(int(g[y, x]))
    hy = Y0 + 8 * TS + 12
    img[hy:hy + TS, X0 + 150:X0 + 150 + TS] = tile(int(g[n, 0]))
    border(img, (X0 + 150, hy), (90, 90, 100), 2)
    if marks:
        for c in marks.get("route", ()):
            px, py = at(lay, c)
            img[py:py + TS, px:px + TS] = (img[py:py + TS, px:px + TS] * 0.6 + np.array(CYAN) * 0.4).astype(np.uint8)
        for key, colour, t, inset in (("stand", GREEN, 4, 4), ("waypoints", WHITE, 4, 10), ("condition", RED, 6, 0),
                                      ("target", YELLOW, 5, 8)):
            for c in marks.get(key, ()):
                border(img, at(lay, c), colour, t, inset)
    return img


def nice(name):
    w = name.split()
    if w[0] == "key":
        return f"{w[1]} key"
    if "door" in w:
        return f"{w[0]} {w[-1]} door"
    if w[0] == "switch":
        return f"switch ({w[1]})"
    return name


def describe(item, f, nm):
    k = item[0]
    if k == "end":
        return "reach the goal"
    if k == "face":
        return f"face the {nice(nm(item[2]))}, to {VERB.get(item[1], ACT.get(item[1], '?'))}"
    if k == "walk":
        v = item[1]
        if v == "blocked":
            return "no clear way there"
        if v == "approach":
            return "{\\c&HFFC800&}System 1: learned approach (no imagined steps){\\c&HFFFFFF&}"
        if v == "waypoint":
            return "{\\c&HFFC800&}System 1: approach a waypoint first{\\c&HFFFFFF&}"
        if isinstance(v, (int, np.integer)):
            return "{\\c&H3C3CF0&}condition: the " + nice(nm(int(f[v]))) + " tile walkable{\\c&HFFFFFF&}"
        return str(v)
    if k == "part":
        what = f"{ACT[item[2]]} the {nice(nm(item[3]))}"
        return (f"condition: hold the right thing to {what}" if item[1] == VP.HELDP
                else f"condition: change what is around, so as to {what}")
    if k == "act":
        return f"do it: {ACT[item[1]]}"
    return str(item)


def hops(pl, ch, c):
    """The placements a least chain from c passes: (placement, whether it is the need's)."""
    out, i, k = [], ch.ix.get(int(c)), None
    if i is None or ch.cost(i) >= MV.INF:
        return out
    while not ch.Pm[i]:
        j, direct, k = ch.hop(i, k)
        out.append((int(ch.S[j]), direct))
        i = j
        if direct:
            break
    return out


def marks_of(pl, st, lay, res, f):
    M = F.M
    m = {"route": [], "waypoints": [], "stand": [], "target": [], "condition": []}
    tr = res.trace
    for item in tr:
        if item[0] == "walk" and isinstance(item[1], (int, np.integer)):
            m["condition"].append(MV.cell_of(lay, int(item[1])))
    faces = [i for i, item in enumerate(tr) if item[0] == "face"]
    if not faces:
        return m
    need = tr[faces[-1]]
    toks = [need[3]] if need[3] is not None else pl.showing(f, need[2])
    m["target"] = [MV.cell_of(lay, t) for t in toks]
    if tr[-1][0] == "walk" and tr[-1][1] in ("approach", "waypoint"):
        ch = pl.plan(st[0], pl.face_pid(st[0], need))
        m["route"] = [MV.cell_of(lay, t) for t in pl.stepped(ch, st[1])]
        for p, direct in hops(pl, ch, st[1]):
            (m["stand"] if direct else m["waypoints"]).append(MV.cell_of(lay, M.cidx[p]))
    return m


def episode(W, lay):
    nm = lambda h: W.judge.name(int(h))
    W.reset()
    pl = VP.VPlan(W)
    s = ld.start_state(lay)
    V = VP.see(lay, s)
    facts, st, _, _ = pl.observe(None, V)
    steps, rng = [], np.random.default_rng(0)
    for t in range(VP.BUDGET):
        res = pl.choose(st)
        f = pl.facts[st[0]]
        if res is None:
            a, lines, marks = int(rng.integers(6)), ["no plan: a random move"], {}
        else:
            a = res.action
            lines = [describe(x, f, nm) for x in res.trace]
            marks = marks_of(pl, st, lay, res, f)
        steps.append({"s": s, "a": a, "lines": lines, "marks": marks, "imagined": pl.walk_moves,
                      "kinds": [x[1] for x in (res.trace if res else []) if x[0] == "walk"]})
        pred = pl.step(st, a)
        s2, end = ld.step(lay, s, a)
        V2 = VP.see(lay, s2)
        facts, st, _, _ = pl.observe(facts, V2, end, prefer=pred[1])
        W.learn_try(V, a, V2, end)
        pl.forget()
        s, V = s2, V2
        if end:
            return steps, s, True
    return steps, s, False


def ts(sec):
    return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    seed, out = int(get("--seed", "400")), Path(get("--out", "runs/045_demo_unseen.mp4"))
    cached = out.with_suffix(".episodes.pkl")
    if cached.exists() and "--rerun" not in args:
        chosen = pickle.loads(cached.read_bytes())
        EP = ()
    else:
        chosen, EP = [], EPISODES
        CS.install()
        MV.install("045")
        import slot_planner as SP_
        SP_.install()
        VP.Kind, VP.vectors_of = MV.CR.kind, MV.CR.vectors_of
        VP.T.configure()
        cache = pickle.loads(VP.T.MEMORY.read_bytes())
        groups = VP.T.use_groups(cache["mem"], "cuda")
        z, parts, _ = VP.vectors_of("A", seed, cache["tiles"], cache["pairs"], groups, "cuda", lambda *a: None)
        data = pickle.loads(VP.T.DATA.read_bytes())["data"]
        U = pickle.loads(MV.UNSEEN.read_bytes())
        S = VP.Store(z)
        lut = np.zeros(256, np.uint8)
        lut[:len(S.hof)] = S.hof
        VP.Kd.APP = lut
    worlds = {}
    for world, room in EP:
        if world not in worlds:
            worlds[world] = VP.World(S, parts, data[world], "cuda", lambda *a: None)
        W = worlds[world]
        VP.WORLD, F.M = W, W.M
        best = None
        for idx, lay in enumerate(U[world][room]["test"][:60]):
            steps, last, done = episode(W, lay)
            kinds = {k for st_ in steps for k in st_["kinds"]}
            score = (done, "waypoint" in kinds, any(isinstance(k, (int, np.integer)) for k in kinds), 12 <= len(steps) <= 28)
            if best is None or score > best[0]:
                best = (score, idx, lay, steps, last, done)
            if all(score):
                break
        _, idx, lay, steps, last, done = best
        chosen.append((world, room, idx, lay, steps, last, done, U[world][room]["shortest_each"][idx]))
        print(world, room, "layout", idx, "steps", len(steps), "done", done, "shortest",
              U[world][room]["shortest_each"][idx], "imagined", steps[-1]["imagined"], flush=True)

    cached.write_bytes(pickle.dumps(chosen))
    ass = [f"[Script Info]\nScriptType: v4.00+\nPlayResX: {WID}\nPlayResY: {HEI}\nWrapStyle: 0\n\n[V4+ Styles]\n"
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding\n"
           "Style: D,DejaVu Sans,22,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
    tx = 680
    frames, t = [], 0.0

    def show(img, sec, text, held_label=True):
        nonlocal t
        n = max(1, round(sec * FPS))
        frames.append((img, n))
        body = text + ("" if not held_label else "")
        ass.append(f"Dialogue: 0,{ts(t)},{ts(t + n / FPS)},D,,{tx},30,{Y0},,{body}\n")      # wraps in the panel
        if held_label:
            ass.append(f"Dialogue: 0,{ts(t)},{ts(t + n / FPS)},D,,0,0,0,,{{\\pos({X0},{Y0 + 8 * TS + 34})}}Holding:\n")
        t += n / FPS

    legend = ("\\N\\N{\\fs19}{\\c&HFFC800&}cyan{\\c&HFFFFFF&} predicted route   {\\c&HFFFFFF&}white{\\c&HFFFFFF&} waypoint   "
              "{\\c&H64DC50&}green{\\c&HFFFFFF&} where it will stand\\N{\\c&H00D7FF&}yellow{\\c&HFFFFFF&} the thing it is "
              "heading for   {\\c&H3C3CF0&}red{\\c&HFFFFFF&} tile made a condition")
    intro = np.empty((HEI, WID, 3), np.uint8)
    intro[:] = BG
    frames.append((intro, 6 * FPS))
    ass.append(f"Dialogue: 0,{ts(0)},{ts(6)},D,,80,80,140,,{{\\fs40}}A learned world model in rooms it "
               "has never seen{\\fs24}\\N\\NArchitecture 7 with card 045: walking through conditions.\\N\\N"
               "The agent learned what its actions do from its own experience in 8 x 8 rooms.\\NIt plans backward from "
               "the goal over conditions (what must be true first).\\NWalking is a learned System 1 approach; an "
               "obstacle on its route becomes a condition.\\NNo step is imagined while walking.\\N\\NHere: rooms of size "
               "6 and 7, and mirrored rooms. Seed 400's encoder.\n")
    t = 6.0
    for k, (world, room, idx, lay, steps, last, done, short) in enumerate(chosen):
        head = (f"{{\\fs30}}Unseen room {k + 1} of {len(chosen)}{{\\fs23}}\\N{ROOMS[room]}\\N"
                f"Rule: {RULES[world]}")
        show(frame(lay, steps[0]["s"]), 3.0, head + "\\N\\NThe agent starts knowing nothing about this room.")
        for i, st_ in enumerate(steps):
            plan = "\\N".join(("     " * min(j, 1)) + ("← " if j else "") + x for j, x in enumerate(st_["lines"][:9]))
            text = (f"{head}\\N\\N{{\\fs26}}Step {i + 1}: {ACT[st_['a']]}{{\\fs23}}\\N\\N"
                    f"Why (each line is needed for the one above):\\N{plan}\\N\\N"
                    f"Imagined steps while walking so far: {st_['imagined']}" + legend)
            show(frame(lay, st_["s"], st_["marks"]), 0.8, text)
        end = ("{\\c&H64DC50&}Goal reached{\\c&HFFFFFF&}" if done else "{\\c&H3C3CF0&}Goal not reached{\\c&HFFFFFF&}")
        show(frame(lay, last), 2.5, f"{head}\\N\\N{{\\fs30}}{end} in {len(steps)} steps{{\\fs23}}\\N"
                                    f"Shortest possible: {short} steps")

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
    print("wrote", out, f"{t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
