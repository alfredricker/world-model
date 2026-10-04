"""Card 052, step 1: the generator, and gate 1.

BabyAI's room grids (minigrid 3.1.0, `minigrid.core.roomgrid`, with BabyAI's verifier for success conditions),
changed as the card's step 1 asks:
  - colours: 12 hues instead of MiniGrid's 6; pink, brown and teal are held out of training (tests only);
  - every kind in every training colour: keys, balls, boxes, doors (closed, locked, or opened by a switch), and
    switches; pairings of keys, doors and switches are drawn per episode;
  - success conditions: go to, pick up, open, put next to, unlock (BabyAI's verifier), each ending the episode;
  - rooms: 1-4 rooms of size 5-10; a single room is split by a wall with a door 60% of the time;
  - obstacles: walls, lava, boxes, closed doors;
  - nuisance: per-episode floor tint and pixel noise (sigma 4/255) in the rendered 8-pixel tiles, and decorative
    coloured floor tiles that no action or goal involves.

A switch door looks like a locked door of its colour and opens when toggled while its switch (same colour,
reachable) is on; a switch is a plate with a lever, toggled on and off.

Gate 1 (the card's section 2, step 1):
  - event counts per kind and colour in 5,000 episodes of random play (100 steps each): pick-ups, door opens,
    unlocks (key and switch doors), switch presses, each success condition met; any below 30 per training colour
    gets play starts (the agent starts facing the event's object, holding the key for an unlock and the object to
    move for a put-next) until it reaches 30;
  - rendering: under random tint and noise, every (kind, state, hue) tile is nearest (L1) to its own clean tile
    in every draw: hues stay distinct from their neighbours at 8 pixels and the nuisance never makes two kinds
    or two colours look alike.

  bin/prun python tools/card052/generator.py --gate --episodes 5000 --out runs/052/gate1.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from minigrid.core import constants as K
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball, Box, Door, Floor, Key, Lava, Wall, WorldObj
from minigrid.envs.babyai.core import verifier as VF
from minigrid.utils.rendering import downsample, fill_coords, point_in_rect

# ---------------------------------------------------------------- the palette

NEW_HUES = {"orange": (255, 128, 0), "cyan": (0, 255, 255), "pink": (255, 105, 180), "brown": (150, 90, 30),
            "white": (230, 230, 230), "teal": (0, 128, 128)}
for _n, _c in NEW_HUES.items():                       # minigrid's tables are shared module objects: extend them
    if _n not in K.COLORS:
        K.COLORS[_n] = np.array(_c)
        K.COLOR_TO_IDX[_n] = len(K.COLOR_TO_IDX)
        K.IDX_TO_COLOR[K.COLOR_TO_IDX[_n]] = _n
        K.COLOR_NAMES.append(_n)
if "switch" not in K.OBJECT_TO_IDX:
    K.OBJECT_TO_IDX["switch"] = max(K.OBJECT_TO_IDX.values()) + 1
    K.IDX_TO_OBJECT[K.OBJECT_TO_IDX["switch"]] = "switch"
HUES = ["red", "green", "blue", "purple", "yellow", "grey"] + list(NEW_HUES)
HELD_OUT = ("pink", "brown", "teal")
TRAIN = [h for h in HUES if h not in HELD_OUT]
TILE = 8
NOISE = 4 / 255
TINT = 24 / 255                                       # largest floor tint per channel
MISSIONS = ("goto", "pickup", "open", "putnext", "unlock")


class Switch(WorldObj):
    def __init__(self, color):
        super().__init__("switch", color)
        self.is_on = False

    def can_overlap(self):
        return False

    def toggle(self, env, pos):
        self.is_on = not self.is_on
        return True

    def encode(self):
        return (K.OBJECT_TO_IDX[self.type], K.COLOR_TO_IDX[self.color], int(self.is_on))

    def render(self, img):
        c = K.COLORS[self.color]
        fill_coords(img, point_in_rect(0.15, 0.85, 0.15, 0.85), (70, 70, 70))
        if self.is_on:
            fill_coords(img, point_in_rect(0.38, 0.62, 0.22, 0.58), c)
        else:
            fill_coords(img, point_in_rect(0.38, 0.62, 0.42, 0.78), c // 2)


class SwitchDoor(Door):
    """A locked-looking door that opens when toggled while its switch is on (no key opens it)."""

    def __init__(self, color, switch=None):
        super().__init__(color, is_open=False, is_locked=True)
        self.switch = switch

    def toggle(self, env, pos):
        if self.is_locked:
            if self.switch is not None and self.switch.is_on:
                self.is_locked = False
                self.is_open = True
                return True
            return False
        self.is_open = not self.is_open
        return True


# ---------------------------------------------------------------- the levels

class Level(RoomGrid):
    """One configuration of rooms; a new layout, objects and success condition every reset."""

    def __init__(self, rows, cols, size, max_steps=100, colours=TRAIN, start=None):
        self.colours = list(colours)
        self.start = start                             # play start: (event, kind, colour) or None
        super().__init__(room_size=size, num_rows=rows, num_cols=cols, max_steps=max_steps,
                         mission_space=MissionSpace(mission_func=lambda: ""))

    # -- helpers
    def _col(self):
        return self._rand_elem(self.colours)

    def _reach(self, blocked_doors=True):
        """Cells reachable from the agent on foot, locked and switch doors counted closed for good."""
        W, H = self.grid.width, self.grid.height
        seen, todo = {tuple(self.agent_pos)}, [tuple(self.agent_pos)]
        while todo:
            x, y = todo.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (x + dx, y + dy)
                if p in seen or not (0 <= p[0] < W and 0 <= p[1] < H):
                    continue
                o = self.grid.get(*p)
                if o is None or o.type == "floor" or (o.type == "door" and not o.is_locked):
                    seen.add(p)
                    todo.append(p)
        return seen

    def _put(self, obj, cells=None):
        """Put obj on a random empty cell (of `cells` when given), not on the agent; False if none."""
        if cells is None:
            pos = self.place_obj(obj, max_tries=2000)
            return pos
        free = [p for p in sorted(cells) if self.grid.get(*p) is None and p != tuple(self.agent_pos)]
        if not free:
            return None
        p = free[self._rand_int(0, len(free))]
        self.grid.set(*p, obj)
        obj.init_pos, obj.cur_pos = p, p
        return p

    # -- generation
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self._sc = self._col()
        self.place_agent()
        want = self._door_start()                     # a play start that needs a door: (kind, colour)
        doors = self.connect_all(door_colors=self.colours) if self.num_rows * self.num_cols > 1 else []
        if self.num_rows * self.num_cols == 1 and (want or (self.room_size >= 6 and self._rand_float(0, 1) < 0.6)):
            doors = [self._split()]
        doors = [d for d in doors if d is not None]
        self.doors, self.switches = [], []
        for n, d in enumerate(doors):
            pos = d.cur_pos
            u = self._rand_float(0, 1)
            if n == 0 and want:
                d.color = want[1]
                u = {"locked": 0.0, "switch": 0.5, "closed": 0.9}[want[0]]
            if u < 0.35:                              # locked: its key where the agent can reach it
                d.is_locked, d.is_open = True, False
                self._put(Key(d.color), self._reach())
            elif u < 0.6:                             # opened by a switch of its colour
                sw = Switch(d.color)
                nd = SwitchDoor(d.color, sw)
                self.grid.set(*pos, nd)
                nd.init_pos, nd.cur_pos = pos, pos
                self._put(sw, self._reach())
                self.switches.append(sw)
                d = nd
            else:
                d.is_open = self._rand_float(0, 1) < 0.2 and not (n == 0 and want)
            self.doors.append(d)
        area = (self.room_size - 2) ** 2 * self.num_rows * self.num_cols
        for _ in range(min(self._rand_int(2, 7), area // 6)):    # objects of every kind, in training colours
            kind = self._rand_elem(["key", "ball", "box"])
            self._try_place({"key": Key, "ball": Ball, "box": Box}[kind](self._col()))
        if self._rand_float(0, 1) < 0.3:              # obstacles: lava
            self._try_place(Lava())
        for _ in range(min(self._rand_int(0, 4), area // 12)):  # decoration: coloured floor nothing involves
            self._try_place(Floor(self._col()))
        if self.start is not None:
            self._play_start()
        self.instrs = self._mission()
        self.mission = self.instrs.surface(self)

    def _try_place(self, obj):
        try:
            return self.place_obj(obj, max_tries=500)
        except RecursionError:
            return None

    def _door_start(self):
        if self.start is None:
            return None
        ev, kind, colour = self.start
        if colour == "-":                              # a success start: the door's colour drawn per episode
            colour = self._sc
        if ev in ("unlock_key", "success:unlock"):
            return ("locked", colour)
        if ev in ("unlock_switch", "switch"):
            return ("switch", colour)
        if ev in ("open", "success:open"):
            return ("closed", colour)
        return None

    def _split(self):
        """A single room split by a wall with a door (the key world's shape)."""
        s = self.room_size
        x = self._rand_int(2, s - 2)
        for y in range(1, s - 1):
            self.grid.set(x, y, Wall())
        y = self._rand_int(1, s - 1)
        d = Door(self._col())
        self.grid.set(x, y, d)
        d.init_pos, d.cur_pos = (x, y), (x, y)
        ax = self._rand_int(1, x) if self._rand_float(0, 1) < 0.5 else self._rand_int(x + 1, s - 1)
        ay = self._rand_int(1, s - 1)
        if self.grid.get(ax, ay) is None:
            self.agent_pos = np.array((ax, ay))
        return d

    def _movables(self):
        return [o for o in self.grid.grid if o is not None and o.type in ("key", "ball", "box")]

    def _mission(self):
        types = ["goto", "pickup", "putnext"]
        if self.doors:
            types.append("open")
        if any(d.is_locked for d in self.doors):
            types.append("unlock")
        if self.start is not None and self.start[0].startswith("success:"):
            types = [self.start[0].split(":")[1]]
        t = self._rand_elem(types)
        mov = self._movables()
        if self.start is not None and self.start[0] in ("success:goto", "success:pickup", "success:putnext"):
            x, y = self.agent_pos
            dx, dy = K.DIR_TO_VEC[self.agent_dir]
            f = self.grid.get(x + dx, y + dy)
            if f is not None and f.type in ("key", "ball", "box"):
                self.mtype = t
                if t == "goto":
                    return VF.GoToInstr(VF.ObjDesc(f.type, f.color))
                if t == "pickup":
                    return VF.PickupInstr(VF.ObjDesc(f.type, f.color))
                if getattr(self, "_pair", None) is not None:
                    a, o = self._pair
                    return VF.PutNextInstr(VF.ObjDesc(a.type, a.color), VF.ObjDesc(o.type, o.color))
        if t == "goto" or (t in ("pickup", "putnext") and len(mov) < 2):
            o = self._rand_elem(mov + self.doors) if (mov or self.doors) else None
            self.mtype = "goto"
            return VF.GoToInstr(VF.ObjDesc(o.type, o.color)) if o is not None else VF.GoToInstr(VF.ObjDesc("wall"))
        if t == "pickup":
            o = self._rand_elem(mov)
            self.mtype = "pickup"
            return VF.PickupInstr(VF.ObjDesc(o.type, o.color))
        if t == "putnext":
            a, b = self._rand_subset(mov, 2)
            self.mtype = "putnext"
            return VF.PutNextInstr(VF.ObjDesc(a.type, a.color), VF.ObjDesc(b.type, b.color))
        cand = [d for d in self.doors if d.is_locked] if t == "unlock" else self.doors
        d = self._rand_elem(cand)
        self.mtype = t
        return VF.OpenInstr(VF.ObjDesc("door", d.color))

    def _play_start(self):
        """Start facing the event's object (holding the key for an unlock, the object to move for a put-next)."""
        ev, kind, colour = self.start
        x, y = self.agent_pos
        dx, dy = K.DIR_TO_VEC[self.agent_dir]
        front = (x + dx, y + dy)
        want = self._door_start()
        if want is not None and self.doors:
            d = self.doors[0]
            if ev == "switch" and self.switches:
                self._face(self.switches[0].cur_pos)
                return
            if ev == "unlock_switch" and isinstance(d, SwitchDoor):
                d.switch.is_on = True                 # the switch already on: start facing its door
            self._face(d.cur_pos)
            if want[0] == "locked":
                self.carrying = Key(d.color)
                self.carrying.cur_pos = np.array([-1, -1])
            return
        if ev.startswith("success:goto") or ev.startswith("success:pickup") or ev.startswith("success:putnext"):
            kind = self._rand_elem(["key", "ball", "box"])
            colour = self._col()
        obj = {"key": Key, "ball": Ball, "box": Box}.get(kind, Ball)(colour)
        g = self.grid.get(*front)
        placed = 0 < front[0] < self.grid.width - 1 and 0 < front[1] < self.grid.height - 1 and g is None
        if placed:
            self.grid.set(*front, obj)
            obj.init_pos, obj.cur_pos = front, front
        self._pair = None
        if ev == "success:putnext" and placed:                   # the object to move in front, the other two ahead
            two = (x + 2 * dx, y + 2 * dy)
            if 0 < two[0] < self.grid.width - 1 and 0 < two[1] < self.grid.height - 1 and self.grid.get(*two) is None:
                other = {"key": Key, "ball": Ball, "box": Box}[self._rand_elem(["key", "ball", "box"])](self._col())
                self.grid.set(*two, other)
                other.init_pos, other.cur_pos = two, two
                self._pair = (obj, other)

    def _face(self, pos):
        """Put the agent on a free cell next to pos, facing it."""
        for d, (dx, dy) in enumerate(K.DIR_TO_VEC):
            p = (pos[0] - dx, pos[1] - dy)
            if 0 < p[0] < self.grid.width - 1 and 0 < p[1] < self.grid.height - 1 and self.grid.get(*p) is None:
                self.agent_pos = np.array(p)
                self.agent_dir = d
                return

    # -- acting
    def reset(self, **kw):
        obs = super().reset(**kw)
        self.instrs.reset_verifier(self)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if action == self.actions.drop:
            self.instrs.update_objs_poss()
        status = self.instrs.verify(action)
        info["success"] = status == "success"
        if status == "success":
            terminated = True
        return obs, reward, terminated, truncated, info


CONFIGS = [(1, 1, s) for s in range(5, 11)] + [(1, 2, s) for s in range(5, 9)] + [(2, 1, s) for s in range(5, 9)] \
    + [(2, 2, s) for s in range(5, 8)]


class Generator:
    """Episodes over the configurations, one Level per configuration (reused)."""

    def __init__(self, seed, colours=TRAIN, start=None):
        self.rng = np.random.default_rng(seed)
        self.levels = {}
        self.colours, self.start = colours, start

    def episode(self):
        cfg = CONFIGS[int(self.rng.integers(len(CONFIGS)))]
        key = (cfg, self.start)
        if key not in self.levels:
            self.levels[key] = Level(*cfg, colours=self.colours, start=self.start)
        env = self.levels[key]
        env.reset(seed=int(self.rng.integers(2 ** 31)))
        return env


# ---------------------------------------------------------------- events (evaluator side)

def snapshot(env):
    st = {}
    for o in env.grid.grid:
        if o is None:
            continue
        if o.type == "door":
            st[id(o)] = ("door", o.color, o.is_open, o.is_locked, isinstance(o, SwitchDoor))
        elif o.type == "switch":
            st[id(o)] = ("switch", o.color, o.is_on)
    return st, env.carrying


def events(env, before, after, action, info):
    out = []
    (s0, c0), (s1, c1) = before, after
    if c0 is None and c1 is not None:
        out.append(("pickup", c1.type, c1.color))
    for k, v in s0.items():
        w = s1.get(k)
        if w is None:
            continue
        if v[0] == "door" and not v[2] and w[2]:
            out.append(("unlock_switch" if v[4] else "unlock_key" if v[3] else "open", "door", v[1]))
        if v[0] == "switch" and v[2] != w[2]:
            out.append(("switch", "switch", v[1]))
    if info.get("success"):
        out.append(("success:" + env.mtype, "mission", "-"))
    return out


def random_play(gen, episodes, steps, rng):
    counts = Counter()
    t0 = time.monotonic()
    n_steps = 0
    for _ in range(episodes):
        env = gen.episode()
        for _ in range(steps):
            a = int(rng.integers(6))
            before = snapshot(env)
            _, _, term, trunc, info = env.step(a)
            n_steps += 1
            for e in events(env, before, snapshot(env), a, info):
                counts[e] += 1
            if term or trunc:
                break
    return counts, n_steps, time.monotonic() - t0


# ---------------------------------------------------------------- rendering check

def clean_tile(obj):
    """MiniGrid's tile drawing at 8 pixels (3 x 3 supersampling), without its grid lines: they are drawn in
    the grey of a grey box's outline, so an empty tile read as a grey box under noise (gate 1, first run)."""
    img = np.zeros((TILE * 3, TILE * 3, 3), dtype=np.uint8)
    if obj is not None:
        obj.render(img)
    return downsample(img, 3)


def tile(obj, tint, rng, noise=True):
    img = clean_tile(obj).astype(np.float64) / 255.0
    bg = img.sum(-1) == 0
    img[bg] += tint                                   # the floor tint shows on the background
    if noise:
        img = img + rng.normal(0, NOISE, img.shape)
    return np.clip(img, 0, 1)


def render_check(draws, rng):
    objs = {}
    for h in HUES:
        objs[("key", h)] = Key(h)
        objs[("ball", h)] = Ball(h)
        objs[("box", h)] = Box(h)
        objs[("door closed", h)] = Door(h)
        objs[("door open", h)] = Door(h, is_open=True)
        objs[("door locked", h)] = Door(h, is_locked=True)
        objs[("floor", h)] = Floor(h)
        s = Switch(h)
        objs[("switch off", h)] = s
        s2 = Switch(h)
        s2.is_on = True
        objs[("switch on", h)] = s2
    objs[("lava", "-")] = Lava()
    objs[("empty", "-")] = None
    names = list(objs)
    clean = np.stack([tile(objs[n], np.zeros(3), rng, noise=False).ravel() for n in names])
    wrong = Counter()
    nearest_other = {}
    same = Counter()
    for i, n in enumerate(names):
        d = np.abs(clean - clean[i]).sum(1)
        d[i] = np.inf
        j = int(d.argmin())
        nearest_other[f"{n[0]} {n[1]}"] = [f"{names[j][0]} {names[j][1]}", round(float(d[j]) / (TILE * TILE * 3), 4)]
        for _ in range(draws):
            tint = rng.uniform(0, TINT, 3)
            t = tile(objs[n], tint, rng).ravel()
            k = int(np.abs(clean - t).sum(1).argmin())        # against untinted tiles (tint unknown)
            if k != i:
                wrong[(f"{n[0]} {n[1]}", f"{names[k][0]} {names[k][1]}")] += 1
            tinted = np.stack([tile(objs[m], tint, rng, noise=False).ravel() for m in names])
            k = int(np.abs(tinted - t).sum(1).argmin())       # against the same tint (the episode's)
            if k != i:
                same[(f"{n[0]} {n[1]}", f"{names[k][0]} {names[k][1]}")] += 1
    closest = sorted(nearest_other.items(), key=lambda x: x[1][1])[:10]
    return {"tiles": len(names), "draws_each": draws,
            "misread_same_tint": {f"{a} -> {b}": v for (a, b), v in same.items()},
            "misread": {f"{a} -> {b}": v for (a, b), v in wrong.items()},
            "closest_pairs_mean_abs_per_channel": closest,
            "noise_mean_abs_per_channel": round(float(np.sqrt(2 / np.pi) * NOISE), 4)}


# ---------------------------------------------------------------- gate 1

def deficient(counts):
    need = []
    for c in TRAIN:
        for k in ("key", "ball", "box"):
            if counts[("pickup", k, c)] < 30:
                need.append(("pickup", k, c))
        for ev in ("open", "unlock_key", "unlock_switch"):
            if counts[(ev, "door", c)] < 30:
                need.append((ev, "door", c))
        if counts[("switch", "switch", c)] < 30:
            need.append(("switch", "switch", c))
    for m in MISSIONS:
        if counts[("success:" + m, "mission", "-")] < 30:
            need.append(("success:" + m, "mission", "-"))
    return need


def render_only(out, seed=52000):
    r = json.loads(Path(out).read_text())
    r["render_first_run_with_grid_lines"] = {"misread": r["render"]["misread"]}
    r["render"] = render_check(200, np.random.default_rng(seed + 7))
    Path(out).write_text(json.dumps(r, indent=1) + "\n")
    print("same tint", r["render"]["misread_same_tint"], "tint unknown", r["render"]["misread"],
          r["render"]["closest_pairs_mean_abs_per_channel"][:4])


def gate(episodes, out, seed=52000):
    rng = np.random.default_rng(seed)
    gen = Generator(seed)
    counts, n_steps, secs = random_play(gen, episodes, 100, rng)
    base = Counter(counts)
    need = deficient(counts)
    print("random play:", episodes, "episodes,", n_steps, "steps,", round(secs, 1), "s; below 30:", len(need), flush=True)
    starts = Counter()
    rounds = 0
    while need and rounds < 40:
        rounds += 1
        for ev in need:
            evname = ev[0]
            st = (evname, ev[1], ev[2])
            g = Generator(seed + 1000 * rounds + hash(ev) % 1000, start=st)
            c, n, _ = random_play(g, 50, 100, rng)
            starts[ev] += 50
            counts.update(c)
        need = deficient(counts)
        print("play starts round", rounds, "still below 30:", len(need), flush=True)
    rc = render_check(200, rng)
    table = {}
    for c in TRAIN + list(HELD_OUT):
        table[c] = {f"pickup {k}": counts[("pickup", k, c)] for k in ("key", "ball", "box")}
        table[c].update({ev: counts[(ev, "door", c)] for ev in ("open", "unlock_key", "unlock_switch")})
        table[c]["switch"] = counts[("switch", "switch", c)]
    res = {"note": "Card 052 gate 1, tools/card052/generator.py", "episodes": episodes, "steps": n_steps,
           "seconds": round(secs, 1), "train_colours": TRAIN, "held_out": list(HELD_OUT),
           "random_play_only": {f"{a} {b} {c}": v for (a, b, c), v in sorted(base.items())},
           "with_play_starts": table,
           "successes": {m: counts[("success:" + m, "mission", "-")] for m in MISSIONS},
           "successes_random_only": {m: base[("success:" + m, "mission", "-")] for m in MISSIONS},
           "play_start_episodes": {f"{a} {b} {c}": v for (a, b, c), v in starts.items()},
           "still_below_30": [f"{a} {b} {c}" for a, b, c in need],
           "render": rc}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("random_play_only", "with_play_starts")}, indent=1)[:4000])


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    if "--render" in args:
        render_only(get("--out", "runs/052/gate1.json"))
    elif "--gate" in args:
        gate(int(get("--episodes", "5000")), get("--out", "runs/052/gate1.json"))


if __name__ == "__main__":
    main()
