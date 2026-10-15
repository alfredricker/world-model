"""Card 102's breaking worlds (CHARTER rule 10): small grid worlds, each built to defeat one environment-specific
assumption. They give pixels and actions only, drawn as Crafter's frames are (64 x 64; a 9 x 7 grid of 7 x 7-pixel
tiles above two rows of carried items), so Crafter's adapter reads them unchanged. The evaluator keeps the state.

World 1, "near, anywhere" (card 102.1): a room seen whole (the view does not follow the agent). The agent moves in
four directions; walking into a box pushes it when the cell beyond is free. A lamp lights while a box stands in one
of the four cells beside it. Lamps stand in interior columns 1-3 in training worlds and 5-7 in test worlds, so the
relation "box beside lamp" must carry to positions never seen.

  bin/prun python tools/card102/worlds.py 1      → runs/card102/world1_{train,test}.npz
  bin/prun python tools/card102/worlds.py 2      → runs/card102/world2_{train,test}.npz

World 2, "shapes" (card 102.2): shapes of 2-5 tiles, half of one colour (so a moved bar changes only at its ends),
that the agent pushes as wholes, and a door of 2-3 tiles in the top or bottom wall that opens or closes as a whole
when the agent walks into any of its tiles. Test worlds use free shapes never seen in training.

World 3, "out of sight" (card 102.3): a 20 x 20 room with internal walls seen through a 9 x 7 view that follows the
agent; a ball rolling and reversing where blocked, cats wandering, boxes that stay, lights that vanish. The
evaluator keeps every object's identity and world place (obj rows: step, id, kind, dx, dy, x, y).

  bin/prun python tools/card102/worlds.py 3      → runs/card102/world3_{train,test}.npz

World 4b, "no body" (card 102.5): world 1's room seen whole, with no player. Each arrow action moves every blue tile
one cell in its direction where the cell is free (a blue tile behind another moves with it); red tiles never move.
World 1 and world 2 are world 4a, "a body in a fixed view".

  bin/prun python tools/card102/worlds.py 4      → runs/card102/world4_{train,test}.npz

World 5, "clicks" (card 102.6): world 2's room and shapes seen whole, with no player. An action is "wait" (0) or a
click at one of the 63 cells (1 + row * 9 + column). A click on a shape's tile recolours every tile of the shape to
the next colour in a fixed cycle; a click on a lamp toggles it; elsewhere nothing happens. Training never clicks
column 6; test worlds use world 2's held-out shapes and click everywhere.

  bin/prun python tools/card102/worlds.py 5      → runs/card102/world5_{train,test}.npz

World 6, "searched caves" (card 109): a corridor with six dead-end caves seen through a 9 x 7 view that follows the
agent; a cave's end shows only from inside. Dark caves may hold a gem, which never comes back once taken; sand caves
hold a bush whose berry regrows 30-60 steps after it is eaten. The evaluator keeps the whole grid and the events.

  bin/prun python tools/card102/worlds.py 6      → runs/card102/world6_{train,test}.npz
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "card102"
W, H, T = 9, 7, 7                                     # grid columns, rows, tile size in pixels
FLOOR, WALL, AGENT, BOX, LAMP, LIT = range(6)
NAMES = ["floor", "wall", "agent", "box", "lamp", "lit lamp"]
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # Crafter's ids: left, right, up, down; 0 is noop


def tile(kind):
    """A 7 x 7 tile: a background colour with a simple mark, so every kind differs in pixels."""
    t = np.zeros((T, T, 3), np.uint8)
    if kind == FLOOR:
        t[:] = (70, 70, 70)
    elif kind == WALL:
        t[:] = (150, 110, 60)
        t[3, :] = (110, 80, 40)
    elif kind == AGENT:
        t[:] = (70, 70, 70)
        t[1:6, 1:6] = (60, 160, 230)
        t[2:5, 2:5] = (240, 240, 240)
    elif kind == BOX:
        t[:] = (70, 70, 70)
        t[1:6, 1:6] = (200, 150, 50)
        t[1, 1:6] = t[5, 1:6] = t[1:6, 1] = t[1:6, 5] = (120, 80, 20)
    elif kind in (LAMP, LIT):
        t[:] = (70, 70, 70)
        t[2:5, 1:6] = t[1:6, 2:5] = (90, 90, 160) if kind == LAMP else (255, 230, 80)
        t[3, 3] = (30, 30, 60) if kind == LAMP else (255, 255, 255)
    return t


TILES = np.stack([tile(k) for k in range(6)])


def render(grid):
    """grid (H, W) of kinds → a 64 x 64 frame (the carried rows empty, the last pixel row and column padding)."""
    f = np.zeros((64, 64, 3), np.uint8)
    f[:H * T, :W * T] = TILES[grid].transpose(0, 2, 1, 3, 4).reshape(H * T, W * T, 3)
    return f


class World1:
    """Boxes and lamps in a 9 x 7 room (interior columns 1-7, rows 1-5)."""

    def __init__(self, rng, test):
        self.rng = rng
        g = np.full((H, W), FLOOR, np.int64)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
        cols = [5, 6, 7] if test else [1, 2, 3]
        free = lambda: [(x, y) for y in range(1, H - 1) for x in range(1, W - 1) if g[y, x] == FLOOR]
        for _ in range(int(rng.integers(1, 3))):
            opts = [(x, y) for (x, y) in free() if x in cols]
            x, y = opts[rng.integers(len(opts))]
            g[y, x] = LAMP
        for _ in range(int(rng.integers(2, 4))):
            opts = free()
            x, y = opts[rng.integers(len(opts))]
            g[y, x] = BOX
        opts = free()
        self.agent = np.array(opts[rng.integers(len(opts))])
        self.g = g
        self.light()

    def light(self):
        for y, x in zip(*np.nonzero((self.g == LAMP) | (self.g == LIT))):
            near = any(self.g[y + dy, x + dx] == BOX for dx, dy in MOVES.values())
            self.g[y, x] = LIT if near else LAMP

    def grid(self):
        g = self.g.copy()
        g[self.agent[1], self.agent[0]] = AGENT
        return g

    def step(self, a):
        """Returns whether a box was pushed."""
        if a not in MOVES:
            return False
        d = np.array(MOVES[a])
        nx, ny = self.agent + d
        pushed = False
        if self.g[ny, nx] == FLOOR:
            self.agent = np.array([nx, ny])
        elif self.g[ny, nx] == BOX:
            bx, by = np.array([nx, ny]) + d
            if self.g[by, bx] == FLOOR:
                self.g[by, bx], self.g[ny, nx] = BOX, FLOOR
                self.agent = np.array([nx, ny])
                pushed = True
        self.light()
        return pushed


def play(world, n_ep, steps, seed, test):
    """Random play, stored in card 098.5's format: obs, act, ep, done, sem (the evaluator's kinds, 7 x 9)."""
    rng = np.random.default_rng(seed)
    n = n_ep * steps
    obs, act = np.zeros((n, 64, 64, 3), np.uint8), np.zeros(n, np.int8)
    sem, ep, done, push = np.zeros((n, H, W), np.int16), np.zeros(n, np.int32), np.zeros(n, bool), np.zeros(n, bool)
    t = 0
    for e in range(n_ep):
        w = world(rng, test)
        for s in range(steps):
            g = w.grid()
            obs[t], sem[t], ep[t] = render(g), g, e
            a = int(rng.integers(0, 5))
            act[t] = a
            push[t] = w.step(a)
            done[t] = s == steps - 1
            t += 1
    return {"obs": obs, "act": act, "sem": sem, "ep": ep, "done": done, "push": push,
            "light": np.ones(n, np.float32), "names": np.array(NAMES)}


# ------------------------------------------------------------------------------------------------ world 2, "shapes"
DOOR, OPEN = 6, 7                                      # world 2's kinds beyond world 1's; shape tiles are 8 + colour
COLOURS = [(220, 60, 60), (60, 180, 80), (70, 110, 230), (230, 200, 50), (200, 90, 210), (60, 200, 210)]
NAMES2 = NAMES + ["door", "open door"] + [f"shape tile {i}" for i in range(len(COLOURS))]


def tile2(kind):
    if kind < 6:
        return tile(kind if kind != 2 else AGENT)
    t = np.zeros((T, T, 3), np.uint8)
    t[:] = (70, 70, 70)
    if kind == DOOR:
        t[:] = (120, 60, 20)
        t[:, 1] = t[:, 3] = t[:, 5] = (60, 30, 10)
    elif kind == OPEN:
        t[:] = (25, 25, 25)
        t[0, :] = t[-1, :] = (120, 60, 20)
    else:
        t[:] = COLOURS[kind - 8]
        t[0, :] = t[:, 0] = tuple(int(0.6 * c) for c in COLOURS[kind - 8])
    return t


TILES2 = np.stack([tile2(k) for k in range(8 + len(COLOURS))])


def render2(grid):
    f = np.zeros((64, 64, 3), np.uint8)
    f[:H * T, :W * T] = TILES2[grid].transpose(0, 2, 1, 3, 4).reshape(H * T, W * T, 3)
    return f


def polyominoes(nmax=5):
    """Every fixed polyomino of 2..nmax cells, grouped by free shape (rotations and reflections)."""
    shapes = {frozenset([(0, 0)])}
    out = {}
    for n in range(2, nmax + 1):
        grown = set()
        for sh in shapes:
            for x, y in sh:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c = (x + dx, y + dy)
                    if c not in sh:
                        g = sh | {c}
                        mx, my = min(p[0] for p in g), min(p[1] for p in g)
                        grown.add(frozenset((p[0] - mx, p[1] - my) for p in g))
        shapes = grown
        for sh in shapes:
            forms = []
            for rot in range(4):
                for refl in (False, True):
                    q = list(sh)
                    for _ in range(rot):
                        q = [(-y, x) for x, y in q]
                    if refl:
                        q = [(-x, y) for x, y in q]
                    mx, my = min(p[0] for p in q), min(p[1] for p in q)
                    forms.append(tuple(sorted((x - mx, y - my) for x, y in q)))
            out.setdefault(min(forms), set()).add(tuple(sorted(sh)))
    return {k: sorted(v) for k, v in out.items()}


FREE = polyominoes()
FREE_KEYS = sorted(FREE, key=lambda k: (len(k), k))
TEST_FREE = set(FREE_KEYS[i] for i in (2, 5, 7, 11, 14, 17, 19))  # held out: 1 of 2 trominoes, 2 of 5 tetrominoes, 4 of 12 pentominoes


class World2:
    """Shapes of 2-5 tiles that the agent pushes as wholes, and a door of 2-3 tiles in the top or bottom wall that
    opens or closes as a whole when the agent walks into any of its tiles. The room is seen whole (9 x 7)."""

    def __init__(self, rng, test):
        self.rng = rng
        g = np.full((H, W), FLOOR, np.int64)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
        obj = np.zeros((H, W), np.int64)
        row = int(rng.choice([0, H - 1]))
        n = int(rng.integers(2, 4))
        x0 = int(rng.integers(1, W - n))
        g[row, x0:x0 + n] = DOOR
        obj[row, x0:x0 + n] = 1
        self.door = [(x, row) for x in range(x0, x0 + n)]
        keys = [k for k in FREE_KEYS if (k in TEST_FREE) == test]
        nid = 2
        for _ in range(int(rng.integers(2, 4))):
            for _try in range(50):
                k = keys[rng.integers(len(keys))]
                forms = FREE[k]
                cells = forms[rng.integers(len(forms))]
                ox, oy = int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1))
                pos = [(ox + x, oy + y) for x, y in cells]
                if all(1 <= x < W - 1 and 1 <= y < H - 1 and g[y, x] == FLOOR for x, y in pos):
                    one = rng.random() < 0.5
                    c0 = int(rng.integers(len(COLOURS)))
                    for x, y in pos:
                        g[y, x] = 8 + (c0 if one else int(rng.integers(len(COLOURS))))
                        obj[y, x] = nid
                    nid += 1
                    break
        free = [(x, y) for y in range(1, H - 1) for x in range(1, W - 1) if g[y, x] == FLOOR]
        self.agent = np.array(free[rng.integers(len(free))])
        self.g, self.obj = g, obj

    def grid(self):
        g = self.g.copy()
        g[self.agent[1], self.agent[0]] = AGENT
        return g

    def objects(self):
        o = self.obj.copy()
        o[self.agent[1], self.agent[0]] = 99              # the agent is an object of one tile
        return o

    def step(self, a):
        """Returns 'push', 'door' or ''."""
        if a not in MOVES:
            return ""
        d = np.array(MOVES[a])
        nx, ny = self.agent + d
        k = self.g[ny, nx]
        if k == FLOOR:
            self.agent = np.array([nx, ny])
        elif k in (DOOR, OPEN):
            for x, y in self.door:
                self.g[y, x] = OPEN if self.g[y, x] == DOOR else DOOR
            return "door"
        elif k >= 8:
            oid = self.obj[ny, nx]
            cells = list(zip(*np.nonzero(self.obj == oid)))  # (y, x)
            dest = [(y + d[1], x + d[0]) for y, x in cells]
            if all(self.obj[y, x] in (0, oid) and self.g[y, x] in (FLOOR,) + tuple(range(8, 8 + len(COLOURS)))
                   for y, x in dest):
                kinds = [self.g[y, x] for y, x in cells]
                for y, x in cells:
                    self.g[y, x], self.obj[y, x] = FLOOR, 0
                for (y, x), kk in zip(dest, kinds):
                    self.g[y, x], self.obj[y, x] = kk, oid
                self.agent = np.array([nx, ny])
                return "push"
        return ""


def play2(n_ep, steps, seed, test):
    rng = np.random.default_rng(seed)
    n = n_ep * steps
    obs = np.zeros((n, 64, 64, 3), np.uint8)
    act, ep, done = np.zeros(n, np.int8), np.zeros(n, np.int32), np.zeros(n, bool)
    sem, obj = np.zeros((n, H, W), np.int16), np.zeros((n, H, W), np.int16)
    event = np.zeros(n, "U4")
    t = 0
    for e in range(n_ep):
        w = World2(rng, test)
        for s in range(steps):
            g = w.grid()
            obs[t], sem[t], obj[t], ep[t] = render2(g), g, w.objects(), e
            a = int(rng.integers(0, 5))
            act[t], event[t] = a, w.step(a)
            done[t] = s == steps - 1
            t += 1
    return {"obs": obs, "act": act, "sem": sem, "obj": obj, "ep": ep, "done": done, "event": event,
            "light": np.ones(n, np.float32), "names": np.array(NAMES2)}


# ------------------------------------------------------------------------------------------------ world 3, "out of sight"
F0, F1, F2, WALL3, AGENT3, BALL, CAT, BOX3, LIGHT = range(9)
NAMES3 = ["floor a", "floor b", "floor c", "wall", "agent", "ball", "cat", "box", "light"]
KINDS3 = {BALL: "ball", CAT: "cat", BOX3: "box", LIGHT: "light"}
SIZE, PAD = 20, 5                                      # the room, and walls beyond it so the view never leaves the world


def tile3(kind):
    t = np.zeros((T, T, 3), np.uint8)
    base = {F0: (70, 70, 70), F1: (82, 76, 64), F2: (62, 72, 80)}
    if kind in base:
        t[:] = base[kind]
    elif kind == WALL3:
        t[:] = (150, 110, 60)
        t[3, :] = (110, 80, 40)
    else:
        t[:] = (70, 70, 70)
        if kind == AGENT3:
            t[1:6, 1:6] = (60, 160, 230)
            t[2:5, 2:5] = (240, 240, 240)
        elif kind == BALL:
            t[2:5, 1:6] = t[1:6, 2:5] = (230, 70, 60)
        elif kind == CAT:
            t[2:6, 1:6] = (240, 150, 40)
            t[1, 1] = t[1, 5] = (240, 150, 40)
            t[3, 2] = t[3, 4] = (20, 20, 20)
        elif kind == BOX3:
            t[1:6, 1:6] = (200, 150, 50)
            t[1, 1:6] = t[5, 1:6] = t[1:6, 1] = t[1:6, 5] = (120, 80, 20)
        elif kind == LIGHT:
            t[2:5, 1:6] = t[1:6, 2:5] = (255, 230, 80)
            t[3, 3] = (255, 255, 255)
    return t


TILES3 = np.stack([tile3(k) for k in range(9)])


class World3:
    """A 20 x 20 room with internal walls, seen through a 9 x 7 view centred on the agent. A ball rolls one tile per
    step and reverses where it is blocked; cats step to a random free neighbour with probability 0.5; boxes stay;
    lights vanish 10-30 steps after they appear, and a new one appears with probability 0.05 per step. Every object
    blocks the others. The floor has three shades, fixed per world, as Crafter's ground does."""

    def __init__(self, rng):
        self.rng = rng
        n = SIZE + 2 * PAD
        g = rng.integers(0, 3, (n, n))
        g[:PAD + 1, :] = g[-PAD - 1:, :] = g[:, :PAD + 1] = g[:, -PAD - 1:] = WALL3
        for _ in range(int(rng.integers(3, 6))):          # internal walls: segments with one gap
            L = int(rng.integers(4, 10))
            x, y = (int(v) for v in rng.integers(PAD + 2, PAD + SIZE - 2, 2))
            horiz = rng.random() < 0.5
            gap = int(rng.integers(0, L))
            for k in range(L):
                xx, yy = (x + k, y) if horiz else (x, y + k)
                if k != gap and PAD < xx < PAD + SIZE - 1 and PAD < yy < PAD + SIZE - 1:
                    g[yy, xx] = WALL3
        self.g = g
        self.objs = {}                                     # id -> [kind, x, y, extra]
        self.nid = 0
        self.agent = self.free()
        self.add(BALL, [(1, 0), (-1, 0), (0, 1), (0, -1)][rng.integers(4)])
        for _ in range(int(rng.integers(1, 3))):
            self.add(CAT, None)
        for _ in range(int(rng.integers(3, 6))):
            self.add(BOX3, None)
        for _ in range(2):
            self.add(LIGHT, int(rng.integers(10, 31)))

    def occupied(self):
        occ = {(int(self.agent[0]), int(self.agent[1]))} if hasattr(self, "agent") else set()
        occ |= {(o[1], o[2]) for o in self.objs.values()}
        return occ

    def free(self, near=None):
        occ = self.occupied()
        while True:
            x, y = (int(v) for v in self.rng.integers(PAD + 1, PAD + SIZE - 1, 2))
            if self.g[y, x] != WALL3 and (x, y) not in occ:
                return np.array([x, y])

    def add(self, kind, extra):
        x, y = self.free()
        self.objs[self.nid] = [kind, int(x), int(y), extra]
        self.nid += 1

    def open_(self, x, y, occ):
        return self.g[y, x] != WALL3 and (x, y) not in occ

    def step(self, a):
        rng = self.rng
        if a in MOVES:
            dx, dy = MOVES[a]
            x, y = self.agent + (dx, dy)
            if self.open_(x, y, self.occupied()):
                self.agent = np.array([x, y])
        for i in list(self.objs):
            kind, x, y, extra = self.objs[i]
            occ = self.occupied() - {(x, y)}
            if kind == BALL:
                dx, dy = extra
                if not self.open_(x + dx, y + dy, occ):
                    dx, dy = -dx, -dy
                    self.objs[i][3] = (dx, dy)
                if self.open_(x + dx, y + dy, occ):
                    self.objs[i][1:3] = [x + dx, y + dy]
            elif kind == CAT and rng.random() < 0.5:
                opts = [(x + dx, y + dy) for dx, dy in MOVES.values() if self.open_(x + dx, y + dy, occ)]
                if opts:
                    self.objs[i][1:3] = list(opts[rng.integers(len(opts))])
            elif kind == LIGHT:
                self.objs[i][3] -= 1
                if self.objs[i][3] <= 0:
                    del self.objs[i]
        if rng.random() < 0.05:
            self.add(LIGHT, int(rng.integers(10, 31)))

    def view(self):
        """The 7 x 9 kinds around the agent (agent at row 3, column 4)."""
        ax, ay = self.agent
        v = self.g[ay - 3:ay + 4, ax - 4:ax + 5].copy()
        for kind, x, y, _ in self.objs.values():
            if abs(x - ax) <= 4 and abs(y - ay) <= 3:
                v[y - ay + 3, x - ax + 4] = kind
        v[3, 4] = AGENT3
        return v


def play3(n_ep, steps, seed):
    rng = np.random.default_rng(seed)
    n = n_ep * steps
    obs = np.zeros((n, 64, 64, 3), np.uint8)
    act, ep, done = np.zeros(n, np.int8), np.zeros(n, np.int32), np.zeros(n, bool)
    sem, pos = np.zeros((n, H, W), np.int16), np.zeros((n, 2), np.int16)
    rows = []
    t = 0
    for e in range(n_ep):
        w = World3(rng)
        base = e * 1000
        for s in range(steps):
            v = w.view()
            f = np.zeros((64, 64, 3), np.uint8)
            f[:H * T, :W * T] = TILES3[v].transpose(0, 2, 1, 3, 4).reshape(H * T, W * T, 3)
            obs[t], sem[t], pos[t], ep[t] = f, v, w.agent, e
            ax, ay = w.agent
            for i, (kind, x, y, _) in w.objs.items():
                if abs(x - ax) <= 4 and abs(y - ay) <= 3:
                    rows.append((t, base + i, kind, x - ax, y - ay, x, y))
            a = int(rng.integers(0, 5))
            act[t] = a
            w.step(a)
            done[t] = s == steps - 1
            t += 1
    return {"obs": obs, "act": act, "sem": sem, "pos": pos, "ep": ep, "done": done,
            "obj": np.array(rows, np.int32), "light": np.ones(n, np.float32), "sleep": np.zeros(n, bool),
            "names": np.array(NAMES3)}


# ------------------------------------------------------------------------------------------------ world 4b, "no body"
BLUE, RED = 2, 3                                       # world 4's kinds: floor, wall, blue (moved by the actions), red
NAMES4 = ["floor", "wall", "blue", "red"]


def tile4(kind):
    t = tile(FLOOR)
    if kind == WALL:
        return tile(WALL)
    if kind == BLUE:
        t[1:6, 1:6] = (50, 90, 220)
        t[3, 1:6] = (200, 220, 255)
    elif kind == RED:
        t[1:6, 1:6] = (210, 60, 50)
        t[1:6, 3] = (255, 200, 190)
    return t


TILES4 = np.stack([tile4(k) for k in range(4)])


class World4:
    def __init__(self, rng, test=False):
        g = np.full((H, W), FLOOR, np.int64)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
        cells = [(x, y) for y in range(1, H - 1) for x in range(1, W - 1)]
        k = rng.permutation(len(cells))
        nb, nr = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        for i, j in enumerate(k[:nb + nr]):
            x, y = cells[j]
            g[y, x] = BLUE if i < nb else RED
        self.g = g

    def grid(self):
        return self.g.copy()

    def step(self, a):
        if a not in MOVES:
            return False
        dx, dy = MOVES[a]
        ys, xs = np.nonzero(self.g == BLUE)
        order = np.argsort(-(xs * dx + ys * dy))            # the leading blue tiles first
        moved = False
        for y, x in zip(ys[order], xs[order]):
            if self.g[y + dy, x + dx] == FLOOR:
                self.g[y + dy, x + dx], self.g[y, x] = BLUE, FLOOR
                moved = True
        return moved


def render4(grid):
    f = np.zeros((64, 64, 3), np.uint8)
    f[:H * T, :W * T] = TILES4[grid].transpose(0, 2, 1, 3, 4).reshape(H * T, W * T, 3)
    return f


# ------------------------------------------------------------------------------------------------ world 5, "clicks"
HELD_COL = 6                                           # never clicked in training


class World5:
    def __init__(self, rng, test):
        g = np.full((H, W), FLOOR, np.int64)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
        obj = np.zeros((H, W), np.int64)
        keys = [k for k in FREE_KEYS if (k in TEST_FREE) == test]
        nid = 2
        for _ in range(int(rng.integers(2, 4))):
            for _try in range(50):
                k = keys[rng.integers(len(keys))]
                cells = FREE[k][rng.integers(len(FREE[k]))]
                ox, oy = int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1))
                pos = [(ox + x, oy + y) for x, y in cells]
                if all(1 <= x < W - 1 and 1 <= y < H - 1 and g[y, x] == FLOOR for x, y in pos):
                    c0 = int(rng.integers(len(COLOURS)))
                    one = rng.random() < 0.5
                    for x, y in pos:
                        g[y, x] = 8 + (c0 if one else int(rng.integers(len(COLOURS))))
                        obj[y, x] = nid
                    nid += 1
                    break
        free = [(x, y) for y in range(1, H - 1) for x in range(1, W - 1) if g[y, x] == FLOOR]
        for j in rng.permutation(len(free))[:int(rng.integers(1, 3))]:
            x, y = free[j]
            g[y, x] = LAMP if rng.random() < 0.5 else LIT
        self.g, self.obj = g, obj

    def step(self, a):
        """a: 0 wait, else a click at cell a - 1. Returns 'shape', 'lamp' or ''."""
        if a == 0:
            return ""
        y, x = divmod(a - 1, W)
        k = self.g[y, x]
        if k >= 8:
            m = self.obj == self.obj[y, x]
            self.g[m] = 8 + (self.g[m] - 8 + 1) % len(COLOURS)
            return "shape"
        if k in (LAMP, LIT):
            self.g[y, x] = LIT if k == LAMP else LAMP
            return "lamp"
        return ""


# ------------------------------------------------------------------------------------------------ world 6, "searched caves"
FLOOR6, WALL6, AGENT6, DARK, SAND, GEM, RIPE, BARE = range(8)
NAMES6 = ["floor", "wall", "agent", "dark floor", "sand", "gem", "ripe bush", "bare bush"]
GW6, GH6, PAD6, DEPTH6 = 25, 15, 4, 4                  # grid width and height, wall padding, cave depth
CORRIDOR6 = (6, 7, 8)                                  # the corridor's rows; up caves rows 5..2, down caves 9..12


def tile6(kind):
    t = np.zeros((T, T, 3), np.uint8)
    base = {FLOOR6: (70, 70, 70), DARK: (34, 30, 40), SAND: (190, 170, 110)}
    if kind == WALL6:
        t[:] = (150, 110, 60)
        t[3, :] = (110, 80, 40)
    elif kind == AGENT6:
        t[:] = (70, 70, 70)
        t[1:6, 1:6] = (60, 160, 230)
        t[2:5, 2:5] = (240, 240, 240)
    elif kind == GEM:
        t[:] = base[DARK]
        t[2:5, 1:6] = t[1:6, 2:5] = (90, 220, 240)
        t[3, 3] = (240, 255, 255)
    elif kind in (RIPE, BARE):
        t[:] = base[SAND]
        t[1:6, 2:5] = t[2:5, 1:6] = (40, 120, 40)
        if kind == RIPE:
            t[2, 2] = t[4, 4] = t[2, 4] = (220, 30, 60)
    else:
        t[:] = base[kind]
    return t


TILES6 = np.stack([tile6(k) for k in range(8)])


class World6:
    """A corridor 3 tiles high with six dead-end caves, 1 wide and 4 deep (three up, three down), seen through a 9 x 7
    view centred on the agent. From the corridor a cave shows its first one to three tiles; its end only from inside.
    A cave is dark (its end holds a gem with probability 0.5) or sand (its end holds a bush, ripe with probability
    0.7, else bare and ripening 30-60 steps from the start). Walking into a gem takes it, and gems never come back.
    Walking into a ripe bush eats the berry; the bare bush ripens again 30-60 steps later, seen or not. Walls and bushes
    block; the agent stays where it was when it takes or eats."""

    def __init__(self, rng, layout=None):
        self.rng = rng
        self.g = np.full((GH6 + 2 * PAD6, GW6 + 2 * PAD6), WALL6, np.int8)
        for y in CORRIDOR6:
            self.g[PAD6 + y, PAD6 + 1:PAD6 + GW6 - 1] = FLOOR6
        self.caves = []                                # [end x, end y, kind (0 dark, 1 sand), side (-1 up, 1 down)]
        self.ripen = {}                                # end place -> step at which its bare bush ripens
        self.t = 0
        for side in (-1, 1):
            cols = layout["cols"][side] if layout else self.columns()
            for c in cols:
                kind = layout["kind"][(side, c)] if layout else int(rng.random() < 0.5)
                y0 = CORRIDOR6[0] - 1 if side < 0 else CORRIDOR6[-1] + 1
                for d in range(DEPTH6):
                    self.g[PAD6 + y0 + side * d, PAD6 + c] = DARK if kind == 0 else SAND
                ex, ey = c, y0 + side * (DEPTH6 - 1)
                if layout:
                    content = layout["end"][(side, c)]
                elif kind == 0:
                    content = GEM if rng.random() < 0.5 else DARK
                else:
                    content = RIPE if rng.random() < 0.7 else BARE
                self.g[PAD6 + ey, PAD6 + ex] = content
                if content == BARE:
                    self.ripen[(ex, ey)] = int(rng.integers(30, 61))
                self.caves.append([ex, ey, kind, side])
        self.agent = layout["start"] if layout else np.array([int(rng.integers(1, GW6 - 1)), int(rng.choice(CORRIDOR6))])
        self.gems = self.berries = 0

    def columns(self):
        while True:
            c = np.sort(self.rng.choice(np.arange(2, GW6 - 2), 3, replace=False))
            if np.diff(c).min() >= 3:
                return [int(v) for v in c]

    def kind(self, x, y):
        return int(self.g[PAD6 + y, PAD6 + x])

    def walkable(self, x, y):
        return self.kind(x, y) in (FLOOR6, DARK, SAND)

    def step(self, a):
        """One step; returns the event: ("take", end) or ("eat", end) or None."""
        ev = None
        if a in MOVES:
            dx, dy = MOVES[a]
            x, y = int(self.agent[0] + dx), int(self.agent[1] + dy)
            k = self.kind(x, y)
            if k == GEM:
                self.g[PAD6 + y, PAD6 + x] = DARK
                self.gems += 1
                ev = ("take", (x, y))
            elif k == RIPE:
                self.g[PAD6 + y, PAD6 + x] = BARE
                self.berries += 1
                self.ripen[(x, y)] = self.t + 1 + int(self.rng.integers(30, 61))
                ev = ("eat", (x, y))
            elif self.walkable(x, y):
                self.agent = np.array([x, y])
        self.t += 1
        for (x, y), r in list(self.ripen.items()):
            if r <= self.t and self.kind(x, y) == BARE:
                self.g[PAD6 + y, PAD6 + x] = RIPE
                del self.ripen[(x, y)]
        return ev

    def view(self):
        ax, ay = int(self.agent[0]) + PAD6, int(self.agent[1]) + PAD6
        v = self.g[ay - 3:ay + 4, ax - 4:ax + 5].copy()
        v[3, 4] = AGENT6
        return v

    def frame(self):
        """The 9 x 7 view above two rows of carried items (gems in slot 0, berries in slot 1; a bar per item, up to 7)."""
        f = np.zeros((64, 64, 3), np.uint8)
        f[:H * T, :W * T] = TILES6[self.view()].transpose(0, 2, 1, 3, 4).reshape(H * T, W * T, 3)
        for slot, (n, kind) in enumerate(((self.gems, GEM), (self.berries, RIPE))):
            if n:
                f[H * T:H * T + T, slot * T:(slot + 1) * T] = TILES6[kind]
                f[H * T + T:H * T + T + 2, slot * T:slot * T + min(n, 7)] = (255, 255, 255)
        return f

    def path_step(self, goal):
        """The first move of a shortest walk to goal (breadth-first over walkable places), or 0 if there is none."""
        start = (int(self.agent[0]), int(self.agent[1]))
        if start == goal:
            return 0
        prev = {start: None}
        q = [start]
        for cur in q:
            if cur == goal:
                break
            for a, (dx, dy) in MOVES.items():
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt not in prev and (self.walkable(*nxt) or nxt == goal):
                    prev[nxt] = (cur, a)
                    q.append(nxt)
        if goal not in prev:
            return 0
        cur = goal
        while prev[cur][0] != start:
            cur = prev[cur][0]
        return prev[cur][1]


class Visitor6:
    """The play (declared, C3): walk to a random cave's end and into it, back to a random place in the corridor's
    middle row, then another cave; a random action with probability 0.1, a noop with probability 0.05."""

    def __init__(self, rng):
        self.rng, self.plan = rng, None

    def act(self, w):
        r = self.rng.random()
        if r < 0.05:
            return 0
        if r < 0.15:
            return int(self.rng.integers(0, 5))
        here = (int(w.agent[0]), int(w.agent[1]))
        if self.plan is None:                          # ("in", the tile before the end, the end)
            ex, ey, _, side = w.caves[int(self.rng.integers(len(w.caves)))]
            self.plan = ("in", (ex, ey - side), (ex, ey))
        if self.plan[0] == "in":
            if here == self.plan[1]:                   # walk into the end, then back to the corridor
                ex, ey = self.plan[2]
                self.plan = ("out", (int(self.rng.integers(1, GW6 - 1)), CORRIDOR6[1]), None)
                return next(a for a, d in MOVES.items() if d == (ex - here[0], ey - here[1]))
            return w.path_step(self.plan[1]) or int(self.rng.integers(1, 5))
        if here == self.plan[1]:
            self.plan = None
            return 0
        return w.path_step(self.plan[1]) or int(self.rng.integers(1, 5))


def play6(n_ep, steps, seed):
    """Play world 6 with Visitor6. The evaluator keeps the whole grid each step, the caves and the events."""
    rng = np.random.default_rng(seed)
    n = n_ep * steps
    obs = np.zeros((n, 64, 64, 3), np.uint8)
    act, ep, done = np.zeros(n, np.int8), np.zeros(n, np.int32), np.zeros(n, bool)
    sem, pos = np.zeros((n, H, W), np.int8), np.zeros((n, 2), np.int16)
    grid = np.zeros((n, GH6, GW6), np.int8)
    caves = np.zeros((n_ep, 6, 4), np.int16)
    event = np.full((n, 3), -1, np.int16)              # kind (0 take, 1 eat), end x, end y
    t = 0
    for e in range(n_ep):
        w, pol = World6(rng), Visitor6(rng)
        caves[e] = w.caves
        for s in range(steps):
            obs[t], sem[t], pos[t], ep[t] = w.frame(), w.view(), w.agent, e
            grid[t] = w.g[PAD6:PAD6 + GH6, PAD6:PAD6 + GW6]
            a = pol.act(w)
            act[t] = a
            ev = w.step(a)
            if ev:
                event[t] = (0 if ev[0] == "take" else 1, *ev[1])
            done[t] = s == steps - 1
            t += 1
    return {"obs": obs, "act": act, "sem": sem, "pos": pos, "ep": ep, "done": done, "grid": grid, "caves": caves,
            "event": event, "light": np.ones(n, np.float32), "sleep": np.zeros(n, bool), "names": np.array(NAMES6)}


if __name__ == "__main__":
    if sys.argv[1:] == ["6"]:
        for split, seed, n_ep in (("train", 1091, 500), ("test", 1092, 200)):
            z = play6(n_ep, 300, seed)
            np.savez_compressed(OUT / f"world6_{split}.npz", **z)
            ev = z["event"]
            print(split, "steps", len(z["act"]), "gems taken", int((ev[:, 0] == 0).sum()), "berries eaten",
                  int((ev[:, 0] == 1).sum()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["5"]:
        for split, seed, test in (("train", 1061, False), ("test", 1062, True)):
            rng = np.random.default_rng(seed)
            n_ep, steps = (2000 if split == "train" else 500), 100
            n = n_ep * steps
            obs, act = np.zeros((n, 64, 64, 3), np.uint8), np.zeros(n, np.int16)
            sem, obj = np.zeros((n, H, W), np.int16), np.zeros((n, H, W), np.int16)
            ep, done, event = np.zeros(n, np.int32), np.zeros(n, bool), np.zeros(n, "U5")
            cells = [c for c in range(H * W) if test or c % W != HELD_COL]
            t = 0
            for e in range(n_ep):
                w = World5(rng, test)
                for s_ in range(steps):
                    obs[t], sem[t], obj[t], ep[t] = render2(w.g), w.g, w.obj, e
                    a = 0 if rng.random() < 0.1 else 1 + int(cells[rng.integers(len(cells))])
                    act[t] = a
                    event[t] = w.step(a)
                    done[t] = s_ == steps - 1
                    t += 1
            np.savez_compressed(OUT / f"world5_{split}.npz", obs=obs, act=act, sem=sem, obj=obj, ep=ep, done=done,
                                event=event, light=np.ones(n, np.float32), names=np.array(NAMES2))
            print(split, "steps", n, "shape recolours", int((event == "shape").sum()), "lamp toggles",
                  int((event == "lamp").sum()), "clicks in column 6", int(((act - 1) % W == HELD_COL)[act > 0].sum()),
                  flush=True)
    if sys.argv[1:] == ["4"]:
        for split, seed in (("train", 1051), ("test", 1052)):
            rng = np.random.default_rng(seed)
            n_ep, steps = (2000 if split == "train" else 500), 100
            n = n_ep * steps
            obs, act = np.zeros((n, 64, 64, 3), np.uint8), np.zeros(n, np.int8)
            sem, ep, done = np.zeros((n, H, W), np.int16), np.zeros(n, np.int32), np.zeros(n, bool)
            t = 0
            for e in range(n_ep):
                w = World4(rng)
                for s_ in range(steps):
                    g = w.grid()
                    obs[t], sem[t], ep[t] = render4(g), g, e
                    act[t] = int(rng.integers(0, 5))
                    w.step(int(act[t]))
                    done[t] = s_ == steps - 1
                    t += 1
            np.savez_compressed(OUT / f"world4_{split}.npz", obs=obs, act=act, sem=sem, ep=ep, done=done,
                                light=np.ones(n, np.float32), names=np.array(NAMES4))
            moved = ((sem[1:] != sem[:-1]).any((1, 2)) & (ep[1:] == ep[:-1]))
            print(split, "steps", n, "steps where blue tiles moved", int(moved.sum()), flush=True)
    if sys.argv[1:] == ["3"]:
        for split, seed in (("train", 1041), ("test", 1042)):
            z = play3(500, 300, seed)
            np.savez_compressed(OUT / f"world3_{split}.npz", **z)
            o = z["obj"]
            print(split, "steps", len(z["act"]), "object sightings by kind",
                  {KINDS3[k]: int((o[:, 2] == k).sum()) for k in KINDS3}, flush=True)
    if sys.argv[1:] == ["2"]:
        for split, seed, test in (("train", 1031, False), ("test", 1032, True)):
            z = play2(2000 if split == "train" else 500, 100, seed, test)
            np.savez_compressed(OUT / f"world2_{split}.npz", **z)
            ev = z["event"]
            sizes = [int((o == k).sum()) for o in z["obj"][::100] for k in np.unique(o) if 2 <= k < 99]
            print(split, "steps", len(ev), "pushes", int((ev == "push").sum()), "door changes", int((ev == "door").sum()),
                  "shapes per episode", round(len(sizes) / (len(ev) / 100), 2), "sizes", np.bincount(sizes).tolist(),
                  flush=True)
    if sys.argv[1:] == ["1"]:
        for split, seed, test in (("train", 1021, False), ("test", 1022, True)):
            z = play(World1, 3000 if split == "train" else 1000, 100, seed, test)
            np.savez_compressed(OUT / f"world1_{split}.npz", **z)
            sem = z["sem"].reshape(len(z["act"]), -1)
            same = (z["ep"][1:] == z["ep"][:-1])
            lit_on = ((sem[1:] == LIT) & (sem[:-1] == LAMP)).any(1) & same
            lit_off = ((sem[1:] == LAMP) & (sem[:-1] == LIT)).any(1) & same
            print(split, "steps", len(z["act"]), "pushes", int(z["push"].sum()), "lamps lit / put out",
                  int(lit_on.sum()), int(lit_off.sum()), "steps with a lit lamp", int((sem == LIT).any(1).sum()),
                  flush=True)
