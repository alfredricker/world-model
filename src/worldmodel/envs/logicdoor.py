"""Logic-door rooms for card 010.

Card 003's room with six actions (left, right, forward, pickup, drop,
toggle), a switch (toggles on and off), a vase (a toggle breaks it for
good; nothing depends on it), the matching key and a distractor key. The
door checks its requirement every time it is opened:

- "key":    holding the matching key
- "switch": the switch is on
- "either": matching key or switch on
- "both":   matching key and switch on

State: (x, y, dir, carry, key, distractor, door_open, switch_on, vase_broken)
with carry 0 none / 1 matching key / 2 distractor, and key / distractor the
object's cell or None while carried. Evaluator machinery only (C1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .keydoor import COLOURS, DIR_VEC

LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = range(6)
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle")
RULES = ("key", "switch", "either", "both")


@dataclass(frozen=True)
class Layout:
    size: int
    wall_x: int
    door_y: int
    door_colour: str
    distractor_colour: str
    key: tuple[int, int]
    distractor: tuple[int, int]
    switch: tuple[int, int]
    vase: tuple[int, int]
    goal: tuple[int, int]
    start: tuple[int, int, int]
    rule: str

    @property
    def door(self):
        return (self.wall_x, self.door_y)


def start_state(layout: Layout, carry: int = 0, switch_on: int = 0) -> tuple:
    key = None if carry == 1 else layout.key
    distractor = None if carry == 2 else layout.distractor
    return (*layout.start, carry, key, distractor, 0, switch_on, 0)


def cell(layout: Layout, s: tuple, pos) -> str:
    x, y = pos
    n = layout.size
    if x <= 0 or y <= 0 or x >= n - 1 or y >= n - 1:
        return "wall"
    if x == layout.wall_x:
        return "door" if y == layout.door_y else "wall"
    if pos == s[4]:
        return "key"
    if pos == s[5]:
        return "distractor"
    if pos == layout.switch:
        return "switch"
    if pos == layout.vase and not s[8]:
        return "vase"
    if pos == layout.goal:
        return "goal"
    return "empty"


def front(s: tuple):
    dx, dy = DIR_VEC[s[2]]
    return (s[0] + dx, s[1] + dy)


def door_requirement(layout: Layout, s: tuple) -> bool:
    key, sw = s[3] == 1, bool(s[7])
    return {"key": key, "switch": sw, "either": key or sw, "both": key and sw}[layout.rule]


def step(layout: Layout, s: tuple, a: int) -> tuple[tuple, bool]:
    x, y, d, carry, key, dist, door, sw, vase = s
    if a == LEFT:
        return (x, y, (d - 1) % 4, *s[3:]), False
    if a == RIGHT:
        return (x, y, (d + 1) % 4, *s[3:]), False
    f = front(s)
    here = cell(layout, s, f)
    if a == FORWARD:
        if here in ("empty", "goal") or (here == "door" and door):
            return (f[0], f[1], d, *s[3:]), here == "goal"
        return s, False
    if a == PICKUP:
        if carry == 0 and here == "key":
            return (x, y, d, 1, None, dist, door, sw, vase), False
        if carry == 0 and here == "distractor":
            return (x, y, d, 2, key, None, door, sw, vase), False
        return s, False
    if a == DROP:
        if carry and here == "empty" and f != layout.goal:
            if carry == 1:
                return (x, y, d, 0, f, dist, door, sw, vase), False
            return (x, y, d, 0, key, f, door, sw, vase), False
        return s, False
    if a == TOGGLE:
        if here == "door":
            if door:
                return (x, y, d, carry, key, dist, 0, sw, vase), False
            if door_requirement(layout, s):
                return (x, y, d, carry, key, dist, 1, sw, vase), False
            return s, False
        if here == "switch":
            return (x, y, d, carry, key, dist, door, 1 - sw, vase), False
        if here == "vase":
            return (x, y, d, carry, key, dist, door, sw, 1), False
    return s, False


def _walk(layout, frm, blocked):
    seen, todo = {frm}, [frm]
    while todo:
        x, y = todo.pop()
        for dx, dy in DIR_VEC:
            p = (x + dx, y + dy)
            if p in seen or p in blocked:
                continue
            if p[0] <= 0 or p[1] <= 0 or p[0] >= layout.size - 1 or p[1] >= layout.size - 1:
                continue
            if p[0] == layout.wall_x and p[1] != layout.door_y:
                continue
            seen.add(p)
            todo.append(p)
    return seen


def make_layout(size: int, rule: str, rng: np.random.Generator) -> Layout:
    """Random layout; the key, switch and door front reachable with the
    objects as obstacles, and nothing placed in front of the door."""
    while True:
        wall_x = int(rng.integers(3, size - 2))
        door_y = int(rng.integers(1, size - 1))
        colours = rng.permutation(len(COLOURS))
        left = [(x, y) for x in range(1, wall_x) for y in range(1, size - 1) if (x, y) != (wall_x - 1, door_y)]
        right = [(x, y) for x in range(wall_x + 1, size - 1) for y in range(1, size - 1)]
        if len(left) < 5:
            continue
        key, dist, switch, vase, agent = [left[i] for i in rng.permutation(len(left))[:5]]
        goal = right[int(rng.integers(len(right)))]
        lay = Layout(size, wall_x, door_y, COLOURS[colours[0]], COLOURS[colours[1]], key, dist, switch, vase,
                     goal, (*agent, int(rng.integers(4))), rule)
        walk = _walk(lay, agent, {key, dist, switch, vase})
        near = lambda p: any((p[0] + dx, p[1] + dy) in walk for dx, dy in DIR_VEC)
        if near(key) and near(switch) and (wall_x - 1, door_y) in walk:
            return lay


def variables(layout: Layout, s: tuple) -> dict:
    x, y, d, carry, key, dist, door, sw, vase = s
    f = cell(layout, s, front(s))
    colour = {"door": layout.door_colour, "key": layout.door_colour,
              "distractor": layout.distractor_colour}.get(f, "none")
    return {
        "front": {"distractor": "key"}.get(f, f),
        "front_colour": colour,
        "front_matches_door": f in ("key", "distractor") and colour == layout.door_colour,
        "carrying": "none" if carry == 0 else "key",
        "carrying_colour": ("none", layout.door_colour, layout.distractor_colour)[carry],
        "carrying_matches_door": carry == 1,
        "door": "open" if door else "closed",
        "switch": "on" if sw else "off",
        "vase": "broken" if vase else "intact",
        "side": "left" if x < layout.wall_x else ("doorway" if x == layout.wall_x else "right"),
        "dir": d,
        "x": x,
        "y": y,
    }


GOALS = {
    "door_open": lambda lay, s: bool(s[6]),
    "holding_matching_key": lambda lay, s: s[3] == 1,
    "switch_on": lambda lay, s: bool(s[7]),
    "on_goal": lambda lay, s: (s[0], s[1]) == lay.goal,
}


def collect(rule: str, episodes: int, seed: int, size: int = 8, max_steps: int = 640,
            play_starts: float = 0.0) -> list[dict]:
    """Uniform random play. play_starts: share of episodes that begin
    holding the matching key, with the switch on, or both (declared)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(episodes):
        lay = make_layout(size, rule, rng)
        carry, sw = 0, 0
        if rng.random() < play_starts:
            carry, sw = [(1, 0), (0, 1), (1, 1)][int(rng.integers(3))]
        s = start_state(lay, carry, sw)
        states, actions = [s], []
        for a in rng.integers(6, size=max_steps):
            s, done = step(lay, s, int(a))
            states.append(s)
            actions.append(int(a))
            if done:
                break
        out.append({"layout": lay, "states": states, "actions": actions})
    return out
