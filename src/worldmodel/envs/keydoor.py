"""Key-door room: the small world for card 003.

One room split by a wall with a locked door, the key of the door's colour, a
distractor key of another colour, and a goal square behind the door. The
agent has the chained-rooms action set (0 left, 1 right, 2 forward, 3 pickup,
4 toggle; no drop), so picking up the distractor key makes the door
unreachable.

The dynamics are written twice: as a MiniGrid environment (frames, for
learners later) and as a small symbolic step function over
(x, y, dir, carry, door) for exact computation. A test checks the two agree.
Everything symbolic here is evaluator machinery (C1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

COLOURS = ("red", "green", "blue")
DIR_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))    # MiniGrid: right, down, left, up
CARRY = ("none", "matching", "distractor")
DOOR = ("locked", "closed", "open")
LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = range(5)
ACTION_MAP = (0, 1, 2, 3, 5)   # opaque ID -> MiniGrid left, right, forward, pickup, toggle


@dataclass(frozen=True)
class Layout:
    size: int                   # grid side, outer walls included
    wall_x: int                 # column of the dividing wall
    door_y: int                 # row of the door in that wall
    door_colour: str
    distractor_colour: str
    key: tuple[int, int]        # matching key
    distractor: tuple[int, int]
    goal: tuple[int, int]
    start: tuple[int, int, int]  # x, y, dir

    @property
    def door(self) -> tuple[int, int]:
        return (self.wall_x, self.door_y)


# State: (x, y, dir, carry, door) with carry in CARRY indices, door in DOOR indices.
def start_state(layout: Layout) -> tuple:
    return (*layout.start, 0, 0)


def cell(layout: Layout, state: tuple, pos: tuple[int, int]) -> str:
    """What occupies pos: wall, door, key, distractor, goal or empty."""
    x, y = pos
    s = layout.size
    if x <= 0 or y <= 0 or x >= s - 1 or y >= s - 1:
        return "wall"
    if x == layout.wall_x:
        return "door" if y == layout.door_y else "wall"
    if pos == layout.key and state[3] != 1:
        return "key"
    if pos == layout.distractor and state[3] != 2:
        return "distractor"
    if pos == layout.goal:
        return "goal"
    return "empty"


def front(state: tuple) -> tuple[int, int]:
    dx, dy = DIR_VEC[state[2]]
    return (state[0] + dx, state[1] + dy)


def step(layout: Layout, state: tuple, action: int) -> tuple[tuple, bool]:
    """Next state and whether the episode ends (the goal square is entered)."""
    x, y, d, carry, door = state
    if action == LEFT:
        return (x, y, (d - 1) % 4, carry, door), False
    if action == RIGHT:
        return (x, y, (d + 1) % 4, carry, door), False
    f = front(state)
    here = cell(layout, state, f)
    if action == FORWARD:
        if here in ("empty", "goal") or (here == "door" and door == 2):
            return (f[0], f[1], d, carry, door), here == "goal"
        return state, False
    if action == PICKUP:
        if carry == 0 and here in ("key", "distractor"):
            return (x, y, d, 1 if here == "key" else 2, door), False
        return state, False
    if action == TOGGLE and here == "door":
        if door == 0:
            return ((x, y, d, carry, 2) if carry == 1 else state), False
        return (x, y, d, carry, 3 - door), False     # closed <-> open
    return state, False


def _reachable(layout: Layout, frm: tuple[int, int], blocked: set) -> set:
    """Cells reachable by walking, treating the door as passable."""
    seen, todo = {frm}, [frm]
    while todo:
        x, y = todo.pop()
        for dx, dy in DIR_VEC:
            p = (x + dx, y + dy)
            if p in seen or p in blocked:
                continue
            c = cell(layout, (0, 0, 0, 0, 2), p)
            if c == "wall":
                continue
            seen.add(p)
            todo.append(p)
    return seen


def make_layout(size: int, rng: np.random.Generator) -> Layout:
    """Random layout whose matching key and door can both be reached."""
    while True:
        wall_x = int(rng.integers(3, size - 2))       # left room at least 2 wide
        door_y = int(rng.integers(1, size - 1))
        colours = rng.permutation(len(COLOURS))
        left = [(x, y) for x in range(1, wall_x) for y in range(1, size - 1)]
        right = [(x, y) for x in range(wall_x + 1, size - 1) for y in range(1, size - 1)]
        spots = [left[i] for i in rng.permutation(len(left))[:3]]
        key, distractor, agent = spots
        goal = right[int(rng.integers(len(right)))]
        layout = Layout(size, wall_x, door_y, COLOURS[colours[0]], COLOURS[colours[1]],
                        key, distractor, goal, (*agent, int(rng.integers(4))))
        # Solvable without touching the distractor: the agent reaches a cell
        # next to the key and the cell in front of the door with the
        # distractor treated as a wall, and the goal is behind the door.
        walk = _reachable(layout, agent, {key, distractor})
        near_key = any((key[0] + dx, key[1] + dy) in walk for dx, dy in DIR_VEC)
        if near_key and (wall_x - 1, door_y) in walk and goal in _reachable(layout, (wall_x, door_y), {key, distractor}):
            return layout


# ---- variables for the condition analysis (evaluator vocabulary) ----

def variables(layout: Layout, state: tuple) -> dict:
    """The candidate variables of a state, as name -> value."""
    x, y, d, carry, door = state
    f = cell(layout, state, front(state))
    colour = {"door": layout.door_colour, "key": layout.door_colour,
              "distractor": layout.distractor_colour}.get(f, "none")
    carried_colour = ("none", layout.door_colour, layout.distractor_colour)[carry]
    return {
        "front": {"distractor": "key"}.get(f, f),
        "front_colour": colour,
        "front_matches_door": f in ("key", "distractor") and colour == layout.door_colour,
        "carrying": "none" if carry == 0 else "key",
        "carrying_colour": carried_colour,
        "carrying_matches_door": carry == 1,
        "door": DOOR[door],
        "side": "left" if x < layout.wall_x else ("doorway" if x == layout.wall_x else "right"),
        "dir": d,
        "x": x,
        "y": y,
    }


# ---- MiniGrid version (frames) ----

class KeyDoorEnv(MiniGridEnv):
    def __init__(self, layout: Layout, max_steps: int = 256, **kwargs):
        self.layout = layout
        super().__init__(mission_space=MissionSpace(mission_func=lambda: "reach the goal"),
                         grid_size=layout.size, max_steps=max_steps, see_through_walls=False, **kwargs)

    def _gen_grid(self, width, height):
        lay = self.layout
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(lay.wall_x, 0)
        self.grid.set(*lay.door, Door(lay.door_colour, is_locked=True))
        self.grid.set(*lay.key, Key(lay.door_colour))
        self.grid.set(*lay.distractor, Key(lay.distractor_colour))
        self.grid.set(*lay.goal, Goal())
        self.agent_pos = lay.start[:2]
        self.agent_dir = lay.start[2]

    def step(self, action):
        """Takes this world's opaque action IDs (4 is toggle)."""
        return super().step(ACTION_MAP[action])


def env_state(env: KeyDoorEnv) -> tuple:
    """The symbolic state of a running MiniGrid KeyDoorEnv."""
    lay = env.layout
    door = env.grid.get(*lay.door)
    carry = 0
    if env.carrying is not None:
        carry = 1 if env.carrying.color == lay.door_colour else 2
    return (int(env.agent_pos[0]), int(env.agent_pos[1]), int(env.agent_dir), carry,
            0 if door.is_locked else (2 if door.is_open else 1))
