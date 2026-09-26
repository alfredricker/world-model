"""Chained rooms: an RGB-only MiniGrid family for temporally extended abstraction.

Rooms in a row, separated by doors of four mechanisms: key-locked (colour must
match a key), switch (a remote switch must have been pressed; the door looks
the same either way), plain, and timed (closes again after a delay). Keys may
hide in boxes and are consumed on use. Lava ends the episode. Everything below
the port is evaluator machinery; the learner receives copied RGB, opaque action
IDs, reward and episode flags only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, COLORS, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Door, Goal, Key, Lava, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_circle, point_in_rect

SWITCH_COLOR = "grey"
DOOR_TYPES = ("locked", "switch", "plain", "timed")
SPLITS = ("train", "development", "transfer_combo", "transfer_colour")
EVENTS = ("pickup", "matching_pickup", "unlock", "door_open", "switch_press",
          "box_open", "lava", "goal", "crossing", "locked_attempt", "switch_attempt")
FRONT_TYPES = ("none", "wall", "door", "key", "box", "switch", "lava", "goal")


@dataclass(frozen=True)
class RoomsConfig:
    split: str = "train"
    rooms: tuple[int, ...] = (3, 4)
    room_size: int = 6
    max_steps: int = 512
    tile_size: int = 6
    agent_view: int = 7
    timed_delay: int = 15
    consume_keys: bool = True
    lava_per_room: tuple[int, ...] = (0, 0, 1)
    play_probability: float = 0.0   # diverse-configuration starts; declared curriculum

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split {self.split}")
        if min(self.rooms) < 2 or self.room_size < 4 or self.max_steps < 1 or self.tile_size < 1:
            raise ValueError("Invalid world geometry or budget")
        if self.agent_view % 2 == 0 or self.agent_view < 3:
            raise ValueError("Agent view must be odd and at least three")
        if not 0 <= self.play_probability <= 1:
            raise ValueError("Play probability must be in [0, 1]")

    @property
    def colours(self) -> list[str]:
        # Grey is reserved for switch doors and switches. Two colours are held
        # out entirely for the transfer_colour split.
        if self.split == "transfer_colour":
            return ["purple", "yellow"]
        return ["red", "green", "blue"]


class Switch(WorldObj):
    """A wall-mounted switch. Toggling it opens its remote door for good."""

    def __init__(self):
        super().__init__("ball", SWITCH_COLOR)
        self.on = False
        self.door: SwitchDoor | None = None

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def toggle(self, env, pos):
        if not self.on:
            self.on = True
            if self.door is not None:
                self.door.enabled = True
        return True

    def encode(self):
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], int(self.on))

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.15, 0.85, 0.15, 0.85), c)
        inner = (0, 0, 0) if not self.on else (255, 255, 255)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.22), inner)


class SwitchDoor(Door):
    """Looks like a locked grey door whether or not its switch has been pressed."""

    def __init__(self):
        super().__init__(SWITCH_COLOR, is_open=False, is_locked=True)
        self.enabled = False

    def toggle(self, env, pos):
        if self.is_open:
            self.is_open = False
            return True
        if self.enabled:
            self.is_open = True
            return True
        return False

    def encode(self):
        # Rendering must not reveal the switch state.
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 2 if not self.is_open else 0)


class TimedDoor(Door):
    """A plain door that closes again a fixed number of steps after opening."""

    def __init__(self, color: str, delay: int):
        super().__init__(color, is_open=False, is_locked=False)
        self.delay = delay
        self.opened_at: int | None = None


@dataclass
class DoorRecord:
    index: int
    kind: str
    color: str
    pos: tuple[int, int]
    obj: Door
    key_pos: tuple[int, int] | None = None
    key_room: int | None = None
    key_boxed: bool = False
    switch_pos: tuple[int, int] | None = None
    switch_room: int | None = None


class ChainedRoomsEnv(MiniGridEnv):
    def __init__(self, config: RoomsConfig):
        self.config = config
        self.room_count = max(config.rooms)
        width = self.room_count * (config.room_size + 1) + 1
        height = config.room_size + 2
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "reach the last room"),
            width=width, height=height, max_steps=config.max_steps,
            agent_view_size=config.agent_view, see_through_walls=False,
        )
        self.doors: list[DoorRecord] = []
        self.switches: list[Switch] = []
        self.boxes: list[tuple[tuple[int, int], Box]] = []
        self.lava_cells: list[tuple[int, int]] = []
        self.play_start = False

    def reset(self, *, seed=None, options=None):
        # MiniGridEnv.reset clears ``carrying`` after ``_gen_grid``; restore the
        # play-start key placed there.
        self._play_carrying = None
        result = super().reset(seed=seed, options=options)
        if self._play_carrying is not None:
            self.carrying = self._play_carrying
            self._play_carrying = None
            result = self.gen_obs(), result[1]
        return result

    # -- world generation ---------------------------------------------------

    def room_of(self, x: int) -> int:
        return min(max((x - 1) // (self.config.room_size + 1), 0), self.rooms - 1)

    def _room_cells(self, room: int):
        s = self.config.room_size
        x0 = 1 + room * (s + 1)
        return [(x, y) for x in range(x0, x0 + s) for y in range(1, s + 1)]

    def _free_cell(self, room: int, avoid: set[tuple[int, int]]):
        cells = [c for c in self._room_cells(room) if self.grid.get(*c) is None and c not in avoid]
        if not cells:
            raise RuntimeError("No free cell in room")
        return cells[self._rand_int(0, len(cells))]

    def _gen_grid(self, width: int, height: int) -> None:
        c = self.config
        self.rooms = c.rooms[self._rand_int(0, len(c.rooms))]
        s = c.room_size
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        used_width = self.rooms * (s + 1) + 1
        for x in range(used_width, width):
            for y in range(height):
                self.grid.set(x, y, Wall())
        for room in range(1, self.rooms):
            x = room * (s + 1)
            for y in range(1, height - 1):
                self.grid.set(x, y, Wall())
        self.doors, self.switches, self.boxes, self.lava_cells = [], [], [], []
        colours = c.colours
        kinds = self._door_kinds()
        avoid: set[tuple[int, int]] = set()
        for i, kind in enumerate(kinds):
            x = (i + 1) * (s + 1)
            y = self._rand_int(1, s + 1)
            pos = (x, y)
            avoid |= {(x - 1, y), (x + 1, y)}
            if kind == "locked":
                colour = self._door_colour(i, colours)
                door = Door(colour, is_locked=True)
                record = DoorRecord(i, kind, colour, pos, door)
                key_room = self._rand_int(0, i + 1)
                key_pos = self._free_cell(key_room, avoid)
                boxed = self._rand_float(0, 1) < 0.5
                key = Key(colour)
                self.grid.set(*key_pos, Box("grey", contains=key) if boxed else key)
                if boxed:
                    self.boxes.append((key_pos, self.grid.get(*key_pos)))
                record.key_pos, record.key_room, record.key_boxed = key_pos, key_room, boxed
            elif kind == "switch":
                door = SwitchDoor()
                record = DoorRecord(i, kind, SWITCH_COLOR, pos, door)
                switch_room = self._rand_int(0, i + 1)
                switch_pos = self._free_cell(switch_room, avoid)
                switch = Switch()
                switch.door = door
                self.grid.set(*switch_pos, switch)
                self.switches.append(switch)
                record.switch_pos, record.switch_room = switch_pos, switch_room
            elif kind == "timed":
                colour = colours[self._rand_int(0, len(colours))]
                door = TimedDoor(colour, c.timed_delay)
                record = DoorRecord(i, kind, colour, pos, door)
            else:
                colour = colours[self._rand_int(0, len(colours))]
                door = Door(colour, is_locked=False)
                record = DoorRecord(i, kind, colour, pos, door)
            self.grid.set(x, y, door)
            self.doors.append(record)
        # Distractor keys, possibly boxed.
        for _ in range(self._rand_int(1, 3)):
            room = self._rand_int(0, self.rooms)
            colour = colours[self._rand_int(0, len(colours))]
            pos = self._free_cell(room, avoid)
            key = Key(colour)
            if self._rand_float(0, 1) < 0.4:
                box = Box("grey", contains=key)
                self.grid.set(*pos, box)
                self.boxes.append((pos, box))
            else:
                self.grid.set(*pos, key)
        # Lava, never adjacent to a door.
        for room in range(self.rooms):
            count = c.lava_per_room[self._rand_int(0, len(c.lava_per_room))]
            for _ in range(count):
                pos = self._free_cell(room, avoid)
                self.grid.set(*pos, Lava())
                self.lava_cells.append(pos)
        goal = self._free_cell(self.rooms - 1, avoid)
        self.grid.set(*goal, Goal())
        self.goal_pos = goal
        # Agent start: ordinary (first room) or play (declared curriculum).
        self.play_start = self._rand_float(0, 1) < c.play_probability
        start_room = self._rand_int(0, self.rooms) if self.play_start else 0
        hold_matching = False
        if self.play_start and self._rand_float(0, 1) < 0.5:
            # Declared curriculum: start in the room before a locked door, holding its key.
            locked = [r for r in self.doors if r.kind == "locked"]
            if locked:
                start_room = locked[self._rand_int(0, len(locked))].index
                hold_matching = True
        self.agent_pos = self._free_cell(start_room, avoid)
        self.agent_dir = self._rand_int(0, 4)
        if self.play_start:
            self._play_configuration(start_room, colours, hold_matching)
        self.mission = "reach the last room"

    def _door_kinds(self) -> list[str]:
        count = self.rooms - 1
        kinds = [DOOR_TYPES[self._rand_int(0, 4)] for _ in range(count)]
        if "locked" not in kinds:
            kinds[self._rand_int(0, count)] = "locked"
        if count >= 2 and "switch" not in kinds:
            free = [i for i, k in enumerate(kinds) if k != "locked"] or [0]
            kinds[free[self._rand_int(0, len(free))]] = "switch"
        return kinds

    def _door_colour(self, index: int, colours: list[str]) -> str:
        split = self.config.split
        if split == "transfer_colour":
            return colours[self._rand_int(0, len(colours))]
        held = split == "transfer_combo"
        options = [k for k in range(len(colours)) if ((k + index) % 2 == 1) == held]
        return colours[options[self._rand_int(0, len(options))]]

    def _play_configuration(self, start_room: int, colours: list[str], hold_matching: bool = False) -> None:
        for record in self.doors:
            if record.index < start_room or self._rand_float(0, 1) < 0.3:
                if self._rand_float(0, 1) < 0.7:
                    door = record.obj
                    if record.kind == "locked":
                        door.is_locked = False
                        door.is_open = True
                        if self.config.consume_keys and record.key_pos is not None:
                            self.grid.set(*record.key_pos, None)
                    elif record.kind == "switch":
                        switch = self.grid.get(*record.switch_pos)
                        switch.on, door.enabled, door.is_open = True, True, True
                    else:
                        door.is_open = True
                        if isinstance(door, TimedDoor):
                            door.opened_at = 0
        for pos, box in self.boxes:
            if self.grid.get(*pos) is box and self._rand_float(0, 1) < 0.5:
                self.grid.set(*pos, box.contains)
        if hold_matching or self._rand_float(0, 1) < 0.5:
            keys = [(x, y) for x in range(self.width) for y in range(self.height)
                    if isinstance(self.grid.get(x, y), Key)]
            ahead = [r for r in self.doors if r.kind == "locked" and r.obj.is_locked and r.index >= start_room]
            if ahead and (hold_matching or self._rand_float(0, 1) < 0.5):
                # Declared curriculum: hold the key matching the next locked door.
                record = ahead[0]
                self._play_carrying = Key(record.color)
                self._play_carrying.cur_pos = np.array([-1, -1])
                if record.key_pos is not None and isinstance(self.grid.get(*record.key_pos), Key):
                    self.grid.set(*record.key_pos, None)
            elif keys:
                pos = keys[self._rand_int(0, len(keys))]
                self._play_carrying = self.grid.get(*pos)
                self._play_carrying.cur_pos = np.array([-1, -1])
                self.grid.set(*pos, None)

    # -- stepping with evaluator labels ---------------------------------------

    def step(self, action):
        before_cell = self.grid.get(*self.front_pos)
        before_carrying = self.carrying
        before_room = self.room_of(self.agent_pos[0])
        before_locked = {r.index: (r.obj.is_locked, r.obj.is_open) for r in self.doors}
        before_switch = [s.on for s in self.switches]
        before_boxes = [self.grid.get(*pos) is box for pos, box in self.boxes]
        events = dict.fromkeys(EVENTS, False)
        if action == self.actions.toggle and isinstance(before_cell, Door):
            if before_cell.is_locked and not isinstance(before_cell, SwitchDoor):
                events["locked_attempt"] = True
            if isinstance(before_cell, SwitchDoor) and not before_cell.is_open:
                events["switch_attempt"] = True
        if action == self.actions.pickup and self.carrying is not None and isinstance(before_cell, Key):
            # Swap: the carried key goes where the picked key was.
            swapped = self.carrying
            self.carrying = before_cell
            self.carrying.cur_pos = np.array([-1, -1])
            self.grid.set(*self.front_pos, swapped)
            swapped.cur_pos = np.array(self.front_pos)
            events["pickup"] = True
        obs, reward, terminated, truncated, info = super().step(action)
        for record in self.doors:
            door = record.obj
            was_locked, was_open = before_locked[record.index]
            if was_locked and not door.is_locked:
                events["unlock"] = True
                if self.config.consume_keys and self.carrying is before_carrying and before_carrying is not None:
                    self.carrying = None
            if not was_open and door.is_open:
                events["door_open"] = True
                if isinstance(door, TimedDoor):
                    door.opened_at = self.step_count
            if isinstance(door, TimedDoor) and door.is_open and door.opened_at is not None:
                if self.step_count - door.opened_at >= door.delay and tuple(self.agent_pos) != record.pos:
                    door.is_open = False
                    door.opened_at = None
        if self.carrying is not None and self.carrying is not before_carrying:
            events["pickup"] = True
            events["matching_pickup"] = any(
                r.kind == "locked" and r.obj.is_locked and r.color == self.carrying.color for r in self.doors)
        events["switch_press"] = any(not b and s.on for b, s in zip(before_switch, self.switches, strict=True))
        events["box_open"] = any(b and self.grid.get(*pos) is not box
                                 for b, (pos, box) in zip(before_boxes, self.boxes, strict=True))
        room = self.room_of(self.agent_pos[0])
        events["crossing"] = room != before_room
        events["goal"] = bool(terminated and reward > 0)
        events["lava"] = bool(terminated and reward == 0)
        info = {"events": events, "labels": self.labels(), "obs_grid": obs}
        return obs, reward, terminated, truncated, info

    def labels(self) -> dict:
        front = self.grid.get(*self.front_pos)
        front_type = "none"
        front_color = -1
        if front is not None:
            front_color = COLOR_TO_IDX[front.color]
            if isinstance(front, Switch):
                front_type = "switch"
            elif front.type in FRONT_TYPES:
                front_type = front.type
        carrying = -1 if self.carrying is None else COLOR_TO_IDX[self.carrying.color]
        doors = []
        for r in self.doors:
            doors.append({"kind": r.kind, "color": COLOR_TO_IDX[r.color], "open": bool(r.obj.is_open),
                          "locked": bool(r.obj.is_locked), "in_view": bool(self.agent_sees(*r.pos)),
                          "enabled": bool(getattr(r.obj, "enabled", not r.obj.is_locked)),
                          "facing": tuple(self.front_pos) == r.pos})
        switches = [{"on": bool(s.on), "in_view": bool(self.agent_sees(*pos)),
                     "facing": tuple(self.front_pos) == pos}
                    for pos, s in ((r.switch_pos, self.grid.get(*r.switch_pos)) for r in self.doors if r.kind == "switch")]
        boxes = [{"opened": self.grid.get(*pos) is not box, "facing": tuple(self.front_pos) == pos}
                 for pos, box in self.boxes]
        return {"room": self.room_of(self.agent_pos[0]), "rooms": self.rooms, "x": int(self.agent_pos[0]),
                "y": int(self.agent_pos[1]), "dir": int(self.agent_dir), "carrying": carrying,
                "front_type": front_type, "front_color": front_color, "doors": doors,
                "switches": switches, "boxes": boxes, "play_start": self.play_start,
                "step": int(self.step_count)}

    def combination(self) -> list[list]:
        def opt(v):
            return None if v is None else int(v)
        return [[r.kind, r.color, int(r.index), opt(r.key_room), bool(r.key_boxed), opt(r.switch_room)]
                for r in self.doors]


@dataclass(frozen=True)
class PixelStep:
    rgb: np.ndarray
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


ACTION_MAP = (0, 1, 2, 3, 5)   # opaque ID -> MiniGrid left, right, forward, pickup, toggle


class RoomsPort:
    """Opaque five-action RGB interface (no drop: a carried key persists until
    used or swapped). Evaluator labels are available only through
    ``last_labels``/``last_events`` and never reach the learner."""

    action_count = len(ACTION_MAP)

    def __init__(self, config: RoomsConfig):
        self.config = config
        self._env = ChainedRoomsEnv(config)
        self._ended = True
        self.last_labels: dict | None = None
        self.last_events: dict | None = None

    def _frame(self) -> np.ndarray:
        return self._env.get_frame(tile_size=self.config.tile_size, agent_pov=True).copy()

    def reset(self, seed: int) -> PixelStep:
        self._env.reset(seed=seed)
        self._ended = False
        self.last_labels = self._env.labels()
        self.last_events = dict.fromkeys(EVENTS, False)
        return PixelStep(self._frame())

    def step(self, action: int) -> PixelStep:
        if self._ended:
            raise RuntimeError("Reset required before stepping an ended episode")
        if not 0 <= int(action) < self.action_count:
            raise ValueError("Action outside opaque action space")
        _, reward, terminated, truncated, info = self._env.step(ACTION_MAP[int(action)])
        self._ended = bool(terminated or truncated)
        self.last_labels, self.last_events = info["labels"], info["events"]
        return PixelStep(self._frame(), float(reward), bool(terminated), bool(truncated))

    def close(self) -> None:
        self._env.close()


def persistent_actions(rng: np.random.Generator, count: int, action_count: int = len(ACTION_MAP),
                       repeat: float = 0.5) -> np.ndarray:
    """Uniform actions repeated for a geometric number of extra steps.

    Domain-general: no action meaning is used. Raises spatial coverage."""
    out = np.empty(count, np.int64)
    i = 0
    while i < count:
        a = int(rng.integers(action_count))
        n = 1 + int(rng.geometric(1 - repeat)) - 1
        out[i:i + n] = a
        i += n
    return out
