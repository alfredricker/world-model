"""Fast frames for the key-door room (card 005).

A frame is the whole room seen from above in MiniGrid's own 8-pixel tiles,
plus a ninth row whose first tile shows what the agent carries. States are
stored as small grids of tile codes and turned into pixels on the GPU by
indexing a table of tile images drawn by MiniGrid itself; a test checks the
result against MiniGrid's full render pixel for pixel.
"""
from __future__ import annotations

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Wall

from .keydoor import COLOURS, Layout

TILE = 8
OBJECTS = ([None, "wall", "goal"] + [("door", c, st) for c in COLOURS for st in range(3)]
           + [("key", c) for c in COLOURS]
           # card 010 (appended so card 005's codes keep their numbers): switch off / on, vase
           + [("ball", "grey"), ("ball", "yellow"), ("box", "purple")])
OBJ_INDEX = {o: i for i, o in enumerate(OBJECTS)}
N_CODES = len(OBJECTS) * 5          # object x (no agent, agent facing 0..3)


def code(obj, agent_dir: int | None = None) -> int:
    return OBJ_INDEX[obj] * 5 + (0 if agent_dir is None else agent_dir + 1)


def _minigrid_obj(obj):
    if obj is None:
        return None
    if obj == "wall":
        return Wall()
    if obj == "goal":
        return Goal()
    if obj[0] == "door":
        return Door(obj[1], is_open=obj[2] == 2, is_locked=obj[2] == 0)
    if obj[0] == "ball":
        return Ball(obj[1])
    if obj[0] == "box":
        return Box(obj[1])
    return Key(obj[1])


def tile_images() -> np.ndarray:
    """(N_CODES, TILE, TILE, 3) uint8, drawn by MiniGrid."""
    out = np.zeros((N_CODES, TILE, TILE, 3), np.uint8)
    for o in OBJECTS:
        for a in (None, 0, 1, 2, 3):
            out[code(o, a)] = Grid.render_tile(_minigrid_obj(o), agent_dir=a, highlight=False, tile_size=TILE)
    return out


def _base(layout: Layout, carry: int, door: int) -> np.ndarray:
    s = layout.size
    g = np.full((s + 1, s), code(None), np.uint8)
    for x in range(s):
        for y in range(s):
            if x in (0, s - 1) or y in (0, s - 1) or (x == layout.wall_x and y != layout.door_y):
                g[y, x] = code("wall")
    g[layout.door_y, layout.wall_x] = code(("door", layout.door_colour, door))
    if carry != 1:
        g[layout.key[1], layout.key[0]] = code(("key", layout.door_colour))
    if carry != 2:
        g[layout.distractor[1], layout.distractor[0]] = code(("key", layout.distractor_colour))
    g[layout.goal[1], layout.goal[0]] = code("goal")
    held = (None, ("key", layout.door_colour), ("key", layout.distractor_colour))[carry]
    g[s, 0] = code(held)
    return g


def encode_states(layout: Layout, states: np.ndarray) -> np.ndarray:
    """(T, 5) states -> (T, size + 1, size) uint8 tile codes."""
    bases = np.stack([np.stack([_base(layout, c, d) for d in range(3)]) for c in range(3)])
    states = np.asarray(states)
    x, y, d, c, o = states.T
    out = bases[c, o].copy()
    t = np.arange(len(states))
    under = out[t, y, x]                        # the object code under the agent (no agent part)
    out[t, y, x] = under + d + 1
    return out


def render(codes, tiles):
    """Tile codes (B, H, W) -> frames (B, H*TILE, W*TILE, 3); works on numpy or torch."""
    img = tiles[codes]                          # B, H, W, T, T, 3
    b, h, w = codes.shape
    if hasattr(img, "permute"):
        return img.permute(0, 1, 3, 2, 4, 5).reshape(b, h * TILE, w * TILE, 3)
    return img.transpose(0, 1, 3, 2, 4, 5).reshape(b, h * TILE, w * TILE, 3)


def encode_logic_states(layout, states) -> np.ndarray:
    """Card 010's logic-door states -> tile codes: the closed door is drawn
    locked, the switch as a ball (grey off, yellow on), the vase as a
    purple box (gone once broken), keys wherever they lie."""
    n = layout.size
    base = np.full((n + 1, n), code(None), np.uint8)
    for x in range(n):
        for y in range(n):
            if x in (0, n - 1) or y in (0, n - 1) or (x == layout.wall_x and y != layout.door_y):
                base[y, x] = code("wall")
    base[layout.goal[1], layout.goal[0]] = code("goal")
    out = np.repeat(base[None], len(states), 0)
    for t, s in enumerate(states):
        x, y, d, carry, key, dist, door, sw, vase = s
        g = out[t]
        g[layout.door_y, layout.wall_x] = code(("door", layout.door_colour, 2 if door else 0))
        g[layout.switch[1], layout.switch[0]] = code(("ball", "yellow" if sw else "grey"))
        if not vase:
            g[layout.vase[1], layout.vase[0]] = code(("box", "purple"))
        if key is not None:
            g[key[1], key[0]] = code(("key", layout.door_colour))
        if dist is not None:
            g[dist[1], dist[0]] = code(("key", layout.distractor_colour))
        held = (None, ("key", layout.door_colour), ("key", layout.distractor_colour))[carry]
        g[n, 0] = code(held)
        g[y, x] = g[y, x] + d + 1
    return out
