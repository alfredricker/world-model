import numpy as np

from worldmodel.envs.logicdoor import (DROP, FORWARD, PICKUP, TOGGLE, Layout, cell, make_layout, start_state,
                                       step)


def _facing(lay: Layout, s: tuple, pos) -> tuple:
    """Put the agent next to pos, facing it (test helper; ignores walls)."""
    x, y = pos
    for d, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        ax, ay = x - dx, y - dy
        if cell(lay, s, (ax, ay)) == "empty":
            return (ax, ay, d, *s[3:])
    raise AssertionError("no free side")


def test_door_rules():
    rng = np.random.default_rng(0)
    for rule in ("key", "switch", "either", "both"):
        lay = make_layout(8, rule, rng)
        for carry in (0, 1, 2):
            for sw in (0, 1):
                s = _facing(lay, start_state(lay, carry, sw), lay.door)
                t, _ = step(lay, s, TOGGLE)
                key = carry == 1
                expected = {"key": key, "switch": sw == 1, "either": key or sw == 1, "both": key and sw == 1}[rule]
                assert t[6] == int(expected)
                if t[6]:                       # closing needs nothing; reopening needs the rule again
                    c, _ = step(lay, t, TOGGLE)
                    assert c[6] == 0


def test_drop_pickup_switch_vase():
    rng = np.random.default_rng(1)
    lay = make_layout(8, "key", rng)
    s = _facing(lay, start_state(lay), lay.key)
    s, _ = step(lay, s, PICKUP)
    assert s[3] == 1 and s[4] is None
    if cell(lay, s, (s[0] + (1, 0, -1, 0)[s[2]], s[1] + (0, 1, 0, -1)[s[2]])) == "empty":
        d, _ = step(lay, s, DROP)
        assert d[3] == 0 and d[4] is not None
    s = _facing(lay, start_state(lay), lay.switch)
    on, _ = step(lay, s, TOGGLE)
    off, _ = step(lay, on, TOGGLE)
    assert on[7] == 1 and off[7] == 0
    s = _facing(lay, start_state(lay), lay.vase)
    b, _ = step(lay, s, TOGGLE)
    assert b[8] == 1 and cell(lay, b, lay.vase) == "empty"
    b2, _ = step(lay, b, TOGGLE)
    assert b2[8] == 1                          # breaking cannot be undone


def test_logic_frames_are_distinct_per_state():
    from worldmodel.envs.keydoor_render import encode_logic_states, render, tile_images
    from worldmodel.envs.logicdoor import collect
    tiles = tile_images()
    assert len({t.tobytes() for t in tiles}) == len(tiles)
    for rule in ("key", "both"):
        for ep in collect(rule, 5, 3):
            codes = encode_logic_states(ep["layout"], ep["states"])
            seen = {}
            for s, c in zip(ep["states"], codes):
                seen.setdefault(c.tobytes(), set()).add(s)
            assert all(len(v) == 1 for v in seen.values())
            assert render(codes[:2], tiles).shape == (2, 72, 64, 3)
