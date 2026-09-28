"""Check full-tree gate exceptions against the simulator, not just sample counts."""
import importlib.util
from pathlib import Path

import numpy as np

from worldmodel import discover_logic as dl
from worldmodel.envs import logicdoor as ld


spec = importlib.util.spec_from_file_location("card021_bench", Path(__file__).resolve().parents[1] /
                                              "tools/card021/bench_full_tree.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_two_toggles_are_real_but_third_adds_no_access():
    # Matching key held; distractor and vase trap the agent in the corner.
    # Breaking the vase makes the door accessible; opening it enables the goal.
    lay = ld.Layout(8, 5, 4, "red", "blue", (2, 3), (2, 1), (2, 4), (1, 2),
                    (6, 3), (3, 6, 1), "key")
    ex = dl.Exact(lay, [-1, 0, 1, 2, 3], [-1, ld.FORWARD, ld.TOGGLE, ld.TOGGLE, ld.TOGGLE])
    trapped = (1, 1, 1, 1, None, (2, 1), 0, 0, 0)
    assert not ex.holds(2, trapped)
    assert ex.ready(2, ld.TOGGLE, trapped)
    assert ex.holds(3, trapped)
    for state in bench.configurations(lay, np.random.default_rng(21)):
        for pose in ex.component(state[3:])[0]:
            query = (*pose, *state[3:])
            assert ex.holds(3, query) == ex.holds(4, query)
            assert not ex.ready(3, ld.TOGGLE, query)


def test_constructed_states_have_consistent_inventory_and_no_overlapping_objects():
    rng = np.random.default_rng(210)
    for _ in range(4):
        lay = ld.make_layout(8, "key", rng)
        for s in bench.configurations(lay, rng):
            assert (s[4] is None) == (s[3] == 1)
            assert (s[5] is None) == (s[3] == 2)
            objects = [p for p in (s[4], s[5], lay.switch, lay.vase, lay.goal) if p is not None]
            assert len(set(objects)) == len(objects)
            assert ld.cell(lay, s, s[:2]) in ("empty", "goal")


def test_inactive_exception_cannot_hide_a_positive_or_bad_false_positive_rate():
    inactive = {"positive": 0, "negative": 100, "recall": None, "false_positive": 0.}
    perfect = {"positive": 100, "negative": 100, "recall": 1., "false_positive": 0.}
    result = {"readiness": {bench.INACTIVE: {tag: dict(inactive) for tag in
               ("agent", "other_poses", "unseen_wall")}}, "conditions": {"condition": perfect}}
    assert bench.gate_passes(result, True)[0]
    result["readiness"][bench.INACTIVE]["agent"]["positive"] = 1
    assert not bench.gate_passes(result, True)[0]
    result["readiness"][bench.INACTIVE]["agent"] = {**inactive, "false_positive": .02}
    assert not bench.gate_passes(result, True)[0]
    result["readiness"] = {"another way": {tag: dict(inactive) for tag in
                          ("agent", "other_poses", "unseen_wall")}}
    assert not bench.gate_passes(result, True)[0]
