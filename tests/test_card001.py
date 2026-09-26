"""Tests for the card-001 machinery: symbolic states, fork extras, the hub's
quasimetric head and losses, the harness's scoring, and the batch sampler."""
from pathlib import Path

import numpy as np
import pytest
import torch

from worldmodel.envs import rooms_goals as G
from worldmodel.envs.rooms import RoomsConfig, RoomsPort, persistent_actions
from worldmodel.envs.rooms_data import collect, world_seed
from worldmodel.envs.rooms_fork_extras import _pairs, shortest_path_goal
from worldmodel.envs.rooms_symbolic import TYPES, annotate, load_symbolic, symbolic
from worldmodel.fork_eval import SCORED, rank, spearman, summarise, top1_rows
from worldmodel.models.hub import Hub, HubConfig, MRNHead, sigreg


def _world(index=0, split="development", play=0.0):
    port = RoomsPort(RoomsConfig(split=split, play_probability=play))
    port.reset(world_seed(0, split, index))
    env = port._env
    return port, env, G.layout_of(env)


# --- symbolic states -------------------------------------------------------

def test_symbolic_marks_agent_doors_and_carried():
    port, env, layout = _world()
    state = G.state_of(env, layout)
    grid, carried = symbolic(layout, state)
    x, y, d = state[:3]
    assert grid[y, x, 5] == d + 1 and (grid[..., 5] > 0).sum() == 1
    for (px, py), kind, colour, _ in layout.doors:
        assert grid[py, px, 0] == TYPES.index("door")
    assert carried[0] == (0 if state[3] is None else 1 + (state[3][0] == "box"))


def test_symbolic_follows_state_through_an_episode():
    """Symbolic states differ exactly when abstract states differ."""
    port, env, layout = _world(index=3, play=1.0)
    rng = np.random.default_rng(0)
    seen = {}
    for action in persistent_actions(rng, 300, 5, 0.5):
        st = G.state_of(env, layout)
        g, c = symbolic(layout, st)
        key = g.tobytes() + c.tobytes()
        if key in seen:
            assert seen[key] == st
        seen[key] = st
        if port.step(int(action)).terminated:
            break


def test_annotate_replays_frames(tmp_path):
    collect(tmp_path / "d", RoomsConfig(max_steps=40, play_probability=0.5), 3, seed=5, workers=1)
    annotate(tmp_path / "d", workers=1)
    grids, carried = load_symbolic(tmp_path / "d")
    import json
    meta = json.loads((tmp_path / "d" / "dataset.json").read_text())
    for g, c, rec in zip(grids, carried, meta["episodes"]):
        assert g.shape == (rec["length"] + 1, 8, 29, 6) and c.shape == (rec["length"] + 1, 2)


# --- fork extras ------------------------------------------------------------

@pytest.fixture(scope="module")
def searched():
    port, env, layout = _world(index=1)
    start = G.state_of(env, layout)
    graph = G.reachable(layout, start)
    goals = G.goals_of(layout, start)
    dist = {g: G.distances(layout, graph, g) for g in goals}
    return layout, start, graph, dist


def test_shortest_path_goal_holds_and_is_that_far(searched):
    layout, start, graph, dist = searched
    rng = np.random.default_rng(0)
    for g, d in dist.items():
        ids = np.flatnonzero((d != G.INF) & (d > 0))
        for sid in rng.choice(ids, size=min(20, len(ids)), replace=False):
            target = shortest_path_goal(graph, d, int(sid))
            assert G.holds(layout, graph.states[target], g)
            # Walk back: the path found has exactly d[sid] steps.
            cur, steps = int(sid), 0
            while cur != target:
                cur = next(int(j) for j in graph.succ[cur] if j >= 0 and d[j] == d[cur] - 1)
                steps += 1
            assert steps == d[sid]


def test_pairs_differ_in_one_condition(searched):
    layout, start, graph, dist = searched
    pairs = _pairs(layout, graph, start, dist, np.random.default_rng(0), per_kind=5, scan=40000)
    assert pairs
    for kind, g, a, b in pairs:
        sa, sb = graph.states[a], graph.states[b]
        assert sa[:3] == sb[:3] and sa[6] == sb[6]
        if kind == "key":
            assert sa[3] == ("key", g[1]) and sb[3] is None and sa[5] == sb[5]
        elif kind == "box":
            assert sa[3] == sb[3] and sa[5] == sb[5]
            diff = set(sa[4]) ^ set(sb[4])
            assert {item[0] for _, item in diff} == {"key", "box"} and len({p for p, _ in diff}) == 1
        else:
            changed = [i for i, (x, y) in enumerate(zip(sa[5], sb[5])) if x != y]
            assert len(changed) == 1 and layout.doors[changed[0]][1] == "plain"
            assert sa[5][changed[0]][1] and not sb[5][changed[0]][1]
            assert sa[3:5] == sb[3:5]


# --- hub ----------------------------------------------------------------------

def test_head_is_a_quasimetric_and_pooling_is_a_lower_bound():
    torch.manual_seed(0)
    head = MRNHead(HubConfig())
    z = torch.randn(64, 192)
    f = head.features(z)
    x = (f[0][:32], f[1][:32])
    y = (f[0][32:], f[1][32:])
    assert torch.allclose(head.distance(x, x), torch.zeros(32), atol=1e-5)
    # Triangle inequality over all triples of 16 points.
    p = (f[0][:16], f[1][:16])
    d = head.distance((p[0][:, None], p[1][:, None]), (p[0][None], p[1][None]))
    for i in range(16):
        for j in range(16):
            assert (d[i, j] <= d[i] + d[:, j] + 1e-4).all()
    # Pooled distance to a set is at most the distance to each member.
    ys = (y[0].view(8, 4, -1), y[1].view(8, 4, -1))
    xs = (x[0][:8], x[1][:8])
    pooled = head.set_distance(xs, ys, "pooled")
    each = head.distance((xs[0][:, None], xs[1][:, None]), ys)
    assert (pooled[:, None] <= each + 1e-4).all()
    assert torch.allclose(head.set_distance(xs, ys, "min"), each.amin(1))


def test_transition_starts_as_nothing_happens():
    hub = Hub(HubConfig())
    z = torch.randn(10, 192)
    for a in range(5):
        # Only the output layer is not zero-initialised; with zero-gated blocks
        # T(z, a) = z + out(inp(z)), identical for every action.
        assert torch.allclose(hub.transition(z, torch.full((10,), a)), hub.transition(z, torch.zeros(10, dtype=torch.long)))


def test_sigreg_prefers_gaussian():
    torch.manual_seed(0)
    gaussian = torch.randn(1024, 32)
    collapsed = torch.randn(1024, 1).repeat(1, 32) * 0.1
    assert sigreg(gaussian, 256) < 0.1 * sigreg(collapsed, 256)


# --- harness --------------------------------------------------------------------

def test_top1_and_ties():
    best = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]], bool)
    pred = np.array([[0, 1, 1, 1, 1], [5, 0, 0, 0, 1.0]])
    assert np.allclose(top1_rows(pred, best), [1, 2 / 3])


def test_spearman_and_ranks():
    assert np.allclose(rank(np.array([3, 1, 1, 2])), [3, 0.5, 0.5, 2])
    a = np.arange(20.0)
    assert spearman(a, a ** 3) == pytest.approx(1)
    assert spearman(a, -a) == pytest.approx(-1)


def test_summarise_truth_scores_one():
    class FS:
        pass
    fs = FS()
    rng = np.random.default_rng(0)
    n = 60
    fs.strata = np.array([SCORED[i % 6] for i in range(n)])
    fs.dist = rng.integers(1, 9, (n, 5)).astype(float)
    fs.best = fs.dist == fs.dist.min(1, keepdims=True)
    fs.no_effect = np.zeros((n, 5), bool)
    fs.goal = np.zeros(n)
    assert summarise(fs, fs.dist)["scored_top1"] == pytest.approx(1)
    assert summarise(fs, -fs.dist)["scored_top1"] < 0.5


def test_egocentric_view_is_minigrids_observation():
    from worldmodel.envs.rooms_symbolic import egocentric
    port, env, layout = _world(index=2, play=1.0)
    rng = np.random.default_rng(1)
    for action in persistent_actions(rng, 100, 5, 0.5):
        view, carried = egocentric(env)
        assert np.array_equal(view, env.gen_obs()["image"])
        if env.carrying is None:
            assert not carried.any()
        if port.step(int(action)).terminated:
            break


def test_goal_sets_come_from_a_later_stretch(tmp_path):
    """Sampler: set members lie in the anchor's episode, after it, in order."""
    from worldmodel.train import Data, TrainConfig
    collect(tmp_path / "d", RoomsConfig(max_steps=60, play_probability=0.5), 4, seed=2, workers=1)
    cfg = TrainConfig(out=tmp_path / "o", data=tmp_path / "d", sets=0.5, next_goal=0.2, batch=64)
    data = Data(cfg, torch.device("cpu"))
    gen = torch.Generator()
    gen.manual_seed(0)
    tr, f, goal, bound, members, set_bound, cond, examples = data.sample(cfg, gen)
    ns = len(members)
    assert ns == 32
    end = data.t_end[tr[:ns]]
    assert (members[:, 0] > f[:ns]).all() and (members <= end[:, None]).all()
    assert (members[:, 1:] >= members[:, :-1]).all()
    assert torch.equal(set_bound, (members[:, 0] - f[:ns]).float())
    finite = torch.isfinite(bound)
    assert (goal[finite] > f[finite]).all() and torch.equal(bound[finite], (goal - f)[finite].float())



def test_condition_goals_and_labels(tmp_path):
    """Supplied condition goals: examples show the condition; next_hold is
    the first frame at or after i where it holds, within i's episode."""
    from worldmodel.train import Data, TrainConfig
    collect(tmp_path / "d", RoomsConfig(max_steps=200, play_probability=1.0), 6, seed=3, workers=1)
    cfg = TrainConfig(out=tmp_path / "o", data=tmp_path / "d", conditions=0.5, batch=64)
    data = Data(cfg, torch.device("cpu"))
    gen = torch.Generator()
    gen.manual_seed(0)
    *_, cond, examples = data.sample(cfg, gen)
    assert len(cond) == 32 and examples.shape == (32, 4)
    assert data.holds[examples, cond[:, None]].all()
    assert not data.holds[:, 3].any()                    # no grey keys
    h, nxt = data.holds.numpy(), data.next_hold.numpy()
    for i in range(0, len(h) - 1, 7):
        for c in range(8):
            j = nxt[i, c]
            if j < len(h):
                assert h[j, c] and j >= i and not h[i:j, c].any()
