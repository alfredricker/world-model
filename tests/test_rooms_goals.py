import copy

import numpy as np
import pytest
from minigrid.core.world_object import Door, Key

from worldmodel.envs import rooms_goals as G
from worldmodel.envs.rooms import ACTION_MAP, RoomsConfig, RoomsPort, persistent_actions
from worldmodel.envs.rooms_data import world_seed


def _port(split="development", play=0.0, index=0, seed=0):
    port = RoomsPort(RoomsConfig(split=split, play_probability=play))
    port.reset(world_seed(seed, split, index))
    return port


@pytest.mark.parametrize("play", [0.0, 0.5, 1.0])
def test_rules_match_environment(play):
    """The re-implemented rules agree with the real environment on every step
    of random episodes, including terminations."""
    rng = np.random.default_rng(3)
    steps = 0
    for index in range(12):
        port = _port(play=play, index=index, seed=11)
        env = port._env
        layout = G.layout_of(env)
        for action in persistent_actions(rng, 512, 5, 0.5):
            state = G.state_of(env, layout)
            predicted, term = G.advance(layout, state, int(action))
            observed = port.step(int(action))
            assert G.state_of(env, layout) == predicted, (index, steps, int(action))
            assert term == observed.terminated
            steps += 1
            if observed.terminated or observed.truncated:
                break
    assert steps > 3000


def test_rules_match_environment_at_interactions():
    """Random episodes rarely unlock; also check every action from every state
    where an object is in front, including play starts holding a matching key."""
    checked = 0
    for index in range(30):
        port = _port(play=1.0, index=index, seed=5)
        env = port._env
        layout = G.layout_of(env)
        rng = np.random.default_rng(index)
        for action in persistent_actions(rng, 200, 5, 0.5):
            front = env.grid.get(*env.front_pos)
            if front is not None and front.type != "wall":
                state = G.state_of(env, layout)
                for a in range(5):
                    branch = copy.deepcopy(env)
                    _, _, term, _, _ = branch.step(ACTION_MAP[a])
                    predicted, pterm = G.advance(layout, state, a)
                    assert G.state_of(branch, layout) == predicted
                    assert pterm == term
                    checked += 1
            observed = port.step(int(action))
            if observed.terminated or observed.truncated:
                break
    assert checked > 500


def _hand_state(**kw):
    """A tiny corridor world: walls around a 1-by-5 strip, a door at x = 4."""
    blocked = frozenset({(x, 0) for x in range(8)} | {(x, 2) for x in range(8)} | {(0, 1), (7, 1)})
    doors = kw.get("doors", (((4, 1), "locked", "blue", 0),))
    layout = G.Layout(blocked, frozenset(), (6, 1), doors, (), True)
    state = (1, 1, 0, kw.get("carrying"), kw.get("objects", ()), kw.get("door_state", ((True, False, 0),)), ())
    return layout, state


def test_hand_counted_distances():
    # Agent at x = 1 facing east; blue key at x = 2 on the floor; locked blue door at x = 4.
    layout, start = _hand_state(objects=(((2, 1), ("key", "blue")),))
    graph = G.reachable(layout, start)
    sid = graph.index[start]
    door = G.distances(layout, graph, ("door_open", "blue"))
    # pickup (1), forward to x = 2 (1), forward to x = 3 (1), toggle (1) = 4 steps.
    assert door[sid] == 4
    ad = G.action_distances(graph, door, sid)
    assert ad[G.PICKUP] == 4
    # toggle and forward change nothing here (a key blocks the way), so they cost one step more.
    assert ad[G.TOGGLE] == 5 and ad[G.FORWARD] == 5
    # Turning away costs the turn and the turn back: 2 + 4.
    assert ad[G.LEFT] == 6 and ad[G.RIGHT] == 6
    cls, best = G.classify(ad)
    assert cls == "interaction" and best.tolist() == [False, False, False, True, False]
    assert G.no_effect(graph, sid).tolist() == [False, False, True, False, True]
    key = G.distances(layout, graph, ("key_held", "blue"))
    assert key[sid] == 1


def test_unreachable_goal_is_infinite():
    # No key anywhere: the locked door can never open.
    layout, start = _hand_state()
    graph = G.reachable(layout, start)
    dist = G.distances(layout, graph, ("door_open", "blue"))
    ad = G.action_distances(graph, dist, graph.index[start])
    assert (ad == G.INF).all()
    assert G.classify(ad)[0] == "indifferent"


def test_boxed_key_needs_toggle_then_pickup():
    layout, start = _hand_state(objects=(((2, 1), ("box", "blue")),))
    graph = G.reachable(layout, start)
    sid = graph.index[start]
    dist = G.distances(layout, graph, ("key_held", "blue"))
    ad = G.action_distances(graph, dist, sid)
    assert ad[G.TOGGLE] == 2          # open the box, then pick up the key
    assert ad[G.PICKUP] == G.INF      # carrying the box forever: no drop action
    assert G.classify(ad)[0] == "interaction"


def test_movement_decisive_fork():
    # Door already unlocked and closed, two cells away: walking is best.
    layout, _ = _hand_state(door_state=((False, False, 0),))
    start = (2, 1, 0, None, (), ((False, False, 0),), ())
    graph = G.reachable(layout, start)
    sid = graph.index[start]
    ad = G.action_distances(graph, G.distances(layout, graph, ("door_open", "blue")), sid)
    assert ad[G.FORWARD] == 2
    assert G.classify(ad)[0] == "movement"


def test_timed_door_closes_after_delay():
    doors = (((4, 1), "timed", "red", 3),)
    layout, state = _hand_state(doors=doors, door_state=((False, False, 0),))
    state = (3, 1, 0, None, (), ((False, False, 0),), ())
    state, _ = G.advance(layout, state, G.TOGGLE)
    assert state[5][0][1] and state[5][0][2] == 0
    for elapsed in (1, 2):
        state, _ = G.advance(layout, state, G.LEFT)
        assert state[5][0] == (False, True, elapsed)
    state, _ = G.advance(layout, state, G.LEFT)
    assert state[5][0] == (False, False, 0)


def _timed_world():
    """A real episode with a timed door, the agent placed facing it from the west."""
    for index in range(200):
        port = _port(index=index, seed=21)
        env = port._env
        timed = [r for r in env.doors if r.kind == "timed"]
        if timed and env.grid.get(timed[0].pos[0] - 1, timed[0].pos[1]) is None:
            env.agent_pos = (timed[0].pos[0] - 1, timed[0].pos[1])
            env.agent_dir = 0
            return port, timed[0]
    raise AssertionError("no episode with a reachable timed door")


@pytest.mark.parametrize("plan", ["turn_away", "stand_in_door"])
def test_timed_door_matches_environment(plan):
    """Timed doors close on their own; random episodes almost never show it,
    so drive the real environment through it and compare every step."""
    port, record = _timed_world()
    env = port._env
    layout = G.layout_of(env)
    delay = record.obj.delay
    if plan == "turn_away":
        actions = [G.TOGGLE] + [G.LEFT] * (delay + 3)
    else:   # step into the doorway, wait past the delay, step out, then wait
        actions = [G.TOGGLE, G.FORWARD] + [G.LEFT, G.RIGHT] * delay + [G.FORWARD] + [G.LEFT] * 3
    closed_by_itself = False
    for action in actions:
        state = G.state_of(env, layout)
        predicted, _ = G.advance(layout, state, action)
        port.step(action)
        observed = G.state_of(env, layout)
        assert observed == predicted
        i = layout.doors.index(next(d for d in layout.doors if d[0] == tuple(record.pos)))
        if state[5][i][1] and not observed[5][i][1] and action != G.TOGGLE:
            closed_by_itself = True
    assert closed_by_itself


def test_top1_and_chance():
    best = np.array([0, 0, 0, 1, 0])
    assert G.top1(np.array([5, 5, 5, 1, 5]), best) == 1.0
    assert G.top1(np.array([1, 5, 5, 5, 5]), best) == 0.0
    assert G.top1(np.array([1, 1, 1, 1, 1]), best) == pytest.approx(0.2)
    assert G.chance(best) == pytest.approx(0.2)
    # A shuffled ranking scores at chance on average.
    rng = np.random.default_rng(0)
    scores = [G.top1(rng.permutation(5).astype(float), best) for _ in range(5000)]
    assert abs(np.mean(scores) - 0.2) < 0.02


def test_goal_probe_examples_show_their_goal(tmp_path):
    config = RoomsConfig(split="development", play_probability=1.0, max_steps=128)
    meta = G.goal_probe(tmp_path / "gp", config, 6, 0)
    episodes, loaded = G.load_goal_probe(tmp_path / "gp")
    assert loaded["sha256"] == meta["sha256"]
    assert all(r["complete"] for r in meta["episodes"])
    names = meta["goal_names"]
    # Re-run one episode and check every visible-goal flag against simulator state.
    port = RoomsPort(config)
    port.reset(world_seed(700000, config.split, 0))
    env = port._env
    rng = np.random.default_rng(np.random.SeedSequence([0, 991]))
    policy = persistent_actions(rng, config.max_steps, 5, 0.5)
    vis = episodes[0]["visible"]
    for t, action in enumerate(policy[: len(vis) - 1]):
        for g in np.flatnonzero(vis[t]):
            kind, colour = names[g].split(":")
            if kind == "key_held":
                assert isinstance(env.carrying, Key) and env.carrying.color == colour
            else:
                assert any(isinstance(r.obj, Door) and r.obj.is_open and r.color == colour and env.agent_sees(*r.pos)
                           for r in env.doors)
        observed = port.step(int(action))
        if observed.terminated or observed.truncated:
            break
    # Stored fork rows are consistent: best actions have the minimum distance.
    for ep in episodes:
        for row in ep["forks"]:
            dist, best = row[6:11], row[11:16].astype(bool)
            if row[2] != G.CLASSES.index("indifferent"):
                assert (dist[best] == dist.min()).all() and (dist[~best] > dist.min()).all()


def test_set_state_round_trip_and_rules():
    """Setting the real environment to any reachable state reproduces that
    state, its frame-relevant front kind, and the rules from there."""
    rng = np.random.default_rng(0)
    checked = 0
    for index in range(4):
        port = _port(index=index, seed=31)
        env = port._env
        layout = G.layout_of(env)
        world = world_seed(31, "development", index)
        graph = G.reachable(layout, G.state_of(env, layout))
        for sid in rng.choice(len(graph.states), size=60, replace=False):
            state = graph.states[sid]
            if graph.terminal[sid]:
                continue
            G.set_state(env, layout, state)   # no reset: set_state must overwrite everything that moves
            assert G.state_of(env, layout) == state
            from worldmodel.envs.rooms_probes import _front
            assert G.front_kind_of(layout, state) == _front(env)[0]
            for a in range(5):
                branch = copy.deepcopy(env)
                _, _, term, _, _ = branch.step(ACTION_MAP[a])
                predicted, pterm = G.advance(layout, state, a)
                assert G.state_of(branch, layout) == predicted and pterm == term
            checked += 1
    assert checked > 150


def test_vectorised_distances_match_single_state():
    layout, start = _hand_state(objects=(((2, 1), ("key", "blue")),))
    graph = G.reachable(layout, start)
    dist = G.distances(layout, graph, ("door_open", "blue"))
    table = G.all_action_distances(graph, dist)
    for sid in range(len(graph.states)):
        assert (table[sid] == G.action_distances(graph, dist, sid)).all()


def test_sampled_probe(tmp_path):
    config = RoomsConfig(split="development")
    meta = G.sampled_probe(tmp_path / "sp", config, 4, 0, per_stratum=20, movement=40, examples_per_goal=40)
    archive = np.load(tmp_path / "sp" / "sampled_probe.npz")
    forks, frames = archive["forks"], archive["fork_frames"]
    assert len(forks) == len(frames) > 0
    assert all(r["complete"] for r in meta["worlds"])
    for row in forks:
        dist, best = row[5:10], row[10:15].astype(bool)
        assert (dist[best] == dist.min()).all() and (dist[~best] > dist.min()).all()
        if G.CLASSES[row[3]] == "interaction":
            assert not best[:3].any()
        else:
            assert not best[3:].any()
    # Stored frames are the frames of the stored states.
    world = world_seed(700000, "development", int(forks[0][0]))
    port = RoomsPort(config)
    port.reset(world)
    layout = G.layout_of(port._env)
    graph = G.reachable(layout, G.state_of(port._env, layout))
    port.reset(world)
    G.set_state(port._env, layout, graph.states[int(forks[0][1])])
    assert (G._frame(port._env, config) == frames[0]).all()
