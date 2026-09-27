import numpy as np

from worldmodel.envs.keydoor import KeyDoorEnv, env_state, make_layout, start_state, step


def test_symbolic_step_matches_minigrid():
    rng = np.random.default_rng(0)
    for episode in range(40):
        layout = make_layout(int(rng.integers(6, 9)), rng)
        env = KeyDoorEnv(layout, max_steps=10_000)
        env.reset(seed=episode)
        state = start_state(layout)
        assert env_state(env) == state
        # Bias toward interactions so pickups, unlocks and closes all occur.
        for a in rng.choice(5, size=600, p=[0.2, 0.2, 0.3, 0.15, 0.15]):
            _, _, terminated, _, _ = env.step(int(a))
            state, done = step(layout, state, int(a))
            assert env_state(env) == state
            assert terminated == done
            if done:
                break
        env.close()


def test_layouts_are_solvable_and_varied():
    rng = np.random.default_rng(1)
    layouts = [make_layout(7, rng) for _ in range(50)]
    assert len({(l.wall_x, l.door_y, l.key, l.goal) for l in layouts}) > 40
    for l in layouts:
        assert l.door_colour != l.distractor_colour
        assert l.key != l.distractor and l.start[:2] not in (l.key, l.distractor)


def test_rendered_frames_match_minigrid_and_are_distinct():
    from worldmodel.envs.keydoor_render import TILE, encode_states, render, tile_images
    tiles = tile_images()
    assert len({t.tobytes() for t in tiles}) == len(tiles)
    rng = np.random.default_rng(2)
    for episode in range(10):
        layout = make_layout(8, rng)
        env = KeyDoorEnv(layout, max_steps=10_000)
        env.reset(seed=episode)
        states, frames = [start_state(layout)], [env.get_frame(highlight=False, tile_size=TILE)]
        for a in rng.choice(5, size=300, p=[0.2, 0.2, 0.3, 0.15, 0.15]):
            _, _, terminated, _, _ = env.step(int(a))
            states.append(env_state(env))
            frames.append(env.get_frame(highlight=False, tile_size=TILE))
            if terminated:
                break
        env.close()
        codes = encode_states(layout, np.array(states))
        ours = render(codes, tiles)
        assert np.array_equal(ours[:, :8 * TILE], np.stack(frames))
        # Different states of a layout give different frames (the carried row included).
        by_state = {}
        for s, c in zip(states, codes):
            by_state.setdefault(c.tobytes(), set()).add(s)
        assert all(len(v) == 1 for v in by_state.values())
