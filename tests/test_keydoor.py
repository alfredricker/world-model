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
