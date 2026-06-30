################################################################################
# Lightweight smoke checks for the constrained navigation environment.
#
# These tests intentionally verify only milestone-critical behavior:
# Gymnasium reset/step API, fixed shapes/dtypes, finite values, seeded reset,
# forced success/collision, and a short random rollout.
#
# They are not a full contract suite and do not test every private helper.
################################################################################

from __future__ import annotations
import numpy as np


REQUIRED_INFO_KEYS = {
    "success",
    "collision",
    "distance_to_goal",
    "min_obstacle_distance",
    "step_count",
}


def make_env(**kwargs):
    from environments.constrained_navigation import ConstrainedNavigationEnv
    return ConstrainedNavigationEnv(**kwargs)


def assert_basic_obs(env, obs):
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert obs.shape == (21,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def assert_basic_info(info):
    assert isinstance(info, dict)
    assert REQUIRED_INFO_KEYS.issubset(info.keys())

# Deterministic layout with no active obstacles.
def no_obstacle_options(env, *, start=(0.0, 0.0), theta=0.0, goal=(1.0, 0.0)):
    return {
        "start": np.asarray(start, dtype=np.float32),
        "theta": float(theta),
        "goal": np.asarray(goal, dtype=np.float32),
        "obstacle_centers": np.zeros((env.max_obstacles, 2), dtype=np.float32),
        "obstacle_radii": np.zeros(env.max_obstacles, dtype=np.float32),
        "obstacle_mask": np.zeros(env.max_obstacles, dtype=bool),
    }

# Deterministic layout where the agent starts inside one active obstacle.
def collision_options(env):
    centers = np.zeros((env.max_obstacles, 2), dtype=np.float32)
    radii = np.zeros(env.max_obstacles, dtype=np.float32)
    mask = np.zeros(env.max_obstacles, dtype=bool)

    centers[0] = np.asarray([0.0, 0.0], dtype=np.float32)
    radii[0] = 0.5
    mask[0] = True

    return {
        "start": np.asarray([0.0, 0.0], dtype=np.float32),
        "theta": 0.0,
        "goal": np.asarray([5.0, 0.0], dtype=np.float32),
        "obstacle_centers": centers,
        "obstacle_radii": radii,
        "obstacle_mask": mask,
    }


def test_reset_step_api_shape_dtype_and_finite_values():
    env = make_env()

    obs, info = env.reset(seed=0, options=no_obstacle_options(env))

    assert env.action_space.shape == (2,)
    assert_basic_obs(env, obs)
    assert_basic_info(info)

    action = np.asarray([0.1, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert_basic_obs(env, obs)
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert_basic_info(info)


def test_same_seed_produces_same_reset_observation():
    env1 = make_env()
    env2 = make_env()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    np.testing.assert_allclose(obs1, obs2)
    assert info1["distance_to_goal"] == info2["distance_to_goal"]
    assert info1["min_obstacle_distance"] == info2["min_obstacle_distance"]


def test_forced_success_case_using_reset_options():
    env = make_env()

    env.reset(
        seed=0,
        options=no_obstacle_options(
            env,
            start=(0.0, 0.0),
            goal=(0.0, 0.0),
        ),
    )

    obs, reward, terminated, truncated, info = env.step(
        np.asarray([0.0, 0.0], dtype=np.float32)
    )

    assert_basic_obs(env, obs)
    assert np.isfinite(reward)
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["collision"] is False


def test_forced_collision_case_using_reset_options():
    env = make_env()

    env.reset(seed=0, options=collision_options(env))

    obs, reward, terminated, truncated, info = env.step(
        np.asarray([0.0, 0.0], dtype=np.float32)
    )

    assert_basic_obs(env, obs)
    assert np.isfinite(reward)
    assert terminated is True
    assert truncated is False
    assert info["collision"] is True
    assert info["success"] is False


def test_short_random_rollout_has_finite_observations_and_rewards():
    env = make_env(max_episode_steps=20)

    env.action_space.seed(0)
    obs, info = env.reset(seed=0)

    assert_basic_obs(env, obs)
    assert_basic_info(info)

    done = False
    steps = 0

    while not done and steps < env.max_episode_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert_basic_obs(env, obs)
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert_basic_info(info)

        done = terminated or truncated
        steps += 1

    assert steps > 0
    assert done is True


