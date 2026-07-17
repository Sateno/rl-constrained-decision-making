################################################################################
# Lightweight smoke checks for the CBF-QP projection wrapper.
#
# These tests intentionally verify only scientifically critical behavior:
# physical-action projection before environment execution, projection
# diagnostics in the info dictionary, and normalized-action wrapper ordering.
################################################################################

from __future__ import annotations

import numpy as np

from environments.factory import make_env
from environments.action_wrappers import NormalizedActionWrapper
from environments.constrained_navigation import ConstrainedNavigationEnv
from projection.cbf_qp_projection import ProjectionParams
from projection.cbf_qp_wrapper import CbfQpProjectionWrapper


PROJECTION_INFO_KEYS = {
    "projection_enabled",
    "projection_action_raw",
    "projection_action_exec",
    "projection_correction",
    "projection_intervened",
    "projection_correction_norm",
    "projection_slack_values",
    "projection_slack_max",
    "projection_slack_sum",
    "projection_success",
    "projection_solver_status",
    "projection_active_constraint_count",
}


# Return a deterministic layout with one active obstacle in front of the agent.
def projection_obstacle_options(env: ConstrainedNavigationEnv) -> dict:
#{
    obstacle_centers = np.zeros((env.max_obstacles, 2), dtype=np.float64)
    obstacle_radii = np.zeros(env.max_obstacles, dtype=np.float64)
    obstacle_mask = np.zeros(env.max_obstacles, dtype=bool)

    obstacle_centers[0] = np.asarray([1.35, 0.0], dtype=np.float64)
    obstacle_radii[0] = 0.25
    obstacle_mask[0] = True

    return {
        "start": np.asarray([0.0, 0.0], dtype=np.float64),
        "theta": 0.0,
        "goal": np.asarray([5.0, 0.0], dtype=np.float64),
        "obstacle_centers": obstacle_centers,
        "obstacle_radii": obstacle_radii,
        "obstacle_mask": obstacle_mask,
    }

#} End function projection_obstacle_options


# Return a deterministic layout with no active obstacles.
def no_obstacle_options(env: ConstrainedNavigationEnv) -> dict:
#{
    return {
        "start": np.asarray([0.0, 0.0], dtype=np.float64),
        "theta": 0.0,
        "goal": np.asarray([5.0, 0.0], dtype=np.float64),
        "obstacle_centers": np.zeros((env.max_obstacles, 2), dtype=np.float64),
        "obstacle_radii": np.zeros(env.max_obstacles, dtype=np.float64),
        "obstacle_mask": np.zeros(env.max_obstacles, dtype=bool),
    }

#} End function no_obstacle_options


# Verify that the wrapper projects a physical action and reports diagnostics.
def test_projection_wrapper_projects_physical_action_and_reports_diagnostics() -> None:
#{
    base_env = ConstrainedNavigationEnv()
    params = ProjectionParams(lookahead_distance=0.25, alpha=2.0, extra_clearance=0.0, slack_penalty=10000.0)
    env = CbfQpProjectionWrapper(base_env, params=params)

    env.reset(seed=0, options=projection_obstacle_options(base_env))

    raw_action = np.asarray([1.0, 0.0], dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(raw_action)

    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert PROJECTION_INFO_KEYS.issubset(info.keys())
    assert info["projection_enabled"] is True
    assert info["projection_success"] is True
    assert info["projection_solver_status"] in {"optimal", "optimal_inaccurate"}
    assert info["projection_active_constraint_count"] == 1
    assert info["projection_intervened"] is True
    assert info["projection_correction_norm"] > 0.0
    assert info["projection_slack_max"] >= 0.0
    assert info["projection_slack_sum"] >= 0.0
    np.testing.assert_allclose(info["projection_action_raw"], raw_action)
    np.testing.assert_allclose(
        info["projection_action_exec"] - info["projection_action_raw"],
        info["projection_correction"],
    )
    np.testing.assert_allclose(
        np.linalg.norm(info["projection_correction"]),
        info["projection_correction_norm"],
    )
    assert info["projection_slack_values"].shape == (base_env.max_obstacles,)
    np.testing.assert_allclose(
        info["projection_slack_values"][1:],
        np.zeros(base_env.max_obstacles - 1),
    )
    np.testing.assert_allclose(
        np.sum(info["projection_slack_values"]),
        info["projection_slack_sum"],
    )
    np.testing.assert_allclose(
        np.max(info["projection_slack_values"]),
        info["projection_slack_max"],
    )

    # This layout requires correction only when the wrapper includes agent_radius.
    # A smaller displacement confirms both radius inflation and projected execution.
    assert 0.0 < base_env.position[0] < raw_action[0] * base_env.dt
    np.testing.assert_allclose(base_env.position[1], 0.0, atol=1.0e-8)

#} End function test_projection_wrapper_projects_physical_action_and_reports_diagnostics


# Verify that the factory places the normalized-action wrapper outside the projection wrapper.
def test_normalized_wrapper_is_outside_projection_wrapper() -> None:
#{
    env_factory = make_env(
        env_index=0,
        record_episode_statistics=False,
        normalize_actions=True,
        enable_projection=True,
        projection_params=None,
    )

    env = env_factory()
    base_env = env.unwrapped

    assert isinstance(env, NormalizedActionWrapper)
    assert isinstance(env.env, CbfQpProjectionWrapper)
    assert isinstance(base_env, ConstrainedNavigationEnv)

    env.reset(seed=0, options=no_obstacle_options(base_env))

    np.testing.assert_allclose(env.action_space.low, np.asarray([-1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(env.action_space.high, np.asarray([1.0, 1.0], dtype=np.float32))

    normalized_action = np.asarray([0.0, 0.0], dtype=np.float32)
    _, _, terminated, truncated, info = env.step(normalized_action)

    expected_position = np.asarray([0.5 * base_env.v_max * base_env.dt, 0.0], dtype=np.float64)

    assert terminated is False
    assert truncated is False
    assert PROJECTION_INFO_KEYS.issubset(info.keys())
    assert info["projection_enabled"] is True
    assert info["projection_success"] is True
    assert info["projection_solver_status"] == "no_active_constraints"
    assert info["projection_active_constraint_count"] == 0
    assert info["projection_intervened"] is False
    np.testing.assert_allclose(info["projection_action_raw"], np.asarray([0.5, 0.0]))
    np.testing.assert_allclose(info["projection_action_exec"], info["projection_action_raw"])
    np.testing.assert_allclose(info["projection_correction"], np.zeros(2))
    np.testing.assert_allclose(info["projection_slack_values"], np.zeros(base_env.max_obstacles))
    np.testing.assert_allclose(base_env.position, expected_position, atol=1.0e-8)

#} End function test_normalized_wrapper_is_outside_projection_wrapper