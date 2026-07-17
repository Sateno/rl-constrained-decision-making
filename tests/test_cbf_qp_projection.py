# Smoke checks for the physical-action CBF-QP projection layer.

import numpy as np
import pytest

from projection.cbf_qp_projection import ProjectionParams, build_cbf_constraints, project_physical_action


def test_obstacle_free_action_is_unchanged() -> None:
    params = ProjectionParams()
    raw_action = np.array([0.5, 0.25], dtype=np.float64)

    result = project_physical_action(
        position=np.array([0.0, 0.0], dtype=np.float64),
        heading=0.0,
        obstacle_centers=np.zeros((0, 2), dtype=np.float64),
        obstacle_radii=np.zeros(0, dtype=np.float64),
        obstacle_mask=np.zeros(0, dtype=bool),
        agent_radius=0.10,
        raw_action=raw_action,
        params=params,
    )

    assert result.success
    assert result.solver_status == "no_active_constraints"
    assert result.active_constraint_count == 0
    assert not result.intervened
    np.testing.assert_allclose(result.action_exec, raw_action)
    np.testing.assert_allclose(result.slack_values, np.zeros(0, dtype=np.float64))


def test_obstacle_free_action_is_clipped_to_physical_bounds() -> None:
    params = ProjectionParams(v_max=1.0, omega_max=2.0)

    result = project_physical_action(
        position=np.array([0.0, 0.0], dtype=np.float64),
        heading=0.0,
        obstacle_centers=np.zeros((0, 2), dtype=np.float64),
        obstacle_radii=np.zeros(0, dtype=np.float64),
        obstacle_mask=np.zeros(0, dtype=bool),
        agent_radius=0.10,
        raw_action=np.array([2.0, -3.0], dtype=np.float64),
        params=params,
    )

    assert result.success
    assert result.solver_status == "no_active_constraints"
    assert result.intervened
    np.testing.assert_allclose(result.action_exec, np.array([1.0, -2.0]))


# Verify that the projection radius includes obstacle geometry, agent footprint, and extra clearance.
def test_projection_radius_combines_obstacle_agent_and_extra_clearance() -> None:
    params = ProjectionParams(lookahead_distance=0.0, extra_clearance=0.20)

    constraint_data = build_cbf_constraints(
        position=np.array([0.0, 0.0], dtype=np.float64),
        heading=0.0,
        obstacle_centers=np.array([[1.0, 0.0]], dtype=np.float64),
        obstacle_radii=np.array([0.50], dtype=np.float64),
        obstacle_mask=np.array([True]),
        agent_radius=0.10,
        params=params,
    )

    expected_projection_radius = 0.50 + 0.10 + 0.20
    expected_h_value = 1.0 ** 2 - expected_projection_radius ** 2

    assert constraint_data.active_indices.tolist() == [0]
    np.testing.assert_allclose(constraint_data.h_values, np.array([expected_h_value]))

#} End function test_projection_radius_combines_obstacle_agent_and_extra_clearance


def test_active_cbf_constraint_modifies_forward_action() -> None:
    pytest.importorskip("cvxpy")

    params = ProjectionParams(
        v_max=1.0,
        omega_max=2.0,
        lookahead_distance=0.25,
        alpha=2.0,
        extra_clearance=0.0,
        slack_penalty=10000.0,
        solver_name="OSQP",
    )

    result = project_physical_action(
        position=np.array([0.0, 0.0], dtype=np.float64),
        heading=0.0,
        obstacle_centers=np.array([[0.75, 0.0]], dtype=np.float64),
        obstacle_radii=np.array([0.25], dtype=np.float64),
        obstacle_mask=np.array([True]),
        agent_radius=0.0,
        raw_action=np.array([1.0, 0.0], dtype=np.float64),
        params=params,
    )

    assert result.success
    assert result.solver_status in {"optimal", "optimal_inaccurate"}
    assert result.active_constraint_count == 1
    assert result.intervened
    assert result.action_exec.shape == (2,)
    assert np.all(np.isfinite(result.action_exec))
    assert 0.0 <= result.action_exec[0] <= params.v_max
    assert -params.omega_max <= result.action_exec[1] <= params.omega_max
    assert result.action_exec[0] < 0.5
    assert result.slack_values.shape == (1,)
    assert np.all(result.slack_values >= -1.0e-7)

# Verify that a solver failure returns a stationary fail-safe action.
def test_solver_failure_returns_stop_action() -> None:
#{
    pytest.importorskip("cvxpy")

    params = ProjectionParams(solver_name="INVALID_SOLVER")

    result = project_physical_action(
        position=np.array([0.0, 0.0], dtype=np.float64),
        heading=0.0,
        obstacle_centers=np.array([[0.75, 0.0]], dtype=np.float64),
        obstacle_radii=np.array([0.25], dtype=np.float64),
        obstacle_mask=np.array([True]),
        agent_radius=0.0,
        raw_action=np.array([1.0, 0.0], dtype=np.float64),
        params=params,
    )

    assert not result.success
    assert result.solver_status == "solver_error"
    assert result.active_constraint_count == 1
    assert result.intervened
    np.testing.assert_allclose(result.action_exec, np.zeros(2, dtype=np.float64))
    assert result.slack_values.shape == (1,)
    assert np.all(np.isnan(result.slack_values))

#} End function test_solver_failure_returns_stop_action

