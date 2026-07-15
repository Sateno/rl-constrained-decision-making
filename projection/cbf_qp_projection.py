from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#############################################################################
# region Data classes

# Numerical parameters for the physical-action projection problem.
@dataclass(frozen=True)
class ProjectionParams:
#{
    v_max: float = 1.0
    omega_max: float = 2.0
    lookahead_distance: float = 0.25
    alpha: float = 2.0
    safety_margin: float = 0.0
    slack_penalty: float = 1000.0
    correction_tolerance: float = 1.0e-6
    solver_name: str = "OSQP"
    solver_verbose: bool = False

    def __post_init__(self) -> None:
        numeric_values = {
            "v_max": self.v_max,
            "omega_max": self.omega_max,
            "lookahead_distance": self.lookahead_distance,
            "alpha": self.alpha,
            "safety_margin": self.safety_margin,
            "slack_penalty": self.slack_penalty,
            "correction_tolerance": self.correction_tolerance,
        }

        for name, value in numeric_values.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.v_max <= 0.0:
            raise ValueError("v_max must be positive.")
        if self.omega_max <= 0.0:
            raise ValueError("omega_max must be positive.")
        if self.lookahead_distance < 0.0:
            raise ValueError("lookahead_distance must be nonnegative.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be nonnegative.")
        if self.slack_penalty <= 0.0:
            raise ValueError("slack_penalty must be positive.")
        if self.correction_tolerance < 0.0:
            raise ValueError("correction_tolerance must be nonnegative.")
        if not isinstance(self.solver_name, str) or not self.solver_name:
            raise ValueError("solver_name must be a nonempty string.")

#} End dataclass ProjectionParams

# Affine CBF constraint data for active obstacles.
@dataclass(frozen=True)
class CbfConstraintData:
#{
    A_cbf: FloatArray
    b_cbf: FloatArray
    h_values: FloatArray
    active_indices: IntArray

#} End dataclass CbfConstraintData


# Physical-action projection result and diagnostics.
@dataclass(frozen=True)
class ProjectionResult:
#{
    action_raw: FloatArray
    action_exec: FloatArray
    correction: FloatArray
    correction_norm: float
    intervened: bool
    slack_values: FloatArray
    slack_max: float
    slack_sum: float
    active_constraint_count: int
    solver_status: str
    success: bool
    objective_value: Optional[float] = None

#} End dataclass ProjectionResult

# end region

##############################################################################
# region Helpers

# Return a 1D array with finite values.
def _as_vector(name: str, value: ArrayLike, length: int) -> FloatArray:
#{
    array = np.asarray(value, dtype=np.float64)
    if array.size != length:
        raise ValueError(f"{name} must contain {length} elements.")
    array = array.reshape(length)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array

#} End function _as_vector

# Return an array of obstacle centers with shape (N, 2)
def _as_obstacle_centers(value: ArrayLike) -> FloatArray:
 #{
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        array = array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("obstacle_centers must have shape (N, 2).")
    if not np.all(np.isfinite(array)):
        raise ValueError("obstacle_centers must contain only finite values.")
    return array

#} End function _as_obstacle_centers

# Return an array of obstacle radii with shape (N,)
def _as_obstacle_radii(value: ArrayLike, expected_count: int) -> FloatArray:
#{
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        array = array.reshape(0)
    if array.ndim != 1 or array.shape[0] != expected_count:
        raise ValueError("obstacle_radii must have shape (N,).")
    if not np.all(np.isfinite(array)):
        raise ValueError("obstacle_radii must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError("obstacle_radii must be nonnegative.")
    return array

#} End function _as_obstacle_radii

# Return a boolean mask array with shape (N,) indicating active obstacles
def _as_obstacle_mask(value: ArrayLike, expected_count: int) -> NDArray[np.bool_]:
#{
    array = np.asarray(value, dtype=bool)
    if array.size == 0:
        array = array.reshape(0)
    if array.ndim != 1 or array.shape[0] != expected_count:
        raise ValueError("obstacle_mask must have shape (N,).")
    return array

#} End function _as_obstacle_mask

# Create results for a physical-action projection problem.
def _make_result(
    *,
    action_raw: FloatArray,
    action_exec: FloatArray,
    slack_values: FloatArray,
    active_constraint_count: int,
    solver_status: str,
    success: bool,
    objective_value: Optional[float],
    params: ProjectionParams,
) -> ProjectionResult:
#{
    correction = action_exec - action_raw
    correction_norm = float(np.linalg.norm(correction, ord=2))
    intervened = bool(correction_norm > params.correction_tolerance)

    if slack_values.size == 0:
        slack_max = 0.0
        slack_sum = 0.0
    elif np.all(np.isfinite(slack_values)):
        slack_max = float(np.max(slack_values))
        slack_sum = float(np.sum(slack_values))
    else:
        slack_max = float("nan")
        slack_sum = float("nan")

    return ProjectionResult(
        action_raw=action_raw.astype(np.float64, copy=False),
        action_exec=action_exec.astype(np.float64, copy=False),
        correction=correction.astype(np.float64, copy=False),
        correction_norm=correction_norm,
        intervened=intervened,
        slack_values=slack_values.astype(np.float64, copy=False),
        slack_max=slack_max,
        slack_sum=slack_sum,
        active_constraint_count=int(active_constraint_count),
        solver_status=str(solver_status),
        success=bool(success),
        objective_value=objective_value,
    )

#} End function _make_result

# end region

#########################################################################################
# region Interface functions

# Compute the lookahead point p_L = p + L [cos(phi), sin(phi)].
def compute_lookahead_point(position: ArrayLike, heading: float, lookahead_distance: float) -> FloatArray:
#{
    position_array = _as_vector("position", position, 2)
    heading_float = float(heading)
    lookahead_float = float(lookahead_distance)

    if not np.isfinite(heading_float):
        raise ValueError("heading must be finite.")
    if not np.isfinite(lookahead_float):
        raise ValueError("lookahead_distance must be finite.")
    if lookahead_float < 0.0:
        raise ValueError("lookahead_distance must be nonnegative.")

    direction = np.asarray([np.cos(heading_float), np.sin(heading_float)], dtype=np.float64)

    return position_array + lookahead_float * direction

#} End function compute_lookahead_point

# Compute the Jacobian mapping [v, omega] to lookahead-point velocity.
def compute_lookahead_jacobian(heading: float, lookahead_distance: float) -> FloatArray:
#{
    heading_float = float(heading)
    lookahead_float = float(lookahead_distance)

    if not np.isfinite(heading_float):
        raise ValueError("heading must be finite.")
    if not np.isfinite(lookahead_float):
        raise ValueError("lookahead_distance must be finite.")
    if lookahead_float < 0.0:
        raise ValueError("lookahead_distance must be nonnegative.")

    c = float(np.cos(heading_float))
    s = float(np.sin(heading_float))
    L = lookahead_float

    return np.asarray(
        [
            [c, -L * s],
            [s, L * c],
        ],
        dtype=np.float64,
    )

#} End function compute_lookahead_jacobian

# Build affine CBF constraints A_cbf u + xi >= b_cbf for active obstacles.
def build_cbf_constraints(
    position: ArrayLike,
    heading: float,
    obstacle_centers: ArrayLike,
    obstacle_radii: ArrayLike,
    obstacle_mask: ArrayLike,
    params: ProjectionParams,
) -> CbfConstraintData:
#{
    position_array = _as_vector("position", position, 2)
    heading_float = float(heading)
    if not np.isfinite(heading_float):
        raise ValueError("heading must be finite.")

    centers = _as_obstacle_centers(obstacle_centers)
    radii = _as_obstacle_radii(obstacle_radii, centers.shape[0])
    mask = _as_obstacle_mask(obstacle_mask, centers.shape[0])

    active_indices = np.flatnonzero(mask).astype(np.int64)
    if active_indices.size == 0:
        return CbfConstraintData(
            A_cbf=np.zeros((0, 2), dtype=np.float64),
            b_cbf=np.zeros(0, dtype=np.float64),
            h_values=np.zeros(0, dtype=np.float64),
            active_indices=active_indices,
        )

    p_lookahead = compute_lookahead_point(position_array, heading_float, params.lookahead_distance)
    J_lookahead = compute_lookahead_jacobian(heading_float, params.lookahead_distance)

    A_rows: list[FloatArray] = []
    b_values: list[float] = []
    h_values: list[float] = []

    for index in active_indices:
        obstacle_center = centers[index]
        effective_radius = float(radii[index] + params.safety_margin)
        displacement = p_lookahead - obstacle_center
        h_value = float(displacement @ displacement - effective_radius * effective_radius)
        grad_h = 2.0 * displacement
        a_row = grad_h @ J_lookahead
        b_value = -params.alpha * h_value

        A_rows.append(np.asarray(a_row, dtype=np.float64).reshape(2))
        b_values.append(float(b_value))
        h_values.append(float(h_value))

    return CbfConstraintData(
        A_cbf=np.vstack(A_rows).astype(np.float64, copy=False),
        b_cbf=np.asarray(b_values, dtype=np.float64),
        h_values=np.asarray(h_values, dtype=np.float64),
        active_indices=active_indices,
    )

#} End function build_cbf_constraints


# Clip a physical action to [0, v_max] x [-omega_max, omega_max].
def clip_physical_action(action: ArrayLike, params: ProjectionParams) -> FloatArray:
#{
    action_array = _as_vector("action", action, 2)
    low = np.asarray([0.0, -params.omega_max], dtype=np.float64)
    high = np.asarray([params.v_max, params.omega_max], dtype=np.float64)

    return np.clip(action_array, low, high).astype(np.float64, copy=False)

#} End function clip_physical_action

# Solve the CBF-QP projection problem for one physical action.
def solve_projection_qp(raw_action: ArrayLike, constraint_data: CbfConstraintData, params: ProjectionParams) -> ProjectionResult:
#{
    action_raw = _as_vector("raw_action", raw_action, 2)
    active_constraint_count = int(constraint_data.A_cbf.shape[0])

    # Validate
    if constraint_data.A_cbf.shape != (active_constraint_count, 2):
        raise ValueError("A_cbf must have shape (M, 2).")
    if constraint_data.b_cbf.shape != (active_constraint_count,):
        raise ValueError("b_cbf must have shape (M,).")
    if constraint_data.h_values.shape != (active_constraint_count,):
        raise ValueError("h_values must have shape (M,).")
    if constraint_data.active_indices.shape != (active_constraint_count,):
        raise ValueError("active_indices must have shape (M,).")
    if not np.all(np.isfinite(constraint_data.A_cbf)):
        raise ValueError("A_cbf must contain only finite values.")
    if not np.all(np.isfinite(constraint_data.b_cbf)):
        raise ValueError("b_cbf must contain only finite values.")
    if not np.all(np.isfinite(constraint_data.h_values)):
        raise ValueError("h_values must contain only finite values.")

    if active_constraint_count == 0:
        action_exec = clip_physical_action(action_raw, params)
        objective_value = 0.5 * float(np.sum((action_exec - action_raw) ** 2))
        return _make_result(
            action_raw=action_raw,
            action_exec=action_exec,
            slack_values=np.zeros(0, dtype=np.float64),
            active_constraint_count=0,
            solver_status="no_active_constraints",
            success=True,
            objective_value=objective_value,
            params=params,
        )

    # Use a stationary action whenever the projection solver cannot return a valid solution.
    stop_action = np.zeros(2, dtype=np.float64)

    try:
        import cvxpy as cp
    except ImportError:
        action_exec = stop_action
        return _make_result(
            action_raw=action_raw,
            action_exec=action_exec,
            slack_values=np.full(active_constraint_count, np.nan, dtype=np.float64),
            active_constraint_count=active_constraint_count,
            solver_status="cvxpy_not_installed",
            success=False,
            objective_value=None,
            params=params,
        )

    # Solve the QP using cvxpy
    u = cp.Variable(2)
    xi = cp.Variable(active_constraint_count)

    objective = cp.Minimize(0.5 * cp.sum_squares(u - action_raw) + 0.5 * params.slack_penalty * cp.sum_squares(xi))

    constraints = [
        u[0] >= 0.0,
        u[0] <= params.v_max,
        u[1] >= -params.omega_max,
        u[1] <= params.omega_max,
        constraint_data.A_cbf @ u + xi >= constraint_data.b_cbf,
        xi >= 0.0,
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=params.solver_name, verbose=params.solver_verbose, warm_start=True)

    except Exception:
        action_exec = stop_action
        return _make_result(
            action_raw=action_raw,
            action_exec=action_exec,
            slack_values=np.full(active_constraint_count, np.nan, dtype=np.float64),
            active_constraint_count=active_constraint_count,
            solver_status="solver_error",
            success=False,
            objective_value=None,
            params=params,
        )

    solver_status = str(problem.status)
    status_success = solver_status in {"optimal", "optimal_inaccurate"}

    if status_success and u.value is not None and xi.value is not None:
        action_exec = np.asarray(u.value, dtype=np.float64).reshape(2)
        action_exec = clip_physical_action(action_exec, params)
        slack_values = np.asarray(xi.value, dtype=np.float64).reshape(active_constraint_count)
        slack_values = np.maximum(slack_values, 0.0)
        objective_value = None if problem.value is None else float(problem.value)

        if not np.all(np.isfinite(action_exec)) or not np.all(np.isfinite(slack_values)):
            action_exec = stop_action
            slack_values = np.full(active_constraint_count, np.nan, dtype=np.float64)
            solver_status = "nonfinite_solution"
            status_success = False
            objective_value = None
    else:
        action_exec = stop_action
        slack_values = np.full(active_constraint_count, np.nan, dtype=np.float64)
        objective_value = None

    return _make_result(
        action_raw=action_raw,
        action_exec=action_exec,
        slack_values=slack_values,
        active_constraint_count=active_constraint_count,
        solver_status=solver_status,
        success=status_success,
        objective_value=objective_value,
        params=params,
    )

#} End function solve_projection_qp

# Project a raw physical action through the CBF-QP safety filter.
def project_physical_action(
    position: ArrayLike,
    heading: float,
    obstacle_centers: ArrayLike,
    obstacle_radii: ArrayLike,
    obstacle_mask: ArrayLike,
    raw_action: ArrayLike,
    params: ProjectionParams | None = None,
) -> ProjectionResult:
#{    
    projection_params = ProjectionParams() if params is None else params
    constraint_data = build_cbf_constraints(
        position=position,
        heading=heading,
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        obstacle_mask=obstacle_mask,
        params=projection_params,
    )
    return solve_projection_qp(raw_action=raw_action, constraint_data=constraint_data, params=projection_params)

#} End function project_physical_action

# end region

######### Module exports
__all__ = [
    "ProjectionParams",
    "CbfConstraintData",
    "ProjectionResult",
    "compute_lookahead_point",
    "compute_lookahead_jacobian",
    "build_cbf_constraints",
    "clip_physical_action",
    "solve_projection_qp",
    "project_physical_action",
]
