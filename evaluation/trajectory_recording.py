from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from environments.action_wrappers import NormalizedActionWrapper
from environments.constrained_navigation import ConstrainedNavigationEnv
from projection.cbf_qp_projection import ProjectionParams


#################################################################################
# region Data classes

# One complete evaluator trajectory with state samples and transition diagnostics.
@dataclass
class EpisodeTrajectory:
#{
    policy: str
    checkpoint: str
    episode: int
    seed: int
    goal: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray
    obstacle_mask: np.ndarray
    agent_radius: float
    dt: float
    v_max: float
    omega_max: float
    positions: list[np.ndarray] = field(default_factory=list)
    headings: list[float] = field(default_factory=list)
    state_step_count: list[int] = field(default_factory=list)
    state_distance_to_goal: list[float] = field(default_factory=list)
    state_min_obstacle_clearance: list[float] = field(default_factory=list)
    state_success: list[bool] = field(default_factory=list)
    state_collision: list[bool] = field(default_factory=list)
    action_raw_normalized: list[np.ndarray] = field(default_factory=list)
    action_raw_physical: list[np.ndarray] = field(default_factory=list)
    action_exec_physical: list[np.ndarray] = field(default_factory=list)
    action_correction_physical: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    projection_enabled: list[bool] = field(default_factory=list)
    projection_intervened: list[bool] = field(default_factory=list)
    projection_correction_norm: list[float] = field(default_factory=list)
    projection_slack_values: list[np.ndarray] = field(default_factory=list)
    projection_slack_sum: list[float] = field(default_factory=list)
    projection_slack_max: list[float] = field(default_factory=list)
    projection_success: list[bool] = field(default_factory=list)
    projection_solver_status: list[str] = field(default_factory=list)
    projection_active_constraint_count: list[int] = field(default_factory=list)
#} End dataclass EpisodeTrajectory

# end region Data classes


#################################################################################
# region Helpers

# Return the project-specific base environment used by the evaluator.
def _get_base_env(env: gym.Env) -> ConstrainedNavigationEnv:
#{
    base_env = env.unwrapped

    if not isinstance(base_env, ConstrainedNavigationEnv):
        raise TypeError("Trajectory recording requires ConstrainedNavigationEnv.")

    return base_env

#} End function _get_base_env


# Return a finite two-element action vector.
def _as_action(name: str, action: Any) -> np.ndarray:
#{
    action_array = np.asarray(action, dtype=np.float64).reshape(2)

    if not np.all(np.isfinite(action_array)):
        raise RuntimeError(f"{name} contains non-finite values: {action_array}")

    return action_array

#} End function _as_action


# Convert one evaluator action from normalized policy coordinates to physical coordinates.
def physical_action_from_policy_action(env: gym.Env, action: Any) -> np.ndarray:
#{
    if not isinstance(env, NormalizedActionWrapper):
        raise TypeError(
            "Trajectory recording requires NormalizedActionWrapper as the evaluator outer wrapper."
        )

    physical_action = env.action(np.asarray(action, dtype=np.float32).reshape(2))
    return _as_action("Physical raw action", physical_action)

#} End function physical_action_from_policy_action


# Validate state and transition list lengths before archive serialization.
def _validate_episode_trajectory(trajectory: EpisodeTrajectory) -> None:
#{
    transition_count = len(trajectory.rewards)
    state_count = transition_count + 1

    state_fields = {
        "positions": trajectory.positions,
        "headings": trajectory.headings,
        "state_step_count": trajectory.state_step_count,
        "state_distance_to_goal": trajectory.state_distance_to_goal,
        "state_min_obstacle_clearance": trajectory.state_min_obstacle_clearance,
        "state_success": trajectory.state_success,
        "state_collision": trajectory.state_collision,
    }
    transition_fields = {
        "action_raw_normalized": trajectory.action_raw_normalized,
        "action_raw_physical": trajectory.action_raw_physical,
        "action_exec_physical": trajectory.action_exec_physical,
        "action_correction_physical": trajectory.action_correction_physical,
        "terminated": trajectory.terminated,
        "truncated": trajectory.truncated,
        "projection_enabled": trajectory.projection_enabled,
        "projection_intervened": trajectory.projection_intervened,
        "projection_correction_norm": trajectory.projection_correction_norm,
        "projection_slack_values": trajectory.projection_slack_values,
        "projection_slack_sum": trajectory.projection_slack_sum,
        "projection_slack_max": trajectory.projection_slack_max,
        "projection_success": trajectory.projection_success,
        "projection_solver_status": trajectory.projection_solver_status,
        "projection_active_constraint_count": trajectory.projection_active_constraint_count,
    }

    for field_name, values in state_fields.items():
        if len(values) != state_count:
            raise RuntimeError(
                f"Trajectory state field {field_name} has {len(values)} values; expected {state_count}."
            )

    for field_name, values in transition_fields.items():
        if len(values) != transition_count:
            raise RuntimeError(
                f"Trajectory transition field {field_name} has {len(values)} values; "
                f"expected {transition_count}."
            )

#} End function _validate_episode_trajectory


# Add one scalar run-metadata value to an NPZ archive dictionary.
def _add_run_metadata(arrays: dict[str, np.ndarray], name: str, value: object) -> None:
#{
    if value is None:
        return

    if isinstance(value, Path):
        value = str(value)

    if not isinstance(value, (str, bool, int, float, np.generic)):
        raise TypeError(f"Unsupported trajectory metadata type for {name}: {type(value).__name__}")

    arrays[f"run_{name}"] = np.asarray(value)

#} End function _add_run_metadata

# end region Helpers


#################################################################################
# region Interface functions

# Initialize one trajectory immediately after env.reset(...).
def create_episode_trajectory(
    *,
    env: gym.Env,
    initial_info: dict[str, Any],
    policy_name: str,
    checkpoint_path: str,
    episode: int,
    seed: int,
) -> EpisodeTrajectory:
#{
    if not isinstance(env, NormalizedActionWrapper):
        raise TypeError(
            "Trajectory recording requires NormalizedActionWrapper as the evaluator outer wrapper."
        )

    base_env = _get_base_env(env)

    return EpisodeTrajectory(
        policy=str(policy_name),
        checkpoint=str(checkpoint_path),
        episode=int(episode),
        seed=int(seed),
        goal=base_env.goal.astype(np.float64, copy=True),
        obstacle_centers=base_env.obstacle_centers.astype(np.float64, copy=True),
        obstacle_radii=base_env.obstacle_radii.astype(np.float64, copy=True),
        obstacle_mask=base_env.obstacle_mask.astype(bool, copy=True),
        agent_radius=float(base_env.agent_radius),
        dt=float(base_env.dt),
        v_max=float(base_env.v_max),
        omega_max=float(base_env.omega_max),
        positions=[base_env.position.astype(np.float64, copy=True)],
        headings=[float(base_env.theta)],
        state_step_count=[int(initial_info.get("step_count", base_env.step_count))],
        state_distance_to_goal=[float(initial_info.get("distance_to_goal", np.inf))],
        state_min_obstacle_clearance=[
            float(initial_info.get("min_obstacle_clearance", np.nan))
        ],
        state_success=[bool(initial_info.get("success", False))],
        state_collision=[bool(initial_info.get("collision", False))],
    )

#} End function create_episode_trajectory


# Append one executed transition and its post-step state to a trajectory.
def append_episode_transition(
    *,
    trajectory: EpisodeTrajectory,
    env: gym.Env,
    action_raw_normalized: Any,
    action_raw_physical: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> None:
#{
    base_env = _get_base_env(env)
    normalized_action = _as_action("Normalized raw action", action_raw_normalized)
    physical_raw_action = _as_action("Physical raw action", action_raw_physical)
    projection_enabled = bool(info.get("projection_enabled", False))

    if projection_enabled:
    #{
        reported_raw_action = _as_action("Projection raw action", info["projection_action_raw"])
        physical_exec_action = _as_action("Projection executed action", info["projection_action_exec"])
        correction = _as_action("Projection correction", info["projection_correction"])
        slack_values = np.asarray(info["projection_slack_values"], dtype=np.float64).reshape(
            base_env.max_obstacles
        )
        projection_intervened = bool(info["projection_intervened"])
        correction_norm = float(info["projection_correction_norm"])
        slack_sum = float(info["projection_slack_sum"])
        slack_max = float(info["projection_slack_max"])
        projection_success = bool(info["projection_success"])
        solver_status = str(info["projection_solver_status"])
        active_constraint_count = int(info["projection_active_constraint_count"])

        if not np.allclose(reported_raw_action, physical_raw_action, rtol=1.0e-6, atol=1.0e-7):
            raise RuntimeError(
                "Projection raw action does not match the normalized-wrapper physical action."
            )

        if active_constraint_count != int(np.count_nonzero(base_env.obstacle_mask)):
            raise RuntimeError(
                "Projection active-constraint count does not match the environment obstacle mask."
            )

        active_slack_values = slack_values[base_env.obstacle_mask]

        if projection_success:
        #{
            if not np.all(np.isfinite(slack_values)):
                raise RuntimeError(
                    "Successful projection reported non-finite per-obstacle slack values."
                )

            expected_slack_sum = float(np.sum(active_slack_values))
            expected_slack_max = (
                float(np.max(active_slack_values))
                if active_slack_values.size > 0
                else 0.0
            )

            if not np.isclose(slack_sum, expected_slack_sum, rtol=1.0e-7, atol=1.0e-9):
                raise RuntimeError(
                    "Projection slack sum does not match the slot-aligned slack vector."
                )

            if not np.isclose(slack_max, expected_slack_max, rtol=1.0e-7, atol=1.0e-9):
                raise RuntimeError(
                    "Projection maximum slack does not match the slot-aligned slack vector."
                )

        #} End if projection_success
        else:
        #{
            if not np.isnan(slack_sum) or not np.isnan(slack_max):
                raise RuntimeError(
                    "Failed projection must report unknown summed and maximum slack."
                )

            if active_slack_values.size > 0 and not np.all(np.isnan(active_slack_values)):
                raise RuntimeError(
                    "Failed projection must preserve unknown slack for every active obstacle."
                )

        #} End else projection_failure

    #} End if projection_enabled
    else:
    #{
        physical_exec_action = physical_raw_action.copy()
        correction = np.zeros(2, dtype=np.float64)
        slack_values = np.zeros(base_env.max_obstacles, dtype=np.float64)
        projection_intervened = False
        correction_norm = 0.0
        slack_sum = 0.0
        slack_max = 0.0
        projection_success = True
        solver_status = "disabled"
        active_constraint_count = 0

    #} End else projection_disabled

    expected_correction = physical_exec_action - physical_raw_action

    if not np.allclose(correction, expected_correction, rtol=1.0e-7, atol=1.0e-9):
        raise RuntimeError("Projection correction does not equal action_exec - action_raw.")

    expected_correction_norm = float(np.linalg.norm(correction, ord=2))

    if not np.isclose(correction_norm, expected_correction_norm, rtol=1.0e-7, atol=1.0e-9):
        raise RuntimeError("Projection correction norm does not match the correction vector.")

    trajectory.action_raw_normalized.append(normalized_action.astype(np.float32, copy=False))
    trajectory.action_raw_physical.append(physical_raw_action)
    trajectory.action_exec_physical.append(physical_exec_action)
    trajectory.action_correction_physical.append(correction)
    trajectory.rewards.append(float(reward))
    trajectory.terminated.append(bool(terminated))
    trajectory.truncated.append(bool(truncated))
    trajectory.projection_enabled.append(projection_enabled)
    trajectory.projection_intervened.append(projection_intervened)
    trajectory.projection_correction_norm.append(correction_norm)
    trajectory.projection_slack_values.append(slack_values)
    trajectory.projection_slack_sum.append(slack_sum)
    trajectory.projection_slack_max.append(slack_max)
    trajectory.projection_success.append(projection_success)
    trajectory.projection_solver_status.append(solver_status)
    trajectory.projection_active_constraint_count.append(active_constraint_count)

    trajectory.positions.append(base_env.position.astype(np.float64, copy=True))
    trajectory.headings.append(float(base_env.theta))
    trajectory.state_step_count.append(int(info.get("step_count", base_env.step_count)))
    trajectory.state_distance_to_goal.append(float(info.get("distance_to_goal", np.inf)))
    trajectory.state_min_obstacle_clearance.append(
        float(info.get("min_obstacle_clearance", np.nan))
    )
    trajectory.state_success.append(bool(info.get("success", False)))
    trajectory.state_collision.append(bool(info.get("collision", False)))

#} End function append_episode_transition


# Write one compressed NPZ archive containing all recorded episodes.
def write_trajectory_archive(
    *,
    trajectories: list[EpisodeTrajectory],
    output_path: str | Path,
    projection_params: ProjectionParams,
    run_metadata: dict[str, object] | None = None,
) -> None:
#{
    if not trajectories:
        raise ValueError("Cannot write an empty trajectory archive.")

    path = Path(output_path)

    if path.suffix.lower() != ".npz":
        raise ValueError("Trajectory output path must use the .npz extension.")

    path.parent.mkdir(parents=True, exist_ok=True)

    episode_keys = [f"episode_{index:04d}" for index in range(len(trajectories))]
    arrays: dict[str, np.ndarray] = {
        "trajectory_archive_version": np.asarray("evaluation_trajectory_v1"),
        "episode_count": np.asarray(len(trajectories), dtype=np.int64),
        "episode_keys": np.asarray(episode_keys, dtype=np.str_),
        "projection_v_max": np.asarray(projection_params.v_max, dtype=np.float64),
        "projection_omega_max": np.asarray(projection_params.omega_max, dtype=np.float64),
        "projection_lookahead_distance": np.asarray(
            projection_params.lookahead_distance,
            dtype=np.float64,
        ),
        "projection_alpha": np.asarray(projection_params.alpha, dtype=np.float64),
        "projection_extra_clearance": np.asarray(
            projection_params.extra_clearance,
            dtype=np.float64,
        ),
        "projection_slack_penalty": np.asarray(
            projection_params.slack_penalty,
            dtype=np.float64,
        ),
        "projection_correction_tolerance": np.asarray(
            projection_params.correction_tolerance,
            dtype=np.float64,
        ),
        "projection_solver_name": np.asarray(projection_params.solver_name),
    }

    if run_metadata is not None:
        for name, value in run_metadata.items():
            _add_run_metadata(arrays, name, value)

    seen_episode_keys: set[tuple[int, int]] = set()

    for archive_key, trajectory in zip(episode_keys, trajectories):
    #{
        _validate_episode_trajectory(trajectory)
        result_key = (trajectory.episode, trajectory.seed)

        if result_key in seen_episode_keys:
            raise ValueError(f"Duplicate trajectory episode key: {result_key}")

        seen_episode_keys.add(result_key)
        transition_count = len(trajectory.rewards)
        obstacle_count = int(trajectory.obstacle_mask.shape[0])

        arrays[f"{archive_key}_policy"] = np.asarray(trajectory.policy)
        arrays[f"{archive_key}_checkpoint"] = np.asarray(trajectory.checkpoint)
        arrays[f"{archive_key}_episode"] = np.asarray(trajectory.episode, dtype=np.int64)
        arrays[f"{archive_key}_seed"] = np.asarray(trajectory.seed, dtype=np.int64)
        arrays[f"{archive_key}_goal"] = np.asarray(trajectory.goal, dtype=np.float64).reshape(2)
        arrays[f"{archive_key}_obstacle_centers"] = np.asarray(
            trajectory.obstacle_centers,
            dtype=np.float64,
        ).reshape(obstacle_count, 2)
        arrays[f"{archive_key}_obstacle_radii"] = np.asarray(
            trajectory.obstacle_radii,
            dtype=np.float64,
        ).reshape(obstacle_count)
        arrays[f"{archive_key}_obstacle_mask"] = np.asarray(
            trajectory.obstacle_mask,
            dtype=bool,
        ).reshape(obstacle_count)
        arrays[f"{archive_key}_agent_radius"] = np.asarray(
            trajectory.agent_radius,
            dtype=np.float64,
        )
        arrays[f"{archive_key}_dt"] = np.asarray(trajectory.dt, dtype=np.float64)
        arrays[f"{archive_key}_v_max"] = np.asarray(trajectory.v_max, dtype=np.float64)
        arrays[f"{archive_key}_omega_max"] = np.asarray(
            trajectory.omega_max,
            dtype=np.float64,
        )
        arrays[f"{archive_key}_positions"] = np.asarray(
            trajectory.positions,
            dtype=np.float64,
        ).reshape(transition_count + 1, 2)
        arrays[f"{archive_key}_headings"] = np.asarray(
            trajectory.headings,
            dtype=np.float64,
        ).reshape(transition_count + 1)
        arrays[f"{archive_key}_state_step_count"] = np.asarray(
            trajectory.state_step_count,
            dtype=np.int64,
        ).reshape(transition_count + 1)
        arrays[f"{archive_key}_state_distance_to_goal"] = np.asarray(
            trajectory.state_distance_to_goal,
            dtype=np.float64,
        ).reshape(transition_count + 1)
        arrays[f"{archive_key}_state_min_obstacle_clearance"] = np.asarray(
            trajectory.state_min_obstacle_clearance,
            dtype=np.float64,
        ).reshape(transition_count + 1)
        arrays[f"{archive_key}_state_success"] = np.asarray(
            trajectory.state_success,
            dtype=bool,
        ).reshape(transition_count + 1)
        arrays[f"{archive_key}_state_collision"] = np.asarray(
            trajectory.state_collision,
            dtype=bool,
        ).reshape(transition_count + 1)
        arrays[f"{archive_key}_action_raw_normalized"] = np.asarray(
            trajectory.action_raw_normalized,
            dtype=np.float32,
        ).reshape(transition_count, 2)
        arrays[f"{archive_key}_action_raw_physical"] = np.asarray(
            trajectory.action_raw_physical,
            dtype=np.float64,
        ).reshape(transition_count, 2)
        arrays[f"{archive_key}_action_exec_physical"] = np.asarray(
            trajectory.action_exec_physical,
            dtype=np.float64,
        ).reshape(transition_count, 2)
        arrays[f"{archive_key}_action_correction_physical"] = np.asarray(
            trajectory.action_correction_physical,
            dtype=np.float64,
        ).reshape(transition_count, 2)
        arrays[f"{archive_key}_rewards"] = np.asarray(
            trajectory.rewards,
            dtype=np.float64,
        ).reshape(transition_count)
        arrays[f"{archive_key}_terminated"] = np.asarray(
            trajectory.terminated,
            dtype=bool,
        ).reshape(transition_count)
        arrays[f"{archive_key}_truncated"] = np.asarray(
            trajectory.truncated,
            dtype=bool,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_enabled"] = np.asarray(
            trajectory.projection_enabled,
            dtype=bool,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_intervened"] = np.asarray(
            trajectory.projection_intervened,
            dtype=bool,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_correction_norm"] = np.asarray(
            trajectory.projection_correction_norm,
            dtype=np.float64,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_slack_values"] = np.asarray(
            trajectory.projection_slack_values,
            dtype=np.float64,
        ).reshape(transition_count, obstacle_count)
        arrays[f"{archive_key}_projection_slack_sum"] = np.asarray(
            trajectory.projection_slack_sum,
            dtype=np.float64,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_slack_max"] = np.asarray(
            trajectory.projection_slack_max,
            dtype=np.float64,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_success"] = np.asarray(
            trajectory.projection_success,
            dtype=bool,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_solver_status"] = np.asarray(
            trajectory.projection_solver_status,
            dtype=np.str_,
        ).reshape(transition_count)
        arrays[f"{archive_key}_projection_active_constraint_count"] = np.asarray(
            trajectory.projection_active_constraint_count,
            dtype=np.int64,
        ).reshape(transition_count)

    #} End loop trajectories

    np.savez_compressed(path, **arrays)

#} End function write_trajectory_archive

# end region Interface functions


######### Module exports
__all__ = [
    "EpisodeTrajectory",
    "append_episode_transition",
    "create_episode_trajectory",
    "physical_action_from_policy_action",
    "write_trajectory_archive",
]
