from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from environments.constrained_navigation import ConstrainedNavigationEnv
from projection.cbf_qp_projection import ProjectionParams, ProjectionResult, project_physical_action


# Gymnasium wrapper that projects physical actions before environment execution.
class CbfQpProjectionWrapper(gym.Wrapper):
#{
    # Initialize the physical-action projection wrapper.
    def __init__(self, env: ConstrainedNavigationEnv, params: ProjectionParams | None = None) -> None:
    #{
        # Store the wrapped physical environment.
        super().__init__(env)
        self.base_env = env

        # Use the physical action bounds defined by the environment.
        # The agent footprint is supplied separately on every projection call.
        if params is None:
            params = ProjectionParams(v_max=env.v_max, omega_max=env.omega_max)
        else:
            params = replace(params, v_max=env.v_max, omega_max=env.omega_max)

        self.params = params

    #} End function __init__

    # Project and execute one physical action.
    def step(self, action: np.ndarray):
    #{
        # The wrapper receives the raw physical action.
        raw_action = action

        # The numerical projector computes u_exec from u_raw using the current environment state.
        result = project_physical_action(
            position=self.base_env.position,
            heading=self.base_env.theta,
            obstacle_centers=self.base_env.obstacle_centers,
            obstacle_radii=self.base_env.obstacle_radii,
            obstacle_mask=self.base_env.obstacle_mask,
            agent_radius=self.base_env.agent_radius,
            raw_action=raw_action,
            params=self.params,
        )

        # Environment transition.
        # self.base_env is the wrapped ConstrainedNavigationEnv.
        # This calls ConstrainedNavigationEnv.step(u_exec).
        observation, reward, terminated, truncated, info = self.base_env.step(result.action_exec)

        # The wrapper adds projection diagnostics after the environment transition into the info dictionary.
        info = self._add_projection_info(info=info, result=result)

        return observation, reward, terminated, truncated, info

    #} End function step

    # Attach scalar projection diagnostics to the environment info dictionary.
    def _add_projection_info(self, info: dict[str, Any], result: ProjectionResult) -> dict[str, Any]:
    #{
        updated_info = dict(info)
        updated_info["projection_enabled"] = True
        updated_info["projection_intervened"] = bool(result.intervened)
        updated_info["projection_correction_norm"] = float(result.correction_norm)
        updated_info["projection_slack_max"] = float(result.slack_max)
        updated_info["projection_slack_sum"] = float(result.slack_sum)
        updated_info["projection_success"] = bool(result.success)
        updated_info["projection_solver_status"] = str(result.solver_status)
        updated_info["projection_active_constraint_count"] = int(result.active_constraint_count)

        return updated_info

    #} End function _add_projection_info

#} End class CbfQpProjectionWrapper


######### Module exports
__all__ = ["CbfQpProjectionWrapper"]