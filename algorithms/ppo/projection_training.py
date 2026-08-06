from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Accumulate projection diagnostics for one PPO rollout iteration.
@dataclass
class ProjectionTrainingDiagnostics:
#{
    transition_count: int = 0
    intervention_count: int = 0
    correction_sum: float = 0.0
    correction_max: float = 0.0
    slack_sum: float = 0.0
    slack_max: float = 0.0
    solver_failure_count: int = 0

    # Consume one vectorized projection-info mapping.
    def _update_vector(self, infos: dict, num_envs: int) -> None:
    #{
        if "projection_success" not in infos:
            return

        projection_mask = np.asarray(
            infos.get("_projection_success", np.ones(num_envs, dtype=bool)),
            dtype=bool,
        )
        projection_success = np.asarray(infos["projection_success"], dtype=bool)
        failed_indices = np.flatnonzero(projection_mask & ~projection_success)

        if failed_indices.size:
            self.solver_failure_count += int(failed_indices.size)
            solver_status = np.asarray(infos["projection_solver_status"], dtype=object)
            failure_details = ", ".join(
                f"env {index}: {solver_status[index]}" for index in failed_indices
            )
            raise RuntimeError(
                f"Projection solve failed during training ({failure_details})."
            )

        valid_indices = np.flatnonzero(projection_mask)

        if not valid_indices.size:
            return

        correction_norms = np.asarray(
            infos["projection_correction_norm"],
            dtype=np.float64,
        )[valid_indices]
        step_slack_sums = np.asarray(
            infos["projection_slack_sum"],
            dtype=np.float64,
        )[valid_indices]
        step_slack_maxima = np.asarray(
            infos["projection_slack_max"],
            dtype=np.float64,
        )[valid_indices]

        if not (
            np.all(np.isfinite(correction_norms))
            and np.all(np.isfinite(step_slack_sums))
            and np.all(np.isfinite(step_slack_maxima))
        ):
            raise RuntimeError("Successful projection steps reported non-finite diagnostics.")

        self.transition_count += int(valid_indices.size)
        self.intervention_count += int(
            np.sum(np.asarray(infos["projection_intervened"], dtype=bool)[valid_indices])
        )
        self.correction_sum += float(np.sum(correction_norms))
        self.correction_max = max(
            self.correction_max,
            float(np.max(correction_norms)),
        )
        self.slack_sum += float(np.sum(step_slack_sums))
        self.slack_max = max(self.slack_max, float(np.max(step_slack_maxima)))
    #} End function _update_vector

    # Consume ordinary and same-step terminal projection diagnostics.
    def update(self, infos: dict, num_envs: int) -> None:
    #{
        self._update_vector(infos, num_envs)

        final_info = infos.get("final_info")

        if final_info is not None:
            self._update_vector(final_info, num_envs)
    #} End function update

    # Write rollout-level projection burden to TensorBoard.
    def write_tensorboard(self, writer: Any, global_step: int) -> None:
    #{
        if self.transition_count == 0:
            return

        writer.add_scalar(
            "projection/transition_count",
            self.transition_count,
            global_step,
        )
        writer.add_scalar(
            "projection/intervention_count",
            self.intervention_count,
            global_step,
        )
        writer.add_scalar(
            "projection/intervention_frequency",
            self.intervention_count / self.transition_count,
            global_step,
        )
        writer.add_scalar(
            "projection/correction_norm",
            self.correction_sum / self.transition_count,
            global_step,
        )
        writer.add_scalar(
            "projection/correction_norm_max",
            self.correction_max,
            global_step,
        )
        writer.add_scalar(
            "projection/slack_sum",
            self.slack_sum / self.transition_count,
            global_step,
        )
        writer.add_scalar("projection/slack_max", self.slack_max, global_step)
        writer.add_scalar(
            "projection/solver_failure_count",
            self.solver_failure_count,
            global_step,
        )
    #} End function write_tensorboard

#} End class ProjectionTrainingDiagnostics


__all__ = ["ProjectionTrainingDiagnostics"]
