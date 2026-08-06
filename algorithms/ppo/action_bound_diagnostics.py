from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Accumulate clipping imposed by the normalized action bounds [-1, 1]^2.
@dataclass
class ActionBoundDiagnostics:
#{
    transition_count: int = 0
    clipping_count: int = 0
    speed_clipping_count: int = 0
    turn_rate_clipping_count: int = 0
    clipping_norm_sum: float = 0.0
    clipping_norm_max: float = 0.0

    # Consume one action or a batch of raw normalized actions.
    def update(self, actions: Any) -> None:
    #{
        action_array = np.asarray(actions, dtype=np.float64)

        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        if action_array.ndim != 2 or action_array.shape[1] != 2:
            raise ValueError(
                "Action-bound diagnostics require actions with shape (N, 2)."
            )
        if not np.all(np.isfinite(action_array)):
            raise RuntimeError("Raw normalized actions contain non-finite values.")

        bounded = np.clip(action_array, -1.0, 1.0)
        clipping = action_array - bounded
        clipped_components = clipping != 0.0
        clipping_norms = np.linalg.norm(clipping, axis=1)

        self.transition_count += int(action_array.shape[0])
        self.clipping_count += int(np.count_nonzero(np.any(clipped_components, axis=1)))
        self.speed_clipping_count += int(np.count_nonzero(clipped_components[:, 0]))
        self.turn_rate_clipping_count += int(np.count_nonzero(clipped_components[:, 1]))
        self.clipping_norm_sum += float(np.sum(clipping_norms))
        self.clipping_norm_max = max(
            self.clipping_norm_max,
            float(np.max(clipping_norms, initial=0.0)),
        )
    #} End function update

    # Return finite transition-level clipping metrics.
    def metrics(self) -> dict[str, float | int]:
    #{
        if self.transition_count == 0:
            return {
                "transition_count": 0,
                "clipping_count": 0,
                "clipping_frequency": 0.0,
                "speed_clipping_count": 0,
                "speed_clipping_frequency": 0.0,
                "turn_rate_clipping_count": 0,
                "turn_rate_clipping_frequency": 0.0,
                "clipping_norm": 0.0,
                "clipping_norm_max": 0.0,
            }

        return {
            "transition_count": self.transition_count,
            "clipping_count": self.clipping_count,
            "clipping_frequency": self.clipping_count / self.transition_count,
            "speed_clipping_count": self.speed_clipping_count,
            "speed_clipping_frequency": (
                self.speed_clipping_count / self.transition_count
            ),
            "turn_rate_clipping_count": self.turn_rate_clipping_count,
            "turn_rate_clipping_frequency": (
                self.turn_rate_clipping_count / self.transition_count
            ),
            "clipping_norm": self.clipping_norm_sum / self.transition_count,
            "clipping_norm_max": self.clipping_norm_max,
        }
    #} End function metrics

    # Write rollout-level action-bound clipping diagnostics to TensorBoard.
    def write_tensorboard(self, writer: Any, global_step: int) -> None:
    #{
        if self.transition_count == 0:
            return

        metrics = self.metrics()
        writer.add_scalar(
            "action_bounds/clipping_frequency",
            metrics["clipping_frequency"],
            global_step,
        )
        writer.add_scalar(
            "action_bounds/speed_clipping_frequency",
            metrics["speed_clipping_frequency"],
            global_step,
        )
        writer.add_scalar(
            "action_bounds/turn_rate_clipping_frequency",
            metrics["turn_rate_clipping_frequency"],
            global_step,
        )
        writer.add_scalar(
            "action_bounds/clipping_norm",
            metrics["clipping_norm"],
            global_step,
        )
        writer.add_scalar(
            "action_bounds/clipping_norm_max",
            metrics["clipping_norm_max"],
            global_step,
        )
    #} End function write_tensorboard

#} End class ActionBoundDiagnostics


__all__ = ["ActionBoundDiagnostics"]
