from __future__ import annotations

import numpy as np
import pytest

from algorithms.ppo.action_bound_diagnostics import ActionBoundDiagnostics


# Collect scalar writes without importing TensorBoard.
class ScalarWriter:
#{
    def __init__(self) -> None:
        self.values: dict[str, tuple[float, int]] = {}

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.values[tag] = (float(value), int(step))

#} End class ScalarWriter


# Verify clipping is separated by action dimension and includes zero-correction steps.
def test_action_bound_diagnostics_compute_transition_metrics() -> None:
#{
    diagnostics = ActionBoundDiagnostics()
    diagnostics.update(
        np.asarray(
            [
                [0.0, 0.0],
                [1.5, 0.0],
                [0.0, -2.0],
                [1.5, 2.0],
            ],
            dtype=np.float64,
        )
    )
    metrics = diagnostics.metrics()

    assert metrics["transition_count"] == 4
    assert metrics["clipping_count"] == 3
    assert metrics["speed_clipping_count"] == 2
    assert metrics["turn_rate_clipping_count"] == 2
    np.testing.assert_allclose(metrics["clipping_frequency"], 0.75)
    np.testing.assert_allclose(metrics["speed_clipping_frequency"], 0.50)
    np.testing.assert_allclose(metrics["turn_rate_clipping_frequency"], 0.50)
    expected_norms = np.asarray([0.0, 0.5, 1.0, np.sqrt(1.25)])
    np.testing.assert_allclose(metrics["clipping_norm"], np.mean(expected_norms))
    np.testing.assert_allclose(metrics["clipping_norm_max"], np.max(expected_norms))

    writer = ScalarWriter()
    diagnostics.write_tensorboard(writer, global_step=1024)
    assert writer.values["action_bounds/clipping_frequency"] == (0.75, 1024)
    np.testing.assert_allclose(
        writer.values["action_bounds/clipping_norm"][0],
        np.mean(expected_norms),
    )
    np.testing.assert_allclose(
        writer.values["action_bounds/clipping_norm_max"][0],
        np.max(expected_norms),
    )

#} End function test_action_bound_diagnostics_compute_transition_metrics


# Empty diagnostics use explicit zero values and produce no TensorBoard event.
def test_action_bound_diagnostics_empty_state_is_explicit() -> None:
#{
    diagnostics = ActionBoundDiagnostics()
    metrics = diagnostics.metrics()
    assert set(metrics.values()) == {0, 0.0}

    writer = ScalarWriter()
    diagnostics.write_tensorboard(writer, global_step=0)
    assert writer.values == {}

#} End function test_action_bound_diagnostics_empty_state_is_explicit


# Invalid action batches cannot silently enter the research diagnostics.
@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 3), dtype=np.float64),
        np.asarray([0.0, np.nan], dtype=np.float64),
    ],
)
def test_action_bound_diagnostics_reject_invalid_actions(actions: np.ndarray) -> None:
#{
    diagnostics = ActionBoundDiagnostics()

    with pytest.raises((ValueError, RuntimeError)):
        diagnostics.update(actions)

#} End function test_action_bound_diagnostics_reject_invalid_actions
