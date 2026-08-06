from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.training_diagnostics import (
    ACTION_BOUND_TAGS,
    EPISODE_REQUIRED_TAGS,
    PROJECTION_TAGS,
    aggregate_episode_curve,
    training_episode_diagnostics,
    training_rollout_diagnostics,
)


IDENTITY = {
    "method": "ppo_train_projection",
    "display_name": "PPO trained with projection",
    "train_seed": 1,
    "checkpoint_sha256": "a" * 64,
    "training_projection_enabled": True,
    "run_dir": "runs/example",
}


# Return long-form TensorBoard points for one tag.
def scalar_points(tag: str, steps: list[int], values: list[float]) -> pd.DataFrame:
#{
    frame = pd.DataFrame(
        {
            "training_diagnostics_schema_version": "training_diagnostics_v1",
            "event_index": np.arange(len(steps), dtype=np.int64),
            "step": steps,
            "value": values,
            "tag": tag,
        }
    )

    for column, value in IDENTITY.items():
        frame[column] = value

    return frame

#} End function scalar_points


# Episode events sharing one vector step remain aligned by within-step occurrence.
def test_training_episode_diagnostics_build_cumulative_and_rolling_outcomes() -> None:
#{
    steps = [200, 200, 400]
    values = {
        "charts/episodic_return": [1.0, 2.0, 3.0],
        "charts/episodic_length": [50.0, 60.0, 70.0],
        "safety/success": [0.0, 1.0, 0.0],
        "safety/collision": [1.0, 0.0, 0.0],
        "safety/timeout": [0.0, 0.0, 1.0],
    }
    points = pd.concat(
        [scalar_points(tag, steps, values[tag]) for tag in EPISODE_REQUIRED_TAGS],
        ignore_index=True,
    )

    table = training_episode_diagnostics(
        points,
        rolling_window=2,
        require_complete=True,
    )

    assert table["step"].tolist() == [200, 200, 400]
    assert table["step_occurrence"].tolist() == [0, 1, 0]
    assert table["completed_episode"].tolist() == [1, 2, 3]
    assert table["cumulative_collision_count"].tolist() == [1, 1, 1]
    assert table["cumulative_success_count"].tolist() == [0, 1, 1]
    assert table["cumulative_timeout_count"].tolist() == [0, 0, 1]
    np.testing.assert_allclose(table["rolling_collision_rate"], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(table["rolling_success_rate"], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(table["rolling_timeout_rate"], [0.0, 0.0, 0.5])

#} End function test_training_episode_diagnostics_build_cumulative_and_rolling_outcomes


# Missing or contradictory episode outcomes cannot enter a final training dataset.
def test_training_episode_diagnostics_reject_inconsistent_outcomes() -> None:
#{
    steps = [200]
    values = {
        "charts/episodic_return": [1.0],
        "charts/episodic_length": [50.0],
        "safety/success": [1.0],
        "safety/collision": [1.0],
        "safety/timeout": [0.0],
    }
    points = pd.concat(
        [scalar_points(tag, steps, values[tag]) for tag in EPISODE_REQUIRED_TAGS],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        training_episode_diagnostics(
            points,
            rolling_window=20,
            require_complete=True,
        )

#} End function test_training_episode_diagnostics_reject_inconsistent_outcomes


# Rollout export preserves clipping and complete mean/maximum projection burden.
def test_training_rollout_diagnostics_exports_complete_projection_metrics() -> None:
#{
    steps = [1024, 2048]
    values = {
        "action_bounds/clipping_frequency": [0.5, 0.4],
        "action_bounds/speed_clipping_frequency": [0.3, 0.2],
        "action_bounds/turn_rate_clipping_frequency": [0.25, 0.25],
        "action_bounds/clipping_norm": [0.20, 0.15],
        "action_bounds/clipping_norm_max": [1.20, 1.00],
        "projection/transition_count": [1024.0, 1024.0],
        "projection/intervention_count": [100.0, 80.0],
        "projection/intervention_frequency": [100.0 / 1024.0, 80.0 / 1024.0],
        "projection/correction_norm": [0.05, 0.04],
        "projection/correction_norm_max": [0.40, 0.35],
        "projection/slack_sum": [0.001, 0.0005],
        "projection/slack_max": [0.01, 0.008],
        "projection/solver_failure_count": [0.0, 0.0],
    }
    points = pd.concat(
        [
            scalar_points(tag, steps, values[tag])
            for tag in (*ACTION_BOUND_TAGS, *PROJECTION_TAGS)
        ],
        ignore_index=True,
    )

    table = training_rollout_diagnostics(points, require_complete=True)

    assert table["step"].tolist() == steps
    np.testing.assert_allclose(
        table["action_bound_clipping_frequency"],
        [0.5, 0.4],
    )
    np.testing.assert_allclose(
        table["max_action_bound_clipping_norm"],
        [1.2, 1.0],
    )
    np.testing.assert_allclose(
        table["max_projection_correction_norm"],
        [0.4, 0.35],
    )
    np.testing.assert_allclose(table["max_projection_slack"], [0.01, 0.008])
    assert not table["projection_solver_failure_count"].any()

#} End function test_training_rollout_diagnostics_exports_complete_projection_metrics


# Optional episode scalars are omitted when they cannot be aligned safely.
def test_training_episode_diagnostics_omits_unaligned_optional_scalar() -> None:
#{
    steps = [200, 200]
    values = {
        "charts/episodic_return": [1.0, 2.0],
        "charts/episodic_length": [50.0, 60.0],
        "safety/success": [1.0, 0.0],
        "safety/collision": [0.0, 1.0],
        "safety/timeout": [0.0, 0.0],
    }
    points = [
        scalar_points(tag, steps, values[tag])
        for tag in EPISODE_REQUIRED_TAGS
    ]
    points.append(
        scalar_points("safety/final_obstacle_clearance", [200], [0.25])
    )

    table = training_episode_diagnostics(
        pd.concat(points, ignore_index=True),
        rolling_window=20,
        require_complete=True,
    )

    assert "final_obstacle_clearance" not in table.columns

#} End function test_training_episode_diagnostics_omits_unaligned_optional_scalar


# Rollout tags must share one exact environment-step grid.
def test_training_rollout_diagnostics_rejects_misaligned_steps() -> None:
#{
    steps = [1024, 2048]
    frames = []

    for tag in ACTION_BOUND_TAGS:
        tag_steps = [1024, 3072] if tag == ACTION_BOUND_TAGS[-1] else steps
        frames.append(scalar_points(tag, tag_steps, [0.0, 0.0]))

    for tag in PROJECTION_TAGS:
        values = [1024.0, 1024.0] if tag == "projection/transition_count" else [0.0, 0.0]
        frames.append(scalar_points(tag, steps, values))

    with pytest.raises(ValueError, match="not aligned"):
        training_rollout_diagnostics(
            pd.concat(frames, ignore_index=True),
            require_complete=True,
        )

#} End function test_training_rollout_diagnostics_rejects_misaligned_steps


# Projection frequency must agree with the recorded rollout counts.
def test_training_rollout_diagnostics_rejects_projection_count_mismatch() -> None:
#{
    steps = [1024]
    values = {
        "action_bounds/clipping_frequency": [0.0],
        "action_bounds/speed_clipping_frequency": [0.0],
        "action_bounds/turn_rate_clipping_frequency": [0.0],
        "action_bounds/clipping_norm": [0.0],
        "action_bounds/clipping_norm_max": [0.0],
        "projection/transition_count": [1024.0],
        "projection/intervention_count": [100.0],
        "projection/intervention_frequency": [0.50],
        "projection/correction_norm": [0.0],
        "projection/correction_norm_max": [0.0],
        "projection/slack_sum": [0.0],
        "projection/slack_max": [0.0],
        "projection/solver_failure_count": [0.0],
    }
    points = pd.concat(
        [
            scalar_points(tag, steps, values[tag])
            for tag in (*ACTION_BOUND_TAGS, *PROJECTION_TAGS)
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="disagrees with counts"):
        training_rollout_diagnostics(points, require_complete=True)

#} End function test_training_rollout_diagnostics_rejects_projection_count_mismatch


# Cumulative training outcomes are aggregated as step functions, not linearly.
def test_aggregate_episode_curve_uses_post_step_cumulative_values() -> None:
#{
    episodes = pd.DataFrame(
        [
            {
                "method": "ppo_baseline",
                "display_name": "PPO baseline",
                "train_seed": 1,
                "step": 100,
                "completed_episode": 1,
                "cumulative_collision_count": 1,
            },
            {
                "method": "ppo_baseline",
                "display_name": "PPO baseline",
                "train_seed": 1,
                "step": 300,
                "completed_episode": 2,
                "cumulative_collision_count": 2,
            },
            {
                "method": "ppo_baseline",
                "display_name": "PPO baseline",
                "train_seed": 2,
                "step": 200,
                "completed_episode": 1,
                "cumulative_collision_count": 0,
            },
            {
                "method": "ppo_baseline",
                "display_name": "PPO baseline",
                "train_seed": 2,
                "step": 300,
                "completed_episode": 2,
                "cumulative_collision_count": 1,
            },
        ]
    )

    curve = aggregate_episode_curve(episodes, "cumulative_collision_count")

    assert curve["step"].tolist() == [0.0, 100.0, 200.0, 300.0]
    np.testing.assert_allclose(
        curve["value_mean"],
        [0.0, 0.5, 0.5, 1.5],
    )
    assert curve["seed_count"].tolist() == [2, 2, 2, 2]

#} End function test_aggregate_episode_curve_uses_post_step_cumulative_values
