from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from environments.factory import make_env
from evaluation.evaluate_policy import (
    main,
    make_projection_params,
    parse_args,
    print_summary,
    run_episode,
    summarize_results,
    write_results_csv,
)
from projection.cbf_qp_projection import ProjectionParams
from projection.cbf_qp_wrapper import CbfQpProjectionWrapper
from evaluation.trajectory_recording import write_trajectory_archive


# Return one deterministic normalized action for evaluator smoke checks.
def zero_normalized_action(env, obs) -> np.ndarray:
#{
    del env, obs
    return np.zeros(2, dtype=np.float32)

#} End function zero_normalized_action


# Deterministic environment stub for evaluator-only projection metric checks.
class ScriptedProjectionEnv:
#{
    def __init__(self, slack_sums: list[float], slack_maxes: list[float]) -> None:
    #{
        self.slack_sums = list(slack_sums)
        self.slack_maxes = list(slack_maxes)
        self.max_episode_steps = len(self.slack_sums)
        self.unwrapped = self
        self.step_index = 0

    #} End function __init__

    def reset(self, *, seed: int | None = None):
    #{
        del seed
        self.step_index = 0
        info = {
            "success": False,
            "collision": False,
            "distance_to_goal": 1.0,
            "min_obstacle_clearance": 1.0,
        }
        return np.zeros(1, dtype=np.float32), info

    #} End function reset

    def step(self, action):
    #{
        del action
        index = self.step_index
        self.step_index += 1
        truncated = self.step_index >= self.max_episode_steps
        info = {
            "success": False,
            "collision": False,
            "distance_to_goal": 1.0,
            "min_obstacle_clearance": 1.0,
            "projection_enabled": True,
            "projection_intervened": False,
            "projection_correction_norm": 0.0,
            "projection_slack_max": self.slack_maxes[index],
            "projection_slack_sum": self.slack_sums[index],
            "projection_success": True,
            "projection_solver_status": "optimal",
            "projection_active_constraint_count": 1,
        }
        return np.zeros(1, dtype=np.float32), 0.0, False, truncated, info

    #} End function step

#} End class ScriptedProjectionEnv


# Verify that evaluator projection arguments reach the wrapper and the CSV artifact.
def test_evaluator_projection_parameters_reach_wrapper_and_csv(monkeypatch, tmp_path) -> None:
#{
    output_path = tmp_path / "projection_parameters.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_policy.py",
            "--episodes",
            "1",
            "--seed",
            "7",
            "--max-episode-steps",
            "1",
            "--num-active-obstacles",
            "0",
            "--enable-projection",
            "--projection-lookahead-distance",
            "0.40",
            "--projection-alpha",
            "3.50",
            "--projection-slack-penalty",
            "2500.0",
            "--projection-extra-clearance",
            "0.08",
            "--no-cuda",
            "--output",
            str(output_path),
        ],
    )

    args = parse_args()
    projection_params = make_projection_params(args)

    assert projection_params.lookahead_distance == 0.40
    assert projection_params.alpha == 3.50
    assert projection_params.slack_penalty == 2500.0
    assert projection_params.extra_clearance == 0.08

    env_factory = make_env(
        env_index=0,
        env_kwargs={
            "max_episode_steps": args.max_episode_steps,
            "max_obstacles": args.max_obstacles,
            "num_active_obstacles": args.num_active_obstacles,
        },
        record_episode_statistics=False,
        normalize_actions=True,
        enable_projection=args.enable_projection,
        projection_params=projection_params,
    )

    env = env_factory()

    try:
        projection_wrapper = env.env
        assert isinstance(projection_wrapper, CbfQpProjectionWrapper)
        assert projection_wrapper.params.lookahead_distance == 0.40
        assert projection_wrapper.params.alpha == 3.50
        assert projection_wrapper.params.slack_penalty == 2500.0
        assert projection_wrapper.params.extra_clearance == 0.08

        result = run_episode(
            env=env,
            action_provider=zero_normalized_action,
            seed=args.seed,
            episode=0,
            policy_name=args.policy,
        )

    finally:
        env.close()

    assert result.projection_enabled
    assert result.projection_solver_failure_count == 0
    assert result.mean_projection_slack_sum == 0.0
    assert result.max_projection_slack == 0.0
    assert np.isnan(result.min_obstacle_clearance)

    summary = summarize_results([result])
    assert summary["mean_projection_slack_sum"] == 0.0
    assert summary["max_projection_slack"] == 0.0
    assert "mean_min_obstacle_clearance" in summary
    assert "mean_min_obstacle_distance" not in summary
    assert np.isnan(summary["mean_min_obstacle_clearance"])

    finite_clearance_result = replace(result, min_obstacle_clearance=0.50)
    mixed_summary = summarize_results([result, finite_clearance_result])
    np.testing.assert_allclose(mixed_summary["mean_min_obstacle_clearance"], 0.50)

    write_results_csv(
        results=[result],
        output_path=output_path,
        projection_params=projection_params,
    )

    row = pd.read_csv(output_path).iloc[0]

    assert "min_obstacle_clearance" in row.index
    assert "min_obstacle_distance" not in row.index
    assert pd.isna(row["min_obstacle_clearance"])
    np.testing.assert_allclose(row["mean_projection_slack_sum"], 0.0)

    np.testing.assert_allclose(row["projection_lookahead_distance"], 0.40)
    np.testing.assert_allclose(row["projection_alpha"], 3.50)
    np.testing.assert_allclose(row["projection_slack_penalty"], 2500.0)
    np.testing.assert_allclose(row["projection_extra_clearance"], 0.08)

#} End function test_evaluator_projection_parameters_reach_wrapper_and_csv


# Verify that the evaluator averages per-step slack sums, including zero-slack steps.
def test_evaluator_computes_mean_projection_slack_sum(tmp_path, capsys) -> None:
#{
    env = ScriptedProjectionEnv(
        slack_sums=[0.0, 0.30, 0.90],
        slack_maxes=[0.0, 0.20, 0.50],
    )

    result = run_episode(
        env=env,
        action_provider=zero_normalized_action,
        seed=17,
        episode=0,
        policy_name="random",
    )

    assert result.episode_length == 3
    assert result.projection_enabled
    assert result.projection_solver_failure_count == 0
    np.testing.assert_allclose(result.mean_projection_slack_sum, 0.40)
    np.testing.assert_allclose(result.max_projection_slack, 0.50)

    summary = summarize_results([result])
    np.testing.assert_allclose(summary["mean_projection_slack_sum"], 0.40)
    np.testing.assert_allclose(summary["max_projection_slack"], 0.50)

    second_result = replace(
        result,
        episode=1,
        episode_length=1,
        mean_projection_slack_sum=0.80,
        max_projection_slack=0.70,
    )
    multi_episode_summary = summarize_results([result, second_result])
    np.testing.assert_allclose(multi_episode_summary["mean_projection_slack_sum"], 0.60)
    np.testing.assert_allclose(multi_episode_summary["max_projection_slack"], 0.70)

    output_path = tmp_path / "projection_slack_metrics.csv"
    projection_params = ProjectionParams()
    write_results_csv(
        results=[result],
        output_path=output_path,
        projection_params=projection_params,
    )

    row = pd.read_csv(output_path).iloc[0]
    np.testing.assert_allclose(row["mean_projection_slack_sum"], 0.40)
    np.testing.assert_allclose(row["max_projection_slack"], 0.50)

    print_summary(summary, output_path)
    printed = capsys.readouterr().out
    assert "mean_projection_slack_sum:          0.400000" in printed
    assert "max_projection_slack:               0.500000" in printed

#} End function test_evaluator_computes_mean_projection_slack_sum


# Verify that projection solver failures preserve unknown slack diagnostics.
def test_solver_failure_preserves_unknown_slack_diagnostics(tmp_path, capsys) -> None:
#{
    projection_params = ProjectionParams(solver_name="INVALID_SOLVER")
    env_factory = make_env(
        env_index=0,
        env_kwargs={"max_episode_steps": 2},
        record_episode_statistics=False,
        normalize_actions=True,
        enable_projection=True,
        projection_params=projection_params,
    )

    env = env_factory()
    trajectory_records = []

    try:
        result = run_episode(
            env=env,
            action_provider=zero_normalized_action,
            seed=11,
            episode=0,
            policy_name="random",
            trajectory_records=trajectory_records,
        )

    finally:
        env.close()

    assert result.episode_length == 2
    assert result.projection_enabled
    assert result.projection_solver_failure_count == 2
    assert np.isnan(result.mean_projection_slack_sum)
    assert np.isnan(result.max_projection_slack)

    summary = summarize_results([result])

    assert summary["total_projection_solver_failures"] == 2
    assert np.isnan(summary["mean_projection_slack_sum"])
    assert np.isnan(summary["max_projection_slack"])

    successful_result = replace(
        result,
        mean_projection_slack_sum=0.0,
        max_projection_slack=0.0,
        projection_solver_failure_count=0,
    )
    mixed_summary = summarize_results([successful_result, result])
    assert np.isnan(mixed_summary["mean_projection_slack_sum"])
    assert np.isnan(mixed_summary["max_projection_slack"])

    output_path = tmp_path / "projection_solver_failure.csv"
    write_results_csv(
        results=[result],
        output_path=output_path,
        projection_params=projection_params,
    )

    row = pd.read_csv(output_path).iloc[0]
    assert row["projection_solver_failure_count"] == 2
    assert pd.isna(row["mean_projection_slack_sum"])
    assert pd.isna(row["max_projection_slack"])

    print_summary(summary, output_path)
    printed = capsys.readouterr().out
    assert "mean_projection_slack_sum:          N/A" in printed
    assert "max_projection_slack:               N/A" in printed

    trajectory_path = tmp_path / "projection_solver_failure_trajectories.npz"
    write_trajectory_archive(
        trajectories=trajectory_records,
        output_path=trajectory_path,
        projection_params=projection_params,
    )

    with np.load(trajectory_path, allow_pickle=False) as archive:
        key = archive["episode_keys"].tolist()[0]
        assert not archive[f"{key}_projection_success"].any()
        assert archive[f"{key}_projection_solver_status"].tolist() == [
            "solver_error",
            "solver_error",
        ]
        np.testing.assert_allclose(
            archive[f"{key}_action_exec_physical"],
            np.zeros((2, 2)),
        )
        assert np.isnan(archive[f"{key}_projection_slack_values"]).all()

#} End function test_solver_failure_preserves_unknown_slack_diagnostics

# Verify the single-mode evaluator can write a self-contained trajectory archive.
def test_single_evaluator_writes_trajectory_archive(monkeypatch, tmp_path) -> None:
#{
    output_path = tmp_path / "random_evaluation.csv"
    trajectory_path = tmp_path / "random_evaluation_trajectories.npz"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_policy.py",
            "--policy",
            "random",
            "--episodes",
            "1",
            "--seed",
            "23",
            "--max-episode-steps",
            "1",
            "--num-active-obstacles",
            "0",
            "--no-cuda",
            "--output",
            str(output_path),
            "--trajectory-output",
            str(trajectory_path),
        ],
    )

    main()

    assert output_path.exists()
    assert trajectory_path.exists()

    with np.load(trajectory_path, allow_pickle=False) as archive:
        assert str(archive["trajectory_archive_version"]) == "evaluation_trajectory_v1"
        assert int(archive["episode_count"]) == 1
        assert archive["episode_keys"].tolist() == ["episode_0000"]

        key = "episode_0000"
        assert int(archive[f"{key}_episode"]) == 0
        assert int(archive[f"{key}_seed"]) == 23
        assert archive[f"{key}_positions"].shape == (2, 2)
        assert archive[f"{key}_headings"].shape == (2,)
        assert archive[f"{key}_action_raw_normalized"].shape == (1, 2)
        assert archive[f"{key}_action_raw_physical"].shape == (1, 2)
        assert archive[f"{key}_action_exec_physical"].shape == (1, 2)
        assert archive[f"{key}_projection_slack_values"].shape == (1, 3)

        raw_normalized = archive[f"{key}_action_raw_normalized"][0]
        raw_physical = archive[f"{key}_action_raw_physical"][0]
        expected_physical = np.asarray(
            [
                0.5 * (np.clip(raw_normalized[0], -1.0, 1.0) + 1.0),
                2.0 * np.clip(raw_normalized[1], -1.0, 1.0),
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(raw_physical, expected_physical, rtol=2.0e-7, atol=1.0e-7)
        np.testing.assert_allclose(
            archive[f"{key}_action_exec_physical"],
            archive[f"{key}_action_raw_physical"],
        )
        np.testing.assert_allclose(
            archive[f"{key}_action_correction_physical"],
            np.zeros((1, 2)),
        )
        assert not archive[f"{key}_projection_enabled"].any()
        assert archive[f"{key}_projection_success"].all()
        assert archive[f"{key}_projection_solver_status"].tolist() == ["disabled"]

#} End function test_single_evaluator_writes_trajectory_archive

