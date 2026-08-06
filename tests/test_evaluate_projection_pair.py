from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from algorithms.ppo.agent import Agent
from evaluation.evaluate_projection_pair import main, make_output_paths


# Verify the complete paired command path with a small deterministic checkpoint.
def test_paired_projection_evaluation_writes_aligned_artifacts(monkeypatch, tmp_path) -> None:
#{
    checkpoint_path = tmp_path / "agent.pt"
    output_prefix = tmp_path / "paired_evaluation"

    agent = Agent(obs_dim=21, action_dim=2)
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "obs_dim": 21,
            "action_dim": 2,
            "args": {
                "method": "ppo_baseline",
                "seed": 1,
                "collision_penalty": 10.0,
                "enable_projection": False,
            },
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_projection_pair.py",
            "--checkpoint",
            str(checkpoint_path),
            "--method",
            "ppo_baseline",
            "--train-seed",
            "1",
            "--episodes",
            "2",
            "--seed",
            "31",
            "--max-episode-steps",
            "1",
            "--num-active-obstacles",
            "0",
            "--collision-penalty",
            "10.0",
            "--projection-lookahead-distance",
            "0.40",
            "--projection-alpha",
            "3.50",
            "--projection-slack-penalty",
            "2500.0",
            "--projection-extra-clearance",
            "0.08",
            "--no-cuda",
            "--output-prefix",
            str(output_prefix),
        ],
    )

    main()

    output_paths = make_output_paths(output_prefix)

    for output_path in output_paths.values():
        assert output_path.exists()

    projection_disabled = pd.read_csv(output_paths["projection_disabled"])
    projection_enabled = pd.read_csv(output_paths["projection_enabled"])
    paired_episodes = pd.read_csv(output_paths["paired_episodes"])
    paired_summary = pd.read_csv(output_paths["paired_summary"]).iloc[0]
    expected_checkpoint_sha256 = str(paired_summary["checkpoint_sha256"])

    for frame in (projection_disabled, projection_enabled, paired_episodes):
        assert frame["method"].tolist() == ["ppo_baseline"] * len(frame)
        assert frame["train_seed"].tolist() == [1] * len(frame)
        assert frame["evaluation_collision_penalty"].tolist() == [10.0] * len(frame)
        assert frame["evaluation_policy_mode"].tolist() == ["deterministic"] * len(frame)

    assert paired_summary["method"] == "ppo_baseline"
    assert paired_summary["train_seed"] == 1
    assert paired_summary["evaluation_collision_penalty"] == 10.0
    assert paired_summary["evaluation_policy_mode"] == "deterministic"

    with np.load(output_paths["projection_disabled_trajectories"], allow_pickle=False) as archive:
        assert int(archive["episode_count"]) == 2
        assert archive["episode_keys"].tolist() == ["episode_0000", "episode_0001"]
        assert str(archive["run_projection_mode"]) == "disabled"
        assert str(archive["run_method"]) == "ppo_baseline"
        assert int(archive["run_train_seed"]) == 1
        assert str(archive["run_evaluation_policy_mode"]) == "deterministic"
        assert float(archive["run_evaluation_collision_penalty"]) == 10.0
        assert str(archive["run_checkpoint_sha256"]) == expected_checkpoint_sha256
        assert int(archive["run_base_seed"]) == 31
        assert int(archive["run_last_seed"]) == 32
        assert [
            int(archive[f"{key}_seed"])
            for key in archive["episode_keys"].tolist()
        ] == [31, 32]

        for key in archive["episode_keys"].tolist():
            assert archive[f"{key}_positions"].shape == (2, 2)
            assert archive[f"{key}_action_raw_normalized"].shape == (1, 2)
            assert archive[f"{key}_action_raw_physical"].shape == (1, 2)
            assert archive[f"{key}_action_exec_physical"].shape == (1, 2)
            np.testing.assert_allclose(
                archive[f"{key}_action_exec_physical"],
                archive[f"{key}_action_raw_physical"],
            )
            np.testing.assert_allclose(
                archive[f"{key}_action_correction_physical"],
                np.zeros((1, 2)),
            )
            assert not archive[f"{key}_projection_enabled"].any()
            assert archive[f"{key}_projection_solver_status"].tolist() == ["disabled"]

    with np.load(output_paths["projection_enabled_trajectories"], allow_pickle=False) as archive:
        assert int(archive["episode_count"]) == 2
        assert archive["episode_keys"].tolist() == ["episode_0000", "episode_0001"]
        assert str(archive["run_projection_mode"]) == "enabled"
        assert str(archive["run_method"]) == "ppo_baseline"
        assert int(archive["run_train_seed"]) == 1
        assert str(archive["run_evaluation_policy_mode"]) == "deterministic"
        assert float(archive["run_evaluation_collision_penalty"]) == 10.0
        assert str(archive["run_checkpoint_sha256"]) == expected_checkpoint_sha256
        assert int(archive["run_base_seed"]) == 31
        assert int(archive["run_last_seed"]) == 32
        assert [
            int(archive[f"{key}_seed"])
            for key in archive["episode_keys"].tolist()
        ] == [31, 32]

        for key in archive["episode_keys"].tolist():
            assert archive[f"{key}_projection_enabled"].all()
            assert archive[f"{key}_projection_success"].all()
            assert archive[f"{key}_projection_solver_status"].tolist() == [
                "no_active_constraints"
            ]
            np.testing.assert_allclose(
                archive[f"{key}_action_exec_physical"],
                archive[f"{key}_action_raw_physical"],
            )
            np.testing.assert_allclose(
                archive[f"{key}_projection_slack_values"],
                np.zeros((1, 3)),
            )

    assert projection_disabled["episode"].tolist() == [0, 1]
    assert projection_enabled["episode"].tolist() == [0, 1]
    assert projection_disabled["seed"].tolist() == [31, 32]
    assert projection_enabled["seed"].tolist() == [31, 32]
    assert not projection_disabled["projection_enabled"].any()
    assert projection_enabled["projection_enabled"].all()
    np.testing.assert_allclose(
        projection_disabled["action_bound_clipping_rate"],
        0.0,
    )
    np.testing.assert_allclose(
        projection_enabled["action_bound_clipping_rate"],
        0.0,
    )

    np.testing.assert_allclose(
        projection_disabled["episode_return"],
        projection_enabled["episode_return"],
    )
    np.testing.assert_allclose(
        projection_disabled["final_distance_to_goal"],
        projection_enabled["final_distance_to_goal"],
    )

    assert paired_episodes["episode"].tolist() == [0, 1]
    assert paired_episodes["seed"].tolist() == [31, 32]
    assert paired_episodes["checkpoint_sha256"].nunique() == 1
    assert paired_episodes["num_active_obstacles"].tolist() == [0, 0]
    np.testing.assert_allclose(
        paired_episodes["episode_return_delta_enabled_minus_disabled"],
        0.0,
    )
    np.testing.assert_allclose(
        paired_episodes["action_bound_clipping_rate_delta_enabled_minus_disabled"],
        0.0,
    )
    np.testing.assert_allclose(
        paired_episodes["with_projection_mean_slack_sum"],
        0.0,
    )
    np.testing.assert_allclose(
        paired_episodes["with_projection_max_slack"],
        0.0,
    )

    assert paired_summary["requested_episodes"] == 2
    assert paired_summary["base_seed"] == 31
    assert paired_summary["last_seed"] == 32
    assert paired_summary["num_active_obstacles"] == 0
    assert paired_summary["projection_lookahead_distance"] == 0.40
    assert paired_summary["projection_alpha"] == 3.50
    assert paired_summary["projection_slack_penalty"] == 2500.0
    assert paired_summary["projection_extra_clearance"] == 0.08
    assert paired_summary["mean_return_delta_enabled_minus_disabled"] == 0.0
    assert paired_summary["success_rate_delta_enabled_minus_disabled"] == 0.0
    assert paired_summary["collision_rate_delta_enabled_minus_disabled"] == 0.0
    assert paired_summary[
        "mean_action_bound_clipping_rate_delta_enabled_minus_disabled"
    ] == 0.0
    assert paired_summary["with_projection_total_interventions"] == 0
    assert paired_summary["with_projection_total_solver_failures"] == 0
    assert paired_summary["projection_disabled_trajectory_npz"] == str(
        output_paths["projection_disabled_trajectories"]
    )
    assert paired_summary["projection_enabled_trajectory_npz"] == str(
        output_paths["projection_enabled_trajectories"]
    )

#} End function test_paired_projection_evaluation_writes_aligned_artifacts
