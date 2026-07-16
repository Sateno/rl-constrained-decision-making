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
            "--episodes",
            "2",
            "--seed",
            "31",
            "--max-episode-steps",
            "1",
            "--num-active-obstacles",
            "0",
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

    assert projection_disabled["episode"].tolist() == [0, 1]
    assert projection_enabled["episode"].tolist() == [0, 1]
    assert projection_disabled["seed"].tolist() == [31, 32]
    assert projection_enabled["seed"].tolist() == [31, 32]
    assert not projection_disabled["projection_enabled"].any()
    assert projection_enabled["projection_enabled"].all()

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
    assert paired_summary["with_projection_total_interventions"] == 0
    assert paired_summary["with_projection_total_solver_failures"] == 0

#} End function test_paired_projection_evaluation_writes_aligned_artifacts
