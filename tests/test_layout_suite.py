from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from algorithms.ppo.agent import Agent
from evaluation.evaluate_layout_suite import main
from evaluation.layout_suite import file_sha256, load_navigation_layout_suite


# Write one small valid suite used by loader and evaluator checks.
def write_suite(path) -> None:
#{
    path.write_text(
        json.dumps(
            {
                "schema_version": "navigation_layout_suite_v1",
                "suite_id": "test_navigation_layouts",
                "max_obstacles": 3,
                "agent_radius": 0.1,
                "goal_radius": 0.25,
                "layouts": [
                    {
                        "layout_id": "open_route",
                        "start": [0.0, 0.0],
                        "theta": 0.0,
                        "goal": [2.0, 0.0],
                        "obstacles": [],
                    },
                    {
                        "layout_id": "offset_obstacle",
                        "start": [0.0, 0.0],
                        "theta": 0.0,
                        "goal": [2.0, 0.0],
                        "obstacles": [
                            {"center": [1.0, 0.6], "radius": 0.2},
                        ],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

#} End function write_suite


# Verify deterministic hashing, padded geometry, and fresh reset arrays.
def test_layout_suite_loads_and_pads_geometry(tmp_path) -> None:
#{
    suite_path = tmp_path / "layouts.json"
    write_suite(suite_path)

    suite = load_navigation_layout_suite(suite_path)

    assert suite.schema_version == "navigation_layout_suite_v1"
    assert suite.suite_id == "test_navigation_layouts"
    assert suite.sha256 == file_sha256(suite_path)
    assert suite.max_obstacles == 3
    assert len(suite.layouts) == 2

    layout = suite.layouts[1]
    assert layout.obstacle_centers.shape == (3, 2)
    assert layout.obstacle_radii.shape == (3,)
    np.testing.assert_array_equal(
        layout.obstacle_mask,
        np.asarray([True, False, False], dtype=bool),
    )

    reset_options = layout.reset_options()
    reset_options["obstacle_centers"][0, 0] = 99.0
    assert layout.obstacle_centers[0, 0] == 1.0

#} End function test_layout_suite_loads_and_pads_geometry


# Verify that duplicate scenario identities are rejected before evaluation.
def test_layout_suite_rejects_duplicate_layout_ids(tmp_path) -> None:
#{
    suite_path = tmp_path / "duplicate_layouts.json"
    write_suite(suite_path)
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    data["layouts"][1]["layout_id"] = "open_route"
    suite_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate layout_id"):
        load_navigation_layout_suite(suite_path)

#} End function test_layout_suite_rejects_duplicate_layout_ids


# Verify the complete checkpoint-to-layout CSV and NPZ path with a tiny run.
def test_layout_evaluator_writes_provenance_and_trajectory(monkeypatch, tmp_path) -> None:
#{
    suite_path = tmp_path / "layouts.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "layout_results.csv"
    trajectory_path = tmp_path / "layout_trajectories.npz"
    write_suite(suite_path)

    agent = Agent(obs_dim=21, action_dim=2)
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "obs_dim": 21,
            "action_dim": 2,
            "args": {
                "method": "ppo_baseline",
                "seed": 7,
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
            "evaluate_layout_suite.py",
            "--checkpoint",
            str(checkpoint_path),
            "--layout-suite",
            str(suite_path),
            "--method",
            "ppo_baseline",
            "--train-seed",
            "7",
            "--projection-mode",
            "enabled",
            "--seed",
            "31",
            "--max-episode-steps",
            "1",
            "--collision-penalty",
            "10.0",
            "--no-cuda",
            "--output",
            str(output_path),
            "--trajectory-output",
            str(trajectory_path),
        ],
    )

    main()

    results = pd.read_csv(output_path)
    expected_hash = file_sha256(suite_path)

    assert results["layout_id"].tolist() == ["open_route", "offset_obstacle"]
    assert results["evaluation_seed"].tolist() == [31, 32]
    assert results["episode"].tolist() == [0, 1]
    assert results["method"].tolist() == ["ppo_baseline", "ppo_baseline"]
    assert results["train_seed"].tolist() == [7, 7]
    assert results["layout_suite_sha256"].tolist() == [expected_hash, expected_hash]
    assert results["projection_mode"].tolist() == ["enabled", "enabled"]
    assert results["projection_enabled"].all()
    assert results["training_collision_penalty"].tolist() == [10.0, 10.0]
    assert not results["training_projection_enabled"].any()
    assert results["evaluation_collision_penalty"].tolist() == [10.0, 10.0]
    assert results["evaluation_policy_mode"].tolist() == ["deterministic", "deterministic"]
    assert results["max_episode_steps"].tolist() == [1, 1]

    with np.load(trajectory_path, allow_pickle=False) as archive:
        assert str(archive["trajectory_archive_version"]) == "evaluation_trajectory_v1"
        assert str(archive["run_layout_suite_id"]) == "test_navigation_layouts"
        assert str(archive["run_layout_suite_sha256"]) == expected_hash
        assert str(archive["run_projection_mode"]) == "enabled"
        assert int(archive["episode_count"]) == 2

        first_key, second_key = archive["episode_keys"].tolist()
        assert int(archive[f"{first_key}_episode"]) == 0
        assert int(archive[f"{second_key}_episode"]) == 1
        np.testing.assert_array_equal(
            archive[f"{first_key}_obstacle_mask"],
            np.asarray([False, False, False], dtype=bool),
        )
        np.testing.assert_array_equal(
            archive[f"{second_key}_obstacle_mask"],
            np.asarray([True, False, False], dtype=bool),
        )
        np.testing.assert_allclose(
            archive[f"{second_key}_obstacle_centers"][0],
            np.asarray([1.0, 0.6]),
        )

#} End function test_layout_evaluator_writes_provenance_and_trajectory
