import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.aggregate_projection_results import build_result_tables
from evaluation.layout_suite import load_navigation_layout_suite


#################################################################################
# region Fixtures

# Write one two-layout suite and return its path.
def write_layout_suite(root: Path) -> Path:
#{
    path = root / "layouts.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "navigation_layout_suite_v1",
                "suite_id": "analysis_test_layouts",
                "max_obstacles": 3,
                "agent_radius": 0.1,
                "goal_radius": 0.25,
                "layouts": [
                    {
                        "layout_id": "layout_a",
                        "start": [0.0, 0.0],
                        "theta": 0.0,
                        "goal": [4.0, 0.0],
                        "obstacles": [],
                    },
                    {
                        "layout_id": "layout_b",
                        "start": [0.0, 0.0],
                        "theta": 0.0,
                        "goal": [4.0, 0.0],
                        "obstacles": [
                            {"center": [2.0, 0.3], "radius": 0.25}
                        ],
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

#} End function write_layout_suite


# Write one protocol containing two independent training seeds.
def write_protocol(root: Path, layout_suite_path: Path) -> Path:
#{
    path = root / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "projection_analysis_protocol_v1",
                "study_id": "analysis_test",
                "layout_suite": layout_suite_path.name,
                "expected_train_seeds": [1, 2],
                "expected_repeats_per_layout": 1,
                "evaluation_policy_mode": "deterministic",
                "evaluation_collision_penalty": 10.0,
                "projection_parameters": {
                    "lookahead_distance": 0.25,
                    "alpha": 2.0,
                    "slack_penalty": 1000.0,
                    "extra_clearance": 0.0,
                },
                "representative_layout_id": "layout_b",
                "require_complete_artifacts": False,
                "methods": [
                    {
                        "method": "ppo_baseline",
                        "display_name": "PPO baseline",
                        "required_projection_modes": ["disabled", "enabled"],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

#} End function write_protocol


# Return one exact-schema common-layout episode row.
def episode_row(
    *,
    layout_suite_sha256: str,
    train_seed: int,
    projection_mode: str,
    layout_id: str,
    episode: int,
    episode_return: float,
    success: bool,
    collision: bool,
) -> dict[str, object]:
#{
    enabled = projection_mode == "enabled"
    return {
        "method": "ppo_baseline",
        "train_seed": train_seed,
        "checkpoint": f"arbitrary/checkpoint-{train_seed}.pt",
        "checkpoint_sha256": str(train_seed) * 64,
        "layout_suite_id": "analysis_test_layouts",
        "layout_suite_sha256": layout_suite_sha256,
        "layout_id": layout_id,
        "layout_repeat": 0,
        "evaluation_seed": 1000 + episode,
        "evaluation_collision_penalty": 10.0,
        "evaluation_policy_mode": "deterministic",
        "projection_mode": projection_mode,
        "projection_enabled": enabled,
        "episode": episode,
        "episode_return": episode_return,
        "episode_length": 40 + episode,
        "success": success,
        "collision": collision,
        "min_obstacle_clearance": np.nan if layout_id == "layout_a" else 0.2,
        "action_bound_clipping_count": 0,
        "action_bound_clipping_rate": 0.0,
        "speed_action_bound_clipping_count": 0,
        "speed_action_bound_clipping_rate": 0.0,
        "turn_rate_action_bound_clipping_count": 0,
        "turn_rate_action_bound_clipping_rate": 0.0,
        "mean_action_bound_clipping_norm": 0.0,
        "max_action_bound_clipping_norm": 0.0,
        "projection_intervention_rate": 0.1 if enabled else 0.0,
        "mean_projection_correction_norm": 0.02 if enabled else 0.0,
        "max_projection_correction_norm": 0.05 if enabled else 0.0,
        "mean_projection_slack_sum": 0.001 if enabled else 0.0,
        "max_projection_slack": 0.003 if enabled else 0.0,
        "projection_solver_failure_count": 0,
        "projection_lookahead_distance": 0.25,
        "projection_alpha": 2.0,
        "projection_slack_penalty": 1000.0,
        "projection_extra_clearance": 0.0,
    }

#} End function episode_row


# Write all required CSV shards under arbitrary filenames.
def write_complete_evaluations(root: Path, layout_suite_path: Path) -> list[Path]:
#{
    suite_hash = load_navigation_layout_suite(layout_suite_path).sha256
    evaluation_dir = root / "evaluation"
    evaluation_dir.mkdir()
    paths = []
    returns = {
        (1, "disabled"): [1.0, 3.0],
        (1, "enabled"): [2.0, 5.0],
        (2, "disabled"): [5.0, 7.0],
        (2, "enabled"): [6.0, 10.0],
    }

    for index, ((train_seed, projection_mode), values) in enumerate(returns.items()):
        rows = [
            episode_row(
                layout_suite_sha256=suite_hash,
                train_seed=train_seed,
                projection_mode=projection_mode,
                layout_id="layout_a",
                episode=0,
                episode_return=values[0],
                success=True,
                collision=False,
            ),
            episode_row(
                layout_suite_sha256=suite_hash,
                train_seed=train_seed,
                projection_mode=projection_mode,
                layout_id="layout_b",
                episode=1,
                episode_return=values[1],
                success=projection_mode == "enabled",
                collision=projection_mode == "disabled",
            ),
        ]
        path = evaluation_dir / f"unrelated-name-{index}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths.append(path)

    pd.DataFrame({"other": [1]}).to_csv(evaluation_dir / "not-an-evaluation.csv", index=False)
    return paths

#} End function write_complete_evaluations

# end region Fixtures


#################################################################################
# region Tests

# Layouts are averaged within checkpoints before independent training seeds are aggregated.
def test_result_build_aggregates_layouts_before_seeds_and_preserves_pairs(tmp_path: Path):
#{
    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    write_complete_evaluations(tmp_path, layout_suite_path)
    output_dir = tmp_path / "tables"
    outputs = build_result_tables(protocol_path, tmp_path / "evaluation", output_dir)
    checkpoint_summary = pd.read_csv(outputs["checkpoints"])
    method_summary = pd.read_csv(outputs["methods"])
    paired = pd.read_csv(outputs["paired"])
    paired_summary = pd.read_csv(outputs["paired_summary"])
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))

    disabled_checkpoints = checkpoint_summary[
        checkpoint_summary["projection_mode"] == "disabled"
    ].sort_values("train_seed")
    np.testing.assert_allclose(disabled_checkpoints["episode_return"], [2.0, 6.0])

    disabled_method = method_summary[
        method_summary["projection_mode"] == "disabled"
    ].iloc[0]
    assert disabled_method["seed_count"] == 2
    assert disabled_method["episode_return_mean"] == pytest.approx(4.0)
    assert disabled_method["episode_return_std"] == pytest.approx(np.sqrt(8.0))

    paired = paired.sort_values("train_seed")
    np.testing.assert_allclose(
        paired["episode_return_delta_enabled_minus_disabled"],
        [1.5, 2.0],
    )
    assert paired_summary.iloc[0][
        "episode_return_delta_enabled_minus_disabled_mean"
    ] == pytest.approx(1.75)
    assert outputs["method_latex"].read_text(encoding="utf-8").strip()
    paired_latex = outputs["paired_latex"].read_text(encoding="utf-8")
    assert paired_latex.strip()
    assert "PPO baseline & " in paired_latex
    assert any(
        line.startswith("PPO baseline & ") and line.endswith('\\\\')
        for line in paired_latex.splitlines()
    )
    assert audit["selected_csv_count"] == 4
    assert audit["discovered_csv_count"] == 5
    assert audit["projection_solver_failure_count"] == 0

#} End function test_result_build_aggregates_layouts_before_seeds_and_preserves_pairs


# A final table must not be produced from incomplete method/seed/layout coverage.
def test_result_build_rejects_missing_layout_coverage(tmp_path: Path):
#{
    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    paths = write_complete_evaluations(tmp_path, layout_suite_path)
    frame = pd.read_csv(paths[0]).iloc[:1]
    frame.to_csv(paths[0], index=False)

    with pytest.raises(ValueError, match="Expected 2 rows"):
        build_result_tables(protocol_path, tmp_path / "evaluation", tmp_path / "tables")

#} End function test_result_build_rejects_missing_layout_coverage


# Duplicate episode keys are rejected even when generated files have arbitrary names.
def test_result_build_rejects_duplicate_episode_keys(tmp_path: Path):
#{
    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    paths = write_complete_evaluations(tmp_path, layout_suite_path)
    frame = pd.read_csv(paths[0])
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(paths[0], index=False)

    with pytest.raises(ValueError, match="Duplicate common-layout episode keys"):
        build_result_tables(protocol_path, tmp_path / "evaluation", tmp_path / "tables")

#} End function test_result_build_rejects_duplicate_episode_keys


# Every method and seed must use one identical layout/repeat/evaluation-seed key set.
def test_result_build_rejects_cross_seed_evaluation_key_mismatch(tmp_path: Path):
#{
    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    paths = write_complete_evaluations(tmp_path, layout_suite_path)
    frame = pd.read_csv(paths[2])
    frame.loc[0, "evaluation_seed"] = 9999
    frame.to_csv(paths[2], index=False)

    with pytest.raises(ValueError, match="Evaluation keys differ"):
        build_result_tables(protocol_path, tmp_path / "evaluation", tmp_path / "tables")
#} End function test_result_build_rejects_cross_seed_evaluation_key_mismatch


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("episode_return", np.nan, "episode_return must be finite"),
        (
            "projection_solver_failure_count",
            np.nan,
            "projection_solver_failure_count must contain finite integers",
        ),
        (
            "mean_projection_correction_norm",
            np.nan,
            "mean_projection_correction_norm must be finite",
        ),
    ],
)
# Non-finite evidence fields cannot be interpreted as valid zero-valued results.
def test_result_build_rejects_nonfinite_evidence(
    tmp_path: Path,
    column: str,
    value: float,
    message: str,
):
#{
    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    paths = write_complete_evaluations(tmp_path, layout_suite_path)
    frame = pd.read_csv(paths[1])
    frame.loc[0, column] = value
    frame.to_csv(paths[1], index=False)

    with pytest.raises(ValueError, match=message):
        build_result_tables(protocol_path, tmp_path / "evaluation", tmp_path / "tables")
#} End function test_result_build_rejects_nonfinite_evidence


# A completed result-table directory is never overwritten silently.
def test_result_build_refuses_nonempty_output_directory(tmp_path: Path):
#{
    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    write_complete_evaluations(tmp_path, layout_suite_path)
    output_dir = tmp_path / "tables"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        build_result_tables(protocol_path, tmp_path / "evaluation", output_dir)
#} End function test_result_build_refuses_nonempty_output_directory


# Files from another layout suite are ignored rather than selected by filename conventions.
def test_result_build_discovers_artifacts_from_metadata_not_filenames(tmp_path: Path):
#{
    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    paths = write_complete_evaluations(tmp_path, layout_suite_path)
    foreign = pd.read_csv(paths[0])
    foreign["layout_suite_sha256"] = "f" * 64
    foreign.to_csv(tmp_path / "evaluation" / "looks-important.csv", index=False)
    outputs = build_result_tables(protocol_path, tmp_path / "evaluation", tmp_path / "tables")
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))

    assert audit["selected_csv_count"] == 4
    assert any(
        item["reason"] == "different_layout_suite"
        for item in audit["ignored_csvs"]
    )

#} End function test_result_build_discovers_artifacts_from_metadata_not_filenames

# end region Tests


# TensorBoard runs are matched by recorded checkpoint metadata without importing PyTorch.
def test_training_curve_discovery_uses_tensorboard_checkpoint_metadata(tmp_path: Path, monkeypatch):
#{
    import builtins
    import hashlib

    from analysis import training_diagnostics

    runs_dir = tmp_path / "runs"
    checkpoint_dir = runs_dir / "checkpoints"
    run_dir = runs_dir / "arbitrary-tensorboard-directory"
    checkpoint_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    (run_dir / "events.out.tfevents.test").write_bytes(b"event-placeholder")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_summary = pd.DataFrame(
        [
            {
                "method": "ppo_baseline",
                "display_name": "PPO baseline",
                "train_seed": 1,
                "checkpoint_sha256": checkpoint_hash,
            }
        ]
    )

    def fake_tensorboard_run(path: Path):
    #{
        assert path == run_dir
        return (
            {
                "checkpoint_path": "runs/checkpoints/model.pt",
                "seed": "1",
                "method": "ppo_baseline",
            },
            {
                "charts/episodic_return": pd.DataFrame(
                    {"step": [10, 20], "value": [1.0, 2.0]}
                )
            },
        )

    #} End function fake_tensorboard_run

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
    #{
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Result plotting must not import PyTorch.")
        return original_import(name, *args, **kwargs)

    #} End function guarded_import

    monkeypatch.setattr(training_diagnostics, "load_tensorboard_run", fake_tensorboard_run)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    points, skipped = training_diagnostics.training_points(checkpoint_summary, runs_dir)

    assert {
        item["tag"]
        for item in skipped
        if item.get("reason") == "scalar_not_found"
    } == {
        "charts/episodic_length",
        "safety/success",
        "safety/collision",
        "safety/timeout",
        "action_bounds/clipping_frequency",
        "action_bounds/speed_clipping_frequency",
        "action_bounds/turn_rate_clipping_frequency",
        "action_bounds/clipping_norm",
        "action_bounds/clipping_norm_max",
    }
    assert points["tag"].unique().tolist() == ["charts/episodic_return"]
    assert points["step"].tolist() == [10, 20]
    assert points["value"].tolist() == [1.0, 2.0]

#} End function test_training_curve_discovery_uses_tensorboard_checkpoint_metadata




# Development builds tolerate the absence of optional TensorBoard curve data.
def test_training_curve_aggregation_accepts_empty_optional_data() -> None:
#{
    from analysis.training_diagnostics import aggregate_curve

    assert aggregate_curve(pd.DataFrame(), "charts/episodic_return").empty
#} End function test_training_curve_aggregation_accepts_empty_optional_data

# Final protocols require a TensorBoard training-return source for every checkpoint.
def test_final_result_build_rejects_missing_training_run(tmp_path: Path) -> None:
#{
    from analysis.training_diagnostics import training_points

    checkpoint_summary = pd.DataFrame(
        [
            {
                "method": "ppo_baseline",
                "display_name": "PPO baseline",
                "train_seed": 1,
                "checkpoint_sha256": "1" * 64,
            }
        ]
    )
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    with pytest.raises(ValueError, match="Required TensorBoard run was not found"):
        training_points(checkpoint_summary, runs_dir, require_complete=True)
#} End function test_final_result_build_rejects_missing_training_run


# Final protocols reject partial representative-trajectory evidence.
def test_final_result_build_rejects_missing_representative_trajectory(tmp_path: Path) -> None:
#{
    from analysis.aggregate_projection_results import load_protocol
    from analysis.plot_projection_results import plot_trajectories

    layout_suite_path = write_layout_suite(tmp_path)
    protocol_path = write_protocol(tmp_path, layout_suite_path)
    protocol = load_protocol(protocol_path)
    protocol["require_complete_artifacts"] = True
    paths = write_complete_evaluations(tmp_path, layout_suite_path)
    episodes = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    with pytest.raises(ValueError, match="Required representative trajectories are incomplete"):
        plot_trajectories(
            protocol,
            episodes,
            tmp_path / "evaluation",
            tmp_path / "figure.pdf",
            tmp_path / "selection.csv",
            require_complete=True,
        )
#} End function test_final_result_build_rejects_missing_representative_trajectory
