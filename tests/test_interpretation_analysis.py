"""Synthetic and portability tests for the explicit-input interpretation CLIs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.interpretation._evaluation_evidence import (
    EvidenceError,
    canonical_json_sha256,
    compare_numeric_exact_nan,
    load_design,
    load_episode_table,
    reconstruct_checkpoint_summary,
    reconstruct_method_summary,
    reconstruct_paired_deltas,
    reconstruct_paired_summary,
)
from analysis.interpretation.summarize_layout_transfer_performance import (
    _effect_concentration,
)
from analysis.interpretation.summarize_paired_terminal_outcome_correspondences import (
    _conditional_matrix,
    _pair_episode_outcomes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "analysis" / "interpretation"
INVENTORY_SCRIPT = SCRIPT_DIRECTORY / "inventory_evaluation_table.py"
VALIDATION_SCRIPT = SCRIPT_DIRECTORY / "validate_evaluation_tables.py"
SUMMARY_SCRIPT = SCRIPT_DIRECTORY / "summarize_absolute_evaluation_performance.py"
PAIRED_EFFECT_SCRIPT = SCRIPT_DIRECTORY / "summarize_paired_projection_effects.py"
LAYOUT_TRANSFER_SCRIPT = SCRIPT_DIRECTORY / "summarize_layout_transfer_performance.py"
TERMINAL_CORRESPONDENCE_SCRIPT = (
    SCRIPT_DIRECTORY / "summarize_paired_terminal_outcome_correspondences.py"
)
RESULT_SCHEMA = "synthetic_result_tables_v1"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_synthetic_evidence(directory: Path) -> dict[str, Path]:
    """Build a complete tiny evidence family under arbitrary filenames."""

    directory.mkdir(parents=True)
    layout_suite = {
        "schema_version": "synthetic_layout_suite_v1",
        "suite_id": "synthetic_worlds",
        "max_obstacles": 2,
        "agent_radius": 0.1,
        "goal_radius": 0.2,
        "layouts": [
            {
                "layout_id": "corridor_a",
                "obstacles": [{"center": [1.0, 0.0], "radius": 0.2}],
            },
            {"layout_id": "open_b", "obstacles": []},
        ],
    }
    protocol = {
        "schema_version": "projection_analysis_protocol_v1",
        "study_id": "synthetic_study",
        "layout_suite": "not-used-for-file-discovery.json",
        "expected_train_seeds": [7, 11],
        "expected_repeats_per_layout": 2,
        "evaluation_policy_mode": "synthetic_mode",
        "evaluation_collision_penalty": 3.0,
        "evaluation_base_seed": 400,
        "evaluation_last_seed": 403,
        "max_episode_steps": 20,
        "projection_parameters": {
            "lookahead_distance": 0.4,
            "alpha": 1.5,
            "slack_penalty": 250.0,
            "extra_clearance": 0.02,
        },
        "methods": [
            {
                "method": "method_alpha",
                "display_name": "Method Alpha",
                "training_collision_penalty": 3.0,
                "training_projection_enabled": False,
                "required_projection_modes": ["disabled", "enabled"],
            },
            {
                "method": "method_beta",
                "display_name": "Method Beta",
                "training_collision_penalty": 9.0,
                "training_projection_enabled": True,
                "required_projection_modes": ["disabled", "enabled"],
            },
        ],
    }
    paths = {
        "protocol": directory / "design-source.weird.json",
        "layout_suite": directory / "world-catalog.unusual.json",
        "episodes": directory / "observations with spaces.csv",
        "checkpoint_summary": directory / "policies-summary.csv",
        "method_summary": directory / "controller-summary.csv",
        "paired_deltas": directory / "paired-effects.csv",
        "paired_summary": directory / "paired-effects-summary.csv",
        "build_audit": directory / "producer-audit.json",
    }
    _write_json(paths["protocol"], protocol)
    _write_json(paths["layout_suite"], layout_suite)

    layout_hash = canonical_json_sha256(layout_suite)
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(protocol["methods"]):
        for seed_index, train_seed in enumerate(protocol["expected_train_seeds"]):
            checkpoint = f"arbitrary-vault/{method['method']}-{train_seed}.weights"
            checkpoint_hash = _sha(f"{method['method']}::{train_seed}")
            for mode_index, projection_mode in enumerate(("disabled", "enabled")):
                episode_index = 0
                for layout in layout_suite["layouts"]:
                    for layout_repeat in range(protocol["expected_repeats_per_layout"]):
                        episode_length = 5 + episode_index
                        outcome = (
                            method_index + seed_index + mode_index + episode_index
                        ) % 3
                        success = outcome == 0
                        collision = outcome == 1
                        truncated = outcome == 2
                        clipping_count = episode_index % 2
                        intervention_count = (
                            1
                            if projection_mode == "enabled" and episode_index % 2 == 0
                            else 0
                        )
                        correction_mean = 0.1 if intervention_count else 0.0
                        correction_max = 0.2 if intervention_count else 0.0
                        slack = 0.001 if intervention_count else 0.0
                        rows.append(
                            {
                                "method": method["method"],
                                "train_seed": train_seed,
                                "training_collision_penalty": method[
                                    "training_collision_penalty"
                                ],
                                "training_projection_enabled": method[
                                    "training_projection_enabled"
                                ],
                                "checkpoint": checkpoint,
                                "checkpoint_sha256": checkpoint_hash,
                                "layout_suite_schema_version": layout_suite[
                                    "schema_version"
                                ],
                                "layout_suite_id": layout_suite["suite_id"],
                                "layout_suite_sha256": layout_hash,
                                "layout_id": layout["layout_id"],
                                "layout_repeat": layout_repeat,
                                "evaluation_seed": 400 + episode_index,
                                "evaluation_collision_penalty": 3.0,
                                "evaluation_policy_mode": "synthetic_mode",
                                "max_episode_steps": 20,
                                "layout_max_obstacles": 2,
                                "layout_agent_radius": 0.1,
                                "layout_goal_radius": 0.2,
                                "projection_mode": projection_mode,
                                "episode": episode_index,
                                "seed": 400 + episode_index,
                                "episode_return": float(
                                    method_index * 2
                                    + seed_index
                                    + mode_index
                                    - episode_index
                                ),
                                "episode_length": episode_length,
                                "success": success,
                                "collision": collision,
                                "terminated": success or collision,
                                "truncated": truncated,
                                "final_distance_to_goal": 0.0 if success else 1.0,
                                "min_obstacle_clearance": (
                                    np.nan
                                    if not layout["obstacles"]
                                    else 0.25 - 0.01 * episode_index
                                ),
                                "action_bound_clipping_count": clipping_count,
                                "action_bound_clipping_rate": clipping_count
                                / episode_length,
                                "speed_action_bound_clipping_count": clipping_count,
                                "speed_action_bound_clipping_rate": clipping_count
                                / episode_length,
                                "turn_rate_action_bound_clipping_count": 0,
                                "turn_rate_action_bound_clipping_rate": 0.0,
                                "mean_action_bound_clipping_norm": (
                                    0.02 if clipping_count else 0.0
                                ),
                                "max_action_bound_clipping_norm": (
                                    0.03 if clipping_count else 0.0
                                ),
                                "projection_enabled": projection_mode == "enabled",
                                "projection_intervention_count": intervention_count,
                                "projection_intervention_rate": intervention_count
                                / episode_length,
                                "mean_projection_correction_norm": correction_mean,
                                "max_projection_correction_norm": correction_max,
                                "mean_projection_slack_sum": slack,
                                "max_projection_slack": slack,
                                "projection_solver_failure_count": 0,
                                "projection_lookahead_distance": 0.4,
                                "projection_alpha": 1.5,
                                "projection_slack_penalty": 250.0,
                                "projection_extra_clearance": 0.02,
                                "result_build_schema_version": RESULT_SCHEMA,
                            }
                        )
                        episode_index += 1
    episodes = pd.DataFrame(rows)
    episodes.to_csv(paths["episodes"], index=False)

    design = load_design(str(paths["protocol"]), str(paths["layout_suite"]))
    parsed_episodes = load_episode_table(str(paths["episodes"]))
    checkpoints = reconstruct_checkpoint_summary(parsed_episodes)
    checkpoints.insert(0, "result_build_schema_version", RESULT_SCHEMA)
    checkpoints.insert(
        2,
        "display_name",
        checkpoints["method"].map(
            {method["method"]: method["display_name"] for method in protocol["methods"]}
        ),
    )
    checkpoints.to_csv(paths["checkpoint_summary"], index=False)

    methods = reconstruct_method_summary(design, checkpoints)
    methods.insert(0, "result_build_schema_version", RESULT_SCHEMA)
    methods.insert(
        2,
        "display_name",
        methods["method"].map(
            {method["method"]: method["display_name"] for method in protocol["methods"]}
        ),
    )
    methods.to_csv(paths["method_summary"], index=False)

    paired = reconstruct_paired_deltas(design, parsed_episodes)
    paired.insert(0, "result_build_schema_version", RESULT_SCHEMA)
    paired.insert(
        2,
        "display_name",
        paired["method"].map(
            {method["method"]: method["display_name"] for method in protocol["methods"]}
        ),
    )
    paired.to_csv(paths["paired_deltas"], index=False)

    paired_summary = reconstruct_paired_summary(paired)
    paired_summary.insert(0, "result_build_schema_version", RESULT_SCHEMA)
    paired_summary.insert(
        2,
        "display_name",
        paired_summary["method"].map(
            {method["method"]: method["display_name"] for method in protocol["methods"]}
        ),
    )
    paired_summary.to_csv(paths["paired_summary"], index=False)

    audit = {
        "status": "PASS",
        "result_build_schema_version": RESULT_SCHEMA,
        "protocol_sha256": canonical_json_sha256(protocol),
        "layout_suite_id": layout_suite["suite_id"],
        "layout_suite_sha256": layout_hash,
        "layout_count": len(layout_suite["layouts"]),
        "expected_train_seeds": protocol["expected_train_seeds"],
        "episode_row_count": len(episodes),
        "checkpoint_row_count": len(checkpoints),
        "method_row_count": len(methods),
        "paired_checkpoint_row_count": len(paired),
        "paired_method_row_count": len(paired_summary),
        "projection_solver_failure_count": 0,
    }
    _write_json(paths["build_audit"], audit)
    return paths


def relative_arguments(paths: dict[str, Path], working_directory: Path) -> dict[str, str]:
    return {
        name: os.path.relpath(path, working_directory)
        for name, path in paths.items()
    }


def validation_arguments(
    evidence: dict[str, Path],
    output: Path,
    *,
    episodes: Path | None = None,
    protocol: Path | None = None,
) -> list[str]:
    return [
        "--episodes",
        str(episodes or evidence["episodes"]),
        "--protocol",
        str(protocol or evidence["protocol"]),
        "--layout-suite",
        str(evidence["layout_suite"]),
        "--checkpoint-summary",
        str(evidence["checkpoint_summary"]),
        "--method-summary",
        str(evidence["method_summary"]),
        "--paired-deltas",
        str(evidence["paired_deltas"]),
        "--paired-summary",
        str(evidence["paired_summary"]),
        "--build-audit",
        str(evidence["build_audit"]),
        "--output",
        str(output),
    ]


def correspondence_arguments(
    evidence: dict[str, Path],
    output: Path | str,
    *,
    episodes: Path | str | None = None,
    checkpoint_summary: Path | str | None = None,
    method_summary: Path | str | None = None,
    paired_deltas: Path | str | None = None,
    paired_summary: Path | str | None = None,
    build_audit: Path | str | None = None,
) -> list[str]:
    return [
        "--episodes",
        str(episodes or evidence["episodes"]),
        "--protocol",
        str(evidence["protocol"]),
        "--layout-suite",
        str(evidence["layout_suite"]),
        "--checkpoint-summary",
        str(checkpoint_summary or evidence["checkpoint_summary"]),
        "--method-summary",
        str(method_summary or evidence["method_summary"]),
        "--paired-deltas",
        str(paired_deltas or evidence["paired_deltas"]),
        "--paired-summary",
        str(paired_summary or evidence["paired_summary"]),
        "--build-audit",
        str(build_audit or evidence["build_audit"]),
        "--output",
        str(output),
    ]


class InterpretationAnalysisTests(unittest.TestCase):
    def run_cli(
        self,
        script: Path,
        arguments: list[str],
        working_directory: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_all_entrypoints_run_from_unrelated_directory_with_arbitrary_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "evidence folder x")
            working = root / "unrelated current directory"
            working.mkdir()
            inputs = relative_arguments(evidence, working)
            comparison = root / "more evidence" / "different-name.csv"
            comparison.parent.mkdir()
            shutil.copyfile(evidence["episodes"], comparison)
            comparison_argument = os.path.relpath(comparison, working)

            inventory_output = "outputs/schema result.json"
            inventory = self.run_cli(
                INVENTORY_SCRIPT,
                [
                    "--episodes",
                    inputs["episodes"],
                    "--output",
                    inventory_output,
                    "--label",
                    "arbitrary inventory",
                ],
                working,
            )
            self.assertEqual(inventory.returncode, 0, inventory.stderr)
            inventory_data = json.loads((working / inventory_output).read_text())
            self.assertEqual(inventory_data["input"]["path"], inputs["episodes"])
            self.assertEqual(inventory_data["row_count"], 32)

            validation_output = "outputs/validation result.json"
            validation = self.run_cli(
                VALIDATION_SCRIPT,
                [
                    "--episodes",
                    inputs["episodes"],
                    "--protocol",
                    inputs["protocol"],
                    "--layout-suite",
                    inputs["layout_suite"],
                    "--checkpoint-summary",
                    inputs["checkpoint_summary"],
                    "--method-summary",
                    inputs["method_summary"],
                    "--paired-deltas",
                    inputs["paired_deltas"],
                    "--paired-summary",
                    inputs["paired_summary"],
                    "--build-audit",
                    inputs["build_audit"],
                    "--compare-checkpoints-with",
                    comparison_argument,
                    "--output",
                    validation_output,
                ],
                working,
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)
            validation_data = json.loads((working / validation_output).read_text())
            self.assertEqual(validation_data["status"], "PASS")
            self.assertEqual(validation_data["design"]["expected_episode_rows"], 32)
            self.assertTrue(
                validation_data["cross_table_checkpoint_comparisons"][0]["exact_match"]
            )

            summary_output = "outputs/absolute result.json"
            summary = self.run_cli(
                SUMMARY_SCRIPT,
                [
                    "--episodes",
                    inputs["episodes"],
                    "--protocol",
                    inputs["protocol"],
                    "--layout-suite",
                    inputs["layout_suite"],
                    "--checkpoint-summary",
                    inputs["checkpoint_summary"],
                    "--method-summary",
                    inputs["method_summary"],
                    "--worked-example-method",
                    "method_beta",
                    "--worked-example-projection-mode",
                    "enabled",
                    "--output",
                    summary_output,
                ],
                working,
            )
            self.assertEqual(summary.returncode, 0, summary.stderr)
            summary_data = json.loads((working / summary_output).read_text())
            self.assertEqual(len(summary_data["absolute_method_summary"]), 4)
            self.assertEqual(
                summary_data["absolute_method_summary"][0]["training_run_count"],
                2,
            )
            self.assertEqual(
                summary_data["replication"]["independent_unit"],
                "training_run_represented_by_final_checkpoint",
            )
            self.assertEqual(summary_data["worked_sample_sd"]["method"], "method_beta")
            self.assertEqual(
                summary_data["inputs"]["protocol"]["path"], inputs["protocol"]
            )

            paired_output = "outputs/paired effect result.json"
            paired_effect = self.run_cli(
                PAIRED_EFFECT_SCRIPT,
                [
                    "--episodes",
                    inputs["episodes"],
                    "--protocol",
                    inputs["protocol"],
                    "--layout-suite",
                    inputs["layout_suite"],
                    "--paired-deltas",
                    inputs["paired_deltas"],
                    "--paired-summary",
                    inputs["paired_summary"],
                    "--worked-example-method",
                    "method_alpha",
                    "--worked-example-train-seed",
                    "7",
                    "--output",
                    paired_output,
                ],
                working,
            )
            self.assertEqual(paired_effect.returncode, 0, paired_effect.stderr)
            paired_data = json.loads((working / paired_output).read_text())
            self.assertEqual(len(paired_data["training_run_paired_effects"]), 4)
            self.assertEqual(paired_data["coverage"]["training_run_count_per_method"], 2)
            self.assertEqual(
                paired_data["coverage"]["independent_unit"],
                "training_run_represented_by_final_checkpoint",
            )
            self.assertTrue(paired_data["outcome_identity_checks"]["all_checks_pass"])
            self.assertEqual(
                paired_data["worked_outcome_pair"]["train_seed"], 7
            )

            layout_output = "outputs/layout transfer result.json"
            layout_transfer = self.run_cli(
                LAYOUT_TRANSFER_SCRIPT,
                [
                    "--episodes",
                    inputs["episodes"],
                    "--protocol",
                    inputs["protocol"],
                    "--layout-suite",
                    inputs["layout_suite"],
                    "--checkpoint-summary",
                    inputs["checkpoint_summary"],
                    "--method-summary",
                    inputs["method_summary"],
                    "--paired-deltas",
                    inputs["paired_deltas"],
                    "--paired-summary",
                    inputs["paired_summary"],
                    "--build-audit",
                    inputs["build_audit"],
                    "--worked-example-method",
                    "method_alpha",
                    "--worked-example-train-seed",
                    "7",
                    "--worked-example-layout",
                    "corridor_a",
                    "--output",
                    layout_output,
                ],
                working,
            )
            self.assertEqual(layout_transfer.returncode, 0, layout_transfer.stderr)
            layout_data = json.loads((working / layout_output).read_text())
            self.assertTrue(layout_data["aggregation_identity_checks"]["all_checks_pass"])
            self.assertEqual(layout_data["coverage"]["training_run_count_per_method"], 2)
            self.assertEqual(
                layout_data["coverage"]["independent_unit"],
                "training_run_represented_by_final_checkpoint",
            )
            self.assertEqual(layout_data["worked_example"]["layout_id"], "corridor_a")
            self.assertEqual(
                layout_data["inputs"]["build_audit"]["path"], inputs["build_audit"]
            )

            correspondence_output = "outputs/terminal correspondence result.json"
            terminal_correspondence = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                [
                    "--episodes",
                    inputs["episodes"],
                    "--protocol",
                    inputs["protocol"],
                    "--layout-suite",
                    inputs["layout_suite"],
                    "--checkpoint-summary",
                    inputs["checkpoint_summary"],
                    "--method-summary",
                    inputs["method_summary"],
                    "--paired-deltas",
                    inputs["paired_deltas"],
                    "--paired-summary",
                    inputs["paired_summary"],
                    "--build-audit",
                    inputs["build_audit"],
                    "--output",
                    correspondence_output,
                    "--label",
                    "arbitrary correspondence analysis",
                ],
                working,
            )
            self.assertEqual(
                terminal_correspondence.returncode, 0, terminal_correspondence.stderr
            )
            correspondence_data = json.loads(
                (working / correspondence_output).read_text(encoding="utf-8")
            )
            self.assertEqual(
                correspondence_data["analysis_schema"],
                "paired_terminal_outcome_correspondences_v1",
            )
            self.assertEqual(correspondence_data["status"], "PASS")
            self.assertEqual(
                correspondence_data["coverage"]["independent_unit"],
                "training_run_represented_by_final_checkpoint",
            )
            self.assertEqual(
                correspondence_data["coverage"]["total_paired_evaluation_observations"],
                16,
            )
            self.assertEqual(
                correspondence_data["inputs"]["episodes"]["path"], inputs["episodes"]
            )
            self.assertEqual(
                correspondence_data["label"], "arbitrary correspondence analysis"
            )

    def test_terminal_correspondence_matrix_matches_hand_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            output = root / "correspondences.json"
            process = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                correspondence_arguments(evidence, output),
                root,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            training_run = next(
                row
                for row in result["training_run_correspondence_matrices"]
                if row["method"] == "method_alpha" and row["train_seed"] == 7
            )
            self.assertEqual(
                training_run["correspondence_count_matrix"],
                [[0, 2, 0], [0, 0, 1], [1, 0, 0]],
            )
            self.assertEqual(training_run["projection_disabled_outcome_counts"], [2, 1, 1])
            self.assertEqual(training_run["projection_enabled_outcome_counts"], [1, 2, 1])
            self.assertEqual(
                training_run["enabled_minus_disabled_outcome_count_deltas"],
                [-1, 1, 0],
            )
            self.assertEqual(len(training_run["correspondence_cells"]), 9)
            zero_cell = next(
                cell
                for cell in training_run["correspondence_cells"]
                if cell["projection_disabled_outcome"] == "success"
                and cell["projection_enabled_outcome"] == "success"
            )
            self.assertEqual(zero_cell["count"], 0)

            method = next(
                row
                for row in result["method_correspondence_summaries"]
                if row["method"] == "method_alpha"
            )
            self.assertEqual(
                method["pooled_correspondence_count_matrix"],
                [[0, 3, 0], [0, 0, 3], [2, 0, 0]],
            )
            success_to_collision = next(
                cell
                for cell in method["training_run_correspondence_cell_summaries"]
                if cell["projection_disabled_outcome"] == "success"
                and cell["projection_enabled_outcome"] == "collision"
            )
            self.assertAlmostEqual(
                success_to_collision["mean_training_run_proportion_of_all_pairs"],
                0.375,
            )
            self.assertAlmostEqual(
                success_to_collision[
                    "sample_sd_training_run_proportion_of_all_pairs"
                ],
                math.sqrt(0.03125),
            )
            self.assertTrue(
                result["reconciliation"]["terminal_outcome_correspondences"][
                    "all_checks_pass"
                ]
            )
            self.assertEqual(
                _conditional_matrix(
                    np.asarray([[0, 0, 0], [1, 1, 0], [0, 0, 2]], dtype=int)
                )[0],
                [None, None, None],
            )

    def test_terminal_correspondence_rejects_missing_or_duplicate_pairs_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            original = pd.read_csv(evidence["episodes"])

            missing = root / "missing-pair.csv"
            original.iloc[:-1].to_csv(missing, index=False)
            missing_output = root / "missing-output.json"
            missing_process = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                correspondence_arguments(evidence, missing_output, episodes=missing),
                root,
            )
            self.assertNotEqual(missing_process.returncode, 0)
            self.assertFalse(missing_output.exists())

            duplicated = root / "duplicate-pair.csv"
            pd.concat([original, original.iloc[[0]]], ignore_index=True).to_csv(
                duplicated, index=False
            )
            duplicate_output = root / "duplicate-output.json"
            duplicate_process = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                correspondence_arguments(evidence, duplicate_output, episodes=duplicated),
                root,
            )
            self.assertNotEqual(duplicate_process.returncode, 0)
            self.assertFalse(duplicate_output.exists())

    def test_terminal_correspondence_pairing_checks_are_exercised_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            design = load_design(
                str(evidence["protocol"]), str(evidence["layout_suite"])
            )
            episodes = load_episode_table(str(evidence["episodes"]))

            duplicate_disabled = pd.concat(
                [
                    episodes,
                    episodes[episodes["projection_mode"].eq("disabled")].iloc[[0]],
                ],
                ignore_index=True,
            )
            with self.assertRaisesRegex(
                EvidenceError, "disabled terminal-outcome pairing keys are duplicated"
            ):
                _pair_episode_outcomes(design, duplicate_disabled)

            enabled_index = episodes[
                episodes["projection_mode"].eq("enabled")
            ].index[0]
            missing_enabled = episodes.drop(index=enabled_index)
            with self.assertRaisesRegex(
                EvidenceError, "terminal-outcome pair is missing"
            ):
                _pair_episode_outcomes(design, missing_enabled)

    def test_terminal_correspondence_rejects_invalid_outcome_partition_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            changed = pd.read_csv(evidence["episodes"])
            changed.loc[0, "success"] = True
            changed.loc[0, "collision"] = True
            changed.loc[0, "terminated"] = True
            malformed = root / "invalid-outcome.csv"
            changed.to_csv(malformed, index=False)
            output = root / "must-not-exist.json"
            process = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                correspondence_arguments(evidence, output, episodes=malformed),
                root,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("must equal one", process.stderr.lower())
            self.assertFalse(output.exists())

    def test_terminal_correspondence_rejects_summary_and_audit_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")

            checkpoint = pd.read_csv(evidence["checkpoint_summary"])
            checkpoint.loc[0, "success_rate"] += 0.125
            checkpoint_path = root / "tampered-checkpoint.csv"
            checkpoint.to_csv(checkpoint_path, index=False)

            paired = pd.read_csv(evidence["paired_deltas"])
            paired.loc[0, "success_delta_enabled_minus_disabled"] += 0.125
            paired_path = root / "tampered-paired.csv"
            paired.to_csv(paired_path, index=False)

            method = pd.read_csv(evidence["method_summary"])
            method.loc[0, "success_rate_mean"] += 0.125
            method_path = root / "tampered-method.csv"
            method.to_csv(method_path, index=False)

            paired_summary = pd.read_csv(evidence["paired_summary"])
            paired_summary.loc[
                0, "success_delta_enabled_minus_disabled_mean"
            ] += 0.125
            paired_summary_path = root / "tampered-paired-summary.csv"
            paired_summary.to_csv(paired_summary_path, index=False)

            audit = json.loads(evidence["build_audit"].read_text(encoding="utf-8"))
            audit["episode_row_count"] += 1
            audit_path = root / "tampered-audit.json"
            _write_json(audit_path, audit)

            cases = (
                {"checkpoint_summary": checkpoint_path},
                {"method_summary": method_path},
                {"paired_deltas": paired_path},
                {"paired_summary": paired_summary_path},
                {"build_audit": audit_path},
            )
            for index, overrides in enumerate(cases):
                with self.subTest(overrides=overrides):
                    output = root / f"tampered-output-{index}.json"
                    process = self.run_cli(
                        TERMINAL_CORRESPONDENCE_SCRIPT,
                        correspondence_arguments(evidence, output, **overrides),
                        root,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertFalse(output.exists())

    def test_terminal_correspondence_is_deterministic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            first_output = root / "first.json"
            second_output = root / "second.json"
            first = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                correspondence_arguments(evidence, first_output),
                root,
            )
            second = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                correspondence_arguments(evidence, second_output),
                root,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())

            first_output.write_text("preserve me\n", encoding="utf-8")
            existing = self.run_cli(
                TERMINAL_CORRESPONDENCE_SCRIPT,
                correspondence_arguments(evidence, first_output),
                root,
            )
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn("refusing to overwrite", existing.stderr)
            self.assertEqual(first_output.read_text(encoding="utf-8"), "preserve me\n")

    def test_hand_calculated_training_run_method_loo_and_paired_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            design = load_design(
                str(evidence["protocol"]),
                str(evidence["layout_suite"]),
            )
            episodes = load_episode_table(str(evidence["episodes"]))
            checkpoints = reconstruct_checkpoint_summary(episodes)
            methods = reconstruct_method_summary(design, checkpoints)
            paired = reconstruct_paired_deltas(design, episodes)
            paired_methods = reconstruct_paired_summary(paired)

            beta_enabled_seed7 = checkpoints[
                checkpoints["method"].eq("method_beta")
                & checkpoints["projection_mode"].eq("enabled")
                & checkpoints["train_seed"].eq(7)
            ].iloc[0]
            self.assertEqual(int(beta_enabled_seed7["episode_count"]), 4)
            self.assertAlmostEqual(float(beta_enabled_seed7["success_rate"]), 0.25)
            self.assertAlmostEqual(float(beta_enabled_seed7["collision_rate"]), 0.25)
            self.assertAlmostEqual(float(beta_enabled_seed7["episode_return"]), 1.5)
            self.assertAlmostEqual(float(beta_enabled_seed7["episode_length"]), 6.5)
            self.assertAlmostEqual(
                float(beta_enabled_seed7["min_obstacle_clearance"]), 0.245
            )

            beta_enabled = methods[
                methods["method"].eq("method_beta")
                & methods["projection_mode"].eq("enabled")
            ].iloc[0]
            expected_sample_sd = math.sqrt(0.03125)
            self.assertAlmostEqual(float(beta_enabled["success_rate_mean"]), 0.375)
            self.assertAlmostEqual(
                float(beta_enabled["success_rate_std"]), expected_sample_sd
            )
            self.assertAlmostEqual(float(beta_enabled["episode_return_mean"]), 2.0)
            self.assertAlmostEqual(
                float(beta_enabled["episode_return_std"]), math.sqrt(0.5)
            )

            alpha_seed7 = paired[
                paired["method"].eq("method_alpha")
                & paired["train_seed"].eq(7)
            ].iloc[0]
            self.assertAlmostEqual(
                float(alpha_seed7["episode_return_delta_enabled_minus_disabled"]),
                1.0,
            )
            self.assertAlmostEqual(
                float(alpha_seed7["success_delta_enabled_minus_disabled"]),
                -0.25,
            )
            self.assertAlmostEqual(
                float(alpha_seed7["collision_delta_enabled_minus_disabled"]),
                0.25,
            )
            self.assertAlmostEqual(
                float(alpha_seed7["min_obstacle_clearance_delta_enabled_minus_disabled"]),
                0.0,
            )
            alpha_paired = paired_methods[
                paired_methods["method"].eq("method_alpha")
            ].iloc[0]
            self.assertAlmostEqual(
                float(alpha_paired["success_delta_enabled_minus_disabled_mean"]),
                -0.125,
            )
            self.assertAlmostEqual(
                float(alpha_paired["success_delta_enabled_minus_disabled_std"]),
                expected_sample_sd,
            )

            output = root / "absolute.json"
            process = self.run_cli(
                SUMMARY_SCRIPT,
                [
                    "--episodes",
                    str(evidence["episodes"]),
                    "--protocol",
                    str(evidence["protocol"]),
                    "--layout-suite",
                    str(evidence["layout_suite"]),
                    "--checkpoint-summary",
                    str(evidence["checkpoint_summary"]),
                    "--method-summary",
                    str(evidence["method_summary"]),
                    "--worked-example-method",
                    "method_beta",
                    "--worked-example-projection-mode",
                    "enabled",
                    "--output",
                    str(output),
                ],
                root,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            summary_row = next(
                row
                for row in result["absolute_method_summary"]
                if row["method"] == "method_beta"
                and row["projection_mode"] == "enabled"
            )
            self.assertAlmostEqual(summary_row["success_rate_mean"], 0.375)
            self.assertAlmostEqual(
                summary_row["success_rate_sample_sd"], expected_sample_sd
            )
            checkpoint_row = next(
                row
                for row in result["training_run_absolute_metrics"]
                if row["method"] == "method_beta"
                and row["projection_mode"] == "enabled"
                and row["train_seed"] == 7
            )
            for metric in (
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "episode_return",
                "episode_length",
                "min_obstacle_clearance",
            ):
                self.assertIn(metric, checkpoint_row)
            self.assertAlmostEqual(checkpoint_row["timeout_rate"], 0.5)
            omitted_seed7 = next(
                row
                for row in result["leave_one_seed_out"]
                if row["method"] == "method_beta"
                and row["projection_mode"] == "enabled"
                and row["omitted_train_seed"] == 7
            )
            self.assertAlmostEqual(omitted_seed7["success_rate_mean"], 0.5)
            worked = result["worked_sample_sd"]["calculations"]["success_rate"]
            self.assertEqual(worked["variance_denominator_n_minus_1"], 1)
            self.assertAlmostEqual(worked["squared_deviation_sum"], 0.03125)
            self.assertAlmostEqual(worked["sample_sd"], expected_sample_sd)

            paired_output = root / "paired.json"
            paired_process = self.run_cli(
                PAIRED_EFFECT_SCRIPT,
                [
                    "--episodes",
                    str(evidence["episodes"]),
                    "--protocol",
                    str(evidence["protocol"]),
                    "--layout-suite",
                    str(evidence["layout_suite"]),
                    "--paired-deltas",
                    str(evidence["paired_deltas"]),
                    "--paired-summary",
                    str(evidence["paired_summary"]),
                    "--worked-example-method",
                    "method_alpha",
                    "--worked-example-train-seed",
                    "7",
                    "--output",
                    str(paired_output),
                ],
                root,
            )
            self.assertEqual(paired_process.returncode, 0, paired_process.stderr)
            paired_result = json.loads(paired_output.read_text(encoding="utf-8"))
            alpha_summary = next(
                row
                for row in paired_result["paired_method_summary"]
                if row["method"] == "method_alpha"
            )
            self.assertAlmostEqual(
                alpha_summary[
                    "success_rate_delta_enabled_minus_disabled_mean"
                ],
                -0.125,
            )
            self.assertEqual(
                alpha_summary[
                    "success_rate_delta_enabled_minus_disabled_negative_count"
                ],
                1,
            )
            self.assertEqual(
                alpha_summary[
                    "success_rate_delta_enabled_minus_disabled_zero_count"
                ],
                1,
            )
            self.assertEqual(
                alpha_summary[
                    "success_rate_delta_enabled_minus_disabled_positive_count"
                ],
                0,
            )
            worked_pair = paired_result["worked_outcome_pair"]
            self.assertAlmostEqual(worked_pair["outcome_delta_sum"], 0.0)
            self.assertEqual(
                paired_result["coverage"]["paired_evaluation_observations_per_training_run"],
                4,
            )

    def test_layout_transfer_summary_matches_hand_calculations_and_robustness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            output = root / "layout-transfer.json"
            process = self.run_cli(
                LAYOUT_TRANSFER_SCRIPT,
                [
                    *validation_arguments(evidence, output)[:-2],
                    "--worked-example-method",
                    "method_alpha",
                    "--worked-example-train-seed",
                    "7",
                    "--worked-example-layout",
                    "corridor_a",
                    "--output",
                    str(output),
                ],
                root,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))

            alpha = next(
                row
                for row in result["paired_method_summary"]
                if row["method"] == "method_alpha"
            )
            self.assertAlmostEqual(
                alpha["success_rate_delta_enabled_minus_disabled_mean"], -0.125
            )
            self.assertAlmostEqual(
                alpha["collision_rate_delta_enabled_minus_disabled_mean"], 0.0
            )
            self.assertAlmostEqual(
                alpha["timeout_rate_delta_enabled_minus_disabled_mean"], 0.125
            )

            corridor = next(
                row
                for row in result["layout_paired_effects"]
                if row["method"] == "method_alpha"
                and row["layout_id"] == "corridor_a"
            )
            self.assertAlmostEqual(
                corridor[
                    "success_rate_delta_enabled_minus_disabled_mean_across_training_runs"
                ],
                0.0,
            )
            self.assertAlmostEqual(
                corridor[
                    "collision_rate_delta_enabled_minus_disabled_mean_across_training_runs"
                ],
                -0.25,
            )
            self.assertAlmostEqual(
                corridor[
                    "timeout_rate_delta_enabled_minus_disabled_mean_across_training_runs"
                ],
                0.25,
            )
            self.assertEqual(
                corridor[
                    "success_rate_delta_enabled_minus_disabled_negative_training_run_count"
                ],
                1,
            )
            self.assertEqual(
                corridor[
                    "success_rate_delta_enabled_minus_disabled_positive_training_run_count"
                ],
                1,
            )

            omit_corridor = next(
                row
                for row in result["leave_one_layout_out"]
                if row["method"] == "method_alpha"
                and row["omitted_layout_id"] == "corridor_a"
            )
            self.assertAlmostEqual(
                omit_corridor[
                    "success_rate_delta_enabled_minus_disabled_mean"
                ],
                -0.25,
            )
            self.assertAlmostEqual(
                omit_corridor[
                    "collision_rate_delta_enabled_minus_disabled_mean"
                ],
                0.25,
            )
            concentration = next(
                row
                for row in result["layout_effect_concentration"]
                if row["method"] == "method_alpha"
                and row["metric"]
                == "collision_rate_delta_enabled_minus_disabled"
            )
            self.assertEqual(
                concentration[
                    "minimum_layouts_for_80_percent_of_absolute_magnitude"
                ],
                2,
            )
            self.assertFalse(concentration["zero_mass"])
            self.assertAlmostEqual(
                concentration["largest_layout_share_of_absolute_magnitude"],
                0.5,
            )
            self.assertAlmostEqual(
                concentration["effective_layout_count_from_absolute_shares"],
                2.0,
            )
            self.assertAlmostEqual(
                concentration[
                    "cancellation_ratio_abs_signed_sum_over_absolute_sum"
                ],
                0.0,
            )
            accounting = next(
                row
                for row in result["outcome_redistribution_accounting"]
                if row["method"] == "method_alpha"
            )
            self.assertEqual(accounting["net_outcome_count_sum"], 0)
            self.assertEqual(
                result["worked_example"]["selected_training_run_layout_cell"][
                    "projection_modes"
                ][0]["evaluation_observation_count"],
                2,
            )
            self.assertTrue(result["aggregation_identity_checks"]["all_checks_pass"])

    def test_layout_concentration_marks_zero_effect_mass_as_undefined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            design = load_design(
                str(evidence["protocol"]),
                str(evidence["layout_suite"]),
            )
            rows = []
            for method in design.paired_methods:
                for layout_index, layout_id in enumerate(design.layout_ids):
                    rows.append(
                        {
                            "method": method.name,
                            "layout_id": layout_id,
                            "layout_index": layout_index,
                            "success_rate_delta_enabled_minus_disabled_mean_across_training_runs": 0.0,
                            "collision_rate_delta_enabled_minus_disabled_mean_across_training_runs": 0.0,
                            "timeout_rate_delta_enabled_minus_disabled_mean_across_training_runs": 0.0,
                        }
                    )
            concentration = _effect_concentration(design, pd.DataFrame(rows))
            self.assertTrue(concentration)
            for record in concentration:
                self.assertTrue(record["zero_mass"])
                self.assertIsNone(
                    record["minimum_layouts_for_50_percent_of_absolute_magnitude"]
                )
                self.assertIsNone(
                    record["minimum_layouts_for_80_percent_of_absolute_magnitude"]
                )
                self.assertIsNone(
                    record["largest_layout_share_of_absolute_magnitude"]
                )
                self.assertIsNone(
                    record["effective_layout_count_from_absolute_shares"]
                )
                self.assertIsNone(
                    record[
                        "cancellation_ratio_abs_signed_sum_over_absolute_sum"
                    ]
                )
                self.assertTrue(
                    all(
                        item["share_of_total_absolute_magnitude"] is None
                        for item in record["top_layouts_by_absolute_magnitude"]
                    )
                )

    def test_layout_transfer_requires_complete_worked_example_and_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            common = validation_arguments(evidence, root / "unused.json")[:-2]
            partial_output = root / "partial.json"
            partial = self.run_cli(
                LAYOUT_TRANSFER_SCRIPT,
                [
                    *common,
                    "--worked-example-method",
                    "method_alpha",
                    "--output",
                    str(partial_output),
                ],
                root,
            )
            self.assertNotEqual(partial.returncode, 0)
            self.assertIn("requires --worked-example-method", partial.stderr)
            self.assertFalse(partial_output.exists())

            existing_output = root / "existing.json"
            existing_output.write_text("preserve me\n", encoding="utf-8")
            existing = self.run_cli(
                LAYOUT_TRANSFER_SCRIPT,
                [*common, "--output", str(existing_output)],
                root,
            )
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn("refusing to overwrite", existing.stderr)
            self.assertEqual(existing_output.read_text(), "preserve me\n")

    def test_paired_effect_partial_worked_example_and_existing_output_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            base_arguments = [
                "--episodes",
                str(evidence["episodes"]),
                "--protocol",
                str(evidence["protocol"]),
                "--layout-suite",
                str(evidence["layout_suite"]),
                "--paired-deltas",
                str(evidence["paired_deltas"]),
                "--paired-summary",
                str(evidence["paired_summary"]),
            ]
            partial_output = root / "partial.json"
            partial = self.run_cli(
                PAIRED_EFFECT_SCRIPT,
                [
                    *base_arguments,
                    "--worked-example-method",
                    "method_alpha",
                    "--output",
                    str(partial_output),
                ],
                root,
            )
            self.assertNotEqual(partial.returncode, 0)
            self.assertIn("requires both", partial.stderr)
            self.assertFalse(partial_output.exists())

            existing_output = root / "existing.json"
            existing_output.write_text("preserve me\n", encoding="utf-8")
            existing = self.run_cli(
                PAIRED_EFFECT_SCRIPT,
                [*base_arguments, "--output", str(existing_output)],
                root,
            )
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn("refusing to overwrite", existing.stderr)
            self.assertEqual(existing_output.read_text(), "preserve me\n")

    def test_duplicate_csv_header_fails_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            with evidence["episodes"].open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.reader(stream))
            duplicate_index = rows[0].index("episode_return")
            rows[0].append("episode_return")
            for row in rows[1:]:
                row.append(row[duplicate_index])
            duplicate = root / "duplicate-header.csv"
            with duplicate.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(rows)
            output = root / "must-not-exist.json"
            process = self.run_cli(
                INVENTORY_SCRIPT,
                ["--episodes", str(duplicate), "--output", str(output)],
                root,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("duplicate column names", process.stderr)
            self.assertFalse(output.exists())

    def test_duplicate_json_keys_and_nonstandard_constants_fail_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            original = evidence["protocol"].read_text(encoding="utf-8")
            schema_field = '"schema_version": "projection_analysis_protocol_v1",'
            duplicate_protocol = root / "duplicate-key.json"
            duplicate_protocol.write_text(
                original.replace(
                    schema_field,
                    f"{schema_field}\n  {schema_field}",
                    1,
                ),
                encoding="utf-8",
            )
            duplicate_output = root / "duplicate-output.json"
            duplicate_process = self.run_cli(
                VALIDATION_SCRIPT,
                validation_arguments(
                    evidence,
                    duplicate_output,
                    protocol=duplicate_protocol,
                ),
                root,
            )
            self.assertNotEqual(duplicate_process.returncode, 0)
            self.assertIn("duplicate object key", duplicate_process.stderr)
            self.assertFalse(duplicate_output.exists())

            numeric_field = '"evaluation_collision_penalty": 3.0'
            for token in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(token=token):
                    malformed_protocol = root / f"nonstandard-{token}.json"
                    malformed_protocol.write_text(
                        original.replace(
                            numeric_field,
                            f'"evaluation_collision_penalty": {token}',
                            1,
                        ),
                        encoding="utf-8",
                    )
                    output = root / f"nonstandard-{token}-output.json"
                    process = self.run_cli(
                        VALIDATION_SCRIPT,
                        validation_arguments(
                            evidence,
                            output,
                            protocol=malformed_protocol,
                        ),
                        root,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("non-standard numeric constant", process.stderr)
                    self.assertFalse(output.exists())

    def test_seed_and_episode_mapping_tampering_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            original = pd.read_csv(evidence["episodes"])
            alterations = {
                "seed": ("seed", original["seed"] + 1, "seed must equal"),
                "episode": (
                    "episode",
                    original["episode"] + 100,
                    "Evaluation mapping differs",
                ),
            }
            for name, (column, values, message) in alterations.items():
                with self.subTest(name=name):
                    changed = original.copy()
                    changed[column] = values
                    changed_path = root / f"changed-{name}.csv"
                    changed.to_csv(changed_path, index=False)
                    output = root / f"changed-{name}-output.json"
                    process = self.run_cli(
                        VALIDATION_SCRIPT,
                        validation_arguments(
                            evidence,
                            output,
                            episodes=changed_path,
                        ),
                        root,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn(message, process.stderr)
                    self.assertFalse(output.exists())

    def test_entrypoints_contain_no_frozen_locations_or_study_labels(self) -> None:
        forbidden = (
            "b005123cb2c6c754a991d1e7fdc709437b90e915",
            "results/tables",
            "fixed_training_geometry",
            "core_layout_transfer",
            "ppo_baseline",
            "ppo_high_penalty",
            "ppo_train_projection",
        )
        for source in (
            INVENTORY_SCRIPT,
            VALIDATION_SCRIPT,
            SUMMARY_SCRIPT,
            PAIRED_EFFECT_SCRIPT,
            LAYOUT_TRANSFER_SCRIPT,
            TERMINAL_CORRESPONDENCE_SCRIPT,
            SCRIPT_DIRECTORY / "_evaluation_evidence.py",
        ):
            text = source.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, f"{source.name} contains {value!r}")
            if source == TERMINAL_CORRESPONDENCE_SCRIPT:
                self.assertNotIn("Step 9", text)
                self.assertNotIn("step9", text.lower())

    def test_malformed_boolean_fails_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            frame = pd.read_csv(evidence["episodes"])
            frame["success"] = frame["success"].astype(object)
            frame.loc[0, "success"] = "not-a-boolean"
            malformed = root / "malformed.csv"
            frame.to_csv(malformed, index=False)
            output = root / "must-not-exist.json"
            process = self.run_cli(
                INVENTORY_SCRIPT,
                ["--episodes", str(malformed), "--output", str(output)],
                root,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("not Boolean", process.stderr)
            self.assertFalse(output.exists())

    def test_checkpoint_comparison_mismatch_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            changed = pd.read_csv(evidence["episodes"])
            mask = changed["method"].eq("method_alpha") & changed["train_seed"].eq(7)
            changed.loc[mask, "checkpoint"] = "elsewhere/replaced.weights"
            changed.loc[mask, "checkpoint_sha256"] = _sha("replacement")
            comparison = root / "comparison.csv"
            changed.to_csv(comparison, index=False)
            output = root / "must-not-exist.json"
            process = self.run_cli(
                VALIDATION_SCRIPT,
                [
                    "--episodes",
                    str(evidence["episodes"]),
                    "--protocol",
                    str(evidence["protocol"]),
                    "--layout-suite",
                    str(evidence["layout_suite"]),
                    "--checkpoint-summary",
                    str(evidence["checkpoint_summary"]),
                    "--method-summary",
                    str(evidence["method_summary"]),
                    "--paired-deltas",
                    str(evidence["paired_deltas"]),
                    "--paired-summary",
                    str(evidence["paired_summary"]),
                    "--build-audit",
                    str(evidence["build_audit"]),
                    "--compare-checkpoints-with",
                    str(comparison),
                    "--output",
                    str(output),
                ],
                root,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Checkpoint identities differ", process.stderr)
            self.assertFalse(output.exists())

    def test_structural_nan_mismatch_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            changed = pd.read_csv(evidence["episodes"])
            active_index = changed.index[changed["layout_id"].eq("corridor_a")][0]
            changed.loc[active_index, "min_obstacle_clearance"] = np.nan
            changed_path = root / "bad-clearance.csv"
            changed.to_csv(changed_path, index=False)
            output = root / "must-not-exist.json"
            process = self.run_cli(
                SUMMARY_SCRIPT,
                [
                    "--episodes",
                    str(changed_path),
                    "--protocol",
                    str(evidence["protocol"]),
                    "--layout-suite",
                    str(evidence["layout_suite"]),
                    "--checkpoint-summary",
                    str(evidence["checkpoint_summary"]),
                    "--method-summary",
                    str(evidence["method_summary"]),
                    "--output",
                    str(output),
                ],
                root,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("NaNs do not exactly match", process.stderr)
            self.assertFalse(output.exists())

    def test_output_is_deterministic_and_existing_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = build_synthetic_evidence(root / "inputs")
            first = root / "first.json"
            second = root / "second.json"
            common = ["--episodes", str(evidence["episodes"])]
            one = self.run_cli(INVENTORY_SCRIPT, [*common, "--output", str(first)], root)
            two = self.run_cli(INVENTORY_SCRIPT, [*common, "--output", str(second)], root)
            self.assertEqual(one.returncode, 0, one.stderr)
            self.assertEqual(two.returncode, 0, two.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            original = first.read_bytes()
            refused = self.run_cli(
                INVENTORY_SCRIPT,
                [*common, "--output", str(first)],
                root,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("refusing to overwrite", refused.stderr)
            self.assertEqual(first.read_bytes(), original)

    def test_numeric_reconciliation_requires_exact_nan_positions(self) -> None:
        self.assertEqual(
            compare_numeric_exact_nan(
                pd.Series([1.0, np.nan, 3.0]),
                pd.Series([1.0, np.nan, 3.0]),
                "same",
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "NaN patterns differ"):
            compare_numeric_exact_nan(
                pd.Series([1.0, np.nan, 3.0]),
                pd.Series([1.0, 2.0, np.nan]),
                "different",
            )


if __name__ == "__main__":
    unittest.main()
