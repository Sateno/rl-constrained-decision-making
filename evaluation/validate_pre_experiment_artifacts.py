from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.layout_suite import load_navigation_layout_suite


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONDA_ENVIRONMENT = "RL_PROJECTS"
REQUIRED_GYMNASIUM_VERSION = "1.3.0"
MINIMUM_TEST_COUNT = 42
EXPECTED_TRAJECTORY_VERSION = "evaluation_trajectory_v1"
EXPECTED_DEVELOPMENT_LAYOUT_SHA256 = (
    "4ce25b8ca82fb59b0aff8fc10b202aca972ff8e0d4e53f923a1b0e6f4dda8f6a"
)
EXPECTED_CORE_LAYOUT_SHA256 = (
    "1027141797052240e83b941398e5a32031e9ca67d001e4a0cf1e19b0f96dd466"
)

CHECKPOINT_FIXTURE = Path("runs/checkpoints/ppo_baseline_51200_seed1.pt")
DEVELOPMENT_LAYOUT_SUITE = Path(
    "evaluation/layouts/development_navigation_layouts.json"
)
CORE_LAYOUT_SUITE = Path("evaluation/layouts/core_navigation_layouts.json")
DEVELOPMENT_PROTOCOL = Path(
    "experiments/development_projection_analysis_protocol.json"
)
CANONICAL_SUMMARY = Path(
    "runs/validation/evaluation_time_projection_validation_summary.txt"
)

BASE_TENSORBOARD_TAGS = {
    "charts/episodic_return",
    "charts/episodic_length",
    "losses/policy_loss",
    "losses/value_loss",
    "losses/entropy",
    "losses/approx_kl",
    "safety/success",
    "safety/collision",
    "safety/final_obstacle_clearance",
}
PROJECTION_TENSORBOARD_TAGS = {
    "projection/intervention_frequency",
    "projection/correction_norm",
    "projection/slack_sum",
    "projection/slack_max",
}
EXPECTED_RESULT_PDFS = {
    "evaluation_return.pdf",
    "evaluation_success_rate.pdf",
    "evaluation_collision_rate.pdf",
    "evaluation_min_obstacle_clearance.pdf",
    "evaluation_projection_intervention.pdf",
    "evaluation_projection_correction.pdf",
    "evaluation_projection_slack.pdf",
    "representative_trajectories.pdf",
}
EXPECTED_RESULT_TABLES = {
    "evaluation_episode_results.csv",
    "checkpoint_summary.csv",
    "method_summary.csv",
    "paired_projection_deltas.csv",
    "paired_projection_summary.csv",
    "generated_method_summary.tex",
    "generated_paired_projection_deltas.tex",
}


# Reject one failed validation condition.
def require(condition: bool, message: str) -> None:
#{
    if not condition:
        raise RuntimeError(message)
#} End function require


# Return one file SHA-256.
def file_sha256(path: Path) -> str:
#{
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()
#} End function file_sha256


# Parse a simple key=value summary file.
def parse_key_value_file(path: Path) -> dict[str, str]:
#{
    require(path.is_file(), f"Expected validation record was not created: {path}")
    values = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    return values
#} End function parse_key_value_file


# Convert one CSV Boolean column into a validated Boolean array.
def csv_boolean_values(series: pd.Series, column_name: str) -> np.ndarray:
#{
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.casefold()
    invalid = set(normalized.unique()) - {"true", "false"}
    require(
        not invalid,
        f"CSV column {column_name} contains invalid Boolean values: {sorted(invalid)}",
    )
    return normalized.eq("true").to_numpy(dtype=bool)
#} End function csv_boolean_values


# Write one JSON audit file.
def write_json(path: Path, value: dict[str, Any]) -> None:
#{
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
#} End function write_json


# Validate the local environment and committed scientific inputs.
def validate_preflight() -> None:
#{
    import cvxpy as cp
    import gymnasium

    active_environment = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    prefix_name = Path(conda_prefix).name if conda_prefix else ""
    require(
        active_environment.casefold() == REQUIRED_CONDA_ENVIRONMENT.casefold()
        or prefix_name.casefold() == REQUIRED_CONDA_ENVIRONMENT.casefold(),
        f"Activate the {REQUIRED_CONDA_ENVIRONMENT} Conda environment first.",
    )
    require(
        gymnasium.__version__ == REQUIRED_GYMNASIUM_VERSION,
        (
            f"Expected Gymnasium {REQUIRED_GYMNASIUM_VERSION}, "
            f"received {gymnasium.__version__}."
        ),
    )
    require("OSQP" in cp.installed_solvers(), "CVXPY does not report OSQP as installed.")
    require(CHECKPOINT_FIXTURE.is_file(), f"Regression checkpoint not found: {CHECKPOINT_FIXTURE}")
    require(DEVELOPMENT_PROTOCOL.is_file(), f"Analysis protocol not found: {DEVELOPMENT_PROTOCOL}")

    development = load_navigation_layout_suite(DEVELOPMENT_LAYOUT_SUITE)
    core = load_navigation_layout_suite(CORE_LAYOUT_SUITE)
    require(
        development.sha256 == EXPECTED_DEVELOPMENT_LAYOUT_SHA256,
        "Development layout identity does not match the reviewed source.",
    )
    require(
        core.sha256 == EXPECTED_CORE_LAYOUT_SHA256,
        "Core layout identity does not match the reviewed source.",
    )

    print("Preflight validation passed.")
    print(f"gymnasium={gymnasium.__version__}")
    print(f"installed_solvers={cp.installed_solvers()}")
    print(f"development_layout_sha256={development.sha256}")
    print(f"core_layout_sha256={core.sha256}")
#} End function validate_preflight


# Verify the established runtime validator passed the patched test floor.
def validate_canonical_summary() -> None:
#{
    summary = parse_key_value_file(CANONICAL_SUMMARY)
    require(summary.get("status") == "PASS", "Canonical runtime summary is not PASS.")
    require(
        int(summary.get("pytest_passed", "0")) >= MINIMUM_TEST_COUNT,
        f"Canonical runtime validation recorded fewer than {MINIMUM_TEST_COUNT} tests.",
    )
    require(
        summary.get("trajectory_archive_version") == EXPECTED_TRAJECTORY_VERSION,
        "Canonical trajectory archive version mismatch.",
    )
    print("Canonical runtime summary audit passed.")
    print(f"pytest_passed={summary['pytest_passed']}")
#} End function validate_canonical_summary


# Require finite scalar events for one TensorBoard tag.
def scalar_event_count(accumulator: Any, tag: str) -> int:
#{
    events = accumulator.Scalars(tag)
    require(events, f"TensorBoard tag contains no events: {tag}")
    require(
        np.isfinite([event.value for event in events]).all(),
        f"TensorBoard tag contains non-finite values: {tag}",
    )
    return len(events)
#} End function scalar_event_count


# Audit the three short training checkpoints and their TensorBoard runs.
def validate_training(checkpoint_directory: Path, output: Path) -> None:
#{
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    specs = (
        ("baseline.pt", "ppo_baseline", False, 10.0),
        ("high_penalty.pt", "ppo_high_penalty", False, 50.0),
        ("train_projection.pt", "ppo_train_projection", True, 10.0),
    )
    audits = []

    for filename, method, projection_enabled, collision_penalty in specs:
        checkpoint_path = checkpoint_directory / filename
        require(checkpoint_path.is_file(), f"Smoke checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        require(args["method"] == method, f"Method mismatch in {checkpoint_path}.")
        require(int(args["seed"]) == 9901, f"Seed mismatch in {checkpoint_path}.")
        require(int(args["total_timesteps"]) == 2048, f"Timestep mismatch in {checkpoint_path}.")
        require(bool(args["enable_projection"]) == projection_enabled, f"Projection flag mismatch in {checkpoint_path}.")
        require(float(args["collision_penalty"]) == collision_penalty, f"Collision penalty mismatch in {checkpoint_path}.")

        run_directory = Path("runs") / checkpoint["run_name"]
        require(run_directory.is_dir(), f"TensorBoard run directory not found: {run_directory}")
        accumulator = EventAccumulator(str(run_directory))
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        required_tags = set(BASE_TENSORBOARD_TAGS)

        if projection_enabled:
            required_tags.update(PROJECTION_TENSORBOARD_TAGS)

        missing = sorted(required_tags - scalar_tags)
        require(not missing, f"TensorBoard run {run_directory} is missing tags: {missing}")
        counts = {tag: scalar_event_count(accumulator, tag) for tag in sorted(required_tags)}

        if projection_enabled:
            values = np.asarray(
                [event.value for event in accumulator.Scalars("projection/intervention_frequency")],
                dtype=np.float64,
            )
            require(np.all((0.0 <= values) & (values <= 1.0)), "Projection intervention frequency is outside [0, 1].")

        audits.append(
            {
                "method": method,
                "projection_enabled": projection_enabled,
                "collision_penalty": collision_penalty,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "run_directory": str(run_directory),
                "scalar_event_counts": counts,
            }
        )

    write_json(output, {"status": "PASS", "runs": audits})
    print("Training smoke artifact audit passed.")
    print(f"audit={output}")
#} End function validate_training


# Audit fresh development-layout CSV and NPZ artifacts.
def validate_layouts(evaluation_directory: Path, output: Path) -> None:
#{
    suite = load_navigation_layout_suite(DEVELOPMENT_LAYOUT_SUITE)
    disabled_csv = evaluation_directory / "baseline_projection_disabled.csv"
    enabled_csv = evaluation_directory / "baseline_projection_enabled.csv"
    disabled_npz = evaluation_directory / "baseline_projection_disabled_trajectories.npz"
    enabled_npz = evaluation_directory / "baseline_projection_enabled_trajectories.npz"
    disabled_rows = pd.read_csv(disabled_csv)
    enabled_rows = pd.read_csv(enabled_csv)
    keys = ["layout_id", "layout_repeat", "evaluation_seed"]

    require(len(disabled_rows) == len(enabled_rows) == len(suite.layouts), "Development layout row count mismatch.")
    require(disabled_rows[keys].equals(enabled_rows[keys]), "Projection mode evaluation keys differ.")
    require(disabled_rows["layout_suite_sha256"].eq(suite.sha256).all(), "Disabled rows contain the wrong layout identity.")
    require(enabled_rows["layout_suite_sha256"].eq(suite.sha256).all(), "Enabled rows contain the wrong layout identity.")
    require(not csv_boolean_values(disabled_rows["projection_enabled"], "projection_enabled").any(), "Disabled rows contain an enabled flag.")
    require(csv_boolean_values(enabled_rows["projection_enabled"], "projection_enabled").all(), "Enabled rows contain a disabled flag.")
    require(np.isfinite(disabled_rows["episode_return"]).all(), "Disabled evaluation contains non-finite returns.")
    require(np.isfinite(enabled_rows["episode_return"]).all(), "Enabled evaluation contains non-finite returns.")

    solver_failures = int(enabled_rows["projection_solver_failure_count"].sum())
    interventions = int(enabled_rows["projection_intervention_count"].sum())
    require(solver_failures == 0, "Development projection evaluation reported solver failure.")
    require(interventions > 0, "Development projection evaluation produced no intervention.")

    with np.load(disabled_npz, allow_pickle=False) as disabled, np.load(enabled_npz, allow_pickle=False) as enabled:
        episode_keys = disabled["episode_keys"].tolist()
        require(disabled["trajectory_archive_version"].item() == EXPECTED_TRAJECTORY_VERSION, "Disabled trajectory schema mismatch.")
        require(enabled["trajectory_archive_version"].item() == EXPECTED_TRAJECTORY_VERSION, "Enabled trajectory schema mismatch.")
        require(disabled["run_layout_suite_sha256"].item() == suite.sha256, "Disabled trajectory layout identity mismatch.")
        require(enabled["run_layout_suite_sha256"].item() == suite.sha256, "Enabled trajectory layout identity mismatch.")
        require(episode_keys == enabled["episode_keys"].tolist(), "Trajectory episode keys differ.")

        for key in episode_keys:
            require(disabled[f"{key}_positions"].shape[0] == disabled[f"{key}_action_raw_physical"].shape[0] + 1, f"Disabled trajectory alignment failed for {key}.")
            require(enabled[f"{key}_positions"].shape[0] == enabled[f"{key}_action_exec_physical"].shape[0] + 1, f"Enabled trajectory alignment failed for {key}.")
            np.testing.assert_array_equal(disabled[f"{key}_action_raw_physical"], disabled[f"{key}_action_exec_physical"])
            np.testing.assert_allclose(
                enabled[f"{key}_action_exec_physical"] - enabled[f"{key}_action_raw_physical"],
                enabled[f"{key}_action_correction_physical"],
                rtol=0.0,
                atol=1.0e-12,
            )
            require(bool(enabled[f"{key}_projection_success"].all()), f"Enabled trajectory {key} contains a failed solve.")

    write_json(
        output,
        {
            "status": "PASS",
            "layout_suite_sha256": suite.sha256,
            "layout_count": len(suite.layouts),
            "projection_interventions": interventions,
            "projection_solver_failures": solver_failures,
            "evaluation_directory": str(evaluation_directory),
        },
    )
    print("Common-layout artifact audit passed.")
    print(f"projection_interventions={interventions}")
    print(f"audit={output}")
#} End function validate_layouts


# Audit the saved-result build and write a short manual review checklist.
def validate_results(result_directory: Path, output: Path) -> None:
#{
    summary = parse_key_value_file(result_directory / "result_build_summary.txt")
    require(summary.get("status") == "PASS", "Result-build summary is not PASS.")
    tables = result_directory / "tables"
    figures = result_directory / "figures"
    table_audit = json.loads((tables / "result_build_audit.json").read_text(encoding="utf-8"))
    figure_audit = json.loads((figures / "figure_build_audit.json").read_text(encoding="utf-8"))
    require(table_audit.get("status") == "PASS", "Result-table audit is not PASS.")
    require(figure_audit.get("status") == "PASS", "Figure-build audit is not PASS.")
    require(table_audit.get("layout_suite_sha256") == EXPECTED_DEVELOPMENT_LAYOUT_SHA256, "Result build contains the wrong layout identity.")
    require(int(table_audit.get("selected_csv_count", 0)) == 2, "Result build must select exactly two development CSVs.")
    require(int(table_audit.get("projection_solver_failure_count", -1)) == 0, "Result build reports a projection solver failure.")

    pdf_names = {path.name for path in figures.glob("*.pdf")}
    table_names = {path.name for path in tables.iterdir() if path.is_file()}
    require(not (EXPECTED_RESULT_PDFS - pdf_names), f"Missing required PDFs: {sorted(EXPECTED_RESULT_PDFS - pdf_names)}")
    require(not (EXPECTED_RESULT_TABLES - table_names), f"Missing required tables: {sorted(EXPECTED_RESULT_TABLES - table_names)}")

    manual_review = result_directory.parent / "manual_review.txt"
    manual_review.write_text(
        "Automated validation passed.\n\n"
        "Before committing:\n"
        "1. Open the generated figures directory.\n"
        "2. Confirm the PDFs are not blank, clipped, or mislabeled.\n"
        "3. Compare evaluation_return.pdf, evaluation_collision_rate.pdf, and\n"
        "   evaluation_projection_intervention.pdf with tables/method_summary.csv.\n"
        "4. Confirm representative_trajectories.pdf shows the expected obstacles,\n"
        "   goal, and projection-disabled/projection-enabled paths.\n"
        "5. Sparse development results are acceptable; contradictions are not.\n\n"
        f"Figures: {figures}\n"
        f"Method table: {tables / 'method_summary.csv'}\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        output,
        {
            "status": "PASS",
            "generated_pdf_count": len(pdf_names),
            "required_pdfs": sorted(EXPECTED_RESULT_PDFS),
            "figures_directory": str(figures),
            "tables_directory": str(tables),
            "manual_review": str(manual_review),
        },
    )
    print("Saved-result build audit passed.")
    print(f"audit={output}")
    print(f"manual_review={manual_review}")
#} End function validate_results


# Dispatch one focused artifact audit used by the master batch file.
def main() -> None:
#{
    os.chdir(REPOSITORY_ROOT)
    parser = argparse.ArgumentParser(description="Audit pre-experiment validation artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("canonical")

    training_parser = subparsers.add_parser("training")
    training_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    training_parser.add_argument("--output", type=Path, required=True)

    layout_parser = subparsers.add_parser("layouts")
    layout_parser.add_argument("--evaluation-dir", type=Path, required=True)
    layout_parser.add_argument("--output", type=Path, required=True)

    result_parser = subparsers.add_parser("results")
    result_parser.add_argument("--result-dir", type=Path, required=True)
    result_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "preflight":
        validate_preflight()
    elif args.command == "canonical":
        validate_canonical_summary()
    elif args.command == "training":
        validate_training(args.checkpoint_dir, args.output)
    elif args.command == "layouts":
        validate_layouts(args.evaluation_dir, args.output)
    elif args.command == "results":
        validate_results(args.result_dir, args.output)
#} End function main


if __name__ == "__main__":
    main()
