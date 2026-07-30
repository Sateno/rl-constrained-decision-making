from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.layout_suite import file_sha256, load_navigation_layout_suite


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUT_SUITE = Path("evaluation/layouts/core_navigation_layouts.json")
DEFAULT_CHECKPOINT = Path("runs/checkpoints/ppo_baseline_51200_seed1.pt")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/calibration/core_navigation_layouts_v1")
DEFAULT_LAYOUT_SHA256 = "1027141797052240e83b941398e5a32031e9ca67d001e4a0cf1e19b0f96dd466"
DEFAULT_CHECKPOINT_SHA256 = "3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3"
REPRESENTATIVE_LAYOUTS = (
    "control_open_route",
    "single_blocked_central_upper",
    "double_blocked_staggered_upper_first",
    "triple_mild_slalom_upper_first",
)


#################################################################################
# region Helpers

# Resolve one path relative to the repository root.
def repository_path(path: Path) -> Path:
#{
    return path if path.is_absolute() else REPOSITORY_ROOT / path
#} End function repository_path


# Verify that replacement is limited to the calibration output tree.
def validate_cleanup_target(path: Path) -> None:
#{
    allowed_root = (REPOSITORY_ROOT / "runs" / "calibration").resolve()
    resolved = path.resolve()

    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise ValueError(f"Refusing to remove unsafe calibration path: {resolved}")
#} End function validate_cleanup_target


# Convert one CSV Boolean series into validated Boolean values.
def boolean_values(series: pd.Series, column_name: str) -> np.ndarray:
#{
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.casefold()
    invalid = set(normalized.unique()) - {"true", "false"}

    if invalid:
        raise ValueError(
            f"CSV column {column_name} contains invalid Boolean values: {sorted(invalid)}"
        )

    return normalized.eq("true").to_numpy(dtype=bool)
#} End function boolean_values


# Run one visible child command.
def run_command(label: str, command: list[str]) -> None:
#{
    print()
    print(label)
    print("-" * len(label))
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
#} End function run_command


# Return one common-layout evaluation command.
def evaluation_command(
    checkpoint: Path,
    layout_suite: Path,
    method: str,
    train_seed: int,
    projection_mode: str,
    output_csv: Path,
    output_npz: Path,
    seed: int,
    max_episode_steps: int,
    collision_penalty: float,
    lookahead_distance: float,
    alpha: float,
    slack_penalty: float,
    extra_clearance: float,
) -> list[str]:
#{
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_layout_suite",
        "--checkpoint",
        str(checkpoint),
        "--layout-suite",
        str(layout_suite),
        "--method",
        method,
        "--train-seed",
        str(train_seed),
        "--projection-mode",
        projection_mode,
        "--repeats-per-layout",
        "1",
        "--seed",
        str(seed),
        "--max-episode-steps",
        str(max_episode_steps),
        "--collision-penalty",
        str(collision_penalty),
        "--no-cuda",
        "--output",
        str(output_csv),
        "--trajectory-output",
        str(output_npz),
    ]

    if projection_mode == "enabled":
        command.extend(
            [
                "--projection-lookahead-distance",
                str(lookahead_distance),
                "--projection-alpha",
                str(alpha),
                "--projection-slack-penalty",
                str(slack_penalty),
                "--projection-extra-clearance",
                str(extra_clearance),
            ]
        )

    return command
#} End function evaluation_command

# end region Helpers


#################################################################################
# region Artifact validation and summaries

# Audit the paired CSV and trajectory artifacts.
def audit_artifacts(
    suite,
    disabled_csv: Path,
    enabled_csv: Path,
    disabled_npz: Path,
    enabled_npz: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
#{
    disabled = pd.read_csv(disabled_csv)
    enabled = pd.read_csv(enabled_csv)
    keys = ["layout_id", "layout_repeat", "evaluation_seed"]

    if len(disabled) != len(enabled) or len(disabled) != len(suite.layouts):
        raise RuntimeError("Calibration row counts do not match the layout suite.")
    if not disabled[keys].equals(enabled[keys]):
        raise RuntimeError("Projection-disabled and projection-enabled keys differ.")
    if not disabled["layout_suite_sha256"].eq(suite.sha256).all():
        raise RuntimeError("Projection-disabled rows contain the wrong layout identity.")
    if not enabled["layout_suite_sha256"].eq(suite.sha256).all():
        raise RuntimeError("Projection-enabled rows contain the wrong layout identity.")
    if boolean_values(disabled["projection_enabled"], "projection_enabled").any():
        raise RuntimeError("Projection-disabled rows contain an enabled projection flag.")
    if not boolean_values(enabled["projection_enabled"], "projection_enabled").all():
        raise RuntimeError("Projection-enabled rows contain a disabled projection flag.")

    for frame, label in ((disabled, "disabled"), (enabled, "enabled")):
        if not np.isfinite(frame["episode_return"]).all():
            raise RuntimeError(f"{label} evaluation contains non-finite returns.")
        if not np.isfinite(frame["episode_length"]).all() or not (frame["episode_length"] > 0).all():
            raise RuntimeError(f"{label} evaluation contains invalid episode lengths.")

    diagnostic_columns = [
        "projection_intervention_count",
        "projection_intervention_rate",
        "mean_projection_correction_norm",
        "max_projection_correction_norm",
        "mean_projection_slack_sum",
        "max_projection_slack",
        "projection_solver_failure_count",
    ]

    if not np.isfinite(enabled[diagnostic_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Projection-enabled calibration contains non-finite diagnostics.")
    if not enabled["projection_intervention_rate"].between(0.0, 1.0).all():
        raise RuntimeError("Projection intervention rates are outside [0, 1].")

    solver_failures = int(enabled["projection_solver_failure_count"].sum())
    interventions = int(enabled["projection_intervention_count"].sum())
    layouts_with_intervention = int((enabled["projection_intervention_count"] > 0).sum())
    layouts_without_intervention = int((enabled["projection_intervention_count"] == 0).sum())

    if solver_failures != 0:
        raise RuntimeError("Core-layout calibration reported a projection solver failure.")
    if interventions <= 0:
        raise RuntimeError("Core-layout calibration produced no projection intervention.")
    if layouts_without_intervention <= 0:
        raise RuntimeError("Projection intervened in every calibration layout.")

    with np.load(disabled_npz, allow_pickle=False) as disabled_trajectories, np.load(
        enabled_npz,
        allow_pickle=False,
    ) as enabled_trajectories:
        episode_keys = disabled_trajectories["episode_keys"].tolist()

        if episode_keys != enabled_trajectories["episode_keys"].tolist():
            raise RuntimeError("Calibration trajectory episode keys differ.")
        if int(disabled_trajectories["episode_count"]) != len(suite.layouts):
            raise RuntimeError("Disabled calibration trajectory count mismatch.")
        if int(enabled_trajectories["episode_count"]) != len(suite.layouts):
            raise RuntimeError("Enabled calibration trajectory count mismatch.")
        if disabled_trajectories["run_layout_suite_sha256"].item() != suite.sha256:
            raise RuntimeError("Disabled trajectory archive contains the wrong layout identity.")
        if enabled_trajectories["run_layout_suite_sha256"].item() != suite.sha256:
            raise RuntimeError("Enabled trajectory archive contains the wrong layout identity.")

        for key in episode_keys:
            if disabled_trajectories[f"{key}_positions"].shape[0] != disabled_trajectories[f"{key}_action_raw_physical"].shape[0] + 1:
                raise RuntimeError(f"Disabled trajectory alignment failed for {key}.")
            if enabled_trajectories[f"{key}_positions"].shape[0] != enabled_trajectories[f"{key}_action_exec_physical"].shape[0] + 1:
                raise RuntimeError(f"Enabled trajectory alignment failed for {key}.")
            np.testing.assert_array_equal(
                disabled_trajectories[f"{key}_action_raw_physical"],
                disabled_trajectories[f"{key}_action_exec_physical"],
            )
            np.testing.assert_allclose(
                enabled_trajectories[f"{key}_action_exec_physical"]
                - enabled_trajectories[f"{key}_action_raw_physical"],
                enabled_trajectories[f"{key}_action_correction_physical"],
                rtol=0.0,
                atol=1.0e-12,
            )
            if not bool(enabled_trajectories[f"{key}_projection_success"].all()):
                raise RuntimeError(f"Enabled trajectory {key} contains a failed projection solve.")

    audit = {
        "layout_count": len(suite.layouts),
        "projection_interventions": interventions,
        "layouts_with_intervention": layouts_with_intervention,
        "layouts_without_intervention": layouts_without_intervention,
        "projection_solver_failures": solver_failures,
    }
    return disabled, enabled, audit
#} End function audit_artifacts


# Build the family-level and layout-level calibration tables.
def build_tables(
    disabled: pd.DataFrame,
    enabled: pd.DataFrame,
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
#{
    disabled_rows = disabled.copy()
    enabled_rows = enabled.copy()
    disabled_rows["mode"] = "projection_off"
    enabled_rows["mode"] = "projection_on"
    combined = pd.concat([disabled_rows, enabled_rows], ignore_index=True)
    combined["family"] = combined["layout_id"].str.split("_").str[0]
    combined["success_bool"] = boolean_values(combined["success"], "success")
    combined["collision_bool"] = boolean_values(combined["collision"], "collision")
    combined["timeout_bool"] = ~combined["success_bool"] & ~combined["collision_bool"]
    family = (
        combined.groupby(["mode", "family"], sort=False)
        .agg(
            layouts=("layout_id", "size"),
            mean_return=("episode_return", "mean"),
            success_rate=("success_bool", "mean"),
            collision_rate=("collision_bool", "mean"),
            timeout_rate=("timeout_bool", "mean"),
            mean_clearance=("min_obstacle_clearance", "mean"),
            interventions=("projection_intervention_count", "sum"),
            mean_intervention_rate=("projection_intervention_rate", "mean"),
            max_correction=("max_projection_correction_norm", "max"),
            max_slack=("max_projection_slack", "max"),
            solver_failures=("projection_solver_failure_count", "sum"),
        )
        .reset_index()
    )
    keys = ["layout_id", "layout_repeat", "evaluation_seed"]
    disabled_columns = [
        "episode_return",
        "success",
        "collision",
        "episode_length",
        "min_obstacle_clearance",
    ]
    enabled_columns = disabled_columns + [
        "projection_intervention_count",
        "projection_intervention_rate",
        "mean_projection_correction_norm",
        "max_projection_correction_norm",
        "mean_projection_slack_sum",
        "max_projection_slack",
        "projection_solver_failure_count",
    ]
    comparison = disabled[keys + disabled_columns].merge(
        enabled[keys + enabled_columns],
        on=keys,
        suffixes=("_off", "_on"),
        validate="one_to_one",
    )
    comparison["return_delta_on_minus_off"] = (
        comparison["episode_return_on"] - comparison["episode_return_off"]
    )
    comparison["clearance_delta_on_minus_off"] = (
        comparison["min_obstacle_clearance_on"]
        - comparison["min_obstacle_clearance_off"]
    )
    family.to_csv(output_directory / "calibration_by_family.csv", index=False)
    comparison.to_csv(output_directory / "calibration_layout_comparison.csv", index=False)
    return family, comparison
#} End function build_tables


# Return overall success, collision, and timeout counts for one result frame.
def outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
#{
    success = boolean_values(frame["success"], "success")
    collision = boolean_values(frame["collision"], "collision")
    return {
        "successes": int(success.sum()),
        "collisions": int(collision.sum()),
        "timeouts": int((~success & ~collision).sum()),
    }
#} End function outcome_counts

# end region Artifact validation and summaries


#################################################################################
# region Representative trajectories

# Return one episode key from a CSV episode index.
def episode_key(row: pd.Series) -> str:
#{
    return f"episode_{int(row['episode']):04d}"
#} End function episode_key


# Generate the predeclared calibration trajectory figure.
def plot_representative_trajectories(
    suite,
    disabled: pd.DataFrame,
    enabled: pd.DataFrame,
    disabled_npz: Path,
    enabled_npz: Path,
    output_path: Path,
) -> None:
#{
    layout_map = {layout.layout_id: layout for layout in suite.layouts}
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))

    with np.load(disabled_npz, allow_pickle=False) as disabled_trajectories, np.load(
        enabled_npz,
        allow_pickle=False,
    ) as enabled_trajectories:
        for axis, layout_id in zip(axes.flat, REPRESENTATIVE_LAYOUTS):
            layout = layout_map[layout_id]
            disabled_row = disabled.loc[disabled["layout_id"] == layout_id].iloc[0]
            enabled_row = enabled.loc[enabled["layout_id"] == layout_id].iloc[0]
            disabled_key = episode_key(disabled_row)
            enabled_key = episode_key(enabled_row)
            disabled_positions = disabled_trajectories[f"{disabled_key}_positions"]
            enabled_positions = enabled_trajectories[f"{enabled_key}_positions"]
            axis.plot(
                disabled_positions[:, 0],
                disabled_positions[:, 1],
                label="projection off",
            )
            axis.plot(
                enabled_positions[:, 0],
                enabled_positions[:, 1],
                label="projection on",
            )
            axis.scatter(layout.start[0], layout.start[1], marker="o", s=35, label="start")
            axis.scatter(layout.goal[0], layout.goal[1], marker="*", s=95, label="goal")

            for center, radius, active in zip(
                layout.obstacle_centers,
                layout.obstacle_radii,
                layout.obstacle_mask,
            ):
                if not active:
                    continue
                axis.add_patch(
                    plt.Circle(
                        center,
                        radius,
                        fill=False,
                        linewidth=1.5,
                    )
                )
                axis.add_patch(
                    plt.Circle(
                        center,
                        radius + suite.agent_radius,
                        fill=False,
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.6,
                    )
                )

            axis.set_title(layout_id.replace("_", " "))
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, hspace=0.32, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
#} End function plot_representative_trajectories

# end region Representative trajectories


#################################################################################
# region Records

# Write the automated audit and concise manual-review records.
def write_records(
    output_directory: Path,
    suite,
    checkpoint: Path,
    checkpoint_sha256: str,
    disabled: pd.DataFrame,
    enabled: pd.DataFrame,
    audit: dict[str, object],
    family: pd.DataFrame,
    projection_parameters: dict[str, float],
) -> None:
#{
    disabled_counts = outcome_counts(disabled)
    enabled_counts = outcome_counts(enabled)
    control_disabled = disabled[disabled["layout_id"].str.startswith("control_")]
    control_successes = int(boolean_values(control_disabled["success"], "success").sum())
    open_route = control_disabled.loc[
        control_disabled["layout_id"] == "control_open_route"
    ].iloc[0]
    open_route_success = bool(boolean_values(pd.Series([open_route["success"]]), "success")[0])
    maximum_correction = float(enabled["max_projection_correction_norm"].max())
    maximum_slack = float(enabled["max_projection_slack"].max())
    audit_record = {
        "status": "PASS",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "layout_suite_path": str(suite.source_path.resolve()),
        "layout_suite_id": suite.suite_id,
        "layout_suite_sha256": suite.sha256,
        "layout_count": len(suite.layouts),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "projection_parameters": projection_parameters,
        "projection_disabled": disabled_counts,
        "projection_enabled": enabled_counts,
        "control_successes_without_projection": control_successes,
        "open_route_success_without_projection": open_route_success,
        "maximum_projection_correction_norm": maximum_correction,
        "maximum_projection_slack": maximum_slack,
        **audit,
        "manual_review_required": True,
    }
    (output_directory / "calibration_audit.json").write_text(
        json.dumps(audit_record, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "# Core Navigation Layout Calibration",
        "",
        "## Automated technical result",
        "",
        "```text",
        "status=PASS",
        f"suite_id={suite.suite_id}",
        f"suite_sha256={suite.sha256}",
        f"layout_count={len(suite.layouts)}",
        f"checkpoint_sha256={checkpoint_sha256}",
        f"projection_interventions={audit['projection_interventions']}",
        f"layouts_with_intervention={audit['layouts_with_intervention']}",
        f"layouts_without_intervention={audit['layouts_without_intervention']}",
        f"projection_solver_failures={audit['projection_solver_failures']}",
        "```",
        "",
        "## Overall outcomes",
        "",
        "| Mode | Successes | Collisions | Timeouts |",
        "|---|---:|---:|---:|",
        f"| Projection off | {disabled_counts['successes']} | {disabled_counts['collisions']} | {disabled_counts['timeouts']} |",
        f"| Projection on | {enabled_counts['successes']} | {enabled_counts['collisions']} | {enabled_counts['timeouts']} |",
        "",
        "## Control checks",
        "",
        f"- Open route success without projection: `{open_route_success}`",
        f"- Control successes without projection: `{control_successes}/4`",
        "",
        "## Projection burden",
        "",
        f"- Total interventions: `{audit['projection_interventions']}`",
        f"- Maximum correction norm: `{maximum_correction:.6f}`",
        f"- Maximum slack: `{maximum_slack:.6f}`",
        f"- Solver failures: `{audit['projection_solver_failures']}`",
        "",
        "## Manual scientific judgment",
        "",
        "Review the following before freezing the suite:",
        "",
        "1. `calibration_by_family.csv`: confirm the families show meaningfully different geometric demands.",
        "2. `calibration_layout_comparison.csv`: inspect the four controls, every collision, the largest intervention, correction, and slack rows.",
        "3. `figures/calibration_representative_trajectories.pdf`: confirm paths, obstacles, goal, and projection behavior are geometrically plausible.",
        "4. Do not require projection to improve return, success, or collisions during calibration.",
        "",
        "Record the final accept/revise decision separately under `docs/core_layout_calibration_record.md`.",
    ]
    (output_directory / "calibration_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary_text = [
        "status=PASS",
        "manual_review_required=true",
        f"suite_id={suite.suite_id}",
        f"suite_sha256={suite.sha256}",
        f"layout_count={len(suite.layouts)}",
        f"checkpoint_sha256={checkpoint_sha256}",
        f"projection_interventions={audit['projection_interventions']}",
        f"layouts_with_intervention={audit['layouts_with_intervention']}",
        f"layouts_without_intervention={audit['layouts_without_intervention']}",
        f"projection_solver_failures={audit['projection_solver_failures']}",
        f"family_rows={len(family)}",
        f"figures={output_directory / 'figures'}",
    ]
    (output_directory / "calibration_summary.txt").write_text(
        "\n".join(summary_text) + "\n",
        encoding="utf-8",
        newline="\n",
    )
#} End function write_records

# end region Records


#################################################################################
# region Workflow

# Run the complete automated core-layout calibration.
def run_calibration(args: argparse.Namespace) -> Path:
#{
    layout_suite_path = repository_path(args.layout_suite).resolve()
    checkpoint_path = repository_path(args.checkpoint).resolve()
    output_directory = repository_path(args.output_dir).resolve()
    validate_cleanup_target(output_directory)

    if output_directory.exists():
        if not args.replace:
            raise FileExistsError(
                f"Calibration directory already exists: {output_directory}. "
                "Choose a new directory or pass --replace explicitly."
            )
        shutil.rmtree(output_directory)

    suite = load_navigation_layout_suite(layout_suite_path)

    if suite.sha256 != args.expected_layout_sha256:
        raise RuntimeError(
            f"Layout-suite SHA-256 mismatch. Expected {args.expected_layout_sha256}, "
            f"received {suite.sha256}."
        )
    if len(suite.layouts) != 24:
        raise RuntimeError(f"Core calibration expects 24 layouts, received {len(suite.layouts)}.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Calibration checkpoint not found: {checkpoint_path}")

    checkpoint_sha256 = file_sha256(checkpoint_path)

    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch. Expected {args.expected_checkpoint_sha256}, "
            f"received {checkpoint_sha256}."
        )

    output_directory.mkdir(parents=True)
    disabled_csv = output_directory / "baseline_projection_disabled.csv"
    enabled_csv = output_directory / "baseline_projection_enabled.csv"
    disabled_npz = output_directory / "baseline_projection_disabled_trajectories.npz"
    enabled_npz = output_directory / "baseline_projection_enabled_trajectories.npz"
    run_command(
        "Core layouts without projection",
        evaluation_command(
            checkpoint=checkpoint_path,
            layout_suite=layout_suite_path,
            method=args.method,
            train_seed=args.train_seed,
            projection_mode="disabled",
            output_csv=disabled_csv,
            output_npz=disabled_npz,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            collision_penalty=args.collision_penalty,
            lookahead_distance=args.projection_lookahead_distance,
            alpha=args.projection_alpha,
            slack_penalty=args.projection_slack_penalty,
            extra_clearance=args.projection_extra_clearance,
        ),
    )
    run_command(
        "Core layouts with projection",
        evaluation_command(
            checkpoint=checkpoint_path,
            layout_suite=layout_suite_path,
            method=args.method,
            train_seed=args.train_seed,
            projection_mode="enabled",
            output_csv=enabled_csv,
            output_npz=enabled_npz,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            collision_penalty=args.collision_penalty,
            lookahead_distance=args.projection_lookahead_distance,
            alpha=args.projection_alpha,
            slack_penalty=args.projection_slack_penalty,
            extra_clearance=args.projection_extra_clearance,
        ),
    )
    disabled, enabled, audit = audit_artifacts(
        suite=suite,
        disabled_csv=disabled_csv,
        enabled_csv=enabled_csv,
        disabled_npz=disabled_npz,
        enabled_npz=enabled_npz,
    )
    family, _ = build_tables(disabled, enabled, output_directory)
    plot_representative_trajectories(
        suite=suite,
        disabled=disabled,
        enabled=enabled,
        disabled_npz=disabled_npz,
        enabled_npz=enabled_npz,
        output_path=output_directory
        / "figures"
        / "calibration_representative_trajectories.pdf",
    )
    write_records(
        output_directory=output_directory,
        suite=suite,
        checkpoint=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        disabled=disabled,
        enabled=enabled,
        audit=audit,
        family=family,
        projection_parameters={
            "lookahead_distance": args.projection_lookahead_distance,
            "alpha": args.projection_alpha,
            "slack_penalty": args.projection_slack_penalty,
            "extra_clearance": args.projection_extra_clearance,
        },
    )
    print()
    print("CORE-LAYOUT CALIBRATION PASSED AUTOMATED CHECKS")
    print("================================================")
    print(f"Summary: {output_directory / 'calibration_summary.md'}")
    print(f"Figure:  {output_directory / 'figures' / 'calibration_representative_trajectories.pdf'}")
    print("Manual scientific judgment is still required before freezing the suite.")
    return output_directory
#} End function run_calibration

# end region Workflow


#################################################################################
# region Command line

# Parse the core-layout calibration configuration.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Calibrate the frozen core navigation layout candidate."
    )
    parser.add_argument("--layout-suite", type=Path, default=DEFAULT_LAYOUT_SUITE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--expected-layout-sha256", default=DEFAULT_LAYOUT_SHA256)
    parser.add_argument("--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--method", default="ppo_baseline")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--collision-penalty", type=float, default=10.0)
    parser.add_argument("--projection-lookahead-distance", type=float, default=0.25)
    parser.add_argument("--projection-alpha", type=float, default=2.0)
    parser.add_argument("--projection-slack-penalty", type=float, default=1000.0)
    parser.add_argument("--projection-extra-clearance", type=float, default=0.0)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    if not args.method.strip():
        parser.error("--method must be nonempty.")
    if args.train_seed < 0 or args.seed < 0:
        parser.error("Seeds must be nonnegative.")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive.")
    if not np.isfinite(args.collision_penalty) or args.collision_penalty < 0.0:
        parser.error("--collision-penalty must be finite and nonnegative.")
    if not np.isfinite(args.projection_lookahead_distance) or args.projection_lookahead_distance < 0.0:
        parser.error("--projection-lookahead-distance must be finite and nonnegative.")
    if not np.isfinite(args.projection_alpha) or args.projection_alpha <= 0.0:
        parser.error("--projection-alpha must be finite and positive.")
    if not np.isfinite(args.projection_slack_penalty) or args.projection_slack_penalty <= 0.0:
        parser.error("--projection-slack-penalty must be finite and positive.")
    if not np.isfinite(args.projection_extra_clearance) or args.projection_extra_clearance < 0.0:
        parser.error("--projection-extra-clearance must be finite and nonnegative.")

    return args
#} End function parse_args


# Run the automated calibration workflow.
def main() -> int:
#{
    args = parse_args()
    run_calibration(args)
    return 0
#} End function main

# end region Command line


if __name__ == "__main__":
    raise SystemExit(main())
