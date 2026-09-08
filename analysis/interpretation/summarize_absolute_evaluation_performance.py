"""Reconstruct training-run-level absolute performance from supplied evidence.

Each training run is an empirical replicate represented by one retained final
checkpoint. Checkpoint path and hash fields identify that retained artifact.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from ._evaluation_evidence import (
        EvidenceError,
        OUTPUT_SCHEMA,
        frame_records,
        implementation_record,
        input_records,
        load_design,
        load_episode_table,
        reconcile_checkpoint_and_method_tables,
        require,
        validate_episode_table,
        write_json_exclusive,
    )
else:  # Support invocation by absolute script path from any working directory.
    sys.path.insert(0, str(Path(__file__).parent))
    from _evaluation_evidence import (  # type: ignore[no-redef]
        EvidenceError,
        OUTPUT_SCHEMA,
        frame_records,
        implementation_record,
        input_records,
        load_design,
        load_episode_table,
        reconcile_checkpoint_and_method_tables,
        require,
        validate_episode_table,
        write_json_exclusive,
    )


ABSOLUTE_METRICS = (
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "episode_return",
    "episode_length",
    "min_obstacle_clearance",
)


def _with_timeout(checkpoints: pd.DataFrame) -> pd.DataFrame:
    result = checkpoints.copy()
    result["timeout_rate"] = 1.0 - result["success_rate"] - result["collision_rate"]
    require(
        bool(
            (
                (result["timeout_rate"] >= -1.0e-12)
                & (result["timeout_rate"] <= 1.0 + 1.0e-12)
            ).all()
        ),
        "Derived checkpoint timeout rate is outside [0, 1].",
    )
    return result


def _absolute_summary(design: object, checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in design.methods:
        for mode in method.modes:
            group = checkpoints[
                checkpoints["method"].eq(method.name)
                & checkpoints["projection_mode"].eq(mode)
            ].sort_values("train_seed")
            row: dict[str, object] = {
                "training_run_count": int(len(group)),
                "method": method.name,
                "projection_mode": mode,
            }
            for metric in ABSOLUTE_METRICS:
                values = group[metric].astype(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sample_sd"] = float(values.std(ddof=1))
                row[f"{metric}_minimum"] = float(values.min())
                row[f"{metric}_maximum"] = float(values.max())
            rows.append(row)
    return pd.DataFrame(rows)


def _leave_one_seed_out(design: object, checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(design.seeds) < 2:
        return pd.DataFrame(rows)
    for method in design.methods:
        for mode in method.modes:
            group = checkpoints[
                checkpoints["method"].eq(method.name)
                & checkpoints["projection_mode"].eq(mode)
            ].sort_values("train_seed")
            for omitted_seed in design.seeds:
                retained = group[~group["train_seed"].eq(omitted_seed)]
                require(
                    len(retained) == len(design.seeds) - 1,
                    f"Cannot omit seed {omitted_seed} from {method.name!r}, {mode!r}.",
                )
                row: dict[str, object] = {
                    "method": method.name,
                    "omitted_train_seed": int(omitted_seed),
                    "projection_mode": mode,
                    "retained_training_run_count": int(len(retained)),
                }
                for metric in ABSOLUTE_METRICS:
                    row[f"{metric}_mean"] = float(retained[metric].astype(float).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _worked_sample_sd(
    checkpoints: pd.DataFrame,
    method: str,
    projection_mode: str,
) -> dict[str, object]:
    group = checkpoints[
        checkpoints["method"].eq(method)
        & checkpoints["projection_mode"].eq(projection_mode)
    ].sort_values("train_seed")
    require(not group.empty, f"Worked-example method/mode was not found: {method!r}, {projection_mode!r}.")
    require(len(group) >= 2, "Worked sample SD requires at least two training runs.")
    calculations: dict[str, object] = {}
    for metric in ("success_rate", "collision_rate", "timeout_rate"):
        values = group[metric].astype(float).to_numpy()
        mean = float(values.sum() / values.size)
        squared_deviations = np.square(values - mean)
        squared_deviation_sum = float(squared_deviations.sum())
        denominator = int(values.size - 1)
        sample_variance = squared_deviation_sum / denominator
        sample_sd = math.sqrt(sample_variance)
        calculations[metric] = {
            "formula": "sqrt(sum((x_i - mean)^2) / (n - 1))",
            "mean": mean,
            "sample_sd": sample_sd,
            "sample_variance": sample_variance,
            "squared_deviation_sum": squared_deviation_sum,
            "squared_deviations": [float(value) for value in squared_deviations],
            "variance_denominator_n_minus_1": denominator,
            "values_by_train_seed": [
                {"train_seed": int(seed), "value": float(value)}
                for seed, value in zip(group["train_seed"], values)
            ],
        }
    return {
        "calculations": calculations,
        "method": method,
        "projection_mode": projection_mode,
    }


def build_absolute_summary(
    *,
    episodes_path: str,
    protocol_path: str,
    layout_suite_path: str,
    checkpoint_summary_path: str,
    method_summary_path: str,
    worked_example_method: str | None,
    worked_example_projection_mode: str | None,
    label: str | None,
) -> dict[str, object]:
    require(
        (worked_example_method is None) == (worked_example_projection_mode is None),
        "Worked example requires both --worked-example-method and --worked-example-projection-mode.",
    )
    design = load_design(protocol_path, layout_suite_path)
    episodes = load_episode_table(episodes_path)
    episode_checks = validate_episode_table(design, episodes)
    checkpoints, _, reconciliation = reconcile_checkpoint_and_method_tables(
        design,
        episodes,
        checkpoint_summary_path,
        method_summary_path,
        str(episode_checks["result_build_schema_version"]),
    )
    checkpoints = _with_timeout(checkpoints)
    outcome_counts = (
        episodes.groupby(["method", "projection_mode"], sort=True)
        .agg(
            collision_count=("collision", "sum"),
            episode_count=("episode", "size"),
            success_count=("success", "sum"),
            timeout_count=("truncated", "sum"),
        )
        .reset_index()
    )
    require(
        bool(
            outcome_counts[["success_count", "collision_count", "timeout_count"]]
            .sum(axis=1)
            .eq(outcome_counts["episode_count"])
            .all()
        ),
        "Outcome counts do not sum to episode coverage.",
    )
    training_run_absolute_metrics = checkpoints[
        [
            "method",
            "projection_mode",
            "train_seed",
            "checkpoint",
            "checkpoint_sha256",
            "episode_count",
            "layout_count",
            *ABSOLUTE_METRICS,
        ]
    ].sort_values(["method", "projection_mode", "train_seed"])
    worked = None
    if worked_example_method is not None and worked_example_projection_mode is not None:
        worked = _worked_sample_sd(
            checkpoints,
            worked_example_method,
            worked_example_projection_mode,
        )
    result: dict[str, object] = {
        "absolute_method_summary": frame_records(_absolute_summary(design, checkpoints)),
        "training_run_absolute_metrics": frame_records(training_run_absolute_metrics),
        "design": design.record(),
        "episode_checks": episode_checks,
        "implementation": implementation_record(__file__),
        "inputs": input_records(
            {
                "checkpoint_summary": checkpoint_summary_path,
                "episodes": episodes_path,
                "layout_suite": layout_suite_path,
                "method_summary": method_summary_path,
                "protocol": protocol_path,
            }
        ),
        "leave_one_seed_out": frame_records(_leave_one_seed_out(design, checkpoints)),
        "outcome_counts_for_coverage_only": frame_records(outcome_counts),
        "output_schema": OUTPUT_SCHEMA,
        "reconciliation": reconciliation,
        "replication": {
            "independent_unit": "training_run_represented_by_final_checkpoint",
            "training_run_count_per_method": len(design.seeds),
        },
        "script": "summarize_absolute_evaluation_performance",
        "worked_sample_sd": worked,
    }
    if label is not None:
        result["label"] = label
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct training-run-level absolute performance from explicitly supplied "
            "episode, protocol, layout, checkpoint-summary, and method-summary files."
        )
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--layout-suite", required=True)
    parser.add_argument("--checkpoint-summary", required=True)
    parser.add_argument("--method-summary", required=True)
    parser.add_argument("--worked-example-method")
    parser.add_argument(
        "--worked-example-projection-mode",
        choices=("disabled", "enabled"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--label")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_absolute_summary(
            episodes_path=args.episodes,
            protocol_path=args.protocol,
            layout_suite_path=args.layout_suite,
            checkpoint_summary_path=args.checkpoint_summary,
            method_summary_path=args.method_summary,
            worked_example_method=args.worked_example_method,
            worked_example_projection_mode=args.worked_example_projection_mode,
            label=args.label,
        )
        write_json_exclusive(args.output, result)
    except EvidenceError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
