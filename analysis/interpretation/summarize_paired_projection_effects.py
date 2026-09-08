"""Reconstruct and summarize within-training-run projection-on effects.

The estimand is always projection enabled minus projection disabled for the
same method, training run, retained final checkpoint, layout/repeat, and
evaluation seed. Method-level summaries treat the training run as the empirical
replicate; one final checkpoint represents each run.
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
        reconstruct_paired_deltas,
        reconcile_paired_tables,
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
        reconstruct_paired_deltas,
        reconcile_paired_tables,
        require,
        validate_episode_table,
        write_json_exclusive,
    )


ZERO_TOLERANCE = 1.0e-12
PAIR_KEYS = ("layout_id", "layout_repeat", "evaluation_seed")

# These names describe the training-run-level mean of episode-paired differences.
DELTA_METRICS = (
    "success_rate_delta_enabled_minus_disabled",
    "collision_rate_delta_enabled_minus_disabled",
    "timeout_rate_delta_enabled_minus_disabled",
    "episode_return_delta_enabled_minus_disabled",
    "episode_length_delta_enabled_minus_disabled",
    "min_obstacle_clearance_delta_enabled_minus_disabled",
    "action_bound_clipping_rate_delta_enabled_minus_disabled",
    "speed_action_bound_clipping_rate_delta_enabled_minus_disabled",
    "turn_rate_action_bound_clipping_rate_delta_enabled_minus_disabled",
    "mean_action_bound_clipping_norm_delta_enabled_minus_disabled",
    "max_action_bound_clipping_norm_delta_enabled_minus_disabled",
)

OUTCOME_DELTA_METRICS = DELTA_METRICS[:3]

ABSOLUTE_SOURCE_COLUMNS = {
    "episode_return": "episode_return",
    "episode_length": "episode_length",
    "min_obstacle_clearance": "min_obstacle_clearance",
    "action_bound_clipping_rate": "action_bound_clipping_rate",
    "speed_action_bound_clipping_rate": "speed_action_bound_clipping_rate",
    "turn_rate_action_bound_clipping_rate": "turn_rate_action_bound_clipping_rate",
    "mean_action_bound_clipping_norm": "mean_action_bound_clipping_norm",
    "max_action_bound_clipping_norm": "max_action_bound_clipping_norm",
}


def _rename_and_add_timeout(paired: pd.DataFrame) -> pd.DataFrame:
    result = paired.rename(
        columns={
            "paired_layout_count": "paired_evaluation_observation_count",
            "success_delta_enabled_minus_disabled": (
                "success_rate_delta_enabled_minus_disabled"
            ),
            "collision_delta_enabled_minus_disabled": (
                "collision_rate_delta_enabled_minus_disabled"
            ),
        }
    ).copy()
    result["timeout_rate_delta_enabled_minus_disabled"] = -(
        result["success_rate_delta_enabled_minus_disabled"]
        + result["collision_rate_delta_enabled_minus_disabled"]
    )
    outcome_sum = result[list(OUTCOME_DELTA_METRICS)].sum(axis=1)
    require(
        bool(np.allclose(outcome_sum, 0.0, atol=ZERO_TOLERANCE, rtol=0.0)),
        "Training-run outcome deltas do not sum to zero.",
    )
    require(
        set(DELTA_METRICS).issubset(result.columns),
        "Reconstructed paired table is missing a required analysis metric.",
    )
    return result


def _training_run_paired_effects(
    design: object,
    episodes: pd.DataFrame,
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach absolute outcome counts/rates and verify every delta identity."""

    display_names = {method.name: method.display_name for method in design.methods}
    rows: list[dict[str, object]] = []
    direct_timeout_errors: list[float] = []
    outcome_sum_errors: list[float] = []
    absolute_delta_errors: dict[str, list[float]] = {
        metric: [] for metric in ABSOLUTE_SOURCE_COLUMNS
    }
    for paired_row in paired.sort_values(["method", "train_seed"]).to_dict(
        orient="records"
    ):
        method = str(paired_row["method"])
        train_seed = int(paired_row["train_seed"])
        group = episodes[
            episodes["method"].eq(method)
            & episodes["train_seed"].eq(train_seed)
        ]
        row: dict[str, object] = dict(paired_row)
        row["display_name"] = display_names[method]
        mode_values: dict[str, dict[str, float]] = {}
        for mode in ("disabled", "enabled"):
            mode_group = group[group["projection_mode"].eq(mode)]
            episode_count = int(len(mode_group))
            require(
                episode_count
                == int(row["paired_evaluation_observation_count"]),
                f"Paired coverage differs for {method!r}, seed {train_seed}, {mode!r}.",
            )
            counts = {
                "success": int(mode_group["success"].sum()),
                "collision": int(mode_group["collision"].sum()),
                "timeout": int(mode_group["truncated"].sum()),
            }
            require(
                sum(counts.values()) == episode_count,
                f"Outcome counts do not partition {method!r}, seed {train_seed}, {mode!r}.",
            )
            values: dict[str, float] = {
                "success_rate": counts["success"] / episode_count,
                "collision_rate": counts["collision"] / episode_count,
                "timeout_rate": counts["timeout"] / episode_count,
            }
            for metric, source_column in ABSOLUTE_SOURCE_COLUMNS.items():
                values[metric] = float(mode_group[source_column].astype(float).mean())
            mode_values[mode] = values
            row[f"episode_count_{mode}"] = episode_count
            for outcome, count in counts.items():
                row[f"{outcome}_count_{mode}"] = count
                row[f"{outcome}_rate_{mode}"] = values[f"{outcome}_rate"]
            for metric in ABSOLUTE_SOURCE_COLUMNS:
                row[f"{metric}_{mode}"] = values[metric]

        for outcome in ("success", "collision", "timeout"):
            metric = f"{outcome}_rate_delta_enabled_minus_disabled"
            direct_delta = (
                mode_values["enabled"][f"{outcome}_rate"]
                - mode_values["disabled"][f"{outcome}_rate"]
            )
            if outcome == "timeout":
                direct_timeout_errors.append(abs(direct_delta - float(row[metric])))
            require(
                math.isclose(
                    direct_delta,
                    float(row[metric]),
                    abs_tol=ZERO_TOLERANCE,
                    rel_tol=0.0,
                ),
                f"Direct and reconstructed {metric} differ for {method!r}, seed {train_seed}.",
            )
            row[f"{metric}_percentage_points"] = 100.0 * float(row[metric])
        for metric in ABSOLUTE_SOURCE_COLUMNS:
            delta_column = f"{metric}_delta_enabled_minus_disabled"
            direct_delta = mode_values["enabled"][metric] - mode_values["disabled"][metric]
            if math.isnan(direct_delta) and pd.isna(row[delta_column]):
                error = 0.0
            else:
                error = abs(direct_delta - float(row[delta_column]))
            absolute_delta_errors[metric].append(error)
            require(
                error <= ZERO_TOLERANCE,
                f"Direct and reconstructed {delta_column} differ for {method!r}, seed {train_seed}.",
            )
        outcome_sum_error = abs(
            sum(float(row[metric]) for metric in OUTCOME_DELTA_METRICS)
        )
        outcome_sum_errors.append(outcome_sum_error)
        rows.append(row)
    checks: dict[str, object] = {
        "all_checks_pass": True,
        "maximum_abs_training_run_error_for_success_plus_collision_plus_timeout_delta": (
            max(outcome_sum_errors, default=0.0)
        ),
        "maximum_abs_timeout_direct_vs_derived_error": max(
            direct_timeout_errors, default=0.0
        ),
        "maximum_abs_direct_vs_paired_delta_errors": {
            metric: max(errors, default=0.0)
            for metric, errors in absolute_delta_errors.items()
        },
        "tolerance": ZERO_TOLERANCE,
    }
    return pd.DataFrame(rows), checks


def _finite_values(group: pd.DataFrame, metric: str) -> np.ndarray:
    values = group[metric].astype(float).to_numpy()
    return values[np.isfinite(values)]


def _sign_counts(values: np.ndarray) -> tuple[int, int, int]:
    negative = int(np.sum(values < -ZERO_TOLERANCE))
    positive = int(np.sum(values > ZERO_TOLERANCE))
    zero = int(values.size - negative - positive)
    return negative, zero, positive


def _paired_method_summary(design: object, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in design.paired_methods:
        group = paired[paired["method"].eq(method.name)].sort_values("train_seed")
        require(
            len(group) == len(design.seeds),
            f"Paired training-run count differs for {method.name!r}.",
        )
        row: dict[str, object] = {
            "training_run_count": int(len(group)),
            "display_name": method.display_name,
            "method": method.name,
        }
        for metric in DELTA_METRICS:
            values = _finite_values(group, metric)
            negative, zero, positive = _sign_counts(values)
            row[f"{metric}_finite_count"] = int(values.size)
            row[f"{metric}_missing_count"] = int(len(group) - values.size)
            row[f"{metric}_mean"] = (
                float(values.mean()) if values.size else math.nan
            )
            row[f"{metric}_sample_sd"] = (
                float(values.std(ddof=1)) if values.size >= 2 else math.nan
            )
            row[f"{metric}_minimum"] = (
                float(values.min()) if values.size else math.nan
            )
            row[f"{metric}_maximum"] = (
                float(values.max()) if values.size else math.nan
            )
            row[f"{metric}_negative_count"] = negative
            row[f"{metric}_zero_count"] = zero
            row[f"{metric}_positive_count"] = positive
            if metric in OUTCOME_DELTA_METRICS:
                for suffix in ("mean", "sample_sd", "minimum", "maximum"):
                    row[f"{metric}_{suffix}_percentage_points"] = (
                        100.0 * float(row[f"{metric}_{suffix}"])
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def _leave_one_seed_out(design: object, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(design.seeds) < 2:
        return pd.DataFrame(rows)
    for method in design.paired_methods:
        group = paired[paired["method"].eq(method.name)].sort_values("train_seed")
        for omitted_seed in design.seeds:
            retained = group[~group["train_seed"].eq(omitted_seed)]
            require(
                len(retained) == len(design.seeds) - 1,
                f"Cannot omit seed {omitted_seed} from {method.name!r}.",
            )
            row: dict[str, object] = {
                "display_name": method.display_name,
                "method": method.name,
                "omitted_train_seed": int(omitted_seed),
                "retained_training_run_count": int(len(retained)),
            }
            for metric in DELTA_METRICS:
                values = _finite_values(retained, metric)
                row[f"{metric}_mean"] = (
                    float(values.mean()) if values.size else math.nan
                )
                if metric in OUTCOME_DELTA_METRICS:
                    row[f"{metric}_mean_percentage_points"] = (
                        100.0 * float(row[f"{metric}_mean"])
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def _leave_one_seed_out_ranges(
    design: object,
    leave_one_out: pd.DataFrame,
) -> pd.DataFrame:
    if leave_one_out.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for method, group in leave_one_out.groupby("method", sort=True):
        row: dict[str, object] = {
            "display_name": design.method_map[str(method)].display_name,
            "method": str(method),
        }
        for metric in DELTA_METRICS:
            column = f"{metric}_mean"
            values = _finite_values(group.rename(columns={column: metric}), metric)
            row[f"{metric}_minimum_leave_one_out_mean"] = (
                float(values.min()) if values.size else math.nan
            )
            row[f"{metric}_maximum_leave_one_out_mean"] = (
                float(values.max()) if values.size else math.nan
            )
            if metric in OUTCOME_DELTA_METRICS:
                row[f"{metric}_minimum_leave_one_out_mean_percentage_points"] = (
                    100.0 * float(row[f"{metric}_minimum_leave_one_out_mean"])
                )
                row[f"{metric}_maximum_leave_one_out_mean_percentage_points"] = (
                    100.0 * float(row[f"{metric}_maximum_leave_one_out_mean"])
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _worked_outcome_pair(
    episodes: pd.DataFrame,
    method: str,
    train_seed: int,
) -> dict[str, object]:
    group = episodes[
        episodes["method"].eq(method) & episodes["train_seed"].eq(train_seed)
    ]
    require(not group.empty, f"Worked-example method/seed was not found: {method!r}, {train_seed}.")
    require(
        set(group["projection_mode"].astype(str)) == {"disabled", "enabled"},
        "Worked example requires both projection modes.",
    )
    mode_rows: list[dict[str, object]] = []
    rates: dict[str, dict[str, float]] = {}
    for mode in ("disabled", "enabled"):
        mode_group = group[group["projection_mode"].eq(mode)]
        episode_count = int(len(mode_group))
        require(episode_count > 0, f"Worked-example mode {mode!r} has no episodes.")
        counts = {
            "success": int(mode_group["success"].sum()),
            "collision": int(mode_group["collision"].sum()),
            "timeout": int(mode_group["truncated"].sum()),
        }
        require(
            sum(counts.values()) == episode_count,
            f"Worked-example outcome counts do not partition {mode!r} episodes.",
        )
        rates[mode] = {
            outcome: count / episode_count for outcome, count in counts.items()
        }
        mode_rows.append(
            {
                "collision_count": counts["collision"],
                "collision_rate": rates[mode]["collision"],
                "episode_count": episode_count,
                "projection_mode": mode,
                "success_count": counts["success"],
                "success_rate": rates[mode]["success"],
                "timeout_count": counts["timeout"],
                "timeout_rate": rates[mode]["timeout"],
            }
        )
    deltas = {
        f"{outcome}_rate_delta_enabled_minus_disabled": (
            rates["enabled"][outcome] - rates["disabled"][outcome]
        )
        for outcome in ("success", "collision", "timeout")
    }
    outcome_delta_sum = float(sum(deltas.values()))
    require(
        math.isclose(
            outcome_delta_sum,
            0.0,
            abs_tol=ZERO_TOLERANCE,
            rel_tol=0.0,
        ),
        "Worked-example outcome deltas do not sum to zero.",
    )
    return {
        "delta_formula": "rate_enabled - rate_disabled",
        "deltas": deltas,
        "method": method,
        "outcome_delta_sum": outcome_delta_sum,
        "projection_modes": mode_rows,
        "rate_formula": "outcome_count / episode_count",
        "train_seed": int(train_seed),
    }


def _worked_sample_sd(
    paired: pd.DataFrame,
    method: str,
) -> dict[str, object]:
    group = paired[paired["method"].eq(method)].sort_values("train_seed")
    require(not group.empty, f"Worked-example method was not found: {method!r}.")
    require(len(group) >= 2, "Worked sample SD requires at least two training runs.")
    calculations: dict[str, object] = {}
    for metric in OUTCOME_DELTA_METRICS:
        values = group[metric].astype(float).to_numpy()
        mean = float(values.sum() / values.size)
        squared_deviations = np.square(values - mean)
        squared_deviation_sum = float(squared_deviations.sum())
        denominator = int(values.size - 1)
        sample_variance = squared_deviation_sum / denominator
        calculations[metric] = {
            "formula": "sqrt(sum((delta_i - mean_delta)^2) / (n - 1))",
            "mean": mean,
            "sample_sd": math.sqrt(sample_variance),
            "sample_variance": sample_variance,
            "squared_deviation_sum": squared_deviation_sum,
            "squared_deviations": [float(value) for value in squared_deviations],
            "values_by_train_seed": [
                {"train_seed": int(seed), "value": float(value)}
                for seed, value in zip(group["train_seed"], values)
            ],
            "variance_denominator_n_minus_1": denominator,
        }
    return {"calculations": calculations, "method": method}


def _metric_definitions() -> dict[str, object]:
    return {
        "training_run_delta": {
            "formula": (
                "mean_over_paired_evaluations(metric_enabled - metric_disabled)"
            ),
            "pairing_keys": list(PAIR_KEYS),
            "replicate_for_method_summary": "training_run_represented_by_final_checkpoint",
            "sign_convention": "projection_enabled_minus_projection_disabled",
        },
        "method_mean_delta": {
            "formula": "sum(training_run_delta_i) / training_run_count",
        },
        "sample_sd": {
            "formula": "sqrt(sum((training_run_delta_i - mean_delta)^2) / (n - 1))",
            "unit_of_variation": "training_run_level_projection_effect",
        },
        "leave_one_seed_out_mean": {
            "formula": (
                "sum(training_run_delta_i for i != omitted_seed) / "
                "(training_run_count - 1)"
            ),
            "purpose": "descriptive check that one training run does not determine the mean direction",
        },
        "absolute_training_run_values": {
            "aggregation": "mean across evaluation observations within one training run and projection mode",
            "maximum_metric_caveat": (
                "max_action_bound_clipping_norm is first an episode-level maximum; "
                "the reported training-run value is the mean of those episode maxima, "
                "not the maximum across episodes"
            ),
        },
        "outcome_delta_identity": {
            "formula": "delta_success + delta_collision + delta_timeout = 0",
        },
        "outcome_rate_units": {
            "stored_unit": "fraction",
            "reporting_conversion": "percentage_points = 100 * fraction_delta",
        },
        "paired_evaluation_observation_count": {
            "definition": (
                "number of matched layout_id, layout_repeat, and evaluation_seed observations "
                "within one training run, represented by one final checkpoint; not a count of training runs or unique layouts"
            ),
            "frozen_source_column": "paired_layout_count",
        },
        "sign_counts": {
            "negative": f"value < -{ZERO_TOLERANCE}",
            "positive": f"value > {ZERO_TOLERANCE}",
            "zero": f"absolute(value) <= {ZERO_TOLERANCE}",
        },
    }


def build_paired_summary(
    *,
    episodes_path: str,
    protocol_path: str,
    layout_suite_path: str,
    paired_deltas_path: str,
    paired_summary_path: str,
    worked_example_method: str | None,
    worked_example_train_seed: int | None,
    label: str | None,
) -> dict[str, object]:
    require(
        (worked_example_method is None) == (worked_example_train_seed is None),
        "Worked example requires both --worked-example-method and --worked-example-train-seed.",
    )
    design = load_design(protocol_path, layout_suite_path)
    episodes = load_episode_table(episodes_path)
    episode_checks = validate_episode_table(design, episodes)
    reconciliation = reconcile_paired_tables(
        design,
        episodes,
        paired_deltas_path,
        paired_summary_path,
        str(episode_checks["result_build_schema_version"]),
    )
    paired = _rename_and_add_timeout(reconstruct_paired_deltas(design, episodes))
    training_run_effects, identity_checks = _training_run_paired_effects(
        design,
        episodes,
        paired,
    )
    leave_one_out = _leave_one_seed_out(design, paired)
    worked_pair = None
    worked_sd = None
    if worked_example_method is not None and worked_example_train_seed is not None:
        worked_pair = _worked_outcome_pair(
            episodes,
            worked_example_method,
            worked_example_train_seed,
        )
        worked_sd = _worked_sample_sd(paired, worked_example_method)
    result: dict[str, object] = {
        "training_run_paired_effects": frame_records(
            training_run_effects.sort_values(["method", "train_seed"])
        ),
        "coverage": {
            "independent_unit": "training_run_represented_by_final_checkpoint",
            "training_run_count_per_method": len(design.seeds),
            "layout_count": len(design.layout_ids),
            "paired_training_run_effect_count": len(training_run_effects),
            "paired_evaluation_observations_per_training_run": (
                len(design.layout_ids) * design.repeats_per_layout
            ),
            "paired_method_count": len(design.paired_methods),
            "repeats_per_layout": design.repeats_per_layout,
            "replication_warning": (
                "Paired evaluation observations improve evaluation coverage but are not "
                "additional training-run replicates."
            ),
        },
        "design": design.record(),
        "episode_checks": episode_checks,
        "implementation": implementation_record(__file__),
        "inputs": input_records(
            {
                "episodes": episodes_path,
                "layout_suite": layout_suite_path,
                "paired_deltas": paired_deltas_path,
                "paired_summary": paired_summary_path,
                "protocol": protocol_path,
            }
        ),
        "leave_one_seed_out": frame_records(leave_one_out),
        "leave_one_seed_out_ranges": frame_records(
            _leave_one_seed_out_ranges(design, leave_one_out)
        ),
        "metric_definitions": _metric_definitions(),
        "output_schema": OUTPUT_SCHEMA,
        "paired_method_summary": frame_records(
            _paired_method_summary(design, paired)
        ),
        "outcome_identity_checks": identity_checks,
        "reconciliation": reconciliation,
        "script": "summarize_paired_projection_effects",
        "worked_outcome_pair": worked_pair,
        "worked_sample_sd": worked_sd,
    }
    if label is not None:
        result["label"] = label
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct within-training-run projection enabled-minus-disabled "
            "effects from explicitly supplied episode, protocol, layout, paired-delta, "
            "and paired-summary files."
        )
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--layout-suite", required=True)
    parser.add_argument("--paired-deltas", required=True)
    parser.add_argument("--paired-summary", required=True)
    parser.add_argument("--worked-example-method")
    parser.add_argument("--worked-example-train-seed", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_paired_summary(
            episodes_path=args.episodes,
            protocol_path=args.protocol,
            layout_suite_path=args.layout_suite,
            paired_deltas_path=args.paired_deltas,
            paired_summary_path=args.paired_summary,
            worked_example_method=args.worked_example_method,
            worked_example_train_seed=args.worked_example_train_seed,
            label=args.label,
        )
        write_json_exclusive(args.output, result)
    except EvidenceError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
