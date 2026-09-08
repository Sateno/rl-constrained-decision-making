"""Describe performance and projection effects across a supplied layout suite.

The command reconstructs every result from an explicitly supplied episode table,
checks it against the supplied frozen result family, and keeps two sources of
variation separate:

* training runs are the method-level empirical replicates, with one retained
  final checkpoint representing each run; and
* layouts are prespecified repeated task conditions used to assess coverage and
  heterogeneity, not additional independently trained policies.

No repository location, method label, commit, layout count, or expected total is
embedded in this file.  Those facts are derived from the supplied protocol and
layout-suite documents.
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
        reconcile_paired_tables,
        reconstruct_paired_deltas,
        require,
        validate_build_audit,
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
        reconcile_paired_tables,
        reconstruct_paired_deltas,
        require,
        validate_build_audit,
        validate_episode_table,
        write_json_exclusive,
    )


ZERO_TOLERANCE = 1.0e-12
PAIR_KEYS = (
    "method",
    "train_seed",
    "checkpoint_sha256",
    "layout_id",
    "layout_repeat",
    "evaluation_seed",
)
OUTCOMES = ("success", "collision", "timeout")
OUTCOME_RATE_METRICS = tuple(f"{outcome}_rate" for outcome in OUTCOMES)
SUPPORTING_METRICS = (
    "episode_return",
    "episode_length",
    "min_obstacle_clearance",
)
ABSOLUTE_METRICS = (*OUTCOME_RATE_METRICS, *SUPPORTING_METRICS)
DELTA_METRICS = tuple(
    f"{metric}_delta_enabled_minus_disabled" for metric in ABSOLUTE_METRICS
)
OUTCOME_DELTA_METRICS = DELTA_METRICS[:3]


def _layout_metadata(design: object) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, layout in enumerate(design.layout_suite["layouts"]):
        rows.append(
            {
                "layout_id": str(layout["layout_id"]),
                "layout_index": index,
                "obstacle_count": len(layout.get("obstacles", [])),
            }
        )
    return pd.DataFrame(rows)


def _with_timeout(episodes: pd.DataFrame) -> pd.DataFrame:
    result = episodes.copy()
    result["timeout"] = result["truncated"].astype(bool)
    partition = (
        result["success"].astype(int)
        + result["collision"].astype(int)
        + result["timeout"].astype(int)
    )
    require(
        bool(partition.eq(1).all()),
        "Success, collision, and timeout do not partition every episode.",
    )
    return result


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _sample_sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size >= 2 else math.nan


def _sign_counts(values: np.ndarray) -> tuple[int, int, int]:
    negative = int(np.sum(values < -ZERO_TOLERANCE))
    positive = int(np.sum(values > ZERO_TOLERANCE))
    zero = int(values.size - negative - positive)
    return negative, zero, positive


def _mode_outcome_counts(group: pd.DataFrame) -> dict[str, int]:
    return {outcome: int(group[outcome].astype(int).sum()) for outcome in OUTCOMES}


def _absolute_group_row(group: pd.DataFrame) -> dict[str, object]:
    observation_count = int(len(group))
    require(observation_count > 0, "Cannot summarize an empty evaluation group.")
    counts = _mode_outcome_counts(group)
    require(
        sum(counts.values()) == observation_count,
        "Outcome counts do not partition an absolute-performance group.",
    )
    row: dict[str, object] = {"evaluation_observation_count": observation_count}
    for outcome in OUTCOMES:
        row[f"{outcome}_count"] = counts[outcome]
        row[f"{outcome}_rate"] = counts[outcome] / observation_count
    for metric in SUPPORTING_METRICS:
        row[metric] = float(group[metric].astype(float).mean())
    return row


def _training_run_layout_absolute(
    design: object,
    episodes: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    display_names = {method.name: method.display_name for method in design.methods}
    rows: list[dict[str, object]] = []
    keys = [
        "method",
        "train_seed",
        "checkpoint_sha256",
        "projection_mode",
        "layout_id",
    ]
    for key, group in episodes.groupby(keys, sort=True):
        method, train_seed, checkpoint_sha256, projection_mode, layout_id = key
        row = {
            "method": str(method),
            "display_name": display_names[str(method)],
            "train_seed": int(train_seed),
            "checkpoint_sha256": str(checkpoint_sha256),
            "projection_mode": str(projection_mode),
            "layout_id": str(layout_id),
            "repeat_count": int(group["layout_repeat"].nunique()),
            **_absolute_group_row(group),
        }
        rows.append(row)
    result = pd.DataFrame(rows).merge(metadata, on="layout_id", validate="many_to_one")
    return result.sort_values(
        ["method", "projection_mode", "train_seed", "layout_index"]
    ).reset_index(drop=True)


def _training_run_absolute(
    design: object,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    display_names = {method.name: method.display_name for method in design.methods}
    rows: list[dict[str, object]] = []
    keys = ["method", "train_seed", "checkpoint", "checkpoint_sha256", "projection_mode"]
    for key, group in episodes.groupby(keys, sort=True):
        method, train_seed, checkpoint, checkpoint_sha256, projection_mode = key
        rows.append(
            {
                "method": str(method),
                "display_name": display_names[str(method)],
                "train_seed": int(train_seed),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": str(checkpoint_sha256),
                "projection_mode": str(projection_mode),
                "layout_count": int(group["layout_id"].nunique()),
                "repeat_count_per_layout": int(group["layout_repeat"].nunique()),
                **_absolute_group_row(group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["method", "projection_mode", "train_seed"]
    ).reset_index(drop=True)


def _absolute_method_summary(
    design: object,
    training_runs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in design.methods:
        for mode in method.modes:
            group = training_runs[
                training_runs["method"].eq(method.name)
                & training_runs["projection_mode"].eq(mode)
            ]
            require(
                len(group) == len(design.seeds),
                f"Training-run coverage differs for {method.name!r}, {mode!r}.",
            )
            row: dict[str, object] = {
                "method": method.name,
                "display_name": method.display_name,
                "projection_mode": mode,
                "training_run_count": int(len(group)),
                "evaluation_observation_count": int(
                    group["evaluation_observation_count"].sum()
                ),
            }
            for outcome in OUTCOMES:
                row[f"{outcome}_count_for_coverage"] = int(
                    group[f"{outcome}_count"].sum()
                )
            for metric in ABSOLUTE_METRICS:
                values = _finite(group[metric])
                row[f"{metric}_mean"] = (
                    float(values.mean()) if values.size else math.nan
                )
                row[f"{metric}_sample_sd"] = _sample_sd(values)
                row[f"{metric}_minimum"] = (
                    float(values.min()) if values.size else math.nan
                )
                row[f"{metric}_maximum"] = (
                    float(values.max()) if values.size else math.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _layout_absolute_summary(
    design: object,
    training_run_layout: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["method", "display_name", "projection_mode", "layout_id", "layout_index", "obstacle_count"]
    for key, group in training_run_layout.groupby(keys, sort=True):
        method, display_name, projection_mode, layout_id, layout_index, obstacle_count = key
        require(
            len(group) == len(design.seeds),
            f"Training-run coverage differs for layout {layout_id!r}, method {method!r}, {projection_mode!r}.",
        )
        row: dict[str, object] = {
            "method": str(method),
            "display_name": str(display_name),
            "projection_mode": str(projection_mode),
            "layout_id": str(layout_id),
            "layout_index": int(layout_index),
            "obstacle_count": int(obstacle_count),
            "training_run_count": int(len(group)),
            "evaluation_observation_count": int(
                group["evaluation_observation_count"].sum()
            ),
        }
        for outcome in OUTCOMES:
            row[f"{outcome}_count_for_coverage"] = int(
                group[f"{outcome}_count"].sum()
            )
        for metric in ABSOLUTE_METRICS:
            values = _finite(group[metric])
            row[f"{metric}_mean_across_training_runs"] = (
                float(values.mean()) if values.size else math.nan
            )
            row[f"{metric}_sample_sd_across_training_runs"] = _sample_sd(values)
            row[f"{metric}_minimum_across_training_runs"] = (
                float(values.min()) if values.size else math.nan
            )
            row[f"{metric}_maximum_across_training_runs"] = (
                float(values.max()) if values.size else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["method", "projection_mode", "layout_index"]
    ).reset_index(drop=True)


def _pair_episode_rows(design: object, episodes: pd.DataFrame) -> pd.DataFrame:
    metrics = ("success", "collision", "timeout", *SUPPORTING_METRICS)
    paired_rows: list[pd.DataFrame] = []
    for method in design.paired_methods:
        group = episodes[episodes["method"].eq(method.name)]
        disabled = group[group["projection_mode"].eq("disabled")]
        enabled = group[group["projection_mode"].eq("enabled")]
        try:
            merged = disabled[list(PAIR_KEYS) + list(metrics)].merge(
                enabled[list(PAIR_KEYS) + list(metrics)],
                on=list(PAIR_KEYS),
                suffixes=("_disabled", "_enabled"),
                validate="one_to_one",
            )
        except pd.errors.MergeError as error:
            raise EvidenceError(
                f"Cannot pair projection modes for method {method.name!r}."
            ) from error
        require(
            len(merged) == len(disabled) == len(enabled),
            f"Incomplete episode pairing for method {method.name!r}.",
        )
        merged["display_name"] = method.display_name
        for outcome in OUTCOMES:
            merged[f"{outcome}_rate_delta_enabled_minus_disabled"] = (
                merged[f"{outcome}_enabled"].astype(int)
                - merged[f"{outcome}_disabled"].astype(int)
            ).astype(float)
        for metric in SUPPORTING_METRICS:
            merged[f"{metric}_delta_enabled_minus_disabled"] = (
                merged[f"{metric}_enabled"].astype(float)
                - merged[f"{metric}_disabled"].astype(float)
            )
        paired_rows.append(merged)
    require(bool(paired_rows), "The protocol contains no method with both projection modes.")
    result = pd.concat(paired_rows, ignore_index=True)
    outcome_sum = result[list(OUTCOME_DELTA_METRICS)].sum(axis=1)
    require(
        bool(np.allclose(outcome_sum, 0.0, atol=ZERO_TOLERANCE, rtol=0.0)),
        "Episode-paired outcome deltas do not sum to zero.",
    )
    return result


def _training_run_layout_effects(
    design: object,
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["method", "display_name", "train_seed", "checkpoint_sha256", "layout_id"]
    for key, group in pairs.groupby(keys, sort=True):
        method, display_name, train_seed, checkpoint_sha256, layout_id = key
        row: dict[str, object] = {
            "method": str(method),
            "display_name": str(display_name),
            "train_seed": int(train_seed),
            "checkpoint_sha256": str(checkpoint_sha256),
            "layout_id": str(layout_id),
            "paired_evaluation_observation_count": int(len(group)),
        }
        for metric in DELTA_METRICS:
            row[metric] = float(group[metric].astype(float).mean())
            if metric in OUTCOME_DELTA_METRICS:
                row[f"{metric}_percentage_points"] = 100.0 * float(row[metric])
        rows.append(row)
    result = pd.DataFrame(rows).merge(metadata, on="layout_id", validate="many_to_one")
    expected = len(design.paired_methods) * len(design.seeds) * len(design.layout_ids)
    require(len(result) == expected, "Checkpoint-layout effect coverage is incomplete.")
    return result.sort_values(["method", "train_seed", "layout_index"]).reset_index(drop=True)


def _training_run_effects(
    design: object,
    training_run_layout: pd.DataFrame,
    reconstructed_paired: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    for (method, display_name, train_seed, checkpoint_sha256), group in training_run_layout.groupby(
        ["method", "display_name", "train_seed", "checkpoint_sha256"], sort=True
    ):
        require(
            len(group) == len(design.layout_ids),
            f"Layout coverage differs for {method!r}, seed {int(train_seed)}.",
        )
        row: dict[str, object] = {
            "method": str(method),
            "display_name": str(display_name),
            "train_seed": int(train_seed),
            "checkpoint_sha256": str(checkpoint_sha256),
            "layout_count": int(len(group)),
        }
        for metric in DELTA_METRICS:
            row[metric] = float(group[metric].astype(float).mean())
            if metric in OUTCOME_DELTA_METRICS:
                row[f"{metric}_percentage_points"] = 100.0 * float(row[metric])
        rows.append(row)
    result = pd.DataFrame(rows)

    committed_shape = reconstructed_paired.rename(
        columns={
            "success_delta_enabled_minus_disabled": OUTCOME_DELTA_METRICS[0],
            "collision_delta_enabled_minus_disabled": OUTCOME_DELTA_METRICS[1],
        }
    ).copy()
    committed_shape[OUTCOME_DELTA_METRICS[2]] = -(
        committed_shape[OUTCOME_DELTA_METRICS[0]]
        + committed_shape[OUTCOME_DELTA_METRICS[1]]
    )
    comparison_metrics = (
        *OUTCOME_DELTA_METRICS,
        "episode_return_delta_enabled_minus_disabled",
        "episode_length_delta_enabled_minus_disabled",
        "min_obstacle_clearance_delta_enabled_minus_disabled",
    )
    merged = result.merge(
        committed_shape[["method", "train_seed", "checkpoint_sha256", *comparison_metrics]],
        on=["method", "train_seed", "checkpoint_sha256"],
        suffixes=("_layout_reconstruction", "_episode_reconstruction"),
        validate="one_to_one",
    )
    errors: list[float] = []
    for metric in comparison_metrics:
        left = merged[f"{metric}_layout_reconstruction"].to_numpy(dtype=float)
        right = merged[f"{metric}_episode_reconstruction"].to_numpy(dtype=float)
        both_nan = np.isnan(left) & np.isnan(right)
        require(
            bool(np.array_equal(np.isnan(left), np.isnan(right))),
            f"Layout and episode reconstructions disagree on missingness for {metric}.",
        )
        finite = ~both_nan
        error = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        errors.append(error)
        require(
            error <= ZERO_TOLERANCE,
            f"Layout and episode reconstructions differ for {metric}.",
        )
    return result.sort_values(["method", "train_seed"]), max(errors, default=0.0)


def _paired_method_summary(
    design: object,
    training_runs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in design.paired_methods:
        group = training_runs[training_runs["method"].eq(method.name)]
        require(
            len(group) == len(design.seeds),
            f"Paired training-run coverage differs for {method.name!r}.",
        )
        row: dict[str, object] = {
            "method": method.name,
            "display_name": method.display_name,
            "training_run_count": int(len(group)),
        }
        for metric in DELTA_METRICS:
            values = _finite(group[metric])
            negative, zero, positive = _sign_counts(values)
            row[f"{metric}_finite_count"] = int(values.size)
            row[f"{metric}_mean"] = float(values.mean()) if values.size else math.nan
            row[f"{metric}_sample_sd"] = _sample_sd(values)
            row[f"{metric}_minimum"] = float(values.min()) if values.size else math.nan
            row[f"{metric}_maximum"] = float(values.max()) if values.size else math.nan
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


def _layout_paired_summary(
    design: object,
    training_run_layout: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["method", "display_name", "layout_id", "layout_index", "obstacle_count"]
    for key, group in training_run_layout.groupby(keys, sort=True):
        method, display_name, layout_id, layout_index, obstacle_count = key
        require(
            len(group) == len(design.seeds),
            f"Paired training-run coverage differs for {method!r}, layout {layout_id!r}.",
        )
        row: dict[str, object] = {
            "method": str(method),
            "display_name": str(display_name),
            "layout_id": str(layout_id),
            "layout_index": int(layout_index),
            "obstacle_count": int(obstacle_count),
            "training_run_count": int(len(group)),
        }
        for metric in DELTA_METRICS:
            values = _finite(group[metric])
            negative, zero, positive = _sign_counts(values)
            row[f"{metric}_finite_count"] = int(values.size)
            row[f"{metric}_mean_across_training_runs"] = (
                float(values.mean()) if values.size else math.nan
            )
            row[f"{metric}_sample_sd_across_training_runs"] = _sample_sd(values)
            row[f"{metric}_minimum_across_training_runs"] = (
                float(values.min()) if values.size else math.nan
            )
            row[f"{metric}_maximum_across_training_runs"] = (
                float(values.max()) if values.size else math.nan
            )
            row[f"{metric}_negative_training_run_count"] = negative
            row[f"{metric}_zero_training_run_count"] = zero
            row[f"{metric}_positive_training_run_count"] = positive
            if metric in OUTCOME_DELTA_METRICS:
                row[f"{metric}_mean_across_training_runs_percentage_points"] = (
                    100.0 * float(row[f"{metric}_mean_across_training_runs"])
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "layout_index"]).reset_index(drop=True)


def _layout_effect_coverage(
    design: object,
    layout_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in design.paired_methods:
        group = layout_summary[layout_summary["method"].eq(method.name)]
        require(len(group) == len(design.layout_ids), f"Layout coverage differs for {method.name!r}.")
        for metric in DELTA_METRICS:
            column = f"{metric}_mean_across_training_runs"
            values = _finite(group[column])
            negative, zero, positive = _sign_counts(values)
            rows.append(
                {
                    "method": method.name,
                    "display_name": method.display_name,
                    "metric": metric,
                    "finite_layout_count": int(values.size),
                    "negative_layout_count": negative,
                    "zero_layout_count": zero,
                    "positive_layout_count": positive,
                }
            )
    return pd.DataFrame(rows)


def _leave_one_training_run_out(
    design: object,
    training_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    if len(design.seeds) >= 2:
        for method in design.paired_methods:
            group = training_runs[training_runs["method"].eq(method.name)]
            for omitted_seed in design.seeds:
                retained = group[~group["train_seed"].eq(omitted_seed)]
                row: dict[str, object] = {
                    "method": method.name,
                    "display_name": method.display_name,
                    "omitted_train_seed": int(omitted_seed),
                    "retained_training_run_count": int(len(retained)),
                }
                for metric in DELTA_METRICS:
                    values = _finite(retained[metric])
                    row[f"{metric}_mean"] = (
                        float(values.mean()) if values.size else math.nan
                    )
                rows.append(row)
    details = pd.DataFrame(rows)
    return details, _robustness_ranges(
        design,
        details,
        omitted_column="omitted_train_seed",
        mean_suffix="_mean",
        prefix="leave_one_training_run_out",
    )


def _leave_one_layout_out(
    design: object,
    training_run_layout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    if len(design.layout_ids) >= 2:
        for method in design.paired_methods:
            method_group = training_run_layout[training_run_layout["method"].eq(method.name)]
            for omitted_layout in design.layout_ids:
                retained = method_group[~method_group["layout_id"].eq(omitted_layout)]
                seed_means = retained.groupby("train_seed", sort=True)[list(DELTA_METRICS)].mean()
                require(
                    len(seed_means) == len(design.seeds),
                    f"Cannot omit layout {omitted_layout!r} for {method.name!r}.",
                )
                row: dict[str, object] = {
                    "method": method.name,
                    "display_name": method.display_name,
                    "omitted_layout_id": omitted_layout,
                    "retained_layout_count": len(design.layout_ids) - 1,
                    "training_run_count": int(len(seed_means)),
                }
                for metric in DELTA_METRICS:
                    values = _finite(seed_means[metric])
                    row[f"{metric}_mean"] = (
                        float(values.mean()) if values.size else math.nan
                    )
                rows.append(row)
    details = pd.DataFrame(rows)
    return details, _robustness_ranges(
        design,
        details,
        omitted_column="omitted_layout_id",
        mean_suffix="_mean",
        prefix="leave_one_layout_out",
    )


def _robustness_ranges(
    design: object,
    details: pd.DataFrame,
    *,
    omitted_column: str,
    mean_suffix: str,
    prefix: str,
) -> pd.DataFrame:
    del omitted_column  # Its presence documents the detail-table unit; ranges use all rows.
    if details.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for method in design.paired_methods:
        group = details[details["method"].eq(method.name)]
        row: dict[str, object] = {
            "method": method.name,
            "display_name": method.display_name,
        }
        for metric in DELTA_METRICS:
            values = _finite(group[f"{metric}{mean_suffix}"])
            row[f"{metric}_{prefix}_minimum_mean"] = (
                float(values.min()) if values.size else math.nan
            )
            row[f"{metric}_{prefix}_maximum_mean"] = (
                float(values.max()) if values.size else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _effect_concentration(
    design: object,
    layout_summary: pd.DataFrame,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for method in design.paired_methods:
        group = layout_summary[layout_summary["method"].eq(method.name)]
        for metric in OUTCOME_DELTA_METRICS:
            column = f"{metric}_mean_across_training_runs"
            finite = group[np.isfinite(group[column].astype(float))][
                ["layout_id", "layout_index", column]
            ].copy()
            finite["absolute_magnitude"] = finite[column].abs()
            finite = finite.sort_values(
                ["absolute_magnitude", "layout_index"], ascending=[False, True]
            )
            total_absolute = float(finite["absolute_magnitude"].sum())
            signed_sum = float(finite[column].sum())
            zero_mass = total_absolute <= ZERO_TOLERANCE
            ranked: list[dict[str, object]] = []
            cumulative = 0.0
            count_for_50 = 0
            count_for_80 = 0
            squared_weight_sum = 0.0
            for rank, (_, row) in enumerate(finite.iterrows(), start=1):
                magnitude = float(row["absolute_magnitude"])
                share = None if zero_mass else magnitude / total_absolute
                if share is not None:
                    cumulative += share
                    squared_weight_sum += share * share
                if not zero_mass and count_for_50 == 0 and cumulative >= 0.5 - ZERO_TOLERANCE:
                    count_for_50 = rank
                if not zero_mass and count_for_80 == 0 and cumulative >= 0.8 - ZERO_TOLERANCE:
                    count_for_80 = rank
                if rank <= min(5, len(finite)):
                    ranked.append(
                        {
                            "rank": rank,
                            "layout_id": str(row["layout_id"]),
                            "signed_delta": float(row[column]),
                            "share_of_total_absolute_magnitude": share,
                        }
                    )
            records.append(
                {
                    "method": method.name,
                    "display_name": method.display_name,
                    "metric": metric,
                    "finite_layout_count": int(len(finite)),
                    "zero_mass": zero_mass,
                    "nonzero_layout_count": int(
                        (finite["absolute_magnitude"] > ZERO_TOLERANCE).sum()
                    ),
                    "signed_sum_of_layout_mean_deltas": signed_sum,
                    "sum_of_absolute_layout_mean_deltas": total_absolute,
                    "minimum_layouts_for_50_percent_of_absolute_magnitude": (
                        None if zero_mass else count_for_50
                    ),
                    "minimum_layouts_for_80_percent_of_absolute_magnitude": (
                        None if zero_mass else count_for_80
                    ),
                    "largest_layout_share_of_absolute_magnitude": (
                        None
                        if zero_mass or finite.empty
                        else float(finite.iloc[0]["absolute_magnitude"])
                        / total_absolute
                    ),
                    "effective_layout_count_from_absolute_shares": (
                        None
                        if zero_mass or squared_weight_sum <= ZERO_TOLERANCE
                        else 1.0 / squared_weight_sum
                    ),
                    "cancellation_ratio_abs_signed_sum_over_absolute_sum": (
                        None if zero_mass else abs(signed_sum) / total_absolute
                    ),
                    "top_layouts_by_absolute_magnitude": ranked,
                    "interpretation_warning": (
                        "Absolute-magnitude concentration describes where the aggregate layout effect is numerically concentrated. "
                        "It does not make layouts independent replicates and it does not identify causal mechanisms."
                    ),
                }
            )
    return records


def _outcome_redistribution_accounting(
    design: object,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in design.paired_methods:
        method_group = episodes[episodes["method"].eq(method.name)]
        modes: dict[str, tuple[int, dict[str, int]]] = {}
        for mode in ("disabled", "enabled"):
            group = method_group[method_group["projection_mode"].eq(mode)]
            modes[mode] = (len(group), _mode_outcome_counts(group))
        require(modes["disabled"][0] == modes["enabled"][0], f"Mode coverage differs for {method.name!r}.")
        observation_count = modes["enabled"][0]
        row: dict[str, object] = {
            "method": method.name,
            "display_name": method.display_name,
            "evaluation_observation_count_per_mode": observation_count,
        }
        net_sum = 0
        for outcome in OUTCOMES:
            disabled_count = modes["disabled"][1][outcome]
            enabled_count = modes["enabled"][1][outcome]
            net = enabled_count - disabled_count
            net_sum += net
            row[f"{outcome}_count_disabled"] = disabled_count
            row[f"{outcome}_count_enabled"] = enabled_count
            row[f"{outcome}_net_count_enabled_minus_disabled"] = net
            row[f"{outcome}_rate_delta_enabled_minus_disabled"] = net / observation_count
            row[f"{outcome}_rate_delta_enabled_minus_disabled_percentage_points"] = (
                100.0 * net / observation_count
            )
        require(net_sum == 0, f"Net outcome counts do not sum to zero for {method.name!r}.")
        row["net_outcome_count_sum"] = net_sum
        row["interpretation_warning"] = (
            "These are marginal net differences across paired coverage, not counts of episode-level outcome transitions."
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregation_checks(
    training_run_absolute: pd.DataFrame,
    layout_absolute: pd.DataFrame,
    training_run_layout_effects: pd.DataFrame,
    training_run_effects: pd.DataFrame,
    layout_effects: pd.DataFrame,
    method_summary: pd.DataFrame,
    reconstruction_error: float,
) -> dict[str, object]:
    absolute_training_run_error = float(
        np.max(
            np.abs(
                training_run_absolute[list(OUTCOME_RATE_METRICS)].sum(axis=1).to_numpy(dtype=float)
                - 1.0
            )
        )
    )
    layout_rate_columns = [f"{metric}_mean_across_training_runs" for metric in OUTCOME_RATE_METRICS]
    absolute_layout_error = float(
        np.max(np.abs(layout_absolute[layout_rate_columns].sum(axis=1).to_numpy(dtype=float) - 1.0))
    )
    training_run_layout_delta_error = float(
        np.max(np.abs(training_run_layout_effects[list(OUTCOME_DELTA_METRICS)].sum(axis=1).to_numpy(dtype=float)))
    )
    training_run_delta_error = float(
        np.max(np.abs(training_run_effects[list(OUTCOME_DELTA_METRICS)].sum(axis=1).to_numpy(dtype=float)))
    )
    layout_delta_columns = [f"{metric}_mean_across_training_runs" for metric in OUTCOME_DELTA_METRICS]
    layout_delta_error = float(
        np.max(np.abs(layout_effects[layout_delta_columns].sum(axis=1).to_numpy(dtype=float)))
    )
    aggregation_errors: list[float] = []
    for method in sorted(training_run_effects["method"].unique()):
        training_run_group = training_run_effects[training_run_effects["method"].eq(method)]
        layout_group = layout_effects[layout_effects["method"].eq(method)]
        summary_row = method_summary[method_summary["method"].eq(method)].iloc[0]
        for metric in DELTA_METRICS:
            training_run_mean = float(training_run_group[metric].mean())
            layout_mean = float(layout_group[f"{metric}_mean_across_training_runs"].mean())
            summary_mean = float(summary_row[f"{metric}_mean"])
            if math.isnan(training_run_mean) and math.isnan(layout_mean) and math.isnan(summary_mean):
                aggregation_errors.append(0.0)
            else:
                aggregation_errors.extend(
                    [abs(training_run_mean - layout_mean), abs(training_run_mean - summary_mean)]
                )
    maximum_error = max(
        absolute_training_run_error,
        absolute_layout_error,
        training_run_layout_delta_error,
        training_run_delta_error,
        layout_delta_error,
        reconstruction_error,
        max(aggregation_errors, default=0.0),
    )
    require(maximum_error <= ZERO_TOLERANCE, "An aggregation identity failed.")
    return {
        "all_checks_pass": True,
        "maximum_abs_training_run_absolute_outcome_sum_minus_one": absolute_training_run_error,
        "maximum_abs_layout_absolute_outcome_sum_minus_one": absolute_layout_error,
        "maximum_abs_training_run_layout_outcome_delta_sum": training_run_layout_delta_error,
        "maximum_abs_training_run_outcome_delta_sum": training_run_delta_error,
        "maximum_abs_layout_outcome_delta_sum": layout_delta_error,
        "maximum_abs_layout_weighting_vs_training_run_weighting_error": max(
            aggregation_errors, default=0.0
        ),
        "maximum_abs_layout_reconstruction_vs_episode_reconstruction_error": reconstruction_error,
        "tolerance": ZERO_TOLERANCE,
    }


def _outcome_summary(group: pd.DataFrame) -> dict[str, object]:
    modes: list[dict[str, object]] = []
    rates: dict[str, dict[str, float]] = {}
    for mode in ("disabled", "enabled"):
        mode_group = group[group["projection_mode"].eq(mode)]
        require(not mode_group.empty, f"Worked example has no {mode!r} observations.")
        row = {"projection_mode": mode, **_absolute_group_row(mode_group)}
        rates[mode] = {outcome: float(row[f"{outcome}_rate"]) for outcome in OUTCOMES}
        modes.append(frame_records(pd.DataFrame([row]))[0])
    deltas = {
        f"{outcome}_rate_delta_enabled_minus_disabled": (
            rates["enabled"][outcome] - rates["disabled"][outcome]
        )
        for outcome in OUTCOMES
    }
    require(abs(sum(deltas.values())) <= ZERO_TOLERANCE, "Worked outcome deltas do not sum to zero.")
    return {
        "projection_modes": modes,
        "deltas": deltas,
        "outcome_delta_sum": float(sum(deltas.values())),
    }


def _worked_example(
    episodes: pd.DataFrame,
    method: str,
    train_seed: int,
    layout_id: str,
) -> dict[str, object]:
    method_rows = episodes[episodes["method"].eq(method)]
    require(not method_rows.empty, f"Worked-example method was not found: {method!r}.")
    training_run_rows = method_rows[method_rows["train_seed"].eq(train_seed)]
    require(
        not training_run_rows.empty,
        f"Worked-example method/seed was not found: {method!r}, {train_seed}.",
    )
    layout_rows = method_rows[method_rows["layout_id"].eq(layout_id)]
    require(not layout_rows.empty, f"Worked-example layout was not found: {layout_id!r}.")
    cell_rows = training_run_rows[training_run_rows["layout_id"].eq(layout_id)]
    require(not cell_rows.empty, "Worked-example training-run/layout cell was not found.")
    return {
        "method": method,
        "train_seed": int(train_seed),
        "layout_id": layout_id,
        "rate_formula": "outcome_count / evaluation_observation_count",
        "delta_formula": "rate_enabled - rate_disabled",
        "selected_training_run_across_all_layouts": _outcome_summary(training_run_rows),
        "selected_layout_across_all_training_runs": _outcome_summary(layout_rows),
        "selected_training_run_layout_cell": _outcome_summary(cell_rows),
    }


def _metric_definitions() -> dict[str, object]:
    return {
        "absolute_training_run_metric": {
            "formula": "mean across evaluation observations within one training run and projection mode",
            "replicate_for_method_summary": "training_run_represented_by_final_checkpoint",
        },
        "training_run_layout_effect": {
            "formula": "mean over repeats within one training run and layout of metric_enabled - metric_disabled",
            "pairing_keys": ["layout_id", "layout_repeat", "evaluation_seed"],
        },
        "layout_effect": {
            "formula": "mean across training-run-level layout effects",
            "layout_role": "prespecified repeated task condition, not a training-run replicate",
        },
        "method_effect": {
            "formula": "mean across training-run effects; each training-run effect is the equal-layout mean",
        },
        "sample_sd": {
            "formula": "sqrt(sum((value_i - mean_value)^2) / (n - 1))",
            "method_level_unit": "training_run_represented_by_final_checkpoint",
            "layout_level_unit": "training-run-specific layout effect",
        },
        "outcome_identity": {
            "absolute": "success_rate + collision_rate + timeout_rate = 1",
            "paired": "delta_success + delta_collision + delta_timeout = 0",
        },
        "rate_units": {
            "stored": "fraction",
            "reported_delta": "percentage_points = 100 * fraction_delta",
        },
        "leave_one_training_run_out": {
            "purpose": "check whether one training run determines the aggregate direction",
        },
        "leave_one_layout_out": {
            "purpose": "check whether one prespecified layout determines the aggregate direction",
        },
        "sign_counts": {
            "negative": f"value < -{ZERO_TOLERANCE}",
            "zero": f"absolute(value) <= {ZERO_TOLERANCE}",
            "positive": f"value > {ZERO_TOLERANCE}",
        },
    }


def build_layout_transfer_summary(
    *,
    episodes_path: str,
    protocol_path: str,
    layout_suite_path: str,
    checkpoint_summary_path: str,
    method_summary_path: str,
    paired_deltas_path: str,
    paired_summary_path: str,
    build_audit_path: str,
    worked_example_method: str | None,
    worked_example_train_seed: int | None,
    worked_example_layout: str | None,
    label: str | None,
) -> dict[str, object]:
    worked_arguments = (
        worked_example_method,
        worked_example_train_seed,
        worked_example_layout,
    )
    require(
        all(value is None for value in worked_arguments)
        or all(value is not None for value in worked_arguments),
        "Worked example requires --worked-example-method, --worked-example-train-seed, and --worked-example-layout together.",
    )
    design = load_design(protocol_path, layout_suite_path)
    episodes = _with_timeout(load_episode_table(episodes_path))
    episode_checks = validate_episode_table(design, episodes)
    schema = str(episode_checks["result_build_schema_version"])
    _, _, absolute_reconciliation = reconcile_checkpoint_and_method_tables(
        design,
        episodes,
        checkpoint_summary_path,
        method_summary_path,
        schema,
    )
    paired_reconciliation = reconcile_paired_tables(
        design,
        episodes,
        paired_deltas_path,
        paired_summary_path,
        schema,
    )
    build_audit_checks = validate_build_audit(design, build_audit_path, schema, episodes)

    metadata = _layout_metadata(design)
    training_run_layout_absolute = _training_run_layout_absolute(design, episodes, metadata)
    training_run_absolute = _training_run_absolute(design, episodes)
    absolute_method_summary = _absolute_method_summary(design, training_run_absolute)
    layout_absolute = _layout_absolute_summary(design, training_run_layout_absolute)
    episode_pairs = _pair_episode_rows(design, episodes)
    training_run_layout_effects = _training_run_layout_effects(design, episode_pairs, metadata)
    reconstructed_paired = reconstruct_paired_deltas(design, episodes)
    training_run_effects, reconstruction_error = _training_run_effects(
        design, training_run_layout_effects, reconstructed_paired
    )
    paired_method_summary = _paired_method_summary(design, training_run_effects)
    layout_effects = _layout_paired_summary(design, training_run_layout_effects)
    leave_training_run, leave_training_run_ranges = _leave_one_training_run_out(
        design, training_run_effects
    )
    leave_layout, leave_layout_ranges = _leave_one_layout_out(
        design, training_run_layout_effects
    )
    identity_checks = _aggregation_checks(
        training_run_absolute,
        layout_absolute,
        training_run_layout_effects,
        training_run_effects,
        layout_effects,
        paired_method_summary,
        reconstruction_error,
    )
    worked = None
    if all(value is not None for value in worked_arguments):
        worked = _worked_example(
            episodes,
            str(worked_example_method),
            int(worked_example_train_seed),
            str(worked_example_layout),
        )

    result: dict[str, object] = {
        "output_schema": OUTPUT_SCHEMA,
        "script": "summarize_layout_transfer_performance",
        "implementation": implementation_record(__file__),
        "inputs": input_records(
            {
                "build_audit": build_audit_path,
                "checkpoint_summary": checkpoint_summary_path,
                "episodes": episodes_path,
                "layout_suite": layout_suite_path,
                "method_summary": method_summary_path,
                "paired_deltas": paired_deltas_path,
                "paired_summary": paired_summary_path,
                "protocol": protocol_path,
            }
        ),
        "design": design.record(),
        "coverage": {
            "independent_unit": "training_run_represented_by_final_checkpoint",
            "training_run_count_per_method": len(design.seeds),
            "layout_count": len(design.layout_ids),
            "repeats_per_layout": design.repeats_per_layout,
            "evaluation_observations_per_training_run_mode": (
                len(design.layout_ids) * design.repeats_per_layout
            ),
            "paired_method_count": len(design.paired_methods),
            "replication_warning": (
                "Layouts and repeated evaluation observations improve task coverage but do not create additional training-run replicates."
            ),
        },
        "episode_checks": episode_checks,
        "reconciliation": {
            "checkpoint_and_method_tables": absolute_reconciliation,
            "paired_tables": paired_reconciliation,
            "build_audit": build_audit_checks,
        },
        "aggregation_identity_checks": identity_checks,
        "metric_definitions": _metric_definitions(),
        "training_run_absolute_performance": frame_records(training_run_absolute),
        "absolute_method_summary": frame_records(absolute_method_summary),
        "training_run_layout_absolute_performance": frame_records(
            training_run_layout_absolute
        ),
        "layout_absolute_performance": frame_records(layout_absolute),
        "training_run_layout_paired_effects": frame_records(training_run_layout_effects),
        "training_run_paired_effects": frame_records(training_run_effects),
        "paired_method_summary": frame_records(paired_method_summary),
        "layout_paired_effects": frame_records(layout_effects),
        "layout_effect_sign_coverage": frame_records(
            _layout_effect_coverage(design, layout_effects)
        ),
        "layout_effect_concentration": _effect_concentration(design, layout_effects),
        "leave_one_training_run_out": frame_records(leave_training_run),
        "leave_one_training_run_out_ranges": frame_records(leave_training_run_ranges),
        "leave_one_layout_out": frame_records(leave_layout),
        "leave_one_layout_out_ranges": frame_records(leave_layout_ranges),
        "outcome_redistribution_accounting": frame_records(
            _outcome_redistribution_accounting(design, episodes)
        ),
        "worked_example": worked,
    }
    if label is not None:
        result["label"] = label
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct descriptive layout-transfer performance, heterogeneity, "
            "and within-training-run projection effects from explicitly supplied evidence files."
        )
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--layout-suite", required=True)
    parser.add_argument("--checkpoint-summary", required=True)
    parser.add_argument("--method-summary", required=True)
    parser.add_argument("--paired-deltas", required=True)
    parser.add_argument("--paired-summary", required=True)
    parser.add_argument("--build-audit", required=True)
    parser.add_argument("--worked-example-method")
    parser.add_argument("--worked-example-train-seed", type=int)
    parser.add_argument("--worked-example-layout")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_layout_transfer_summary(
            episodes_path=args.episodes,
            protocol_path=args.protocol,
            layout_suite_path=args.layout_suite,
            checkpoint_summary_path=args.checkpoint_summary,
            method_summary_path=args.method_summary,
            paired_deltas_path=args.paired_deltas,
            paired_summary_path=args.paired_summary,
            build_audit_path=args.build_audit,
            worked_example_method=args.worked_example_method,
            worked_example_train_seed=args.worked_example_train_seed,
            worked_example_layout=args.worked_example_layout,
            label=args.label,
        )
        write_json_exclusive(args.output, result)
    except EvidenceError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
