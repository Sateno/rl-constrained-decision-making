# Reconstruct matched projection-disabled to projection-enabled terminal-outcome
# correspondences without altering the supplied evidence. Disabled outcomes are
# rows and enabled outcomes are columns. The arrow is a paired table direction,
# not a temporal within-trajectory transition or a formal causal claim.

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
        implementation_record,
        input_records,
        load_design,
        load_episode_table,
        reconcile_checkpoint_and_method_tables,
        reconcile_paired_tables,
        reconstruct_paired_deltas,
        reconstruct_paired_summary,
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
        implementation_record,
        input_records,
        load_design,
        load_episode_table,
        reconcile_checkpoint_and_method_tables,
        reconcile_paired_tables,
        reconstruct_paired_deltas,
        reconstruct_paired_summary,
        require,
        validate_build_audit,
        validate_episode_table,
        write_json_exclusive,
    )


ZERO_TOLERANCE = 1.0e-12
ANALYSIS_SCHEMA = "paired_terminal_outcome_transitions_v1"
OUTCOMES = ("success", "collision", "timeout")
PAIR_KEYS = (
    "method",
    "train_seed",
    "checkpoint_sha256",
    "layout_id",
    "layout_repeat",
    "evaluation_seed",
)


def _outcome_series(episodes: pd.DataFrame) -> pd.Series:
    partition = (
        episodes["success"].astype(int)
        + episodes["collision"].astype(int)
        + episodes["truncated"].astype(int)
    )
    require(
        bool(partition.eq(1).all()),
        "Success, collision, and timeout do not partition every episode.",
    )
    values = np.select(
        [
            episodes["success"].to_numpy(dtype=bool),
            episodes["collision"].to_numpy(dtype=bool),
            episodes["truncated"].to_numpy(dtype=bool),
        ],
        list(OUTCOMES),
        default="invalid",
    )
    require(
        set(values).issubset(OUTCOMES),
        "A terminal outcome could not be assigned.",
    )
    return pd.Series(values, index=episodes.index, name="terminal_outcome")


def _pair_episode_outcomes(design: object, episodes: pd.DataFrame) -> pd.DataFrame:
    frame = episodes.copy()
    frame["terminal_outcome"] = _outcome_series(frame)
    paired_method_names = {method.name for method in design.paired_methods}
    frame = frame[frame["method"].isin(paired_method_names)].copy()
    disabled = frame[frame["projection_mode"].eq("disabled")]
    enabled = frame[frame["projection_mode"].eq("enabled")]
    require(
        not bool(disabled.duplicated(list(PAIR_KEYS), keep=False).any()),
        "Projection-disabled terminal-outcome pairing keys are duplicated.",
    )
    require(
        not bool(enabled.duplicated(list(PAIR_KEYS), keep=False).any()),
        "Projection-enabled terminal-outcome pairing keys are duplicated.",
    )
    try:
        paired = disabled[list(PAIR_KEYS) + ["checkpoint", "terminal_outcome"]].merge(
            enabled[list(PAIR_KEYS) + ["checkpoint", "terminal_outcome"]],
            on=list(PAIR_KEYS),
            how="outer",
            validate="one_to_one",
            suffixes=("_disabled", "_enabled"),
            indicator=True,
        )
    except pd.errors.MergeError as error:
        raise EvidenceError(
            "Projection-disabled and projection-enabled terminal outcomes cannot be paired one-to-one."
        ) from error
    require(
        bool(paired["_merge"].eq("both").all()),
        "A projection-disabled or projection-enabled terminal-outcome pair is missing.",
    )
    require(
        bool(
            paired["checkpoint_disabled"].astype(str).eq(
                paired["checkpoint_enabled"].astype(str)
            ).all()
        ),
        "Projection modes use different checkpoint paths within a terminal-outcome pair.",
    )
    expected_per_checkpoint = len(design.layout_ids) * design.repeats_per_layout
    expected_total = (
        len(design.paired_methods) * len(design.seeds) * expected_per_checkpoint
    )
    require(
        len(paired) == expected_total,
        "Paired terminal-outcome count differs from protocol-derived coverage.",
    )
    return paired.drop(columns=["_merge"]).sort_values(list(PAIR_KEYS)).reset_index(
        drop=True
    )


def _count_matrix(group: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(OUTCOMES), len(OUTCOMES)), dtype=int)
    for row_index, disabled_outcome in enumerate(OUTCOMES):
        for column_index, enabled_outcome in enumerate(OUTCOMES):
            matrix[row_index, column_index] = int(
                (
                    group["terminal_outcome_disabled"].eq(disabled_outcome)
                    & group["terminal_outcome_enabled"].eq(enabled_outcome)
                ).sum()
            )
    return matrix


def _integer_matrix(matrix: np.ndarray) -> list[list[int]]:
    return [[int(value) for value in row] for row in matrix.tolist()]


def _float_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def _optional_float_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not math.isfinite(float(value)) else float(value) for value in row]
        for row in matrix.tolist()
    ]


def _conditional_matrix(counts: np.ndarray) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in counts:
        total = int(row.sum())
        if total == 0:
            rows.append([None for _ in OUTCOMES])
        else:
            rows.append([float(value / total) for value in row])
    return rows


def _transition_cells(counts: np.ndarray, denominator: int) -> list[dict[str, object]]:
    row_margins = counts.sum(axis=1)
    cells: list[dict[str, object]] = []
    for row_index, disabled_outcome in enumerate(OUTCOMES):
        for column_index, enabled_outcome in enumerate(OUTCOMES):
            count = int(counts[row_index, column_index])
            row_total = int(row_margins[row_index])
            cells.append(
                {
                    "projection_disabled_outcome": disabled_outcome,
                    "projection_enabled_outcome": enabled_outcome,
                    "count": count,
                    "proportion_of_all_pairs": count / denominator,
                    "conditional_proportion_given_disabled_outcome": (
                        count / row_total if row_total else None
                    ),
                }
            )
    return cells


def _checkpoint_transition_records(
    design: object,
    episodes: pd.DataFrame,
    paired: pd.DataFrame,
) -> list[dict[str, object]]:
    display_names = {method.name: method.display_name for method in design.methods}
    expected_pairs = len(design.layout_ids) * design.repeats_per_layout
    records: list[dict[str, object]] = []
    for method in design.paired_methods:
        for train_seed in design.seeds:
            group = paired[
                paired["method"].eq(method.name)
                & paired["train_seed"].eq(train_seed)
            ]
            require(
                len(group) == expected_pairs,
                f"Terminal-outcome pair count differs for {method.name!r}, seed {train_seed}.",
            )
            checkpoint_hashes = set(group["checkpoint_sha256"].astype(str))
            checkpoint_paths = set(group["checkpoint_disabled"].astype(str))
            require(
                len(checkpoint_hashes) == 1 and len(checkpoint_paths) == 1,
                f"Checkpoint identity is ambiguous for {method.name!r}, seed {train_seed}.",
            )
            counts = _count_matrix(group)
            row_margins = counts.sum(axis=1)
            column_margins = counts.sum(axis=0)
            require(
                int(counts.sum()) == expected_pairs,
                f"Terminal-outcome matrix total differs for {method.name!r}, seed {train_seed}.",
            )
            disabled_episode_group = episodes[
                episodes["method"].eq(method.name)
                & episodes["train_seed"].eq(train_seed)
                & episodes["projection_mode"].eq("disabled")
            ]
            enabled_episode_group = episodes[
                episodes["method"].eq(method.name)
                & episodes["train_seed"].eq(train_seed)
                & episodes["projection_mode"].eq("enabled")
            ]
            direct_disabled = np.array(
                [
                    int(disabled_episode_group["success"].sum()),
                    int(disabled_episode_group["collision"].sum()),
                    int(disabled_episode_group["truncated"].sum()),
                ]
            )
            direct_enabled = np.array(
                [
                    int(enabled_episode_group["success"].sum()),
                    int(enabled_episode_group["collision"].sum()),
                    int(enabled_episode_group["truncated"].sum()),
                ]
            )
            require(
                bool(np.array_equal(row_margins, direct_disabled)),
                f"Transition row margins differ from disabled outcomes for {method.name!r}, seed {train_seed}.",
            )
            require(
                bool(np.array_equal(column_margins, direct_enabled)),
                f"Transition column margins differ from enabled outcomes for {method.name!r}, seed {train_seed}.",
            )
            delta_counts = column_margins - row_margins
            delta_rates = delta_counts.astype(float) / expected_pairs
            require(
                int(delta_counts.sum()) == 0,
                f"Terminal-outcome delta counts do not sum to zero for {method.name!r}, seed {train_seed}.",
            )
            records.append(
                {
                    "method": method.name,
                    "display_name": display_names[method.name],
                    "train_seed": int(train_seed),
                    "checkpoint": next(iter(checkpoint_paths)),
                    "checkpoint_sha256": next(iter(checkpoint_hashes)),
                    "layout_count": len(design.layout_ids),
                    "paired_evaluation_observation_count": expected_pairs,
                    "transition_count_matrix": _integer_matrix(counts),
                    "transition_proportion_matrix": _float_matrix(
                        counts.astype(float) / expected_pairs
                    ),
                    "conditional_enabled_outcome_given_disabled_matrix": (
                        _conditional_matrix(counts)
                    ),
                    "transition_cells": _transition_cells(counts, expected_pairs),
                    "projection_disabled_outcome_counts": [
                        int(value) for value in row_margins
                    ],
                    "projection_enabled_outcome_counts": [
                        int(value) for value in column_margins
                    ],
                    "projection_disabled_outcome_rates": [
                        float(value / expected_pairs) for value in row_margins
                    ],
                    "projection_enabled_outcome_rates": [
                        float(value / expected_pairs) for value in column_margins
                    ],
                    "enabled_minus_disabled_outcome_count_deltas": [
                        int(value) for value in delta_counts
                    ],
                    "enabled_minus_disabled_outcome_rate_deltas": [
                        float(value) for value in delta_rates
                    ],
                    "enabled_minus_disabled_outcome_percentage_point_deltas": [
                        100.0 * float(value) for value in delta_rates
                    ],
                    "same_outcome_count": int(np.trace(counts)),
                    "different_outcome_count": int(counts.sum() - np.trace(counts)),
                }
            )
    return records


def _sample_sd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    if values.shape[axis] < 2:
        shape = values.shape[:axis] + values.shape[axis + 1 :]
        return np.full(shape, np.nan, dtype=float)
    return values.std(axis=axis, ddof=1)


def _sign_counts(values: np.ndarray) -> dict[str, int]:
    negative = int(np.sum(values < -ZERO_TOLERANCE))
    positive = int(np.sum(values > ZERO_TOLERANCE))
    return {
        "negative": negative,
        "zero": int(values.size - negative - positive),
        "positive": positive,
    }


def _method_transition_records(
    design: object,
    checkpoint_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for method in design.paired_methods:
        group = [row for row in checkpoint_records if row["method"] == method.name]
        require(
            len(group) == len(design.seeds),
            f"Checkpoint transition count differs for {method.name!r}.",
        )
        counts = np.asarray(
            [row["transition_count_matrix"] for row in group], dtype=int
        )
        pair_counts = np.asarray(
            [row["paired_evaluation_observation_count"] for row in group],
            dtype=int,
        )
        require(
            bool((pair_counts == pair_counts[0]).all()),
            f"Per-checkpoint pair counts differ for {method.name!r}.",
        )
        proportions = counts.astype(float) / pair_counts[:, None, None]
        pooled_counts = counts.sum(axis=0)
        pooled_pair_count = int(pair_counts.sum())
        pooled_proportions = pooled_counts.astype(float) / pooled_pair_count
        mean_proportions = proportions.mean(axis=0)
        pooled_mean_error = float(
            np.max(np.abs(pooled_proportions - mean_proportions))
        )
        require(
            pooled_mean_error <= ZERO_TOLERANCE,
            f"Pooled and equal-checkpoint transition proportions differ for {method.name!r}.",
        )
        disabled_rates = np.asarray(
            [row["projection_disabled_outcome_rates"] for row in group],
            dtype=float,
        )
        enabled_rates = np.asarray(
            [row["projection_enabled_outcome_rates"] for row in group],
            dtype=float,
        )
        delta_rates = enabled_rates - disabled_rates
        delta_summaries: list[dict[str, object]] = []
        for index, outcome in enumerate(OUTCOMES):
            values = delta_rates[:, index]
            delta_summaries.append(
                {
                    "outcome": outcome,
                    "mean": float(values.mean()),
                    "sample_sd": (
                        float(values.std(ddof=1)) if values.size >= 2 else None
                    ),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "sign_counts": _sign_counts(values),
                }
            )
        cell_summaries: list[dict[str, object]] = []
        pooled_row_margins = pooled_counts.sum(axis=1)
        for row_index, disabled_outcome in enumerate(OUTCOMES):
            for column_index, enabled_outcome in enumerate(OUTCOMES):
                values = proportions[:, row_index, column_index]
                row_total = int(pooled_row_margins[row_index])
                pooled_count = int(pooled_counts[row_index, column_index])
                cell_summaries.append(
                    {
                        "projection_disabled_outcome": disabled_outcome,
                        "projection_enabled_outcome": enabled_outcome,
                        "pooled_count": pooled_count,
                        "pooled_proportion_of_all_pairs": float(
                            pooled_proportions[row_index, column_index]
                        ),
                        "pooled_conditional_proportion_given_disabled_outcome": (
                            pooled_count / row_total if row_total else None
                        ),
                        "mean_checkpoint_proportion_of_all_pairs": float(
                            values.mean()
                        ),
                        "sample_sd_checkpoint_proportion_of_all_pairs": (
                            float(values.std(ddof=1)) if values.size >= 2 else None
                        ),
                        "minimum_checkpoint_proportion_of_all_pairs": float(
                            values.min()
                        ),
                        "maximum_checkpoint_proportion_of_all_pairs": float(
                            values.max()
                        ),
                    }
                )
        records.append(
            {
                "method": method.name,
                "display_name": method.display_name,
                "independent_checkpoint_count": len(group),
                "paired_evaluation_observations_per_checkpoint": int(
                    pair_counts[0]
                ),
                "pooled_paired_evaluation_observation_count": pooled_pair_count,
                "pooled_transition_count_matrix": _integer_matrix(pooled_counts),
                "pooled_transition_proportion_matrix": _float_matrix(
                    pooled_proportions
                ),
                "pooled_transition_cells": _transition_cells(
                    pooled_counts, pooled_pair_count
                ),
                "checkpoint_transition_cell_summaries": cell_summaries,
                "mean_checkpoint_transition_proportion_matrix": _float_matrix(
                    mean_proportions
                ),
                "sample_sd_checkpoint_transition_proportion_matrix": (
                    _optional_float_matrix(_sample_sd(proportions, axis=0))
                ),
                "minimum_checkpoint_transition_proportion_matrix": _float_matrix(
                    proportions.min(axis=0)
                ),
                "maximum_checkpoint_transition_proportion_matrix": _float_matrix(
                    proportions.max(axis=0)
                ),
                "pooled_conditional_enabled_outcome_given_disabled_matrix": (
                    _conditional_matrix(pooled_counts)
                ),
                "pooled_projection_disabled_outcome_counts": [
                    int(value) for value in pooled_counts.sum(axis=1)
                ],
                "pooled_projection_enabled_outcome_counts": [
                    int(value) for value in pooled_counts.sum(axis=0)
                ],
                "mean_checkpoint_projection_disabled_outcome_rates": [
                    float(value) for value in disabled_rates.mean(axis=0)
                ],
                "sample_sd_checkpoint_projection_disabled_outcome_rates": [
                    None if not math.isfinite(float(value)) else float(value)
                    for value in _sample_sd(disabled_rates, axis=0)
                ],
                "mean_checkpoint_projection_enabled_outcome_rates": [
                    float(value) for value in enabled_rates.mean(axis=0)
                ],
                "sample_sd_checkpoint_projection_enabled_outcome_rates": [
                    None if not math.isfinite(float(value)) else float(value)
                    for value in _sample_sd(enabled_rates, axis=0)
                ],
                "enabled_minus_disabled_outcome_rate_delta_summary": (
                    delta_summaries
                ),
                "pooled_enabled_minus_disabled_outcome_count_deltas": [
                    int(value)
                    for value in (
                        pooled_counts.sum(axis=0) - pooled_counts.sum(axis=1)
                    )
                ],
                "pooled_same_outcome_count": int(np.trace(pooled_counts)),
                "pooled_different_outcome_count": int(
                    pooled_counts.sum() - np.trace(pooled_counts)
                ),
                "pooled_vs_equal_checkpoint_proportion_max_abs_error": (
                    pooled_mean_error
                ),
            }
        )
    return records


def _maximum_error(errors: list[float]) -> float:
    return max(errors, default=0.0)


def _transition_reconciliation(
    design: object,
    checkpoint_records: list[dict[str, object]],
    method_records: list[dict[str, object]],
    reconstructed_checkpoints: pd.DataFrame,
    reconstructed_methods: pd.DataFrame,
    reconstructed_paired: pd.DataFrame,
    reconstructed_paired_summary: pd.DataFrame,
) -> dict[str, object]:
    checkpoint_margin_errors: list[float] = []
    checkpoint_delta_errors: list[float] = []
    method_margin_errors: list[float] = []
    method_delta_errors: list[float] = []
    comparison_count = 0

    for record in checkpoint_records:
        method = str(record["method"])
        train_seed = int(record["train_seed"])
        margin_values: dict[str, np.ndarray] = {}
        for mode in ("disabled", "enabled"):
            row = reconstructed_checkpoints[
                reconstructed_checkpoints["method"].eq(method)
                & reconstructed_checkpoints["train_seed"].eq(train_seed)
                & reconstructed_checkpoints["projection_mode"].eq(mode)
            ]
            require(len(row) == 1, "A reconstructed checkpoint margin row is missing.")
            success = float(row.iloc[0]["success_rate"])
            collision = float(row.iloc[0]["collision_rate"])
            expected = np.array([success, collision, 1.0 - success - collision])
            actual = np.asarray(
                record[f"projection_{mode}_outcome_rates"], dtype=float
            )
            checkpoint_margin_errors.extend(np.abs(actual - expected).tolist())
            comparison_count += len(OUTCOMES)
            margin_values[mode] = expected
        paired_row = reconstructed_paired[
            reconstructed_paired["method"].eq(method)
            & reconstructed_paired["train_seed"].eq(train_seed)
        ]
        require(len(paired_row) == 1, "A reconstructed paired checkpoint row is missing.")
        expected_delta = np.array(
            [
                float(
                    paired_row.iloc[0][
                        "success_delta_enabled_minus_disabled"
                    ]
                ),
                float(
                    paired_row.iloc[0][
                        "collision_delta_enabled_minus_disabled"
                    ]
                ),
                0.0,
            ]
        )
        expected_delta[2] = -(expected_delta[0] + expected_delta[1])
        direct_expected = margin_values["enabled"] - margin_values["disabled"]
        checkpoint_delta_errors.extend(
            np.abs(expected_delta - direct_expected).tolist()
        )
        actual_delta = np.asarray(
            record["enabled_minus_disabled_outcome_rate_deltas"], dtype=float
        )
        checkpoint_delta_errors.extend(np.abs(actual_delta - expected_delta).tolist())
        comparison_count += 2 * len(OUTCOMES)

    for record in method_records:
        method = str(record["method"])
        for mode in ("disabled", "enabled"):
            row = reconstructed_methods[
                reconstructed_methods["method"].eq(method)
                & reconstructed_methods["projection_mode"].eq(mode)
            ]
            require(len(row) == 1, "A reconstructed method margin row is missing.")
            mean_actual = np.asarray(
                record[f"mean_checkpoint_projection_{mode}_outcome_rates"],
                dtype=float,
            )
            sd_actual_values = record[
                f"sample_sd_checkpoint_projection_{mode}_outcome_rates"
            ]
            sd_actual = np.asarray(
                [math.nan if value is None else value for value in sd_actual_values],
                dtype=float,
            )
            checkpoint_rows = reconstructed_checkpoints[
                reconstructed_checkpoints["method"].eq(method)
                & reconstructed_checkpoints["projection_mode"].eq(mode)
            ].sort_values("train_seed")
            success_values = checkpoint_rows["success_rate"].to_numpy(dtype=float)
            collision_values = checkpoint_rows["collision_rate"].to_numpy(dtype=float)
            timeout_values = 1.0 - success_values - collision_values
            expected_values = np.column_stack(
                [success_values, collision_values, timeout_values]
            )
            expected_mean = expected_values.mean(axis=0)
            expected_sd = _sample_sd(expected_values, axis=0)
            method_margin_errors.extend(np.abs(mean_actual - expected_mean).tolist())
            finite_sd = np.isfinite(expected_sd)
            method_margin_errors.extend(
                np.abs(sd_actual[finite_sd] - expected_sd[finite_sd]).tolist()
            )
            comparison_count += int(len(OUTCOMES) + finite_sd.sum())
            for index, metric in enumerate(("success_rate", "collision_rate")):
                method_margin_errors.append(
                    abs(float(row.iloc[0][f"{metric}_mean"]) - expected_mean[index])
                )
                if math.isfinite(float(expected_sd[index])):
                    method_margin_errors.append(
                        abs(
                            float(row.iloc[0][f"{metric}_std"])
                            - float(expected_sd[index])
                        )
                    )
                    comparison_count += 1
                comparison_count += 1

        delta_records = {
            str(item["outcome"]): item
            for item in record["enabled_minus_disabled_outcome_rate_delta_summary"]
        }
        paired_method_row = reconstructed_paired_summary[
            reconstructed_paired_summary["method"].eq(method)
        ]
        require(len(paired_method_row) == 1, "A reconstructed paired method row is missing.")
        checkpoint_paired_rows = reconstructed_paired[
            reconstructed_paired["method"].eq(method)
        ].sort_values("train_seed")
        success_delta = checkpoint_paired_rows[
            "success_delta_enabled_minus_disabled"
        ].to_numpy(dtype=float)
        collision_delta = checkpoint_paired_rows[
            "collision_delta_enabled_minus_disabled"
        ].to_numpy(dtype=float)
        expected_deltas = np.column_stack(
            [success_delta, collision_delta, -(success_delta + collision_delta)]
        )
        expected_mean = expected_deltas.mean(axis=0)
        expected_sd = _sample_sd(expected_deltas, axis=0)
        for index, outcome in enumerate(OUTCOMES):
            actual = delta_records[outcome]
            method_delta_errors.append(abs(float(actual["mean"]) - expected_mean[index]))
            comparison_count += 1
            if actual["sample_sd"] is not None:
                method_delta_errors.append(
                    abs(float(actual["sample_sd"]) - expected_sd[index])
                )
                comparison_count += 1
        for index, metric in enumerate(("success", "collision")):
            method_delta_errors.append(
                abs(
                    float(
                        paired_method_row.iloc[0][
                            f"{metric}_delta_enabled_minus_disabled_mean"
                        ]
                    )
                    - expected_mean[index]
                )
            )
            method_delta_errors.append(
                abs(
                    float(
                        paired_method_row.iloc[0][
                            f"{metric}_delta_enabled_minus_disabled_std"
                        ]
                    )
                    - expected_sd[index]
                )
            )
            comparison_count += 2

    maximums = {
        "checkpoint_margin_max_abs_error": _maximum_error(
            checkpoint_margin_errors
        ),
        "checkpoint_paired_delta_max_abs_error": _maximum_error(
            checkpoint_delta_errors
        ),
        "method_margin_max_abs_error": _maximum_error(method_margin_errors),
        "method_paired_delta_max_abs_error": _maximum_error(method_delta_errors),
    }
    require(
        max(maximums.values(), default=0.0) <= ZERO_TOLERANCE,
        "Terminal-outcome matrices do not reconcile with the supplied summary hierarchy.",
    )
    return {
        "all_checks_pass": True,
        "comparison_count": comparison_count,
        "integer_row_and_column_margins_exact": True,
        "outcome_delta_sum_zero_exact": True,
        "maximum_absolute_errors": maximums,
        "tolerance": ZERO_TOLERANCE,
    }


def build_terminal_outcome_transition_summary(
    *,
    episodes_path: str,
    protocol_path: str,
    layout_suite_path: str,
    checkpoint_summary_path: str,
    method_summary_path: str,
    paired_deltas_path: str,
    paired_summary_path: str,
    build_audit_path: str,
    label: str | None,
) -> dict[str, object]:
    design = load_design(protocol_path, layout_suite_path)
    episodes = load_episode_table(episodes_path)
    episode_checks = validate_episode_table(design, episodes)
    schema = str(episode_checks["result_build_schema_version"])
    reconstructed_checkpoints, reconstructed_methods, absolute_reconciliation = (
        reconcile_checkpoint_and_method_tables(
            design,
            episodes,
            checkpoint_summary_path,
            method_summary_path,
            schema,
        )
    )
    paired_reconciliation = reconcile_paired_tables(
        design,
        episodes,
        paired_deltas_path,
        paired_summary_path,
        schema,
    )
    build_audit_checks = validate_build_audit(
        design, build_audit_path, schema, episodes
    )
    reconstructed_paired = reconstruct_paired_deltas(design, episodes)
    reconstructed_paired_summary = reconstruct_paired_summary(reconstructed_paired)
    paired = _pair_episode_outcomes(design, episodes)
    checkpoint_records = _checkpoint_transition_records(design, episodes, paired)
    method_records = _method_transition_records(design, checkpoint_records)
    transition_reconciliation = _transition_reconciliation(
        design,
        checkpoint_records,
        method_records,
        reconstructed_checkpoints,
        reconstructed_methods,
        reconstructed_paired,
        reconstructed_paired_summary,
    )
    result: dict[str, object] = {
        "analysis_schema": ANALYSIS_SCHEMA,
        "output_schema": OUTPUT_SCHEMA,
        "status": "PASS",
        "script": "summarize_paired_terminal_outcome_transitions",
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
        "matrix_definition": {
            "outcome_order": list(OUTCOMES),
            "row_projection_mode": "disabled",
            "column_projection_mode": "enabled",
            "row_outcomes_projection_disabled": list(OUTCOMES),
            "column_outcomes_projection_enabled": list(OUTCOMES),
            "arrow_direction": "projection disabled outcome -> projection enabled outcome",
            "delta_direction": "projection enabled minus projection disabled",
            "pair_keys_excluding_projection_mode": list(PAIR_KEYS),
            "complete_cell_count": len(OUTCOMES) * len(OUTCOMES),
            "count_unit": "matched evaluation observation pair",
            "joint_proportion_denominator": "all matched pairs in the reported checkpoint or pooled method group",
            "conditional_proportion_denominator": "matched pairs having the reported projection-disabled row outcome; null when that row count is zero",
            "timeout_source": "validated truncated indicator in the exhaustive success/collision/timeout outcome partition",
        },
        "coverage": {
            "paired_method_count": len(design.paired_methods),
            "independent_checkpoint_count_per_method": len(design.seeds),
            "paired_evaluation_observations_per_checkpoint": (
                len(design.layout_ids) * design.repeats_per_layout
            ),
            "total_paired_evaluation_observations": len(paired),
            "independent_unit": "trained checkpoint",
        },
        "episode_checks": episode_checks,
        "reconciliation": {
            "checkpoint_and_method_tables": absolute_reconciliation,
            "paired_tables": paired_reconciliation,
            "build_audit": build_audit_checks,
            "terminal_outcome_transitions": transition_reconciliation,
        },
        "checkpoint_transition_matrices": checkpoint_records,
        "method_transition_summaries": method_records,
        "interpretation_boundary": {
            "arrow_meaning": (
                "The arrow maps a matched projection-disabled terminal outcome to the projection-enabled terminal outcome for the same checkpoint and evaluation key."
            ),
            "not_temporal": (
                "It is not a temporal transition within one trajectory; the two controller executions can diverge after their actions differ."
            ),
            "not_mechanism_proof": (
                "The correspondence does not prove deliberate projector reliance, policy-projector co-adaptation, formal safety, or behavior under another projector or arbitrary geometry."
            ),
            "replication": (
                "Checkpoint-level proportions are summarized across independently trained checkpoints; pooled episode counts describe matched evaluation coverage, not independent policy replication."
            ),
        },
    }
    if label is not None:
        result["label"] = label
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct complete matched projection-disabled to projection-enabled terminal-outcome matrices from explicitly supplied evidence files."
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--label")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_terminal_outcome_transition_summary(
            episodes_path=args.episodes,
            protocol_path=args.protocol,
            layout_suite_path=args.layout_suite,
            checkpoint_summary_path=args.checkpoint_summary,
            method_summary_path=args.method_summary,
            paired_deltas_path=args.paired_deltas,
            paired_summary_path=args.paired_summary,
            build_audit_path=args.build_audit,
            label=args.label,
        )
        write_json_exclusive(args.output, result)
    except EvidenceError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
