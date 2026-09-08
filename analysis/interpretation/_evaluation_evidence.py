"""Schema-specific evidence mechanics for explicit-input interpretation commands.

This module deliberately contains the independent reconstruction used by the
interpretation commands.  It does not import the result-building implementation:
reimplementing the documented formulas here lets the validation command detect
disagreement with committed tables instead of reproducing the same implementation
mistake.  Source locations, study labels, commits, method names, and expected row
totals are always supplied by the caller or derived from supplied protocol and
layout-suite files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROTOCOL_SCHEMA = "projection_analysis_protocol_v1"
PROJECTION_MODES = ("disabled", "enabled")
OUTPUT_SCHEMA = "interpretation_analysis_output_v1"

BOOLEAN_COLUMNS = (
    "training_projection_enabled",
    "success",
    "collision",
    "terminated",
    "truncated",
    "projection_enabled",
)
EPISODE_NUMERIC_COLUMNS = (
    "train_seed",
    "training_collision_penalty",
    "layout_repeat",
    "evaluation_seed",
    "evaluation_collision_penalty",
    "max_episode_steps",
    "layout_max_obstacles",
    "layout_agent_radius",
    "layout_goal_radius",
    "episode",
    "seed",
    "episode_return",
    "episode_length",
    "final_distance_to_goal",
    "min_obstacle_clearance",
    "action_bound_clipping_count",
    "action_bound_clipping_rate",
    "speed_action_bound_clipping_count",
    "speed_action_bound_clipping_rate",
    "turn_rate_action_bound_clipping_count",
    "turn_rate_action_bound_clipping_rate",
    "mean_action_bound_clipping_norm",
    "max_action_bound_clipping_norm",
    "projection_intervention_count",
    "projection_intervention_rate",
    "mean_projection_correction_norm",
    "max_projection_correction_norm",
    "mean_projection_slack_sum",
    "max_projection_slack",
    "projection_solver_failure_count",
    "projection_lookahead_distance",
    "projection_alpha",
    "projection_slack_penalty",
    "projection_extra_clearance",
)
INTEGER_EPISODE_COLUMNS = (
    "train_seed",
    "layout_repeat",
    "evaluation_seed",
    "max_episode_steps",
    "layout_max_obstacles",
    "episode",
    "seed",
    "episode_length",
    "action_bound_clipping_count",
    "speed_action_bound_clipping_count",
    "turn_rate_action_bound_clipping_count",
    "projection_intervention_count",
    "projection_solver_failure_count",
)
REQUIRED_EPISODE_COLUMNS = {
    "method",
    "train_seed",
    "training_collision_penalty",
    "training_projection_enabled",
    "checkpoint",
    "checkpoint_sha256",
    "layout_suite_schema_version",
    "layout_suite_id",
    "layout_suite_sha256",
    "layout_id",
    "layout_repeat",
    "evaluation_seed",
    "evaluation_collision_penalty",
    "evaluation_policy_mode",
    "max_episode_steps",
    "layout_max_obstacles",
    "layout_agent_radius",
    "layout_goal_radius",
    "projection_mode",
    "episode",
    "seed",
    "episode_return",
    "episode_length",
    "success",
    "collision",
    "terminated",
    "truncated",
    "final_distance_to_goal",
    "min_obstacle_clearance",
    "action_bound_clipping_count",
    "action_bound_clipping_rate",
    "speed_action_bound_clipping_count",
    "speed_action_bound_clipping_rate",
    "turn_rate_action_bound_clipping_count",
    "turn_rate_action_bound_clipping_rate",
    "mean_action_bound_clipping_norm",
    "max_action_bound_clipping_norm",
    "projection_enabled",
    "projection_intervention_count",
    "projection_intervention_rate",
    "mean_projection_correction_norm",
    "max_projection_correction_norm",
    "mean_projection_slack_sum",
    "max_projection_slack",
    "projection_solver_failure_count",
    "projection_lookahead_distance",
    "projection_alpha",
    "projection_slack_penalty",
    "projection_extra_clearance",
    "result_build_schema_version",
}

CHECKPOINT_KEYS = (
    "method",
    "train_seed",
    "checkpoint",
    "checkpoint_sha256",
    "projection_mode",
)
CHECKPOINT_AGGREGATIONS = {
    "episode_count": ("episode", "size"),
    "layout_count": ("layout_id", "nunique"),
    "episode_return": ("episode_return", "mean"),
    "episode_length": ("episode_length", "mean"),
    "success_rate": ("success", "mean"),
    "collision_rate": ("collision", "mean"),
    "min_obstacle_clearance": ("min_obstacle_clearance", "mean"),
    "action_bound_clipping_rate": ("action_bound_clipping_rate", "mean"),
    "speed_action_bound_clipping_rate": (
        "speed_action_bound_clipping_rate",
        "mean",
    ),
    "turn_rate_action_bound_clipping_rate": (
        "turn_rate_action_bound_clipping_rate",
        "mean",
    ),
    "action_bound_clipping_norm": ("mean_action_bound_clipping_norm", "mean"),
    "action_bound_clipping_norm_max": ("max_action_bound_clipping_norm", "max"),
    "projection_intervention_rate": ("projection_intervention_rate", "mean"),
    "projection_correction_norm": ("mean_projection_correction_norm", "mean"),
    "projection_correction_norm_max": ("max_projection_correction_norm", "max"),
    "projection_slack_sum": ("mean_projection_slack_sum", "mean"),
    "projection_slack_max": ("max_projection_slack", "max"),
    "training_collision_penalty": ("training_collision_penalty", "first"),
    "training_projection_enabled": ("training_projection_enabled", "first"),
}
SUMMARY_METRICS = (
    "episode_return",
    "episode_length",
    "success_rate",
    "collision_rate",
    "min_obstacle_clearance",
    "action_bound_clipping_rate",
    "speed_action_bound_clipping_rate",
    "turn_rate_action_bound_clipping_rate",
    "action_bound_clipping_norm",
    "action_bound_clipping_norm_max",
    "projection_intervention_rate",
    "projection_correction_norm",
    "projection_correction_norm_max",
    "projection_slack_sum",
    "projection_slack_max",
)
PROJECTION_RAW_METRICS = (
    "projection_intervention_rate",
    "mean_projection_correction_norm",
    "max_projection_correction_norm",
    "mean_projection_slack_sum",
    "max_projection_slack",
)
PROJECTION_EPISODE_METRICS = (
    "projection_intervention_count",
    "projection_intervention_rate",
    "mean_projection_correction_norm",
    "max_projection_correction_norm",
    "mean_projection_slack_sum",
    "max_projection_slack",
    "projection_solver_failure_count",
)
PROJECTION_SUMMARY_METRICS = (
    "projection_intervention_rate",
    "projection_correction_norm",
    "projection_correction_norm_max",
    "projection_slack_sum",
    "projection_slack_max",
)
PAIRED_METRICS = (
    "episode_return",
    "episode_length",
    "success",
    "collision",
    "min_obstacle_clearance",
    "action_bound_clipping_rate",
    "speed_action_bound_clipping_rate",
    "turn_rate_action_bound_clipping_rate",
    "mean_action_bound_clipping_norm",
    "max_action_bound_clipping_norm",
)


class EvidenceError(RuntimeError):
    """Raised when an explicitly supplied evidence file fails validation."""


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    modes: tuple[str, ...]
    training_collision_penalty: float | None
    training_projection_enabled: bool | None


@dataclass(frozen=True)
class Design:
    protocol: dict[str, Any]
    layout_suite: dict[str, Any]
    methods: tuple[MethodSpec, ...]
    seeds: tuple[int, ...]
    layout_ids: tuple[str, ...]
    obstacle_free_layout_ids: tuple[str, ...]
    repeats_per_layout: int
    protocol_canonical_sha256: str
    layout_canonical_sha256: str

    @property
    def method_map(self) -> dict[str, MethodSpec]:
        return {method.name: method for method in self.methods}

    @property
    def expected_checkpoint_rows(self) -> int:
        return len(self.seeds) * sum(len(method.modes) for method in self.methods)

    @property
    def expected_method_rows(self) -> int:
        return sum(len(method.modes) for method in self.methods)

    @property
    def expected_episode_rows(self) -> int:
        return (
            self.expected_checkpoint_rows
            * len(self.layout_ids)
            * self.repeats_per_layout
        )

    @property
    def paired_methods(self) -> tuple[MethodSpec, ...]:
        return tuple(
            method
            for method in self.methods
            if {"disabled", "enabled"}.issubset(method.modes)
        )

    @property
    def expected_paired_rows(self) -> int:
        return len(self.paired_methods) * len(self.seeds)

    @property
    def expected_paired_summary_rows(self) -> int:
        return len(self.paired_methods)

    def record(self) -> dict[str, object]:
        return {
            "expected_checkpoint_rows": self.expected_checkpoint_rows,
            "expected_episode_rows": self.expected_episode_rows,
            "expected_method_rows": self.expected_method_rows,
            "expected_paired_checkpoint_rows": self.expected_paired_rows,
            "expected_paired_method_rows": self.expected_paired_summary_rows,
            "expected_train_seeds": list(self.seeds),
            "layout_count": len(self.layout_ids),
            "layout_ids": list(self.layout_ids),
            "methods": [method.name for method in self.methods],
            "projection_modes_by_method": {
                method.name: list(method.modes) for method in self.methods
            },
            "repeats_per_layout": self.repeats_per_layout,
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    try:
        return hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise EvidenceError(f"Cannot read input file {str(path)!r}: {error}") from error


def canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"Value cannot be represented as canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def input_record(path: str) -> dict[str, str]:
    """Record exactly the supplied path string; never expand it to a machine path."""

    return {"path": path, "sha256": sha256_file(path)}


def input_records(paths: Mapping[str, str]) -> dict[str, dict[str, str]]:
    return {role: input_record(paths[role]) for role in sorted(paths)}


def implementation_record(entrypoint: str | Path) -> dict[str, str]:
    entrypoint_path = Path(entrypoint)
    helper_path = Path(__file__)
    return {
        "entrypoint": entrypoint_path.name,
        "entrypoint_sha256": sha256_file(entrypoint_path),
        "evidence_module": helper_path.name,
        "evidence_module_sha256": sha256_file(helper_path),
    }


def read_json_object(path: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(
                key not in result,
                f"JSON input contains duplicate object key {key!r}: {path!r}",
            )
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> None:
        raise EvidenceError(
            f"JSON input contains non-standard numeric constant {value!r}: {path!r}"
        )

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_nonstandard_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"Cannot read JSON object {path!r}: {error}") from error
    require(isinstance(value, dict), f"JSON input must contain an object: {path!r}")
    return value


def read_csv(path: str) -> pd.DataFrame:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream, strict=True), None)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise EvidenceError(f"Cannot read CSV input {path!r}: {error}") from error
    require(header is not None, f"CSV input is empty: {path!r}")
    require(
        len(set(header)) == len(header),
        f"CSV input has duplicate column names: {path!r}",
    )
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise EvidenceError(f"Cannot read CSV input {path!r}: {error}") from error
    require(not frame.empty, f"CSV input is empty: {path!r}")
    return frame


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    require(not missing, f"{label} is missing required columns: {missing}")


def strict_bool_series(series: pd.Series, label: str) -> pd.Series:
    parsed: list[bool] = []
    for row_index, value in series.items():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            require(int(value) in (0, 1), f"{label}[{row_index!r}] is not Boolean: {value!r}")
            parsed.append(bool(value))
        elif isinstance(value, str):
            normalized = value.strip().lower()
            require(
                normalized in {"true", "false", "1", "0"},
                f"{label}[{row_index!r}] is not Boolean: {value!r}",
            )
            parsed.append(normalized in {"true", "1"})
        else:
            raise EvidenceError(f"{label}[{row_index!r}] is not Boolean: {value!r}")
    return pd.Series(parsed, index=series.index, dtype=bool, name=series.name)


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    label: str,
    *,
    allow_nan: bool = False,
    integer: bool = False,
) -> pd.Series:
    require(column in frame.columns, f"{label} is missing required column {column!r}.")
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label}.{column} contains malformed numeric values.") from error
    array = values.to_numpy(dtype=float)
    require(not bool(np.isinf(array).any()), f"{label}.{column} contains an infinity.")
    if not allow_nan:
        require(not bool(np.isnan(array).any()), f"{label}.{column} contains a missing value.")
    if integer:
        finite = array[np.isfinite(array)]
        require(
            bool(np.equal(finite, np.round(finite)).all()),
            f"{label}.{column} must contain integers.",
        )
    return values


def _required_text(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    require(column in frame.columns, f"{label} is missing required column {column!r}.")
    require(not bool(frame[column].isna().any()), f"{label}.{column} contains a missing value.")
    values = frame[column].astype(str)
    require(bool(values.str.strip().ne("").all()), f"{label}.{column} contains an empty value.")
    return values


def load_episode_table(path: str) -> pd.DataFrame:
    frame = read_csv(path)
    require_columns(frame, REQUIRED_EPISODE_COLUMNS, "episode table")
    result = frame.copy()
    for column in BOOLEAN_COLUMNS:
        result[column] = strict_bool_series(result[column], f"episode table.{column}")
    for column in EPISODE_NUMERIC_COLUMNS:
        result[column] = numeric_series(
            result,
            column,
            "episode table",
            allow_nan=column == "min_obstacle_clearance",
            integer=column in INTEGER_EPISODE_COLUMNS,
        )
        if column in INTEGER_EPISODE_COLUMNS:
            result[column] = result[column].astype(int)
    for column in (
        "method",
        "checkpoint",
        "checkpoint_sha256",
        "layout_suite_schema_version",
        "layout_suite_id",
        "layout_suite_sha256",
        "layout_id",
        "evaluation_policy_mode",
        "projection_mode",
        "result_build_schema_version",
    ):
        result[column] = _required_text(result, column, "episode table")
    return result


def load_design(protocol_path: str, layout_suite_path: str) -> Design:
    protocol = read_json_object(protocol_path)
    layout_suite = read_json_object(layout_suite_path)
    require(
        protocol.get("schema_version") == PROTOCOL_SCHEMA,
        f"protocol.schema_version must equal {PROTOCOL_SCHEMA!r}.",
    )

    raw_seeds = protocol.get("expected_train_seeds")
    require(isinstance(raw_seeds, list) and bool(raw_seeds), "Protocol seeds must be nonempty.")
    require(
        all(type(seed) is int for seed in raw_seeds)
        and len(set(raw_seeds)) == len(raw_seeds),
        "Protocol seeds must be unique integers.",
    )

    raw_methods = protocol.get("methods")
    require(isinstance(raw_methods, list) and bool(raw_methods), "Protocol methods must be nonempty.")
    methods: list[MethodSpec] = []
    names: set[str] = set()
    for record in raw_methods:
        require(isinstance(record, dict), "Every protocol method must be an object.")
        name = str(record.get("method", "")).strip()
        display_name = str(record.get("display_name", "")).strip()
        require(name and name not in names, "Protocol method labels must be unique and nonempty.")
        require(bool(display_name), f"Protocol display_name is missing for method {name!r}.")
        raw_modes = record.get("required_projection_modes")
        require(isinstance(raw_modes, list) and bool(raw_modes), f"Protocol modes are missing for {name!r}.")
        modes = tuple(str(mode) for mode in raw_modes)
        require(
            len(set(modes)) == len(modes)
            and all(mode in PROJECTION_MODES for mode in modes),
            f"Protocol modes are invalid for {name!r}.",
        )
        penalty: float | None = None
        if "training_collision_penalty" in record:
            raw_penalty = record["training_collision_penalty"]
            require(
                isinstance(raw_penalty, (int, float)) and not isinstance(raw_penalty, bool),
                f"training_collision_penalty must be numeric for {name!r}.",
            )
            penalty = float(raw_penalty)
            require(math.isfinite(penalty), f"training_collision_penalty must be finite for {name!r}.")
        training_projection: bool | None = None
        if "training_projection_enabled" in record:
            require(
                type(record["training_projection_enabled"]) is bool,
                f"training_projection_enabled must be Boolean for {name!r}.",
            )
            training_projection = bool(record["training_projection_enabled"])
        methods.append(
            MethodSpec(
                name=name,
                display_name=display_name,
                modes=modes,
                training_collision_penalty=penalty,
                training_projection_enabled=training_projection,
            )
        )
        names.add(name)

    repeats = protocol.get("expected_repeats_per_layout")
    require(type(repeats) is int and repeats > 0, "Protocol repeats per layout must be positive.")
    raw_layouts = layout_suite.get("layouts")
    require(isinstance(raw_layouts, list) and bool(raw_layouts), "Layout suite must be nonempty.")
    layout_ids: list[str] = []
    obstacle_free: list[str] = []
    for layout in raw_layouts:
        require(isinstance(layout, dict), "Every layout must be an object.")
        layout_id = str(layout.get("layout_id", "")).strip()
        require(
            bool(layout_id) and layout_id not in layout_ids,
            "Layout IDs must be unique and nonempty.",
        )
        obstacles = layout.get("obstacles", [])
        require(isinstance(obstacles, list), f"Layout obstacles must be an array for {layout_id!r}.")
        layout_ids.append(layout_id)
        if not obstacles:
            obstacle_free.append(layout_id)

    for field in ("suite_id", "schema_version"):
        require(bool(str(layout_suite.get(field, "")).strip()), f"Layout suite {field} is missing.")
    return Design(
        protocol=protocol,
        layout_suite=layout_suite,
        methods=tuple(methods),
        seeds=tuple(int(seed) for seed in raw_seeds),
        layout_ids=tuple(layout_ids),
        obstacle_free_layout_ids=tuple(obstacle_free),
        repeats_per_layout=int(repeats),
        protocol_canonical_sha256=canonical_json_sha256(protocol),
        layout_canonical_sha256=canonical_json_sha256(layout_suite),
    )


def _require_numeric_constant(
    frame: pd.DataFrame,
    column: str,
    expected: object,
    label: str,
) -> None:
    require(
        isinstance(expected, (int, float)) and not isinstance(expected, bool),
        f"{label} expected value is not numeric.",
    )
    values = numeric_series(frame, column, "episode table").to_numpy(dtype=float)
    require(
        bool(np.allclose(values, float(expected), atol=1.0e-12, rtol=0.0)),
        f"Episode rows disagree with protocol/layout value for {label}.",
    )


def checkpoint_identity_map(frame: pd.DataFrame) -> dict[tuple[str, int], tuple[str, str]]:
    require_columns(
        frame,
        {"method", "train_seed", "checkpoint", "checkpoint_sha256", "projection_mode"},
        "episode table",
    )
    result: dict[tuple[str, int], tuple[str, str]] = {}
    for (method, train_seed), group in frame.groupby(["method", "train_seed"], sort=True):
        identities = set(
            zip(group["checkpoint"].astype(str), group["checkpoint_sha256"].astype(str))
        )
        require(
            len(identities) == 1,
            f"Checkpoint identity is ambiguous for method {method!r}, seed {int(train_seed)}.",
        )
        result[(str(method), int(train_seed))] = next(iter(identities))
    hashes = [identity[1] for identity in result.values()]
    require(
        len(set(hashes)) == len(hashes),
        "Distinct method/seed units reuse a checkpoint SHA-256.",
    )
    return result


def validate_episode_table(design: Design, episodes: pd.DataFrame) -> dict[str, object]:
    """Validate complete protocol coverage, pairing, metrics, and structural missingness."""

    require(len(episodes) == design.expected_episode_rows, "Episode row count differs from protocol-derived coverage.")
    method_map = design.method_map
    require(set(episodes["method"].astype(str)) == set(method_map), "Episode methods differ from the protocol.")
    require(set(episodes["train_seed"].astype(int)) == set(design.seeds), "Episode training seeds differ from the protocol.")
    require(
        set(episodes["layout_id"].astype(str)) == set(design.layout_ids),
        "Episode layout IDs differ from the supplied layout suite.",
    )
    require(
        set(episodes["projection_mode"].astype(str)).issubset(PROJECTION_MODES),
        "Episode projection_mode contains an unsupported value.",
    )
    require(
        bool((episodes["projection_enabled"] == episodes["projection_mode"].eq("enabled")).all()),
        "projection_enabled disagrees with projection_mode.",
    )

    expected_string_constants = {
        "layout_suite_schema_version": design.layout_suite["schema_version"],
        "layout_suite_id": design.layout_suite["suite_id"],
        "layout_suite_sha256": design.layout_canonical_sha256,
        "evaluation_policy_mode": design.protocol.get("evaluation_policy_mode"),
    }
    for column, expected in expected_string_constants.items():
        require(expected is not None, f"Protocol/layout value required for {column!r} is missing.")
        require(
            set(episodes[column].astype(str)) == {str(expected)},
            f"Episode rows disagree with protocol/layout value for {column}.",
        )

    projection = design.protocol.get("projection_parameters")
    require(isinstance(projection, dict), "Protocol projection_parameters must be an object.")
    numeric_constants = {
        "evaluation_collision_penalty": design.protocol.get("evaluation_collision_penalty"),
        "max_episode_steps": design.protocol.get("max_episode_steps"),
        "layout_max_obstacles": design.layout_suite.get("max_obstacles"),
        "layout_agent_radius": design.layout_suite.get("agent_radius"),
        "layout_goal_radius": design.layout_suite.get("goal_radius"),
        "projection_lookahead_distance": projection.get("lookahead_distance"),
        "projection_alpha": projection.get("alpha"),
        "projection_slack_penalty": projection.get("slack_penalty"),
        "projection_extra_clearance": projection.get("extra_clearance"),
    }
    for column, expected in numeric_constants.items():
        _require_numeric_constant(episodes, column, expected, column)

    schema_values = set(episodes["result_build_schema_version"].astype(str))
    require(len(schema_values) == 1, "Episode rows contain multiple result-build schemas.")

    duplicate_key = [
        *CHECKPOINT_KEYS,
        "layout_id",
        "layout_repeat",
        "evaluation_seed",
        "episode",
    ]
    require(
        not bool(episodes.duplicated(duplicate_key, keep=False).any()),
        "Episode table contains duplicate checkpoint-identified evaluation keys.",
    )

    expected_group_labels: set[tuple[str, int, str]] = set()
    evaluation_rows_per_group = len(design.layout_ids) * design.repeats_per_layout
    base_seed = design.protocol.get("evaluation_base_seed")
    last_seed = design.protocol.get("evaluation_last_seed")
    require(
        type(base_seed) is int and type(last_seed) is int,
        "Protocol evaluation seed bounds must be integers.",
    )
    expected_evaluation_seeds = set(range(base_seed, last_seed + 1))
    require(
        len(expected_evaluation_seeds) == evaluation_rows_per_group,
        "Protocol evaluation seed bounds disagree with layout/repeat coverage.",
    )
    expected_evaluation_mapping: set[tuple[str, int, int, int, int]] = set()
    episode_index = 0
    for layout_id in design.layout_ids:
        for layout_repeat in range(design.repeats_per_layout):
            evaluation_seed = base_seed + episode_index
            expected_evaluation_mapping.add(
                (
                    layout_id,
                    layout_repeat,
                    evaluation_seed,
                    episode_index,
                    evaluation_seed,
                )
            )
            episode_index += 1
    require(
        bool(
            (
                episodes["seed"].to_numpy(dtype=int)
                == episodes["evaluation_seed"].to_numpy(dtype=int)
            ).all()
        ),
        "Episode seed must equal evaluation_seed on every row.",
    )
    for method in design.methods:
        method_rows = episodes[episodes["method"].eq(method.name)]
        require(
            set(method_rows["projection_mode"].astype(str)) == set(method.modes),
            f"Projection modes differ from the protocol for method {method.name!r}.",
        )
        if method.training_collision_penalty is not None:
            values = method_rows["training_collision_penalty"].to_numpy(dtype=float)
            require(
                bool(np.allclose(values, method.training_collision_penalty, atol=1.0e-12, rtol=0.0)),
                f"Training collision penalty differs for method {method.name!r}.",
            )
        if method.training_projection_enabled is not None:
            require(
                bool((method_rows["training_projection_enabled"] == method.training_projection_enabled).all()),
                f"Training projection flag differs for method {method.name!r}.",
            )
        for seed in design.seeds:
            identities: set[tuple[str, str]] = set()
            for mode in method.modes:
                expected_group_labels.add((method.name, seed, mode))
                group = episodes[
                    episodes["method"].eq(method.name)
                    & episodes["train_seed"].eq(seed)
                    & episodes["projection_mode"].eq(mode)
                ]
                require(
                    len(group) == evaluation_rows_per_group,
                    f"Unexpected coverage for {method.name!r}, seed {seed}, mode {mode!r}.",
                )
                require(
                    set(group["layout_id"].astype(str)) == set(design.layout_ids),
                    f"Incomplete layout coverage for {method.name!r}, seed {seed}, mode {mode!r}.",
                )
                for layout_id, layout_rows in group.groupby("layout_id", sort=True):
                    require(
                        set(layout_rows["layout_repeat"].astype(int))
                        == set(range(design.repeats_per_layout)),
                        f"Incomplete repeats for {method.name!r}, seed {seed}, mode {mode!r}, layout {layout_id!r}.",
                    )
                group_identities = set(
                    zip(group["checkpoint"].astype(str), group["checkpoint_sha256"].astype(str))
                )
                require(
                    len(group_identities) == 1,
                    f"Mixed checkpoint identities for {method.name!r}, seed {seed}, mode {mode!r}.",
                )
                identities.update(group_identities)
                keys = set(
                    zip(
                        group["layout_id"].astype(str),
                        group["layout_repeat"].astype(int),
                        group["evaluation_seed"].astype(int),
                        group["episode"].astype(int),
                        group["seed"].astype(int),
                    )
                )
                require(
                    keys == expected_evaluation_mapping,
                    "Evaluation mapping differs from protocol/layout order, repeats, seeds, or episode indices.",
                )
            require(
                len(identities) == 1,
                f"Projection modes use different checkpoints for {method.name!r}, seed {seed}.",
            )

    actual_group_labels = set(
        zip(
            episodes["method"].astype(str),
            episodes["train_seed"].astype(int),
            episodes["projection_mode"].astype(str),
        )
    )
    require(actual_group_labels == expected_group_labels, "Unexpected method/seed/mode groups are present.")
    identities = checkpoint_identity_map(episodes)
    require(
        len(identities) == len(design.methods) * len(design.seeds),
        "The method/seed checkpoint map is incomplete.",
    )
    for _, checkpoint_sha in identities.values():
        require(
            re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha) is not None,
            f"Invalid checkpoint SHA-256 value: {checkpoint_sha!r}",
        )

    require(
        set(episodes["evaluation_seed"].astype(int)) == expected_evaluation_seeds,
        "Episode evaluation seeds differ from the protocol bounds.",
    )

    lengths = episodes["episode_length"].to_numpy(dtype=float)
    require(bool((lengths > 0).all()), "episode_length must be positive.")
    require(
        bool((lengths <= episodes["max_episode_steps"].to_numpy(dtype=float)).all()),
        "episode_length exceeds max_episode_steps.",
    )
    success = episodes["success"].to_numpy(dtype=bool)
    collision = episodes["collision"].to_numpy(dtype=bool)
    terminated = episodes["terminated"].to_numpy(dtype=bool)
    truncated = episodes["truncated"].to_numpy(dtype=bool)
    outcome_sum = success.astype(int) + collision.astype(int) + truncated.astype(int)
    require(bool((outcome_sum == 1).all()), "success + collision + timeout must equal one on every row.")
    require(
        bool(np.array_equal(terminated, np.logical_or(success, collision))),
        "terminated must equal success OR collision on every row.",
    )

    count_rate_pairs = (
        ("action_bound_clipping_count", "action_bound_clipping_rate"),
        ("speed_action_bound_clipping_count", "speed_action_bound_clipping_rate"),
        ("turn_rate_action_bound_clipping_count", "turn_rate_action_bound_clipping_rate"),
        ("projection_intervention_count", "projection_intervention_rate"),
    )
    rate_errors: dict[str, float] = {}
    for count_column, rate_column in count_rate_pairs:
        counts = episodes[count_column].to_numpy(dtype=float)
        rates = episodes[rate_column].to_numpy(dtype=float)
        require(bool((counts >= 0).all()) and bool((counts <= lengths).all()), f"{count_column} is outside [0, episode_length].")
        require(bool((rates >= 0).all()) and bool((rates <= 1).all()), f"{rate_column} is outside [0, 1].")
        error = float(np.max(np.abs(rates - counts / lengths)))
        require(error <= 1.0e-12, f"{rate_column} does not equal {count_column}/episode_length.")
        rate_errors[rate_column] = error

    nonnegative_metrics = (
        "final_distance_to_goal",
        "mean_action_bound_clipping_norm",
        "max_action_bound_clipping_norm",
        *PROJECTION_EPISODE_METRICS,
    )
    for column in nonnegative_metrics:
        require(
            bool((episodes[column].to_numpy(dtype=float) >= 0).all()),
            f"{column} must be nonnegative.",
        )
    require(
        bool(
            (
                episodes["mean_action_bound_clipping_norm"].to_numpy(dtype=float)
                <= episodes["max_action_bound_clipping_norm"].to_numpy(dtype=float) + 1.0e-12
            ).all()
        ),
        "Mean clipping norm exceeds the episode maximum.",
    )
    require(
        bool(
            (
                episodes["mean_projection_correction_norm"].to_numpy(dtype=float)
                <= episodes["max_projection_correction_norm"].to_numpy(dtype=float) + 1.0e-12
            ).all()
        ),
        "Mean projection-correction norm exceeds the episode maximum.",
    )

    disabled = episodes["projection_mode"].eq("disabled")
    structural_zero_counts: dict[str, int] = {}
    for column in PROJECTION_EPISODE_METRICS:
        values = episodes.loc[disabled, column].to_numpy(dtype=float)
        require(bool((values == 0.0).all()), f"Disabled {column} must be structural zero.")
        structural_zero_counts[column] = int(len(values))

    expected_clearance_nan = episodes["layout_id"].isin(design.obstacle_free_layout_ids).to_numpy()
    actual_clearance_nan = episodes["min_obstacle_clearance"].isna().to_numpy()
    require(
        bool(np.array_equal(actual_clearance_nan, expected_clearance_nan)),
        "min_obstacle_clearance NaNs do not exactly match obstacle-free layouts.",
    )
    return {
        "checkpoint_identity_count": len(identities),
        "clearance_nan_pattern_exact": True,
        "evaluation_mapping_exact": True,
        "outcome_partition_exact": True,
        "pairing_keys_exact": True,
        "rate_formula_max_abs_errors": rate_errors,
        "result_build_schema_version": next(iter(schema_values)),
        "seed_identity_exact": True,
        "structural_zero_counts_when_projection_disabled": structural_zero_counts,
    }


def reconstruct_checkpoint_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = episodes.copy()
    disabled = frame["projection_mode"].eq("disabled")
    for column in PROJECTION_RAW_METRICS:
        frame.loc[disabled, column] = np.nan
    return (
        frame.groupby(list(CHECKPOINT_KEYS), sort=True, dropna=False)
        .agg(**CHECKPOINT_AGGREGATIONS)
        .reset_index()
    )


def reconstruct_method_summary(design: Design, checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in design.methods:
        for mode in method.modes:
            group = checkpoints[
                checkpoints["method"].eq(method.name)
                & checkpoints["projection_mode"].eq(mode)
            ]
            require(len(group) == len(design.seeds), f"Checkpoint count differs for {method.name!r}, {mode!r}.")
            row: dict[str, object] = {
                "method": method.name,
                "projection_mode": mode,
                "seed_count": int(group["train_seed"].nunique()),
            }
            for metric in SUMMARY_METRICS:
                values = group[metric].astype(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def reconstruct_paired_deltas(design: Design, episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pair_columns = ["layout_id", "layout_repeat", "evaluation_seed"]
    for method in design.paired_methods:
        for seed in design.seeds:
            group = episodes[episodes["method"].eq(method.name) & episodes["train_seed"].eq(seed)]
            disabled = group[group["projection_mode"].eq("disabled")]
            enabled = group[group["projection_mode"].eq("enabled")]
            try:
                merged = disabled[pair_columns + ["checkpoint_sha256", *PAIRED_METRICS]].merge(
                    enabled[pair_columns + ["checkpoint_sha256", *PAIRED_METRICS]],
                    on=pair_columns,
                    suffixes=("_disabled", "_enabled"),
                    validate="one_to_one",
                )
            except pd.errors.MergeError as error:
                raise EvidenceError(f"Cannot pair projection modes for {method.name!r}, seed {seed}.") from error
            require(len(merged) == len(disabled) == len(enabled), f"Incomplete projection pairing for {method.name!r}, seed {seed}.")
            require(
                bool((merged["checkpoint_sha256_disabled"] == merged["checkpoint_sha256_enabled"]).all()),
                f"Projection modes use different checkpoint hashes for {method.name!r}, seed {seed}.",
            )
            row: dict[str, object] = {
                "method": method.name,
                "train_seed": seed,
                "checkpoint_sha256": str(merged["checkpoint_sha256_disabled"].iloc[0]),
                "paired_layout_count": int(len(merged)),
            }
            for metric in PAIRED_METRICS:
                delta = (
                    merged[f"{metric}_enabled"].astype(float)
                    - merged[f"{metric}_disabled"].astype(float)
                )
                row[f"{metric}_delta_enabled_minus_disabled"] = float(delta.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def reconstruct_paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame(columns=["method", "seed_count"])
    delta_columns = [column for column in paired.columns if column.endswith("_delta_enabled_minus_disabled")]
    rows: list[dict[str, object]] = []
    for method, group in paired.groupby("method", sort=True):
        row: dict[str, object] = {
            "method": str(method),
            "seed_count": int(group["train_seed"].nunique()),
        }
        for column in delta_columns:
            values = group[column].astype(float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_numeric_exact_nan(
    actual: pd.Series,
    expected: pd.Series,
    label: str,
    *,
    atol: float = 1.0e-12,
) -> float:
    try:
        actual_values = pd.to_numeric(actual, errors="raise").to_numpy(dtype=float)
        expected_values = pd.to_numeric(expected, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"{label} contains malformed numeric values.") from error
    require(actual_values.shape == expected_values.shape, f"{label} shapes differ.")
    actual_nan = np.isnan(actual_values)
    expected_nan = np.isnan(expected_values)
    require(bool(np.array_equal(actual_nan, expected_nan)), f"{label} NaN patterns differ.")
    finite = ~actual_nan
    require(
        bool(np.isfinite(actual_values[finite]).all())
        and bool(np.isfinite(expected_values[finite]).all()),
        f"{label} contains a nonfinite, non-NaN value.",
    )
    if not bool(finite.any()):
        return 0.0
    error = float(np.max(np.abs(actual_values[finite] - expected_values[finite])))
    require(error <= atol, f"{label} maximum absolute error {error} exceeds {atol}.")
    return error


def _merge_one_to_one(
    reconstructed: pd.DataFrame,
    committed: pd.DataFrame,
    keys: Sequence[str],
    label: str,
) -> pd.DataFrame:
    require_columns(reconstructed, keys, f"reconstructed {label}")
    require_columns(committed, keys, f"committed {label}")
    require(not bool(reconstructed.duplicated(list(keys)).any()), f"Reconstructed {label} keys are duplicated.")
    require(not bool(committed.duplicated(list(keys)).any()), f"Committed {label} keys are duplicated.")
    reconstructed_keys = set(map(tuple, reconstructed[list(keys)].itertuples(index=False, name=None)))
    committed_keys = set(map(tuple, committed[list(keys)].itertuples(index=False, name=None)))
    require(reconstructed_keys == committed_keys, f"Reconstructed and committed {label} keys differ.")
    try:
        return reconstructed.merge(
            committed,
            on=list(keys),
            how="inner",
            validate="one_to_one",
            suffixes=("_reconstructed", "_committed"),
        )
    except pd.errors.MergeError as error:
        raise EvidenceError(f"Cannot reconcile {label} one-to-one.") from error


def _validate_table_schema(frame: pd.DataFrame, schema: str, label: str) -> None:
    require_columns(frame, {"result_build_schema_version"}, label)
    require(
        set(_required_text(frame, "result_build_schema_version", label)) == {schema},
        f"{label} result-build schema differs from the episode table.",
    )


def reconcile_checkpoint_and_method_tables(
    design: Design,
    episodes: pd.DataFrame,
    checkpoint_path: str,
    method_path: str,
    schema: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    reconstructed_checkpoints = reconstruct_checkpoint_summary(episodes)
    reconstructed_methods = reconstruct_method_summary(design, reconstructed_checkpoints)
    committed_checkpoints = read_csv(checkpoint_path)
    committed_methods = read_csv(method_path)
    _validate_table_schema(committed_checkpoints, schema, "checkpoint summary")
    _validate_table_schema(committed_methods, schema, "method summary")
    require(len(committed_checkpoints) == design.expected_checkpoint_rows, "Checkpoint-summary row count differs from protocol-derived coverage.")
    require(len(committed_methods) == design.expected_method_rows, "Method-summary row count differs from protocol-derived coverage.")

    committed_checkpoints["train_seed"] = numeric_series(
        committed_checkpoints, "train_seed", "checkpoint summary", integer=True
    ).astype(int)
    for column in ("episode_count", "layout_count", *SUMMARY_METRICS, "training_collision_penalty"):
        committed_checkpoints[column] = numeric_series(
            committed_checkpoints,
            column,
            "checkpoint summary",
            allow_nan=column in PROJECTION_SUMMARY_METRICS,
            integer=column in {"episode_count", "layout_count"},
        )
    committed_checkpoints["training_projection_enabled"] = strict_bool_series(
        committed_checkpoints["training_projection_enabled"],
        "checkpoint summary.training_projection_enabled",
    )
    for column in ("method", "display_name", "checkpoint", "checkpoint_sha256", "projection_mode"):
        committed_checkpoints[column] = _required_text(committed_checkpoints, column, "checkpoint summary")

    checkpoint_merged = _merge_one_to_one(
        reconstructed_checkpoints,
        committed_checkpoints,
        CHECKPOINT_KEYS,
        "checkpoint summary",
    )
    checkpoint_errors: dict[str, float] = {}
    for column in ("episode_count", "layout_count", *SUMMARY_METRICS, "training_collision_penalty"):
        checkpoint_errors[column] = compare_numeric_exact_nan(
            checkpoint_merged[f"{column}_reconstructed"],
            checkpoint_merged[f"{column}_committed"],
            f"checkpoint summary {column}",
        )
    require(
        bool(
            (
                checkpoint_merged["training_projection_enabled_reconstructed"]
                == checkpoint_merged["training_projection_enabled_committed"]
            ).all()
        ),
        "Checkpoint-summary training projection flags differ from reconstruction.",
    )
    display_map = {method.name: method.display_name for method in design.methods}
    require(
        bool(
            checkpoint_merged["display_name"].eq(
                checkpoint_merged["method"].map(display_map)
            ).all()
        ),
        "Checkpoint-summary display names differ from the protocol.",
    )

    committed_methods["seed_count"] = numeric_series(
        committed_methods, "seed_count", "method summary", integer=True
    ).astype(int)
    for metric in SUMMARY_METRICS:
        for suffix in ("mean", "std"):
            column = f"{metric}_{suffix}"
            committed_methods[column] = numeric_series(
                committed_methods,
                column,
                "method summary",
                allow_nan=metric in PROJECTION_SUMMARY_METRICS,
            )
    for column in ("method", "display_name", "projection_mode"):
        committed_methods[column] = _required_text(committed_methods, column, "method summary")
    method_merged = _merge_one_to_one(
        reconstructed_methods,
        committed_methods,
        ("method", "projection_mode"),
        "method summary",
    )
    method_errors: dict[str, float] = {
        "seed_count": compare_numeric_exact_nan(
            method_merged["seed_count_reconstructed"],
            method_merged["seed_count_committed"],
            "method summary seed_count",
        )
    }
    for metric in SUMMARY_METRICS:
        for suffix in ("mean", "std"):
            column = f"{metric}_{suffix}"
            method_errors[column] = compare_numeric_exact_nan(
                method_merged[f"{column}_reconstructed"],
                method_merged[f"{column}_committed"],
                f"method summary {column}",
            )
    require(
        bool(method_merged["display_name"].eq(method_merged["method"].map(display_map)).all()),
        "Method-summary display names differ from the protocol.",
    )
    return reconstructed_checkpoints, reconstructed_methods, {
        "checkpoint_max_abs_errors": checkpoint_errors,
        "method_max_abs_errors": method_errors,
    }


def reconcile_paired_tables(
    design: Design,
    episodes: pd.DataFrame,
    paired_path: str,
    paired_summary_path: str,
    schema: str,
) -> dict[str, object]:
    reconstructed_paired = reconstruct_paired_deltas(design, episodes)
    reconstructed_summary = reconstruct_paired_summary(reconstructed_paired)
    committed_paired = read_csv(paired_path)
    committed_summary = read_csv(paired_summary_path)
    _validate_table_schema(committed_paired, schema, "paired-delta table")
    _validate_table_schema(committed_summary, schema, "paired-summary table")
    require(len(committed_paired) == design.expected_paired_rows, "Paired-delta row count differs from protocol-derived coverage.")
    require(len(committed_summary) == design.expected_paired_summary_rows, "Paired-summary row count differs from protocol-derived coverage.")

    committed_paired["train_seed"] = numeric_series(
        committed_paired, "train_seed", "paired-delta table", integer=True
    ).astype(int)
    committed_paired["paired_layout_count"] = numeric_series(
        committed_paired, "paired_layout_count", "paired-delta table", integer=True
    ).astype(int)
    delta_columns = [f"{metric}_delta_enabled_minus_disabled" for metric in PAIRED_METRICS]
    for column in delta_columns:
        committed_paired[column] = numeric_series(
            committed_paired,
            column,
            "paired-delta table",
            allow_nan=column.startswith("min_obstacle_clearance"),
        )
    for column in ("method", "display_name", "checkpoint_sha256"):
        committed_paired[column] = _required_text(committed_paired, column, "paired-delta table")
    paired_merged = _merge_one_to_one(
        reconstructed_paired,
        committed_paired,
        ("method", "train_seed", "checkpoint_sha256"),
        "paired-delta table",
    )
    paired_errors: dict[str, float] = {
        "paired_layout_count": compare_numeric_exact_nan(
            paired_merged["paired_layout_count_reconstructed"],
            paired_merged["paired_layout_count_committed"],
            "paired-delta paired_layout_count",
        )
    }
    for column in delta_columns:
        paired_errors[column] = compare_numeric_exact_nan(
            paired_merged[f"{column}_reconstructed"],
            paired_merged[f"{column}_committed"],
            f"paired-delta {column}",
        )

    committed_summary["seed_count"] = numeric_series(
        committed_summary, "seed_count", "paired-summary table", integer=True
    ).astype(int)
    summary_columns = [
        f"{column}_{suffix}"
        for column in delta_columns
        for suffix in ("mean", "std")
    ]
    for column in summary_columns:
        committed_summary[column] = numeric_series(
            committed_summary,
            column,
            "paired-summary table",
            allow_nan=column.startswith("min_obstacle_clearance"),
        )
    for column in ("method", "display_name"):
        committed_summary[column] = _required_text(committed_summary, column, "paired-summary table")
    summary_merged = _merge_one_to_one(
        reconstructed_summary,
        committed_summary,
        ("method",),
        "paired-summary table",
    )
    summary_errors: dict[str, float] = {
        "seed_count": compare_numeric_exact_nan(
            summary_merged["seed_count_reconstructed"],
            summary_merged["seed_count_committed"],
            "paired-summary seed_count",
        )
    }
    for column in summary_columns:
        summary_errors[column] = compare_numeric_exact_nan(
            summary_merged[f"{column}_reconstructed"],
            summary_merged[f"{column}_committed"],
            f"paired-summary {column}",
        )
    display_map = {method.name: method.display_name for method in design.methods}
    require(
        bool(paired_merged["display_name"].eq(paired_merged["method"].map(display_map)).all()),
        "Paired-delta display names differ from the protocol.",
    )
    require(
        bool(summary_merged["display_name"].eq(summary_merged["method"].map(display_map)).all()),
        "Paired-summary display names differ from the protocol.",
    )
    return {
        "paired_delta_max_abs_errors": paired_errors,
        "paired_summary_max_abs_errors": summary_errors,
    }


def validate_build_audit(
    design: Design,
    audit_path: str,
    schema: str,
    episodes: pd.DataFrame,
) -> dict[str, object]:
    audit = read_json_object(audit_path)
    require(audit.get("status") == "PASS", "Result-build audit status is not PASS.")
    require(
        audit.get("result_build_schema_version") == schema,
        "Result-build audit schema differs from the episode table.",
    )
    expected_counts = {
        "episode_row_count": design.expected_episode_rows,
        "checkpoint_row_count": design.expected_checkpoint_rows,
        "method_row_count": design.expected_method_rows,
        "paired_checkpoint_row_count": design.expected_paired_rows,
        "paired_method_row_count": design.expected_paired_summary_rows,
    }
    for field, expected in expected_counts.items():
        require(audit.get(field) == expected, f"Result-build audit {field} differs from protocol-derived coverage.")
    require(
        audit.get("protocol_sha256") == design.protocol_canonical_sha256,
        "Result-build audit protocol hash differs from the supplied protocol.",
    )
    require(
        audit.get("layout_suite_sha256") == design.layout_canonical_sha256,
        "Result-build audit layout hash differs from the supplied layout suite.",
    )
    require(
        audit.get("layout_suite_id") == design.layout_suite["suite_id"],
        "Result-build audit layout-suite ID differs from the supplied layout suite.",
    )
    require(audit.get("layout_count") == len(design.layout_ids), "Result-build audit layout count differs.")
    require(audit.get("expected_train_seeds") == list(design.seeds), "Result-build audit training seeds differ.")
    solver_failures = int(episodes["projection_solver_failure_count"].sum())
    require(
        audit.get("projection_solver_failure_count") == solver_failures,
        "Result-build audit solver-failure count differs from the episode table.",
    )
    return {"counts_exact": True, "hashes_exact": True, "status": "PASS"}


def frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        record: dict[str, object] = {}
        for column, value in row.items():
            if pd.isna(value):
                record[str(column)] = None
            elif isinstance(value, (np.bool_, np.integer, np.floating)):
                record[str(column)] = value.item()
            else:
                record[str(column)] = value
        records.append(record)
    return records


def write_json_exclusive(path: str, value: object) -> None:
    """Serialize completely before creating the requested output file exclusively."""

    try:
        rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"Output is not strict JSON: {error}") from error
    destination = Path(path)
    if destination.exists():
        raise EvidenceError(f"Output already exists; refusing to overwrite: {path!r}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except OSError as error:
        raise EvidenceError(f"Cannot create output {path!r}: {error}") from error
