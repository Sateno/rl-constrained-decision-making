from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.layout_suite import load_navigation_layout_suite


PROTOCOL_SCHEMA = "projection_analysis_protocol_v1"
TABLE_SCHEMA = "projection_result_tables_v1"
MODES = ("disabled", "enabled")
REQUIRED_COLUMNS = {
    "method",
    "train_seed",
    "checkpoint",
    "checkpoint_sha256",
    "layout_suite_id",
    "layout_suite_sha256",
    "layout_id",
    "layout_repeat",
    "evaluation_seed",
    "evaluation_collision_penalty",
    "evaluation_policy_mode",
    "projection_mode",
    "projection_enabled",
    "episode",
    "episode_return",
    "episode_length",
    "success",
    "collision",
    "min_obstacle_clearance",
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
}
SUMMARY_METRICS = (
    "episode_return",
    "episode_length",
    "success_rate",
    "collision_rate",
    "min_obstacle_clearance",
    "projection_intervention_rate",
    "projection_correction_norm",
    "projection_correction_norm_max",
    "projection_slack_sum",
    "projection_slack_max",
)


#################################################################################
# region Protocol and discovery

# Return a strict Boolean series from CSV values.
def as_bool(series: pd.Series, name: str) -> pd.Series:
#{
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}

    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} contains non-Boolean values.")

    return normalized.map(mapping).astype(bool)

#} End function as_bool


# Load the frozen analysis protocol and its referenced layout suite.
def load_protocol(path: str | Path) -> dict[str, object]:
#{
    source = Path(path).resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Analysis protocol not found: {source}")

    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))

    if data.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError(f"schema_version must be {PROTOCOL_SCHEMA!r}.")

    seeds = data.get("expected_train_seeds")
    methods = data.get("methods")

    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("expected_train_seeds must contain unique integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("expected_train_seeds must contain unique integers.")
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a nonempty array.")

    method_names = [str(method.get("method", "")).strip() for method in methods]

    if any(not name for name in method_names) or len(set(method_names)) != len(method_names):
        raise ValueError("methods must define unique nonempty method labels.")

    for method in methods:
        modes = method.get("required_projection_modes")
        if not isinstance(modes, list) or not modes or any(mode not in MODES for mode in modes):
            raise ValueError("required_projection_modes must contain disabled and/or enabled.")
        if len(set(modes)) != len(modes):
            raise ValueError("required_projection_modes must not contain duplicates.")

    repeats = data.get("expected_repeats_per_layout")

    if type(repeats) is not int or repeats <= 0:
        raise ValueError("expected_repeats_per_layout must be positive.")

    suite_path = (source.parent / str(data.get("layout_suite", ""))).resolve()
    suite = load_navigation_layout_suite(suite_path)
    layout_ids = {layout.layout_id for layout in suite.layouts}

    if data.get("representative_layout_id") not in layout_ids:
        raise ValueError("representative_layout_id is not present in the layout suite.")

    projection = data.get("projection_parameters", {})
    required_projection_fields = {"lookahead_distance", "alpha", "slack_penalty", "extra_clearance"}

    if not required_projection_fields.issubset(projection):
        raise ValueError("projection_parameters is incomplete.")

    data["_source_path"] = source
    data["_sha256"] = hashlib.sha256(raw).hexdigest()
    data["_layout_suite"] = suite
    data["_method_map"] = {method["method"]: method for method in methods}
    return data

#} End function load_protocol


# Discover common-layout evaluation CSVs from their columns and internal metadata.
def discover_episodes(protocol: dict[str, object], evaluation_dir: str | Path) -> tuple[pd.DataFrame, dict]:
#{
    evaluation_path = Path(evaluation_dir).resolve()

    if not evaluation_path.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_path}")

    suite = protocol["_layout_suite"]
    method_map = protocol["_method_map"]
    expected_seeds = set(protocol["expected_train_seeds"])
    frames = []
    ignored = []
    csv_paths = sorted(evaluation_path.rglob("*.csv"))

    for csv_path in csv_paths:
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception as error:
            ignored.append({"path": str(csv_path), "reason": f"unreadable: {error}"})
            continue

        if "result_build_schema_version" in header.columns:
            ignored.append({"path": str(csv_path), "reason": "generated_table"})
            continue
        if not REQUIRED_COLUMNS.issubset(header.columns):
            ignored.append({"path": str(csv_path), "reason": "not_layout_evaluation"})
            continue

        frame = pd.read_csv(csv_path)

        if frame.empty:
            raise ValueError(f"Evaluation CSV is empty: {csv_path}")
        if set(frame["layout_suite_sha256"].astype(str)) != {suite.sha256}:
            ignored.append({"path": str(csv_path), "reason": "different_layout_suite"})
            continue

        frame = frame[
            frame["method"].isin(method_map)
            & frame["train_seed"].isin(expected_seeds)
        ].copy()
        frame = frame[
            frame.apply(
                lambda row: row["projection_mode"]
                in method_map[row["method"]]["required_projection_modes"],
                axis=1,
            )
        ]

        if frame.empty:
            ignored.append({"path": str(csv_path), "reason": "not_required_by_protocol"})
            continue

        frame["source_csv"] = str(csv_path)
        frames.append(frame)

    if not frames:
        raise ValueError("No evaluation CSVs matched the protocol and layout suite.")

    episodes = pd.concat(frames, ignore_index=True, sort=False)
    episodes["success"] = as_bool(episodes["success"], "success")
    episodes["collision"] = as_bool(episodes["collision"], "collision")
    episodes["projection_enabled"] = as_bool(
        episodes["projection_enabled"],
        "projection_enabled",
    )
    episodes["result_build_schema_version"] = TABLE_SCHEMA

    audit = {
        "evaluation_dir": str(evaluation_path),
        "discovered_csv_count": len(csv_paths),
        "selected_csv_paths": sorted(set(episodes["source_csv"])),
        "ignored_csvs": ignored,
    }
    return episodes, audit

#} End function discover_episodes

# end region Protocol and discovery


#################################################################################
# region Validation and aggregation

# Validate complete protocol coverage and common evaluation configuration.
def validate_episodes(protocol: dict[str, object], episodes: pd.DataFrame) -> None:
#{
    suite = protocol["_layout_suite"]
    methods = protocol["methods"]
    seeds = protocol["expected_train_seeds"]
    repeats = int(protocol["expected_repeats_per_layout"])
    layout_ids = {layout.layout_id for layout in suite.layouts}
    expected_rows = len(layout_ids) * repeats
    projection = protocol["projection_parameters"]

    checks = {
        "layout_suite_id": suite.suite_id,
        "layout_suite_sha256": suite.sha256,
        "evaluation_policy_mode": protocol["evaluation_policy_mode"],
    }

    for column, expected in checks.items():
        if set(episodes[column].astype(str)) != {str(expected)}:
            raise ValueError(f"Evaluation rows do not share the protocol {column}.")

    numeric_checks = {
        "evaluation_collision_penalty": protocol["evaluation_collision_penalty"],
        "projection_lookahead_distance": projection["lookahead_distance"],
        "projection_alpha": projection["alpha"],
        "projection_slack_penalty": projection["slack_penalty"],
        "projection_extra_clearance": projection["extra_clearance"],
    }

    for column, expected in numeric_checks.items():
        values = pd.to_numeric(episodes[column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)) or not np.allclose(values, float(expected), atol=1.0e-12, rtol=0.0):
            raise ValueError(f"Evaluation rows do not share the protocol {column}.")

    if int(pd.to_numeric(episodes["projection_solver_failure_count"]).sum()) != 0:
        raise ValueError("The selected evaluation dataset contains projection solver failures.")

    expected_enabled = episodes["projection_mode"].eq("enabled")
    if not bool((episodes["projection_enabled"] == expected_enabled).all()):
        raise ValueError("projection_enabled disagrees with projection_mode.")

    duplicate_columns = [
        "method",
        "train_seed",
        "checkpoint_sha256",
        "projection_mode",
        "layout_id",
        "layout_repeat",
        "evaluation_seed",
    ]
    duplicates = episodes.duplicated(duplicate_columns, keep=False)

    if bool(duplicates.any()):
        raise ValueError("Duplicate common-layout episode keys were discovered.")

    for method in methods:
        method_name = method["method"]
        expected_penalty = method.get("training_collision_penalty")
        expected_training_projection = method.get("training_projection_enabled")

        for seed in seeds:
            checkpoint_hashes = set()
            mode_keys = []

            for mode in method["required_projection_modes"]:
                group = episodes[
                    (episodes["method"] == method_name)
                    & (episodes["train_seed"] == seed)
                    & (episodes["projection_mode"] == mode)
                ]

                if len(group) != expected_rows:
                    raise ValueError(
                        f"Expected {expected_rows} rows for {method_name}, seed {seed}, {mode}; "
                        f"found {len(group)}."
                    )
                if set(group["layout_id"].astype(str)) != layout_ids:
                    raise ValueError(f"Incomplete layout coverage for {method_name}, seed {seed}, {mode}.")

                for _, layout_rows in group.groupby("layout_id"):
                    if set(layout_rows["layout_repeat"].astype(int)) != set(range(repeats)):
                        raise ValueError(f"Incomplete layout-repeat coverage for {method_name}, seed {seed}, {mode}.")

                hashes = set(group["checkpoint_sha256"].astype(str))
                if len(hashes) != 1:
                    raise ValueError(f"Multiple checkpoints were mixed for {method_name}, seed {seed}, {mode}.")
                checkpoint_hashes.update(hashes)
                mode_keys.append(
                    set(
                        zip(
                            group["layout_id"].astype(str),
                            group["layout_repeat"].astype(int),
                            group["evaluation_seed"].astype(int),
                        )
                    )
                )

            if len(checkpoint_hashes) != 1:
                raise ValueError(f"Projection modes use different checkpoints for {method_name}, seed {seed}.")
            if len(mode_keys) > 1 and any(keys != mode_keys[0] for keys in mode_keys[1:]):
                raise ValueError(f"Projection modes are not paired for {method_name}, seed {seed}.")

        method_rows = episodes[episodes["method"] == method_name]

        if expected_penalty is not None:
            values = pd.to_numeric(method_rows["training_collision_penalty"], errors="coerce").to_numpy(float)
            if not np.allclose(values, float(expected_penalty), atol=1.0e-12, rtol=0.0):
                raise ValueError(f"Unexpected training collision penalty for {method_name}.")

        if expected_training_projection is not None:
            values = as_bool(method_rows["training_projection_enabled"], "training_projection_enabled")
            if not bool((values == bool(expected_training_projection)).all()):
                raise ValueError(f"Unexpected training projection mode for {method_name}.")

#} End function validate_episodes


# Average layouts within each checkpoint.
def checkpoint_summary(protocol: dict[str, object], episodes: pd.DataFrame) -> pd.DataFrame:
#{
    frame = episodes.copy()
    disabled = frame["projection_mode"].eq("disabled")

    for column in (
        "projection_intervention_rate",
        "mean_projection_correction_norm",
        "max_projection_correction_norm",
        "mean_projection_slack_sum",
        "max_projection_slack",
    ):
        frame.loc[disabled, column] = np.nan

    group_columns = ["method", "train_seed", "checkpoint", "checkpoint_sha256", "projection_mode"]
    aggregations = {
        "episode_count": ("episode", "size"),
        "layout_count": ("layout_id", "nunique"),
        "episode_return": ("episode_return", "mean"),
        "episode_length": ("episode_length", "mean"),
        "success_rate": ("success", "mean"),
        "collision_rate": ("collision", "mean"),
        "min_obstacle_clearance": ("min_obstacle_clearance", "mean"),
        "projection_intervention_rate": ("projection_intervention_rate", "mean"),
        "projection_correction_norm": ("mean_projection_correction_norm", "mean"),
        "projection_correction_norm_max": ("max_projection_correction_norm", "max"),
        "projection_slack_sum": ("mean_projection_slack_sum", "mean"),
        "projection_slack_max": ("max_projection_slack", "max"),
    }

    if "training_collision_penalty" in frame.columns:
        aggregations["training_collision_penalty"] = ("training_collision_penalty", "first")
    if "training_projection_enabled" in frame.columns:
        aggregations["training_projection_enabled"] = ("training_projection_enabled", "first")

    summary = frame.groupby(group_columns, as_index=False, sort=False).agg(**aggregations)
    display_names = {method["method"]: method["display_name"] for method in protocol["methods"]}
    summary.insert(0, "result_build_schema_version", TABLE_SCHEMA)
    summary.insert(2, "display_name", summary["method"].map(display_names))
    return summary

#} End function checkpoint_summary


# Aggregate checkpoint scores across independent training seeds.
def method_summary(protocol: dict[str, object], checkpoints: pd.DataFrame) -> pd.DataFrame:
#{
    rows = []

    for (method, display_name, mode), group in checkpoints.groupby(
        ["method", "display_name", "projection_mode"],
        sort=False,
    ):
        row = {
            "result_build_schema_version": TABLE_SCHEMA,
            "method": method,
            "display_name": display_name,
            "projection_mode": mode,
            "seed_count": int(group["train_seed"].nunique()),
        }

        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))

        rows.append(row)

    order = {}
    order_index = 0

    for method in protocol["methods"]:
        for mode in method["required_projection_modes"]:
            order[(method["method"], mode)] = order_index
            order_index += 1
    result = pd.DataFrame(rows)
    result["_order"] = [order[(row.method, row.projection_mode)] for row in result.itertuples()]
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)

#} End function method_summary


# Average enabled-minus-disabled differences within each checkpoint.
def paired_deltas(protocol: dict[str, object], episodes: pd.DataFrame) -> pd.DataFrame:
#{
    rows = []
    pair_columns = ["layout_id", "layout_repeat", "evaluation_seed"]
    metrics = ["episode_return", "episode_length", "success", "collision", "min_obstacle_clearance"]

    for method in protocol["methods"]:
        if not {"disabled", "enabled"}.issubset(method["required_projection_modes"]):
            continue

        for seed in protocol["expected_train_seeds"]:
            group = episodes[(episodes["method"] == method["method"]) & (episodes["train_seed"] == seed)]
            disabled = group[group["projection_mode"] == "disabled"]
            enabled = group[group["projection_mode"] == "enabled"]
            merged = disabled[pair_columns + ["checkpoint_sha256", *metrics]].merge(
                enabled[pair_columns + ["checkpoint_sha256", *metrics]],
                on=pair_columns,
                suffixes=("_disabled", "_enabled"),
                validate="one_to_one",
            )
            row = {
                "result_build_schema_version": TABLE_SCHEMA,
                "method": method["method"],
                "display_name": method["display_name"],
                "train_seed": seed,
                "checkpoint_sha256": merged["checkpoint_sha256_disabled"].iloc[0],
                "paired_layout_count": len(merged),
            }

            for metric in metrics:
                enabled_values = pd.to_numeric(merged[f"{metric}_enabled"], errors="coerce").astype(float)
                disabled_values = pd.to_numeric(merged[f"{metric}_disabled"], errors="coerce").astype(float)
                row[f"{metric}_delta_enabled_minus_disabled"] = float((enabled_values - disabled_values).mean())

            rows.append(row)

    return pd.DataFrame(rows)

#} End function paired_deltas


# Aggregate paired checkpoint deltas across independent training seeds.
def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
#{
    if paired.empty:
        return pd.DataFrame()

    rows = []
    delta_columns = [
        column
        for column in paired.columns
        if column.endswith("_delta_enabled_minus_disabled")
    ]

    for (method, display_name), group in paired.groupby(
        ["method", "display_name"],
        sort=False,
    ):
        row = {
            "result_build_schema_version": TABLE_SCHEMA,
            "method": method,
            "display_name": display_name,
            "seed_count": int(group["train_seed"].nunique()),
        }

        for column in delta_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)

#} End function paired_summary

# end region Validation and aggregation


#################################################################################
# region Output

# Escape one short text value for LaTeX.
def latex_escape(value: object) -> str:
#{
    text = str(value)

    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]:
        text = text.replace(old, new)

    return text

#} End function latex_escape


# Format a mean and standard deviation for LaTeX.
def mean_std(mean_value: object, std_value: object, digits: int = 3) -> str:
#{
    mean_value = float(mean_value)
    std_value = float(std_value)

    if not np.isfinite(mean_value):
        return "--"
    if not np.isfinite(std_value):
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} $\\pm$ {std_value:.{digits}f}"

#} End function mean_std


# Write compact include-ready LaTeX tables.
def write_latex_tables(methods: pd.DataFrame, paired_methods: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
#{
    method_path = output_dir / "generated_method_summary.tex"
    paired_path = output_dir / "generated_paired_projection_deltas.tex"
    method_lines = [
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Method & Projection & Return & Success & Collision & Clearance & Intervention & Correction & Slack \\\\",
        "\\midrule",
    ]

    for row in methods.itertuples(index=False):
        values = [
            latex_escape(row.display_name),
            "On" if row.projection_mode == "enabled" else "Off",
            mean_std(row.episode_return_mean, row.episode_return_std),
            mean_std(row.success_rate_mean, row.success_rate_std),
            mean_std(row.collision_rate_mean, row.collision_rate_std),
            mean_std(row.min_obstacle_clearance_mean, row.min_obstacle_clearance_std),
            mean_std(row.projection_intervention_rate_mean, row.projection_intervention_rate_std),
            mean_std(row.projection_correction_norm_mean, row.projection_correction_norm_std),
            mean_std(row.projection_slack_sum_mean, row.projection_slack_sum_std, 6),
        ]
        method_lines.append(" & ".join(values) + " \\\\")

    method_lines.extend(["\\bottomrule", "\\end{tabular}"])
    method_path.write_text("\n".join(method_lines) + "\n", encoding="utf-8")
    paired_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Return $\\Delta$ & Success $\\Delta$ & Collision $\\Delta$ & Clearance $\\Delta$ \\\\",
        "\\midrule",
    ]

    for row in paired_methods.itertuples(index=False):
        values = []

        for metric in ("episode_return", "success", "collision", "min_obstacle_clearance"):
            column = f"{metric}_delta_enabled_minus_disabled"
            values.append(
                mean_std(
                    getattr(row, f"{column}_mean"),
                    getattr(row, f"{column}_std"),
                )
            )

        paired_lines.append(
            f"{latex_escape(row.display_name)} & " + " & ".join(values) + " \\"
        )

    paired_lines.extend(["\\bottomrule", "\\end{tabular}"])
    paired_path.write_text("\n".join(paired_lines) + "\n", encoding="utf-8")
    return method_path, paired_path

#} End function write_latex_tables


# Build all canonical tables from discovered common-layout episode CSVs.
def build_result_tables(protocol_path: str | Path, evaluation_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
#{
    protocol = load_protocol(protocol_path)
    episodes, discovery_audit = discover_episodes(protocol, evaluation_dir)
    validate_episodes(protocol, episodes)
    checkpoints = checkpoint_summary(protocol, episodes)
    methods = method_summary(protocol, checkpoints)
    paired = paired_deltas(protocol, episodes)
    paired_methods = paired_summary(paired)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "episodes": output / "evaluation_episode_results.csv",
        "checkpoints": output / "checkpoint_summary.csv",
        "methods": output / "method_summary.csv",
        "paired": output / "paired_projection_deltas.csv",
        "paired_summary": output / "paired_projection_summary.csv",
        "audit": output / "result_build_audit.json",
    }
    episodes.to_csv(paths["episodes"], index=False)
    checkpoints.to_csv(paths["checkpoints"], index=False)
    methods.to_csv(paths["methods"], index=False)
    paired.to_csv(paths["paired"], index=False)
    paired_methods.to_csv(paths["paired_summary"], index=False)
    method_latex, paired_latex = write_latex_tables(methods, paired_methods, output)
    paths["method_latex"] = method_latex
    paths["paired_latex"] = paired_latex
    suite = protocol["_layout_suite"]
    audit = {
        "status": "PASS",
        "result_build_schema_version": TABLE_SCHEMA,
        "study_id": protocol["study_id"],
        "protocol_path": str(protocol["_source_path"]),
        "protocol_sha256": protocol["_sha256"],
        "layout_suite_path": str(suite.source_path.resolve()),
        "layout_suite_id": suite.suite_id,
        "layout_suite_sha256": suite.sha256,
        "layout_count": len(suite.layouts),
        "expected_train_seeds": protocol["expected_train_seeds"],
        **discovery_audit,
        "selected_csv_count": len(discovery_audit["selected_csv_paths"]),
        "episode_row_count": len(episodes),
        "checkpoint_row_count": len(checkpoints),
        "method_row_count": len(methods),
        "paired_checkpoint_row_count": len(paired),
        "paired_method_row_count": len(paired_methods),
        "projection_solver_failure_count": 0,
        "outputs": {name: str(path) for name, path in paths.items() if name != "audit"},
    }
    paths["audit"].write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return paths

#} End function build_result_tables

# end region Output


#################################################################################
# region Command line

# Run one saved-result table build.
def main() -> None:
#{
    parser = argparse.ArgumentParser(description="Audit and aggregate common-layout evaluation CSVs.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = build_result_tables(args.protocol, args.evaluation_dir, args.output_dir)
    print("Result aggregation completed successfully.")

    for name, path in outputs.items():
        print(f"{name}: {path}")

#} End function main

# end region Command line


if __name__ == "__main__":
    main()
