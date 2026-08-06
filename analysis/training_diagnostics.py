from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_DIAGNOSTICS_SCHEMA = "training_diagnostics_v1"

EPISODE_REQUIRED_TAGS = (
    "charts/episodic_return",
    "charts/episodic_length",
    "safety/success",
    "safety/collision",
    "safety/timeout",
)
EPISODE_OPTIONAL_TAGS = (
    "safety/final_obstacle_clearance",
)
ACTION_BOUND_TAGS = (
    "action_bounds/clipping_frequency",
    "action_bounds/speed_clipping_frequency",
    "action_bounds/turn_rate_clipping_frequency",
    "action_bounds/clipping_norm",
    "action_bounds/clipping_norm_max",
)
PROJECTION_TAGS = (
    "projection/transition_count",
    "projection/intervention_count",
    "projection/intervention_frequency",
    "projection/correction_norm",
    "projection/correction_norm_max",
    "projection/slack_sum",
    "projection/slack_max",
    "projection/solver_failure_count",
)
ALL_TRAINING_TAGS = (
    *EPISODE_REQUIRED_TAGS,
    *EPISODE_OPTIONAL_TAGS,
    *ACTION_BOUND_TAGS,
    *PROJECTION_TAGS,
)

TRAINING_CURVES = (
    ("charts/episodic_return", "training_return.pdf", "Episode return"),
    (
        "action_bounds/clipping_frequency",
        "training_action_bound_clipping.pdf",
        "Action-bound clipping frequency",
    ),
    (
        "projection/intervention_frequency",
        "training_projection_intervention.pdf",
        "Projection intervention frequency",
    ),
    (
        "projection/correction_norm",
        "training_projection_correction.pdf",
        "Mean projection correction norm",
    ),
    (
        "projection/correction_norm_max",
        "training_projection_correction_max.pdf",
        "Maximum projection correction norm",
    ),
    (
        "projection/slack_sum",
        "training_projection_slack.pdf",
        "Mean summed projection slack",
    ),
    (
        "projection/slack_max",
        "training_projection_slack_max.pdf",
        "Maximum projection slack",
    ),
)

DERIVED_EPISODE_CURVES = (
    (
        "cumulative_collision_count",
        "training_cumulative_collisions.pdf",
        "Cumulative collisions",
    ),
    (
        "rolling_collision_rate",
        "training_rolling_collision_rate.pdf",
        "Rolling collision rate",
    ),
    (
        "rolling_success_rate",
        "training_rolling_success_rate.pdf",
        "Rolling success rate",
    ),
)

ROLLOUT_COLUMN_NAMES = {
    "action_bounds/clipping_frequency": "action_bound_clipping_frequency",
    "action_bounds/speed_clipping_frequency": "speed_action_bound_clipping_frequency",
    "action_bounds/turn_rate_clipping_frequency": "turn_rate_action_bound_clipping_frequency",
    "action_bounds/clipping_norm": "mean_action_bound_clipping_norm",
    "action_bounds/clipping_norm_max": "max_action_bound_clipping_norm",
    "projection/transition_count": "projection_transition_count",
    "projection/intervention_count": "projection_intervention_count",
    "projection/intervention_frequency": "projection_intervention_frequency",
    "projection/correction_norm": "mean_projection_correction_norm",
    "projection/correction_norm_max": "max_projection_correction_norm",
    "projection/slack_sum": "mean_projection_slack_sum",
    "projection/slack_max": "max_projection_slack",
    "projection/solver_failure_count": "projection_solver_failure_count",
}


#################################################################################
# region TensorBoard discovery

# Return one file SHA-256.
def file_sha256(path: Path) -> str:
#{
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

#} End function file_sha256


# Parse the Markdown table written by CleanRL to the TensorBoard text summary.
def parse_hyperparameters(text: str) -> dict[str, str]:
#{
    values = {}

    for line in text.splitlines():
        if not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.split("|")[1:-1]]

        if len(cells) < 2 or cells[0] in {"param", "-"}:
            continue

        values[cells[0]] = cells[1]

    return values

#} End function parse_hyperparameters


# Read one TensorBoard run without importing the PPO or PyTorch modules.
def load_tensorboard_run(run_dir: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
#{
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0, "tensors": 1},
    )
    accumulator.Reload()
    tags = accumulator.Tags()
    hyperparameters = {}
    text_tag = "hyperparameters/text_summary"

    if text_tag in tags.get("tensors", []):
        events = accumulator.Tensors(text_tag)

        if events and events[-1].tensor_proto.string_val:
            text = events[-1].tensor_proto.string_val[0].decode("utf-8")
            hyperparameters = parse_hyperparameters(text)

    scalars = {}

    for tag in ALL_TRAINING_TAGS:
        if tag not in tags.get("scalars", []):
            continue

        events = accumulator.Scalars(tag)
        scalars[tag] = pd.DataFrame(
            {
                "event_index": np.arange(len(events), dtype=np.int64),
                "step": [event.step for event in events],
                "value": [event.value for event in events],
            }
        )

    return hyperparameters, scalars

#} End function load_tensorboard_run


# Resolve a checkpoint path recorded in TensorBoard against the repository root.
def resolve_recorded_checkpoint(path_text: str, runs_dir: Path) -> Path:
#{
    normalized = Path(path_text.strip().replace("\\", "/"))

    if normalized.is_absolute():
        return normalized

    return (runs_dir.parent / normalized).resolve()

#} End function resolve_recorded_checkpoint


# Return whether checkpoint metadata declares projection-enabled training.
def checkpoint_projection_mode(row: object) -> bool:
#{
    value = getattr(row, "training_projection_enabled", False)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0", "", "nan"}:
            return False
        raise ValueError(f"Invalid training_projection_enabled value: {value!r}")

    if pd.isna(value):
        return False

    return bool(value)

#} End function checkpoint_projection_mode


# Discover TensorBoard runs by checkpoint identity and return all research scalars.
def training_points(
    checkpoint_summary: pd.DataFrame,
    runs_dir: Path,
    require_complete: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
#{
    identity_columns = [
        "method",
        "display_name",
        "train_seed",
        "checkpoint_sha256",
    ]

    if "training_projection_enabled" in checkpoint_summary.columns:
        identity_columns.append("training_projection_enabled")

    unique = checkpoint_summary[identity_columns].drop_duplicates()
    required_hashes = set(unique["checkpoint_sha256"].astype(str))
    run_index = {}
    skipped = []
    event_directories = sorted(
        {path.parent for path in runs_dir.rglob("events.out.tfevents.*")}
    )

    for run_dir in event_directories:
        try:
            hyperparameters, scalars = load_tensorboard_run(run_dir)
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("tensorboard"):
                if require_complete:
                    raise RuntimeError(
                        "TensorBoard is required by the final analysis protocol."
                    ) from error
                return pd.DataFrame(), [
                    {"reason": "tensorboard_unavailable", "details": str(error)}
                ]
            raise
        except Exception as error:
            skipped.append(
                {
                    "run_dir": str(run_dir),
                    "reason": "tensorboard_run_unreadable",
                    "details": str(error),
                }
            )
            continue

        checkpoint_text = hyperparameters.get("checkpoint_path", "").strip()

        if not checkpoint_text:
            skipped.append(
                {"run_dir": str(run_dir), "reason": "checkpoint_path_not_recorded"}
            )
            continue

        checkpoint_path = resolve_recorded_checkpoint(checkpoint_text, runs_dir)

        if not checkpoint_path.is_file():
            skipped.append(
                {
                    "run_dir": str(run_dir),
                    "reason": "recorded_checkpoint_not_found",
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            continue

        checkpoint_hash = file_sha256(checkpoint_path)

        if checkpoint_hash not in required_hashes:
            continue

        if checkpoint_hash in run_index:
            raise ValueError(
                f"Duplicate TensorBoard runs for checkpoint SHA-256 {checkpoint_hash}: "
                f"{run_index[checkpoint_hash][0]} and {run_dir}"
            )

        run_index[checkpoint_hash] = (run_dir, hyperparameters, scalars)

    frames = []

    for row in unique.itertuples(index=False):
        checkpoint_hash = str(row.checkpoint_sha256)
        match = run_index.get(checkpoint_hash)

        if match is None:
            item = {
                "method": row.method,
                "train_seed": int(row.train_seed),
                "reason": "tensorboard_run_not_found",
            }
            if require_complete:
                raise ValueError(
                    "Required TensorBoard run was not found for "
                    f"{row.method}, seed {int(row.train_seed)}."
                )
            skipped.append(item)
            continue

        run_dir, hyperparameters, scalars = match
        recorded_seed = hyperparameters.get("seed", "").strip()
        recorded_method = hyperparameters.get("method", "").strip()
        projection_enabled = checkpoint_projection_mode(row)

        if recorded_seed and int(recorded_seed) != int(row.train_seed):
            raise ValueError(
                f"TensorBoard seed {recorded_seed} does not match training seed "
                f"{row.train_seed} for {run_dir}."
            )

        if recorded_method and recorded_method != row.method:
            raise ValueError(
                f"TensorBoard method {recorded_method!r} does not match evaluation method "
                f"{row.method!r} for {run_dir}."
            )

        required_tags = set(EPISODE_REQUIRED_TAGS) | set(ACTION_BOUND_TAGS)
        if projection_enabled:
            required_tags.update(PROJECTION_TAGS)

        for tag in ALL_TRAINING_TAGS:
            frame = scalars.get(tag, pd.DataFrame())

            if frame.empty:
                if tag not in required_tags:
                    continue

                item = {
                    "method": row.method,
                    "train_seed": int(row.train_seed),
                    "tag": tag,
                    "reason": "scalar_not_found",
                }
                if require_complete:
                    raise ValueError(
                        f"Required {tag} scalar was not found for "
                        f"{row.method}, seed {int(row.train_seed)}."
                    )
                skipped.append(item)
                continue

            frame = frame.copy()
            frame.insert(0, "training_diagnostics_schema_version", TRAINING_DIAGNOSTICS_SCHEMA)
            frame["method"] = row.method
            frame["display_name"] = row.display_name
            frame["train_seed"] = int(row.train_seed)
            frame["checkpoint_sha256"] = checkpoint_hash
            frame["training_projection_enabled"] = projection_enabled
            frame["run_dir"] = str(run_dir)
            frame["tag"] = tag
            frames.append(frame)

    points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return points, skipped

#} End function training_points

# end region TensorBoard discovery


#################################################################################
# region Structured training tables

# Add an occurrence index for multiple events written at one environment step.
def add_step_occurrence(frame: pd.DataFrame) -> pd.DataFrame:
#{
    result = frame.sort_values(["event_index", "step"]).copy()
    result["step_occurrence"] = result.groupby("step", sort=False).cumcount()
    return result

#} End function add_step_occurrence


# Build one complete episode-level training-safety table.
def training_episode_diagnostics(
    points: pd.DataFrame,
    rolling_window: int,
    require_complete: bool = False,
) -> pd.DataFrame:
#{
    if type(rolling_window) is not int or rolling_window <= 0:
        raise ValueError("rolling_window must be a positive integer.")
    if points.empty:
        return pd.DataFrame()

    rows = []
    group_columns = [
        "method",
        "display_name",
        "train_seed",
        "checkpoint_sha256",
        "training_projection_enabled",
        "run_dir",
    ]

    for key, group in points.groupby(group_columns, sort=False, dropna=False):
        tag_frames = {}

        for tag in (*EPISODE_REQUIRED_TAGS, *EPISODE_OPTIONAL_TAGS):
            frame = group[group["tag"] == tag]
            if frame.empty:
                if tag in EPISODE_REQUIRED_TAGS and require_complete:
                    raise ValueError(
                        f"Required training episode tag {tag} is missing for {key}."
                    )
                continue

            frame = add_step_occurrence(frame)
            tag_frames[tag] = frame[
                ["step", "step_occurrence", "value"]
            ].rename(columns={"value": tag})

        missing = [tag for tag in EPISODE_REQUIRED_TAGS if tag not in tag_frames]
        if missing:
            if require_complete:
                raise ValueError(
                    f"Incomplete training episode diagnostics for {key}: {missing}"
                )
            continue

        reference_keys = set(
            zip(
                tag_frames[EPISODE_REQUIRED_TAGS[0]]["step"].astype(int),
                tag_frames[EPISODE_REQUIRED_TAGS[0]]["step_occurrence"].astype(int),
            )
        )

        for tag in EPISODE_REQUIRED_TAGS[1:]:
            keys = set(
                zip(
                    tag_frames[tag]["step"].astype(int),
                    tag_frames[tag]["step_occurrence"].astype(int),
                )
            )
            if keys != reference_keys:
                raise ValueError(
                    f"Training episode tag {tag} is not aligned with "
                    f"{EPISODE_REQUIRED_TAGS[0]} for {key}."
                )

        for tag in EPISODE_OPTIONAL_TAGS:
            if tag not in tag_frames:
                continue
            optional_keys = set(
                zip(
                    tag_frames[tag]["step"].astype(int),
                    tag_frames[tag]["step_occurrence"].astype(int),
                )
            )
            if optional_keys != reference_keys:
                # Optional scalars may be omitted for undefined values. Without
                # an explicit episode identifier they cannot be aligned safely.
                del tag_frames[tag]

        merged = tag_frames[EPISODE_REQUIRED_TAGS[0]]

        for tag in (*EPISODE_REQUIRED_TAGS[1:], *EPISODE_OPTIONAL_TAGS):
            if tag not in tag_frames:
                continue
            merged = merged.merge(
                tag_frames[tag],
                on=["step", "step_occurrence"],
                how="left" if tag in EPISODE_OPTIONAL_TAGS else "inner",
                validate="one_to_one",
            )

        merged = merged.sort_values(["step", "step_occurrence"]).reset_index(drop=True)
        merged = merged.rename(
            columns={
                "charts/episodic_return": "episode_return",
                "charts/episodic_length": "episode_length",
                "safety/success": "success",
                "safety/collision": "collision",
                "safety/timeout": "timeout",
                "safety/final_obstacle_clearance": "final_obstacle_clearance",
            }
        )
        merged["episode_length"] = pd.to_numeric(
            merged["episode_length"],
            errors="raise",
        ).astype(int)

        for outcome in ("success", "collision", "timeout"):
            values = pd.to_numeric(merged[outcome], errors="coerce").to_numpy(float)
            if not np.all(np.isfinite(values)) or not np.all(
                np.isclose(values, 0.0) | np.isclose(values, 1.0)
            ):
                raise ValueError(f"Training {outcome} values must be binary for {key}.")
            merged[outcome] = np.rint(values).astype(int)

        outcome_sum = merged[["success", "collision", "timeout"]].sum(axis=1)
        if not bool((outcome_sum == 1).all()):
            raise ValueError(
                f"Training episode outcomes are not mutually exclusive and exhaustive for {key}."
            )

        merged.insert(0, "training_diagnostics_schema_version", TRAINING_DIAGNOSTICS_SCHEMA)
        for column, value in zip(group_columns, key):
            merged[column] = value
        merged["completed_episode"] = np.arange(1, len(merged) + 1, dtype=np.int64)
        merged["cumulative_collision_count"] = merged["collision"].cumsum()
        merged["cumulative_success_count"] = merged["success"].cumsum()
        merged["cumulative_timeout_count"] = merged["timeout"].cumsum()
        merged["rolling_window_episodes"] = int(rolling_window)
        merged["rolling_collision_rate"] = merged["collision"].rolling(
            window=rolling_window,
            min_periods=1,
        ).mean()
        merged["rolling_success_rate"] = merged["success"].rolling(
            window=rolling_window,
            min_periods=1,
        ).mean()
        merged["rolling_timeout_rate"] = merged["timeout"].rolling(
            window=rolling_window,
            min_periods=1,
        ).mean()
        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

#} End function training_episode_diagnostics


# Build one wide rollout-level action and projection diagnostic table.
def training_rollout_diagnostics(
    points: pd.DataFrame,
    require_complete: bool = False,
) -> pd.DataFrame:
#{
    if points.empty:
        return pd.DataFrame()

    rows = []
    group_columns = [
        "method",
        "display_name",
        "train_seed",
        "checkpoint_sha256",
        "training_projection_enabled",
        "run_dir",
    ]

    for key, group in points.groupby(group_columns, sort=False, dropna=False):
        projection_enabled = bool(key[group_columns.index("training_projection_enabled")])
        required_tags = set(ACTION_BOUND_TAGS)
        if projection_enabled:
            required_tags.update(PROJECTION_TAGS)

        tag_frames = {}

        for tag in (*ACTION_BOUND_TAGS, *PROJECTION_TAGS):
            frame = group[group["tag"] == tag]
            if frame.empty:
                if tag in required_tags and require_complete:
                    raise ValueError(
                        f"Required rollout tag {tag} is missing for {key}."
                    )
                continue
            if frame["step"].duplicated().any():
                raise ValueError(f"Rollout tag {tag} has duplicate steps for {key}.")

            tag_frames[tag] = frame[["step", "value"]].rename(
                columns={"value": ROLLOUT_COLUMN_NAMES[tag]}
            )

        missing = [tag for tag in required_tags if tag not in tag_frames]
        if missing:
            if require_complete:
                raise ValueError(
                    f"Incomplete training rollout diagnostics for {key}: {missing}"
                )
            continue

        base_tag = ACTION_BOUND_TAGS[0]
        reference_steps = set(tag_frames[base_tag]["step"].astype(int))

        for tag in required_tags - {base_tag}:
            steps = set(tag_frames[tag]["step"].astype(int))
            if steps != reference_steps:
                raise ValueError(
                    f"Training rollout tag {tag} is not aligned with {base_tag} for {key}."
                )

        merged = tag_frames[base_tag]

        for tag in (*ACTION_BOUND_TAGS[1:], *PROJECTION_TAGS):
            if tag not in tag_frames:
                continue
            merged = merged.merge(
                tag_frames[tag],
                on="step",
                how="left" if tag in PROJECTION_TAGS and not projection_enabled else "inner",
                validate="one_to_one",
            )

        merged = merged.sort_values("step").reset_index(drop=True)
        merged.insert(0, "training_diagnostics_schema_version", TRAINING_DIAGNOSTICS_SCHEMA)
        for column, value in zip(group_columns, key):
            merged[column] = value

        frequency_columns = [
            "action_bound_clipping_frequency",
            "speed_action_bound_clipping_frequency",
            "turn_rate_action_bound_clipping_frequency",
        ]
        if projection_enabled:
            frequency_columns.append("projection_intervention_frequency")

        for column in frequency_columns:
            values = pd.to_numeric(merged[column], errors="coerce").to_numpy(float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"{column} must be finite and within [0, 1] for {key}.")

        nonnegative_columns = [
            "mean_action_bound_clipping_norm",
            "max_action_bound_clipping_norm",
        ]
        if projection_enabled:
            nonnegative_columns.extend(
                [
                    "projection_transition_count",
                    "projection_intervention_count",
                    "mean_projection_correction_norm",
                    "max_projection_correction_norm",
                    "mean_projection_slack_sum",
                    "max_projection_slack",
                    "projection_solver_failure_count",
                ]
            )

        for column in nonnegative_columns:
            values = pd.to_numeric(merged[column], errors="coerce").to_numpy(float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{column} must be finite and nonnegative for {key}.")

        action_mean = pd.to_numeric(
            merged["mean_action_bound_clipping_norm"],
            errors="coerce",
        ).to_numpy(float)
        action_max = pd.to_numeric(
            merged["max_action_bound_clipping_norm"],
            errors="coerce",
        ).to_numpy(float)
        if np.any(action_mean > action_max + 1.0e-12):
            raise ValueError(
                f"Mean action-bound clipping norm exceeds the maximum for {key}."
            )

        if projection_enabled:
            transition_counts = pd.to_numeric(
                merged["projection_transition_count"],
                errors="coerce",
            ).to_numpy(float)
            intervention_counts = pd.to_numeric(
                merged["projection_intervention_count"],
                errors="coerce",
            ).to_numpy(float)
            intervention_frequencies = pd.to_numeric(
                merged["projection_intervention_frequency"],
                errors="coerce",
            ).to_numpy(float)
            failures = pd.to_numeric(
                merged["projection_solver_failure_count"],
                errors="coerce",
            ).to_numpy(float)

            for name, values in (
                ("projection_transition_count", transition_counts),
                ("projection_intervention_count", intervention_counts),
                ("projection_solver_failure_count", failures),
            ):
                if not np.allclose(values, np.round(values), atol=0.0, rtol=0.0):
                    raise ValueError(f"{name} must contain integer values for {key}.")

            if np.any(transition_counts <= 0.0):
                raise ValueError(f"Projection transition counts must be positive for {key}.")
            if np.any(intervention_counts > transition_counts):
                raise ValueError(
                    f"Projection intervention counts exceed transition counts for {key}."
                )
            if not np.allclose(
                intervention_frequencies,
                intervention_counts / transition_counts,
                atol=1.0e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Projection intervention frequency disagrees with counts for {key}."
                )

            correction_mean = pd.to_numeric(
                merged["mean_projection_correction_norm"],
                errors="coerce",
            ).to_numpy(float)
            correction_max = pd.to_numeric(
                merged["max_projection_correction_norm"],
                errors="coerce",
            ).to_numpy(float)
            if np.any(correction_mean > correction_max + 1.0e-12):
                raise ValueError(
                    f"Mean projection correction exceeds the maximum for {key}."
                )
            if np.any(failures != 0.0):
                raise ValueError(
                    f"Accepted training diagnostics contain projection solver failures for {key}."
                )

        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

#} End function training_rollout_diagnostics

# end region Structured training tables


#################################################################################
# region Seed aggregation

# Interpolate seed curves onto one common within-method grid.
def aggregate_curve(points: pd.DataFrame, tag: str) -> pd.DataFrame:
#{
    if points.empty or "tag" not in points.columns:
        return pd.DataFrame()

    rows = []

    for (method, display_name), method_rows in points[points["tag"] == tag].groupby(
        ["method", "display_name"],
        sort=False,
    ):
        seed_curves = []

        for _, seed_rows in method_rows.groupby("train_seed"):
            seed_rows = (
                seed_rows.groupby("step", as_index=False)["value"]
                .mean()
                .sort_values("step")
            )
            if not seed_rows.empty:
                seed_curves.append(
                    (
                        seed_rows["step"].to_numpy(float),
                        seed_rows["value"].to_numpy(float),
                    )
                )

        if not seed_curves:
            continue
        if len(seed_curves) == 1:
            grid = seed_curves[0][0]
        else:
            start = max(curve[0][0] for curve in seed_curves)
            end = min(curve[0][-1] for curve in seed_curves)
            if end < start:
                continue
            grid = np.linspace(
                start,
                end,
                min(200, max(len(curve[0]) for curve in seed_curves)),
            )

        values = np.vstack(
            [np.interp(grid, steps, scalar) for steps, scalar in seed_curves]
        )
        rows.append(
            pd.DataFrame(
                {
                    "training_diagnostics_schema_version": TRAINING_DIAGNOSTICS_SCHEMA,
                    "method": method,
                    "display_name": display_name,
                    "tag": tag,
                    "step": grid,
                    "value_mean": values.mean(axis=0),
                    "value_std": (
                        values.std(axis=0, ddof=1)
                        if len(values) > 1
                        else np.nan
                    ),
                    "seed_count": len(values),
                }
            )
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

#} End function aggregate_curve


# Aggregate one episode-completion diagnostic as a step function across seeds.
def aggregate_episode_curve(
    episodes: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
#{
    if episodes.empty or value_column not in episodes.columns:
        return pd.DataFrame()

    rows = []
    tag = f"training/{value_column}"

    for (method, display_name), method_rows in episodes.groupby(
        ["method", "display_name"],
        sort=False,
    ):
        seed_curves = []

        for _, seed_rows in method_rows.groupby("train_seed", sort=False):
            ordered = seed_rows.sort_values(
                ["step", "completed_episode"]
            )
            # Multiple vector environments can complete episodes at the same
            # global step. The final event at that step represents the complete
            # post-step cumulative or rolling state.
            ordered = ordered.groupby("step", as_index=False, sort=True).last()
            steps = ordered["step"].to_numpy(float)
            values = pd.to_numeric(
                ordered[value_column],
                errors="coerce",
            ).to_numpy(float)

            if not np.all(np.isfinite(steps)) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Derived training curve {value_column} contains non-finite values."
                )

            if value_column.startswith("cumulative_"):
                steps = np.concatenate(([0.0], steps))
                values = np.concatenate(([0.0], values))

            if steps.size:
                seed_curves.append((steps, values))

        if not seed_curves:
            continue

        if len(seed_curves) == 1:
            grid = seed_curves[0][0]
        else:
            start = max(curve[0][0] for curve in seed_curves)
            end = min(curve[0][-1] for curve in seed_curves)
            if end < start:
                continue
            grid = np.asarray(
                sorted(
                    {
                        float(step)
                        for steps, _ in seed_curves
                        for step in steps
                        if start <= step <= end
                    }
                ),
                dtype=np.float64,
            )

        values = []
        for steps, scalar in seed_curves:
            indices = np.searchsorted(steps, grid, side="right") - 1
            if np.any(indices < 0):
                raise ValueError(
                    f"Derived training curve {value_column} requires extrapolation."
                )
            values.append(scalar[indices])

        stacked = np.vstack(values)
        rows.append(
            pd.DataFrame(
                {
                    "training_diagnostics_schema_version": TRAINING_DIAGNOSTICS_SCHEMA,
                    "method": method,
                    "display_name": display_name,
                    "tag": tag,
                    "step": grid,
                    "value_mean": stacked.mean(axis=0),
                    "value_std": (
                        stacked.std(axis=0, ddof=1)
                        if len(stacked) > 1
                        else np.nan
                    ),
                    "seed_count": len(stacked),
                }
            )
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

#} End function aggregate_episode_curve

# end region Seed aggregation


__all__ = [
    "ACTION_BOUND_TAGS",
    "ALL_TRAINING_TAGS",
    "DERIVED_EPISODE_CURVES",
    "EPISODE_REQUIRED_TAGS",
    "PROJECTION_TAGS",
    "TRAINING_CURVES",
    "TRAINING_DIAGNOSTICS_SCHEMA",
    "aggregate_curve",
    "aggregate_episode_curve",
    "load_tensorboard_run",
    "parse_hyperparameters",
    "resolve_recorded_checkpoint",
    "training_episode_diagnostics",
    "training_points",
    "training_rollout_diagnostics",
]
