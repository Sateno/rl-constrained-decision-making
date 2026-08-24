from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.aggregate_projection_results import load_protocol
from analysis.training_diagnostics import (
    DERIVED_EPISODE_CURVES,
    TRAINING_CURVES,
    aggregate_curve,
    aggregate_episode_curve,
    training_episode_diagnostics,
    training_points,
    training_rollout_diagnostics,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

PDF_AUTHOR = "Salvador Tenorio"
PDF_SUBJECT = "Predictive action projection with PPO"
SHARED_TRAINING_TABLE_FILENAMES = (
    "training_scalar_events.csv",
    "training_episode_diagnostics.csv",
    "training_rollout_diagnostics.csv",
    "training_curve_points.csv",
)
BOUNDED_TRAINING_FIGURES = frozenset(
    {
        "training_action_bound_clipping.pdf",
        "training_projection_intervention.pdf",
        "training_rolling_collision_rate.pdf",
        "training_rolling_success_rate.pdf",
    }
)


EVALUATION_PLOTS = (
    ("episode_return", "evaluation_return.pdf", "Mean episode return", False),
    ("success_rate", "evaluation_success_rate.pdf", "Success rate", False),
    ("collision_rate", "evaluation_collision_rate.pdf", "Collision rate", False),
    ("timeout_rate", "evaluation_timeout_rate.pdf", "Timeout rate", False),
    (
        "min_obstacle_clearance",
        "evaluation_min_obstacle_clearance.pdf",
        "Mean minimum obstacle clearance",
        False,
    ),
    (
        "action_bound_clipping_rate",
        "evaluation_action_bound_clipping.pdf",
        "Action-bound clipping rate",
        False,
    ),
    (
        "action_bound_clipping_norm",
        "evaluation_action_bound_clipping_norm.pdf",
        "Mean action-bound clipping norm",
        False,
    ),
    (
        "projection_intervention_rate",
        "evaluation_projection_intervention.pdf",
        "Projection intervention rate",
        True,
    ),
    (
        "projection_correction_norm",
        "evaluation_projection_correction.pdf",
        "Mean projection correction norm",
        True,
    ),
    (
        "projection_correction_norm_max",
        "evaluation_projection_correction_max.pdf",
        "Maximum projection correction norm",
        True,
    ),
    (
        "projection_slack_sum",
        "evaluation_projection_slack.pdf",
        "Mean summed projection slack",
        True,
    ),
    (
        "projection_slack_max",
        "evaluation_projection_slack_max.pdf",
        "Maximum projection slack",
        True,
    ),
)


#################################################################################
# region Basic plotting

# Return a human-readable title for one generated artifact.
def artifact_title(path: Path) -> str:
#{
    return path.stem.replace("_", " ").strip().title()

#} End function artifact_title


# Save without automatic layout engines that can fail inside native libraries.
def save_figure(
    figure: plt.Figure,
    path: Path,
    bottom: float = 0.30,
    *,
    left: float = 0.16,
    right: float = 0.98,
    top: float = 0.90,
    wspace: float | None = None,
    title: str | None = None,
) -> None:
#{
    path.parent.mkdir(parents=True, exist_ok=True)
    adjustments = {"left": left, "right": right, "top": top, "bottom": bottom}

    if wspace is not None:
        adjustments["wspace"] = wspace

    figure.subplots_adjust(**adjustments)

    figure.savefig(
        path,
        metadata={
            "Author": PDF_AUTHOR,
            "Title": title or artifact_title(path),
            "Subject": PDF_SUBJECT,
        },
    )
    plt.close(figure)

#} End function save_figure


# Return one compact display label without rotated tick text.
def method_tick_label(display_name: str) -> str:
#{
    return fill(display_name, width=18)

#} End function method_tick_label


# Describe incompatible training-return scales from checkpoint metadata.
def training_return_caveat(checkpoints: pd.DataFrame) -> str | None:
#{
    required = {"method", "display_name", "training_collision_penalty"}

    if not required.issubset(checkpoints.columns):
        return None

    penalties = checkpoints[
        ["method", "display_name", "training_collision_penalty"]
    ].drop_duplicates()
    penalties["training_collision_penalty"] = pd.to_numeric(
        penalties["training_collision_penalty"],
        errors="coerce",
    )
    penalties = penalties[np.isfinite(penalties["training_collision_penalty"])]

    if penalties.empty or penalties["training_collision_penalty"].nunique() <= 1:
        return None

    per_method = penalties.groupby("method")["training_collision_penalty"].nunique()

    if (per_method > 1).any():
        raise ValueError("A training method has inconsistent collision-penalty metadata.")

    counts = penalties.groupby("training_collision_penalty")["method"].nunique()

    if len(counts) == 2 and int(counts.min()) == 1 and int(counts.max()) > 1:
        outlier_penalty = float(counts.idxmin())
        common_penalty = float(counts.idxmax())
        outlier_name = str(
            penalties.loc[
                penalties["training_collision_penalty"] == outlier_penalty,
                "display_name",
            ].iloc[0]
        )
        return (
            "Not directly comparable: collision penalty "
            f"{outlier_penalty:g} ({outlier_name}) vs {common_penalty:g} (others)"
        )

    return "Not directly comparable: collision penalties differ across methods"

#} End function training_return_caveat


# Plot checkpoint-level evaluation values while preserving off/on pairing.
def plot_evaluation_checkpoints(
    checkpoints: pd.DataFrame,
    metric: str,
    path: Path,
    ylabel: str,
    projection_only: bool,
) -> bool:
#{
    frame = checkpoints.copy()

    if metric == "timeout_rate":
        frame[metric] = 1.0 - frame["success_rate"] - frame["collision_rate"]

    if projection_only:
        frame = frame[frame["projection_mode"] == "enabled"]

    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[np.isfinite(frame[metric])]

    if frame.empty:
        return False

    methods = frame[["method", "display_name"]].drop_duplicates().reset_index(drop=True)
    x = np.arange(len(methods), dtype=float)
    figure, axis = plt.subplots(figsize=(max(6.6, 2.05 * len(methods)), 4.5))

    if projection_only:
        for method_index, method_row in methods.iterrows():
            rows = frame[frame["method"] == method_row["method"]].sort_values("train_seed")
            values = rows[metric].to_numpy(float)
            jitter = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(
                np.full(len(values), x[method_index]) + jitter,
                values,
                color="#0072B2",
                alpha=0.80,
                s=34,
                zorder=3,
                label="Training checkpoint" if method_index == 0 else None,
            )
            axis.scatter(
                [x[method_index]],
                [float(np.mean(values))],
                marker="D",
                color="black",
                s=42,
                zorder=4,
                label="Across-checkpoint mean" if method_index == 0 else None,
            )
    else:
        offsets = {"disabled": -0.14, "enabled": 0.14}
        colors = {"disabled": "#777777", "enabled": "#0072B2"}

        for method_index, method_row in methods.iterrows():
            rows = frame[frame["method"] == method_row["method"]]
            paired = rows.pivot(index="train_seed", columns="projection_mode", values=metric)

            if {"disabled", "enabled"}.issubset(paired.columns):
                paired = paired.dropna(subset=["disabled", "enabled"])

                for values in paired[["disabled", "enabled"]].itertuples(index=False, name=None):
                    axis.plot(
                        [x[method_index] + offsets["disabled"], x[method_index] + offsets["enabled"]],
                        values,
                        color="#B8B8B8",
                        linewidth=1.0,
                        alpha=0.85,
                        zorder=1,
                    )

            for mode in ("disabled", "enabled"):
                values = rows.loc[rows["projection_mode"] == mode, metric].to_numpy(float)

                if not len(values):
                    continue

                position = x[method_index] + offsets[mode]
                axis.scatter(
                    np.full(len(values), position),
                    values,
                    color=colors[mode],
                    alpha=0.82,
                    s=32,
                    zorder=3,
                    label=("Projection off" if mode == "disabled" else "Projection on")
                    if method_index == 0
                    else None,
                )
                axis.scatter(
                    [position],
                    [float(np.mean(values))],
                    marker="D",
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.5,
                    s=48,
                    zorder=4,
                    label="Across-checkpoint mean"
                    if method_index == 0 and mode == "disabled"
                    else None,
                )

    axis.set_xticks(x, [method_tick_label(name) for name in methods["display_name"]])
    axis.set_ylabel(ylabel)
    axis.set_title(ylabel)
    axis.grid(axis="y", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    preferred = (
        ["Training checkpoint", "Across-checkpoint mean"]
        if projection_only
        else ["Projection off", "Projection on", "Across-checkpoint mean"]
    )
    legend_items = {label: handle for handle, label in zip(handles, labels)}
    axis.legend(
        [legend_items[label] for label in preferred if label in legend_items],
        [label for label in preferred if label in legend_items],
        frameon=False,
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
    )

    if metric in {
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "action_bound_clipping_rate",
        "projection_intervention_rate",
    }:
        axis.set_ylim(-0.02, 1.02)

    save_figure(
        figure,
        path,
        bottom=0.30,
        right=0.94 if projection_only else 0.98,
        title=ylabel,
    )
    return True

#} End function plot_evaluation_checkpoints

# end region Basic plotting


#################################################################################
# region TensorBoard curves

# Plot one available training curve.
def plot_curve(
    frame: pd.DataFrame,
    path: Path,
    ylabel: str,
    *,
    caveat: str | None = None,
) -> None:
#{
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    bounded_rate = path.name in BOUNDED_TRAINING_FIGURES

    for (_, display_name), rows in frame.groupby(["method", "display_name"], sort=False):
        steps = rows["step"].to_numpy(float)
        mean = rows["value_mean"].to_numpy(float)
        std = rows["value_std"].to_numpy(float)
        line = axis.plot(steps, mean, label=display_name)[0]
        finite = np.isfinite(std)

        if finite.any():
            lower = mean[finite] - std[finite]
            upper = mean[finite] + std[finite]

            if bounded_rate:
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)

            axis.fill_between(
                steps[finite],
                lower,
                upper,
                color=line.get_color(),
                alpha=0.18,
            )

    axis.set_xlabel("Environment transitions")
    axis.set_ylabel(ylabel)

    if caveat:
        axis.set_title(f"{ylabel}\n{caveat}", fontsize=10)
    else:
        axis.set_title(ylabel)

    if bounded_rate:
        axis.set_ylim(0.0, 1.0)

    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(
        figure,
        path,
        bottom=0.16,
        top=0.86 if caveat else 0.90,
    )

#} End function plot_curve

# end region TensorBoard curves


#################################################################################
# region Trajectories

# Return one scalar from a trajectory archive.
def npz_scalar(archive: np.lib.npyio.NpzFile, key: str) -> object:
#{
    return np.asarray(archive[key]).item()

#} End function npz_scalar


# Index trajectory archives by internal metadata rather than filenames.
def trajectory_index(protocol: dict[str, object], evaluation_dir: Path) -> dict[tuple[str, int, str, str], Path]:
#{
    suite_hash = protocol["_layout_suite"].sha256
    index = {}

    for path in sorted(evaluation_dir.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "trajectory_archive_version",
                    "run_method",
                    "run_train_seed",
                    "run_projection_mode",
                    "run_checkpoint_sha256",
                    "run_layout_suite_sha256",
                }
                if not required.issubset(archive.files):
                    continue
                if npz_scalar(archive, "trajectory_archive_version") != "evaluation_trajectory_v1":
                    continue
                if npz_scalar(archive, "run_layout_suite_sha256") != suite_hash:
                    continue
                key = (
                    str(npz_scalar(archive, "run_method")),
                    int(npz_scalar(archive, "run_train_seed")),
                    str(npz_scalar(archive, "run_projection_mode")),
                    str(npz_scalar(archive, "run_checkpoint_sha256")),
                )
        except Exception:
            continue

        if key in index:
            raise ValueError(f"Duplicate trajectory metadata key {key}: {index[key]} and {path}")
        index[key] = path

    return index

#} End function trajectory_index


# Find one episode key by the CSV episode index and evaluation seed.
def episode_key(archive: np.lib.npyio.NpzFile, episode: int, seed: int) -> str:
#{
    for key in archive["episode_keys"].astype(str):
        if int(npz_scalar(archive, f"{key}_episode")) == episode and int(npz_scalar(archive, f"{key}_seed")) == seed:
            return key

    raise ValueError(f"Trajectory episode={episode}, seed={seed} was not found.")

#} End function episode_key


# Return the exhaustive outcome label for one evaluation row.
def episode_outcome(row: pd.Series) -> str:
#{
    required = {"success", "collision", "terminated", "truncated"}

    if not required.issubset(row.index):
        missing = sorted(required.difference(row.index))
        raise ValueError(f"Trajectory outcome row is missing fields: {missing}")

    success = bool(row["success"])
    collision = bool(row["collision"])
    terminated = bool(row["terminated"])
    truncated = bool(row["truncated"])

    if success and not collision and terminated and not truncated:
        return "success"
    if collision and not success and terminated and not truncated:
        return "collision"
    if truncated and not success and not collision and not terminated:
        return "timeout"

    raise ValueError(
        "Trajectory outcome is not exactly one valid success, collision, or timeout: "
        f"success={success}, collision={collision}, terminated={terminated}, truncated={truncated}"
    )

#} End function episode_outcome


# Return a human-readable label for one stored layout identifier.
def layout_display_name(layout_id: str) -> str:
#{
    return layout_id.replace("_", " ").strip().title()

#} End function layout_display_name


# Plot one protocol-selected layout for the lowest training seed of each method/mode.
def plot_trajectories(
    protocol: dict[str, object],
    episodes: pd.DataFrame,
    evaluation_dir: Path,
    figure_path: Path,
    selection_path: Path,
    require_complete: bool = False,
) -> bool:
#{
    index = trajectory_index(protocol, evaluation_dir)
    selected = []
    trajectories = []
    missing = []
    seed = min(protocol["expected_train_seeds"])

    for method in protocol["methods"]:
        for mode in method["required_projection_modes"]:
            rows = episodes[
                (episodes["method"] == method["method"])
                & (episodes["train_seed"] == seed)
                & (episodes["projection_mode"] == mode)
                & (episodes["layout_id"] == protocol["representative_layout_id"])
                & (episodes["layout_repeat"] == 0)
            ]

            if len(rows) != 1:
                missing.append(
                    f"{method['method']}, seed {seed}, {mode}: expected one evaluation row"
                )
                continue

            row = rows.iloc[0]
            key = (method["method"], seed, mode, str(row["checkpoint_sha256"]))
            archive_path = index.get(key)

            if archive_path is None:
                missing.append(
                    f"{method['method']}, seed {seed}, {mode}: trajectory archive not found"
                )
                continue

            with np.load(archive_path, allow_pickle=False) as archive:
                item_key = episode_key(archive, int(row["episode"]), int(row["evaluation_seed"]))
                trajectories.append(
                    {
                        "method": method["method"],
                        "display_name": method["display_name"],
                        "projection_mode": mode,
                        "outcome": episode_outcome(row),
                        "episode_length": int(row["episode_length"]),
                        "intervention_rate": float(row["projection_intervention_rate"]),
                        "positions": np.asarray(archive[f"{item_key}_positions"], dtype=float),
                        "goal": np.asarray(archive[f"{item_key}_goal"], dtype=float),
                        "centers": np.asarray(archive[f"{item_key}_obstacle_centers"], dtype=float),
                        "radii": np.asarray(archive[f"{item_key}_obstacle_radii"], dtype=float),
                        "mask": np.asarray(archive[f"{item_key}_obstacle_mask"], dtype=bool),
                    }
                )

            selected.append(
                {
                    "method": method["method"],
                    "train_seed": seed,
                    "projection_mode": mode,
                    "layout_id": protocol["representative_layout_id"],
                    "evaluation_seed": int(row["evaluation_seed"]),
                    "checkpoint_sha256": str(row["checkpoint_sha256"]),
                    "trajectory_archive": str(archive_path),
                    "episode": int(row["episode"]),
                }
            )

    if missing and require_complete:
        raise ValueError(
            "Required representative trajectories are incomplete: " + "; ".join(missing)
        )
    if not trajectories:
        return False

    pd.DataFrame(selected).to_csv(selection_path, index=False)
    methods = [
        method
        for method in protocol["methods"]
        if any(item["method"] == method["method"] for item in trajectories)
    ]
    figure, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4.2 * len(methods), 5.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    path_styles = {
        "disabled": {"color": "#D55E00", "linestyle": "--", "linewidth": 2.0},
        "enabled": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.7},
    }
    outcome_markers = {"success": "P", "collision": "X", "timeout": "s"}

    for method_index, method in enumerate(methods):
        axis = axes[0, method_index]
        items = [item for item in trajectories if item["method"] == method["method"]]
        first = items[0]

        for center, radius, active in zip(first["centers"], first["radii"], first["mask"]):
            if active:
                axis.add_patch(
                    plt.Circle(center, radius, fill=False, linewidth=1.3, color="black")
                )

        axis.scatter(
            [first["positions"][0, 0]],
            [first["positions"][0, 1]],
            marker="o",
            s=45,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
        axis.scatter(
            [first["goal"][0]],
            [first["goal"][1]],
            marker="*",
            s=115,
            color="#E69F00",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )

        # Draw enabled first so an exactly coincident dashed disabled path remains visible.
        for item in sorted(items, key=lambda value: value["projection_mode"] == "disabled"):
            mode = item["projection_mode"]
            style = path_styles[mode]
            positions = item["positions"]
            axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=2 if mode == "enabled" else 3,
            )
            endpoint = positions[-1]
            axis.scatter(
                [endpoint[0]],
                [endpoint[1]],
                marker=outcome_markers[item["outcome"]],
                s=72,
                color=style["color"],
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )

        axis.set_xlabel("x position")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.22)

        by_mode = {item["projection_mode"]: item for item in items}
        title_lines = [method["display_name"]]

        for mode, prefix in (("disabled", "Off"), ("enabled", "On")):
            item = by_mode.get(mode)

            if item is None:
                continue

            summary = f"{prefix}: {item['outcome']}, {item['episode_length']} steps"

            if mode == "enabled" and np.isfinite(item["intervention_rate"]):
                summary += f", intervention {item['intervention_rate']:.1%}"

            title_lines.append(summary)

        if len(items) == 2:
            positions_a = items[0]["positions"]
            positions_b = items[1]["positions"]

            if positions_a.shape == positions_b.shape and np.allclose(positions_a, positions_b):
                title_lines.append("Off/on paths coincide")

        axis.set_title("\n".join(title_lines), fontsize=9.5)

        if method_index == 0:
            axis.set_ylabel("y position")

    layout_name = layout_display_name(str(protocol["representative_layout_id"]))
    figure.suptitle(
        f"Prespecified illustrative trajectories — {layout_name}",
        fontsize=13,
        y=0.95,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=path_styles["disabled"]["color"],
                linestyle="--",
                linewidth=2.0,
                label="Projection off",
            ),
            Line2D(
                [0],
                [0],
                color=path_styles["enabled"]["color"],
                linestyle="-",
                linewidth=1.7,
                label="Projection on",
            ),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="white", linestyle="None", label="Start"),
            Line2D(
                [0],
                [0],
                marker="*",
                color="#E69F00",
                markeredgecolor="black",
                linestyle="None",
                markersize=10,
                label="Goal",
            ),
            Line2D(
                [0],
                [0],
                marker=outcome_markers["success"],
                color="black",
                linestyle="None",
                label="Success endpoint",
            ),
            Line2D(
                [0],
                [0],
                marker=outcome_markers["collision"],
                color="black",
                linestyle="None",
                label="Collision endpoint",
            ),
            Line2D(
                [0],
                [0],
                marker=outcome_markers["timeout"],
                color="black",
                linestyle="None",
                label="Timeout endpoint",
            ),
        ],
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=8,
    )
    save_figure(
        figure,
        figure_path,
        bottom=0.16,
        left=0.07,
        right=0.99,
        top=0.70,
        wspace=0.18,
        title=f"Prespecified illustrative trajectories: {layout_name}",
    )
    return True

#} End function plot_trajectories

# end region Trajectories


#################################################################################
# region Build and command line

# Generate all available figures from audited tables and discoverable artifacts.
def build_result_figures(
    protocol_path: str | Path,
    tables_dir: str | Path,
    evaluation_dir: str | Path,
    figures_dir: str | Path,
    runs_dir: str | Path | None = None,
    *,
    include_training_diagnostics: bool = True,
) -> dict[str, Path]:
#{
    protocol = load_protocol(protocol_path)
    tables = Path(tables_dir)
    evaluation = Path(evaluation_dir).resolve()
    figures = Path(figures_dir)

    if figures.exists() and (not figures.is_dir() or any(figures.iterdir())):
        raise FileExistsError(
            f"Result figure directory already exists and is not empty: {figures}"
        )

    figures.mkdir(parents=True, exist_ok=True)
    require_complete = bool(protocol["require_complete_artifacts"])
    checkpoint_summary = pd.read_csv(tables / "checkpoint_summary.csv")
    episodes = pd.read_csv(tables / "evaluation_episode_results.csv")
    generated = {}
    skipped = []

    if not include_training_diagnostics:
        stale_training_tables = [
            tables / filename
            for filename in SHARED_TRAINING_TABLE_FILENAMES
            if (tables / filename).exists()
        ]

        if stale_training_tables:
            joined = ", ".join(str(path) for path in stale_training_tables)
            raise FileExistsError(
                "Training diagnostics were explicitly disabled, but stale generated training "
                f"tables remain: {joined}"
            )

    for metric, filename, ylabel, projection_only in EVALUATION_PLOTS:
        path = figures / filename
        if plot_evaluation_checkpoints(checkpoint_summary, metric, path, ylabel, projection_only):
            generated[filename] = path
        else:
            skipped.append({"figure": filename, "reason": "no_finite_values"})

    if include_training_diagnostics and require_complete and runs_dir is None:
        raise ValueError("The final analysis protocol requires a runs directory.")

    training_diagnostics_counts = {
        "included": False,
        "reason": "disabled_by_command" if not include_training_diagnostics else "runs_dir_unavailable",
    }

    if include_training_diagnostics and runs_dir is not None:
        points, missing = training_points(
            checkpoint_summary,
            Path(runs_dir).resolve(),
            require_complete=require_complete,
        )
        skipped.extend(missing)
        rolling_window = int(protocol["training_episode_rolling_window"])
        episode_diagnostics = training_episode_diagnostics(
            points,
            rolling_window=rolling_window,
            require_complete=require_complete,
        )
        rollout_diagnostics = training_rollout_diagnostics(
            points,
            require_complete=require_complete,
        )

        if require_complete and points.empty:
            raise ValueError("Required training scalar events are unavailable.")
        if require_complete and episode_diagnostics.empty:
            raise ValueError("Required training episode diagnostics are unavailable.")
        if require_complete and rollout_diagnostics.empty:
            raise ValueError("Required training rollout diagnostics are unavailable.")

        training_tables = (
            ("training_scalar_events.csv", points),
            ("training_episode_diagnostics.csv", episode_diagnostics),
            ("training_rollout_diagnostics.csv", rollout_diagnostics),
        )

        for filename, frame in training_tables:
            if frame.empty:
                skipped.append({"table": filename, "reason": "training_data_unavailable"})
                continue
            path = tables / filename
            frame.to_csv(path, index=False)
            generated[filename] = path

        curve_frames = []

        return_caveat = training_return_caveat(checkpoint_summary)

        for tag, filename, ylabel in TRAINING_CURVES:
            curve = aggregate_curve(points, tag)
            if curve.empty:
                if require_complete:
                    raise ValueError(f"Required training curve could not be aggregated: {tag}")
                skipped.append({"figure": filename, "reason": "scalar_unavailable"})
                continue
            path = figures / filename
            plot_curve(
                curve,
                path,
                ylabel,
                caveat=return_caveat if filename == "training_return.pdf" else None,
            )
            generated[filename] = path
            curve_frames.append(curve)

        for value_column, filename, ylabel in DERIVED_EPISODE_CURVES:
            curve = aggregate_episode_curve(episode_diagnostics, value_column)
            if curve.empty:
                if require_complete:
                    raise ValueError(
                        f"Required derived training curve could not be aggregated: {value_column}"
                    )
                skipped.append({"figure": filename, "reason": "episode_data_unavailable"})
                continue
            path = figures / filename
            plot_curve(curve, path, ylabel)
            generated[filename] = path
            curve_frames.append(curve)

        if curve_frames:
            path = tables / "training_curve_points.csv"
            pd.concat(curve_frames, ignore_index=True).to_csv(path, index=False)
            generated[path.name] = path

        training_diagnostics_counts = {
            "included": True,
            "scalar_event_rows": int(len(points)),
            "episode_rows": int(len(episode_diagnostics)),
            "rollout_rows": int(len(rollout_diagnostics)),
            "rolling_window_episodes": rolling_window,
        }

    selection = tables / "representative_trajectory_selection.csv"
    trajectory_figure = figures / "representative_trajectories.pdf"

    if plot_trajectories(
        protocol,
        episodes,
        evaluation,
        trajectory_figure,
        selection,
        require_complete=require_complete,
    ):
        generated[trajectory_figure.name] = trajectory_figure
        generated[selection.name] = selection
    else:
        skipped.append({"figure": trajectory_figure.name, "reason": "matching_trajectory_archives_unavailable"})

    audit_path = figures / "figure_build_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "study_id": protocol["study_id"],
                "protocol_path": str(protocol["_source_path"]),
                "protocol_sha256": protocol["_sha256"],
                "evaluation_dir": str(evaluation),
                "runs_dir": str(Path(runs_dir).resolve()) if runs_dir is not None else None,
                "training_diagnostics_included": bool(include_training_diagnostics),
                "artifact_scope": (
                    "evaluation_and_training"
                    if include_training_diagnostics
                    else "evaluation_only"
                ),
                "intentional_omissions": (
                    []
                    if include_training_diagnostics
                    else [
                        {
                            "category": "shared_training_diagnostics",
                            "reason": "disabled_by_command",
                        }
                    ]
                ),
                "training_diagnostics": training_diagnostics_counts,
                "generated": {name: str(path) for name, path in generated.items()},
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    generated[audit_path.name] = audit_path
    return generated

#} End function build_result_figures


# Run one saved-result figure build.
def main() -> None:
#{
    parser = argparse.ArgumentParser(description="Generate figures from audited saved results.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-training-diagnostics",
        action="store_true",
        help="Generate only evaluation and trajectory outputs for this result suite.",
    )
    args = parser.parse_args()
    outputs = build_result_figures(
        args.protocol,
        args.tables_dir,
        args.evaluation_dir,
        args.figures_dir,
        args.runs_dir,
        include_training_diagnostics=not args.skip_training_diagnostics,
    )
    print("Result figure generation completed successfully.")

    for name, path in outputs.items():
        print(f"{name}: {path}")

#} End function main

# end region Build and command line


if __name__ == "__main__":
    main()
