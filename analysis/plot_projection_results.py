from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


EVALUATION_PLOTS = (
    ("episode_return", "evaluation_return.pdf", "Mean episode return", False),
    ("success_rate", "evaluation_success_rate.pdf", "Success rate", False),
    ("collision_rate", "evaluation_collision_rate.pdf", "Collision rate", False),
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

# Save without automatic layout engines that can fail inside native libraries.
def save_figure(figure: plt.Figure, path: Path, bottom: float = 0.30) -> None:
#{
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=bottom)

    figure.savefig(path)
    plt.close(figure)

#} End function save_figure


# Plot one method-level evaluation metric.
def plot_evaluation_bar(summary: pd.DataFrame, metric: str, path: Path, ylabel: str, projection_only: bool) -> bool:
#{
    frame = summary.copy()

    if projection_only:
        frame = frame[frame["projection_mode"] == "enabled"]

    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    frame = frame[np.isfinite(pd.to_numeric(frame[mean_column], errors="coerce"))]

    if frame.empty:
        return False

    labels = [
        f"{row.display_name}\n{'projection on' if row.projection_mode == 'enabled' else 'projection off'}"
        for row in frame.itertuples(index=False)
    ]
    values = pd.to_numeric(frame[mean_column]).to_numpy(float)
    errors = pd.to_numeric(frame[std_column], errors="coerce").fillna(0.0).to_numpy(float)
    x = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(max(6.0, 1.45 * len(frame)), 4.4))
    axis.bar(x, values, yerr=errors, capsize=4)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(ylabel)
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, path, bottom=0.34)
    return True

#} End function plot_evaluation_bar

# end region Basic plotting


#################################################################################
# region TensorBoard curves

# Plot one available training curve.
def plot_curve(frame: pd.DataFrame, path: Path, ylabel: str) -> None:
#{
    figure, axis = plt.subplots(figsize=(7.0, 4.4))

    for (_, display_name), rows in frame.groupby(["method", "display_name"], sort=False):
        steps = rows["step"].to_numpy(float)
        mean = rows["value_mean"].to_numpy(float)
        std = rows["value_std"].to_numpy(float)
        line = axis.plot(steps, mean, label=display_name)[0]
        finite = np.isfinite(std)

        if finite.any():
            axis.fill_between(
                steps[finite],
                mean[finite] - std[finite],
                mean[finite] + std[finite],
                color=line.get_color(),
                alpha=0.18,
            )

    axis.set_xlabel("Environment transitions")
    axis.set_ylabel(ylabel)
    axis.set_title(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, path, bottom=0.16)

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
                        "label": f"{method['display_name']} — {'projection on' if mode == 'enabled' else 'projection off'}",
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
    figure, axis = plt.subplots(figsize=(7.0, 5.0))

    for trajectory in trajectories:
        positions = trajectory["positions"]
        axis.plot(positions[:, 0], positions[:, 1], label=trajectory["label"])

    first = trajectories[0]
    for center, radius, active in zip(first["centers"], first["radii"], first["mask"]):
        if active:
            axis.add_patch(plt.Circle(center, radius, fill=False, linewidth=1.5))

    axis.scatter([first["goal"][0]], [first["goal"][1]], marker="*", s=120, label="Goal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"Representative layout: {protocol['representative_layout_id']}")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    save_figure(figure, figure_path, bottom=0.16)
    return True

#} End function plot_trajectories

# end region Trajectories


#################################################################################
# region Build and command line

# Generate all available figures from audited tables and discoverable artifacts.
def build_result_figures(protocol_path: str | Path, tables_dir: str | Path, evaluation_dir: str | Path, figures_dir: str | Path, runs_dir: str | Path | None = None) -> dict[str, Path]:
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
    method_summary = pd.read_csv(tables / "method_summary.csv")
    checkpoint_summary = pd.read_csv(tables / "checkpoint_summary.csv")
    episodes = pd.read_csv(tables / "evaluation_episode_results.csv")
    generated = {}
    skipped = []

    for metric, filename, ylabel, projection_only in EVALUATION_PLOTS:
        path = figures / filename
        if plot_evaluation_bar(method_summary, metric, path, ylabel, projection_only):
            generated[filename] = path
        else:
            skipped.append({"figure": filename, "reason": "no_finite_values"})

    if require_complete and runs_dir is None:
        raise ValueError("The final analysis protocol requires a runs directory.")

    training_diagnostics_counts = None

    if runs_dir is not None:
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

        for tag, filename, ylabel in TRAINING_CURVES:
            curve = aggregate_curve(points, tag)
            if curve.empty:
                if require_complete:
                    raise ValueError(f"Required training curve could not be aggregated: {tag}")
                skipped.append({"figure": filename, "reason": "scalar_unavailable"})
                continue
            path = figures / filename
            plot_curve(curve, path, ylabel)
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
    args = parser.parse_args()
    outputs = build_result_figures(
        args.protocol,
        args.tables_dir,
        args.evaluation_dir,
        args.figures_dir,
        args.runs_dir,
    )
    print("Result figure generation completed successfully.")

    for name, path in outputs.items():
        print(f"{name}: {path}")

#} End function main

# end region Build and command line


if __name__ == "__main__":
    main()
