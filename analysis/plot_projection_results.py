from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.aggregate_projection_results import load_protocol


TRAINING_TAGS = (
    ("charts/episodic_return", "training_return.pdf", "Episode return"),
    (
        "projection/intervention_frequency",
        "training_projection_intervention.pdf",
        "Projection intervention frequency",
    ),
    (
        "projection/correction_norm",
        "training_projection_correction.pdf",
        "Projection correction norm",
    ),
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
        "projection_slack_sum",
        "evaluation_projection_slack.pdf",
        "Mean summed projection slack",
        True,
    ),
)


#################################################################################
# region Basic plotting

# Return one file SHA-256.
def file_sha256(path: Path) -> str:
#{
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

#} End function file_sha256


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

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0, "tensors": 1})
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

    for tag, _, _ in TRAINING_TAGS:
        if tag not in tags.get("scalars", []):
            continue

        events = accumulator.Scalars(tag)
        scalars[tag] = pd.DataFrame(
            {
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


# Discover TensorBoard runs by the checkpoint path recorded in their own metadata.
def training_points(
    checkpoint_summary: pd.DataFrame,
    runs_dir: Path,
    require_complete: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
#{
    unique = checkpoint_summary[
        ["method", "display_name", "train_seed", "checkpoint_sha256"]
    ].drop_duplicates()
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
        match = run_index.get(str(row.checkpoint_sha256))

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

        if recorded_seed and int(recorded_seed) != int(row.train_seed):
            raise ValueError(
                f"TensorBoard seed {recorded_seed} does not match evaluation seed "
                f"{row.train_seed} for {run_dir}."
            )

        if recorded_method and recorded_method != row.method:
            raise ValueError(
                f"TensorBoard method {recorded_method!r} does not match evaluation method "
                f"{row.method!r} for {run_dir}."
            )

        for tag, _, _ in TRAINING_TAGS:
            frame = scalars.get(tag, pd.DataFrame())

            if frame.empty:
                item = {
                    "method": row.method,
                    "train_seed": int(row.train_seed),
                    "tag": tag,
                    "reason": "scalar_not_found",
                }
                if require_complete and tag == "charts/episodic_return":
                    raise ValueError(
                        "Required charts/episodic_return scalar was not found for "
                        f"{row.method}, seed {int(row.train_seed)}."
                    )
                skipped.append(item)
                continue

            frame = frame.copy()
            frame["method"] = row.method
            frame["display_name"] = row.display_name
            frame["train_seed"] = int(row.train_seed)
            frame["tag"] = tag
            frames.append(frame)

    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), skipped

#} End function training_points


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
            seed_rows = seed_rows.groupby("step", as_index=False)["value"].mean().sort_values("step")
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
            grid = np.linspace(start, end, min(200, max(len(curve[0]) for curve in seed_curves)))

        values = np.vstack([np.interp(grid, steps, scalar) for steps, scalar in seed_curves])
        rows.append(
            pd.DataFrame(
                {
                    "method": method,
                    "display_name": display_name,
                    "tag": tag,
                    "step": grid,
                    "value_mean": values.mean(axis=0),
                    "value_std": values.std(axis=0, ddof=1) if len(values) > 1 else np.nan,
                    "seed_count": len(values),
                }
            )
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

#} End function aggregate_curve


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

    if runs_dir is not None:
        points, missing = training_points(
            checkpoint_summary,
            Path(runs_dir).resolve(),
            require_complete=require_complete,
        )
        skipped.extend(missing)
        curve_frames = []

        for tag, filename, ylabel in TRAINING_TAGS:
            curve = aggregate_curve(points, tag)
            if curve.empty:
                if require_complete and tag == "charts/episodic_return":
                    raise ValueError("Required training-return curve could not be aggregated.")
                skipped.append({"figure": filename, "reason": "scalar_unavailable"})
                continue
            path = figures / filename
            plot_curve(curve, path, ylabel)
            generated[filename] = path
            curve_frames.append(curve)

        if curve_frames:
            path = tables / "training_curve_points.csv"
            pd.concat(curve_frames, ignore_index=True).to_csv(path, index=False)
            generated[path.name] = path

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
