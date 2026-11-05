"""Verify the published trajectory bundle without running a policy or simulator.

Only NumPy and the Python standard library are required. Historical endpoint
outcomes are checked against the frozen tables; this is not a swept-path audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = "results/trajectories/pap_evaluation_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data, *, text=False):
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def safe_path(root, relative):
    """Allow only canonical relative POSIX paths contained in the chosen root."""
    require(isinstance(relative, str) and bool(relative), "Invalid bundle path")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative
            and ":" not in relative and path.as_posix() == relative,
            f"Invalid bundle path: {relative}")
    destination = root.joinpath(*path.parts)
    require(destination.resolve().is_relative_to(root.resolve()),
            f"Bundle path escapes its root: {relative}")
    return destination


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def scalar(value, name):
    value = np.asarray(value)
    require(value.shape == (), f"Expected scalar: {name}")
    return value.item()


def boolean(text):
    require(text in ("True", "False"), f"Invalid table Boolean: {text}")
    return text == "True"


def validate_episode(data, row, mode, label):
    """Validate one saved episode; return its OFF/ON initial-condition signature."""
    n = int(row["episode_length"])
    require(n > 0, f"Invalid episode length: {label}")
    for name, shape in (("positions", (n + 1, 2)), ("headings", (n + 1,)),
                        ("state_step_count", (n + 1,)),
                        ("state_distance_to_goal", (n + 1,)),
                        ("action_raw_normalized", (n, 2)),
                        ("action_raw_physical", (n, 2)),
                        ("action_exec_physical", (n, 2)),
                        ("action_correction_physical", (n, 2)), ("rewards", (n,)),
                        ("projection_correction_norm", (n,)),
                        ("projection_slack_sum", (n,)),
                        ("projection_slack_max", (n,)),
                        ("projection_active_constraint_count", (n,))):
        require(data[name].shape == shape and np.isfinite(data[name]).all(),
                f"Invalid shape or nonfinite {name}: {label}")
    for name in ("state_success", "state_collision", "terminated", "truncated",
                 "projection_enabled", "projection_intervened", "projection_success"):
        shape = (n + 1,) if name.startswith("state_") else (n,)
        require(data[name].shape == shape and data[name].dtype.kind == "b",
                f"Invalid Boolean array {name}: {label}")
    require(data["projection_solver_status"].shape == (n,),
            f"Misaligned solver status: {label}")
    goal, centers = data["goal"], data["obstacle_centers"]
    require(goal.shape == (2,) and np.isfinite(goal).all(), f"Invalid goal: {label}")
    require(centers.ndim == 2 and centers.shape[1] == 2 and np.isfinite(centers).all(),
            f"Invalid obstacle centers: {label}")
    k = len(centers)
    require(data["obstacle_radii"].shape == (k,) and np.isfinite(data["obstacle_radii"]).all()
            and (data["obstacle_radii"] >= 0).all(), f"Invalid obstacle radii: {label}")
    require(data["obstacle_mask"].shape == (k,) and data["obstacle_mask"].dtype.kind == "b",
            f"Invalid obstacle mask: {label}")
    require(data["projection_slack_values"].shape == (n, k)
            and np.isfinite(data["projection_slack_values"]).all()
            and (data["projection_slack_values"] >= 0).all(), f"Invalid slack: {label}")
    clearance = data["state_min_obstacle_clearance"]
    require(clearance.shape == (n + 1,), f"Misaligned clearance: {label}")
    require(np.isfinite(clearance).all() if data["obstacle_mask"].any()
            else np.isnan(clearance).all(), f"Invalid recorded clearance: {label}")
    for name in ("agent_radius", "dt", "v_max", "omega_max"):
        value = float(scalar(data[name], name))
        require(np.isfinite(value) and value > 0, f"Invalid {name}: {label}")
    require(int(scalar(data["episode"], "episode")) == int(row["episode"]),
            f"Episode identity differs: {label}")
    require(int(scalar(data["seed"], "seed")) == int(row["evaluation_seed"]),
            f"Evaluation seed differs: {label}")
    require(str(scalar(data["policy"], "policy")) == row["policy"],
            f"Policy identity differs: {label}")
    require(np.array_equal(data["state_step_count"], np.arange(n + 1)),
            f"State step counts differ: {label}")
    require(np.all(data["projection_enabled"] == (mode == "enabled")),
            f"Projection mode differs: {label}")
    require(bool(data["projection_success"].all()), f"Recorded projection failure: {label}")
    require(np.allclose(data["action_exec_physical"] - data["action_raw_physical"],
                        data["action_correction_physical"], atol=1e-12, rtol=0),
            f"Action correction differs: {label}")
    if mode == "disabled":
        require(np.array_equal(data["action_raw_physical"], data["action_exec_physical"]),
                f"Projection-OFF action differs: {label}")
    require(np.isclose(data["rewards"].sum(), float(row["episode_return"]), atol=1e-10, rtol=0),
            f"Episode return differs: {label}")
    for name, column in (("state_success", "success"), ("state_collision", "collision"),
                         ("terminated", "terminated"), ("truncated", "truncated")):
        require(bool(data[name][-1]) == boolean(row[column]), f"Terminal {column} differs: {label}")
        require(not bool(data[name][:-1].any()), f"Early terminal flag in {name}: {label}")
    success, collision = bool(data["state_success"][-1]), bool(data["state_collision"][-1])
    terminated, truncated = bool(data["terminated"][-1]), bool(data["truncated"][-1])
    require(not (success and collision) and terminated == (success or collision)
            and terminated != truncated, f"Inconsistent terminal outcome: {label}")
    require(np.isclose(data["state_distance_to_goal"][-1], float(row["final_distance_to_goal"]),
                       atol=1e-12, rtol=0), f"Final goal distance differs: {label}")
    # Hash only paired initial conditions. OFF and ON paths are expected to diverge.
    fields = {name: data[name] for name in ("goal", "obstacle_centers", "obstacle_radii",
                                           "obstacle_mask", "agent_radius", "dt", "v_max", "omega_max", "seed")}
    fields.update(position=data["positions"][0], heading=data["headings"][0])
    return {name: np.asarray(value).copy() for name, value in fields.items()}


def verify_trajectory_bundle(root, bundle_relative=DEFAULT_BUNDLE):
    """Check every published archive and episode; return counts on success."""
    root = Path(root).resolve()
    bundle = safe_path(root, bundle_relative)
    manifest = read_json(bundle / "manifest.json")
    require(manifest.get("schema_version") == "pap_trajectory_bundle_v1", "Unsupported bundle schema")
    require(manifest.get("dataset_id") == "pap_evaluation_v1", "Unexpected published dataset")
    require(manifest.get("trajectory_archive_version") == "evaluation_trajectory_v1",
            "Unsupported trajectory schema")
    archives, tables = manifest["archives"], manifest["tables"]
    require(isinstance(archives, dict) and archives and isinstance(tables, dict) and tables,
            "Empty or invalid bundle inventory")
    original = read_json(root / "verification/repository_manifest.json")["local_archive_files"]
    prefix = "runs/evaluation/final/"
    expected_sources = {name for name in original if name.startswith(prefix)
                        and name.endswith("_trajectories.npz")}
    actual_files = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*.npz")}
    require(actual_files == set(archives), "Bundle archive inventory differs (missing or extra NPZ)")
    rows_by_archive = {}
    all_rows = {}
    for suite, item in sorted(tables.items()):
        require(re.fullmatch(r"[A-Za-z0-9_-]+", suite) is not None, "Invalid suite name")
        expected_table = f"results/tables/{suite}/evaluation_episode_results.csv"
        require(item["path"] == expected_table, "Unexpected bundle table path")
        table = safe_path(root, item["path"])
        require(digest(table.read_bytes(), text=True) == item["sha256"], f"Table hash differs: {suite}")
        with table.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                method, seed, episode, mode = row["method"], int(row["train_seed"]), int(row["episode"]), row["projection_mode"]
                require(re.fullmatch(r"[A-Za-z0-9_-]+", method) is not None and seed >= 0 and episode >= 0
                        and mode in ("disabled", "enabled"), f"Invalid table identity: {suite}")
                key = (suite, method, seed, episode, mode)
                require(key not in all_rows, f"Duplicate episode row: {key}")
                all_rows[key] = row
                relative = f"{suite}/{method}/seed{seed}/projection_{mode}_trajectories.npz"
                rows_by_archive.setdefault(relative, {})[episode] = row
    require(set(rows_by_archive) == set(archives), "Table/archive coverage differs")
    sources = set()
    signatures = {}
    for relative, item in sorted(archives.items()):
        path = safe_path(bundle, relative)
        source = prefix + relative
        require(item["source_archive"] == source and source in original, f"Original archive identity differs: {relative}")
        sources.add(source)
        require(original[source]["text"] is False and original[source]["sha256"] == item["sha256"],
                f"Frozen archive provenance differs: {relative}")
        payload = path.read_bytes()
        require(len(payload) == item["size_bytes"] and digest(payload) == item["sha256"],
                f"Archive size/hash differs: {relative}")
        del payload
        rows = rows_by_archive[relative]
        require(type(item["episodes"]) is int and item["episodes"] == len(rows),
                f"Archive episode count differs: {relative}")
        suite, method, seed_component, filename = PurePosixPath(relative).parts
        seed, mode = int(seed_component[4:]), filename[len("projection_"):-len("_trajectories.npz")]
        with np.load(path, allow_pickle=False) as archive:
            require(str(scalar(archive["trajectory_archive_version"], "schema")) == manifest["trajectory_archive_version"],
                    f"Archive schema differs: {relative}")
            keys = archive["episode_keys"].tolist()
            require(isinstance(keys, list) and len(keys) == len(set(keys))
                    and set(keys) == {f"episode_{episode:04d}" for episode in rows},
                    f"Archive episode keys differ: {relative}")
            actual_keys = {match.group(1) for name in archive.files
                           if (match := re.match(r"^(episode_\d+)_", name))}
            require(actual_keys == set(keys), f"Unlisted episode payload: {relative}")
            metadata = {name: scalar(archive[name], name) for name in (
                "run_checkpoint_sha256", "run_layout_suite_sha256", "run_method", "run_train_seed",
                "run_projection_mode", "run_stochastic", "run_goal_radius")}
            require(metadata["run_method"] == method and int(metadata["run_train_seed"]) == seed
                    and metadata["run_projection_mode"] == mode, f"Run identity differs: {relative}")
            require(isinstance(metadata["run_stochastic"], bool), f"Invalid run action mode: {relative}")
            for episode, row in sorted(rows.items()):
                label = f"{relative}:episode_{episode:04d}"
                require(metadata["run_checkpoint_sha256"] == row["checkpoint_sha256"], f"Checkpoint hash differs: {label}")
                require(metadata["run_layout_suite_sha256"] == row["layout_suite_sha256"], f"Layout hash differs: {label}")
                require(row["evaluation_policy_mode"] in ("stochastic", "deterministic")
                        and metadata["run_stochastic"] == (row["evaluation_policy_mode"] == "stochastic"),
                        f"Action mode differs: {label}")
                require(float(metadata["run_goal_radius"]) == float(row["layout_goal_radius"]),
                        f"Goal radius differs: {label}")
                episode_prefix = f"episode_{episode:04d}_"
                data = {name[len(episode_prefix):]: archive[name] for name in archive.files if name.startswith(episode_prefix)}
                signatures[(suite, method, seed, episode, mode)] = validate_episode(data, row, mode, label)
    require(sources == expected_sources, "Bundle does not contain the complete frozen final trajectory inventory")
    for key, row in all_rows.items():
        other_key = (*key[:-1], "enabled" if key[-1] == "disabled" else "disabled")
        require(other_key in all_rows, f"Missing paired row: {key}")
        other = all_rows[other_key]
        for field in ("checkpoint_sha256", "evaluation_seed", "layout_id", "layout_repeat", "evaluation_policy_mode", "layout_suite_sha256"):
            require(row[field] == other[field], f"Paired table {field} differs: {key}")
        for field, value in signatures[key].items():
            require(np.array_equal(value, signatures[other_key][field]), f"Paired initial condition {field} differs: {key}")
    counts = {"archives": len(archives), "episodes": len(all_rows), "pairs": len(all_rows) // 2}
    require(manifest["counts"] == counts, "Declared bundle counts differ")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        counts = verify_trajectory_bundle(args.repo_root)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        print(f"TRAJECTORY_BUNDLE FAIL: {error}")
        return 1
    print(f"TRAJECTORY_BUNDLE PASS: {counts['archives']} unchanged archives, "
          f"{counts['episodes']} episodes, {counts['pairs']} OFF/ON pairs; "
          "hashes, frozen provenance, tables and trajectory alignment verified (read-only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
