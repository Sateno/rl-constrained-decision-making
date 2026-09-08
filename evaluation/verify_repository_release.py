"""Verify the completed release using saved files only; never execute a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "verification/repository_manifest.json"
EXTERNAL_MANUALS = frozenset({
    "docs/guides/Orientation_Guide_to_Predictive_Action_Projection_with_PPO.pdf",
    "docs/guides/Predictive_Action_Projection_Software_Companion.pdf",
    "docs/design/predictive_action_projection_implementation_design.pdf",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes, *, text: bool) -> str:
    """Only newline normalization is permitted for declared text artifacts."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def local_recorded_path(root: Path, value: str) -> Path:
    """Relocate historical Windows/POSIX run paths without guessing filenames."""
    parts = value.replace("\\", "/").split("/")
    require("runs" in parts, "Recorded artifact does not identify a runs/ path.")
    relative = PurePosixPath(*parts[parts.index("runs"):])
    require(".." not in relative.parts, "Recorded artifact escapes the repository.")
    return root / relative


def git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=False
    )
    require(result.returncode == 0, "Git check failed: " + result.stderr.decode(errors="replace").strip())
    return result.stdout


def verify_files(root: Path, manifest: dict, *, require_tracked: bool) -> int:
    require(not any(path.startswith("runs/") for path in manifest["files"]),
            "Run archives must not be required repository files.")
    tracked_runs = git(root, "ls-files", "-z", "--", "runs/")
    require(not tracked_runs,
            "Remove runs/ from Git's index; preserve its local files and archive. "
            "If newly staged, unstage runs/ in your editor.")
    manuals = set(manifest.get("externally_compiled_manuals", []))
    require(manuals <= EXTERNAL_MANUALS, "Only the three software manuals may use external PDF verification.")
    require(not manuals.intersection(manifest["files"]), "External manuals must not also have compiler-specific hashes.")
    for relative, item in manifest["files"].items():
        path = root / relative
        require(path.is_file(), f"Missing release file: {relative}")
        require(
            digest(path.read_bytes(), text=item["text"]) == item["sha256"],
            f"Release content changed: {relative}",
        )
        if path.suffix == ".py":
            compile(path.read_bytes(), relative, "exec")
    for relative in sorted(manuals):
        path = root / relative
        require(path.is_file(), f"Missing compiled manual: {relative}")
        data = path.read_bytes()
        require(data.startswith(b"%PDF-"), f"Invalid PDF header: {relative}")
        ending = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", data[-4096:])
        require(ending is not None, f"Incomplete PDF end marker: {relative}")
        offset = int(ending.group(1))
        require(0 <= offset < len(data) and (
            data[offset:].startswith(b"xref")
            or re.match(rb"\d+\s+\d+\s+obj\b", data[offset:]) is not None
        ), f"Invalid PDF cross-reference pointer: {relative}")
    for relative in manifest["excluded_paths"]:
        require(not (root / relative).exists(), f"Remove retired paper asset: {relative}")
    for relative in manifest.get("excluded_trees", []):
        path = root / relative
        require(not path.is_file() and not any(p.is_file() for p in path.rglob("*")),
                f"Remove excluded source/build tree: {relative}")
    expected = set(manifest["files"]) | manuals | {MANIFEST}
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--stdin"],
        input="\n".join(sorted(expected)).encode(), capture_output=True, check=False,
    )
    require(ignored.returncode in (0, 1), "Cannot check Git inclusion rules.")
    require(not ignored.stdout, "Required files are ignored by Git:\n" + ignored.stdout.decode())
    if require_tracked:
        tracked = set(git(root, "ls-files", "-z").decode().rstrip("\0").split("\0"))
        missing = sorted(expected - tracked)
        extra = sorted(tracked - expected)
        require(not missing, "Stage these required release files:\n" + "\n".join(missing))
        require(not extra, "Unexpected tracked files:\n" + "\n".join(extra))
        # The Git index must contain the audited content, not an older staged copy.
        for relative in sorted(expected):
            item = manifest["files"].get(relative, {"text": relative not in manuals})
            staged = git(root, "show", ":" + relative)
            require(digest(staged, text=item["text"]) == digest((root / relative).read_bytes(), text=item["text"]),
                    f"Staged content differs from the verified file: {relative}")
    return len(expected)


def verify_base(root: Path, manifest: dict, reference: str) -> str:
    commit = git(root, "rev-parse", "--verify", reference + "^{commit}").decode().strip()
    for relative in manifest["protected_against_main"]:
        original = git(root, "show", commit + ":" + relative)
        require(digest(original, text=True) == digest((root / relative).read_bytes(), text=True),
                f"Frozen runtime/protocol differs from {reference}: {relative}")
    return commit


def verify_paper(root: Path, manifest: dict, archive_path: Path) -> int:
    with ZipFile(archive_path) as archive:
        for paper_path, repository_path in manifest["paper_evidence_map"].items():
            matches = [n for n in archive.namelist() if n == paper_path or n.endswith("/" + paper_path)]
            require(len(matches) == 1, f"Paper input missing or ambiguous: {paper_path}")
            require(digest(archive.read(matches[0]), text=True) == digest((root / repository_path).read_bytes(), text=True),
                    f"Paper evidence differs: {paper_path}")
    return len(manifest["paper_evidence_map"])


def verify_links(root: Path, manifest: dict) -> int:
    count = 0
    for relative in manifest["active_markdown"]:
        path = root / relative
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            require((path.parent / target).exists(), f"Broken link in {relative}: {target}")
            count += 1
    return count


def verify_archive_files(archive_root: Path, files: dict) -> int:
    """Check a local runs archive without requiring Git or changing its files."""
    for relative, item in files.items():
        parts = PurePosixPath(relative).parts
        require(parts and parts[0] == "runs" and ".." not in parts,
                f"Invalid local archive path: {relative}")
        path = archive_root.joinpath(*parts)
        require(path.is_file(), f"Missing local archive file: {relative}")
        require(digest(path.read_bytes(), text=item["text"]) == item["sha256"],
                f"Local archive content changed: {relative}")
    return len(files)


def verify_saved_evidence(root: Path, *, archive_root: Path | None = None,
                          archive_files: dict | None = None) -> dict:
    import numpy as np
    import pandas as pd
    from analysis.interpretation.validate_evaluation_tables import build_validation

    if archive_root is not None:
        import torch
        from analysis.training_diagnostics import training_points, training_episode_diagnostics, training_rollout_diagnostics
        from evaluation.validate_pre_experiment_artifacts import validate_action_bound_trajectory_metrics
        require(archive_files is not None, "Local archive inventory is missing.")
        count = verify_archive_files(archive_root, archive_files)
        print(f"PASS local archive hashes: {count} files (outside Git inventory)", flush=True)

    suites = (
        ("fixed_training_geometry", "fixed_training_geometry_analysis_protocol.json", "fixed_training_geometry.json"),
        ("core_layout_transfer", "projection_analysis_protocol.json", "core_navigation_layouts.json"),
    )
    tables = []
    trajectory_count = 0
    step_count = 0
    solver_calls = 0
    for suite, protocol, layout in suites:
        table_root = root / "results/tables" / suite
        other = "core_layout_transfer" if suite == "fixed_training_geometry" else "fixed_training_geometry"
        build_validation(
            episodes_path=str(table_root / "evaluation_episode_results.csv"),
            protocol_path=str(root / "experiments" / protocol),
            layout_suite_path=str(root / "evaluation/layouts" / layout),
            checkpoint_summary_path=str(table_root / "checkpoint_summary.csv"),
            method_summary_path=str(table_root / "method_summary.csv"),
            paired_deltas_path=str(table_root / "paired_projection_deltas.csv"),
            paired_summary_path=str(table_root / "paired_projection_summary.csv"),
            build_audit_path=str(table_root / "result_build_audit.json"),
            compare_checkpoint_paths=[str(root / "results/tables" / other / "evaluation_episode_results.csv")],
            label=None,
        )
        episodes = pd.read_csv(table_root / "evaluation_episode_results.csv")
        tables.append(episodes)
        print(f"PASS {suite}: {len(episodes)} published rows, summaries, and OFF/ON pairing", flush=True)
        if archive_root is None:
            continue
        for recorded_path, selected in episodes.groupby("source_csv", sort=True):
            csv_path = local_recorded_path(archive_root, recorded_path)
            raw = pd.read_csv(csv_path).sort_values("episode").reset_index(drop=True)
            selected = selected.sort_values("episode").reset_index(drop=True)
            pd.testing.assert_frame_equal(raw, selected[list(raw.columns)], check_dtype=False, atol=1e-12, rtol=0)
            npz_path = csv_path.with_name(csv_path.stem + "_trajectories.npz")
            with np.load(npz_path, allow_pickle=False) as archive:
                require(archive["run_checkpoint_sha256"].item() == raw.iloc[0].checkpoint_sha256, "Trajectory checkpoint identity differs.")
                require(archive["run_layout_suite_sha256"].item() == raw.iloc[0].layout_suite_sha256, "Trajectory layout identity differs.")
                keys = archive["episode_keys"].tolist()
                require(len(keys) == len(raw), "Trajectory coverage differs from CSV.")
                validate_action_bound_trajectory_metrics(raw, archive, str(npz_path.relative_to(archive_root)))
                for key in keys:
                    row = raw.loc[raw.episode == int(archive[f"{key}_episode"])].iloc[0]
                    require(int(archive[f"{key}_seed"]) == int(row.seed), "Trajectory seed differs.")
                    positions = archive[f"{key}_positions"]
                    nominal = archive[f"{key}_action_raw_physical"]
                    executed = archive[f"{key}_action_exec_physical"]
                    require(len(executed) == int(row.episode_length) and len(positions) == len(executed) + 1, "Trajectory alignment differs.")
                    np.testing.assert_allclose(executed - nominal, archive[f"{key}_action_correction_physical"], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(archive[f"{key}_rewards"].sum(), row.episode_return, atol=1e-10, rtol=0)
                    for field, column in (("state_success", "success"), ("state_collision", "collision"), ("terminated", "terminated"), ("truncated", "truncated")):
                        require(bool(archive[f"{key}_{field}"][-1]) == bool(row[column]), f"Trajectory endpoint differs: {column}")
                    require(bool(archive[f"{key}_projection_success"].all()), "Recorded trajectory contains a projection failure.")
                    if row.projection_mode == "disabled":
                        np.testing.assert_array_equal(nominal, executed)
                    else:
                        step_count += len(executed)
                        solver_calls += int(np.count_nonzero(archive[f"{key}_projection_active_constraint_count"]))
                    trajectory_count += 1
        print(f"PASS local archive {suite}: {len(episodes)} raw rows and trajectories", flush=True)

    if archive_root is None:
        return {"episode_rows": sum(len(x) for x in tables)}

    checkpoints = tables[0][["method", "train_seed", "checkpoint", "checkpoint_sha256"]].drop_duplicates()
    require(len(checkpoints) == 15, "Expected 15 distinct final checkpoints.")
    for row in checkpoints.itertuples(index=False):
        path = local_recorded_path(archive_root, row.checkpoint)
        require(digest(path.read_bytes(), text=False) == row.checkpoint_sha256, f"Checkpoint hash differs: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        args = payload["args"]
        require(payload["global_step"] == 51200 and payload["obs_dim"] == 21 and payload["action_dim"] == 2, "Checkpoint budget or shape differs.")
        require(args["method"] == row.method and args["seed"] == row.train_seed, "Checkpoint training identity differs.")
        require(payload["device"] == "cpu" and args["num_envs"] == 4 and args["num_steps"] == 256, "Checkpoint training configuration differs.")
        require(all(bool(torch.isfinite(value).all()) for value in payload["agent_state_dict"].values()), "Checkpoint contains nonfinite parameters.")

    fixed_root = root / "results/tables/fixed_training_geometry"
    summary = pd.read_csv(fixed_root / "checkpoint_summary.csv")
    points, _ = training_points(summary, archive_root / "runs", require_complete=True)
    training_frames = {
        "training_scalar_events.csv": points,
        "training_episode_diagnostics.csv": training_episode_diagnostics(points, rolling_window=20, require_complete=True),
        "training_rollout_diagnostics.csv": training_rollout_diagnostics(points, require_complete=True),
    }
    for filename, rebuilt in training_frames.items():
        saved = pd.read_csv(fixed_root / filename)
        # Event-directory paths are historical provenance and relocate across machines.
        for frame in (saved, rebuilt):
            for column in frame.select_dtypes(include="object").columns:
                frame[column] = frame[column].map(lambda x: str(local_recorded_path(Path("."), x)).replace("\\", "/") if isinstance(x, str) and "runs/" in x.replace("\\", "/") else x)
        pd.testing.assert_frame_equal(saved.reset_index(drop=True), rebuilt[list(saved.columns)].reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
    require(step_count == 298984 and solver_calls == 296269, "Recorded solver exposure differs from the paper.")
    print("PASS 15 final checkpoints and final TensorBoard exports; 296269 recorded QP calls, zero recorded failures", flush=True)
    return {"episode_rows": sum(len(x) for x in tables), "trajectories": trajectory_count, "checkpoints": len(checkpoints), "qp_calls": solver_calls}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="main", help="Local base branch or commit for protected runtime/protocol comparison.")
    parser.add_argument("--require-tracked", action="store_true", help="Also require the exact release inventory and content in Git's index; use after staging.")
    parser.add_argument("--paper-project", type=Path, help="Optional separate Overleaf ZIP; compare its six frozen evidence inputs.")
    parser.add_argument("--archive-root", type=Path, help="Optional local archive directory containing runs/. Verify raw artifacts without adding them to Git.")
    args = parser.parse_args()
    try:
        manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
        count = verify_files(ROOT, manifest, require_tracked=args.require_tracked)
        print(f"PASS release inventory and content: {count} files", flush=True)
        if manifest.get("externally_compiled_manuals"):
            print("PASS external manual PDF headers, end markers, and cross-reference pointers; review rendered content separately", flush=True)
        commit = verify_base(ROOT, manifest, args.base_ref)
        print(f"PASS protected runtime and protocols equal {args.base_ref} ({commit})", flush=True)
        print(f"PASS documentation links: {verify_links(ROOT, manifest)}", flush=True)
        if args.paper_project:
            print(f"PASS separate paper evidence: {verify_paper(ROOT, manifest, args.paper_project)} matching inputs", flush=True)
        verify_saved_evidence(ROOT, archive_root=args.archive_root,
                              archive_files=manifest.get("local_archive_files"))
        if args.archive_root is not None:
            print("LOCAL_ARCHIVE PASS (read-only; no Git inclusion required)", flush=True)
        else:
            print("Local run archive not requested; repository verification is complete without it.", flush=True)
    except (ValueError, OSError, AssertionError, KeyError) as error:
        print(f"REPOSITORY_RELEASE FAIL: {error}", flush=True)
        return 1
    print("REPOSITORY_RELEASE PASS (repository files and tables; no training or policy evaluation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
