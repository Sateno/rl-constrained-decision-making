from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_ROOT = Path("runs/validation/pre_experiment_codebase")
DEFAULT_SEED = 9901
DEFAULT_TIMESTEPS = 2048


#################################################################################
# region Logging and process helpers

# Resolve one path relative to the repository root.
def repository_path(path: Path) -> Path:
#{
    return path if path.is_absolute() else REPOSITORY_ROOT / path
#} End function repository_path


# Verify that a repeatable validation target is below runs/validation.
def validate_cleanup_target(path: Path) -> None:
#{
    allowed_root = (REPOSITORY_ROOT / "runs" / "validation").resolve()
    resolved = path.resolve()

    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise ValueError(f"Refusing to remove unsafe validation path: {resolved}")
#} End function validate_cleanup_target


# Print and append one validation line.
def log(message: str, log_path: Path) -> None:
#{
    print(message)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(message + "\n")
#} End function log


# Run one child command and retain its complete output in the master log.
def run_command(label: str, command: list[str], log_path: Path) -> None:
#{
    log("", log_path)
    log(label, log_path)
    log("-" * len(label), log_path)
    log(subprocess.list2cmdline(command), log_path)
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(f"{label} did not expose a readable output stream.")

    for line in process.stdout:
        log(line.rstrip("\r\n"), log_path)

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}."
        )

    log("[PASS]", log_path)
#} End function run_command

# end region Logging and process helpers


#################################################################################
# region Cleanup and command construction

# Remove only artifacts reserved for this repeatable validation.
def prepare_validation_directory(
    validation_root: Path,
    seed: int,
    total_timesteps: int,
    replace: bool,
) -> tuple[Path, Path, Path]:
#{
    resolved_root = repository_path(validation_root).resolve()
    validate_cleanup_target(resolved_root)

    if resolved_root.exists():
        if not replace:
            raise FileExistsError(
                f"Validation directory already exists: {resolved_root}. "
                "Pass --replace only for the dedicated repeatable validation directory."
            )
        shutil.rmtree(resolved_root)

    runs_directory = (REPOSITORY_ROOT / "runs").resolve()
    patterns = (
        f"ConstrainedNavigation-v0__ppo_baseline_{total_timesteps}_seed{seed}__{seed}__*",
        f"ConstrainedNavigation-v0__ppo_high_penalty_{total_timesteps}_seed{seed}__{seed}__*",
        f"ConstrainedNavigation-v0__ppo_train_projection_{total_timesteps}_seed{seed}__{seed}__*",
    )

    for pattern in patterns:
        for path in runs_directory.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)

    checkpoint_directory = resolved_root / "checkpoints"
    layout_directory = resolved_root / "layout_evaluation"
    result_directory = resolved_root / "result_build"
    checkpoint_directory.mkdir(parents=True)
    layout_directory.mkdir(parents=True)
    return checkpoint_directory, layout_directory, result_directory
#} End function prepare_validation_directory


# Return one common-layout evaluation command.
def layout_evaluation_command(
    projection_mode: str,
    output_csv: Path,
    output_npz: Path,
) -> list[str]:
#{
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_layout_suite",
        "--checkpoint",
        "runs/checkpoints/ppo_baseline_51200_seed1.pt",
        "--layout-suite",
        "evaluation/layouts/development_navigation_layouts.json",
        "--method",
        "ppo_baseline",
        "--train-seed",
        "1",
        "--projection-mode",
        projection_mode,
        "--repeats-per-layout",
        "1",
        "--seed",
        "1000",
        "--max-episode-steps",
        "200",
        "--collision-penalty",
        "10.0",
        "--no-cuda",
        "--output",
        str(output_csv),
        "--trajectory-output",
        str(output_npz),
    ]

    if projection_mode == "enabled":
        command.extend(
            [
                "--projection-lookahead-distance",
                "0.25",
                "--projection-alpha",
                "2.0",
                "--projection-slack-penalty",
                "1000.0",
                "--projection-extra-clearance",
                "0.0",
            ]
        )

    return command
#} End function layout_evaluation_command

# end region Cleanup and command construction


#################################################################################
# region Validation workflow

# Write the final PASS marker after every automated phase succeeds.
def write_summary(
    validation_root: Path,
    result_directory: Path,
    seed: int,
    total_timesteps: int,
    log_path: Path,
) -> Path:
#{
    summary_path = validation_root / "pre_experiment_validation_summary.txt"
    lines = [
        "status=PASS",
        f"completed_at={datetime.now(timezone.utc).isoformat()}",
        f"validation_seed={seed}",
        f"validation_timesteps={total_timesteps}",
        "canonical_summary=runs/validation/evaluation_time_projection_validation_summary.txt",
        f"training_audit={validation_root / 'training_smoke_audit.json'}",
        f"layout_audit={validation_root / 'layout_evaluation_audit.json'}",
        f"result_audit={validation_root / 'result_build_audit.json'}",
        f"figures={result_directory / 'figures'}",
        f"tables={result_directory / 'tables'}",
        f"manual_review={validation_root / 'manual_review.txt'}",
        "manual_review_required=true",
        f"master_log={log_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary_path
#} End function write_summary


# Execute the complete pre-experiment validation workflow.
def run_validation(args: argparse.Namespace) -> Path:
#{
    validation_root = repository_path(args.validation_root).resolve()
    checkpoint_directory, layout_directory, result_directory = prepare_validation_directory(
        validation_root=validation_root,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        replace=args.replace,
    )
    log_path = validation_root / "pre_experiment_validation.log"
    log_path.unlink(missing_ok=True)
    log("Pre-Experiment Codebase Validation", log_path)
    log("=================================", log_path)
    log(f"started_at={datetime.now(timezone.utc).isoformat()}", log_path)
    log(f"repository_root={REPOSITORY_ROOT}", log_path)
    log(f"validation_root={validation_root}", log_path)

    run_command(
        "Preflight",
        [
            sys.executable,
            "-m",
            "evaluation.validate_pre_experiment_artifacts",
            "preflight",
        ],
        log_path,
    )
    run_command(
        "Active source compilation",
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "algorithms",
            "environments",
            "evaluation",
            "analysis",
            "projection",
            "experiments",
            "tests",
        ],
        log_path,
    )
    run_command(
        "Canonical evaluation-time regression",
        [sys.executable, "-m", "evaluation.validate_evaluation_time_projection"],
        log_path,
    )
    run_command(
        "Canonical summary audit",
        [
            sys.executable,
            "-m",
            "evaluation.validate_pre_experiment_artifacts",
            "canonical",
        ],
        log_path,
    )

    training_specs = (
        ("baseline", checkpoint_directory / "baseline.pt", "Baseline training smoke"),
        ("high_penalty", checkpoint_directory / "high_penalty.pt", "High-penalty training smoke"),
        ("projection", checkpoint_directory / "train_projection.pt", "Projection training smoke"),
    )

    for variant, checkpoint_path, label in training_specs:
        run_command(
            label,
            [
                sys.executable,
                "-m",
                "experiments.train_ppo_variant",
                variant,
                str(args.seed),
                str(args.total_timesteps),
                str(checkpoint_path),
            ],
            log_path,
        )

    run_command(
        "Training artifact audit",
        [
            sys.executable,
            "-m",
            "evaluation.validate_pre_experiment_artifacts",
            "training",
            "--checkpoint-dir",
            str(checkpoint_directory),
            "--output",
            str(validation_root / "training_smoke_audit.json"),
            "--seed",
            str(args.seed),
            "--total-timesteps",
            str(args.total_timesteps),
        ],
        log_path,
    )
    run_command(
        "Development layouts without projection",
        layout_evaluation_command(
            projection_mode="disabled",
            output_csv=layout_directory / "baseline_projection_disabled.csv",
            output_npz=layout_directory / "baseline_projection_disabled_trajectories.npz",
        ),
        log_path,
    )
    run_command(
        "Development layouts with projection",
        layout_evaluation_command(
            projection_mode="enabled",
            output_csv=layout_directory / "baseline_projection_enabled.csv",
            output_npz=layout_directory / "baseline_projection_enabled_trajectories.npz",
        ),
        log_path,
    )
    run_command(
        "Common-layout artifact audit",
        [
            sys.executable,
            "-m",
            "evaluation.validate_pre_experiment_artifacts",
            "layouts",
            "--evaluation-dir",
            str(layout_directory),
            "--output",
            str(validation_root / "layout_evaluation_audit.json"),
        ],
        log_path,
    )
    run_command(
        "Saved-result table and figure build",
        [
            sys.executable,
            "-m",
            "analysis.build_projection_results",
            "experiments/development_projection_analysis_protocol.json",
            str(layout_directory),
            str(result_directory),
            "runs",
        ],
        log_path,
    )
    run_command(
        "Saved-result artifact audit",
        [
            sys.executable,
            "-m",
            "evaluation.validate_pre_experiment_artifacts",
            "results",
            "--result-dir",
            str(result_directory),
            "--output",
            str(validation_root / "result_build_audit.json"),
        ],
        log_path,
    )
    summary_path = write_summary(
        validation_root=validation_root,
        result_directory=result_directory,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        log_path=log_path,
    )
    log("", log_path)
    log("PRE-EXPERIMENT VALIDATION PASSED", log_path)
    log("================================", log_path)
    log(f"summary={summary_path}", log_path)
    log("Next: scripts\\inspect_pre_experiment_validation.bat", log_path)
    return summary_path
#} End function run_validation

# end region Validation workflow


#################################################################################
# region Command line

# Parse the repeatable validation configuration.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Run the complete pre-experiment codebase validation."
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=DEFAULT_VALIDATION_ROOT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps must be positive.")

    return args
#} End function parse_args


# Run validation and preserve a readable failure record.
def main() -> int:
#{
    args = parse_args()
    validation_root = repository_path(args.validation_root).resolve()
    log_path = validation_root / "pre_experiment_validation.log"

    try:
        run_validation(args)
    except Exception as error:
        validation_root.mkdir(parents=True, exist_ok=True)
        log("", log_path)
        log("PRE-EXPERIMENT VALIDATION FAILED", log_path)
        log("================================", log_path)
        log(str(error), log_path)
        log(traceback.format_exc().rstrip(), log_path)
        print(f"Review: {log_path}")
        return 1

    return 0
#} End function main

# end region Command line


if __name__ == "__main__":
    raise SystemExit(main())
