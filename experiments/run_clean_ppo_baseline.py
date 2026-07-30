from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = Path("runs")
SUMMARY_COLUMNS = [
    "episode_return",
    "episode_length",
    "success",
    "collision",
    "final_distance_to_goal",
    "min_obstacle_clearance",
]


#################################################################################
# region Helpers

# Return one repository-relative path as an absolute path.
def repository_path(path: Path) -> Path:
#{
    return path if path.is_absolute() else REPOSITORY_ROOT / path
#} End function repository_path


# Verify that a destructive target is below the repository root.
def validate_removal_target(path: Path) -> None:
#{
    resolved_root = REPOSITORY_ROOT.resolve()
    resolved_path = path.resolve()

    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to remove unsafe path: {resolved_path}")
#} End function validate_removal_target


# Run one visible pipeline command and stop on failure.
def run_command(label: str, command: list[str]) -> None:
#{
    print()
    print(label)
    print("-" * len(label))
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
#} End function run_command


# Build one evaluator command.
def evaluation_command(
    checkpoint: Path | None,
    output: Path,
    episodes: int,
    evaluation_seed: int,
    max_episode_steps: int,
    stochastic: bool,
) -> list[str]:
#{
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_policy",
        "--policy",
        "ppo" if checkpoint is not None else "random",
        "--episodes",
        str(episodes),
        "--seed",
        str(evaluation_seed),
        "--max-episode-steps",
        str(max_episode_steps),
        "--output",
        str(output),
    ]

    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if stochastic:
        command.append("--stochastic")

    return command
#} End function evaluation_command


# Print and save one compact comparison of generated evaluation CSVs.
def write_summary(
    random_csv: Path,
    deterministic_csv: Path,
    stochastic_csv: Path,
    summary_path: Path,
) -> None:
#{
    frames = {
        "random": pd.read_csv(random_csv),
        "ppo_deterministic": pd.read_csv(deterministic_csv),
        "ppo_stochastic": pd.read_csv(stochastic_csv),
    }
    lines = [
        "status=PASS",
        f"completed_at={datetime.now(timezone.utc).isoformat()}",
    ]

    for name, frame in frames.items():
        means = frame[SUMMARY_COLUMNS].mean(numeric_only=True)
        print()
        print(name.replace("_", " "))
        print(means)
        lines.append(f"[{name}]")

        for metric, value in means.items():
            lines.append(f"{metric}={float(value)}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
#} End function write_summary

# end region Helpers


#################################################################################
# region Pipeline

# Run the historical clean baseline pipeline through Python orchestration.
def run_pipeline(args: argparse.Namespace) -> None:
#{
    runs_directory = repository_path(RUNS_DIRECTORY).resolve()
    validate_removal_target(runs_directory)

    if runs_directory.exists():
        if not args.replace_runs:
            raise FileExistsError(
                f"Runs directory already exists: {runs_directory}. "
                "Pass --replace-runs only when a clean baseline run is intended."
            )
        shutil.rmtree(runs_directory)

    checkpoint_directory = runs_directory / "checkpoints"
    evaluation_directory = runs_directory / "evaluation"
    validation_directory = runs_directory / "validation"
    checkpoint_directory.mkdir(parents=True)
    evaluation_directory.mkdir(parents=True)
    validation_directory.mkdir(parents=True)
    checkpoint = checkpoint_directory / f"ppo_baseline_{args.total_timesteps}_seed{args.seed}.pt"
    random_csv = evaluation_directory / f"random_policy_seed{args.evaluation_seed}.csv"
    deterministic_csv = evaluation_directory / f"ppo_baseline_{args.total_timesteps}_seed{args.seed}_eval.csv"
    stochastic_csv = evaluation_directory / f"ppo_baseline_{args.total_timesteps}_seed{args.seed}_eval_stochastic.csv"
    summary_path = validation_directory / "clean_ppo_baseline_summary.txt"

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
            "projection",
            "tests",
        ],
    )
    run_command(
        "Complete lightweight test suite",
        [sys.executable, "-m", "pytest", "-q"],
    )
    run_command(
        "Random-policy evaluation",
        evaluation_command(
            checkpoint=None,
            output=random_csv,
            episodes=args.episodes,
            evaluation_seed=args.evaluation_seed,
            max_episode_steps=args.max_episode_steps,
            stochastic=False,
        ),
    )
    run_command(
        "PPO baseline training",
        [
            sys.executable,
            "-m",
            "experiments.train_ppo_variant",
            "baseline",
            str(args.seed),
            str(args.total_timesteps),
            str(checkpoint),
            "--experiment-name",
            f"ppo_baseline_{args.total_timesteps}_seed{args.seed}_clean",
        ],
    )
    run_command(
        "Deterministic checkpoint evaluation",
        evaluation_command(
            checkpoint=checkpoint,
            output=deterministic_csv,
            episodes=args.episodes,
            evaluation_seed=args.evaluation_seed,
            max_episode_steps=args.max_episode_steps,
            stochastic=False,
        ),
    )
    run_command(
        "Stochastic checkpoint evaluation",
        evaluation_command(
            checkpoint=checkpoint,
            output=stochastic_csv,
            episodes=args.episodes,
            evaluation_seed=args.evaluation_seed,
            max_episode_steps=args.max_episode_steps,
            stochastic=True,
        ),
    )
    print()
    print("Summary comparison")
    print("------------------")
    write_summary(
        random_csv=random_csv,
        deterministic_csv=deterministic_csv,
        stochastic_csv=stochastic_csv,
        summary_path=summary_path,
    )
    print()
    print("Clean PPO baseline pipeline completed successfully.")
    print(f"Checkpoint: {checkpoint}")
    print(f"Summary:    {summary_path}")
    print(f"TensorBoard: tensorboard --logdir {runs_directory}")
#} End function run_pipeline

# end region Pipeline


#################################################################################
# region Command line

# Parse one clean baseline pipeline configuration.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Run the complete clean PPO baseline regression pipeline."
    )
    parser.add_argument("--replace-runs", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=51200)
    parser.add_argument("--evaluation-seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    args = parser.parse_args()

    if args.seed < 0 or args.evaluation_seed < 0:
        parser.error("Seeds must be nonnegative.")
    if args.total_timesteps <= 0 or args.episodes <= 0 or args.max_episode_steps <= 0:
        parser.error("Timesteps, episodes, and maximum episode steps must be positive.")

    return args
#} End function parse_args


# Run the clean baseline pipeline.
def main() -> int:
#{
    args = parse_args()
    run_pipeline(args)
    return 0
#} End function main

# end region Command line


if __name__ == "__main__":
    raise SystemExit(main())
