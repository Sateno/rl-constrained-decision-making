from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


#################################################################################
# region Configuration

# One explicit PPO training variant.
@dataclass(frozen=True)
class TrainingVariant:
#{
    name: str
    method: str
    experiment_prefix: str
    collision_penalty: float
    projection_enabled: bool

#} End dataclass TrainingVariant


TRAINING_VARIANTS = {
    "baseline": TrainingVariant(
        name="baseline",
        method="ppo_baseline",
        experiment_prefix="ppo_baseline",
        collision_penalty=10.0,
        projection_enabled=False,
    ),
    "high_penalty": TrainingVariant(
        name="high_penalty",
        method="ppo_high_penalty",
        experiment_prefix="ppo_high_penalty",
        collision_penalty=50.0,
        projection_enabled=False,
    ),
    "projection": TrainingVariant(
        name="projection",
        method="ppo_train_projection",
        experiment_prefix="ppo_train_projection",
        collision_penalty=10.0,
        projection_enabled=True,
    ),
}

# end region Configuration


#################################################################################
# region Helpers

# Resolve one path relative to the repository root.
def repository_path(path: Path) -> Path:
#{
    return path if path.is_absolute() else REPOSITORY_ROOT / path
#} End function repository_path


# Return the default checkpoint path for one training run.
def default_checkpoint_path(variant: TrainingVariant, seed: int, total_timesteps: int) -> Path:
#{
    filename = f"{variant.experiment_prefix}_{total_timesteps}_seed{seed}.pt"
    return Path("runs/checkpoints") / filename
#} End function default_checkpoint_path


# Build the exact PPO command for one declared training variant.
def training_command(
    variant: TrainingVariant,
    seed: int,
    total_timesteps: int,
    checkpoint_path: Path,
    experiment_name: str | None = None,
) -> list[str]:
#{
    selected_experiment_name = experiment_name or (
        f"{variant.experiment_prefix}_{total_timesteps}_seed{seed}"
    )
    command = [
        sys.executable,
        "-m",
        "algorithms.ppo.ppo_continuous_action",
        "--method",
        variant.method,
        "--exp-name",
        selected_experiment_name,
        "--env-id",
        "ConstrainedNavigation-v0",
        "--total-timesteps",
        str(total_timesteps),
        "--num-envs",
        "4",
        "--num-steps",
        "256",
        "--num-minibatches",
        "8",
        "--update-epochs",
        "4",
        "--max-episode-steps",
        "200",
        "--max-obstacles",
        "3",
        "--num-active-obstacles",
        "3",
        "--collision-penalty",
        str(variant.collision_penalty),
        "--seed",
        str(seed),
        "--save-model",
        "--checkpoint-path",
        str(checkpoint_path),
    ]

    if variant.projection_enabled:
        command.extend(
            [
                "--enable-projection",
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
    else:
        command.append("--no-enable-projection")

    return command
#} End function training_command


# Execute one training variant and verify that its checkpoint was created.
def run_training(
    variant_name: str,
    seed: int,
    total_timesteps: int,
    checkpoint_path: Path | None,
    experiment_name: str | None = None,
) -> Path:
#{
    if seed < 0:
        raise ValueError("Training seed must be nonnegative.")
    if total_timesteps <= 0:
        raise ValueError("Total timesteps must be positive.")

    variant = TRAINING_VARIANTS[variant_name]
    selected_path = checkpoint_path or default_checkpoint_path(
        variant,
        seed,
        total_timesteps,
    )
    absolute_checkpoint = repository_path(selected_path).resolve()

    if absolute_checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {absolute_checkpoint}")

    absolute_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    command = training_command(
        variant,
        seed,
        total_timesteps,
        absolute_checkpoint,
        experiment_name,
    )
    print(f"Training variant: {variant.name}")
    print(f"Seed:             {seed}")
    print(f"Total timesteps:  {total_timesteps}")
    print(f"Checkpoint:       {absolute_checkpoint}")
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    if not absolute_checkpoint.is_file():
        raise RuntimeError(f"Training completed without creating checkpoint: {absolute_checkpoint}")

    print("Training completed successfully.")
    print(f"Checkpoint: {absolute_checkpoint}")
    return absolute_checkpoint
#} End function run_training

# end region Helpers


#################################################################################
# region Command line

# Parse the thin launcher-compatible training command.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Train one declared PPO experiment variant."
    )
    parser.add_argument("variant", choices=sorted(TRAINING_VARIANTS))
    parser.add_argument("seed", type=int, nargs="?", default=1)
    parser.add_argument("total_timesteps", type=int, nargs="?", default=51200)
    parser.add_argument("checkpoint_path", type=Path, nargs="?", default=None)
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()
#} End function parse_args


# Run the requested training variant.
def main() -> int:
#{
    args = parse_args()
    run_training(
        variant_name=args.variant,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        checkpoint_path=args.checkpoint_path,
        experiment_name=args.experiment_name,
    )
    return 0
#} End function main

# end region Command line


if __name__ == "__main__":
    raise SystemExit(main())
