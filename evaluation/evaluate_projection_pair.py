from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from environments.factory import make_env
from evaluation.evaluate_policy import (
    EpisodeResult,
    load_ppo_agent,
    make_ppo_action_provider,
    make_projection_params,
    print_summary,
    projection_parameter_metadata,
    run_episode,
    summarize_results,
    validate_checkpoint_environment_compatibility,
    write_results_csv,
)
from evaluation.trajectory_recording import EpisodeTrajectory, write_trajectory_archive
from projection.cbf_qp_projection import ProjectionParams


#################################################################################
# region functions

# Parse one configuration that will be applied to both projection modes.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one PPO checkpoint with projection disabled and enabled "
            "under the same episode seeds and environment configuration."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the PPO checkpoint used in both evaluation modes.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Training-method identifier recorded in every paired artifact.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        required=True,
        help="Independent training seed associated with the checkpoint.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of paired evaluation episodes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Paired episode i uses seed + i in both modes.",
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("runs/evaluation/ppo_projection_pair"),
        help="Output path prefix without a file extension.",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Maximum number of environment steps per episode.",
    )

    parser.add_argument(
        "--max-obstacles",
        type=int,
        default=3,
        help="Fixed obstacle capacity. This must match the checkpoint observation dimension.",
    )

    parser.add_argument(
        "--num-active-obstacles",
        type=int,
        default=None,
        help="Number of active obstacles in the built-in layout. The default is min(3, max_obstacles).",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=10.0,
        help="Common evaluation collision penalty applied in both projection modes.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the PPO policy instead of using the actor mean.",
    )

    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA for PPO policy evaluation when available.",
    )

    default_projection_params = ProjectionParams()

    parser.add_argument(
        "--projection-lookahead-distance",
        type=float,
        default=default_projection_params.lookahead_distance,
        help="Nonnegative lookahead distance used by the CBF projection geometry.",
    )

    parser.add_argument(
        "--projection-alpha",
        type=float,
        default=default_projection_params.alpha,
        help="Positive CBF barrier-rate parameter.",
    )

    parser.add_argument(
        "--projection-slack-penalty",
        type=float,
        default=default_projection_params.slack_penalty,
        help="Positive quadratic penalty applied to CBF slack variables.",
    )

    parser.add_argument(
        "--projection-extra-clearance",
        type=float,
        default=default_projection_params.extra_clearance,
        help="Nonnegative projection-only clearance beyond collision geometry.",
    )

    args = parser.parse_args()

    if not args.method.strip():
        parser.error("--method must be nonempty.")

    if args.train_seed < 0:
        parser.error("--train-seed must be nonnegative.")

    if args.episodes <= 0:
        parser.error("--episodes must be positive.")

    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive.")

    if args.max_obstacles <= 0:
        parser.error("--max-obstacles must be positive.")

    if args.num_active_obstacles is not None:
        if args.num_active_obstacles < 0 or args.num_active_obstacles > args.max_obstacles:
            parser.error("--num-active-obstacles must be between 0 and --max-obstacles.")
        if args.num_active_obstacles > 3:
            parser.error("The built-in layout defines at most three active obstacles.")

    if not np.isfinite(args.collision_penalty) or args.collision_penalty < 0.0:
        parser.error("--collision-penalty must be finite and nonnegative.")

    if not np.isfinite(args.projection_lookahead_distance) or args.projection_lookahead_distance < 0.0:
        parser.error("--projection-lookahead-distance must be finite and nonnegative.")

    if not np.isfinite(args.projection_alpha) or args.projection_alpha <= 0.0:
        parser.error("--projection-alpha must be finite and positive.")

    if not np.isfinite(args.projection_slack_penalty) or args.projection_slack_penalty <= 0.0:
        parser.error("--projection-slack-penalty must be finite and positive.")

    if not np.isfinite(args.projection_extra_clearance) or args.projection_extra_clearance < 0.0:
        parser.error("--projection-extra-clearance must be finite and nonnegative.")

    return args
#} End function parse_args


# Return deterministic output paths derived from one run prefix.
def make_output_paths(output_prefix: str | Path) -> dict[str, Path]:
#{
    prefix = Path(output_prefix)
    parent = prefix.parent
    name = prefix.name

    return {
        "projection_disabled": parent / f"{name}_projection_disabled.csv",
        "projection_enabled": parent / f"{name}_projection_enabled.csv",
        "projection_disabled_trajectories": parent / f"{name}_projection_disabled_trajectories.npz",
        "projection_enabled_trajectories": parent / f"{name}_projection_enabled_trajectories.npz",
        "paired_episodes": parent / f"{name}_paired_episodes.csv",
        "paired_summary": parent / f"{name}_paired_summary.csv",
    }

#} End function make_output_paths


# Compute a stable checkpoint identifier for the paired artifact.
def file_sha256(path: str | Path) -> str:
#{
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

#} End function file_sha256


# Resolve the active count exactly as the environment constructor does.
def resolve_num_active_obstacles(max_obstacles: int, num_active_obstacles: int | None) -> int:
#{
    if num_active_obstacles is None:
        return min(3, int(max_obstacles))

    return int(num_active_obstacles)

#} End function resolve_num_active_obstacles


# Run one projection mode with the exact episode seeds used by the other mode.
def run_evaluation_mode(
    *,
    env_factory: Callable,
    action_provider,
    episodes: int,
    base_seed: int,
    checkpoint_path: str,
    method: str,
    projection_enabled: bool,
    device: torch.device,
    trajectory_records: list[EpisodeTrajectory],
) -> list[EpisodeResult]:
#{
    results: list[EpisodeResult] = []

    for episode_index in range(episodes):
    #{
        episode_seed = base_seed + episode_index
        env = env_factory()

        try:
            env.action_space.seed(episode_seed)
            torch.manual_seed(episode_seed)

            if device.type == "cuda":
                torch.cuda.manual_seed_all(episode_seed)

            result = run_episode(
                env=env,
                action_provider=action_provider,
                seed=episode_seed,
                episode=episode_index,
                policy_name=method,
                checkpoint_path=checkpoint_path,
                trajectory_records=trajectory_records,
            )

            if result.projection_enabled != projection_enabled:
                raise RuntimeError(
                    "Environment projection mode did not match the paired evaluation configuration."
                )

            results.append(result)

        finally:
            env.close()

    #} End loop episodes

    return results

#} End function run_evaluation_mode


# Validate explicit labels and return training fields persisted by the checkpoint.
def checkpoint_training_metadata(
    checkpoint: dict,
    *,
    method: str,
    train_seed: int,
) -> dict[str, object]:
#{
    checkpoint_args = checkpoint.get("args")

    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    checkpoint_method = checkpoint_args.get("method")
    checkpoint_seed = checkpoint_args.get("seed")

    if checkpoint_method is not None and str(checkpoint_method) != method:
        raise ValueError(
            f"Checkpoint method={checkpoint_method!r} does not match --method={method!r}."
        )
    if checkpoint_seed is not None and int(checkpoint_seed) != int(train_seed):
        raise ValueError(
            f"Checkpoint seed={checkpoint_seed} does not match --train-seed={train_seed}."
        )

    return {
        "method": method,
        "train_seed": int(train_seed),
        "training_collision_penalty": float(
            checkpoint_args.get("collision_penalty", np.nan)
        ),
        "training_projection_enabled": bool(
            checkpoint_args.get("enable_projection", False)
        ),
    }

#} End function checkpoint_training_metadata


# Build metadata shared by both modes and every paired artifact row.
def make_pair_metadata(
    *,
    args: argparse.Namespace,
    projection_params: ProjectionParams,
    checkpoint_sha256: str,
    training_metadata: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
#{
    metadata: dict[str, object] = {
        **training_metadata,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_policy_mode": (
            "stochastic" if args.stochastic else "deterministic"
        ),
        "stochastic": bool(args.stochastic),
        "device": str(device),
        "requested_episodes": int(args.episodes),
        "base_seed": int(args.seed),
        "last_seed": int(args.seed + args.episodes - 1),
        "max_episode_steps": int(args.max_episode_steps),
        "max_obstacles": int(args.max_obstacles),
        "num_active_obstacles": resolve_num_active_obstacles(
            args.max_obstacles,
            args.num_active_obstacles,
        ),
        "evaluation_collision_penalty": float(args.collision_penalty),
    }
    metadata.update(projection_parameter_metadata(projection_params))

    return metadata

#} End function make_pair_metadata


# Return a one-to-one index for episode results and reject duplicate pair keys.
def index_episode_results(results: list[EpisodeResult], mode_name: str) -> dict[tuple[int, int], EpisodeResult]:
#{
    indexed: dict[tuple[int, int], EpisodeResult] = {}

    for result in results:
        key = (int(result.episode), int(result.seed))

        if key in indexed:
            raise ValueError(f"Duplicate {mode_name} episode key: {key}")

        indexed[key] = result

    return indexed

#} End function index_episode_results


# Build one wide row per paired episode for direct off/on comparison.
def build_paired_episode_table(
    *,
    projection_disabled_results: list[EpisodeResult],
    projection_enabled_results: list[EpisodeResult],
    pair_metadata: dict[str, object],
) -> pd.DataFrame:
#{
    disabled_by_key = index_episode_results(projection_disabled_results, "projection-disabled")
    enabled_by_key = index_episode_results(projection_enabled_results, "projection-enabled")

    if disabled_by_key.keys() != enabled_by_key.keys():
        raise ValueError("Projection-disabled and projection-enabled episode keys do not match.")

    task_fields = (
        "episode_return",
        "episode_length",
        "success",
        "collision",
        "terminated",
        "truncated",
        "final_distance_to_goal",
        "min_obstacle_clearance",
        "action_bound_clipping_count",
        "action_bound_clipping_rate",
        "speed_action_bound_clipping_count",
        "speed_action_bound_clipping_rate",
        "turn_rate_action_bound_clipping_count",
        "turn_rate_action_bound_clipping_rate",
        "mean_action_bound_clipping_norm",
        "max_action_bound_clipping_norm",
    )
    projection_fields = {
        "projection_intervention_count": "intervention_count",
        "projection_intervention_rate": "intervention_rate",
        "mean_projection_correction_norm": "mean_correction_norm",
        "max_projection_correction_norm": "max_correction_norm",
        "mean_projection_slack_sum": "mean_slack_sum",
        "max_projection_slack": "max_slack",
        "projection_solver_failure_count": "solver_failure_count",
    }
    delta_fields = (
        "episode_return",
        "episode_length",
        "success",
        "collision",
        "final_distance_to_goal",
        "min_obstacle_clearance",
        "action_bound_clipping_rate",
        "speed_action_bound_clipping_rate",
        "turn_rate_action_bound_clipping_rate",
        "mean_action_bound_clipping_norm",
        "max_action_bound_clipping_norm",
    )

    rows: list[dict[str, object]] = []

    for key in sorted(disabled_by_key):
    #{
        disabled_result = disabled_by_key[key]
        enabled_result = enabled_by_key[key]

        if disabled_result.policy != enabled_result.policy:
            raise ValueError(f"Paired episode {key} has different policy labels.")

        if disabled_result.checkpoint != enabled_result.checkpoint:
            raise ValueError(f"Paired episode {key} has different checkpoint paths.")

        if disabled_result.projection_enabled:
            raise ValueError(f"Projection-disabled episode {key} reports projection enabled.")

        if not enabled_result.projection_enabled:
            raise ValueError(f"Projection-enabled episode {key} reports projection disabled.")

        disabled_values = asdict(disabled_result)
        enabled_values = asdict(enabled_result)

        row: dict[str, object] = {
            "policy": disabled_result.policy,
            "checkpoint": disabled_result.checkpoint,
            "episode": int(disabled_result.episode),
            "seed": int(disabled_result.seed),
        }
        row.update(pair_metadata)

        for field_name in task_fields:
            row[f"without_projection_{field_name}"] = disabled_values[field_name]
            row[f"with_projection_{field_name}"] = enabled_values[field_name]

        for field_name, output_name in projection_fields.items():
            row[f"with_projection_{output_name}"] = enabled_values[field_name]

        for field_name in delta_fields:
            disabled_value = disabled_values[field_name]
            enabled_value = enabled_values[field_name]
            row[f"{field_name}_delta_enabled_minus_disabled"] = (
                int(enabled_value) - int(disabled_value)
                if isinstance(disabled_value, bool)
                else float(enabled_value) - float(disabled_value)
            )

        rows.append(row)

    #} End loop paired episodes

    return pd.DataFrame(rows)

#} End function build_paired_episode_table


# Build one wide summary row with direct enabled-minus-disabled deltas.
def build_paired_summary_table(
    *,
    projection_disabled_summary: dict[str, float | int | bool],
    projection_enabled_summary: dict[str, float | int | bool],
    pair_metadata: dict[str, object],
    checkpoint_path: str,
    output_paths: dict[str, Path],
) -> pd.DataFrame:
#{
    if bool(projection_disabled_summary["projection_enabled"]):
        raise ValueError("Projection-disabled summary reports projection enabled.")

    if not bool(projection_enabled_summary["projection_enabled"]):
        raise ValueError("Projection-enabled summary reports projection disabled.")

    task_fields = (
        "episodes",
        "mean_return",
        "mean_length",
        "success_rate",
        "collision_rate",
        "mean_min_obstacle_clearance",
        "total_action_bound_clipping_count",
        "mean_action_bound_clipping_rate",
        "mean_speed_action_bound_clipping_rate",
        "mean_turn_rate_action_bound_clipping_rate",
        "mean_action_bound_clipping_norm",
        "max_action_bound_clipping_norm",
    )
    projection_fields = {
        "total_projection_interventions": "total_interventions",
        "mean_projection_intervention_rate": "mean_intervention_rate",
        "mean_projection_correction_norm": "mean_correction_norm",
        "max_projection_correction_norm": "max_correction_norm",
        "mean_projection_slack_sum": "mean_slack_sum",
        "max_projection_slack": "max_slack",
        "total_projection_solver_failures": "total_solver_failures",
    }
    delta_fields = (
        "mean_return",
        "mean_length",
        "success_rate",
        "collision_rate",
        "mean_min_obstacle_clearance",
        "mean_action_bound_clipping_rate",
        "mean_speed_action_bound_clipping_rate",
        "mean_turn_rate_action_bound_clipping_rate",
        "mean_action_bound_clipping_norm",
        "max_action_bound_clipping_norm",
    )

    row: dict[str, object] = {
        "policy": str(pair_metadata["method"]),
        "checkpoint": checkpoint_path,
    }
    row.update(pair_metadata)

    for field_name in task_fields:
        row[f"without_projection_{field_name}"] = projection_disabled_summary[field_name]
        row[f"with_projection_{field_name}"] = projection_enabled_summary[field_name]

    for field_name, output_name in projection_fields.items():
        row[f"with_projection_{output_name}"] = projection_enabled_summary[field_name]

    for field_name in delta_fields:
        row[f"{field_name}_delta_enabled_minus_disabled"] = (
            float(projection_enabled_summary[field_name])
            - float(projection_disabled_summary[field_name])
        )

    row["projection_disabled_csv"] = str(output_paths["projection_disabled"])
    row["projection_enabled_csv"] = str(output_paths["projection_enabled"])
    row["projection_disabled_trajectory_npz"] = str(
        output_paths["projection_disabled_trajectories"]
    )
    row["projection_enabled_trajectory_npz"] = str(
        output_paths["projection_enabled_trajectories"]
    )
    row["paired_episodes_csv"] = str(output_paths["paired_episodes"])
    row["paired_summary_csv"] = str(output_paths["paired_summary"])

    return pd.DataFrame([row])

#} End function build_paired_summary_table


# Write a DataFrame while creating its parent directory when required.
def write_table(table: pd.DataFrame, output_path: str | Path) -> None:
#{
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

#} End function write_table


# Print the paired comparison and all generated artifact locations.
def print_paired_summary(
    *,
    projection_disabled_summary: dict[str, float | int | bool],
    projection_enabled_summary: dict[str, float | int | bool],
    output_paths: dict[str, Path],
) -> None:
#{
    print("\nProjection-disabled evaluation")
    print("==============================")
    print_summary(projection_disabled_summary, output_paths["projection_disabled"])

    print("\nProjection-enabled evaluation")
    print("=============================")
    print_summary(projection_enabled_summary, output_paths["projection_enabled"])

    print("\nPaired comparison")
    print("-----------------")
    print(
        "mean_return_delta_enabled_minus_disabled:        "
        f"{float(projection_enabled_summary['mean_return']) - float(projection_disabled_summary['mean_return']):.6f}"
    )
    print(
        "success_rate_delta_enabled_minus_disabled:       "
        f"{float(projection_enabled_summary['success_rate']) - float(projection_disabled_summary['success_rate']):.6f}"
    )
    collision_rate_delta = (
        float(projection_enabled_summary["collision_rate"])
        - float(projection_disabled_summary["collision_rate"])
    )
    print(
        "collision_rate_delta_enabled_minus_disabled:     "
        f"{collision_rate_delta:.6f}"
    )
    print(
        "projection_disabled_trajectory_npz:             "
        f"{output_paths['projection_disabled_trajectories']}"
    )
    print(
        "projection_enabled_trajectory_npz:              "
        f"{output_paths['projection_enabled_trajectories']}"
    )
    print(f"paired_episodes_csv:                               {output_paths['paired_episodes']}")
    print(f"paired_summary_csv:                                {output_paths['paired_summary']}")

#} End function print_paired_summary


# Main paired evaluation routine.
def main() -> None:
#{
    args = parse_args()
    projection_params = make_projection_params(args)
    output_paths = make_output_paths(args.output_prefix)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = {
        "max_episode_steps": args.max_episode_steps,
        "max_obstacles": args.max_obstacles,
        "num_active_obstacles": args.num_active_obstacles,
        "collision_penalty": args.collision_penalty,
    }

    projection_disabled_factory = make_env(
        env_index=0,
        env_kwargs=env_kwargs,
        record_episode_statistics=False,
        normalize_actions=True,
        enable_projection=False,
        projection_params=projection_params,
    )
    projection_enabled_factory = make_env(
        env_index=0,
        env_kwargs=env_kwargs,
        record_episode_statistics=False,
        normalize_actions=True,
        enable_projection=True,
        projection_params=projection_params,
    )

    checkpoint_sha256 = file_sha256(args.checkpoint)
    agent, checkpoint = load_ppo_agent(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    training_metadata = checkpoint_training_metadata(
        checkpoint,
        method=args.method,
        train_seed=args.train_seed,
    )

    validate_checkpoint_environment_compatibility(
        checkpoint=checkpoint,
        env_factory=projection_disabled_factory,
    )
    validate_checkpoint_environment_compatibility(
        checkpoint=checkpoint,
        env_factory=projection_enabled_factory,
    )

    action_provider = make_ppo_action_provider(
        agent=agent,
        device=device,
        deterministic=not args.stochastic,
    )
    checkpoint_path = str(args.checkpoint)
    projection_disabled_trajectories: list[EpisodeTrajectory] = []
    projection_enabled_trajectories: list[EpisodeTrajectory] = []

    projection_disabled_results = run_evaluation_mode(
        env_factory=projection_disabled_factory,
        action_provider=action_provider,
        episodes=args.episodes,
        base_seed=args.seed,
        checkpoint_path=checkpoint_path,
        method=args.method,
        projection_enabled=False,
        device=device,
        trajectory_records=projection_disabled_trajectories,
    )
    projection_enabled_results = run_evaluation_mode(
        env_factory=projection_enabled_factory,
        action_provider=action_provider,
        episodes=args.episodes,
        base_seed=args.seed,
        checkpoint_path=checkpoint_path,
        method=args.method,
        projection_enabled=True,
        device=device,
        trajectory_records=projection_enabled_trajectories,
    )

    if file_sha256(args.checkpoint) != checkpoint_sha256:
        raise RuntimeError("Checkpoint file changed during paired evaluation.")

    pair_metadata = make_pair_metadata(
        args=args,
        projection_params=projection_params,
        checkpoint_sha256=checkpoint_sha256,
        training_metadata=training_metadata,
        device=device,
    )
    write_results_csv(
        results=projection_disabled_results,
        output_path=output_paths["projection_disabled"],
        projection_params=projection_params,
        row_metadata={**pair_metadata, "projection_mode": "disabled"},
    )
    write_results_csv(
        results=projection_enabled_results,
        output_path=output_paths["projection_enabled"],
        projection_params=projection_params,
        row_metadata={**pair_metadata, "projection_mode": "enabled"},
    )

    write_trajectory_archive(
        trajectories=projection_disabled_trajectories,
        output_path=output_paths["projection_disabled_trajectories"],
        projection_params=projection_params,
        run_metadata={**pair_metadata, "projection_mode": "disabled"},
    )
    write_trajectory_archive(
        trajectories=projection_enabled_trajectories,
        output_path=output_paths["projection_enabled_trajectories"],
        projection_params=projection_params,
        run_metadata={**pair_metadata, "projection_mode": "enabled"},
    )
    paired_episode_table = build_paired_episode_table(
        projection_disabled_results=projection_disabled_results,
        projection_enabled_results=projection_enabled_results,
        pair_metadata=pair_metadata,
    )
    write_table(paired_episode_table, output_paths["paired_episodes"])

    projection_disabled_summary = summarize_results(projection_disabled_results)
    projection_enabled_summary = summarize_results(projection_enabled_results)
    paired_summary_table = build_paired_summary_table(
        projection_disabled_summary=projection_disabled_summary,
        projection_enabled_summary=projection_enabled_summary,
        pair_metadata=pair_metadata,
        checkpoint_path=checkpoint_path,
        output_paths=output_paths,
    )
    write_table(paired_summary_table, output_paths["paired_summary"])

    print_paired_summary(
        projection_disabled_summary=projection_disabled_summary,
        projection_enabled_summary=projection_enabled_summary,
        output_paths=output_paths,
    )

#} End function main

# end region functions

if __name__ == "__main__":
    main()
