from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import gymnasium as gym
import numpy as np
import pandas as pd

from environments.factory import make_env
from algorithms.ppo.agent import Agent
from projection.cbf_qp_projection import ProjectionParams
from evaluation.trajectory_recording import (
    EpisodeTrajectory,
    append_episode_transition,
    create_episode_trajectory,
    physical_action_from_policy_action,
    write_trajectory_archive,
)

# Episode-level evaluation metrics returned by run_episode.
@dataclass
class EpisodeResult:
 #{
    policy: str
    checkpoint: str
    episode: int
    seed: int
    episode_return: float
    episode_length: int
    success: bool
    collision: bool
    terminated: bool
    truncated: bool
    final_distance_to_goal: float
    min_obstacle_clearance: float
    projection_enabled: bool
    projection_intervention_count: int
    projection_intervention_rate: float
    mean_projection_correction_norm: float
    max_projection_correction_norm: float
    mean_projection_slack_sum: float
    max_projection_slack: float
    projection_solver_failure_count: int
#} End dataclass EpisodeResult


# Define the signature for an action provider function.
ActionProvider = Callable[[gym.Env, np.ndarray], np.ndarray]


#################################################################################
# region functions

# Parse arguments for the evaluation
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Evaluate policies on the constrained navigation environment."
    )

    parser.add_argument(
        "--policy",
        choices=["random", "ppo"],
        default="random",
        help="Policy source to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a PPO checkpoint when --policy ppo.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Episode i uses seed + i.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/evaluation/random_policy_evaluation.csv"),
        help="CSV output path for one-row-per-episode results.",
    )

    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help="Optional compressed NPZ output for state, action, and projection trajectories.",
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
        help="Fixed obstacle capacity. This changes the observation dimension and must match a PPO checkpoint.",
    )

    parser.add_argument(
        "--num-active-obstacles",
        type=int,
        default=None,
        help="Number of active obstacles in the built-in layout. The default is min(3, max_obstacles).",
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

    parser.add_argument(
        "--enable-projection",
        action="store_true",
        help="Enable CBF-QP physical-action projection during evaluation.",
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

    if args.episodes <= 0:
        parser.error("--episodes must be positive.")

    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive.")

    if args.max_obstacles <= 0:
        parser.error("--max-obstacles must be positive.")

    if args.trajectory_output is not None and args.trajectory_output.suffix.lower() != ".npz":
        parser.error("--trajectory-output must use the .npz extension.")

    if args.num_active_obstacles is not None:
        if args.num_active_obstacles < 0 or args.num_active_obstacles > args.max_obstacles:
            parser.error("--num-active-obstacles must be between 0 and --max-obstacles.")
        if args.num_active_obstacles > 3:
            parser.error("The built-in layout defines at most three active obstacles.")

    if not np.isfinite(args.projection_lookahead_distance) or args.projection_lookahead_distance < 0.0:
        parser.error("--projection-lookahead-distance must be finite and nonnegative.")

    if not np.isfinite(args.projection_alpha) or args.projection_alpha <= 0.0:
        parser.error("--projection-alpha must be finite and positive.")

    if not np.isfinite(args.projection_slack_penalty) or args.projection_slack_penalty <= 0.0:
        parser.error("--projection-slack-penalty must be finite and positive.")

    if not np.isfinite(args.projection_extra_clearance) or args.projection_extra_clearance < 0.0:
        parser.error("--projection-extra-clearance must be finite and nonnegative.")

    if args.policy == "ppo" and args.checkpoint is None:
        parser.error("--checkpoint is required when --policy ppo.")

    return args
#} End function parse_args


# Build one explicit projection configuration from evaluator arguments.
def make_projection_params(args: argparse.Namespace) -> ProjectionParams:
#{
    return ProjectionParams(
        lookahead_distance=float(args.projection_lookahead_distance),
        alpha=float(args.projection_alpha),
        slack_penalty=float(args.projection_slack_penalty),
        extra_clearance=float(args.projection_extra_clearance),
    )

#} End function make_projection_params


# Return one random action from the environment action space.
# The observation argument is accepted for signature compatibility with later
# policy-based action providers, but it is unused here.
def random_action(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    del obs
    return env.action_space.sample()


# Run one complete episode and return episode-level metrics.
def run_episode(
    env: gym.Env,
    action_provider: ActionProvider,
    seed: int,
    episode: int,
    policy_name: str,
    checkpoint_path: str = "",
    trajectory_records: list[EpisodeTrajectory] | None = None,
    reset_options: dict | None = None,
) -> EpisodeResult:
#{
    if reset_options is None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset(seed=seed, options=reset_options)

    if not np.all(np.isfinite(obs)):
        raise RuntimeError("Non-finite observation returned by env.reset().")

    episode_trajectory = (
        create_episode_trajectory(
            env=env,
            initial_info=info,
            policy_name=policy_name,
            checkpoint_path=checkpoint_path,
            episode=episode,
            seed=seed,
        )
        if trajectory_records is not None
        else None
    )

    result = EpisodeResult(
        policy=policy_name,
        checkpoint=checkpoint_path,
        episode=int(episode),
        seed=int(seed),
        episode_return=0.0,
        episode_length=0,
        success=bool(info.get("success", False)),
        collision=bool(info.get("collision", False)),
        terminated=False,
        truncated=False,
        final_distance_to_goal=float(info.get("distance_to_goal", np.inf)),
        min_obstacle_clearance=float(info.get("min_obstacle_clearance", np.nan)),
        projection_enabled=False,
        projection_intervention_count=0,
        projection_intervention_rate=0.0,
        mean_projection_correction_norm=0.0,
        max_projection_correction_norm=0.0,
        mean_projection_slack_sum=0.0,
        max_projection_slack=0.0,
        projection_solver_failure_count=0,
    )

    terminated = False
    truncated = False

    # Safety guard against accidental infinite loops if an environment bug
    # prevents termination or truncation.
    unwrapped_env = getattr(env, "unwrapped", env)
    max_steps_guard = int(getattr(unwrapped_env, "max_episode_steps", 10_000)) + 1
    
    # Accumulate per-step values before computing episode means.
    projection_correction_sum = 0.0
    projection_slack_sum_total = 0.0
    
    while not (terminated or truncated):
    #{
        action = action_provider(env, obs)
        action_raw_physical = (
            physical_action_from_policy_action(env, action)
            if episode_trajectory is not None
            else None
        )

        obs, reward, terminated, truncated, info = env.step(action)

        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Non-finite observation encountered in episode {episode}.")

        if not np.isfinite(reward):
            raise RuntimeError(f"Non-finite reward encountered in episode {episode}.")

        if episode_trajectory is not None:
            append_episode_transition(
                trajectory=episode_trajectory,
                env=env,
                action_raw_normalized=action,
                action_raw_physical=action_raw_physical,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

        # Accumulate projection diagnostics reported for this environment step.
        if bool(info.get("projection_enabled", False)):
        #{
            result.projection_enabled = True

            projection_intervened = bool(info["projection_intervened"])
            projection_correction_norm = float(info["projection_correction_norm"])
            projection_slack_max = float(info["projection_slack_max"])
            projection_slack_sum = float(info["projection_slack_sum"])
            projection_success = bool(info["projection_success"])

            if projection_intervened:
                result.projection_intervention_count += 1

            projection_correction_sum += projection_correction_norm

            result.max_projection_correction_norm = max(
                result.max_projection_correction_norm,
                projection_correction_norm,
            )

            # A failed solve has no trustworthy slack solution. Preserve NaN for
            # both episode slack statistics instead of reporting a numeric value.
            if not projection_success:
                result.projection_solver_failure_count += 1
                result.mean_projection_slack_sum = float("nan")
                result.max_projection_slack = float("nan")
            elif not np.isfinite(projection_slack_max) or not np.isfinite(projection_slack_sum):
                raise RuntimeError(
                    "Projection reported success with non-finite slack diagnostics."
                )
            else:
                projection_slack_sum_total += projection_slack_sum

                if np.isfinite(result.max_projection_slack):
                    result.max_projection_slack = max(
                        result.max_projection_slack,
                        projection_slack_max,
                    )

        #} End if projection_enabled

        result.episode_return += float(reward)
        result.episode_length += 1

        result.success = bool(info.get("success", False))
        result.collision = bool(info.get("collision", False))
        result.terminated = bool(terminated)
        result.truncated = bool(truncated)
        result.final_distance_to_goal = float(info.get("distance_to_goal", np.inf))

        current_min_clearance = float(info.get("min_obstacle_clearance", np.nan))

        if np.isnan(result.min_obstacle_clearance):
            result.min_obstacle_clearance = current_min_clearance
        elif not np.isnan(current_min_clearance):
            result.min_obstacle_clearance = min(
                result.min_obstacle_clearance,
                current_min_clearance,
            )

        if result.episode_length > max_steps_guard:
            raise RuntimeError(
                "Episode exceeded the safety step guard. "
                "Check environment termination/truncation logic."
            )

    #} End loop

    # Compute episode-level projection statistics.
    if result.projection_enabled and result.episode_length > 0:
        result.projection_intervention_rate = result.projection_intervention_count / result.episode_length
        result.mean_projection_correction_norm = projection_correction_sum / result.episode_length

        if result.projection_solver_failure_count == 0:
            result.mean_projection_slack_sum = projection_slack_sum_total / result.episode_length

    if episode_trajectory is not None:
        if len(episode_trajectory.rewards) != result.episode_length:
            raise RuntimeError("Trajectory length does not match the episode result length.")

        trajectory_records.append(episode_trajectory)

    return result
#} End function run_episode


# Load a PPO agent from a checkpoint and return the agent and checkpoint dictionary.
def load_ppo_agent(*, checkpoint_path: str | Path, device: torch.device) -> tuple[Agent, dict]:
#{
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    obs_dim = int(checkpoint["obs_dim"])
    action_dim = int(checkpoint["action_dim"])

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    return agent, checkpoint
#} End function load_ppo_agent


# Create an action provider compatible with run_episode(...).
#
# The evaluator passes one observation at a time as a NumPy array with
# shape (obs_dim,). The PPO agent expects a Torch tensor with batch shape
# (1, obs_dim).
#
# deterministic=True uses the actor mean, which is the standard first
# choice for evaluation. deterministic=False samples from the Gaussian
# policy, matching the stochastic training-time policy behavior.
def make_ppo_action_provider(*, agent: Agent, device: torch.device, deterministic: bool) -> ActionProvider:
#{
    agent.eval()

    def ppo_action(env: gym.Env, obs: np.ndarray) -> np.ndarray:
        # env is accepted to match the ActionProvider signature.
        # It is not used here because the PPO action depends only on obs.
        _ = env

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action_tensor = agent.actor_mean(obs_tensor)
            else:
                action_tensor, _, _, _ = agent.get_action_and_value(obs_tensor)

        action = action_tensor.squeeze(0).detach().cpu().numpy()

        if not np.all(np.isfinite(action)):
            raise RuntimeError(f"PPO produced non-finite action: {action}")

        return action.astype(np.float32, copy=False)

    return ppo_action
#} End function make_ppo_action_provider


# Create one environment only to inspect observation/action spaces.
# This does not run an episode and does not call PPO.
def validate_checkpoint_environment_compatibility(*, checkpoint: dict, env_factory) -> None:
#{
    env = env_factory()

    try:
        env_obs_dim = int(np.prod(env.observation_space.shape))
        env_action_dim = int(np.prod(env.action_space.shape))

        checkpoint_obs_dim = int(checkpoint["obs_dim"])
        checkpoint_action_dim = int(checkpoint["action_dim"])

        if env_obs_dim != checkpoint_obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={checkpoint_obs_dim} does not match "
                f"environment obs_dim={env_obs_dim}."
            )

        if env_action_dim != checkpoint_action_dim:
            raise ValueError(
                f"Checkpoint action_dim={checkpoint_action_dim} does not match "
                f"environment action_dim={env_action_dim}."
            )

    finally:
        env.close()

#} End function validate_checkpoint_environment_compatibility


# Return the projection hyperparameters persisted with every evaluation row.
def projection_parameter_metadata(params: ProjectionParams) -> dict[str, float]:
#{
    return {
        "projection_lookahead_distance": float(params.lookahead_distance),
        "projection_alpha": float(params.alpha),
        "projection_slack_penalty": float(params.slack_penalty),
        "projection_extra_clearance": float(params.extra_clearance),
    }

#} End function projection_parameter_metadata


# Write one CSV row per evaluated episode.
def write_results_csv(
    *,
    results: list[EpisodeResult],
    output_path: str | Path,
    projection_params: ProjectionParams,
) -> None:
#{
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    projection_metadata = projection_parameter_metadata(projection_params)
    rows = [{**asdict(result), **projection_metadata} for result in results]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

#} End function write_results_csv


# Compute a compact summary from episode results.
def summarize_results(results: list[EpisodeResult]) -> dict[str, float | int | bool]:
#{
    if not results:
        raise ValueError("Cannot summarize an empty result list.")

    returns = np.asarray([result.episode_return for result in results], dtype=np.float64)
    lengths = np.asarray([result.episode_length for result in results], dtype=np.float64)
    successes = np.asarray([float(result.success) for result in results], dtype=np.float64)
    collisions = np.asarray([float(result.collision) for result in results], dtype=np.float64)
    min_clearances = np.asarray([result.min_obstacle_clearance for result in results], dtype=np.float64)
    defined_clearances = min_clearances[np.isfinite(min_clearances)]
    mean_min_obstacle_clearance = (
        float(np.mean(defined_clearances))
        if defined_clearances.size > 0
        else float("nan")
    )

    projection_enabled = bool(results[0].projection_enabled)

    summary: dict[str, float | int | bool] = {
        "episodes": len(results),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_min_obstacle_clearance": mean_min_obstacle_clearance,
        "projection_enabled": projection_enabled,
    }

    if projection_enabled:
        projection_intervention_rates = np.asarray([result.projection_intervention_rate for result in results], dtype=np.float64)
        projection_mean_corrections = np.asarray([result.mean_projection_correction_norm for result in results], dtype=np.float64)
        projection_max_corrections = np.asarray([result.max_projection_correction_norm for result in results], dtype=np.float64)
        projection_mean_slack_sums = np.asarray([result.mean_projection_slack_sum for result in results], dtype=np.float64)
        projection_max_slacks = np.asarray([result.max_projection_slack for result in results], dtype=np.float64)
        total_projection_solver_failures = int(sum(result.projection_solver_failure_count for result in results))
        mean_projection_slack_sum = (
            float(np.mean(projection_mean_slack_sums))
            if total_projection_solver_failures == 0 and np.all(np.isfinite(projection_mean_slack_sums))
            else float("nan")
        )
        max_projection_slack = (
            float(np.max(projection_max_slacks))
            if total_projection_solver_failures == 0 and np.all(np.isfinite(projection_max_slacks))
            else float("nan")
        )

        summary["total_projection_interventions"] = int(sum(result.projection_intervention_count for result in results))
        summary["mean_projection_intervention_rate"] = float(np.mean(projection_intervention_rates))
        summary["mean_projection_correction_norm"] = float(np.mean(projection_mean_corrections))
        summary["max_projection_correction_norm"] = float(np.max(projection_max_corrections))
        summary["mean_projection_slack_sum"] = mean_projection_slack_sum
        summary["max_projection_slack"] = max_projection_slack
        summary["total_projection_solver_failures"] = total_projection_solver_failures

    return summary

#} End function summarize_results

# Print a concise command-line summary of the evaluation results.
def print_summary(summary: dict[str, float | int | bool], output_path: str | Path) -> None:
#{
    projection_enabled = bool(summary["projection_enabled"])
    mean_min_obstacle_clearance = float(summary["mean_min_obstacle_clearance"])
    mean_min_obstacle_clearance_text = (
        f"{mean_min_obstacle_clearance:.3f}"
        if np.isfinite(mean_min_obstacle_clearance)
        else "N/A"
    )

    print("Evaluation summary")
    print("------------------")
    print(f"episodes:                           {int(summary['episodes'])}")
    print(f"mean_return:                        {float(summary['mean_return']):.3f}")
    print(f"mean_length:                        {float(summary['mean_length']):.3f}")
    print(f"success_rate:                       {float(summary['success_rate']):.3f}")
    print(f"collision_rate:                     {float(summary['collision_rate']):.3f}")
    print(f"mean_min_obstacle_clearance:        {mean_min_obstacle_clearance_text}")
    print(f"projection_enabled:                 {projection_enabled}")

    if projection_enabled:
    #{
        mean_projection_slack_sum = float(summary["mean_projection_slack_sum"])
        mean_projection_slack_sum_text = (
            f"{mean_projection_slack_sum:.6f}"
            if np.isfinite(mean_projection_slack_sum)
            else "N/A"
        )
        max_projection_slack = float(summary["max_projection_slack"])
        max_projection_slack_text = (
            f"{max_projection_slack:.6f}"
            if np.isfinite(max_projection_slack)
            else "N/A"
        )

        print(f"total_projection_interventions:     {int(summary['total_projection_interventions'])}")
        print(f"mean_projection_intervention_rate:  {float(summary['mean_projection_intervention_rate']):.3f}")
        print(f"mean_projection_correction_norm:    {float(summary['mean_projection_correction_norm']):.3f}")
        print(f"max_projection_correction_norm:     {float(summary['max_projection_correction_norm']):.3f}")
        print(f"mean_projection_slack_sum:          {mean_projection_slack_sum_text}")
        print(f"max_projection_slack:               {max_projection_slack_text}")
        print(f"total_projection_solver_failures:   {int(summary['total_projection_solver_failures'])}")

    #} End if projection_enabled

    print(f"csv_output:                         {Path(output_path)}")

#} End function print_summary


# Main evaluation routine.
def main() -> None:
#{
    args = parse_args()
    projection_params = make_projection_params(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = {
        "max_episode_steps": args.max_episode_steps,
        "max_obstacles": args.max_obstacles,
        "num_active_obstacles": args.num_active_obstacles,
    }

    env_factory = make_env(
        env_index=0,
        env_kwargs=env_kwargs,
        record_episode_statistics=False,
        normalize_actions=True,
        enable_projection=args.enable_projection,
        projection_params=projection_params,
    )

    if args.policy == "random":
        action_provider = random_action
        checkpoint_label = ""

    elif args.policy == "ppo":
        agent, checkpoint = load_ppo_agent(
            checkpoint_path=args.checkpoint,
            device=device,
        )

        validate_checkpoint_environment_compatibility(checkpoint=checkpoint, env_factory=env_factory)

        action_provider = make_ppo_action_provider(agent=agent, device=device, deterministic=not args.stochastic)

        checkpoint_label = str(args.checkpoint)

    else:
        raise ValueError(f"Unsupported policy: {args.policy}")

    results: list[EpisodeResult] = []
    trajectory_records: list[EpisodeTrajectory] | None = (
        [] if args.trajectory_output is not None else None
    )

    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index

        env = env_factory()

        try:
            env.action_space.seed(episode_seed)
            torch.manual_seed(episode_seed)

            result = run_episode(
                env=env,
                action_provider=action_provider,
                seed=episode_seed,
                episode=episode_index,
                policy_name=args.policy,
                checkpoint_path=checkpoint_label,
                trajectory_records=trajectory_records,
            )

            results.append(result)

        finally:
            env.close()

    write_results_csv(results=results, output_path=args.output, projection_params=projection_params)

    if trajectory_records is not None:
        resolved_active_count = (
            min(3, args.max_obstacles)
            if args.num_active_obstacles is None
            else args.num_active_obstacles
        )
        write_trajectory_archive(
            trajectories=trajectory_records,
            output_path=args.trajectory_output,
            projection_params=projection_params,
            run_metadata={
                "policy": args.policy,
                "stochastic": bool(args.stochastic),
                "device": str(device),
                "max_episode_steps": int(args.max_episode_steps),
                "max_obstacles": int(args.max_obstacles),
                "num_active_obstacles": int(resolved_active_count),
            },
        )

    summary = summarize_results(results)
    print_summary(summary, args.output)

    if args.trajectory_output is not None:
        print(f"trajectory_output:                   {args.trajectory_output}")

#} End function main

# end region functions

if __name__ == "__main__":
    main()