from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd

from environments.factory import make_env


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    episode_return: float
    episode_length: int
    success: bool
    collision: bool
    terminated: bool
    truncated: bool
    final_distance_to_goal: float
    min_obstacle_distance: float


ActionProvider = Callable[[gym.Env, np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the P1 constrained navigation environment."
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
        default=Path("results/tables/random_policy_evaluation.csv"),
        help="CSV output path for one-row-per-episode results.",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Maximum number of environment steps per episode.",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive.")

    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive.")

    return args


def random_action(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    # Return one random action from the environment action space.
    # The observation argument is accepted for signature compatibility with later
    # policy-based action providers, but it is unused here.
    
    del obs
    return env.action_space.sample()


def run_episode(
    env: gym.Env,
    action_provider: ActionProvider,
    seed: int,
    episode: int,
) -> EpisodeResult:
    #Run one complete episode and return episode-level metrics.

    obs, info = env.reset(seed=seed)

    if not np.all(np.isfinite(obs)):
        raise RuntimeError("Non-finite observation returned by env.reset().")

    result = EpisodeResult(
        episode=int(episode),
        seed=int(seed),
        episode_return=0.0,
        episode_length=0,
        success=bool(info.get("success", False)),
        collision=bool(info.get("collision", False)),
        terminated=False,
        truncated=False,
        final_distance_to_goal=float(info.get("distance_to_goal", np.inf)),
        min_obstacle_distance=float(info.get("min_obstacle_distance", np.inf)),
    )

    terminated = False
    truncated = False

    # Safety guard against accidental infinite loops if an environment bug
    # prevents termination or truncation.
    unwrapped_env = getattr(env, "unwrapped", env)
    max_steps_guard = int(getattr(unwrapped_env, "max_episode_steps", 10_000)) + 1

    while not (terminated or truncated):
        action = action_provider(env, obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if not np.all(np.isfinite(obs)):
            raise RuntimeError(
                f"Non-finite observation encountered in episode {episode}."
            )

        if not np.isfinite(reward):
            raise RuntimeError(
                f"Non-finite reward encountered in episode {episode}."
            )

        result.episode_return += float(reward)
        result.episode_length += 1

        result.success = bool(info.get("success", False))
        result.collision = bool(info.get("collision", False))
        result.terminated = bool(terminated)
        result.truncated = bool(truncated)

        result.final_distance_to_goal = float(info.get("distance_to_goal", np.inf))

        current_min_distance = float(info.get("min_obstacle_distance", np.inf))
        result.min_obstacle_distance = min(result.min_obstacle_distance, current_min_distance)

        if result.episode_length > max_steps_guard:
            raise RuntimeError(
                "Episode exceeded the safety step guard. "
                "Check environment termination/truncation logic."
            )

    return result


def write_results_csv(*, results: list[EpisodeResult], output_path: str | Path) -> None:
    # Write one CSV row per evaluated episode.
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(result) for result in results]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def summarize_results(results: list[EpisodeResult]) -> dict[str, float | int]:
    # Compute a compact terminal summary from episode results.

    if not results:
        raise ValueError("Cannot summarize an empty result list.")

    returns = np.asarray([r.episode_return for r in results], dtype=np.float64)
    lengths = np.asarray([r.episode_length for r in results], dtype=np.float64)
    successes = np.asarray([float(r.success) for r in results], dtype=np.float64)
    collisions = np.asarray([float(r.collision) for r in results], dtype=np.float64)
    min_distances = np.asarray([r.min_obstacle_distance for r in results], dtype=np.float64)

    return {
        "episodes": len(results),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_min_obstacle_distance": float(np.mean(min_distances)),
    }


def print_summary(summary: dict[str, float | int], output_path: str | Path) -> None:
    # Print a concise command-line summary.

    print("Evaluation summary")
    print("------------------")
    print(f"episodes:                  {summary['episodes']}")
    print(f"mean_return:               {summary['mean_return']:.3f}")
    print(f"mean_length:               {summary['mean_length']:.3f}")
    print(f"success_rate:              {summary['success_rate']:.3f}")
    print(f"collision_rate:            {summary['collision_rate']:.3f}")
    print(
        "mean_min_obstacle_distance:"
        f" {summary['mean_min_obstacle_distance']:.3f}"
    )
    print(f"csv_output:                {Path(output_path)}")


def main() -> None:
    args = parse_args()

    env_kwargs = {
        "max_episode_steps": args.max_episode_steps,
    }

    env_factory = make_env(
        env_index=0,
        env_kwargs=env_kwargs,
        record_episode_statistics=False,
    )

    results: list[EpisodeResult] = []

    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index

        env = env_factory()

        try:
            # Seed random action sampling for reproducibility.
            env.action_space.seed(episode_seed)

            result = run_episode(
                env=env,
                action_provider=random_action,
                seed=episode_seed,
                episode=episode_index,
            )

            results.append(result)

        finally:
            env.close()

    write_results_csv(
        results=results,
        output_path=args.output,
    )

    summary = summarize_results(results)
    print_summary(summary, args.output)


if __name__ == "__main__":
    main()