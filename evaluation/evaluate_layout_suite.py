from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from environments.factory import make_env
from evaluation.evaluate_policy import (
    load_ppo_agent,
    make_ppo_action_provider,
    make_projection_params,
    print_summary,
    projection_parameter_metadata,
    run_episode,
    summarize_results,
    validate_checkpoint_environment_compatibility,
)
from evaluation.layout_suite import file_sha256, load_navigation_layout_suite
from evaluation.trajectory_recording import EpisodeTrajectory, write_trajectory_archive
from projection.cbf_qp_projection import ProjectionParams


#################################################################################
# region Functions

# Parse one common-layout checkpoint evaluation configuration.
def parse_args() -> argparse.Namespace:
#{
    parser = argparse.ArgumentParser(
        description="Evaluate one PPO checkpoint on a deterministic navigation layout suite."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layout-suite", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--projection-mode", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--repeats-per-layout", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--collision-penalty", type=float, default=10.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, default=None)

    defaults = ProjectionParams()
    parser.add_argument(
        "--projection-lookahead-distance",
        type=float,
        default=defaults.lookahead_distance,
    )
    parser.add_argument("--projection-alpha", type=float, default=defaults.alpha)
    parser.add_argument(
        "--projection-slack-penalty",
        type=float,
        default=defaults.slack_penalty,
    )
    parser.add_argument(
        "--projection-extra-clearance",
        type=float,
        default=defaults.extra_clearance,
    )
    args = parser.parse_args()

    if not args.method.strip():
        parser.error("--method must be nonempty.")
    if args.repeats_per_layout <= 0:
        parser.error("--repeats-per-layout must be positive.")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive.")
    if not np.isfinite(args.collision_penalty) or args.collision_penalty < 0.0:
        parser.error("--collision-penalty must be finite and nonnegative.")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must use the .csv extension.")
    if args.trajectory_output is not None and args.trajectory_output.suffix.lower() != ".npz":
        parser.error("--trajectory-output must use the .npz extension.")
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


# Validate explicit labels and return the training fields stored in the checkpoint.
def checkpoint_metadata(checkpoint: dict, *, method: str, train_seed: int) -> dict[str, object]:
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
        "training_collision_penalty": float(
            checkpoint_args.get("collision_penalty", np.nan)
        ),
        "training_projection_enabled": bool(
            checkpoint_args.get("enable_projection", False)
        ),
    }

#} End function checkpoint_metadata


# Run one complete common-layout checkpoint evaluation.
def main() -> None:
#{
    args = parse_args()
    suite = load_navigation_layout_suite(args.layout_suite)
    projection_params = make_projection_params(args)
    projection_enabled = args.projection_mode == "enabled"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env_factory = make_env(
        env_index=0,
        env_kwargs={
            "max_episode_steps": args.max_episode_steps,
            "max_obstacles": suite.max_obstacles,
            "num_active_obstacles": 0,
            "agent_radius": suite.agent_radius,
            "goal_radius": suite.goal_radius,
            "collision_penalty": args.collision_penalty,
        },
        record_episode_statistics=False,
        normalize_actions=True,
        enable_projection=projection_enabled,
        projection_params=projection_params,
    )

    agent, checkpoint = load_ppo_agent(checkpoint_path=args.checkpoint, device=device)
    validate_checkpoint_environment_compatibility(checkpoint=checkpoint, env_factory=env_factory)
    training_metadata = checkpoint_metadata(
        checkpoint,
        method=args.method,
        train_seed=args.train_seed,
    )
    checkpoint_sha256 = file_sha256(args.checkpoint)
    action_provider = make_ppo_action_provider(
        agent=agent,
        device=device,
        deterministic=not args.stochastic,
    )
    projection_metadata = projection_parameter_metadata(projection_params)
    policy_mode = "stochastic" if args.stochastic else "deterministic"
    trajectories: list[EpisodeTrajectory] | None = (
        [] if args.trajectory_output is not None else None
    )
    results = []
    rows: list[dict[str, object]] = []
    episode_index = 0

    for layout in suite.layouts:
    #{
        for layout_repeat in range(args.repeats_per_layout):
        #{
            evaluation_seed = args.seed + episode_index
            env = env_factory()

            try:
                env.action_space.seed(evaluation_seed)
                torch.manual_seed(evaluation_seed)

                if device.type == "cuda":
                    torch.cuda.manual_seed_all(evaluation_seed)

                result = run_episode(
                    env=env,
                    action_provider=action_provider,
                    seed=evaluation_seed,
                    episode=episode_index,
                    policy_name=args.method,
                    checkpoint_path=str(args.checkpoint),
                    trajectory_records=trajectories,
                    reset_options=layout.reset_options(),
                )
            finally:
                env.close()

            if result.projection_enabled != projection_enabled:
                raise RuntimeError(
                    "Environment projection mode did not match the layout evaluation configuration."
                )

            results.append(result)
            rows.append(
                {
                    "method": args.method,
                    "train_seed": int(args.train_seed),
                    **training_metadata,
                    "checkpoint_sha256": checkpoint_sha256,
                    "layout_suite_schema_version": suite.schema_version,
                    "layout_suite_id": suite.suite_id,
                    "layout_suite_sha256": suite.sha256,
                    "layout_suite_path": str(suite.source_path),
                    "layout_id": layout.layout_id,
                    "layout_repeat": int(layout_repeat),
                    "evaluation_seed": int(evaluation_seed),
                    "evaluation_collision_penalty": float(args.collision_penalty),
                    "evaluation_policy_mode": policy_mode,
                    "max_episode_steps": int(args.max_episode_steps),
                    "layout_max_obstacles": int(suite.max_obstacles),
                    "layout_agent_radius": float(suite.agent_radius),
                    "layout_goal_radius": float(suite.goal_radius),
                    "projection_mode": args.projection_mode,
                    **asdict(result),
                    **projection_metadata,
                }
            )
            episode_index += 1

        #} End loop layout repeats
    #} End loop layouts

    if file_sha256(args.checkpoint) != checkpoint_sha256:
        raise RuntimeError("Checkpoint file changed during layout-suite evaluation.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    if trajectories is not None:
        write_trajectory_archive(
            trajectories=trajectories,
            output_path=args.trajectory_output,
            projection_params=projection_params,
            run_metadata={
                "method": args.method,
                "train_seed": int(args.train_seed),
                **training_metadata,
                "checkpoint_sha256": checkpoint_sha256,
                "layout_suite_schema_version": suite.schema_version,
                "layout_suite_id": suite.suite_id,
                "layout_suite_sha256": suite.sha256,
                "layout_suite_path": str(suite.source_path),
                "layout_count": len(suite.layouts),
                "repeats_per_layout": int(args.repeats_per_layout),
                "base_seed": int(args.seed),
                "last_seed": int(args.seed + len(results) - 1),
                "projection_mode": args.projection_mode,
                "stochastic": bool(args.stochastic),
                "device": str(device),
                "max_episode_steps": int(args.max_episode_steps),
                "evaluation_collision_penalty": float(args.collision_penalty),
                "max_obstacles": int(suite.max_obstacles),
                "agent_radius": float(suite.agent_radius),
                "goal_radius": float(suite.goal_radius),
            },
        )

    print("Layout suite")
    print("------------")
    print(f"suite_id:                           {suite.suite_id}")
    print(f"suite_sha256:                       {suite.sha256}")
    print(f"layouts:                            {len(suite.layouts)}")
    print(f"repeats_per_layout:                 {args.repeats_per_layout}")
    print(f"projection_mode:                    {args.projection_mode}")
    print_summary(summarize_results(results), args.output)

    if args.trajectory_output is not None:
        print(f"trajectory_output:                  {args.trajectory_output}")

#} End function main

# end region Functions


if __name__ == "__main__":
    main()
