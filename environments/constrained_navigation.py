from __future__ import annotations
from logging import info

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ConstrainedNavigationEnv(gym.Env):
    
    metadata = {"render_modes": []}

    ############################################################################
    # Constructor
    ############################################################################

    def __init__(
        self,
        *,
        max_obstacles: int = 3,
        num_active_obstacles: int | None = None,
        dt: float = 0.1,
        v_max: float = 1.0,
        omega_max: float = 2.0,
        goal_radius: float = 0.25,
        agent_radius: float = 0.10,
        max_episode_steps: int = 200,
        time_penalty: float = 0.01,
        timeout_distance_penalty: float = 1.0,
        collision_penalty: float = 10.0,
    ) -> None:
        super().__init__()

        self.max_obstacles = int(max_obstacles)          # fixed observation capacity

        if self.max_obstacles <= 0:
            raise ValueError("max_obstacles must be positive.")

        default_obstacle_count = 3
        default_active_count = min(default_obstacle_count, self.max_obstacles)
        self.num_active_obstacles = (
            default_active_count
            if num_active_obstacles is None
            else int(num_active_obstacles)
        )

        if self.num_active_obstacles < 0 or self.num_active_obstacles > self.max_obstacles:
            raise ValueError("num_active_obstacles must be between 0 and max_obstacles.")
        if self.num_active_obstacles > default_obstacle_count:
            raise ValueError("The built-in layout defines at most three active obstacles.")

        self.dt = float(dt)                              # integration step
        self.v_max = float(v_max)                        # speed bound
        self.omega_max = float(omega_max)                # turn-rate bound
        self.goal_radius = float(goal_radius)            # success threshold
        self.agent_radius = float(agent_radius)          # collision footprint
        self.max_episode_steps = int(max_episode_steps)  # time-limit truncation
        self.time_penalty = float(time_penalty)
        self.timeout_distance_penalty = float(timeout_distance_penalty)
        self.collision_penalty = float(collision_penalty)

        if not np.isfinite(self.collision_penalty) or self.collision_penalty < 0.0:
            raise ValueError("collision_penalty must be finite and nonnegative.")

        self.obs_dim = 6 + 5 * self.max_obstacles        # fixed neural-network input size

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.asarray([0.0, -self.omega_max], dtype=np.float32),
            high=np.asarray([self.v_max, self.omega_max], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal episode state. These are assigned properly in reset().
        self.position = np.zeros(2, dtype=np.float64)
        self.theta = 0.0
        self.goal = np.zeros(2, dtype=np.float64)

        self.obstacle_centers = np.zeros((self.max_obstacles, 2), dtype=np.float64)
        self.obstacle_radii = np.zeros(self.max_obstacles, dtype=np.float64)
        self.obstacle_mask = np.zeros(self.max_obstacles, dtype=bool)

        self.step_count = 0
        self.prev_distance_to_goal = 0.0

        self.progress_weight: float = 1.0
        self.action_penalty: float = 0.01
        self.goal_reward: float = 10.0

    ############################################################################
    # Reset
    # Start a new episode and return the initial observation.
    # The options dictionary is used for deterministic smoke checks and
    # controlled layouts. If options is None, a simple default layout is used.
    ############################################################################
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        options = {} if options is None else dict(options)

        self.step_count = 0

        # Default deterministic layout.
        default_start = np.asarray([0.0, 0.0], dtype=np.float64)
        default_theta = 0.0
        default_goal = np.asarray([4.0, 0.0], dtype=np.float64)

        built_in_obstacle_centers = np.asarray(
            [
                [1.5, 0.5],
                [2.5, -0.5],
                [3.2, 0.4],
            ],
            dtype=np.float64,
        )
        built_in_obstacle_radii = np.asarray([0.30, 0.30, 0.25], dtype=np.float64)

        default_obstacle_centers = np.zeros((self.max_obstacles, 2), dtype=np.float64)
        default_obstacle_radii = np.zeros(self.max_obstacles, dtype=np.float64)
        default_obstacle_mask = np.zeros(self.max_obstacles, dtype=bool)

        built_in_count = min(self.max_obstacles, built_in_obstacle_centers.shape[0])
        default_obstacle_centers[:built_in_count] = built_in_obstacle_centers[:built_in_count]
        default_obstacle_radii[:built_in_count] = built_in_obstacle_radii[:built_in_count]
        default_obstacle_mask[:self.num_active_obstacles] = True

        self.position = np.asarray(options.get("start", default_start), dtype=np.float64).reshape(2)
        self.theta = float(options.get("theta", default_theta))
        self.goal = np.asarray(options.get("goal", default_goal), dtype=np.float64).reshape(2)

        self.obstacle_centers = np.asarray(
            options.get("obstacle_centers", default_obstacle_centers),
            dtype=np.float64,
        ).reshape(self.max_obstacles, 2)

        self.obstacle_radii = np.asarray(
            options.get("obstacle_radii", default_obstacle_radii),
            dtype=np.float64,
        ).reshape(self.max_obstacles)

        self.obstacle_mask = np.asarray(
            options.get("obstacle_mask", default_obstacle_mask),
            dtype=bool,
        ).reshape(self.max_obstacles)

        self.prev_distance_to_goal = self._distance_to_goal()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info


    ############################################################################
    # Action
    #
    # Advance the environment by one timestep.
    #
    # The environment receives an executed action u = [v, omega],
    # applies one explicit-Euler unicycle update, computes reward and
    # termination flags, and returns the Gymnasium five-tuple:
    #     obs, reward, terminated, truncated, info
    ############################################################################
    def step(self, action):
        
        # Convert external action into a clean NumPy vector.
        action = np.asarray(action, dtype=np.float32).reshape(2)

        if not np.all(np.isfinite(action)):
            raise ValueError(f"Action contains non-finite values: {action}")

        # Enforce the action bounds declared by action_space.
        action = np.clip(action, self.action_space.low, self.action_space.high)

        v = float(action[0])
        omega = float(action[1])

        # Store pre-step distance for progress reward.
        prev_distance_to_goal = self._distance_to_goal()

        # Apply unicycle dynamics.
        # Position is updated using the heading at the start of the step.
        x, y = self.position
        theta = self.theta

        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = self._wrap_angle(theta + omega * self.dt)

        self.position = np.asarray([x_next, y_next], dtype=np.float64)
        self.theta = float(theta_next)

        # Advance episode counter before getting info 
        # and before computing truncation.
        self.step_count += 1

        # Compute post-step diagnostics.
        info = self._get_info()

        distance_to_goal = float(info["distance_to_goal"])
        success = bool(info["success"])
        collision = bool(info["collision"])

        # Compute shaped reward.
        # progress > 0 means the action moved the agent closer to the goal.
        progress = prev_distance_to_goal - distance_to_goal
        action_cost = v * v + omega * omega

        reward = ( 
            self.progress_weight * progress 
            - self.action_penalty * action_cost 
            - self.time_penalty
        )

        if success:
            reward += self.goal_reward

        if collision:
            reward -= self.collision_penalty

        # Gymnasium termination flags.
        terminated = success or collision
        truncated = self.step_count >= self.max_episode_steps and not terminated

        if truncated:
            reward -= self.timeout_distance_penalty * distance_to_goal

        # Store distance for debugging / future extensions.
        self.prev_distance_to_goal = distance_to_goal

        # Return Gymnasium five-tuple.
        obs = self._get_obs()

        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Observation contains non-finite values: {obs}")

        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


    ############################################################################
    # Private helper methods for encoding state and diagnostics.
    ############################################################################

    # Return Euclidean distance from the agent to the goal.
    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    # Return collision margins for active obstacles.
    # Positive margin means clear of collision.
    # Zero or negative margin means collision.
    def _obstacle_margins(self) -> np.ndarray:
        delta = self.obstacle_centers - self.position[None, :]
        center_distances = np.linalg.norm(delta, axis=1)
        collision_radii = self.obstacle_radii + self.agent_radius
        margins = center_distances - collision_radii

        # Inactive obstacle slots should not affect min-distance or collision.
        margins = np.where(self.obstacle_mask, margins, np.inf)
        return margins

    # Encode the current internal state as a fixed-size observation
    def _get_obs(self) -> np.ndarray:
        goal_delta = self.goal - self.position
        margins = self._obstacle_margins()

        features: list[float] = [
            float(self.position[0]),
            float(self.position[1]),
            float(np.cos(self.theta)),
            float(np.sin(self.theta)),
            float(goal_delta[0]),
            float(goal_delta[1]),
        ]

        for i in range(self.max_obstacles):
            active = bool(self.obstacle_mask[i])

            if active:
                relpos = self.obstacle_centers[i] - self.position
                margin = float(margins[i])
                radius = float(self.obstacle_radii[i])
                mask_value = 1.0
            else:
                relpos = np.zeros(2, dtype=np.float64)
                margin = 0.0
                radius = 0.0
                mask_value = 0.0

            features.extend(
                [
                    float(relpos[0]),
                    float(relpos[1]),
                    margin,
                    radius,
                    mask_value,
                ]
            )

        obs = np.asarray(features, dtype=np.float32)

        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observation shape {obs.shape} does not match "
                f"{self.observation_space.shape}."
            )

        return obs

    # Return diagnostic information for logging and smoke checks.
    def _get_info(self) -> dict:
        distance_to_goal = self._distance_to_goal()
        margins = self._obstacle_margins()

        if np.any(self.obstacle_mask):
            min_obstacle_clearance = float(np.min(margins))
            collision = bool(min_obstacle_clearance <= 0.0)
        else:
            # Clearance is undefined without active obstacles. NaN prevents a
            # placeholder value from entering evaluation aggregates.
            min_obstacle_clearance = float("nan")
            collision = False

        success = bool(distance_to_goal <= self.goal_radius)

        return {
            "success": success,
            "collision": collision,
            "distance_to_goal": float(distance_to_goal),
            "min_obstacle_clearance": min_obstacle_clearance,
            "step_count": int(self.step_count),
        }

    # Wrap angle to [-pi, pi).
    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return float((theta + np.pi) % (2.0 * np.pi) - np.pi)