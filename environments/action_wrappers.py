from __future__ import annotations

import gymnasium as gym
import numpy as np

# Exposes a symmetric normalized action space [-1, 1]^d to the policy.
# Maps normalized actions to the wrapped environment's physical action space.
class NormalizedActionWrapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("NormalizedActionWrapper requires a Box action space.")

        self.physical_low = env.action_space.low.astype(np.float32)
        self.physical_high = env.action_space.high.astype(np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)


    def action(self, action: np.ndarray) -> np.ndarray:
        normalized_action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)

        normalized_action = np.clip(normalized_action, self.action_space.low, self.action_space.high)

        physical_action = (
            self.physical_low
            + 0.5
            * (normalized_action + 1.0)
            * (self.physical_high - self.physical_low)
        )

        return physical_action.astype(np.float32, copy=False)


    def reverse_action(self, physical_action: np.ndarray) -> np.ndarray:
        physical_action = np.asarray(physical_action, dtype=np.float32).reshape(self.action_space.shape)

        normalized_action = (
            2.0
            * (physical_action - self.physical_low)
            / (self.physical_high - self.physical_low)
            - 1.0
        )

        return np.clip(normalized_action, self.action_space.low, self.action_space.high).astype(np.float32, copy=False)
