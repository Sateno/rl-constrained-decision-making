
from __future__ import annotations
from typing import Callable
import gymnasium as gym
from environments.constrained_navigation import ConstrainedNavigationEnv
from environments.action_wrappers import NormalizedActionWrapper
from projection.cbf_qp_projection import ProjectionParams
from projection.cbf_qp_wrapper import CbfQpProjectionWrapper


#########################################################################
# Return a factory function that creates one fresh environment.
#
# This function does not create the environment immediately.
# It returns a zero-argument callable so Gymnasium vector environments
# can create independent environment instances.
#########################################################################
def make_env(
    *, env_index: int, 
    env_kwargs: dict | None = None, 
    record_episode_statistics: bool = True, 
    normalize_actions: bool = True,
    enable_projection: bool = False,
    projection_params: ProjectionParams | None = None
) -> Callable[[], gym.Env]:
#{    
    # Keep for future seeting/debugging
    _ = env_index

    # Make a copy to avoid mutating the original
    local_env_kwargs = {} if env_kwargs is None else dict(env_kwargs)  

    def create_env() -> gym.Env:
    #{
        env = ConstrainedNavigationEnv(**local_env_kwargs)
        
        if enable_projection:
            env = CbfQpProjectionWrapper(env, params=projection_params)

        if normalize_actions:
            env = NormalizedActionWrapper(env)

        if record_episode_statistics:
            env = gym.wrappers.RecordEpisodeStatistics(env)

        # Optional later:
        # env.action_space.seed(base_seed + env_index)
        
        # We do not call env.reset(...) here. The caller controls episode seeds.

        return env
    #}

    return create_env
#} End function make_env