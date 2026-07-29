from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from algorithms.ppo.ppo_continuous_action import make_vector_env
from algorithms.ppo.projection_training import ProjectionTrainingDiagnostics


# One-step environment used to distinguish same-step from next-step autoreset.
class OneStepActionEnv(gym.Env):
#{
    metadata = {"render_modes": []}

    def __init__(self) -> None:
    #{
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.reset_count = 0
        self.executed_actions: list[float] = []
    #} End function __init__

    def reset(self, *, seed: int | None = None, options: dict | None = None):
    #{
        super().reset(seed=seed)
        _ = options
        self.reset_count += 1
        return np.asarray([self.reset_count], dtype=np.float32), {}
    #} End function reset

    def step(self, action):
    #{
        value = float(np.asarray(action, dtype=np.float32).reshape(1)[0])
        self.executed_actions.append(value)
        return (
            np.asarray([99.0], dtype=np.float32),
            value,
            True,
            False,
            {"executed_action": value},
        )
    #} End function step

#} End class OneStepActionEnv


# The trainer's vector environment must execute the first action after every reset.
def test_vector_env_uses_same_step_autoreset() -> None:
#{
    base_env = OneStepActionEnv()
    envs = make_vector_env([lambda: base_env])

    try:
        envs.reset(seed=7)
        observations, rewards, terminations, truncations, infos = envs.step(
            np.asarray([[0.25]], dtype=np.float32)
        )

        assert envs.autoreset_mode == gym.vector.AutoresetMode.SAME_STEP
        assert rewards.tolist() == pytest.approx([0.25])
        assert terminations.tolist() == [True]
        assert truncations.tolist() == [False]
        assert observations.tolist() == [[2.0]]
        assert infos["final_info"]["executed_action"][0] == pytest.approx(0.25)

        _, rewards, _, _, _ = envs.step(np.asarray([[0.75]], dtype=np.float32))
        assert rewards.tolist() == pytest.approx([0.75])
        assert base_env.executed_actions == pytest.approx([0.25, 0.75])
    finally:
        envs.close()
#} End function test_vector_env_uses_same_step_autoreset


# Same-step autoreset preserves completed episode statistics in final_info.
def test_same_step_autoreset_preserves_episode_statistics() -> None:
#{
    envs = make_vector_env(
        [lambda: gym.wrappers.RecordEpisodeStatistics(OneStepActionEnv())]
    )

    try:
        envs.reset(seed=7)
        _, rewards, terminations, truncations, infos = envs.step(
            np.asarray([[0.25]], dtype=np.float32)
        )

        assert rewards.tolist() == pytest.approx([0.25])
        assert terminations.tolist() == [True]
        assert truncations.tolist() == [False]
        assert infos["final_info"]["_episode"].tolist() == [True]
        assert float(infos["final_info"]["episode"]["r"][0]) == pytest.approx(0.25)
        assert int(infos["final_info"]["episode"]["l"][0]) == 1
        assert infos["final_info"]["executed_action"][0] == pytest.approx(0.25)
    finally:
        envs.close()
#} End function test_same_step_autoreset_preserves_episode_statistics


# Rollout diagnostics include both nonterminal vector fields and terminal final_info.
def test_projection_training_diagnostics_include_terminal_final_info() -> None:
#{
    diagnostics = ProjectionTrainingDiagnostics()
    infos = {
        "projection_success": np.asarray([True, False]),
        "_projection_success": np.asarray([True, False]),
        "projection_solver_status": np.asarray(["optimal", ""], dtype=object),
        "projection_intervened": np.asarray([False, False]),
        "projection_correction_norm": np.asarray([0.0, 0.0]),
        "projection_slack_sum": np.asarray([0.0, 0.0]),
        "projection_slack_max": np.asarray([0.0, 0.0]),
        "final_info": {
            "projection_success": np.asarray([False, True]),
            "_projection_success": np.asarray([False, True]),
            "projection_solver_status": np.asarray(["", "optimal"], dtype=object),
            "projection_intervened": np.asarray([False, True]),
            "projection_correction_norm": np.asarray([0.0, 0.3]),
            "projection_slack_sum": np.asarray([0.0, 0.02]),
            "projection_slack_max": np.asarray([0.0, 0.02]),
        },
        "_final_info": np.asarray([False, True]),
    }

    diagnostics.update(infos, num_envs=2)

    assert diagnostics.transition_count == 2
    assert diagnostics.intervention_count == 1
    assert diagnostics.correction_sum == pytest.approx(0.3)
    assert diagnostics.slack_sum == pytest.approx(0.02)
    assert diagnostics.slack_max == pytest.approx(0.02)
#} End function test_projection_training_diagnostics_include_terminal_final_info


# A solver failure stored in terminal final_info must still stop training.
def test_projection_training_diagnostics_reject_terminal_solver_failure() -> None:
#{
    diagnostics = ProjectionTrainingDiagnostics()
    infos = {
        "final_info": {
            "projection_success": np.asarray([False]),
            "_projection_success": np.asarray([True]),
            "projection_solver_status": np.asarray(["solver_error"], dtype=object),
        },
        "_final_info": np.asarray([True]),
    }

    with pytest.raises(RuntimeError, match="env 0: solver_error"):
        diagnostics.update(infos, num_envs=1)
#} End function test_projection_training_diagnostics_reject_terminal_solver_failure
