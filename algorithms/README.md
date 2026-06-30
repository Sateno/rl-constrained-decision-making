# Algorithms

This package contains project-owned algorithm implementations and adapted algorithm scripts.

## PPO

`algorithms/ppo/ppo_continuous_action.py` contains the adapted continuous-action PPO training script.

`algorithms/ppo/agent.py` contains the shared PPO actor--critic model used by training and evaluation.

The PPO implementation is adapted from CleanRL's continuous-action PPO script. CleanRL provenance should be recorded in the repository documentation rather than by keeping an unused base copy in the source tree.