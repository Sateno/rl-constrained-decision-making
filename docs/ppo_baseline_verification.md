# PPO Baseline and Evaluation Foundation Verification

## Scope

Components:

- `environments.factory.make_env`
- `evaluation.evaluate_policy`
- `scripts/evaluate_policy.bat`
- `algorithms.ppo.ppo_continuous_action`

Environment: `RL_PROJECTS`  
Repository root: `C:\rl_projects\src\repos\rl-constrained-decision-making`

This document records the minimum verification commands for the PPO baseline and evaluation foundation. It excludes repository setup, directory creation, file creation, Git operations, and one-off debugging commands.

## Setup

```cmd
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
conda activate RL_PROJECTS
```

## 1. Compile Key Python Modules

```cmd
python -m py_compile environments\factory.py evaluation\evaluate_policy.py algorithms\ppo\ppo_continuous_action.py
```

Expected result:

```text
no output
```

## 2. Run Current Tests

```cmd
python -m pytest -q
```

Expected result:

```text
all current tests pass
```

## 3. Run Evaluation Wrapper

```cmd
scripts\evaluate_policy.bat
```

Expected properties:

```text
Evaluation summary is printed.
CSV output is written under runs\evaluation\.
No traceback or command-line parsing error occurs.
```

## 4. Run PPO Integration Smoke Test

```cmd
python -m algorithms.ppo.ppo_continuous_action --env-id ConstrainedNavigation-v0 --total-timesteps 256 --num-envs 2 --num-steps 32 --num-minibatches 2 --update-epochs 1 --max-episode-steps 200
```

Expected properties:

```text
The PPO script starts without import errors.
Vector environments construct successfully.
The agent initializes from single_observation_space and single_action_space.
Rollout collection and PPO update execute without shape or dtype errors.
Four SPS lines are printed for the 256-timestep smoke run.
Projection remains disabled.
```

## Verification Exit Criteria

The PPO baseline and evaluation foundation is considered verified when:

- the key modules compile;
- the current test suite passes;
- the evaluation wrapper runs and writes a CSV under `runs\evaluation\`;
- the adapted PPO script completes the small smoke run without import, wrapper, observation-shape, action-shape, or dtype errors.
