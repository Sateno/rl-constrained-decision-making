# PPO Baseline Verification Record

## Scope

Verified components:

- `environments.constrained_navigation.ConstrainedNavigationEnv`
- `environments.action_wrappers.NormalizedActionWrapper`
- `environments.factory.make_env`
- `evaluation.evaluate_policy`
- `algorithms.ppo.agent.Agent`
- `algorithms.ppo.ppo_continuous_action`
- `scripts.run_ppo_baseline_clean.bat`

Environment: `RL_PROJECTS`  
Repository root: `C:\rl_projects\src\repos\rl-constrained-decision-making`

This record contains the canonical verification command set for the PPO baseline and evaluation foundation. It excludes repository setup commands, Git operations, directory creation commands, file creation commands, and one-off debugging commands.

## Verification Command Set

### 1. Activate Environment

```cmd
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
conda activate RL_PROJECTS
```

### 2. Execute Full PPO Baseline Verification Pipeline

```cmd
scripts\run_ppo_baseline_clean.bat
```

The batch script performs the complete baseline verification sequence:

```text
1. Removes the previous generated runs directory.
2. Recreates local run-output folders.
3. Compiles key Python modules.
4. Runs the current pytest suite.
5. Runs random-policy evaluation.
6. Trains the PPO baseline.
7. Verifies checkpoint creation.
8. Evaluates deterministic PPO.
9. Evaluates stochastic PPO.
10. Prints the summary comparison table.
```

Expected generated files:

```text
runs\checkpoints\ppo_baseline_51200_seed1.pt
runs\evaluation\random_policy_seed1000.csv
runs\evaluation\ppo_baseline_51200_seed1_eval.csv
runs\evaluation\ppo_baseline_51200_seed1_eval_stochastic.csv
```

Expected terminal completion message:

```text
Clean PPO baseline run completed successfully.
```

### 3. Inspect TensorBoard Logs

```cmd
tensorboard --logdir runs
```

Expected TensorBoard groups:

```text
charts/SPS
charts/learning_rate
charts/episodic_return
charts/episodic_length
losses/value_loss
losses/policy_loss
losses/entropy
losses/old_approx_kl
losses/approx_kl
losses/clipfrac
losses/explained_variance
```

TensorBoard is used for training-dynamics inspection. The evaluation CSV files are used for policy-performance summaries.

## Current Verified Baseline Result

Verification command:

```cmd
scripts\run_ppo_baseline_clean.bat
```

Evaluation setting:

```text
training seed: 1
evaluation seed base: 1000
evaluation episodes: 20
max episode steps: 200
PPO total timesteps: 51200
PPO num envs: 4
PPO rollout steps: 256
PPO minibatches: 8
PPO update epochs: 4
```

Summary output:

```text
random
episode_return            -8.851189
episode_length            65.450000
success                    0.050000
collision                  0.800000
final_distance_to_goal     2.672536
min_obstacle_distance      0.035840

ppo deterministic
episode_return            12.319942
episode_length            96.000000
success                    1.000000
collision                  0.000000
final_distance_to_goal     0.235396
min_obstacle_distance      0.518163

ppo stochastic
episode_return            -1.187591
episode_length            137.650000
success                    0.400000
collision                  0.250000
final_distance_to_goal     1.179077
min_obstacle_distance      0.266364
```

Primary baseline mode:

```text
deterministic PPO evaluation
```

Diagnostic mode:

```text
stochastic PPO evaluation
```

## Baseline Adjustment Record

### Initial Failure Mode

Initial PPO training with the physical action space

```text
v in [0, 1]
omega in [-2, 2]
```

converged to a conservative idle policy. The deterministic actor mean produced a non-positive forward-speed command, which was clipped to zero by the environment. The policy avoided collisions by not translating and timed out with zero success.

### Reward Adjustment

A per-step time penalty and a terminal timeout-distance penalty were added to reduce the value of idling:

```text
time_penalty = 0.01
timeout_distance_penalty = 1.0
```

The adjustment reduced the value of idling but did not fully resolve the conservative-action failure mode.

### Action-Interface Adjustment

A normalized action wrapper was added. PPO now observes a symmetric action space:

```text
a_v in [-1, 1]
a_omega in [-1, 1]
```

The wrapper maps normalized policy actions to physical controls:

```text
v = 0.5 * v_max * (a_v + 1)
omega = omega_max * a_omega
```

This maps an actor mean near zero to forward motion rather than to the lower forward-speed bound. The adjustment produced a successful deterministic PPO baseline on the current fixed/default evaluation setting.

## Generated Asset Policy

Generated files under `runs\` are local experiment outputs and are excluded from Git.

Current retained local artifacts:

```text
runs\checkpoints\ppo_baseline_51200_seed1.pt
runs\evaluation\random_policy_seed1000.csv
runs\evaluation\ppo_baseline_51200_seed1_eval.csv
runs\evaluation\ppo_baseline_51200_seed1_eval_stochastic.csv
```

Current generated artifacts are not promoted to `results\`. The `results\` directory is reserved for curated tables and figures from final comparison experiments.

## Verification Exit Criteria

The PPO baseline and evaluation foundation is verified when all conditions hold:

```text
The full clean baseline batch completes without error.
A PPO checkpoint is created under runs\checkpoints\.
Random-policy, deterministic-PPO, and stochastic-PPO evaluation CSV files are created under runs\evaluation\.
TensorBoard contains episode-return and episode-length charts.
Deterministic PPO achieves task-solving behavior on the current fixed/default evaluation setting.
Generated run artifacts remain outside Git.
```
