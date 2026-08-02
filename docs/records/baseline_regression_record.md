# Historical PPO Baseline Regression Record

## Status and identity

```text
role=regression and calibration fixture
final_experiment_replicate=false
checkpoint=runs/checkpoints/ppo_baseline_51200_seed1.pt
sha256=3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3
observation shape=(21,)
action shape=(2,)
training seed=1
global step=51200
```

The checkpoint predates the explicit Gymnasium same-step autoreset correction. It remains valid for regression and calibration, but final comparative checkpoints must be trained from the corrected codebase.

## Training configuration

```text
total timesteps=51200
num envs=4
rollout steps=256
batch size=1024
minibatches=8
update epochs=4
learning rate=0.0003
gamma=0.99
gae lambda=0.95
clip coefficient=0.2
```

## Deterministic regression result

Configuration:

```text
20 episodes, seeds 1000-1019
built-in three-obstacle layout
actor mean, projection disabled
maximum 200 steps
```

| Metric | Value |
|---|---:|
| Mean return | `12.319942` |
| Mean length | `96.0` |
| Success rate | `1.0` |
| Collision rate | `0.0` |
| Final distance | approximately `0.235396` |
| Mean minimum clearance | approximately `0.518163` |

## Adjustment history

Initial training used the asymmetric physical action space directly. A Gaussian mean near zero proposed non-positive speed, which the environment clipped to zero. PPO learned a conservative idle policy.

Two reward terms reduced the value of idling:

```text
time_penalty=0.01
timeout_distance_penalty=1.0
```

The decisive correction was `NormalizedActionWrapper`:

```text
v = 0.5 * v_max * (a_v + 1)
omega = omega_max * a_omega
```

A normalized actor mean near zero then mapped to forward motion rather than the lower speed bound.

## Current use

The checkpoint is retained for:

```text
deterministic baseline regression
projection noninterference regression
stochastic active-projection diagnostics
checkpoint compatibility checks
core layout calibration
```

It must not be pooled with fresh post-correction checkpoints in final seed-level statistics.

Generated checkpoint, evaluation CSVs, and TensorBoard events remain under `runs/`.

## Claim boundary

This record establishes learnability of the original fixed benchmark after reward and action-interface corrections. It does not establish broad layout generalization. Core calibration later showed strong specialization to the built-in training geometry.
