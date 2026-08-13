# Predictive Action Projection Final Experimental Protocol

## Freeze status

```text
protocol_id=predictive_action_projection_final_v1
freeze_date=2026-08-12
decision=FROZEN BEFORE FINAL TRAINING
validated_implementation_base_commit=d0548b3e6729571113675b6f4ccad6401f27f167
campaign_source=git tag predictive-action-projection-protocol-v1
final_training_results_inspected=false
final_evaluation_results_inspected=false
runtime_code_changed_by_this_freeze=false
```

This record freezes the final experimental choices before any fresh final
checkpoint is trained. The commit carrying the annotated tag
`predictive-action-projection-protocol-v1` is the authorized campaign source.
The protocol commit may contain only documentation, protocol declarations, and
the exact serialization of the already implemented fixed training geometry.

## Machine-readable analysis declarations

```text
primary analysis protocol:
experiments/fixed_training_geometry_analysis_protocol.json
canonical_sha256=89ea5ec0d2329bfe99ea3a1cb7a5c1339478b0ce2d8304f5b1b810f51f1f0d14

secondary analysis protocol:
experiments/projection_analysis_protocol.json
canonical_sha256=dfc0e1e3de29c0f63eb6152a3f063ad1d17c4461430855f33222f93e827c6e90

fixed training-geometry serialization:
evaluation/layouts/fixed_training_geometry.json
suite_id=fixed_training_geometry_v1
canonical_sha256=c977b0b340e62770327cec65ebed9920cd7ac8d51038c5b6a5bfd8c47ce8625c

frozen transfer suite:
evaluation/layouts/core_navigation_layouts.json
suite_id=core_navigation_layouts_v1
canonical_sha256=1027141797052240e83b941398e5a32031e9ca67d001e4a0cf1e19b0f96dd466
```

The fixed-training-geometry JSON is an exact serialization of the existing
built-in three-obstacle task. It does not introduce a new training environment
or change the environment implementation.

## Device decision

```text
training device=cpu
evaluation device=cpu
benchmark evidence=runs/validation/training_device_benchmark/
benchmark seed=9902
benchmark transitions per condition=10240
```

CUDA completed successfully but was slower under the predeclared weighted
comparison. The device decision is engineering-only and was not based on
return, success, collision, or any other learning outcome.

## Final training matrix

| Method | Training collision penalty | Training projection | Seeds |
|---|---:|---:|---|
| `ppo_baseline` | `10.0` | disabled | `1, 2, 3, 4, 5` |
| `ppo_high_penalty` | `50.0` | disabled | `1, 2, 3, 4, 5` |
| `ppo_train_projection` | `10.0` | enabled | `1, 2, 3, 4, 5` |

```text
independent training runs=15
transitions per run=51200
checkpoint selection=final checkpoint only
training geometry=fixed built-in three-obstacle geometry
```

A training seed initializes one independent PPO run. It is unrelated to the
number of obstacles.

## Shared training configuration

```text
num_envs=4
num_steps=256
batch_size=1024
num_minibatches=8
update_epochs=4
learning_rate=3e-4
anneal_learning_rate=true
gamma=0.99
gae_lambda=0.95
normalize_advantages=true
clip_coefficient=0.2
clip_value_loss=true
entropy_coefficient=0.0
value_coefficient=0.5
maximum_gradient_norm=0.5
target_kl=None
torch_deterministic=true
maximum_episode_steps=200
maximum_obstacle_capacity=3
active_training_obstacles=3
external_tracking=false
video_capture=false
```

The PPO update loop, raw sampled-action likelihood semantics, explicit same-step
autoreset, environment dynamics, and reward terms other than the declared
collision penalty remain unchanged across methods.

Projection parameters for training and evaluation are:

```text
lookahead_distance=0.25
alpha=2.0
slack_penalty=1000.0
extra_clearance=0.0
```

## Required training evidence

Every final run must preserve:

```text
episode return and length
success, collision, and timeout
cumulative success, collision, and timeout counts
rolling success, collision, and timeout rates
action-bound clipping frequency
per-component clipping frequency
mean and maximum clipping-excess norm
```

The rolling window is frozen to:

```text
20 completed episodes
```

Projection-enabled training must additionally preserve:

```text
projection transition count
intervention count and frequency
mean and maximum correction norm
mean summed slack
maximum slack
solver-failure count
```

Training returns are optimization returns under each method's own reward.
Cross-method task comparison uses the common evaluation reward below.

## Complete deployment evaluation matrix

Every final checkpoint is evaluated with projection both disabled and enabled.

| Trained policy | Projection disabled | Projection enabled |
|---|---:|---:|
| Baseline PPO | required | required |
| High-penalty PPO | required | required |
| Projection-trained PPO | required | required |

The common evaluation collision penalty is `10.0` in every cell.

## Primary evaluation

```text
role=stochastic policy behavior on the fixed training geometry
layout suite=evaluation/layouts/fixed_training_geometry.json
layout ID=fixed_training_geometry
policy mode=stochastic
episodes per checkpoint and projection mode=100
evaluation seeds=10000 through 10099
maximum episode steps=200
device=cpu
learning during evaluation=none
```

Primary outcomes:

```text
primary safety outcome=collision rate
principal task outcome=success rate
competing outcome=timeout rate
```

For complete episodes:

```text
timeout = truncated and not terminated
```

Supporting outcomes include common-reward return, episode length, minimum
signed clearance, action-bound clipping, projection intervention, mean and
maximum correction, mean and maximum slack, and solver failures.

Projection-off and projection-on runs use the same episode seeds. They are
matched-seed comparisons, not trajectory-level counterfactuals after the first
changed executed action.

## Secondary evaluation

```text
role=deterministic obstacle-layout transfer and robustness
suite=evaluation/layouts/core_navigation_layouts.json
suite ID=core_navigation_layouts_v1
canonical SHA-256=1027141797052240e83b941398e5a32031e9ca67d001e4a0cf1e19b0f96dd466
policy mode=deterministic actor mean
layouts=24
episodes per layout/checkpoint/projection mode=1
evaluation seeds=1000 through 1023
maximum episode steps=200
device=cpu
learning during evaluation=none
```

Primary fixed-geometry results and secondary transfer results are reported
separately and are never pooled into one headline metric.

## Predeclared comparisons

1. Baseline off versus baseline on: execution-time projection effect.
2. Baseline off versus high-penalty off: reward-shaping effect.
3. Baseline off versus projection-trained off: learned nominal-policy effect.
4. Projection-trained off versus projection-trained on: filter dependence.
5. Baseline on versus projection-trained on: operational comparison with the filter.
6. High-penalty off versus high-penalty on: reward/filter complementarity.

Filter dependence is described through separate enabled-minus-disabled changes
in collision rate, success rate, timeout rate, and common-reward return. No
post hoc composite score is created.

Matched episode or layout outcomes are classified as:

```text
collision, success, or timeout
```

and paired transition counts are reported descriptively. These transition
tables are deterministic transformations of the frozen raw episode rows; they
do not require changes to training, environment, projection, or evaluation
semantics.

## Aggregation and statistical interpretation

The independently trained checkpoint is the empirical replicate.

Primary evaluation:

```text
100 complete episodes within checkpoint
-> one checkpoint summary
-> aggregate five checkpoint summaries across training seeds
```

Secondary evaluation:

```text
24 layouts within checkpoint
-> one checkpoint summary
-> aggregate five checkpoint summaries across training seeds
```

Report all seed-level values, mean, standard deviation, paired seed-level
projection effects, and effect magnitudes. Episodes and layouts are not treated
as independently trained policies. No formal significance test is predeclared.

## Rerun and exclusion rules

A final run may be repeated only for a documented technical failure:

```text
process interruption
missing or corrupted checkpoint
wrong method, seed, budget, or configuration
NaN or Inf in model state or required metrics
device or memory failure
artifact-schema failure
projection solver failure
failure to reach exactly 51200 transitions
```

These are not valid rerun reasons:

```text
low return
high collision rate
failure to learn
unfavorable method comparison
large but valid seed-to-seed variation
```

Every failed attempt is preserved separately with an incident record. A
replacement uses the same seed and frozen configuration.

## Historical exclusion

```text
checkpoint=runs/checkpoints/ppo_baseline_51200_seed1.pt
sha256=3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3
role=historical regression and calibration fixture only
included in final matrix=false
```

No historical smoke, calibration, or pre-autoreset checkpoint may substitute
for a final seed.

## Claim boundaries

Potential claims, only when supported by final evidence:

```text
projection reduced measured collisions in the tested setting
projection changed the measured safety-performance tradeoff
training-time projection changed intervention burden
higher collision penalty and projection produced different behavior
projection diagnostics revealed dependence on the filter
```

Prohibited claims:

```text
global safety
formal optimality
unbiased projected PPO gradients
real-world robotics readiness
general superiority over safe RL or MPC
broad navigation generalization
```

## Change control

After final training begins, a methodological or runtime change requires a new
protocol version, written validity rationale, new source identity, new
validation PASS, and complete reruns of every affected method, seed,
evaluation level, and projection mode. Outcome-driven changes are prohibited.

## Freeze declaration

```text
protocol=predictive_action_projection_final_v1
training methods=3
training seeds=1,2,3,4,5
final checkpoints=15
training budget=51200 transitions per run
training device=cpu
evaluation device=cpu
primary evaluation=100 stochastic episodes on fixed training geometry
secondary evaluation=24 deterministic transfer layouts
all trained methods evaluated projection off and on
final outcomes inspected before freeze=false
```
