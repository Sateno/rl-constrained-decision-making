# Evaluation-Time Projection Regression Record

## Status

```text
validated_source_commit=20eacdee87ddd22b91af7de73c2f81fb3a04618f
validation_date=2026-07-30
status=PASS
checkpoint=runs/checkpoints/ppo_baseline_51200_seed1.pt
checkpoint_sha256=3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3
trajectory_schema=evaluation_trajectory_v1
lightweight_tests=46 passed
```

Operational command:

```bat
scripts\validate_evaluation_time_projection.bat
```

## Coverage

The validator passed source compilation, OSQP availability, checkpoint identity, active-count compatibility, expected capacity rejection, deterministic and stochastic paired evaluation, CSV/NPZ action audits, and a 300-call runtime benchmark.

The 21-dimensional checkpoint remained compatible with three obstacle slots and two active obstacles. It was correctly rejected for five slots (`obs_dim=31`).

## Deterministic paired result

Configuration:

```text
20 episodes, seeds 1000-1019
built-in layout, actor mean, CPU inference
lookahead=0.25, alpha=2.0, slack penalty=1000.0,
extra clearance=0.0
```

| Metric | Projection off | Projection on |
|---|---:|---:|
| Mean return | `12.319942` | `12.319942` |
| Mean length | `96.0` | `96.0` |
| Success rate | `1.0` | `1.0` |
| Collision rate | `0.0` | `0.0` |
| Mean minimum clearance | `0.518163` | `0.518163` |
| Interventions | - | `0` |
| Solver failures | - | `0` |

Interpretation: the deterministic policy was already locally feasible, so projection acted as the identity map. Zero intervention is valid noninterference.

## Stochastic active diagnostic

| Metric | Projection off | Projection on | Delta |
|---|---:|---:|---:|
| Mean return | `-0.611634` | `2.766761` | `3.378395` |
| Success rate | `0.45` | `0.60` | `0.15` |
| Collision rate | `0.30` | `0.00` | `-0.30` |
| Mean minimum clearance | `0.155729` | `0.229752` | `0.074023` |

Projection-on burden:

```text
interventions=201
mean intervention rate=0.063031
mean correction norm=0.026436
maximum correction norm=1.736898
mean summed slack=0.000071
maximum slack=0.016806
solver failures=0
```

This verifies active correction and diagnostic propagation. It is not a final multi-seed method comparison.

## Trajectory and runtime evidence

Twenty deterministic trajectory pairs passed schema, alignment, action-identity, slack, checkpoint, and seed checks.

Projection runtime:

```text
300 calls
4.017653 ms per call
248.902 calls per second
last status=optimal
```

## Evidence locations

```text
runs/validation/evaluation_time_projection_validation.log
runs/validation/evaluation_time_projection_validation_summary.txt
runs/validation/projection_environment.txt
runs/validation/projection_conda_list.txt
runs/validation/projection_runtime.txt
runs/evaluation/ppo_baseline_51200_seed1_projection_pair_*
runs/evaluation/ppo_baseline_51200_seed1_projection_pair_stochastic_*
```

## Limitations

- The deterministic regression uses one fixed built-in layout.
- The checkpoint is a historical fixture, not a final experimental seed.
- The stochastic result is an active-path diagnostic.
- The local softened CBF-QP is not a global route planner or global safety guarantee.
- Runtime applies to the recorded software and hardware configuration.
