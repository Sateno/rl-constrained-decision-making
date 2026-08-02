# Core Navigation Layout Calibration and Freeze Record

## Status

```text
calibration_source_commit=20eacdee87ddd22b91af7de73c2f81fb3a04618f
calibration_date=2026-07-30
decision=ACCEPTED UNCHANGED
suite_role=deterministic transfer and robustness evaluation
final_training_results_inspected=false
```

The 24-layout core suite was calibrated before final training and before final method results were observed. It is frozen as a secondary deterministic obstacle-layout transfer and robustness benchmark, not as the primary in-distribution task benchmark.

## Frozen suite

```text
path=evaluation/layouts/core_navigation_layouts.json
suite_id=core_navigation_layouts_v1
canonical_sha256=1027141797052240e83b941398e5a32031e9ca67d001e4a0cf1e19b0f96dd466
layout_count=24
```

| Family | Count |
|---|---:|
| Control and noninterference | 4 |
| Single-obstacle deviations | 6 |
| Two-obstacle interactions | 8 |
| Three-obstacle slaloms | 6 |

Held constant:

```text
start=[0, 0]
heading=0
goal=[4, 0]
agent radius=0.10
goal radius=0.25
maximum obstacle slots=3
episode limit=200
```

Only obstacle geometry and active count vary. Non-symmetric layouts have reflected counterparts.

Visual references:

- [Original training layout](../assets/layouts/original_training_layout.pdf)
- [Core layout suite](../assets/layouts/core_navigation_layouts_visual_reference.pdf)

## Calibration probe and protocol

```text
checkpoint=runs/checkpoints/ppo_baseline_51200_seed1.pt
checkpoint_sha256=3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3
role=historical regression checkpoint used only for calibration
policy mode=deterministic actor mean
projection modes=disabled and enabled
one episode per layout
base seed=1000
maximum steps=200
collision penalty=10.0
lookahead=0.25
alpha=2.0
slack penalty=1000.0
extra clearance=0.0
```

Calibration was allowed to reject malformed, uniformly trivial, uniformly impossible, non-intervening, constantly intervened, or numerically unstable suites. It did not require projection to improve return, success, or collision count.

## Automated result

```text
status=PASS
projection_interventions=166
layouts_with_intervention=10
layouts_without_intervention=14
projection_solver_failures=0
maximum_correction_norm=0.601044
maximum_slack=0.002209
```

| Mode | Successes | Collisions | Timeouts |
|---|---:|---:|---:|
| Projection off | 7 | 3 | 14 |
| Projection on | 8 | 1 | 15 |

These values describe one historical probe policy and are not final comparative results.

## Family summary

### Projection off

| Family | Mean return | Success | Collision | Timeout | Mean clearance |
|---|---:|---:|---:|---:|---:|
| Control | `-8.454` | `0.000` | `0.250` | `0.750` | `0.337` |
| Single | `-7.834` | `0.000` | `0.167` | `0.833` | `0.345` |
| Double | `-4.957` | `0.125` | `0.125` | `0.750` | `0.344` |
| Triple | `12.279` | `1.000` | `0.000` | `0.000` | `0.244` |

### Projection on

| Family | Mean return | Success | Collision | Timeout | Interventions |
|---|---:|---:|---:|---:|---:|
| Control | `-8.539` | `0.000` | `0.250` | `0.750` | `75` |
| Single | `-6.898` | `0.000` | `0.000` | `1.000` | `45` |
| Double | `-2.241` | `0.250` | `0.000` | `0.750` | `25` |
| Triple | `12.272` | `1.000` | `0.000` | `0.000` | `21` |

## Scientific judgment

```text
geometry=PASS
behavioral diversity=PASS
projection behavior=PASS
solver stability=PASS
revision required=NO
```

No malformed, overlapping, or numerically unstable layout was found. The suite produced successes, collisions, timeouts, active interventions, and nonintervention cases.

The historical policy failed the obstacle-free route and all four controls while succeeding on all six three-obstacle slaloms. Representative trajectories showed a repeated downward arc that passed below the goal on open and sparse layouts. This indicates specialization to the fixed training geometry, not defective control layouts.

Changing the suite to accommodate the probe policy would have adapted the benchmark to the policy. The suite was therefore accepted unchanged.

## Notable cases

### Noninterference

Fourteen layouts produced zero interventions. Enabling projection did not imply continuous action modification.

### Collision prevention without success

```text
layout=single_near_early_lower
off=collision
on=timeout
interventions=37
```

A local filter prevented immediate collision but did not provide the global route behavior required for success.

### Collision prevention with success

```text
layout=double_near_staggered_lower_first
off=collision
on=success
interventions=20
return=-9.308 to 12.368
```

Local corrections were sufficient for the existing policy to recover and reach the goal.

### Persistent unsafe behavior

```text
layout=control_lower_clearance
off=collision at 31 steps
on=collision at 85 steps
interventions=62
minimum clearance=-0.034553 to -0.001855
```

Projection delayed and reduced penetration but did not create a successful route. The local softened filter is not a global planner.

## Final study role

The suite measures:

```text
transfer to changed obstacle geometry
directional bias through mirrored layouts
policy specialization
projection compensation for out-of-distribution actions
filter dependence of projection-trained policies
```

The final protocol should separately define a primary in-distribution stochastic safety evaluation on the original training layout.

## Evidence preservation

Complete raw evidence:

```text
runs/calibration/core_navigation_layouts_v1/
```

Recommended curated repository outputs:

```text
results/calibration/core_navigation_layouts_v1/calibration_by_family.csv
results/calibration/core_navigation_layouts_v1/calibration_layout_comparison.csv
results/calibration/core_navigation_layouts_v1/calibration_representative_trajectories.pdf
```

The raw evidence directory and historical checkpoint must be archived outside normal Git and must not be overwritten.

## Paper notes

Method statement:

> A fixed suite of 24 obstacle configurations was defined and calibrated before final training. Start, heading, goal, dynamics, reward, agent footprint, action bounds, and episode limit were held constant; only obstacle geometry varied. A historical checkpoint was used solely to reject malformed or degenerate scenarios. No final-policy result influenced layout selection.

Interpretation statement:

> Calibration revealed strong specialization of the historical policy to the fixed three-obstacle training geometry. The core suite was therefore interpreted as a deterministic obstacle-layout transfer and robustness benchmark rather than an in-distribution task benchmark.

Limitations:

- Training uses one deterministic built-in layout.
- The core suite is fixed, not sampled from a broad task distribution.
- At most three obstacles are active.
- The environment is a compact simulation.
- Projection is local and softened by slack, not a global safety guarantee.
- Clearance averages exclude the obstacle-free layout because clearance is undefined there.

## Freeze declaration

```text
suite=core_navigation_layouts_v1
decision=accepted unchanged
canonical_sha256=1027141797052240e83b941398e5a32031e9ca67d001e4a0cf1e19b0f96dd466
role=secondary deterministic transfer and robustness evaluation
```

If a genuine defect is later found, create a new version, document the policy-independent reason, and rerun every method, seed, and required projection mode. Do not silently edit this suite or selectively retain favorable prior results.
