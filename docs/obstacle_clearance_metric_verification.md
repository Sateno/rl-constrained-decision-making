# Obstacle Clearance Metric Verification

## Purpose

This document defines and verifies the canonical obstacle-clearance metric used by the constrained-navigation environment and checkpoint evaluator.

The implementation files are:

```text
environments/constrained_navigation.py
evaluation/evaluate_policy.py
```

The focused smoke-test files are:

```text
tests/test_constrained_navigation.py
tests/test_evaluate_policy.py
```

## Problem Addressed

The repository previously used the name:

```text
min_obstacle_distance
```

The stored value was not center-to-center distance. It was the signed clearance between the agent collision disk and the nearest active obstacle collision disk. The old name was therefore ambiguous in environment diagnostics, evaluation CSV files, summaries, scripts, and research tables.

The repository now uses `clearance` consistently and does not retain the ambiguous name as a compatibility alias.

## Canonical Definition

For agent position `p`, active obstacle center `o_i`, obstacle radius `r_i`, and agent radius `r_agent`, define:

```text
clearance_i
    = ||p - o_i||_2 - (r_i + r_agent)
```

The environment metric is:

```text
min_obstacle_clearance
    = min_i clearance_i
```

where the minimum is taken only over active obstacles.

The sign has the following meaning:

| Value | Interpretation |
|---:|---|
| greater than zero | geometric separation from every active collision boundary |
| equal to zero | contact with at least one collision boundary |
| less than zero | collision-boundary penetration |

The collision event remains:

```text
collision = min_obstacle_clearance <= 0
```

when at least one obstacle is active.

## No-Active-Obstacle Convention

When no obstacle is active, a nearest-obstacle clearance does not exist. The environment therefore reports:

```text
min_obstacle_clearance = NaN
collision = False
```

The previous finite sentinel `1.0e6` has been removed. A large finite placeholder could be averaged into a scientific result as though it were an observed geometric clearance.

The evaluator excludes non-finite clearance values when computing the cross-episode mean. If every evaluated episode has no active obstacle, it reports:

```text
mean_min_obstacle_clearance = NaN
```

The terminal summary renders that undefined aggregate as `N/A`. This preserves the distinction between an undefined metric and a very large measured clearance.

## Canonical Names

The current environment and evaluation schema uses:

```text
environment info:       min_obstacle_clearance
EpisodeResult field:    min_obstacle_clearance
CSV column:             min_obstacle_clearance
summary key:            mean_min_obstacle_clearance
terminal label:         mean_min_obstacle_clearance
```

The following names are retired from newly generated artifacts:

```text
min_obstacle_distance
mean_min_obstacle_distance
```

## Compatibility

The change does not alter:

- environment observations or observation dimension;
- environment dynamics, rewards, collision geometry, or termination;
- normalized or physical action semantics;
- projection geometry or solver behavior;
- PPO model architecture or checkpoint dimensions;
- numeric clearance values for episodes with active obstacles.

Existing PPO checkpoints remain compatible because the checkpoint interface depends on observation and action dimensions, not on `info` dictionary key names.

Historical raw CSV files generated before this correction may retain the retired header. They are not silently rewritten. New evaluator runs use the canonical schema.

## Modified Files

The metric contract is implemented and recorded in:

```text
environments/constrained_navigation.py
evaluation/evaluate_policy.py
tests/test_constrained_navigation.py
tests/test_evaluate_policy.py
scripts/run_ppo_baseline_clean.bat
docs/constrained_navigation_verification.md
docs/ppo_baseline_verification.md
docs/projection_evaluation_parameters_verification.md
docs/obstacle_clearance_metric_verification.md
```

## Minimal Verification Commands

Run the commands from the repository root after activating the `RL_PROJECTS` Conda environment.

### 1. Compile the modified Python files

```bat
python -m compileall -q ^
  environments\constrained_navigation.py ^
  evaluation\evaluate_policy.py ^
  tests\test_constrained_navigation.py ^
  tests\test_evaluate_policy.py
```

Acceptance criterion:

```text
The command exits without syntax errors.
```

### 2. Run focused environment and evaluator tests

```bat
python -m pytest tests\test_constrained_navigation.py tests\test_evaluate_policy.py -q -rs
```

Acceptance criterion:

```text
All focused tests pass.
```

### 3. Verify the geometric value

```bat
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(agent_radius=0.10); centers=np.zeros((3,2)); radii=np.zeros(3); mask=np.zeros(3,dtype=bool); centers[0]=[1.0,0.0]; radii[0]=0.25; mask[0]=True; _,info=env.reset(seed=0, options={'start':np.array([0.0,0.0]),'theta':0.0,'goal':np.array([5.0,0.0]),'obstacle_centers':centers,'obstacle_radii':radii,'obstacle_mask':mask}); print(info); assert abs(info['min_obstacle_clearance']-0.65)<1e-12; assert info['collision'] is False; env.close()"
```

Acceptance criterion:

```text
min_obstacle_clearance = 0.65
collision = False
```

### 4. Verify the no-active-obstacle convention

```bat
python -c "from environments.constrained_navigation import ConstrainedNavigationEnv; import numpy as np; env=ConstrainedNavigationEnv(num_active_obstacles=0); _,info=env.reset(seed=0); print(info); assert np.isnan(info['min_obstacle_clearance']); assert info['collision'] is False; env.close()"
```

Acceptance criterion:

```text
min_obstacle_clearance = NaN
collision = False
```

### 5. Verify the evaluator CSV schema

```bat
python -m evaluation.evaluate_policy ^
  --policy random ^
  --episodes 1 ^
  --seed 123 ^
  --max-episode-steps 5 ^
  --no-cuda ^
  --output runs\evaluation\obstacle_clearance_schema.csv
```

```bat
python -c "import pandas as pd; df=pd.read_csv(r'runs\evaluation\obstacle_clearance_schema.csv'); print(df.columns.tolist()); assert 'min_obstacle_clearance' in df.columns; assert 'min_obstacle_distance' not in df.columns"
```

### 6. Run the complete lightweight suite

```bat
python -m pytest -q -rs
```

Acceptance criterion:

```text
All lightweight tests pass.
```

### 7. Verify existing checkpoint compatibility

```bat
python -m evaluation.evaluate_policy ^
  --policy ppo ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --episodes 20 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --no-cuda ^
  --output runs\evaluation\ppo_clearance_metric_validation.csv
```

Acceptance criterion for the existing deterministic checkpoint and default layout:

```text
mean return approximately 12.320
success rate 1.00
collision rate 0.00
mean_min_obstacle_clearance approximately 0.518163
```

## Completion Criteria

The metric issue is complete when all of the following conditions hold:

1. `min_obstacle_clearance` is the only current environment and episode-field name.
2. `mean_min_obstacle_clearance` is the only current evaluator summary name.
3. Active-obstacle values equal signed collision-boundary clearance.
4. No-active-obstacle episodes use `NaN`, not a finite sentinel.
5. Cross-episode means exclude undefined clearance values.
6. Newly generated CSV files use the canonical column.
7. Existing PPO checkpoint dimensions and deterministic baseline behavior remain unchanged.
8. Focused tests and the complete lightweight suite pass.

## Scope Boundary

This issue does not modify:

- projection slack accumulation;
- solver-failure slack reporting;
- paired projection-off/on orchestration;
- trajectory recording;
- training-time projection;
- comparative experiment execution.
