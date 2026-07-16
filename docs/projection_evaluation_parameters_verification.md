# P1.3 Projection Evaluation Parameter Verification

## Purpose

This document records the evaluator-level contract for configuring and persisting the CBF-QP projection hyperparameters used during P1.3 checkpoint evaluation.

The implementation file is:

```text
evaluation/evaluate_policy.py
```

The focused smoke-test file is:

```text
tests/test_evaluate_policy.py
```

The projection geometry and numerical QP mechanics remain defined in:

```text
projection/cbf_qp_projection.py
projection/cbf_qp_wrapper.py
```

## Problem Addressed

The evaluator previously enabled projection while passing:

```python
projection_params=None
```

That behavior always selected repository defaults and provided no command-line path for changing the projection geometry or slack penalty. The resulting episode CSV also omitted those values, so the artifact did not identify the exact projection configuration that produced it.

This was a reproducibility defect rather than a numerical projection defect. The numerical projector already accepted an explicit `ProjectionParams` object.

## Evaluator Contract

The evaluator now exposes these arguments:

```text
--projection-lookahead-distance
--projection-alpha
--projection-slack-penalty
--projection-extra-clearance
```

They map directly to:

```text
ProjectionParams.lookahead_distance
ProjectionParams.alpha
ProjectionParams.slack_penalty
ProjectionParams.extra_clearance
```

The evaluator constructs one `ProjectionParams` object per command invocation. The same object is used for both of the following operations:

1. passing the projection configuration through `environments.factory.make_env(...)` to `CbfQpProjectionWrapper`;
2. writing projection configuration columns into every episode row in the evaluation CSV.

This avoids separate runtime and artifact configurations that could drift apart.

## Validation Rules

The evaluator rejects non-finite or invalid values before environment construction:

| Argument | Required domain |
|---|---|
| `--projection-lookahead-distance` | finite and greater than or equal to zero |
| `--projection-alpha` | finite and greater than zero |
| `--projection-slack-penalty` | finite and greater than zero |
| `--projection-extra-clearance` | finite and greater than or equal to zero |

`ProjectionParams` retains its own numerical validation as the lower-level contract.

## Persisted CSV Columns

Every episode row contains:

```text
projection_lookahead_distance
projection_alpha
projection_slack_penalty
projection_extra_clearance
```

These columns are present for both projection-disabled and projection-enabled evaluations. `projection_enabled` remains the field that states whether the configuration was active for a particular episode.

Keeping the configuration columns in both modes preserves a stable CSV schema and makes paired projection-disabled and projection-enabled files directly comparable.

## Default Behavior

The evaluator CLI defaults are read from `ProjectionParams()` rather than duplicated as independent constants. The current defaults are:

```text
projection_lookahead_distance = 0.25
projection_alpha = 2.0
projection_slack_penalty = 1000.0
projection_extra_clearance = 0.0
```

The projection wrapper continues to replace only `v_max` and `omega_max` with the physical action bounds owned by the environment. The four evaluator-exposed hyperparameters remain unchanged during that handoff.

The solver name, solver verbosity, and correction tolerance remain at the repository defaults in this issue. They are not exposed as evaluator arguments.

## Minimal Verification Commands

Run the following commands from the repository root after activating the `RL_PROJECTS` Conda environment.

### 1. Compile the evaluator and focused smoke test

```bat
python -m compileall -q evaluation/evaluate_policy.py tests/test_evaluate_policy.py
```

Acceptance criterion:

```text
The command exits without syntax errors.
```

### 2. Verify command-line exposure

```bat
python -m evaluation.evaluate_policy --help | findstr /C:"--projection-lookahead-distance" /C:"--projection-alpha" /C:"--projection-slack-penalty" /C:"--projection-extra-clearance"
```

Acceptance criterion:

```text
All four arguments are present in the help output.
```

### 3. Run the focused evaluator smoke test

```bat
python -m pytest tests/test_evaluate_policy.py -q -rs
```

Acceptance criterion:

```text
3 passed
```

The parameter smoke test verifies the complete configuration path:

```text
CLI arguments
    -> ProjectionParams
    -> environment factory
    -> CbfQpProjectionWrapper
    -> evaluation CSV columns
```

It uses zero active obstacles and a one-step episode, so it does not require a QP solve.

### 4. Run a projection-enabled evaluation with explicit parameters

```bat
python -m evaluation.evaluate_policy ^
  --policy random ^
  --episodes 2 ^
  --seed 123 ^
  --max-episode-steps 20 ^
  --enable-projection ^
  --projection-lookahead-distance 0.40 ^
  --projection-alpha 3.50 ^
  --projection-slack-penalty 2500.0 ^
  --projection-extra-clearance 0.08 ^
  --no-cuda ^
  --output runs\evaluation\projection_parameter_smoke.csv
```

### 5. Inspect the persisted values

```bat
python -c "import pandas as pd; p=r'runs\evaluation\projection_parameter_smoke.csv'; df=pd.read_csv(p); cols=['projection_enabled','projection_lookahead_distance','projection_alpha','projection_slack_penalty','projection_extra_clearance']; print(df[cols]); assert df['projection_enabled'].all(); assert set(df['projection_lookahead_distance'])=={0.4}; assert set(df['projection_alpha'])=={3.5}; assert set(df['projection_slack_penalty'])=={2500.0}; assert set(df['projection_extra_clearance'])=={0.08}"
```

Acceptance criterion:

```text
Every row contains the exact values supplied on the command line.
```

### 6. Run the complete lightweight suite

```bat
python -m pytest -q -rs
```

Acceptance criterion:

```text
All lightweight tests pass.
```

## Completion Criteria

The evaluator parameter issue is complete when all of the following conditions hold:

1. All four projection hyperparameters are exposed through explicit CLI arguments.
2. Invalid values are rejected before an evaluation begins.
3. One `ProjectionParams` object is constructed for the run.
4. The same object is passed through the environment factory.
5. The same object supplies the projection metadata written to the CSV.
6. Every episode row contains the four projection configuration columns.
7. Default projection behavior remains numerically unchanged.
8. The focused evaluator smoke test and the complete lightweight suite pass.

## Scope Boundary

This issue does not change:

- environment geometry;
- numerical CBF-QP equations;
- wrapper order;
- PPO policy or checkpoint loading;
- obstacle-clearance metric naming or aggregation;
- slack aggregation or solver-failure semantics;
- trajectory recording;
- training-time projection.

The current obstacle-clearance metric contract is defined separately in:

```text
docs/obstacle_clearance_metric_verification.md
```

The current solver-failure slack reporting contract is defined separately in:

```text
docs/projection_failure_diagnostics_verification.md
```

The current projection slack metric contract is defined separately in:

```text
docs/projection_slack_metrics_verification.md
```

The paired projection-disabled/projection-enabled orchestration contract is defined separately in:

```text
docs/paired_projection_evaluation_verification.md
```
