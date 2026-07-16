# P1.3 Projection Slack Metric Verification

## Purpose

This document defines and verifies the evaluator contract for propagating per-step CBF-QP slack diagnostics into episode CSV rows and multi-episode summaries.

The implementation file is:

```text
evaluation/evaluate_policy.py
```

The focused smoke-test file is:

```text
tests/test_evaluate_policy.py
```

The numerical projector and wrapper remain defined in:

```text
projection/cbf_qp_projection.py
projection/cbf_qp_wrapper.py
```

## Metric Meaning

For one successful projection step, the numerical projector returns one nonnegative slack value for each active CBF constraint. The wrapper reports:

```text
projection_slack_sum = sum of active-obstacle slack values for the step
projection_slack_max = maximum active-obstacle slack value for the step
```

The evaluator now records two complementary episode statistics:

```text
mean_projection_slack_sum
    = arithmetic mean of projection_slack_sum over all episode steps

max_projection_slack
    = maximum projection_slack_max over all episode steps
```

In mathematical form, with episode length `T` and per-step total slack `Xi_t`:

```text
mean_projection_slack_sum = (1 / T) * sum_t Xi_t
```

This is the episode-level statistic required for interpreting average CBF constraint relaxation. `max_projection_slack` remains the worst individual obstacle slack observed during the episode.

## Successful-Step Contract

Every projection-enabled successful step must report finite values for both:

```text
projection_slack_sum
projection_slack_max
```

A successful step with non-finite slack diagnostics raises a runtime interface error.

Zero-slack successful steps are included in the episode mean. This is intentional. A zero value is a valid observation that the QP required no relaxation on that step. Excluding zero-slack steps would overstate average slack burden.

The no-active-constraint path is also a successful zero-slack step:

```text
projection_slack_sum = 0.0
projection_slack_max = 0.0
```

## Solver-Failure Contract

A failed projection solve has no valid slack solution. If any projection step fails, the evaluator records:

```text
projection_solver_failure_count > 0
mean_projection_slack_sum = NaN
max_projection_slack = NaN
```

Later successful steps cannot replace those undefined episode values. The failure count remains the authoritative number of failed steps.

The detailed failure-time contract is also recorded in:

```text
docs/projection_failure_diagnostics_verification.md
```

## Multi-Episode Summary Contract

For projection-enabled evaluation with no solver failures, the summary field:

```text
mean_projection_slack_sum
```

is the arithmetic mean of the episode-level `mean_projection_slack_sum` values. Episodes receive equal weight, consistent with the evaluator's other episode-level summary metrics.

The summary field:

```text
max_projection_slack
```

is the maximum episode-level `max_projection_slack` value.

If any included episode contains a solver failure, both summary slack fields are `NaN`:

```text
mean_projection_slack_sum = NaN
max_projection_slack = NaN
```

The terminal report prints `N/A` for either non-finite aggregate.

## CSV Contract

`EpisodeResult` and every evaluator CSV row now contain:

```text
mean_projection_slack_sum
max_projection_slack
projection_solver_failure_count
```

Projection-disabled rows retain the fixed evaluator schema and use the existing zero-valued projection placeholders together with:

```text
projection_enabled = False
```

Projection-specific summary fields continue to be emitted only when projection is enabled.

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

### 2. Run the focused evaluator tests

```bat
python -m pytest tests/test_evaluate_policy.py -q -rs
```

Acceptance criterion:

```text
3 passed
```

The focused mean-slack test uses three successful scripted projection steps:

```text
projection_slack_sum values: 0.00, 0.30, 0.90
projection_slack_max values: 0.00, 0.20, 0.50
```

The expected episode metrics are:

```text
mean_projection_slack_sum = 0.40
max_projection_slack = 0.50
```

The zero-slack first step is included in the mean.

### 3. Verify successful zero-slack propagation through the real wrapper

```bat
python -m evaluation.evaluate_policy ^
  --policy random ^
  --episodes 1 ^
  --seed 7 ^
  --max-episode-steps 1 ^
  --num-active-obstacles 0 ^
  --enable-projection ^
  --no-cuda ^
  --output runs\evaluation\projection_zero_slack_smoke.csv
```

Inspect the result:

```bat
python -c "import pandas as pd; p=r'runs\evaluation\projection_zero_slack_smoke.csv'; row=pd.read_csv(p).iloc[0]; print(row[['projection_enabled','mean_projection_slack_sum','max_projection_slack','projection_solver_failure_count']]); assert bool(row['projection_enabled']); assert row['mean_projection_slack_sum']==0.0; assert row['max_projection_slack']==0.0; assert row['projection_solver_failure_count']==0"
```

### 4. Verify failure-time undefined slack metrics

```bat
python -c "import numpy as np; from environments.factory import make_env; from evaluation.evaluate_policy import run_episode, summarize_results; from projection.cbf_qp_projection import ProjectionParams; action=lambda env,obs: np.zeros(2,dtype=np.float32); factory=make_env(env_index=0,env_kwargs={'max_episode_steps':2},record_episode_statistics=False,normalize_actions=True,enable_projection=True,projection_params=ProjectionParams(solver_name='INVALID_SOLVER')); env=factory(); result=run_episode(env=env,action_provider=action,seed=11,episode=0,policy_name='random'); env.close(); summary=summarize_results([result]); print(result); print(summary); assert result.projection_solver_failure_count==2; assert np.isnan(result.mean_projection_slack_sum); assert np.isnan(result.max_projection_slack); assert np.isnan(summary['mean_projection_slack_sum']); assert np.isnan(summary['max_projection_slack'])"
```

### 5. Run the complete lightweight suite

```bat
python -m pytest -q -rs
```

Acceptance criterion:

```text
All lightweight tests pass.
```

The paired projection-disabled/projection-enabled orchestration contract is defined separately in:

```text
docs/paired_projection_evaluation_verification.md
```

## Scope Boundary

This issue changes only evaluator propagation and reporting of the existing `projection_slack_sum` diagnostic. It does not change:

- CBF geometry or QP equations;
- slack-variable optimization;
- projection action behavior;
- stationary solver-failure fallback;
- projection hyperparameters;
- obstacle-clearance metrics;
- trajectory recording;
- training-time projection.

## Completion Criteria

The mean-slack propagation issue is complete when all of the following conditions hold:

1. `EpisodeResult` contains `mean_projection_slack_sum`.
2. Every successful projection step contributes its finite `projection_slack_sum` value.
3. Successful zero-slack steps remain in the episode denominator.
4. The episode mean divides by complete episode length.
5. The CSV preserves the episode mean.
6. Projection-enabled summaries average the episode means.
7. A failed projection step makes both episode slack statistics undefined.
8. A summary containing any solver failure keeps both slack statistics undefined.
9. Terminal output prints finite values or `N/A` according to the diagnostic state.
10. The focused evaluator tests and complete lightweight suite pass.
