# P1.3 Projection Solver-Failure Diagnostic Verification

## Purpose

This document defines and verifies the evaluator contract for projection slack diagnostics when the CBF-QP solver does not return a valid solution.

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

## Problem Addressed

The numerical projector already reports unsuccessful solves with:

```text
projection_success = False
projection_slack_max = NaN
projection_slack_sum = NaN
```

The evaluator previously accumulated the episode maximum with:

```python
max(current_max_slack, projection_slack_max)
```

Because comparisons with `NaN` are unordered, a failed step could leave the episode aggregate at its initial value of `0.0`. This incorrectly represented an unavailable slack solution as zero slack.

Zero slack has a specific numerical meaning: the successful QP required no relaxation of the active CBF constraints. It cannot also mean that the solver failed and no valid slack vector exists.

## Diagnostic Contract

The evaluator now uses the following contract.

### Successful projection step

A successful projection step must report finite `projection_slack_max` and `projection_slack_sum` values. The maximum value contributes to the episode maximum, and the sum value contributes to the episode mean. Valid zero-slack steps remain part of both episode statistics. A successful step with non-finite slack diagnostics raises a runtime error because it violates the projector-to-evaluator interface.

### Failed projection step

A failed projection step increments:

```text
projection_solver_failure_count
```

and sets both episode slack fields:

```text
mean_projection_slack_sum = NaN
max_projection_slack = NaN
```

Once a failure occurs, later successful steps do not overwrite the episode `NaN` values. The episode therefore states that complete mean and maximum slack statistics are unavailable.

### Multi-episode summary

The summary fields:

```text
mean_projection_slack_sum
max_projection_slack
```

are finite only when every included projection-enabled episode has zero solver failures and finite episode slack diagnostics. If any included episode contains a solver failure, both summary values are `NaN`.

The separate count:

```text
total_projection_solver_failures
```

remains the authoritative number of failed projection steps.

### CSV and terminal output

The per-episode CSV preserves both undefined slack values as `NaN` when read by pandas. The terminal summary prints:

```text
mean_projection_slack_sum:          N/A
max_projection_slack:               N/A
```

when the aggregates are undefined, followed by the explicit solver-failure count.

## Scope Boundary

This issue changes only the truthfulness of failure-time slack reporting. It does not change the QP, stationary failure fallback, wrapper diagnostics, projection parameters, or training-time projection.

The completed mean per-step slack propagation contract is defined separately in:

```text
docs/projection_slack_metrics_verification.md
```

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

### 3. Reproduce the failure-time episode contract

```bat
python -c "import numpy as np; from environments.factory import make_env; from evaluation.evaluate_policy import run_episode, summarize_results; from projection.cbf_qp_projection import ProjectionParams; action=lambda env,obs: np.zeros(2,dtype=np.float32); factory=make_env(env_index=0,env_kwargs={'max_episode_steps':2},record_episode_statistics=False,normalize_actions=True,enable_projection=True,projection_params=ProjectionParams(solver_name='INVALID_SOLVER')); env=factory(); result=run_episode(env=env,action_provider=action,seed=11,episode=0,policy_name='random'); env.close(); summary=summarize_results([result]); print(result); print(summary); assert result.projection_solver_failure_count==2; assert np.isnan(result.mean_projection_slack_sum); assert np.isnan(result.max_projection_slack); assert summary['total_projection_solver_failures']==2; assert np.isnan(summary['mean_projection_slack_sum']); assert np.isnan(summary['max_projection_slack'])"
```

Acceptance criterion:

```text
The episode reports two solver failures, and both episode-level and summary slack statistics are NaN.
```

### 4. Run the complete lightweight suite

```bat
python -m pytest -q -rs
```

Acceptance criterion:

```text
All lightweight tests pass.
```

## Completion Criteria

The solver-failure diagnostic issue is complete when all of the following conditions hold:

1. A failed projection step increments the solver-failure count.
2. Any failed projection step makes both episode slack statistics undefined.
3. Later successful steps cannot overwrite those undefined episode values.
4. A multi-episode summary keeps both slack statistics undefined if any included episode contains a solver failure.
5. Successful zero-slack steps continue to report valid numeric zeros.
6. A successful projection with non-finite maximum or summed slack is rejected as an interface error.
7. The CSV preserves both undefined values.
8. The terminal summary prints `N/A` rather than `0.000000` for both fields.
9. The focused evaluator tests and complete lightweight suite pass.
