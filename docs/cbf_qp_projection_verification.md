# P1.3 CBF-QP Numerical Projection Verification

## Purpose

This document records the minimal verification procedure for the numerical CBF-QP projection core. The verification scope is limited to the pure physical-action projection module and its smoke tests.

The verified module is:

```text
projection/cbf_qp_projection.py
```

The verified smoke-test file is:

```text
tests/test_cbf_qp_projection.py
```

This document does not cover the Gymnasium projection wrapper, environment factory integration, evaluator integration, PPO retraining, or result-table generation.

## Repository Context

Repository root:

```text
C:\rl_projects\src\repos\rl-constrained-decision-making
```

Conda environment:

```text
RL_PROJECTS
```

The projection core operates in physical action coordinates:

```text
u = [v, omega]
```

The projection core does not accept normalized PPO actions. Normalized PPO actions are converted to physical actions before projection by the normalized action wrapper.

## Verification Rationale

The numerical projection core must be validated before adding the Gymnasium wrapper. This isolates mathematical and solver issues from environment integration issues.

If the numerical tests pass before wrapper integration, later failures are more likely to involve wrapper ordering, environment-state extraction, action-space semantics, or diagnostic propagation. If the numerical tests fail before wrapper integration, the failure should be corrected in the projection core before any repository integration work proceeds.

## Minimal Verification Commands

Run the following commands from Anaconda Prompt or another shell where `conda` is available.

### 1. Activate the project environment

```bat
conda activate RL_PROJECTS
```

### 2. Enter the repository root

```bat
cd /d C:\rl_projects\src\repos\rl-constrained-decision-making
```

### 3. Verify CVXPY and OSQP availability

```bat
python -c "import cvxpy as cp, osqp; print('cvxpy', cp.__version__); print('osqp', osqp.__version__); print('installed_solvers', cp.installed_solvers()); assert 'OSQP' in cp.installed_solvers()"
```

Acceptance criterion:

```text
The command prints CVXPY and OSQP versions, prints the installed solver list, and exits without an AssertionError.
```

### 4. Compile the projection module and smoke-test file

```bat
python -m compileall -q projection/cbf_qp_projection.py tests/test_cbf_qp_projection.py
```

Acceptance criterion:

```text
The command exits without syntax errors.
```

### 5. Run the minimal projection smoke tests

```bat
python -m pytest tests/test_cbf_qp_projection.py -q -rs
```

Acceptance criterion:

```text
All tests pass. No CVXPY-dependent test is skipped after CVXPY and OSQP are installed.
```

Expected result after dependency installation:

```text
4 passed
```

### 6. Run a direct active-obstacle projection check

```bat
python -c "import numpy as np; from projection.cbf_qp_projection import ProjectionParams, project_physical_action; params=ProjectionParams(v_max=1.0, omega_max=2.0, lookahead_distance=0.25, alpha=2.0, safety_margin=0.0, slack_penalty=10000.0, solver_name='OSQP'); result=project_physical_action(position=np.array([0.0, 0.0]), heading=0.0, obstacle_centers=np.array([[0.75, 0.0]]), obstacle_radii=np.array([0.25]), obstacle_mask=np.array([True]), raw_action=np.array([1.0, 0.0]), params=params); print('status=', result.solver_status); print('success=', result.success); print('intervened=', result.intervened); print('action_raw=', result.action_raw); print('action_exec=', result.action_exec); print('correction_norm=', result.correction_norm); print('slack_max=', result.slack_max); assert result.success; assert result.intervened; assert result.action_exec.shape == (2,); assert np.all(np.isfinite(result.action_exec)); assert 0.0 <= result.action_exec[0] <= params.v_max; assert -params.omega_max <= result.action_exec[1] <= params.omega_max; assert result.action_exec[0] < 0.5"
```

Acceptance criterion:

```text
The command reports an optimal or optimal_inaccurate solver status, reports success=True, reports intervened=True, and exits without an AssertionError.
```

### 7. Verify the stationary solver-failure fallback

```bat
python -c "import numpy as np; from projection.cbf_qp_projection import ProjectionParams, project_physical_action; result=project_physical_action(position=np.array([0.0, 0.0]), heading=0.0, obstacle_centers=np.array([[0.75, 0.0]]), obstacle_radii=np.array([0.25]), obstacle_mask=np.array([True]), raw_action=np.array([1.0, 0.0]), params=ProjectionParams(solver_name='INVALID_SOLVER')); print('status=', result.solver_status); print('success=', result.success); print('action_exec=', result.action_exec); assert not result.success; assert result.solver_status == 'solver_error'; np.testing.assert_allclose(result.action_exec, np.zeros(2))"
```

Acceptance criterion:

```text
The command reports success=False, reports solver_error, returns action_exec=[0.0, 0.0], and exits without an AssertionError.
```

## Smoke-Test Coverage

The current minimal smoke tests cover only the following behavior:

| Check | Purpose |
|---|---|
| Obstacle-free in-bounds action remains unchanged | Verifies the no-active-constraint path and result construction. |
| Obstacle-free out-of-bounds action is clipped | Verifies physical action bounds without invoking the QP solver. |
| Forward action near an obstacle is modified | Verifies active CBF constraint construction and OSQP-backed projection. |
| Solver failure returns a stationary action | Verifies that projection failure does not execute the raw action. |

This is intentionally not a full unit-test suite. The purpose is to validate minimal functionality before wrapper integration.

## Numerical Completion Criteria

The numerical projection core is considered ready for wrapper integration when all of the following conditions hold:

1. CVXPY imports successfully.
2. OSQP appears in `cvxpy.installed_solvers()`.
3. `projection/cbf_qp_projection.py` compiles without syntax errors.
4. `tests/test_cbf_qp_projection.py` compiles without syntax errors.
5. `python -m pytest tests/test_cbf_qp_projection.py -q -rs` reports all tests passed.
6. The direct active-obstacle projection command reports a successful intervention and returns a finite bounded physical action.
7. The solver-failure check returns the stationary physical action `[0.0, 0.0]` and reports `success=False`.


