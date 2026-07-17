# CBF-QP Projection Geometry Verification

## Purpose

This document records the minimal verification procedure for the numerical CBF-QP projection core and the environment-to-projector radius handoff.

The verified implementation files are:

```text
environments/constrained_navigation.py
projection/cbf_qp_projection.py
projection/cbf_qp_wrapper.py
```

The verified smoke-test files are:

```text
tests/test_cbf_qp_projection.py
tests/test_cbf_qp_wrapper.py
```

Evaluator parameter exposure and CSV persistence are recorded separately in `docs/projection_evaluation_parameters_verification.md`. Episode mean/max slack propagation is recorded in `docs/projection_slack_metrics_verification.md`, solver-failure slack semantics are recorded in `docs/projection_failure_diagnostics_verification.md`, paired checkpoint evaluation is recorded in `docs/paired_projection_evaluation_verification.md`, and raw/executed action trajectory auditability is recorded in `docs/trajectory_audit_verification.md`. This document does not cover PPO retraining or final result-table generation.

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

## Geometry Contract

The repository uses separate collision and projection radii:

```text
collision_radius
    = obstacle_radius + agent_radius

projection_radius
    = obstacle_radius + agent_radius + extra_clearance
```

`obstacle_radius` is the physical radius stored in `ConstrainedNavigationEnv.obstacle_radii`. `agent_radius` is the physical footprint stored in `ConstrainedNavigationEnv.agent_radius`. `ProjectionParams.extra_clearance` is an optional nonnegative projection-only buffer.

The environment computes collision clearance using `collision_radius`. The projection wrapper passes `agent_radius` explicitly to the numerical projector for both default and custom `ProjectionParams`. The numerical projector then adds `extra_clearance` when constructing each CBF barrier.

The default `extra_clearance=0.0` preserves the existing default projection geometry:

```text
projection_radius = obstacle_radius + agent_radius
```

The unused environment constructor parameter formerly named `safety_margin` has been removed. Projection-only clearance now belongs exclusively to `ProjectionParams.extra_clearance`.

This implementation contract supersedes the ambiguous use of a single safety-margin term in the original design report.

## Verification Rationale

The collision footprint must not depend on whether projection parameters are defaulted or supplied explicitly. Without an explicit agent-radius handoff, a custom `ProjectionParams` object could construct barriers around obstacle disks alone and omit the agent footprint.

The focused tests therefore protect two distinct mechanics:

1. the numerical barrier radius is exactly `obstacle_radius + agent_radius + extra_clearance`;
2. the Gymnasium wrapper supplies the environment agent radius even when custom projection parameters are used.

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

### 4. Compile the modified modules and smoke tests

```bat
python -m compileall -q environments/constrained_navigation.py projection/cbf_qp_projection.py projection/cbf_qp_wrapper.py tests/test_cbf_qp_projection.py tests/test_cbf_qp_wrapper.py
```

Acceptance criterion:

```text
The command exits without syntax errors.
```

### 5. Run the focused projection smoke tests

```bat
python -m pytest tests/test_cbf_qp_projection.py tests/test_cbf_qp_wrapper.py -q -rs
```

Acceptance criterion:

```text
All tests pass. No CVXPY-dependent test is skipped after CVXPY and OSQP are installed.
```

Expected result after dependency installation:

```text
7 passed
```

### 6. Verify the radius equation directly

```bat
python -c "import numpy as np; from projection.cbf_qp_projection import ProjectionParams, build_cbf_constraints; params=ProjectionParams(lookahead_distance=0.0, extra_clearance=0.20); data=build_cbf_constraints(position=np.array([0.0, 0.0]), heading=0.0, obstacle_centers=np.array([[1.0, 0.0]]), obstacle_radii=np.array([0.50]), obstacle_mask=np.array([True]), agent_radius=0.10, params=params); expected_radius=0.50+0.10+0.20; expected_h=1.0-expected_radius**2; print('h=', data.h_values[0]); print('expected_h=', expected_h); np.testing.assert_allclose(data.h_values, np.array([expected_h]))"
```

Acceptance criterion:

```text
The command confirms that the barrier uses projection_radius=0.80 and exits without an AssertionError.
```

### 7. Run a direct active-obstacle projection check

```bat
python -c "import numpy as np; from projection.cbf_qp_projection import ProjectionParams, project_physical_action; params=ProjectionParams(v_max=1.0, omega_max=2.0, lookahead_distance=0.25, alpha=2.0, extra_clearance=0.0, slack_penalty=10000.0, solver_name='OSQP'); result=project_physical_action(position=np.array([0.0, 0.0]), heading=0.0, obstacle_centers=np.array([[0.75, 0.0]]), obstacle_radii=np.array([0.25]), obstacle_mask=np.array([True]), agent_radius=0.0, raw_action=np.array([1.0, 0.0]), params=params); print('status=', result.solver_status); print('success=', result.success); print('intervened=', result.intervened); print('action_raw=', result.action_raw); print('action_exec=', result.action_exec); print('correction_norm=', result.correction_norm); print('slack_max=', result.slack_max); assert result.success; assert result.intervened; assert result.action_exec.shape == (2,); assert np.all(np.isfinite(result.action_exec)); assert 0.0 <= result.action_exec[0] <= params.v_max; assert -params.omega_max <= result.action_exec[1] <= params.omega_max; assert result.action_exec[0] < 0.5"
```

Acceptance criterion:

```text
The command reports an optimal or optimal_inaccurate solver status, reports success=True, reports intervened=True, and exits without an AssertionError.
```

### 8. Verify the stationary solver-failure fallback

```bat
python -c "import numpy as np; from projection.cbf_qp_projection import ProjectionParams, project_physical_action; result=project_physical_action(position=np.array([0.0, 0.0]), heading=0.0, obstacle_centers=np.array([[0.75, 0.0]]), obstacle_radii=np.array([0.25]), obstacle_mask=np.array([True]), agent_radius=0.0, raw_action=np.array([1.0, 0.0]), params=ProjectionParams(solver_name='INVALID_SOLVER')); print('status=', result.solver_status); print('success=', result.success); print('action_exec=', result.action_exec); assert not result.success; assert result.solver_status == 'solver_error'; np.testing.assert_allclose(result.action_exec, np.zeros(2))"
```

Acceptance criterion:

```text
The command reports success=False, reports solver_error, returns action_exec=[0.0, 0.0], and exits without an AssertionError.
```

## Smoke-Test Coverage

| Check | Purpose |
|---|---|
| Obstacle-free in-bounds action remains unchanged | Verifies the no-active-constraint path and result construction. |
| Obstacle-free out-of-bounds action is clipped | Verifies physical action bounds without invoking the QP solver. |
| Projection radius combines obstacle radius, agent radius, and extra clearance | Protects the explicit geometry contract. |
| Forward action near an obstacle is modified | Verifies active CBF constraint construction and OSQP-backed projection. |
| Solver failure returns a stationary action | Verifies that projection failure does not execute the raw action. |
| Custom wrapper parameters still include the environment agent radius | Prevents omission of the robot footprint when parameters are supplied explicitly. |
| Normalized-action wrapper remains outside the projection wrapper | Protects physical-action projection and wrapper ordering. |

This is intentionally not a full unit-test suite. The purpose is to protect the scientifically critical projection mechanics without freezing internal implementation details.

## Completion Criteria

The radius-semantics issue is complete when all of the following conditions hold:

1. The environment collision radius is `obstacle_radius + agent_radius`.
2. `ProjectionParams` exposes `extra_clearance` and no longer exposes the ambiguous `safety_margin` field.
3. The projector requires `agent_radius` as an explicit geometry input.
4. The projection radius is `obstacle_radius + agent_radius + extra_clearance`.
5. The wrapper passes `ConstrainedNavigationEnv.agent_radius` for both default and custom projection parameters.
6. Focused geometry and wrapper tests pass.
7. The stationary solver-failure fallback remains unchanged.
8. The complete lightweight repository test suite passes before validation closure.
