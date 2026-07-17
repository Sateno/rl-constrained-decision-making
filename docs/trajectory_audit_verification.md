# Trajectory and Action Audit Verification

## Purpose

This document defines and verifies the evaluator trajectory artifact used to audit the raw policy action, the physical action presented to the projection layer, the executed physical action, and the resulting state transition.

The implementation files are:

```text
evaluation/trajectory_recording.py
evaluation/evaluate_policy.py
evaluation/evaluate_projection_pair.py
projection/cbf_qp_wrapper.py
```

The focused smoke-test files are:

```text
tests/test_cbf_qp_wrapper.py
tests/test_evaluate_policy.py
tests/test_evaluate_projection_pair.py
```

## Problem Addressed

Episode-level CSV rows preserve aggregate return, task outcomes, correction burden, slack, and solver-failure counts. They do not establish which raw action was proposed at each state or which physical action was executed after projection.

The evaluation action path contains three distinct numerical objects:

```text
normalized policy action
    -> physical raw action
    -> physical executed action
```

The first action belongs to the PPO-facing normalized action space. The second is the normalized wrapper output and the input to the physical CBF-QP projector. The third is the action passed to `ConstrainedNavigationEnv.step(...)`.

A trajectory artifact is required to preserve these distinctions and to support later trajectory figures, intervention inspection, and scientific auditing without rerunning evaluation.

## Wrapper Diagnostic Contract

`CbfQpProjectionWrapper` now adds the following vector diagnostics to the environment `info` dictionary:

```text
projection_action_raw        shape (2,)
projection_action_exec       shape (2,)
projection_correction        shape (2,)
projection_slack_values      shape (max_obstacles,)
```

The existing scalar diagnostics remain unchanged:

```text
projection_enabled
projection_intervened
projection_correction_norm
projection_slack_sum
projection_slack_max
projection_success
projection_solver_status
projection_active_constraint_count
```

The fixed-capacity slack vector is aligned with obstacle slots. Active obstacle entries contain the solver slack values. Inactive entries are zero. A failed solve preserves `NaN` in active entries because no valid slack solution exists.

The following invariants are checked before trajectory storage:

```text
projection_action_raw
    == normalized-wrapper physical action

projection_correction
    == projection_action_exec - projection_action_raw

projection_correction_norm
    == ||projection_correction||_2
```

## State and Transition Alignment

For an episode of length `T`, the trajectory archive stores:

```text
state arrays:       T + 1 samples
transition arrays:  T samples
```

The first state sample is recorded immediately after `env.reset(...)`. Each transition sample is followed by one post-step state sample.

This convention gives direct alignment:

```text
state[t]
action_raw[t]
action_exec[t]
reward[t]
state[t + 1]
```

## NPZ Archive Contract

The archive format identifier is:

```text
evaluation_trajectory_v1
```

One compressed NPZ file may contain multiple variable-length episodes. Each episode has a deterministic key:

```text
episode_0000
episode_0001
...
```

No object arrays are used. The archive can therefore be loaded with:

```python
np.load(path, allow_pickle=False)
```

### Archive-level fields

```text
trajectory_archive_version
episode_count
episode_keys
projection_v_max
projection_omega_max
projection_lookahead_distance
projection_alpha
projection_extra_clearance
projection_slack_penalty
projection_correction_tolerance
projection_solver_name
run_* metadata fields
```

Paired archives include the checkpoint SHA-256, device, common episode range, environment configuration, projection parameters, and projection mode in `run_*` fields.

### Per-episode static fields

For prefix `episode_XXXX`:

```text
<key>_policy
<key>_checkpoint
<key>_episode
<key>_seed
<key>_goal                         shape (2,)
<key>_obstacle_centers             shape (N, 2)
<key>_obstacle_radii               shape (N,)
<key>_obstacle_mask                shape (N,)
<key>_agent_radius
<key>_dt
<key>_v_max
<key>_omega_max
```

### State fields

```text
<key>_positions                    shape (T + 1, 2)
<key>_headings                     shape (T + 1,)
<key>_state_step_count             shape (T + 1,)
<key>_state_distance_to_goal       shape (T + 1,)
<key>_state_min_obstacle_clearance shape (T + 1,)
<key>_state_success                shape (T + 1,)
<key>_state_collision              shape (T + 1,)
```

### Transition and action fields

```text
<key>_action_raw_normalized        shape (T, 2)
<key>_action_raw_physical          shape (T, 2)
<key>_action_exec_physical         shape (T, 2)
<key>_action_correction_physical   shape (T, 2)
<key>_rewards                      shape (T,)
<key>_terminated                   shape (T,)
<key>_truncated                    shape (T,)
```

`action_raw_normalized` is the policy output before `NormalizedActionWrapper` clipping and scaling. `action_raw_physical` is the bounded physical action received by the projector. `action_exec_physical` is the physical action received by the base environment.

### Projection fields

```text
<key>_projection_enabled                   shape (T,)
<key>_projection_intervened                shape (T,)
<key>_projection_correction_norm           shape (T,)
<key>_projection_slack_values              shape (T, N)
<key>_projection_slack_sum                 shape (T,)
<key>_projection_slack_max                 shape (T,)
<key>_projection_success                   shape (T,)
<key>_projection_solver_status             shape (T,)
<key>_projection_active_constraint_count   shape (T,)
```

When projection is disabled:

```text
projection_enabled = False
projection_intervened = False
projection_correction = [0, 0]
projection_slack_values = 0
projection_slack_sum = 0
projection_slack_max = 0
projection_success = True
projection_solver_status = disabled
```

`projection_enabled` and `projection_solver_status` distinguish this identity path from a successful zero-correction QP solve.

## Evaluator Interfaces

### Single-mode evaluator

Trajectory recording is optional:

```bat
python -m evaluation.evaluate_policy ^
  --policy ppo ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --episodes 2 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --max-obstacles 3 ^
  --num-active-obstacles 3 ^
  --enable-projection ^
  --no-cuda ^
  --output runs\evaluation\ppo_projection_enabled.csv ^
  --trajectory-output runs\evaluation\ppo_projection_enabled_trajectories.npz
```

`--trajectory-output` must use the `.npz` extension. Omitting the argument preserves the previous evaluator behavior and memory footprint.

### Paired evaluator

The paired evaluator always writes one trajectory archive per projection mode:

```text
<output-prefix>_projection_disabled_trajectories.npz
<output-prefix>_projection_enabled_trajectories.npz
```

Both archives use the same episode indices, seeds, checkpoint, environment settings, and projection-parameter metadata as the paired CSV artifacts.

## Minimal Verification Commands

Run from the repository root after activating the `RL_PROJECTS` Conda environment.

### 1. Compile the modified modules and focused tests

```bat
python -m compileall -q ^
  projection\cbf_qp_wrapper.py ^
  evaluation\trajectory_recording.py ^
  evaluation\evaluate_policy.py ^
  evaluation\evaluate_projection_pair.py ^
  tests\test_cbf_qp_wrapper.py ^
  tests\test_evaluate_policy.py ^
  tests\test_evaluate_projection_pair.py
```

Acceptance criterion:

```text
The command exits without syntax errors.
```

### 2. Run the focused trajectory and wrapper checks

```bat
python -m pytest ^
  tests\test_cbf_qp_wrapper.py ^
  tests\test_evaluate_policy.py ^
  tests\test_evaluate_projection_pair.py ^
  -q -rs
```

Acceptance criterion:

```text
7 passed
```

### 3. Run the complete lightweight suite

```bat
python -m pytest -q -rs
```

Acceptance criterion:

```text
19 passed
```

### 4. Run the canonical paired evaluation

```bat
scripts\evaluate_projection_pair.bat
```

Acceptance criterion:

```text
The command writes four CSV files and two NPZ trajectory archives.
```

### 5. Verify archive alignment without pickle

```bat
python -c "import numpy as np; p=r'runs\evaluation\ppo_baseline_51200_seed1_projection_pair'; off=np.load(p+'_projection_disabled_trajectories.npz',allow_pickle=False); on=np.load(p+'_projection_enabled_trajectories.npz',allow_pickle=False); assert off['episode_keys'].tolist()==on['episode_keys'].tolist(); k=off['episode_keys'].tolist()[0]; assert off[k+'_positions'].shape[0]==off[k+'_action_raw_physical'].shape[0]+1; assert on[k+'_positions'].shape[0]==on[k+'_action_exec_physical'].shape[0]+1; print(k, off[k+'_positions'].shape, on[k+'_positions'].shape)"
```

### 6. Verify action identities

```bat
python -c "import numpy as np; p=r'runs\evaluation\ppo_baseline_51200_seed1_projection_pair'; off=np.load(p+'_projection_disabled_trajectories.npz',allow_pickle=False); on=np.load(p+'_projection_enabled_trajectories.npz',allow_pickle=False); k=off['episode_keys'].tolist()[0]; np.testing.assert_allclose(off[k+'_action_raw_physical'],off[k+'_action_exec_physical']); np.testing.assert_allclose(on[k+'_action_exec_physical']-on[k+'_action_raw_physical'],on[k+'_action_correction_physical']); print('action audit passed')"
```

## Review-Environment Validation Record

Validation was performed in the available Linux review environment.

```text
modified-module compilation: passed
focused wrapper/evaluator tests: 7 passed
complete lightweight suite: 19 passed
```

The established PPO checkpoint was evaluated for twenty deterministic paired episodes. Both trajectory archives contained twenty episode records. Every state array had exactly one more row than its corresponding action array. The projection-disabled archive recorded identity actions and `disabled` solver status. The projection-enabled archive preserved the previous zero-intervention deterministic result and recorded successful solver status for every step.

A projection-enabled random-policy diagnostic was also recorded. The archive contained nonzero action corrections, slot-aligned per-obstacle slack values, and solver status for the intervened steps. For every stored transition:

```text
action_exec_physical - action_raw_physical
    == action_correction_physical
```

## Scope Boundary

This issue adds evaluator trajectory auditability only. It does not change:

- PPO action sampling or likelihood calculations;
- normalized-action mapping;
- environment dynamics, reward, or termination;
- CBF geometry, QP objective, or solver configuration;
- stationary solver-failure fallback;
- episode-level metric definitions;
- paired seed alignment;
- training-time projection;
- plotting or trajectory-selection rules.

The trajectory archives remain raw run artifacts under `runs/`. Curated trajectory figures belong to the result-generation workflow and must be generated from saved archives rather than live evaluation.

## Completion Criteria

The trajectory-audit item is complete when all of the following conditions hold:

1. The projection wrapper exposes raw action, executed action, correction vector, and fixed-slot slack values.
2. Single-mode trajectory recording is optional and does not change existing evaluation behavior when omitted.
3. The paired evaluator writes one trajectory archive for each projection mode.
4. Every episode stores `T + 1` state samples and `T` transition samples.
5. Normalized raw, physical raw, physical executed, and physical correction actions remain distinct and aligned.
6. Slot-aligned slack values agree with the summed and maximum slack diagnostics on successful solves.
7. Failed solves preserve stationary executed actions, failure status, and undefined active-obstacle slack values.
8. Paired archives preserve checkpoint SHA-256, episode indices, and common seeds.
9. Archives load with `allow_pickle=False` and contain no object arrays.
10. The focused trajectory checks and complete lightweight suite pass.
