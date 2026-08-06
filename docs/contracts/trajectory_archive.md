# Trajectory Archive Contract

## Scope

`evaluation/trajectory_recording.py` writes compressed NumPy archives with schema:

```text
evaluation_trajectory_v1
```

Archives contain no object arrays and must load with:

```python
np.load(path, allow_pickle=False)
```

## Alignment

For `T` executed transitions:

```text
state arrays       length T + 1
transition arrays  length T
```

Index `t` represents:

```text
state[t] -> raw/executed action[t] -> reward[t] -> state[t + 1]
```

The initial state is recorded immediately after reset.

## Archive-level fields

```text
trajectory_archive_version
episode_count
episode_keys
projection_v_max, projection_omega_max
projection_lookahead_distance, projection_alpha
projection_extra_clearance, projection_slack_penalty
projection_correction_tolerance, projection_solver_name
```

Optional scalar run metadata is prefixed with `run_`.

Episode archive keys are deterministic:

```text
episode_0000, episode_0001, ...
```

## Per-episode static fields

For `<episode>`:

```text
policy, checkpoint, episode, seed
goal
obstacle_centers, obstacle_radii, obstacle_mask
agent_radius, dt, v_max, omega_max
```

## State fields, length T + 1

```text
positions                 shape (T + 1, 2)
headings
state_step_count
state_distance_to_goal
state_min_obstacle_clearance
state_success
state_collision
```

## Transition fields, length T

```text
action_raw_normalized     shape (T, 2)
action_raw_physical       shape (T, 2)
action_exec_physical      shape (T, 2)
action_correction_physical shape (T, 2)
rewards
terminated
truncated
```

Action meanings:

```text
raw normalized   sampled policy output; may lie outside [-1, 1]^2
raw physical     componentwise-clipped normalized-wrapper output
executed physical projector output passed to the environment
correction       executed physical - raw physical
```

## Projection fields, length T

```text
projection_enabled
projection_intervened
projection_correction_norm
projection_slack_values   shape (T, max_obstacles)
projection_slack_sum
projection_slack_max
projection_success
projection_solver_status
projection_active_constraint_count
```

Successful projected transition:

```text
active slack finite and nonnegative
inactive slots zero
slack sum and maximum agree with the slot vector
```

Failed projected transition:

```text
active slack NaN
slack sum NaN
slack maximum NaN
projection_success=False
stationary executed action under the failure contract
```

## Projection-disabled convention

```text
projection_enabled=False
projection_intervened=False
executed action=raw physical action
correction=[0, 0]
slack=0
projection_success=True
projection_solver_status=disabled
active constraint count=0
```

`disabled` distinguishes the identity path from a successful zero-correction QP solve.

## Recorder integrity checks

Before writing, the recorder enforces:

```text
T + 1 versus T alignment
finite actions
raw physical action matches normalized-wrapper output
active constraint count matches obstacle mask
correction vector and norm identities
slack vector/sum/maximum consistency
failure-time unknown slack
duplicate episode/seed rejection
```

## Relationship to CSV

```text
CSV: what happened overall
NPZ: how it happened step by step
```

The companion CSV supplies method, training seed, checkpoint hash, layout identity, evaluation seed, projection mode, action-bound clipping summary, and projection summary. The NPZ supplies exact geometry and transition arrays. Action-bound clipping can be independently reconstructed from `action_raw_normalized` without changing the trajectory schema.

## Downstream use and versioning

The archive supports action auditing, trajectory figures, structured-objective data export, and later offline-dataset generation.

Executed actions must never be reinterpreted as PPO likelihood actions; likelihood semantics remain attached to raw normalized actions.

Any incompatible field, shape, alignment, or semantic change requires a new schema identifier.
