# Paired Projection Evaluation Verification

## Purpose

This document defines and verifies the one-command evaluation path for comparing one PPO checkpoint with predictive action projection disabled and enabled.

The implementation files are:

```text
evaluation/evaluate_projection_pair.py
scripts/evaluate_projection_pair.bat
```

The focused smoke-test file is:

```text
tests/test_evaluate_projection_pair.py
```

The existing single-mode evaluator remains:

```text
evaluation/evaluate_policy.py
```

## Problem Addressed

The repository previously required two independent evaluator commands to compare projection-disabled and projection-enabled behavior. A manual comparison could accidentally change the checkpoint, episode seeds, environment configuration, policy evaluation mode, or projection parameters between commands.

Evaluation-time predictive action projection requires the same trained policy to be evaluated in both modes under a common protocol. The paired path therefore owns the comparison orchestration while reusing the existing episode runner, checkpoint loader, projection metrics, CSV writer, and summary functions.

## Paired Evaluation Contract

One command now constructs a single comparison configuration containing:

```text
checkpoint path and SHA-256
number of episodes
base episode seed
maximum episode length
obstacle capacity
active obstacle count
deterministic or stochastic policy mode
device
projection lookahead distance
projection alpha
projection slack penalty
projection extra clearance
```

Two environment factories are then created from the same configuration. They differ only in:

```text
enable_projection=False
enable_projection=True
```

The PPO checkpoint is loaded once. Checkpoint compatibility is verified against both factories. The same action provider is used for both modes.

For paired episode `i`, both modes use:

```text
episode index = i
episode seed  = base_seed + i
```

The environment reset path, environment action space, and PyTorch random generator are seeded identically before each mode runs that episode.

Deterministic evaluation remains the default. Under stochastic evaluation, both modes begin each episode from the same random-generator state. State trajectories may diverge after projection changes an executed action, so later stochastic policy samples are not required to remain numerically identical.

## Output Artifacts

For an output prefix such as:

```text
runs\evaluation\ppo_baseline_51200_seed1_projection_pair
```

the command writes four CSV files and two compressed NPZ trajectory archives.

### Projection-disabled raw episode rows

```text
ppo_baseline_51200_seed1_projection_pair_projection_disabled.csv
```

This file uses the existing fixed `EpisodeResult` schema and records `projection_enabled=False`.

### Projection-enabled raw episode rows

```text
ppo_baseline_51200_seed1_projection_pair_projection_enabled.csv
```

This file uses the same schema and records the intervention, correction, slack, and solver-failure diagnostics produced with `projection_enabled=True`.


### Projection-disabled trajectory archive

```text
ppo_baseline_51200_seed1_projection_pair_projection_disabled_trajectories.npz
```

This archive records the initial state and every transition for each projection-disabled episode. The physical executed action equals the physical raw action, the correction is zero, and the per-step solver status is `disabled`.

### Projection-enabled trajectory archive

```text
ppo_baseline_51200_seed1_projection_pair_projection_enabled_trajectories.npz
```

This archive records normalized raw actions, physical raw actions, physical executed actions, correction vectors, positions, headings, geometry, reward/termination fields, slot-aligned slack values, intervention flags, solver status, and success diagnostics.

Both archives use the `evaluation_trajectory_v1` contract defined in:

```text
docs/trajectory_audit_verification.md
```

### Wide paired episode table

```text
ppo_baseline_51200_seed1_projection_pair_paired_episodes.csv
```

This file contains one row per `(episode, seed)` pair. It records:

- the shared checkpoint path and SHA-256;
- the complete shared evaluation configuration;
- task metrics without projection;
- task metrics with projection;
- projection-burden metrics from the enabled mode;
- enabled-minus-disabled deltas for return, episode length, success, collision, final goal distance, and minimum obstacle clearance.

The writer rejects duplicate episode keys, different key sets, different policy labels, different checkpoint paths, or incorrect projection-mode flags.

### Wide paired summary table

```text
ppo_baseline_51200_seed1_projection_pair_paired_summary.csv
```

This file contains one comparison row. It records:

- shared run metadata;
- projection-disabled aggregate task metrics;
- projection-enabled aggregate task metrics;
- projection-enabled burden and solver diagnostics;
- enabled-minus-disabled aggregate deltas;
- paths to all six comparison artifacts.

The checkpoint SHA-256 is computed before loading and verified again after both modes finish. This identifies the checkpoint contents even if a path is later reused and rejects a checkpoint file that changes during evaluation.

## Command-Line Interface

The Python entry point is:

```bat
python -m evaluation.evaluate_projection_pair ^
  --checkpoint runs\checkpoints\ppo_baseline_51200_seed1.pt ^
  --episodes 20 ^
  --seed 1000 ^
  --max-episode-steps 200 ^
  --max-obstacles 3 ^
  --num-active-obstacles 3 ^
  --projection-lookahead-distance 0.25 ^
  --projection-alpha 2.0 ^
  --projection-slack-penalty 1000.0 ^
  --projection-extra-clearance 0.0 ^
  --no-cuda ^
  --output-prefix runs\evaluation\ppo_baseline_51200_seed1_projection_pair
```

The canonical repository wrapper is:

```bat
scripts\evaluate_projection_pair.bat
```

The batch file uses the established PPO baseline checkpoint, twenty evaluation episodes, seeds `1000` through `1019`, the default three-obstacle environment, explicit projection parameters, and CPU evaluation.

## Minimal Verification Commands

Run the following commands from the repository root after activating the `RL_PROJECTS` Conda environment.

### 1. Compile the paired evaluator and focused smoke test

```bat
python -m compileall -q ^
  evaluation\evaluate_projection_pair.py ^
  tests\test_evaluate_projection_pair.py
```

Acceptance criterion:

```text
The command exits without syntax errors.
```

### 2. Verify command-line exposure

```bat
python -m evaluation.evaluate_projection_pair --help
```

Acceptance criterion:

```text
The help output includes the checkpoint, episode, seed, environment, stochastic-policy, device, projection-parameter, and output-prefix arguments.
```

### 3. Run the focused paired-evaluation smoke test

```bat
python -m pytest tests\test_evaluate_projection_pair.py -q -rs
```

Acceptance criterion:

```text
1 passed
```

The smoke test creates a small temporary PPO checkpoint and runs two one-step, zero-obstacle episodes in both modes. It verifies:

```text
matching episode indices and seeds
projection-disabled and projection-enabled flags
identical noninterference results
persisted projection parameters
checkpoint SHA-256
paired episode deltas
paired summary deltas
state/action array alignment
raw/executed action identity in the disabled mode
projection diagnostics in the enabled mode
all six output artifacts
```

### 4. Run the complete lightweight suite

```bat
python -m pytest -q -rs
```

Acceptance criterion:

```text
19 passed
```

### 5. Run the canonical paired checkpoint evaluation

```bat
scripts\evaluate_projection_pair.bat
```

Acceptance criterion:

```text
The command writes all four CSV artifacts and both NPZ trajectory archives, then exits successfully.
```

### 6. Verify episode alignment and the comparison row

```bat
python -c "import pandas as pd; p=r'runs\evaluation\ppo_baseline_51200_seed1_projection_pair'; off=pd.read_csv(p+'_projection_disabled.csv'); on=pd.read_csv(p+'_projection_enabled.csv'); pair=pd.read_csv(p+'_paired_episodes.csv'); summary=pd.read_csv(p+'_paired_summary.csv').iloc[0]; assert off[['episode','seed']].equals(on[['episode','seed']]); assert pair[['episode','seed']].equals(off[['episode','seed']]); assert not off['projection_enabled'].any(); assert on['projection_enabled'].all(); assert summary['without_projection_episodes']==summary['with_projection_episodes']==20; print(summary.to_string())"
```

## Review-Environment Validation Record

Validation was performed in the available Linux review environment.

### Compilation and tests

```text
paired evaluator compilation: passed
focused paired-evaluation test: 1 passed
complete lightweight suite: 19 passed
```

### Deterministic checkpoint comparison

The established PPO checkpoint was evaluated for twenty episodes with seeds `1000` through `1019`.

| Metric | Without projection | With projection | Enabled minus disabled |
|---|---:|---:|---:|
| Mean return | `12.319941656945442` | `12.319941656945442` | `0.0` |
| Mean episode length | `96.0` | `96.0` | `0.0` |
| Success rate | `1.0` | `1.0` | `0.0` |
| Collision rate | `0.0` | `0.0` | `0.0` |
| Mean minimum obstacle clearance | `0.5181634403654055` | `0.5181634403654055` | `0.0` |

Projection-enabled diagnostics were:

```text
total interventions = 0
mean intervention rate = 0.0
mean correction norm = approximately 1.49e-17
mean slack sum = 0.0
maximum slack = 0.0
solver failures = 0
```

The paired episode table confirmed exact equality of all twenty deterministic episode returns.

### Stochastic checkpoint diagnostic

The same checkpoint and seeds were also evaluated with `--stochastic`.

| Metric | Without projection | With projection | Enabled minus disabled |
|---|---:|---:|---:|
| Mean return | `-0.6116337311344641` | `2.766760867066224` | `3.3783945982006878` |
| Mean episode length | `124.05` | `150.95` | `26.9` |
| Success rate | `0.45` | `0.60` | `0.15` |
| Collision rate | `0.30` | `0.00` | `-0.30` |
| Mean minimum obstacle clearance | `0.1557292229048058` | `0.2297519097817625` | `0.0740226868769566` |

Projection-enabled diagnostics were:

```text
total interventions = 201
mean intervention rate = 0.0630311808856135
mean correction norm = 0.0264355345440324
maximum correction norm = 1.7368978363488647
mean slack sum = 7.142547827530943e-05
maximum slack = 0.0168056268301485
solver failures = 0
```

This stochastic run is a projection-path diagnostic, not a final comparative experiment result.

## Trajectory Audit Contract

The action/state trajectory schema, array shapes, disabled-mode conventions, projection-vector diagnostics, and `allow_pickle=False` loading requirement are defined in:

```text
docs/trajectory_audit_verification.md
```

## Scope Boundary

This issue adds paired evaluation orchestration and paired artifacts only. It does not change:

- environment dynamics, reward, collision geometry, or obstacle encoding;
- the PPO agent, checkpoint format, or likelihood semantics;
- normalized or physical action mappings;
- CBF geometry, QP equations, solver fallback, or wrapper order;
- single-mode episode metrics or summary semantics;
- projection-enabled PPO training;
- comparative experiment execution;
- final result aggregation or plotting.

## Completion Criteria

The paired-evaluation issue is complete when all of the following conditions hold:

1. One command evaluates one checkpoint with projection disabled and enabled.
2. Both modes use the same episode indices and seeds.
3. Both modes use the same environment and projection configuration.
4. The two environment factories differ only in projection enablement.
5. The checkpoint is loaded once, identified by path and SHA-256, and verified unchanged after both modes finish.
6. Checkpoint compatibility is verified for both environment factories.
7. Two raw fixed-schema episode CSVs are preserved.
8. Two `evaluation_trajectory_v1` NPZ archives preserve state, raw action, executed action, correction, slack, and solver diagnostics.
9. One wide paired episode table is written.
10. One wide paired summary table is written and links all six artifacts.
11. Pairing errors are rejected rather than silently merged.
12. The deterministic baseline noninterference result is preserved.
13. The focused smoke test and complete lightweight suite pass.
