# Evaluation and Artifact Contract

## Scope

This document defines stable evaluation, metric, pairing, aggregation, and result-generation behavior implemented under `evaluation/`, `analysis/`, and `experiments/`.

Trajectory serialization is defined in [Trajectory Archive Contract](trajectory_archive.md).

## Evaluation rule

Evaluation loads fixed parameters, runs complete episodes, writes metrics and optional trajectories, and performs no learning.

Action modes:

```text
deterministic actor mean   primary reproducible tables
stochastic sample          robustness and active-projection diagnostics
random action              smoke validation
```

## Episode CSV schema

One row per episode contains:

```text
policy, checkpoint, episode, seed
episode_return, episode_length
success, collision, terminated, truncated
final_distance_to_goal, min_obstacle_clearance
projection_enabled
projection_intervention_count, projection_intervention_rate
mean_projection_correction_norm, max_projection_correction_norm
mean_projection_slack_sum, max_projection_slack
projection_solver_failure_count
```

Every row also records:

```text
projection_lookahead_distance
projection_alpha
projection_slack_penalty
projection_extra_clearance
```

Projection-disabled rows keep the same schema with `projection_enabled=False` and zero projection placeholders.

## Metric meanings

- `episode_return`: sum of rewards.
- `episode_length`: executed transitions.
- `min_obstacle_clearance`: minimum signed collision-boundary clearance during the episode; `NaN` when no obstacle is active.
- intervention rate: intervention count divided by complete episode length.
- mean correction: mean over all transitions, including zero-correction steps.
- mean summed slack: mean per-step sum of active slack, including zero-slack steps.
- maximum slack: largest individual slack observed.

Solver failure:

```text
failure count increments
mean slack=NaN
maximum slack=NaN
```

A valid zero-slack solve remains numeric zero. It is not a failure sentinel.

## Common-layout evaluation

`evaluation.evaluate_layout_suite` adds:

```text
method, train_seed
training_collision_penalty, training_projection_enabled
checkpoint_sha256
layout_suite_schema_version, layout_suite_id, layout_suite_sha256
layout_id, layout_repeat, evaluation_seed
evaluation_collision_penalty
projection_mode
```

Canonical pairing key:

```text
layout_id, layout_repeat, evaluation_seed
```

Layout-suite identity is computed from canonical parsed JSON, independent of LF/CRLF line endings.

Existing CSV or NPZ output is refused unless `--overwrite` is explicit.

The development suite validates software and pipelines. The frozen core suite measures deterministic obstacle-layout transfer and robustness.

## Same-checkpoint paired evaluation

`evaluation.evaluate_projection_pair` runs one checkpoint with projection disabled and enabled under one shared configuration and aligned seeds.

For prefix `<prefix>` it writes:

```text
<prefix>_projection_disabled.csv
<prefix>_projection_enabled.csv
<prefix>_projection_disabled_trajectories.npz
<prefix>_projection_enabled_trajectories.npz
<prefix>_paired_episodes.csv
<prefix>_paired_summary.csv
```

The checkpoint is hashed before and after evaluation. Paired tables contain enabled-minus-disabled deltas.

For stochastic evaluation, both modes begin each episode with the same random state. Trajectories may diverge after projection changes an executed action.

## Interpretation rules

- Zero deterministic intervention is valid noninterference when raw actions are feasible.
- Stochastic paired evaluation verifies active corrections, finite burden metrics, and solver stability; it is not automatically a final method-level result.
- A local filter can prevent collision without guaranteeing route completion.

## Analysis protocol

JSON protocols under `experiments/` declare:

```text
layout suite
expected training seeds
required method/projection groups
representative layout
whether complete training and trajectory evidence is mandatory
```

They do not hardcode generated checkpoint, CSV, NPZ, TensorBoard, timestamped-directory, or machine-specific filenames.

The result builder discovers artifacts from schema and internal metadata, then rejects:

```text
missing methods, seeds, modes, or layouts
duplicate episode keys
mismatched checkpoint or layout identities
different evaluation key sets
non-finite returns or invalid lengths
invalid projection diagnostics
solver failures in accepted final evidence
```

Final protocols require a training-return curve and representative trajectory for every required checkpoint/mode. Projection-specific training curves remain optional.

## Aggregation hierarchy

Layouts are repeated test cases for one checkpoint, not independent trained-policy replicates.

\[
\bar y_{m,s}=\frac{1}{L}\sum_l y_{m,s,l},
\qquad
\bar y_m=\frac{1}{S}\sum_s \bar y_{m,s}.
\]

Thus:

```text
average layouts within checkpoint
then aggregate independently trained checkpoints across seeds
```

Projection off/on differences are paired by checkpoint and layout before seed-level aggregation.

With three seeds, report individual seeds, mean, standard deviation, and effect magnitude. Strong significance claims are not justified by layout count alone.

## Generated outputs

Curated tables include:

```text
evaluation_episode_results.csv
checkpoint_summary.csv
method_summary.csv
paired_projection_deltas.csv
paired_projection_summary.csv
generated_method_summary.tex
generated_paired_projection_deltas.tex
result_build_audit.json
```

Figures may include training return, task/safety comparisons, projection burden, and predeclared trajectories.

Plotting consumes saved artifacts only and never launches training or evaluation. The builder refuses an existing output directory to prevent stale figures.

## Artifact locations

```text
runs/checkpoints/     raw checkpoints
runs/evaluation/      raw CSV and NPZ evidence
runs/validation/      validation logs and smoke artifacts
runs/calibration/     full calibration evidence
results/tables/       curated final tables
results/figures/      curated final figures
docs/records/         concise decisions and provenance
```

## Change-control boundary

Contract review is required for changes to episode-field meaning, parameter metadata, failure-time `NaN` semantics, layout identity, pairing keys, aggregation hierarchy, or required final evidence.
