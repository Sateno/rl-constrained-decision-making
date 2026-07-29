# Constrained Policy Learning with Predictive Action Projection

Research code for a compact constrained continuous-control benchmark with a
CleanRL-style PPO baseline, optional predictive action projection, fixed-layout
evaluation, trajectory recording, and reproducible result generation.

## Implemented workflow

- `ConstrainedNavigationEnv` provides fixed-shape observations, physical
  unicycle actions, rewards, collision checks, and controlled reset layouts.
- PPO training supports baseline, high-collision-penalty, and
  projection-enabled interaction variants.
- The CBF-QP projection layer operates in physical action space and records
  intervention, correction, slack, and solver diagnostics.
- Evaluation supports built-in regression scenarios and frozen named layout
  suites, with per-episode CSV summaries and trajectory NPZ archives.
- Result scripts audit saved artifacts before producing aggregate CSV tables,
  LaTeX tables, and PDF figures.

## Canonical validation

Activate the `RL_PROJECTS` Conda environment, then run:

```bat
python -m pytest -q -rs
scripts\validate_evaluation_time_projection.bat
```

Common-layout evaluation and result-generation commands are defined by the
corresponding scripts and protocol files under `evaluation/`, `experiments/`,
and `scripts/`.

## Artifact policy

Raw checkpoints, TensorBoard logs, evaluation CSV files, and trajectory NPZ
archives are written under `runs/` and are not normal source-control content.
Curated paper tables and figures are generated from saved artifacts only.

## Scope

The repository supports simulated numerical experiments. It does not claim
global safety guarantees, real-world deployment readiness, or an unbiased
projected policy-gradient formulation.
