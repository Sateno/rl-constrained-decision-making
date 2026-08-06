# Constrained Policy Learning with Predictive Action Projection

Research code for a compact constrained continuous-control benchmark with a
CleanRL-style PPO baseline, optional predictive action projection, fixed-layout
evaluation, trajectory recording, and reproducible result generation.

## Start here

Readers new to the study should begin with the
[Orientation Guide to Predictive Action Projection with PPO](docs/guides/orientation_guide_to_predictive_action_projection_with_ppo.pdf).
It provides a concise conceptual introduction to the research question,
architecture, and recommended reading path.

Next, read the
[Predictive Action Projection Software and Artifact Companion](docs/guides/predictive_action_projection_software_companion.pdf).
It explains how the design appears in the repository during practical code
reading, training, evaluation, debugging, artifact inspection, and result
interpretation.

For exact implemented behavior, mathematical definitions, component ownership,
interfaces, algorithms, artifact schemas, failure semantics, and verification
contracts, consult the
[Predictive Action Projection Implementation Design](docs/design/predictive_action_projection_implementation_design.pdf).

The guides are pedagogical. When exact field names, array shapes, formulas,
protocol requirements, or runtime behavior matter, consult the implementation
design, the focused contracts under `docs/contracts/`, the machine-readable
protocols, the executable validators, and the source code.
## Implemented workflow

- `ConstrainedNavigationEnv` provides fixed-shape observations, physical
  unicycle actions, rewards, collision checks, and controlled reset layouts.
- PPO training supports baseline, high-collision-penalty, and
  projection-enabled interaction variants.
- The action path records normalized action-bound clipping separately from
  physical CBF-QP intervention, correction, slack, and solver diagnostics.
- Evaluation supports built-in regression scenarios and frozen named layout
  suites, with per-episode CSV summaries and trajectory NPZ archives.
- Result scripts audit saved artifacts and export episode-level training
  safety, rollout-level intervention burden, aggregate CSV and LaTeX tables,
  and PDF figures.

## Canonical validation

Activate the `RL_PROJECTS` Conda environment, then run the consolidated
pre-experiment validation:

```bat
scripts\validate_pre_experiment_codebase.bat
```

After the consolidated validation passes, the engineering-only CPU/CUDA
throughput comparison is run with:

```bat
scripts\benchmark_training_devices.bat
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
