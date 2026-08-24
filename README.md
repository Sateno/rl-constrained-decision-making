# Constrained Policy Learning with Predictive Action Projection

Research code and audited simulation results for a compact constrained
continuous-control benchmark using a CleanRL-style PPO baseline and an optional
predictive action-projection safety filter.

> **Study status, 24 August 2026:** Training, frozen evaluation, dataset audit,
> and result generation are complete. Scientific interpretation and a short
> empirical technical report are in preparation. The numerical snapshot below
> is deliberately preliminary and does not represent the final conclusions of
> the study.

## Evaluation snapshot

The completed experiment contains:

- three PPO training conditions;
- five independently trained checkpoints per condition;
- 3,000 stochastic fixed-geometry evaluation episodes;
- 720 deterministic episodes across a 24-layout transfer suite; and
- 3,720 evaluated episodes in total, with projection disabled and enabled under
  frozen protocols.

The table reports projection-on minus projection-off paired differences as
mean plus or minus sample standard deviation across five independently trained
checkpoints. Values are percentage-point changes. A negative collision value
means that enabling projection reduced collision rate.

| Training condition | Fixed Δ success | Fixed Δ collision | Transfer Δ success | Transfer Δ collision |
| --- | ---: | ---: | ---: | ---: |
| PPO baseline | +2.6 ± 3.8 pp | -13.8 ± 7.3 pp | 0.0 ± 0.0 pp | -1.7 ± 2.3 pp |
| PPO high penalty | +0.2 ± 0.4 pp | -12.8 ± 5.8 pp | 0.0 ± 0.0 pp | -1.7 ± 3.7 pp |
| PPO trained with projection | +76.2 ± 4.5 pp | -80.4 ± 6.1 pp | +5.8 ± 2.3 pp | -50.8 ± 9.5 pp |

For checkpoints trained with projection, fixed-geometry success increased from
17.4% to 93.6% when projection was enabled, while collision rate decreased
from 81.8% to 1.4%. The projector intervened on 41.4% of executed steps. In the
transfer suite, success increased from 27.5% to 33.3%, collision rate decreased
from 63.3% to 12.5%, and intervention increased to 56.7%. A substantial share
of avoided transfer collisions became timeouts rather than successes.

These observations describe the complete policy-plus-projector controller.
They do not establish that the nominal policy is independently safe, that the
method provides a global safety guarantee, or that the present interpretation
is final. No projection solver failure was recorded in the frozen evaluation
dataset.

## Results and audit trail

The committed result set is intentionally curated. Raw checkpoints,
TensorBoard logs, and trajectory archives remain outside normal source control.

- [Fixed-geometry tables](results/tables/fixed_training_geometry/)
- [Fixed-geometry figures](results/figures/fixed_training_geometry/)
- [Transfer tables](results/tables/core_layout_transfer/)
- [Transfer figures](results/figures/core_layout_transfer/)
- [Analysis record](docs/records/Predictive_Action_Projection_Analysis_Record.md)
- [Exact analysis command record](docs/records/Predictive_Action_Projection_Analysis_Command_Record.md)

Each result suite includes machine-readable build audits. The final committed
set contains 23 primary PDFs and 13 evaluation-only transfer PDFs. All figures
passed direct visual review, metadata and embedded-font checks, and numerical
reconciliation against their committed tables and underlying evidence. The
evaluation tables contain 3,720 episode rows, and the complete repository test
suite passed with 71 tests.

## Start here

Readers new to the study should begin with the
[Orientation Guide to Predictive Action Projection with PPO](docs/guides/Orientation_Guide_to_Predictive_Action_Projection_with_PPO.pdf).
It provides a concise conceptual introduction to the research question,
architecture, and recommended reading path.

Next, read the
[Predictive Action Projection Software and Artifact Companion](docs/guides/Predictive_Action_Projection_Software_Companion.pdf).
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

The exact commands used for the frozen evaluation, aggregation, figure builds,
audits, and final quality gates are preserved in the
[analysis command record](docs/records/Predictive_Action_Projection_Analysis_Command_Record.md).

## Artifact policy

Raw checkpoints, TensorBoard logs, evaluation CSV files, and trajectory NPZ
archives are written under `runs/` and are not normal source-control content.
Curated result tables and figures are generated from saved artifacts only.

## Attribution and licensing

The adapted continuous-action PPO training script is derived from
[CleanRL](https://github.com/vwxyzjn/cleanrl). Its provenance and retained MIT
notice are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Original project source code, machine-readable protocols and configurations,
and executable scripts are available under the MIT terms in
[LICENSE](LICENSE). Generated figures, result tables, datasets, and
documentation are not covered by that source-code license unless a file states
otherwise; they remain copyright Salvador Tenorio and are currently provided
for scholarly inspection.

## Scope

This repository supports simulated numerical experiments. It does not claim
global safety guarantees, real-world deployment readiness, or an unbiased
projected policy-gradient formulation.
