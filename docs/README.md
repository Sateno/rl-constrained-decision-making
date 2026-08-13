# Documentation

This directory contains the active documentation for the predictive action
projection study. Documents are organized by purpose and authority rather than
by implementation history.

## Directory structure

```text
design/      Authoritative compiled implementation design.
guides/      Conceptual and operational guides for reading and using the repository.
contracts/   Stable software, mathematical, metric, and artifact contracts.
validation/  Current executable validation entry points and PASS-evidence locations.
records/     Historical and research decisions tied to specific revisions and artifacts.
assets/      Small static visual references used by the documentation.
```

## Suggested reading order

1. Begin with the
   [Orientation Guide to Predictive Action Projection with PPO](guides/orientation_guide_to_predictive_action_projection_with_ppo.pdf)
   for a concise conceptual introduction.

2. Continue with the
   [Predictive Action Projection Software and Artifact Companion](guides/predictive_action_projection_software_companion.pdf)
   for practical mental models covering code reading, action flow, rollout
   semantics, diagnostics, evaluation, debugging, and artifact interpretation.

3. Consult the
   [Predictive Action Projection Implementation Design](design/predictive_action_projection_implementation_design.pdf)
   for the consolidated mathematical, architectural, algorithmic, artifact,
   and verification specification.

4. Read the
   [Frozen Experimental Protocol](records/predictive_action_projection_experimental_protocol.md)
   before launching or interpreting final training and evaluation.

5. Use the focused Markdown contracts for exact subsystem interfaces, persisted
   schemas, metric meanings, and failure semantics.

The guides are pedagogical and do not supersede the implementation design,
focused contracts, machine-readable protocols, executable validators, or
source code.

## Active documents

### Guides

- [Orientation Guide to Predictive Action Projection with PPO](guides/orientation_guide_to_predictive_action_projection_with_ppo.pdf)
- [Predictive Action Projection Software and Artifact Companion](guides/predictive_action_projection_software_companion.pdf)

The orientation guide provides the shortest conceptual entry into the study,
its motivation, and its main architecture.

The software and artifact companion explains how the design appears during
practical repository work. It covers the complete action path, NumPy and
PyTorch boundaries, Gymnasium same-step autoreset, action-bound and projection
diagnostics, evaluation modes, checkpoints, TensorBoard events, CSV and NPZ
evidence, validation workflows, aggregation, and common interpretation errors.

### Implementation design

- [Predictive Action Projection Implementation Design](design/predictive_action_projection_implementation_design.pdf)

The implementation design is the authoritative consolidated specification for
the implemented predictive action-projection system. It defines the
mathematical model, component ownership, software architecture, algorithms,
artifact contracts, failure semantics, validation structure, and
change-control boundaries.

The editable source is maintained in Overleaf. The repository contains the
compiled PDF as the authoritative implementation-design artifact.

### Frozen experimental protocol

- [Predictive Action Projection Final Experimental Protocol](records/predictive_action_projection_experimental_protocol.md)
- [Primary fixed-training-geometry analysis protocol](../experiments/fixed_training_geometry_analysis_protocol.json)
- [Secondary core-layout transfer analysis protocol](../experiments/projection_analysis_protocol.json)
- [Fixed training-geometry serialization](../evaluation/layouts/fixed_training_geometry.json)

The protocol freezes the methods, independently initialized runs, transition
budget, selected device, evaluation matrix, episode and layout seeds,
diagnostics, aggregation hierarchy, rerun rules, and claim boundaries before
final outcomes are observed.

### Contracts

- [Environment and projection contract](contracts/environment_and_projection.md)
- [Evaluation and artifact contract](contracts/evaluation_and_artifacts.md)
- [Trajectory archive contract](contracts/trajectory_archive.md)

Contracts describe exact repository interfaces and scientific meanings for
focused subsystems. They must be updated when the corresponding runtime
behavior, persisted schema, metric interpretation, or failure semantics
change.

### Validation

- [Runtime validation](validation/runtime_validation.md)

The Python validators and tests are the executable source of truth for
validation behavior. The validation guide identifies the supported entry
points, outputs, PASS markers, and required manual review without duplicating
their internal command sequences.

### Records

- [Historical PPO baseline regression record](records/baseline_regression_record.md)
- [Evaluation-time projection regression record](records/evaluation_time_projection_regression_record.md)
- [Core layout calibration and freeze record](records/core_layout_calibration_record.md)
- [Predictive action projection final experimental protocol](records/predictive_action_projection_experimental_protocol.md)

Records preserve decisions and evidence tied to a particular source revision,
checkpoint, protocol, benchmark, or dataset. Accepted records are not silently
rewritten to match later outcomes. A correction requires a new record or an
explicit amendment.

### Curated calibration evidence

- [Core navigation layout calibration evidence](../results/calibration/core_navigation_layouts_v1/README.md)

The calibration evidence directory contains the compact family table, the
layout-level paired comparison, and the representative trajectory figure. The
complete raw calibration bundle remains outside normal Git under `runs/`.

### Visual references

- [Original training layout](assets/layouts/original_training_layout.pdf)
- [Frozen core layout suite](assets/layouts/core_navigation_layouts_visual_reference.pdf)

These PDFs document benchmark geometry. They are not final paper-result
figures.

## Documentation authority

The documentation layers have different responsibilities:

```text
orientation guide     rapid conceptual introduction
software companion    practical repository and artifact understanding
implementation design consolidated technical specification
focused contracts     exact subsystem interfaces and schemas
protocols             machine-readable experimental requirements
validators and tests  executable verification behavior
source code            final authority for actual runtime behavior
records                revision-specific historical evidence and decisions
```

A disagreement among these layers is a documentation or implementation defect
that must be resolved explicitly. Historical records are not rewritten merely
because the implementation later changes.

## Artifact boundaries

```text
runs/      Raw checkpoints, TensorBoard events, CSV files, NPZ trajectories,
           validation logs, calibration outputs, benchmark outputs, and other
           generated evidence.

results/   Curated tables and figures selected for repository-facing analysis
           and publication artifacts.

docs/      Implementation design, conceptual and operational guides, focused
           contracts, validation guidance, research records, and compact
           visual references.
```

Raw generated evidence is intentionally not copied into this directory.
Records identify the relevant paths, hashes, configurations, and summaries so
that the evidence remains traceable without treating generated runtime output
as ordinary documentation source.
