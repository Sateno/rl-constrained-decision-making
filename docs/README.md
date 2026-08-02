# Documentation

This directory contains the active documentation for the predictive action projection study. Documents are organized by purpose rather than by implementation history.

## Directory structure

```text
design/      Authoritative compiled implementation design.
contracts/   Stable software, mathematical, metric, and artifact contracts.
validation/  Current executable validation entry points and PASS-evidence locations.
records/     Historical and research decisions tied to specific revisions and artifacts.
assets/      Small static visual references used by the documentation.
```

## Active documents

### Implementation design

- [Predictive action projection implementation design](design/predictive_action_projection_implementation_design.pdf)

The editable source is maintained in Overleaf. The repository contains the compiled PDF as the authoritative implementation-design artifact.

### Contracts

- [Environment and projection contract](contracts/environment_and_projection.md)
- [Evaluation and artifact contract](contracts/evaluation_and_artifacts.md)
- [Trajectory archive contract](contracts/trajectory_archive.md)

Contracts describe the current repository interfaces and scientific meanings. They must be updated when the corresponding runtime behavior or persisted schema changes.

### Validation

- [Runtime validation](validation/runtime_validation.md)

The Python validators and tests are the executable source of truth. The validation guide identifies the supported entry points, outputs, PASS markers, and remaining manual review without duplicating their internal command sequences.

### Records

- [Historical PPO baseline regression record](records/baseline_regression_record.md)
- [Evaluation-time projection regression record](records/evaluation_time_projection_regression_record.md)
- [Core layout calibration and freeze record](records/core_layout_calibration_record.md)

Records preserve decisions and evidence tied to a particular revision, checkpoint, protocol, or dataset. Accepted records are not rewritten to match later outcomes. A correction requires a new record or an explicit amendment.

### Visual references

- [Original training layout](assets/layouts/original_training_layout.pdf)
- [Frozen core layout suite](assets/layouts/core_navigation_layouts_visual_reference.pdf)

These PDFs document benchmark geometry. They are not final paper-result figures.

## Artifact boundaries

```text
runs/      Raw checkpoints, TensorBoard events, CSV files, NPZ trajectories,
           validation logs, calibration outputs, and other generated evidence.

results/   Curated tables and figures selected for repository-facing analysis
           and publication artifacts.

docs/      Implementation design, contracts, validation guidance, research
           records, and compact visual references.
```

Raw generated evidence is intentionally not copied into this directory. Records identify the relevant paths, hashes, and summaries so that the evidence remains traceable.
