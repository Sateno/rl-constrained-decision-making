# Core Navigation Layout Calibration Evidence

This directory contains the compact repository-facing evidence used to accept
and freeze the 24-layout core navigation suite. These artifacts describe one
historical calibration checkpoint and are **not final multi-seed comparative
results**.

## Frozen identities

```text
layout_suite=evaluation/layouts/core_navigation_layouts.json
suite_id=core_navigation_layouts_v1
canonical_suite_sha256=1027141797052240e83b941398e5a32031e9ca67d001e4a0cf1e19b0f96dd466
calibration_checkpoint_sha256=3c06bd19ee42914aef49f049de88c165190f745ca1c4cdbb3ac23bb7497da1c3
calibration_source_commit=20eacdee87ddd22b91af7de73c2f81fb3a04618f
```

The accepted scientific judgment and freeze declaration are recorded in
[`docs/records/core_layout_calibration_record.md`](../../../docs/records/core_layout_calibration_record.md).

## Curated files

- `calibration_by_family.csv` summarizes outcomes and projection burden for
  the control, single-obstacle, two-obstacle, and three-obstacle families.
- `calibration_layout_comparison.csv` preserves the paired projection-off and
  projection-on result for each of the 24 layouts.
- `calibration_representative_trajectories.pdf` shows four predeclared
  representative layouts used in the manual geometric review.

## Interpretation boundary

The suite was accepted unchanged as a **secondary deterministic transfer and
robustness benchmark**. The historical probe checkpoint exposed strong
specialization to the fixed training geometry; it is not a final experimental
replicate.

## Raw evidence

The complete generated evidence remains outside normal Git at:

```text
runs/calibration/core_navigation_layouts_v1/
```

That directory includes the two raw episode CSV files, both trajectory NPZ
archives, automated audit outputs, summaries, and the generated figure. The
checkpoint used for calibration remains at:

```text
runs/checkpoints/ppo_baseline_51200_seed1.pt
```

Both must be preserved in a durable external archive and must not be
overwritten.
