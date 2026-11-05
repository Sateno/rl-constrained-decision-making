# Recorded evaluation trajectories: PAP evaluation v1

This frozen dataset makes the completed study playable from a fresh clone.
It contains **60 original NumPy archives, 3,720 evaluation episodes, and 1,860
matched projector-OFF/projector-ON pairs**. It is evaluation evidence, not
training trajectories or newly generated rollouts.

| Evaluation suite | Archives | Episodes | OFF/ON pairs | Action selection |
| --- | ---: | ---: | ---: | --- |
| `fixed_training_geometry` | 30 | 3,000 | 1,500 | Sampled actions |
| `core_layout_transfer` | 30 | 720 | 360 | Actor-mean actions |

Each suite covers three training methods and five independent training seeds,
with both projection modes. The archives occupy 64,335,285 bytes (about 61.4
MiB); the largest archive is about 2 MiB. They are ordinary Git-tracked binary
files, so a clone needs neither Git LFS nor a separate data download.

## Playback

From the repository root:

```sh
python -m pip install -r visualization/trajectory_viewer/requirements.txt
python visualization/trajectory_viewer/viewer.py
```

The explorer selects this dataset automatically when its manifest is present.
No `runs/` directory, model checkpoint, policy execution, QP solve, or training
is needed. Use `--data-source published` to require this dataset explicitly;
use `--data-source local` to select preserved local archives under
`runs/evaluation/final/`. The explorer does not mix the two sources or replace
a missing published file with a local file.

See the [explorer guide](../../../visualization/trajectory_viewer/README.md)
for controls and video export. Browser exports remain in temporary storage
until downloaded; they are not written into this dataset.

## Provenance and interpretation

Every NPZ is a byte-for-byte copy from
`rl-constrained-decision-making(20260907-183013).zip`. The source ZIP and all 60
NPZ hashes matched the pre-existing `input_archives` and `local_archive_files`
records in the repository's [release manifest](../../../verification/repository_manifest.json).
The source repository revision used for this packaging is
`65a312db670e82d2a186bb1f7afb83d64c2484c3`.

[manifest.json](manifest.json) records the original `runs/` path, SHA-256,
byte size and episode count of every archive, together with the hashes of
the corresponding published episode tables. NPZ hashes are exact byte hashes;
table hashes normalize CRLF to LF for cross-platform checkouts. Embedded
historical path strings are preserved as provenance, not rewritten as current
filesystem locations.

**These are the original recorded outcomes under endpoint collision detection.**
This packaging does not apply swept-segment collision corrections, truncate
historical trajectories, change success/collision/timeout labels, or regenerate
the paper's tables. Clear sampled endpoints do not establish that the intervening
segment avoided an obstacle. The viewer displays what was recorded; it does not
certify physical safety. Any corrected evidence should be identified separately
and must not silently replace this dataset.

## Verification and maintenance

```sh
python -m evaluation.verify_trajectory_bundle
```

This read-only check verifies archive inventory and hashes, original provenance,
published-table correspondence, episode coverage, and OFF/ON pairing. It performs
no training or policy evaluation. The normal repository release check includes
this validation as well:

```sh
python -m evaluation.verify_repository_release --base-ref main
```

Treat this directory as frozen input. Training and evaluation tools continue to
write under ignored `runs/`, not here. New studies need separately named datasets
and matching tables. Do not edit a hash to conceal altered evidence. The existing
[license](../../../LICENSE) distinguishes source code from datasets; this addition
does not change that license scope.
