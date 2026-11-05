# Verify the completed repository

With `RL_PROJECTS` active, run from the repository root after staging the
intended source, documentation, and curated results:

```sh
python -m pytest -q -rs
python -m evaluation.verify_repository_release --base-ref main --require-tracked
```

Both must exit with code 0. The second finishes with `REPOSITORY_RELEASE PASS`.
These commands work with `runs/` entirely absent. They do not launch final
training, policy evaluation, calibration, or device benchmarking. Tests use
small temporary fixtures to check software behavior.

The root pytest command includes the trajectory explorer's archive-loader and
local-server tests. These use synthetic recordings and mocked video rendering;
they require neither the preserved run archives nor FFmpeg.

All `runs/` files remain excluded from Git and preserved in local archives.
The 60 final-evaluation trajectory archives also have byte-identical published
copies under `results/trajectories/pap_evaluation_v1/`. Those versioned copies
are required repository files, so playback and trajectory verification work
without restoring `runs/`.
If the previous package caused them to be staged, unstage `runs/` in your editor.
Changing ignore rules alone does not remove an already staged file. Preserve
all local files; `runs/` is not part of the filesystem deletion list.

The normal verifier checks:

- the manifest-defined repository inventory and audited content, including source,
  protocols, environment specification, curated results, and documentation;
- the three externally compiled manual PDFs and exact staged/working bytes;
- absence of tracked `runs/` files and retired source/production trees;
- protected runtime and protocols against the local `main` reference;
- all 3,000 fixed and 720 transfer rows, design coverage, recorded model
  identities, OFF/ON pairing, and reconstruction of the suite summary tables;
- the published trajectory bundle's exact hashes, provenance, complete episode
  coverage, and agreement between recorded arrays and the endpoint tables;
- active Markdown links and the exact staged file inventory/content.

Before staging, omit `--require-tracked` to inspect the working files. Tracked
run files are rejected in either mode. The base comparison uses the local Git
reference and does not query the live remote.

## Published trajectory bundle only

```sh
python -m evaluation.verify_trajectory_bundle
```

This reads the manifest and all 60 NPZ files under
`results/trajectories/pap_evaluation_v1/`, checks their hashes and correspondence
with the 3,720 published episode rows, and verifies the saved trajectory data.
It performs no training, policy evaluation, or QP solve. The default repository
verifier also runs this check when the release manifest declares the bundle.

The bundle preserves the original endpoint-collision contract and labels. Its
verification establishes consistency with those recorded results, not physical
safety or a later swept-collision reanalysis. Do not regenerate data or hashes
to make a failed check pass; locate the changed or missing artifact first.

## Optional local archive verification

The manifest separately retains `local_archive_files`, a checksum inventory of
181 archived raw/provenance files. This is not a list of required Git files.
To verify an archive, specify the directory containing its preserved `runs/`
folder. If the ignored original files remain in this working directory, use:

```sh
python -m evaluation.verify_repository_release --base-ref main --archive-root .
```

For an archive stored elsewhere, replace `.` with its parent directory, such as
`"C:\archives\PAP"` when the files are under `C:\archives\PAP\runs\`.

This additionally verifies all archive checksums, the 60 final raw CSVs and
3,720 trajectories against the repository endpoint tables, all 15 final model
hashes/configurations/finite parameters, the TensorBoard scalar/episode/rollout
exports, and recorded solver exposure. It prints `LOCAL_ARCHIVE PASS` followed
by the repository result. It never stages, deletes, or changes the archive.
Omitting this option still verifies the committed trajectory bundle, but makes
no claim to have checked checkpoint files, raw CSVs, TensorBoard events or other
raw artifacts absent from a public checkout.

## Optional separate paper comparison

```sh
python -m evaluation.verify_repository_release --base-ref main --paper-project "../Runtime_Attribution_in_PPO_with_Predictive_Action_Projection.zip"
```

This reads the six shared scientific inputs in the supplied Overleaf ZIP without
copying the manuscript or paper-production assets into the code repository.

## Integrity boundary

Scientific binaries use exact hashes; declared text allows only CRLF/LF
normalization. The sole external-build exceptions are the three named manuals.
Their PDF structure and staged bytes are checked; their text and layout require
author review after compilation. Recompilation does not require new hashes.

The full dependency specification remains in `environment.yml`. The
[repository audit](../records/repository_release_audit.md) records the observed
CPU validation environment. No CUDA hardware or TeX installation is needed for
the repository checks. The historical pre-experiment runner remains a
development tool and is not an additional merge gate.

The repository supports replaying the exact recorded final-evaluation
trajectories and reproducing numerical analysis from curated tables. Trained
parameters, raw CSV comparisons and raw-log reconstructions require the
separately preserved archive. A new training run is not guaranteed to reproduce
historical bytes.
