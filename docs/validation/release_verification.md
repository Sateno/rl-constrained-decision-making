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

All `runs/` files remain excluded from Git and preserved in local archives.
If the previous package caused them to be staged, unstage `runs/` in your editor.
Changing ignore rules alone does not remove an already staged file. Preserve
all local files; `runs/` is not part of the filesystem deletion list.

The normal verifier checks:

- the 182-file repository inventory and audited content, including source,
  protocols, environment specification, curated results, and documentation;
- the three externally compiled manual PDFs and exact staged/working bytes;
- absence of tracked `runs/` files and retired source/production trees;
- protected runtime and protocols against the local `main` reference;
- all 3,000 fixed and 720 transfer rows, design coverage, recorded model
  identities, OFF/ON pairing, and reconstruction of the suite summary tables;
- active Markdown links and the exact staged file inventory/content.

Before staging, omit `--require-tracked` to inspect the working files. Tracked
run files are rejected in either mode. The base comparison uses the local Git
reference and does not query the live remote.

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
Omitting this option verifies the repository only; it makes no claim to have
checked raw files absent from a public checkout.

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

The repository supports rerunning the specified study and reproducing its
numerical analysis from curated tables. Exact historical trajectories, trained
parameters, and raw-log reconstructions require the separately preserved
archive. A new training run is not guaranteed to reproduce historical bytes.
