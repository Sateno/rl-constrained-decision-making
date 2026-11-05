# Repository inventory and local archive metadata

`repository_manifest.json` keeps two separate inventories:

- `files` contains the source, tests, environment specification, frozen
  protocols, curated tables/exports, analysis outputs, and documentation that
  belong in Git. It also includes the versioned final-evaluation trajectory
  bundle and its provenance manifest. The verifier reports the complete
  manifest-defined file count, including the three manual PDFs.
- `local_archive_files` records checksums for 181 preserved files under `runs/`.
  These are metadata references only. The files themselves are excluded from
  Git and are not required by the normal repository checks.

All `runs/` contents remain ignored, including final checkpoints, TensorBoard
logs, raw evaluation CSVs, trajectory NPZ files, calibration, and validation
records. Keep the existing local archives intact. If the previous patch caused
run files to be staged, unstage them in your editor without deleting them.
The release verifier rejects tracked files under `runs/`.

The 60 final-evaluation trajectory NPZ files have byte-identical published
copies under `results/trajectories/pap_evaluation_v1/`. Its manifest records
exact archive hashes, episode-table hashes, original archive paths and dataset
provenance. These curated copies are included in `files`; the original paths
remain in `local_archive_files` for optional full-archive verification. New
evaluations continue to write to `runs/` and must not overwrite this bundle.

A clean checkout contains all 3,720 final-evaluation trajectories, or 1,860
OFF/ON pairs, for immediate replay. It also contains the inputs and tools to
reconstruct the reported numerical analysis from curated tables and to execute
the specified study. Inspection of historical model parameters, raw evaluation
CSVs and TensorBoard events still requires the local archive. A new training
execution is not guaranteed to reproduce historical checkpoint bytes.

The published trajectories preserve the original endpoint-collision labels.
They are not newly generated or swept-collision-corrected evaluations. The
[bundle guide](../results/trajectories/pap_evaluation_v1/README.md) states this
interpretation boundary and provenance.

The six scientific inputs in the separate paper ZIP match the corresponding
repository tables and geometry files. Their mapping remains in the manifest.
Paper-production sources and manual LaTeX belong to their separate projects.
The three software manual PDFs and the transfer layout atlas are repository
documentation. Historical build audits retain retired output paths as provenance.

For declared text files, SHA-256 is computed after converting CRLF to LF; all
other bytes remain significant. Binary hashes cover exact bytes. The three
manual PDFs use header, end-marker, cross-reference-pointer, and staged-byte
checks because Overleaf serialization can differ. Review their text and layout
after compilation. No scientific artifact uses that PDF exception.

The default verifier checks repository files, the published trajectory bundle,
and reconstruction of the two suites' summary tables without accessing `runs/`.
The bundle can also be checked with `python -m evaluation.verify_trajectory_bundle`.
With `--archive-root`, the repository verifier additionally
verifies archive hashes, raw CSV/trajectory agreement, final checkpoint contents,
and TensorBoard exports. The archive root is the directory containing `runs/`;
it need not be a Git checkout. These checks are read-only and require no
training or evaluation rerun.

See the [release verification guide](../docs/validation/release_verification.md)
for commands. Do not regenerate hashes to conceal a failed check; resolve the
changed or missing artifact first.
