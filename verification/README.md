# Repository inventory and local archive metadata

`repository_manifest.json` keeps two separate inventories:

- `files` contains the source, tests, environment specification, frozen
  protocols, curated tables/exports, analysis outputs, and documentation that
  belong in Git. Together with the manifest and three manual PDFs, the release
  contains 182 repository files.
- `local_archive_files` records checksums for 181 preserved files under `runs/`.
  These are metadata references only. The files themselves are excluded from
  Git and are not required by the normal repository checks.

All `runs/` contents remain ignored, including final checkpoints, TensorBoard
logs, raw evaluation CSVs, trajectory NPZ files, calibration, and validation
records. Keep the existing local archives intact. If the previous patch caused
run files to be staged, unstage them in your editor without deleting them.
The release verifier rejects tracked files under `runs/`.

A clean checkout contains the inputs and tools to reconstruct the reported
numerical analysis from curated tables and to execute the specified study.
Inspection of the exact historical model parameters and per-step trajectories
requires the local archive. A new training execution is not guaranteed to
reproduce historical checkpoint bytes. Raw-archive availability is not claimed
for GitHub.

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

The default verifier checks repository files and reconstructs the two suites'
summary tables without accessing `runs/`. With `--archive-root`, it additionally
verifies archive hashes, raw CSV/trajectory agreement, final checkpoint contents,
and TensorBoard exports. The archive root is the directory containing `runs/`;
it need not be a Git checkout. These checks are read-only and require no
training or evaluation rerun.

See the [release verification guide](../docs/validation/release_verification.md)
for commands. Do not regenerate hashes to conceal a failed check; resolve the
changed or missing artifact first.
