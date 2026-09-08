# Completed-repository audit, revised 8 September 2026

## Source and scope

The supplied development branch was `final_evaluation_scripts`. Its HEAD, local
`main`, and included `origin/main` all identified
`5a5cc2041ad5b6194a86aff9e61460873ce185c9`. There were no commits ahead of `main`.
This audit uses the supplied Git snapshot; it does not assert the current state
of a live remote. The scientific runtime and frozen protocols also match the
protocol-freeze commit `ba64926aed98b08b7b285266cf85989d466f9f1c`.

The separate Overleaf ZIP supplied for this audit is the final-paper reference.
Its six scientific input files match the repository's episode tables, checkpoint
summaries, and fixed/transfer geometry JSONs byte for byte before Git line-ending
normalization. The manifest records the file mapping and input-archive hashes.

No final training, evaluation, calibration, or device benchmark was rerun.
No final checkpoint, raw final CSV/NPZ file, TensorBoard event, scientific table,
protocol, layout, or numerical conclusion was altered.

## Corrections

- Removed the paper-specific `analysis/publication/` package and generated
  `results/publication_assets/` tree, with their publication-export test.
- Removed the four generated LaTeX result tables and their producer/validator
  requirements. CSV/JSON scientific outputs remain supported.
- Retained the transfer atlas as benchmark documentation under `docs/assets/`,
  with its independent renderer and geometry checks under `analysis/`.
- Fixed saved TensorBoard checkpoint lookup for relocated Windows and POSIX
  checkouts. The subsequent checkpoint SHA-256 match remains mandatory.
- Added the read-only release verifier, content inventory, and negative tests
  for changed/missing/ignored evidence, staged mismatches, base-source drift,
  and inconsistent paper inputs.
- Restored the original blanket `runs/` exclusion. All 181 raw/provenance files
  remain preserved locally. Their checksums are archive metadata, separate from
  the 182-file Git inventory. Optional archive verification requires no staging.
- Split repository-table verification from raw-archive verification. The normal
  release gate works with no `runs/` directory and rejects tracked run artifacts.
- Added the completed interpretation JSON outputs and exact reconstruction
  argument arrays, using only the supplied frozen data.
- Corrected the README's fixed paired-success sample SD from 4.5 to 4.6 points,
  matching the paper and the actual five values.
- Updated active documentation and corrected all three software manuals in
  the author-supplied Overleaf sources, maintained outside this repository. Corrections cover the four action representations,
  unbounded Gaussian samples and explicit clipping, reward indexing, completed
  protocols, five-run replication, retrospective attribution, matched outcomes,
  and the release/development verification boundary.

Earlier chronological records and historical build audits retain their
original command/output references. Their new context notes direct readers to
the completed study and release checks. Retired LaTeX paths in those historical
records are provenance, not current required outputs.

## Verification evidence

| Check | Result |
| --- | --- |
| Supplied `main` full test suite | 71 passed |
| Repository-only clean export full test suite | 119 passed; 10 subtests passed; no failures, errors, warnings, or skips |
| Protected runtime/protocol comparison | Matches `main` and the protocol-freeze source, ignoring only CRLF/LF |
| Fixed evaluation | 3,000 repository rows and summaries reconciled; trajectories verified separately against the local archive |
| Transfer evaluation | 720 repository rows and summaries reconciled; trajectories verified separately against the local archive |
| Final model identities | All 15 SHA-256 values, metadata, shapes, and finite parameters checked |
| Summary reconstruction | Both checkpoint/method and paired table families independently reconciled |
| Training evidence | All final TensorBoard scalar, episode, and rollout exports reconstructed |
| Solver exposure | 298,984 ON transitions; 296,269 QP calls; zero recorded failures |
| Separate paper inputs | All six matching scientific inputs |
| Existing development artifact bundle | Accepted after removing retired LaTeX outputs, without new evaluation |
| Software manuals | All three authoritative Overleaf sources compile; no unresolved references or overfull boxes; rendered pages inspected |
| Transfer atlas | Four vector PDF pages covering all 24 layouts; rendered pages inspected |

The Python audit used Python 3.12.13 on Linux with Torch 2.12.0+cpu, Gymnasium 1.3.0,
CVXPY 1.9.2, OSQP 1.1.3, NumPy 2.5.3, pandas 2.2.3, and pytest 9.1.1. Exact
observed versions are in [audit environment](../../verification/audit_environment.json).
This is an actual CPU verification environment, not a claim that a new Windows
Conda environment was recreated. The original training-environment records and
`environment.yml` are retained. The manual sources retain the author's shared preamble and bibliography.
The separately delivered Overleaf patch changes the affected document bodies,
shared diagram labels, and a one-line algorithm-anchor definition guard. Local build details are recorded in the
external handoff; no manual source or TeX build helper belongs in this tree.

The handoff was checked by applying its changed/new files and deletion list
to the supplied branch, exporting only the 182 intended repository files, and
running the full tests and release verifier with `runs/` entirely absent.
The optional archive check was run separately against the preserved original
raw files. No raw file was deleted or changed by this correction. The verifier's staged-content mode
requires the exact intended inventory and checks staged bytes as well as the
working files. Verification does not commit, tag, merge, or push the repository.

## Boundaries for the final handoff

The manifest checks exact scientific binary bytes and text bytes after
CRLF-to-LF normalization. The three manuals are the sole external-build
exceptions: PDF headers, end markers, cross-reference pointers, inclusion, and
staged-byte agreement are checked, while rendered content requires author
review after compilation in Overleaf. It does not ignore scientific values, arbitrary whitespace, or
missing content. The release verifier compares with the local base reference;
the user remains responsible for the final commit, merge, and push.

The previous handoff's proposed availability replacement is withdrawn.
The paper's original statement that the repository does not contain every final
per-step trajectory record remains applicable. Raw checkpoints and logs also
remain local. The manuscript and paper-production assets are outside this
repository; the archive policy does not change the results or conclusions.
