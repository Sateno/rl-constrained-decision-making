# Completed-study interpretation outputs

These JSON files were reconstructed on 7 September 2026 from the supplied final
evidence, using the preserved `analysis/interpretation/` tools. They contain no
new training or evaluation observations. They preserve the final analysis in
the code repository independently of the paper's presentation assets.

| Suffix | Content |
| --- | --- |
| `inventory.json` | Episode schema, field coverage, and missingness |
| `validation.json` | Protocol, pairing, summary, and build-audit reconciliation |
| `absolute.json` | All runs, absolute outcomes, means, sample SDs, ranges, and leave-one-run-out summaries |
| `paired.json` | Within-run ON-minus-OFF effects and their descriptive sensitivity |
| `correspondences.json` | Complete matched terminal-outcome matrices and marginal reconciliation |
| `layouts.json` | Transfer layout-level outcomes, breadth, and sensitivity |

Both suites have the first five outputs; only transfer has the layout summary.
`reconstruction_commands.json` records the exact argument arrays used, with
repository-relative paths. To reconstruct an output, run the corresponding
command with a new `--output` path. The tools deliberately refuse to replace an
existing output.

Every JSON records its implementation and input byte hashes. These are snapshots
of the supplied files. Git normalizes text line endings, so a reconstruction on
another checkout can have different input hashes while preserving identical
scientific content. The release manifest explicitly normalizes CRLF to LF for
text verification and checks binary evidence exactly. No other text changes
are ignored.

Historical interpretation commands and hashes remain in the original
[analysis command record](../../docs/records/Predictive_Action_Projection_Analysis_Command_Record.md).
These release reconstructions do not replace or relabel those earlier runs.
The [final study record](../../docs/records/final_study_record.md) states the
supported conclusions and inferential limits.
