# Scientific interpretation scripts

This directory contains read-only tools that inventory, validate, and summarize
already-generated evaluation evidence. Script names describe the operation they
perform; analysis-phase or step numbers belong in the existing Analysis Command
Record, not in filenames.

## Reproducibility contract

Every evidence location is an explicit command-line argument. The tools do not
search a repository, select a Git commit, infer a suite name from a directory,
or contain study-specific method labels, paths, row totals, or checkpoint names.
Methods, training seeds, required projection modes, layouts, repeats, and
expected table sizes are derived from the protocol and layout files supplied to
the command.

The experimental hierarchy is explicit throughout these outputs: a training
run is the empirical replicate, one final checkpoint represents each run, and
Projector OFF/ON observations are paired within that run through the same final
checkpoint. `checkpoint` and `checkpoint_sha256` identify the retained model
artifact; they do not redefine the statistical unit.

Each successful command writes deterministic strict JSON containing:

- the path string supplied for every input and the input's SHA-256;
- the entrypoint and evidence-module SHA-256 values;
- the protocol-derived design and relevant validation or analysis results; and
- no timestamp or silently expanded machine-specific path.

Output creation is exclusive. An existing output is never overwritten. All
inputs are read and all requested checks complete before the output is created,
so a failed validation produces no result artifact.

## Why the reconstruction formulas are repeated here

`_evaluation_evidence.py` independently reconstructs the checkpoint, method,
paired-delta, and paired-summary tables from the episode table. It deliberately
does not import the result-building implementation. This limited duplication is
a scientific audit control: reusing the producer's aggregation functions could
repeat the producer's implementation error and falsely report agreement. The
two implementations are compared numerically with exact key matching, exact
NaN-pattern matching, and a maximum absolute tolerance of `1e-12`.

The evidence module is schema-specific rather than a general statistics
framework. Shared code is limited to mechanics genuinely needed by multiple
entrypoints: strict parsing, design derivation, evidence validation, canonical
reconstruction, numerical reconciliation, hashing, and deterministic writing.

## Commands

The examples in this section document each command-line interface. They are
usage templates, not evidence that a particular command was executed. Exact
study invocations and their results are recorded in
`docs/records/Predictive_Action_Projection_Analysis_Command_Record.md`.

The multiline command synopses below use POSIX-shell `\` continuations. In
Windows Command Prompt, replace each continuation with `^`. Retain
forward-slash path strings in recorded study invocations on both platforms;
the deterministic JSON preserves caller-supplied path strings, so changing
only their slash direction would prevent a byte-for-byte output match.

### Inventory one episode table

POSIX-shell synopsis:

```sh
python -m analysis.interpretation.inventory_evaluation_table \
  --episodes <evaluation_episode_results.csv> \
  --output <new_inventory.json> \
  [--label <descriptive-label>]
```

The inventory reports each column's parsed type, missingness, cardinality, and
numeric range or categorical examples, both overall and by projection mode.

### Validate a complete evaluation-table family

POSIX-shell synopsis:

```sh
python -m analysis.interpretation.validate_evaluation_tables \
  --episodes <evaluation_episode_results.csv> \
  --protocol <analysis_protocol.json> \
  --layout-suite <layout_suite.json> \
  --checkpoint-summary <checkpoint_summary.csv> \
  --method-summary <method_summary.csv> \
  --paired-deltas <paired_projection_deltas.csv> \
  --paired-summary <paired_projection_summary.csv> \
  --build-audit <result_build_audit.json> \
  [--compare-checkpoints-with <other_evaluation_episode_results.csv>] \
  --output <new_validation.json> \
  [--label <descriptive-label>]
```

`--compare-checkpoints-with` is repeatable. It checks that every supplied table
uses the same method/seed checkpoint path and SHA-256 map as `--episodes`.

The validation covers protocol-derived row counts, method/seed/mode/layout
coverage, pairing keys, checkpoint uniqueness, strict Boolean and numeric
parsing, canonical layout/repeat/episode/seed mapping, outcome partitioning,
count/rate formulas, structural zeros, obstacle-free clearance NaNs,
summary-table reconstruction, paired-table reconstruction, and result-build
audit reconciliation.

### Summarize absolute evaluation performance

POSIX-shell synopsis:

```sh
python -m analysis.interpretation.summarize_absolute_evaluation_performance \
  --episodes <evaluation_episode_results.csv> \
  --protocol <analysis_protocol.json> \
  --layout-suite <layout_suite.json> \
  --checkpoint-summary <checkpoint_summary.csv> \
  --method-summary <method_summary.csv> \
  [--worked-example-method <method-label> \
   --worked-example-projection-mode {disabled,enabled}] \
  --output <new_absolute_performance_summary.json> \
  [--label <descriptive-label>]
```

This command reconstructs each training-run row from episodes, reconciles the
supplied checkpoint and method summaries, derives timeout as `1 - success - collision`,
reports every training run's absolute metrics with its final-checkpoint identity, and then reports method-level
means, sample standard deviations, ranges, and leave-one-seed-out means. Outcome
totals are labeled as evaluation coverage, not as independent policy
replication. The optional worked example exposes every intermediate term in one
sample-standard-deviation calculation; both worked example arguments must be
supplied together.

### Summarize performance across a layout suite

POSIX-shell synopsis:

```sh
python -m analysis.interpretation.summarize_layout_transfer_performance \
  --episodes <evaluation_episode_results.csv> \
  --protocol <analysis_protocol.json> \
  --layout-suite <layout_suite.json> \
  --checkpoint-summary <checkpoint_summary.csv> \
  --method-summary <method_summary.csv> \
  --paired-deltas <paired_projection_deltas.csv> \
  --paired-summary <paired_projection_summary.csv> \
  --build-audit <result_build_audit.json> \
  [--worked-example-method <method-label> \
   --worked-example-train-seed <integer> \
   --worked-example-layout <layout-id>] \
  --output <new_layout_transfer_summary.json> \
  [--label <descriptive-label>]
```

This command reconstructs absolute and projection-enabled-minus-disabled
outcomes by training run and layout, reconciles every supplied frozen table and
the build audit, and reports sign breadth, leave-one-training-run-out and
leave-one-layout-out stability, effect concentration, and aggregation
identities. The layouts are fixed evaluation conditions. They describe task
coverage and heterogeneity, while training runs provide the method-level
replication and one final checkpoint represents each run. Marginal net outcome
accounting is not an episode-transition matrix. All three worked-example arguments must be supplied
together.

### Summarize paired terminal-outcome correspondences

POSIX-shell synopsis:

```sh
python -m analysis.interpretation.summarize_paired_terminal_outcome_correspondences \
  --episodes <evaluation_episode_results.csv> \
  --protocol <analysis_protocol.json> \
  --layout-suite <layout_suite.json> \
  --checkpoint-summary <checkpoint_summary.csv> \
  --method-summary <method_summary.csv> \
  --paired-deltas <paired_projection_deltas.csv> \
  --paired-summary <paired_projection_summary.csv> \
  --build-audit <result_build_audit.json> \
  --output <new_terminal_outcome_correspondence_summary.json> \
  [--label <descriptive-label>]
```

This command pairs projection-disabled and projection-enabled observations by
method, training seed, checkpoint hash, layout, repeat, and evaluation seed.
For every training run and method it emits a complete 3 by 3 count matrix with
disabled outcomes as rows, enabled outcomes as columns, and the outcome order
success, collision, timeout. All nine labeled cells remain present when their
counts are zero. Row and column margins reconcile with the supplied absolute
tables, and enabled-minus-disabled outcome deltas reconcile with the supplied
paired tables. Method-level means and sample standard deviations use training
runs as the empirical units, with one final checkpoint representing each run.

The matrix arrow is a descriptive correspondence between two matched
controller executions. It is not a temporal transition within one trajectory
and does not establish deliberate projector reliance, policy-projector
co-adaptation, formal safety, or behavior under a different projector or
arbitrary geometry.

## Provenance boundary

These commands do not generate episodes or rerun policies. Commands used to
produce the frozen experiment and evaluation evidence remain recorded in
`docs/records/Predictive_Action_Projection_Analysis_Command_Record.md`.
A future campaign runner would need a separate, explicit protocol-driven design;
creating one is outside this interpretation-tool change.

Exact invocations of these interpretation tools, their input and output hashes,
exit status, and the report claims they support are recorded in the same
existing Analysis Command Record. No second scientific-interpretation command
record is used.
