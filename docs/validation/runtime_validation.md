# Runtime Validation

## Principle

Python validators and tests are the executable source of truth. This guide lists supported entry points, outputs, and PASS markers without duplicating internal command sequences.

Run from the repository root with Conda environment `RL_PROJECTS` active.

## Lightweight tests

```bat
python -m pytest -q -rs
```

Purpose: protect deterministic environment, projection, rollout, evaluation, layout, analysis, and orchestration mechanics. The exact pass count may grow; failures, errors, and unexpected skips are not acceptable.

## Evaluation-time projection regression

```bat
scripts\validate_evaluation_time_projection.bat
```

Python module:

```text
evaluation/validate_evaluation_time_projection.py
```

Covers:

```text
environment and solver availability
source compilation
complete test suite
checkpoint identity and compatibility
expected observation-capacity rejection
deterministic same-checkpoint projection-off/on evaluation
CSV pairing and trajectory/action audit
stochastic active-projection diagnostic
projection runtime benchmark
```

Primary evidence:

```text
runs/validation/evaluation_time_projection_validation.log
runs/validation/evaluation_time_projection_validation_summary.txt
runs/validation/projection_environment.txt
runs/validation/projection_conda_list.txt
runs/validation/projection_runtime.txt
```

PASS marker:

```text
runs/validation/evaluation_time_projection_validation_summary.txt
status=PASS
```

## Consolidated pre-experiment validation

```bat
scripts\validate_pre_experiment_codebase.bat
```

Python module:

```text
evaluation/validate_pre_experiment_codebase.py
```

Adds:

```text
short baseline, high-penalty, and projection-training runs
checkpoint and TensorBoard audits
action-bound clipping and complete projection-burden diagnostics
episode success, collision, and timeout consistency
development-layout projection-off/on evaluation
CSV and NPZ audits
saved-result table and figure generation
```

Output root:

```text
runs/validation/pre_experiment_codebase/
```

PASS marker:

```text
runs/validation/pre_experiment_codebase/pre_experiment_validation_summary.txt
status=PASS
```

The launcher replaces only its reserved validation tree. It does not touch final experiment or calibration artifacts.

## Final-training device benchmark

Run only after the consolidated pre-experiment validation passes:

```bat
scripts\benchmark_training_devices.bat
```

Python module:

```text
experiments/benchmark_training_devices.py
```

The benchmark uses diagnostic seed `9902` and `10,240` transitions by default.
It measures baseline and projection-training throughput on CPU and, when
available, CUDA. The weighted decision represents two baseline-like final
methods and one projection-training method:

\[
T_d=2T_{d,\mathrm{baseline}}+T_{d,\mathrm{projection}}.
\]

CUDA is selected only when every CUDA condition completes stably and its
weighted time is at least ten percent lower than CPU. Otherwise CPU is selected.
The benchmark compares runtime and stability only; checkpoint returns are not
research outcomes.

Output root:

```text
runs/validation/training_device_benchmark/
```

PASS and decision records:

```text
runs/validation/training_device_benchmark/benchmark_summary.txt
runs/validation/training_device_benchmark/benchmark_decision.json
```

The benchmark refuses an existing output tree unless `--replace` is explicit.
It performs no Git operation and its checkpoints are excluded from final
experimental evidence.

## Manual inspection of validation figures

```bat
scripts\inspect_pre_experiment_validation.bat
```

Python module:

```text
evaluation/inspect_pre_experiment_validation.py
```

It prints summaries, lists PDFs, and opens the figure directory. Manual review checks only that figures are not blank, clipped, mislabeled, or inconsistent with the method summary table.

## Core layout calibration

```bat
scripts\calibrate_core_layouts.bat
```

Python module:

```text
experiments/calibrate_core_layouts.py
```

Output root:

```text
runs/calibration/core_navigation_layouts_v1/
```

Automated PASS marker:

```text
runs/calibration/core_navigation_layouts_v1/calibration_summary.txt
status=PASS
manual_review_required=true
```

Calibration requires a separate scientific accept/revise decision recorded in:

```text
docs/records/core_layout_calibration_record.md
```

An accepted calibration directory must not be replaced.

## Training entry points

```bat
scripts\train_ppo_baseline.bat
scripts\train_ppo_high_penalty.bat
scripts\train_ppo_projection.bat
```

All call `experiments/train_ppo_variant.py` and expose seed, budget, checkpoint path, and an optional explicit `--device` selection. Existing checkpoint outputs are refused. Every saved checkpoint records the actual device used.

## Historical clean baseline rebuild

```bat
scripts\run_ppo_baseline_clean.bat
```

This intentionally rebuilds the generated `runs` baseline tree through `experiments/run_clean_ppo_baseline.py --replace-runs`. It is not an ordinary validation command and should be used only when a complete historical baseline regeneration is intended.

## Result generation

```bat
scripts\plot_projection_results.bat PROTOCOL EVALUATION_DIR OUTPUT_DIR [RUNS_DIR]
```

Python module:

```text
analysis/build_projection_results.py
```

It discovers artifacts from metadata, validates protocol completeness, exports raw TensorBoard events plus episode-level and rollout-level training diagnostics, aggregates results, generates tables and figures, and writes a PASS summary. It never trains or evaluates policies. The output directory must not already exist.

## Failure handling

A missing PASS marker means the workflow did not complete. Inspect the terminal exception or master log, correct the source/configuration/evidence, and rerun under a new or explicitly replaceable validation path. Do not weaken a gate to accept an invalid artifact.
