# Completed study and evidence map

This record summarizes the final interpretation of **Runtime Attribution in
PPO with Predictive Action Projection: An Independent-Run Fixed-to-Transfer
Audit**. The manuscript and its production assets are maintained separately.
Earlier analysis records preserve the chronology of observations and revisions.

## Design and chronology

The experiment used three PPO training histories, five independently initialized
runs per history, and one final checkpoint per run. Every run used 51,200 CPU
environment transitions, four environments, 256 transitions per rollout per
environment, and 50 rollout/update iterations. The baseline and projection-trained
histories used collision penalty 10; the high-penalty history used 50. Final
evaluation used penalty 10 throughout.

Every checkpoint was evaluated with the projector OFF and ON. Fixed geometry
used 100 stochastic episodes per checkpoint and mode, for 3,000 episodes.
Transfer used one deterministic actor-mean execution per checkpoint, mode, and
each of 24 prespecified layouts, for 720 executions. All training used the same
single fixed geometry. The transfer suite was calibrated before final training.

The OFF/ON conditions and transfer suite were fixed before the final outcomes
were observed. The decision to organize the paper around runtime attribution
was retrospective. These are descriptive results from a completed experiment.
The training run is the empirical replicate, with n=5 per history. OFF/ON is
paired within run through the same final checkpoint; training histories are
unpaired. Episodes, layouts, and solver calls do not increase the independent
training-run sample size.

## Final findings

| Projection-trained history | OFF success / collision / timeout | ON success / collision / timeout | Paired changes in percentage points |
| --- | --- | --- | --- |
| Fixed geometry | 17.4 / 81.8 / 0.8% | 93.6 / 1.4 / 5.0% | +76.2 / -80.4 / +4.2 |
| Prespecified transfer | 27.50 / 63.33 / 9.17% | 33.33 / 12.50 / 54.17% | +5.83 / -50.83 / +45.00 |

Fixed success increased in every projection-trained run. The five changes were
72, 84, 75, 75, and 75 points, with sample SD 4.55 points, reported as 4.6 in the
paper. Leave-one-run-out mean success changes ranged from 74.25 to 77.25 points.
The fixed success result characterizes the deployed policy-plus-projector
composite. It does not establish an independently safe nominal policy or an
additive projector-only causal effect.

Of 409 fixed OFF collisions, 382 corresponded to ON success, 21 to ON timeout,
and six to ON collision. Of 76 transfer OFF collisions, seven corresponded to
ON success, 54 to ON timeout, and 15 to ON collision. These are matched outcomes
from separate executions, not physical conversions or independent confirmation.

In transfer, success increased in all five projection-trained runs, but timeout
increased much more. Mean collision decreased on 20 layouts and was unchanged
on four; success increased on six and was unchanged on 18. Geometry, action
selection, and repetition differ between suites, so their difference cannot
be attributed to geometry alone.

The projection-trained ON intervention rates were 41.39% in fixed geometry and
56.70% in transfer, computed as equal-run means of episode/layout means. Across
all histories there were 298,984 ON transitions, including 296,269 QP calls
and 2,715 obstacle-free bypasses. No solver failure was recorded under the
implemented acceptance rule. This is numerical exposure, not formal safety,
latency evidence, or universal solver reliability.

No high-penalty OFF checkpoint succeeded in either suite. At this one training
budget and configuration, that history did not provide task completion comparable
to projection-trained ON execution. This is a complete-configuration comparison,
not general failure of reward shaping, a penalty-only causal estimate, or an
equivalence/non-inferiority test.

## Evidence map

All `runs/` paths below identify locally archived files excluded from Git.
Curated `results/` tables and exports, source, and protocols are included in
the repository. Raw trajectories and model parameters require the local archive.

| Conclusion or check | Repository evidence or identified local archive |
| --- | --- |
| Fixed and transfer outcomes, run-level means and sample SDs | `results/tables/*/evaluation_episode_results.csv`, `checkpoint_summary.csv`, `method_summary.csv` |
| OFF/ON paired changes | `results/tables/*/paired_projection_deltas.csv`, `paired_projection_summary.csv` |
| Ranges, leave-one-run-out summaries, and matched outcomes | `analysis/interpretation/` and `results/interpretation/` |
| Runtime correction, clipping, slack, solver exposure, and trajectories | Raw CSV/NPZ under `runs/evaluation/final/`, with tables under `results/tables/` |
| Policy identity and training configuration | All 15 checkpoints under `runs/checkpoints/final/` |
| Training diagnostics and final policy distributions | Final TensorBoard event files under `runs/`, final checkpoints, and fixed-suite training CSV exports |
| Evaluation geometry and protocol | `evaluation/layouts/` and the two final analysis JSON protocols under `experiments/` |
| Calibration and device-selection chronology | Recorded calibration/checkpoint and benchmark files under `runs/`; historical records under `docs/records/` |
| Exact release inclusion and content | `verification/repository_manifest.json` and `evaluation/verify_repository_release.py` |

The study introduces no new PPO, CBF, QP, or safety-filter method. Its contribution
is the bounded independent-run empirical audit. No hypothesis test, confidence
interval, equivalence margin, or non-inferiority margin was prespecified or
used. Correction and slack retain their implemented numerical units and do not
by themselves measure safety. There is no arbitrary-geometry or real-world
deployment claim.
