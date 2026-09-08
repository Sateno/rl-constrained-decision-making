# Predictive Action Projection with PPO: Analysis Record

> **Completed-study context, 7 September 2026:** This is a chronological record.
> Statements about pending analysis, historical commands, temporary outputs, or
> retired paper assets describe the stage at which they were written. Current
> findings are in the [final study record](final_study_record.md), and the merge
> checks are in the [release verification guide](../validation/release_verification.md).


**Author:** Salvador Tenorio\
**Status:** Living analysis record\
**Study protocol:** Frozen protocol v1\
**Frozen source:** `ba64926aed98b08b7b285266cf85989d466f9f1c`\
**Repository path:** `docs/records/Predictive_Action_Projection_Analysis_Record.md`\
**Analysis started:** 2026-08-22

## Purpose

This document accumulates verified numerical results and their interpretation as the final study analysis proceeds. It is a working evidence record, not yet final paper prose. Observations, interpretations, and unresolved questions are kept distinct.

## Campaign and data integrity

- Three training methods: PPO baseline, PPO high collision penalty, and PPO trained with projection.
- Five independent training seeds per method, giving 15 checkpoints.
- Primary evaluation: fixed training geometry, stochastic Gaussian policy sampling, 100 episodes per checkpoint and projection mode, 3,000 episodes total.
- Secondary evaluation: 24 deterministic core-layout transfers per checkpoint and projection mode, 720 episodes total.
- Complete evaluation campaign: 60 evaluator invocations and 3,720 episodes.
- All 60 CSV and 60 NPZ raw evidence artifacts passed protocol and schema validation.
- All trajectory archives were indexed and accounted for the expected episode counts.
- Projection solver failures: zero.
- The 120 raw numerical artifacts were recorded in a SHA-256 manifest and verified byte-for-byte against the OneDrive archive.
- Seven execution logs and all dataset-audit files were backed up.
- Both primary and transfer result-table and figure builds passed their generated audit records with no skipped figures.

## Primary evaluation: fixed training geometry

Values below are the mean and sample standard deviation across five independently trained checkpoints. Success and collision are episode rates. Timeout is the remaining terminal outcome and is not displayed in the generated method table.

### Method-level summary

| Training method | Projection | Return | Success | Collision | Minimum clearance | Intervention | Correction norm | Slack sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PPO baseline | Off | -5.086 ± 3.997 | 0.152 ± 0.212 | 0.140 ± 0.072 | 0.375 ± 0.131 | — | — | — |
| PPO baseline | On | -3.912 ± 4.565 | 0.178 ± 0.250 | 0.002 ± 0.004 | 0.414 ± 0.126 | 0.034 ± 0.014 | 0.015 ± 0.008 | 0.000056 ± 0.000036 |
| PPO high penalty | Off | -9.364 ± 0.448 | 0.000 ± 0.000 | 0.134 ± 0.063 | 0.543 ± 0.116 | — | — | — |
| PPO high penalty | On | -8.659 ± 0.695 | 0.002 ± 0.004 | 0.006 ± 0.009 | 0.582 ± 0.102 | 0.028 ± 0.015 | 0.012 ± 0.008 | 0.000048 ± 0.000036 |
| PPO trained with projection | Off | -5.357 ± 1.644 | 0.174 ± 0.080 | 0.818 ± 0.076 | -0.016 ± 0.009 | — | — | — |
| PPO trained with projection | On | 10.004 ± 1.599 | 0.936 ± 0.088 | 0.014 ± 0.021 | 0.067 ± 0.014 | 0.414 ± 0.025 | 0.208 ± 0.041 | 0.000521 ± 0.000170 |

### Paired projection effects

Each delta is projection enabled minus projection disabled for the same checkpoint and matched evaluation episodes. Values are the mean and sample standard deviation of the five checkpoint-level paired effects.

| Training method | Return delta | Success delta | Collision delta | Clearance delta |
|---|---:|---:|---:|---:|
| PPO baseline | +1.173 ± 0.698 | +0.026 ± 0.038 | -0.138 ± 0.073 | +0.039 ± 0.011 |
| PPO high penalty | +0.704 ± 0.276 | +0.002 ± 0.004 | -0.128 ± 0.058 | +0.039 ± 0.015 |
| PPO trained with projection | +15.361 ± 0.833 | +0.762 ± 0.045 | -0.804 ± 0.061 | +0.083 ± 0.006 |

### Verified observations

1. **Baseline PPO is seed-sensitive and unreliable at the frozen training budget.** Mean unprotected success is 15.2%, but its 21.2-point across-seed standard deviation exceeds the mean. Enabling projection nearly eliminates collisions, yet increases mean success by only 2.6 points.

2. **The higher collision penalty does not yield goal-reaching competence.** Mean success is 0% without projection and 0.2% with projection. Projection reduces collision incidence substantially, but the protected controller still almost always times out.

3. **Projection-trained PPO is the only consistently competent composite controller on the training geometry.** With projection enabled, it reaches 93.6% ± 8.8% success, 1.4% ± 2.1% collision, and a mean return of 10.004 ± 1.599.

4. **The projection-trained nominal actor remains strongly dependent on the projector.** Removing projection reduces success to 17.4% ± 8.0% and raises collision to 81.8% ± 7.6%. The protected system intervenes on 41.4% ± 2.5% of steps.

5. **Projection improves constraint outcomes but does not generally create goal-directed behavior.** For baseline and high-penalty PPO, the paired collision reductions of 13.8 and 12.8 percentage points correspond to success increases of only 2.6 and 0.2 points. Because success, collision, and timeout are exhaustive outcomes, most avoided collisions become timeouts.

6. **The composite benefit for projection-trained PPO is large and repeatable across training seeds.** Its paired effects are +76.2 ± 4.5 percentage points in success, -80.4 ± 6.1 points in collision, +15.361 ± 0.833 in return, and +0.083 ± 0.006 in minimum clearance.

7. **Projection use differs sharply by learned policy.** Protected baseline and high-penalty policies trigger projection on only 3.4% and 2.8% of steps, whereas the projection-trained policy triggers it on 41.4% of steps and receives much larger action corrections.

8. **The negative mean minimum clearance without protection for projection-trained PPO is consistent with frequent obstacle penetration or collision.** With projection enabled, the mean minimum clearance becomes positive.

9. **Projection slack is nonzero but numerically small, and solver failures are absent.** Its practical significance will be assessed against the trajectory plots and any relevant geometric scale before stronger language is used.

### Checkpoint-level outcome variation

`S/C/T` denotes success, collision, and timeout percentages over 100 stochastic episodes. Intervention is the percentage of protected steps on which projection modified the bounded nominal action.

| Training method | Seed | Projection off S/C/T | Projection on S/C/T | Return off → on | Protected intervention |
|---|---:|---:|---:|---:|---:|
| PPO baseline | 1 | 8/10/82 | 9/1/90 | -5.245 → -4.492 | 2.2% |
| PPO baseline | 2 | 0/5/95 | 0/0/100 | -8.693 → -8.377 | 1.9% |
| PPO baseline | 3 | 51/17/32 | 60/0/40 | 0.880 → 3.052 | 4.6% |
| PPO baseline | 4 | 17/14/69 | 20/0/80 | -3.652 → -2.350 | 3.1% |
| PPO baseline | 5 | 0/24/76 | 0/0/100 | -8.719 → -7.395 | 5.0% |
| PPO high penalty | 1 | 0/7/93 | 1/0/99 | -9.869 → -9.379 | 1.1% |
| PPO high penalty | 2 | 0/12/88 | 0/0/100 | -8.871 → -8.027 | 4.0% |
| PPO high penalty | 3 | 0/16/84 | 0/2/98 | -9.676 → -9.051 | 2.8% |
| PPO high penalty | 4 | 0/23/77 | 0/1/99 | -8.924 → -7.810 | 4.5% |
| PPO high penalty | 5 | 0/9/91 | 0/0/100 | -9.478 → -9.030 | 1.4% |
| PPO trained with projection | 1 | 6/92/2 | 78/5/17 | -7.676 → 7.178 | 45.0% |
| PPO trained with projection | 2 | 12/88/0 | 96/1/3 | -6.517 → 10.327 | 42.2% |
| PPO trained with projection | 3 | 22/77/1 | 97/0/3 | -4.346 → 10.701 | 41.6% |
| PPO trained with projection | 4 | 24/76/0 | 99/0/1 | -4.007 → 11.007 | 38.8% |
| PPO trained with projection | 5 | 23/76/1 | 98/1/1 | -4.239 → 10.809 | 39.4% |

### Verified checkpoint-level observations

1. **Baseline seed 3 dominates the aggregate baseline success rate.** It reaches 51% without projection and 60% with projection. The other four protected checkpoints reach 0%, 0%, 9%, and 20%; two never succeed at all.

2. **The baseline is capable but not reliable at 51,200 transitions.** Seed 3 proves that the frozen reward and architecture can produce useful behavior, while the five-seed spread shows that this outcome is not repeatable at the chosen budget.

3. **The high-penalty failure is consistent across seeds.** All five unprotected checkpoints have zero success. Across 500 protected episodes, there is only one success, from seed 1.

4. **The projection-trained composite benefit occurs in every checkpoint.** Protected success ranges from 78% to 99%, compared with 6% to 24% without protection. Thus, the method-level effect is not created by one exceptional seed.

5. **Projection-trained seed 1 is the weakest protected checkpoint but remains strongly improved.** Its success rises from 6% to 78% and collision falls from 92% to 5%; it is empirical variation, not a technical anomaly.

6. **Projection removes or nearly removes baseline and high-penalty collisions without repairing non-completion.** Baseline seeds 2 and 5 and four of five high-penalty seeds reach 100% timeout when protected.

7. **Intervention rate alone does not explain goal competence.** Baseline seed 5 has the largest baseline intervention rate yet zero protected success, while seed 3 combines a similar rate with 60% success.

### Action-bound clipping diagnostic

Values are the mean and sample standard deviation across five checkpoints. Rates are fractions of evaluation steps. Clipping norm measures the magnitude of the difference between the sampled normalized action and its bounded value.

| Training method | Projection | Any-component clipping | Speed clipping | Turn-rate clipping | Mean clipping norm |
|---|---|---:|---:|---:|---:|
| PPO baseline | Off | 0.648 ± 0.131 | 0.489 ± 0.196 | 0.309 ± 0.014 | 0.523 ± 0.269 |
| PPO baseline | On | 0.650 ± 0.133 | 0.491 ± 0.198 | 0.309 ± 0.014 | 0.528 ± 0.271 |
| PPO high penalty | Off | 0.719 ± 0.058 | 0.578 ± 0.103 | 0.329 ± 0.032 | 0.635 ± 0.135 |
| PPO high penalty | On | 0.722 ± 0.059 | 0.581 ± 0.105 | 0.329 ± 0.032 | 0.640 ± 0.136 |
| PPO trained with projection | Off | 0.603 ± 0.033 | 0.421 ± 0.048 | 0.314 ± 0.012 | 0.423 ± 0.056 |
| PPO trained with projection | On | 0.592 ± 0.031 | 0.401 ± 0.043 | 0.322 ± 0.014 | 0.406 ± 0.049 |

### Verified clipping observations

1. **Action-bound clipping is common for every method.** Mean any-component clipping ranges from approximately 59% to 72% of evaluation steps.

2. **Greater clipping does not indicate greater competence.** High-penalty PPO has the largest clipping rate and norm but essentially no goal completion.

3. **Projection-trained PPO has the lowest and most consistent clipping burden across seeds.** It nevertheless remains clipped on roughly 60% of steps, so its protected competence cannot be described as arising from unsaturated nominal actions.

4. **Enabling deployment-time projection barely changes aggregate clipping for policies trained without projection.** This is consistent with action bounding occurring before projection in the action path, although projection can still change later states and therefore later nominal actions.

5. **Baseline clipping varies substantially across seeds, especially in the speed component.** This may be diagnostically relevant to seed sensitivity, but clipping alone cannot explain competence: prior checkpoint evidence includes similarly clipped baseline policies with markedly different success.

6. **Across-seed variation is concentrated more strongly in speed clipping than turn-rate clipping.** Turn-rate clipping means and standard deviations are comparatively stable across methods.

### Primary clipping concentration by checkpoint

| Training method | Seed | Any clipping, off → on | Speed clipping, off → on | Turn clipping, off → on | Clipping norm, off → on |
|---|---:|---:|---:|---:|---:|
| PPO baseline | 1 | 0.555 → 0.555 | 0.351 → 0.352 | 0.311 → 0.312 | 0.337 → 0.338 |
| PPO baseline | 2 | 0.844 → 0.844 | 0.776 → 0.776 | 0.304 → 0.304 | 0.945 → 0.946 |
| PPO baseline | 3 | 0.555 → 0.557 | 0.337 → 0.336 | 0.331 → 0.332 | 0.342 → 0.346 |
| PPO baseline | 4 | 0.560 → 0.561 | 0.367 → 0.368 | 0.302 → 0.302 | 0.348 → 0.348 |
| PPO baseline | 5 | 0.724 → 0.733 | 0.613 → 0.621 | 0.295 → 0.297 | 0.644 → 0.661 |
| PPO high penalty | 1 | 0.684 → 0.683 | 0.495 → 0.494 | 0.375 → 0.374 | 0.546 → 0.546 |
| PPO high penalty | 2 | 0.800 → 0.800 | 0.713 → 0.714 | 0.302 → 0.302 | 0.831 → 0.831 |
| PPO high penalty | 3 | 0.650 → 0.651 | 0.463 → 0.464 | 0.346 → 0.346 | 0.487 → 0.491 |
| PPO high penalty | 4 | 0.747 → 0.757 | 0.644 → 0.655 | 0.296 → 0.297 | 0.696 → 0.717 |
| PPO high penalty | 5 | 0.716 → 0.716 | 0.576 → 0.577 | 0.327 → 0.327 | 0.614 → 0.615 |
| PPO trained with projection | 1 | 0.643 → 0.632 | 0.488 → 0.465 | 0.304 → 0.312 | 0.487 → 0.463 |
| PPO trained with projection | 2 | 0.624 → 0.606 | 0.443 → 0.402 | 0.328 → 0.344 | 0.457 → 0.428 |
| PPO trained with projection | 3 | 0.606 → 0.599 | 0.426 → 0.413 | 0.319 → 0.322 | 0.439 → 0.425 |
| PPO trained with projection | 4 | 0.560 → 0.552 | 0.368 → 0.353 | 0.300 → 0.307 | 0.350 → 0.339 |
| PPO trained with projection | 5 | 0.582 → 0.573 | 0.381 → 0.372 | 0.320 → 0.322 | 0.383 → 0.374 |

### Verified primary checkpoint-level clipping observations

1. **The two most saturated baseline checkpoints are both non-completing.** Seeds 2 and 5 have the largest speed-clipping rates and clipping norms, and both have zero success in either deployment mode.

2. **Clipping still does not explain baseline seed sensitivity.** Seeds 1, 3, and 4 have nearly identical any-component clipping rates (approximately 55.5%–56.1%) and similar clipping norms, yet their unprotected success rates are 8%, 51%, and 17%.

3. **Turn-rate clipping is especially uninformative about competence.** It stays near 30%–33% for most checkpoints while success varies from 0% to 99% across the campaign.

4. **High-penalty PPO is heavily clipped but fails across its entire clipping range.** Any-component clipping varies from 65.0% to 80.0%, with zero unprotected success for every checkpoint.

5. **Projection-trained clipping varies only moderately while protected performance remains consistently high.** Seed 1 has the largest clipping burden and weakest protected success, while seed 4 has the smallest burden and strongest success; with only five seeds this is a diagnostic association, not evidence of causation.

6. **Projection's large outcome effects occur with only small clipping changes.** Enabling projection changes checkpoint clipping by at most about two percentage points for nearly all conditions, while projection-trained success rises by 72–84 points and collision falls by 75–87 points.

7. **Speed clipping carries most of the large between-checkpoint variation.** The next diagnostic is the learned policy standard deviation, which can determine whether persistent stochastic spread plausibly contributes to this saturation.

### Availability of the policy-variance diagnostic

Inspection of `training_scalar_events.csv` found action-bound, episodic-outcome, projection, and safety tags, but no policy standard deviation, log standard deviation, or equivalent policy-variance scalar.

The scalar table also carries `training_diagnostics_schema_version`, event index, training step, method, display name, training seed, checkpoint SHA-256, training-projection flag, and run-directory fields. These provide sufficient provenance to associate every exported scalar with the exact frozen checkpoint and training condition; they do not add a policy-variance measurement.

Consequently:

- Policy variance cannot be reconstructed from the generated scalar table.
- Clipping frequency is not a substitute for policy variance because it combines the actor mean, stochastic spread, and visited-state distribution.
- The authoritative final values must be read directly from each frozen checkpoint's state-independent log-standard-deviation parameter and exponentiated to obtain standard deviation.
- This is a read-only diagnostic of the existing campaign and does not modify checkpoints or evaluation evidence.

Inspection of `ppo_baseline_51200_seed1.pt` confirms the checkpoint structure required for that diagnostic:

- The checkpoint is a dictionary containing `agent_state_dict`, `args`, `run_name`, `global_step`, observation/action shapes and dimensions, and device metadata.
- `agent_state_dict` contains the parameter `actor_logstd` alongside the critic and actor-mean network parameters.
- The checkpoint records `global_step = 51200`, observation dimension 21, action dimension 2, and CPU execution.
- The two `actor_logstd` components correspond to normalized speed and turn-rate actions. Exponentiating them yields the final Gaussian standard deviations used by the stochastic policy.

### Final state-independent policy standard deviation

| Training method | Seed | Speed log σ | Speed σ | Turn-rate log σ | Turn-rate σ |
|---|---:|---:|---:|---:|---:|
| PPO baseline | 1 | -0.018345 | 0.981823 | -0.046455 | 0.954608 |
| PPO baseline | 2 | -0.021670 | 0.978563 | -0.036583 | 0.964078 |
| PPO baseline | 3 | 0.008512 | 1.008548 | -0.026727 | 0.973627 |
| PPO baseline | 4 | -0.007358 | 0.992669 | -0.065471 | 0.936626 |
| PPO baseline | 5 | -0.016245 | 0.983887 | -0.059023 | 0.942685 |
| PPO high penalty | 1 | 0.004385 | 1.004395 | -0.026019 | 0.974316 |
| PPO high penalty | 2 | -0.000447 | 0.999553 | -0.041312 | 0.959530 |
| PPO high penalty | 3 | 0.015379 | 1.015498 | -0.010307 | 0.989746 |
| PPO high penalty | 4 | -0.015104 | 0.985009 | -0.056847 | 0.944739 |
| PPO high penalty | 5 | 0.009237 | 1.009280 | -0.042472 | 0.958417 |
| PPO trained with projection | 1 | -0.020712 | 0.979501 | -0.064371 | 0.937657 |
| PPO trained with projection | 2 | 0.002682 | 1.002686 | -0.014533 | 0.985572 |
| PPO trained with projection | 3 | 0.013160 | 1.013247 | -0.047055 | 0.954035 |
| PPO trained with projection | 4 | -0.014938 | 0.985173 | -0.081094 | 0.922107 |
| PPO trained with projection | 5 | 0.010564 | 1.010620 | -0.053571 | 0.947839 |

Method-level descriptive summaries across five checkpoints are:

| Training method | Speed σ | Turn-rate σ |
|---|---:|---:|
| PPO baseline | 0.989 ± 0.012 | 0.954 ± 0.015 |
| PPO high penalty | 1.003 ± 0.012 | 0.965 ± 0.017 |
| PPO trained with projection | 0.998 ± 0.015 | 0.949 ± 0.024 |

Across all 15 checkpoints, speed σ ranges from 0.979 to 1.015 and turn-rate σ from 0.922 to 0.990. These are normalized-action units, whose admissible interval is `[-1, 1]`, not physical velocity units.

### Verified policy-variance observations

1. **Every final policy retains broad stochastic spread relative to the bounded action scale.** Both standard-deviation components remain close to one normalized-action unit at 51,200 transitions.

2. **This quantitatively explains much of the common stochastic clipping floor.** For a zero-mean Gaussian with σ = 1, one component falls outside `[-1, 1]` approximately 31.7% of the time. For two conditionally independent components, at least one clips approximately 53.4% of the time. The actual σ ranges imply a centered two-component clipping floor of roughly 50%–54%, close to the lowest observed primary any-component clipping rates of approximately 55%.

3. **Additional speed clipping must reflect the learned means and visited states.** Final speed σ changes little across checkpoints, while primary speed clipping ranges from 33.6% to 77.6%. The large excess over the centered-Gaussian baseline therefore cannot be assigned to variance alone.

4. **Final variance does not explain baseline competence.** Baseline seed 3 is the competent checkpoint even though it has the largest speed and turn-rate σ among the five baseline checkpoints. Seeds with nearly identical or smaller σ perform much worse.

5. **Final variance does not distinguish the training methods.** Their σ distributions overlap almost completely, yet high-penalty PPO never develops goal-reaching competence and protected projection-trained PPO succeeds on 78%–99% of primary episodes.

6. **Broad variance is not the sole source of unsafe behavior.** All unprotected projection-trained policies exhibit zero clipping during deterministic transfer while still colliding on 58.3%–66.7% of layouts. Unsafe in-bound mean actions therefore remain even when stochastic sampling is removed.

7. **The causal claim must remain narrow.** These final values establish substantial residual variance at the frozen training budget. They do not show its trajectory during training, prove that it caused weak learning, or establish that a larger budget would reduce it.

The supported conclusion is:

> At the frozen training budget, every policy had approximately unit final Gaussian standard deviation in normalized action space. This quantitatively explains much of the ubiquitous clipping under stochastic evaluation, but it does not explain the large seed- and method-dependent differences in navigation competence.

### Interpretation boundary

- The summaries report variation across only five independent training seeds; they are descriptive estimates, not formal population guarantees.
- The paired design isolates the deployment-time effect of projection within each frozen checkpoint. It does not by itself isolate why projection-enabled training produced a projector-dependent nominal actor.
- High clearance alone is not evidence of useful control. For the high-penalty method, it coexists with near-total non-completion.
- No result is a reason to rerun or alter the frozen campaign.

## Secondary evaluation: deterministic core-layout transfer

The secondary evaluation uses the deterministic actor mean on 24 frozen layouts, once per checkpoint and projection mode. Values below are the mean and sample standard deviation across five independently trained checkpoints. Each checkpoint rate is computed over its 24 layouts.

### Method-level summary

| Training method | Projection | Return | Success | Collision | Minimum clearance | Intervention | Correction norm | Slack sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PPO baseline | Off | -3.243 ± 2.762 | 0.075 ± 0.168 | 0.050 ± 0.090 | 0.740 ± 0.368 | — | — | — |
| PPO baseline | On | -3.174 ± 2.830 | 0.075 ± 0.168 | 0.033 ± 0.075 | 0.743 ± 0.364 | 0.031 ± 0.066 | 0.014 ± 0.030 | 0.000028 ± 0.000062 |
| PPO high penalty | Off | -5.801 ± 0.243 | 0.000 ± 0.000 | 0.017 ± 0.037 | 0.860 ± 0.269 | — | — | — |
| PPO high penalty | On | -5.722 ± 0.225 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.861 ± 0.267 | 0.003 ± 0.007 | 0.000 ± 0.001 | 0.000001 ± 0.000002 |
| PPO trained with projection | Off | -2.980 ± 0.735 | 0.275 ± 0.037 | 0.633 ± 0.035 | 0.055 ± 0.016 | — | — | — |
| PPO trained with projection | On | 0.508 ± 0.877 | 0.333 ± 0.042 | 0.125 ± 0.066 | 0.154 ± 0.010 | 0.567 ± 0.017 | 0.400 ± 0.030 | 0.000661 ± 0.000050 |

### Verified observations from the method summary

1. **Baseline transfer competence is concentrated in a minority of training seeds.** Mean success is only 7.5%, with a 16.8-point across-seed standard deviation. Enabling projection does not change success and reduces mean collision by only 1.7 points.

2. **The high-penalty method does not transfer goal-reaching behavior.** It succeeds on none of the 120 checkpoint-layout combinations in either projection mode. Projection removes its small residual collision rate, but completion remains zero.

3. **Projection-trained PPO transfers some nominal goal-directed behavior, but broad competence is absent.** Without projection it succeeds on 27.5% ± 3.7% of layouts and collides on 63.3% ± 3.5%.

4. **Projection substantially improves transfer safety for projection-trained PPO but only modestly improves completion.** With projection, collision falls to 12.5% ± 6.6%, while success rises to 33.3% ± 4.2%. The remaining 54.2% of protected checkpoint-layout evaluations time out.

5. **Transfer requires greater projector involvement than the fixed training geometry.** For projection-trained PPO, intervention rises from 41.4% ± 2.5% of steps in the primary evaluation to 56.7% ± 1.7% in transfer. Mean correction norm rises from 0.208 ± 0.041 to 0.400 ± 0.030.

6. **The projection-trained transfer limitation is repeatable across seeds.** Its relatively small across-seed standard deviations show that the approximately one-third protected success rate is not caused by a single failed checkpoint.

7. **High clearance still does not imply useful navigation.** The high-penalty method has the largest mean clearance but zero success, reinforcing that it generally avoids completion rather than solving the navigation task.

### Paired projection effects

Each delta is projection enabled minus projection disabled for the same checkpoint and matched transfer layouts. Values are the mean and sample standard deviation of the five checkpoint-level paired effects.

| Training method | Return delta | Success delta | Collision delta | Clearance delta |
|---|---:|---:|---:|---:|
| PPO baseline | +0.069 ± 0.095 | 0.000 ± 0.000 | -0.017 ± 0.023 | +0.003 ± 0.007 |
| PPO high penalty | +0.079 ± 0.178 | 0.000 ± 0.000 | -0.017 ± 0.037 | +0.001 ± 0.002 |
| PPO trained with projection | +3.488 ± 0.605 | +0.058 ± 0.023 | -0.508 ± 0.095 | +0.099 ± 0.018 |

### Verified observations from the paired transfer analysis

1. **Deployment-time projection does not improve transfer success for either policy trained without projection.** Baseline and high-penalty PPO both have an exact mean success delta of zero across the five checkpoints.

2. **The small collision reductions for baseline and high-penalty PPO become timeouts rather than successes.** Both methods reduce collision by 1.7 percentage points while success remains unchanged.

3. **Projection has a large, repeatable transfer-safety effect for projection-trained PPO.** Collision falls by 50.8 ± 9.5 percentage points, clearance rises by 0.099 ± 0.018, and return rises by 3.488 ± 0.605.

4. **The transfer completion benefit remains modest.** Projection-trained PPO gains only 5.8 ± 2.3 percentage points of success. Because success, collision, and timeout are exhaustive, the 50.8-point collision reduction decomposes into approximately 5.8 points of additional success and 45.0 points of additional timeout.

5. **The primary and transfer results support the same qualitative separation.** Projection is effective at changing constraint outcomes when the nominal policy encounters obstacles, but it does not independently supply the missing goal-directed behavior needed for broad transfer.

### Checkpoint-level outcome variation

`S/C/T` denotes success, collision, and timeout counts out of 24 deterministic layouts. Intervention is the percentage of protected steps on which projection modified the bounded nominal action.

| Training method | Seed | Projection off S/C/T | Projection on S/C/T | Return off → on | Protected intervention |
|---|---:|---:|---:|---:|---:|
| PPO baseline | 1 | 0/1/23 | 0/0/24 | -3.274 → -3.096 | 0.6% |
| PPO baseline | 2 | 0/0/24 | 0/0/24 | -6.024 → -6.024 | 0.0% |
| PPO baseline | 3 | 9/5/10 | 9/4/11 | 1.153 → 1.322 | 15.0% |
| PPO baseline | 4 | 0/0/24 | 0/0/24 | -2.983 → -2.983 | 0.0% |
| PPO baseline | 5 | 0/0/24 | 0/0/24 | -5.087 → -5.087 | 0.0% |
| PPO high penalty | 1 | 0/2/22 | 0/0/24 | -6.003 → -5.605 | 1.5% |
| PPO high penalty | 2 | 0/0/24 | 0/0/24 | -5.960 → -5.960 | 0.0% |
| PPO high penalty | 3 | 0/0/24 | 0/0/24 | -5.487 → -5.488 | 0.1% |
| PPO high penalty | 4 | 0/0/24 | 0/0/24 | -5.589 → -5.589 | 0.0% |
| PPO high penalty | 5 | 0/0/24 | 0/0/24 | -5.967 → -5.967 | 0.0% |
| PPO trained with projection | 1 | 6/14/4 | 7/4/13 | -3.497 → -0.661 | 55.5% |
| PPO trained with projection | 2 | 6/15/3 | 8/5/11 | -3.249 → 0.337 | 54.6% |
| PPO trained with projection | 3 | 8/15/1 | 9/3/12 | -1.791 → 1.181 | 58.1% |
| PPO trained with projection | 4 | 6/16/2 | 7/1/16 | -3.584 → 0.132 | 58.5% |
| PPO trained with projection | 5 | 7/16/1 | 9/2/13 | -2.779 → 1.551 | 56.8% |

### Verified transfer checkpoint observations

1. **All baseline transfer success comes from seed 3.** It succeeds on 9 of 24 layouts in both projection modes; the other four baseline checkpoints succeed on none.

2. **Baseline seed 3 is consistently the exceptional baseline checkpoint across both evaluations.** It is also the only baseline checkpoint with strong primary success. This supports genuine learned competence in that checkpoint rather than an evaluation artifact.

3. **Projection does not add transfer successes to any baseline checkpoint.** It prevents one collision for seed 1 and one for seed 3; both become timeouts.

4. **High-penalty transfer failure is complete across all checkpoints.** Projection prevents two seed-1 collisions, but all 120 protected checkpoint-layout evaluations end in timeout.

5. **Every projection-trained checkpoint exhibits the same limited-transfer structure.** Unprotected success is 6–8 layouts, while protected success is 7–9. Projection reduces collisions by 10–15 layouts per checkpoint, but adds only 1–2 successes.

6. **The resulting timeout increase is present in every projection-trained checkpoint.** Prevented collisions become 8–14 additional timeouts per checkpoint.

7. **Projector dependence during transfer is highly repeatable.** Protected intervention rates occupy the narrow range 54.6%–58.5%, and all are substantially above the corresponding primary rates.

8. **The transfer gap is method-level rather than an isolated bad seed.** All five projection-trained checkpoints are strong on the protected training geometry and all five remain limited to 7–9 protected transfer successes.

### Transfer action-bound clipping diagnostic

Values are the mean and sample standard deviation across five checkpoints. Unlike the primary evaluation, transfer executes the deterministic actor mean on each of the 24 layouts.

| Training method | Projection | Any-component clipping | Speed clipping | Turn-rate clipping | Mean clipping norm |
|---|---|---:|---:|---:|---:|
| PPO baseline | Off | 0.246 ± 0.433 | 0.246 ± 0.433 | 0.000 ± 0.000 | 0.085 ± 0.166 |
| PPO baseline | On | 0.246 ± 0.433 | 0.246 ± 0.433 | 0.000 ± 0.000 | 0.085 ± 0.166 |
| PPO high penalty | Off | 0.335 ± 0.331 | 0.335 ± 0.331 | 0.000 ± 0.000 | 0.058 ± 0.069 |
| PPO high penalty | On | 0.335 ± 0.331 | 0.335 ± 0.331 | 0.000 ± 0.000 | 0.058 ± 0.069 |
| PPO trained with projection | Off | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| PPO trained with projection | On | 0.022 ± 0.049 | 0.022 ± 0.049 | 0.000 ± 0.000 | 0.000 ± 0.000 at displayed precision |

### Verified transfer clipping observations

1. **Deterministic transfer clips substantially less often than stochastic fixed-geometry evaluation.** Mean any-component clipping ranges from 0% to 33.5%, compared with approximately 59% to 72% in the primary evaluation. Policy mode and geometry both change between suites, so this comparison is diagnostic rather than causal.

2. **Every observed transfer clipping event is a speed-component event.** Turn-rate clipping is exactly zero for every method and deployment mode on the 24 deterministic layouts.

3. **The projection-trained deterministic actor stays within the normalized action bounds on all unprotected transfer trajectories.** Its disabled-mode clipping rate and clipping norm are both zero.

4. **Protected projection-trained trajectories introduce only minor speed clipping.** The 2.2% mean rate arises after projection changes the visited-state distribution; action bounding still occurs before projection at each step.

5. **Baseline and high-penalty clipping summaries are unchanged by projection at the displayed precision.** Their substantial across-seed standard deviations indicate checkpoint concentration that requires checkpoint-level inspection before attribution.

6. **The primary turn-rate clipping is consistent with an important stochastic-sampling contribution.** The deterministic transfer actor means never exceed the turn-rate bound, although the changed geometry prevents isolating sampling as the sole cause.

### Transfer clipping concentration by checkpoint

| Training method | Seed | Any clipping, off → on | Clipping norm, off → on |
|---|---:|---:|---:|
| PPO baseline | 1 | 0.000 → 0.000 | 0.000 → 0.000 |
| PPO baseline | 2 | 1.000 → 1.000 | 0.380 → 0.380 |
| PPO baseline | 3 | 0.000 → 0.000 | 0.000 → 0.000 |
| PPO baseline | 4 | 0.000 → 0.000 | 0.000 → 0.000 |
| PPO baseline | 5 | 0.230 → 0.230 | 0.045 → 0.045 |
| PPO high penalty | 1 | 0.041 → 0.041 | 0.001 → 0.001 |
| PPO high penalty | 2 | 0.719 → 0.719 | 0.136 → 0.136 |
| PPO high penalty | 3 | 0.000 → 0.000 | 0.000 → 0.000 |
| PPO high penalty | 4 | 0.633 → 0.633 | 0.130 → 0.130 |
| PPO high penalty | 5 | 0.280 → 0.280 | 0.022 → 0.022 |
| PPO trained with projection | 1 | 0.000 → 0.111 | 0.000 → 0.001 |
| PPO trained with projection | 2 | 0.000 → 0.000 | 0.000 → 0.000 |
| PPO trained with projection | 3 | 0.000 → 0.000 | 0.000 → 0.000 |
| PPO trained with projection | 4 | 0.000 → 0.000 | 0.000 → 0.000 |
| PPO trained with projection | 5 | 0.000 → 0.000 | 0.000 → 0.000 |

### Verified checkpoint-level clipping observations

1. **Baseline clipping is concentrated in seeds 2 and 5.** Seed 2 clips speed on every deterministic transfer step and has zero success, while seed 3 never clips and is the only competent baseline checkpoint.

2. **Clipping absence is not sufficient for competence.** Baseline seeds 1 and 4 also never clip yet succeed on no transfer layouts.

3. **High-penalty failure cannot be reduced to action saturation.** Its clipping ranges from 0% to 71.9% across seeds, but every checkpoint has zero transfer success.

4. **Projection-trained transfer collisions are produced by in-bound deterministic actions.** Every unprotected projection-trained checkpoint has zero clipping despite collision rates of 58.3%–66.7% across the 24 layouts.

5. **The only protected projection-trained clipping occurs in seed 1 and is numerically tiny.** Its 11.1% clipping rate has a mean norm of only 0.001; it cannot explain the method-wide transfer limitation shared by all five checkpoints.

6. **Deployment-time projection does not change baseline or high-penalty clipping at the reported precision.** Their safety changes therefore are not mediated by reducing action-bound saturation.

## Training diagnostics

### Aggregated training-curve table schema

The generated file `results\tables\fixed_training_geometry\training_curve_points.csv` passed structural inspection:

- Schema: `training_diagnostics_v1`
- Rows: 14,811
- Columns: schema version, method, display name, tag, step, across-seed mean, across-seed sample standard deviation, and seed count
- The displayed baseline episodic-return rows all have `seed_count = 5`, confirming complete seed participation in that initial segment.
- The noninteger step coordinates after the first point show that this is an aligned curve representation for method-level comparison, not a list of raw per-episode event steps.

The first twelve rows cover only PPO baseline episodic return from approximately step 800 to step 3,574. They show an early mean return between roughly -15.4 and -9.9 with appreciable across-seed variation, but they do not support any conclusion about convergence or whether the 51,200-transition budget was sufficient. Full tag coverage and step ranges must be established before selecting endpoint windows or trend comparisons.

### Training-curve coverage inventory

Every reported curve has `seed_min = seed_max = 5`; no method/tag curve loses a training seed.

| Training method | Episodic return | Rolling success/collision | Clipping | Projection diagnostics |
|---|---:|---:|---:|---:|
| PPO baseline | 200 points, through 50,976 | 1,467 each, through 50,976 | 50 points, through 51,200 | Not applicable |
| PPO high penalty | 200 points, through 50,688 | 1,322 each, through 50,688 | 50 points, through 51,200 | Not applicable |
| PPO trained with projection | 200 points, through 51,056 | 1,801 each, through 51,056 | 50 points, through 51,200 | Five tags × 50 points, through 51,200 |

Cumulative-collision curves begin at step 0 and contain 1,487, 1,342, and 1,802 points for baseline, high-penalty, and projection-trained PPO respectively. Their last steps match the final completed-episode diagnostics for each method.

Verified implications:

1. **The generated training diagnostics are complete for the intended comparison.** All curves include all five seeds and reach the end of the training campaign or the last episode completed immediately before it.

2. **The 50 clipping points correspond to the 50 rollout boundaries.** The five projection diagnostics have the same rollout-level coverage for projection-trained PPO.

3. **Projection tags appear only for projection-enabled training, as expected.** Their absence for baseline and high-penalty PPO is structural rather than missing evidence.

4. **The small differences in final episodic step are expected.** Episode summaries end at the final episode boundary before the fixed 51,200-transition cutoff, whereas rollout metrics reach exactly 51,200.

5. **No deeper schema investigation is needed.** Subsequent analysis is restricted to compact summaries that answer whether learning improved or plateaued: return, rolling success, rolling collision, clipping, and projection burden.

### Early-versus-late core training comparison

Values are descriptive means of the aligned five-seed method curves over the first and last 20% of each curve. `Final` is the last aligned curve point and is inherently noisier than a window mean.

| Training method | Metric | Early | Late | Late minus early | Final |
|---|---|---:|---:|---:|---:|
| PPO baseline | Action-bound clipping frequency | 0.550 | 0.653 | +0.103 | 0.640 |
| PPO baseline | Episodic return | -11.947 | -6.013 | +5.934 | -8.552 |
| PPO baseline | Rolling collision rate | 0.477 | 0.119 | -0.358 | 0.080 |
| PPO baseline | Rolling success rate | 0.018 | 0.100 | +0.081 | 0.150 |
| PPO high penalty | Action-bound clipping frequency | 0.567 | 0.720 | +0.153 | 0.717 |
| PPO high penalty | Episodic return | -25.871 | -16.752 | +9.119 | -12.414 |
| PPO high penalty | Rolling collision rate | 0.415 | 0.144 | -0.271 | 0.160 |
| PPO high penalty | Rolling success rate | 0.008 | 0.000 | -0.008 | 0.000 |
| PPO trained with projection | Action-bound clipping frequency | 0.539 | 0.587 | +0.048 | 0.588 |
| PPO trained with projection | Episodic return | -7.967 | 8.684 | +16.651 | 11.377 |
| PPO trained with projection | Rolling collision rate | 0.006 | 0.040 | +0.034 | 0.010 |
| PPO trained with projection | Rolling success rate | 0.094 | 0.864 | +0.770 | 0.930 |

Verified implications:

1. **Baseline PPO learns partial competence but remains mostly non-completing.** Its late rolling collision rate is 11.9% and late success is 10.0%, implying approximately 78.1% timeout. The return and outcome curves improve substantially, but neither the late-window mean nor the noisier final point establishes convergence. Its final 15.0% success closely matches the 15.2% unprotected primary-evaluation mean.

2. **High-penalty PPO learns conservative collision avoidance rather than navigation competence.** Collision falls by 27.1 percentage points and return improves, while late and final success are both zero. The implied late timeout rate is approximately 85.6%. Because this method uses a different reward scale, its return magnitude must not be compared directly with the other methods; its within-method improvement is consistent with avoiding costly collisions without reaching the goal.

3. **Projection-trained PPO clearly learns the protected fixed-geometry task.** Late success reaches 86.4%, final success reaches 93.0%, and late return is positive. Its final training success closely matches the 93.6% protected primary-evaluation mean. This establishes genuine composite-controller learning rather than an evaluation-only effect.

4. **Low protected training collision does not imply a safe nominal actor.** The projector is active during projection-trained learning, while projection-disabled evaluation shows 81.8% mean collision. The training curve describes the composite controller.

5. **Improvement does not arise from reduced action-bound saturation.** Clipping rises for all three methods, including by 4.8 points for projection-trained PPO while its success rises by 77.0 points. High-penalty PPO has the largest late clipping and zero success. Clipping is therefore a shared diagnostic symptom, not a sufficient performance explanation.

6. **There is no single budget-sufficiency verdict.** The 51,200-transition budget is sufficient for strong protected fixed-geometry performance from projection-trained PPO, plausibly insufficient for reliable baseline learning, and produces a high-penalty solution more consistent with reward-driven conservative non-completion than with demonstrated budget insufficiency. These frozen curves do not prove what a longer budget would do.

7. **A narrow seed-level tail check is required to interpret the method average.** Because baseline performance is strongly seed-dependent, the comparison below separates continued improvement in the exceptional checkpoint from the behavior of the failed checkpoints. No broader general curve mining is required.

### Seed-level late-training return and success

Each value is the mean of raw episode events within the indicated fraction of that seed's final recorded training span. The deltas compare two adjacent windows descriptively; they are not formal convergence tests and do not account for episode-count differences, autocorrelation, or sampling noise.

| Training method | Seed | Return 80–90% | Return 90–100% | Return delta | Success 80–90% | Success 90–100% | Success delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO baseline | 1 | -5.521 | -5.951 | -0.430 | 0.074 | 0.036 | -0.038 |
| PPO baseline | 2 | -8.935 | -8.514 | +0.421 | 0.000 | 0.000 | 0.000 |
| PPO baseline | 3 | -2.441 | 2.138 | +4.579 | 0.343 | 0.568 | +0.225 |
| PPO baseline | 4 | -6.964 | -5.279 | +1.685 | 0.000 | 0.036 | +0.036 |
| PPO baseline | 5 | -8.343 | -8.858 | -0.515 | 0.000 | 0.000 | 0.000 |
| PPO high penalty | 1 | -19.594 | -12.850 | +6.744 | 0.000 | 0.000 | 0.000 |
| PPO high penalty | 2 | -14.082 | -11.641 | +2.441 | 0.000 | 0.000 | 0.000 |
| PPO high penalty | 3 | -16.903 | -15.992 | +0.911 | 0.000 | 0.000 | 0.000 |
| PPO high penalty | 4 | -13.745 | -21.838 | -8.092 | 0.000 | 0.000 | 0.000 |
| PPO high penalty | 5 | -14.349 | -14.364 | -0.015 | 0.000 | 0.000 | 0.000 |
| PPO trained with projection | 1 | 4.514 | 7.657 | +3.143 | 0.638 | 0.804 | +0.166 |
| PPO trained with projection | 2 | 7.504 | 8.521 | +1.017 | 0.818 | 0.873 | +0.055 |
| PPO trained with projection | 3 | 7.812 | 10.054 | +2.242 | 0.839 | 0.937 | +0.097 |
| PPO trained with projection | 4 | 10.458 | 10.472 | +0.013 | 0.966 | 0.968 | +0.002 |
| PPO trained with projection | 5 | 8.129 | 10.027 | +1.897 | 0.870 | 0.951 | +0.080 |

Verified implications:

1. **Baseline PPO does not approach a reliable solution uniformly across seeds.** Seed 3 improves strongly in both return and success, seed 4 shows only weak late emergence, seeds 1 and 5 regress, and seed 2 remains unsuccessful.

2. **The method-average baseline improvement is driven principally by seed 3.** Its 56.8% success in the final training decile is coherent with its exceptional 51% unprotected primary-evaluation success. This strengthens the interpretation that seed 3 learned genuinely useful behavior rather than benefiting from an evaluation anomaly.

3. **A larger budget may help some baseline runs, but budget alone is not established as the remedy for failed seeds.** Seed 3 was still improving at the cutoff, yet seeds 1, 2, and 5 show no positive late success trend. A controlled budget study could test the idea, but this remains a private post-study note excluded from the report.

4. **High-penalty PPO exhibits no late goal learning in any checkpoint.** Success is exactly zero in both late windows for all five seeds. Return changes are heterogeneous, including a large seed-4 decline, and cannot be interpreted as navigation progress. Together with the aggregate collision-to-timeout shift, this is more consistent with conservative reward optimization than with demonstrated budget insufficiency.

5. **Projection-trained PPO is consistently competent and still improving in four checkpoints.** Seeds 1, 2, 3, and 5 gain both return and success; seed 4 is effectively at a ceiling near 97% success. The method is not universally plateaued, but the frozen budget was already sufficient for a strong composite controller.

6. **Additional training targets differ by method.** For projection-trained PPO, more transitions might improve already-strong protected performance but would not directly address nominal-actor dependence or transfer. For baseline PPO, a budget-only experiment could be informative; for high-penalty PPO, the frozen evidence gives no sign that budget alone would create goal completion. These are private experiment-design observations, not report claims.

7. **General training-convergence inspection is complete.** The remaining numerical training analysis is limited to the projector's intervention, correction, slack, and solver-failure burden; trajectory inspection then has greater explanatory value than further curve subdivision.

### Training-time projection burden

Values are descriptive means over the first and last 20% of the aligned five-seed projection-training curves. `Final` is the last aligned rollout point.

| Projection metric | Early | Late | Late minus early | Final |
|---|---:|---:|---:|---:|
| Correction norm | 0.073917 | 0.209333 | +0.135415 | 0.216806 |
| Maximum correction norm | 1.860338 | 1.996664 | +0.136326 | 1.952593 |
| Intervention frequency | 0.138672 | 0.393111 | +0.254439 | 0.411133 |
| Maximum slack | 0.014498 | 0.015910 | +0.001412 | 0.015722 |
| Slack sum | 0.000245 | 0.000578 | +0.000333 | 0.000566 |

The training records contain 250 solver-failure summaries, corresponding to 50 rollout boundaries for each of five projection-trained seeds. Their summed failure count is zero.

Verified implications:

1. **Projection burden increases materially during successful learning.** Mean intervention frequency rises from 13.9% to 39.3% and finishes at 41.1%; mean correction norm rises from 0.074 to 0.209 and finishes at 0.217. The policy does not learn away its use of the projector on the frozen training geometry.

2. **The increase is primarily in intervention frequency and average correction, not in increasingly severe rare extremes.** Maximum correction remains near 1.9–2.0, while maximum slack changes only slightly.

3. **The result is consistent with actor–projector co-adaptation, but does not prove intentional exploitation.** As the composite controller becomes more successful, it may visit states closer to constraints or use more direct paths. Because intervention is state-distribution-dependent, its increase cannot by itself distinguish policy behavior from changed visitation.

4. **Training-end and frozen protected evaluation burdens agree closely.** Final training versus primary stochastic evaluation values are 0.411 versus 0.414 for intervention frequency, 0.217 versus 0.208 for correction norm, and 0.000566 versus 0.000521 for slack sum. The protected evaluation therefore reproduces the burden seen at the end of training.

5. **Numerical reliability is excellent.** There are no solver failures across all 250 recorded projection-training rollouts. This establishes solver robustness, not absolute safety: protected collisions and nonzero slack still occur.

6. **The combined evidence establishes substantial composite-controller dependence.** Growing training burden, closely matched protected evaluation burden, and 81.8% collision when projection is removed show that projection is an integral part of the learned system rather than a rarely used emergency layer.

7. **Numerical training diagnostics are complete.** Representative trajectories are now the appropriate evidence for determining whether the projector corrects direct unsafe paths, guides motion along constraints, or converts unsafe motion into stalled behavior.

## Representative trajectories

### Primary fixed-geometry selection provenance

The generated primary selection manifest contains six trajectories: projection off and on for each of the three training methods. Every selection uses training seed 1, layout `fixed_training_geometry`, evaluation seed 10000, and episode 0.

| Training method | Projection | Training seed | Evaluation seed | Episode |
|---|---|---:|---:|---:|
| PPO baseline | Off | 1 | 10000 | 0 |
| PPO baseline | On | 1 | 10000 | 0 |
| PPO high penalty | Off | 1 | 10000 | 0 |
| PPO high penalty | On | 1 | 10000 | 0 |
| PPO trained with projection | Off | 1 | 10000 | 0 |
| PPO trained with projection | On | 1 | 10000 | 0 |

The disabled/enabled pair for each method has the same checkpoint SHA-256, confirming that the plotted checkpoint is held fixed within the projection comparison. Each row also records the exact trajectory archive used.

Verified implications and interpretation rules:

1. **The figure provides a controlled, reproducible illustration.** The same checkpoint index, evaluation seed, episode index, and fixed geometry are used systematically rather than selecting different favorable-looking cases for each condition.

2. **“Representative” means a fixed example, not a statistical summary.** One stochastic episode from seed 1 cannot represent five independently trained checkpoints or 100 evaluation episodes, particularly for seed-sensitive baseline PPO. Quantitative tables remain the performance evidence.

3. **Only projection off versus on within a method is checkpoint-paired.** Cross-method panels use different trained checkpoints and support qualitative comparison only.

4. **Matched evaluation seeds do not make the trajectories exact pointwise counterfactuals.** Once projection changes an executed action, the next state changes; subsequent policy means and sampled actions can therefore diverge even under the same seed.

5. **The primary figure cannot establish transfer behavior.** It covers only the fixed training geometry; the separately generated transfer figure must be inspected for geometry variation.

6. **Visual inspection must illustrate mechanisms already established numerically.** Relevant questions are whether projection creates a useful detour or boundary-following path, merely turns collision into timeout, and whether the projection-trained nominal path is unsafe while the protected path reaches the goal. The plot must not be treated as additional frequency evidence.

### Primary representative-trajectory visual review

The generated artifact is a one-page, single-axis overlay of all six selected trajectories, not six separate panels. It renders without corruption, clipping, or unreadable glyphs, but it is not yet publication-ready.

Verified visual observations:

1. **The legend obstructs substantive evidence.** It covers much of the upper-right plot region, overlaps the goal marker, and obscures sections of several trajectories. This makes endpoints and path differences harder to verify.

2. **Six overlaid paths are difficult to follow.** Several trajectories coincide near the common start and some remain hidden beneath other traces or the legend. The figure does not provide start markers, terminal-outcome markers, intervention locations, or endpoint labels.

3. **The visible protected baseline and high-penalty examples do not display clear goal-directed completion.** The baseline-on path makes a large lower detour, while the high-penalty-on path loops near the start. This is qualitatively consistent with the quantitative timeout-dominated results, but exact outcomes must be read from the episode table rather than inferred from line endpoints.

4. **The projection-trained pair visibly separates near the central constraints.** The unprotected path terminates near the lower central obstacle, whereas the protected path follows a much longer route around the obstacle field. Because the plot has no terminal labels and the legend hides part of the route, it cannot establish the selected episode's outcome by itself.

5. **The plot is illustrative only.** It supports qualitative mechanism review but adds no frequency evidence beyond the audited tables. Exact selected-episode outcomes and metrics must be joined from `evaluation_episode_results.csv`.

6. **The PDF metadata is incomplete.** It identifies Matplotlib as creator and producer but contains no `Author` field. Before final commit, generated PDFs must identify Salvador Tenorio as author and contain no OpenAI or ChatGPT attribution.

7. **Presentation remediation is warranted without changing evidence or selection.** A publication version should move the legend outside the data region and should strongly consider method-separated panels plus explicit start and terminal-outcome markers. Any redesign must preserve the same frozen trajectory selections and numerical data.

This direct review corrects the earlier assumption that the selection manifest represented six separate panels. The manifest establishes provenance; only the PDF establishes the actual visual design.

### Exact primary selected-episode outcomes

The six selection rows were joined one-to-one to `evaluation_episode_results.csv` using method, training seed, projection mode, layout, evaluation seed, checkpoint hash, and episode index. Neither table contains duplicate join keys, all six rows matched, and no row was unmatched.

| Training method | Projection | Outcome | Length | Return | Final goal distance | Minimum clearance | Clipping rate | Intervention rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PPO baseline | Off | Timeout | 200 | -6.393482 | 1.884987 | 0.660428 | 0.540 | 0.000 |
| PPO baseline | On | Timeout | 200 | -6.393482 | 1.884987 | 0.660428 | 0.540 | 0.000 |
| PPO high penalty | Off | Timeout | 200 | -11.106348 | 4.082473 | 0.843644 | 0.650 | 0.000 |
| PPO high penalty | On | Timeout | 200 | -11.106348 | 4.082473 | 0.843644 | 0.650 | 0.000 |
| PPO trained with projection | Off | Collision | 30 | -9.107260 | 1.979758 | -0.001347 | 0.700 | 0.000 |
| PPO trained with projection | On | Timeout | 200 | -6.411670 | 1.994247 | 0.013253 | 0.640 | 0.265 |

The protected projection-trained episode contains 53 interventions, mean correction norm 0.216007, maximum correction norm 2.085313, mean summed slack 0.000710, maximum slack 0.011969, and zero solver failures.

Verified implications:

1. **All six selected primary episodes are failures.** The baseline and high-penalty pairs are timeouts in both modes. The projection-trained pair changes from collision to timeout.

2. **The baseline and high-penalty off/on examples are exactly identical because the projector never intervenes.** Their enabled correction values are only floating-point zero. Overlaying both traces therefore advertises six lines while only four distinct paths are visible.

3. **The protected projection-trained example illustrates collision prevention, not typical protected performance.** It times out even though projection-trained seed 1 succeeds in 78% of protected primary episodes and the five-checkpoint mean is 93.6%. It must not be described as statistically representative.

4. **The prespecified selection is defensible as anti-cherry-picking evidence.** Its purpose is to show a reproducible mechanism example. A publication caption should call it a *prespecified illustrative episode* rather than a representative outcome.

### Transfer selection provenance and exact outcomes

The transfer selection manifest also joins one-to-one without duplicate or unmatched keys. All six rows use training seed 1, deterministic evaluation, layout `triple_mild_slalom_upper_first`, evaluation seed 1018, and episode 18.

| Training method | Projection | Outcome | Length | Return | Final goal distance | Minimum clearance | Clipping rate | Intervention rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PPO baseline | Off | Timeout | 200 | -0.032812 | 0.868776 | 0.501740 | 0.000 | 0.000 |
| PPO baseline | On | Timeout | 200 | -0.032812 | 0.868776 | 0.501740 | 0.000 | 0.000 |
| PPO high penalty | Off | Timeout | 200 | -6.761259 | 3.332893 | 0.535549 | 0.200 | 0.000 |
| PPO high penalty | On | Timeout | 200 | -6.761259 | 3.332893 | 0.535549 | 0.200 | 0.000 |
| PPO trained with projection | Off | Collision | 10 | -9.247317 | 3.052317 | -0.019058 | 0.000 | 0.000 |
| PPO trained with projection | On | Timeout | 200 | -3.235855 | 2.571467 | 0.059369 | 0.895 | 1.000 |

The protected projection-trained transfer episode contains 200 interventions, mean correction norm 0.940610, maximum correction norm 1.021754, mean summed slack 0.001644, maximum slack 0.001875, and zero solver failures.

Verified implications:

1. **The transfer example is a particularly clear collision-to-timeout case.** The unprotected projection-trained actor collides after ten steps. Protection prevents that collision but intervenes on every one of 200 steps and still does not reach the goal.

2. **The baseline and high-penalty off/on paths are again exactly identical.** The projector never intervenes, so the paired deployment conditions provide no distinct path in this selected layout.

3. **The protected clipping increase is coherent with state divergence.** Once projection changes the first executed action, later states and nominal actions differ; deterministic evaluation does not require equal later clipping between modes.

4. **This single episode supports mechanism interpretation only.** Transfer rates remain established by the 720-episode aggregate, not this layout.

### Transfer representative-trajectory visual review

The transfer PDF renders cleanly but has the same structural limitations as the primary figure: a single six-line overlay, obscured goal and upper paths, exact off/on overlaps, no start or terminal markers, no outcome labels, and the raw layout identifier in the title. Its underlying example is scientifically useful, but the present rendering is not publication-ready.

The recommended redesign is three method panels with shared axes. Each panel should show the off/on pair, obstacles, start, goal, terminal markers, outcome and episode length, with the legend outside the data region. The frozen trajectory selections and exact numerical evidence must remain unchanged.

## Complete generated-figure quality audit

The uploaded result bundle contained 74 files, including all tables, audits, and 44 one-page PDFs: 22 for fixed training geometry and 22 for core-layout transfer. Both result-build audits and both figure-build audits report `PASS`; the complete 3,000 primary and 720 transfer episode rows are represented, all expected 30 CSV shards per suite were selected, and solver failures remain zero. All PDFs render cleanly as vector graphics with no corruption, clipped panels, blank pages, or missing visible glyphs.

This establishes technical build validity. It does not establish publication readiness.

### Blocking presentation and archival issues

1. **PDF metadata fails the authorship requirement.** All 44 PDFs omit `Author`, `Title`, and `Subject`. Creator and producer identify Matplotlib only. Every final PDF must name Salvador Tenorio as author and contain no OpenAI or ChatGPT attribution.

2. **All PDFs embed Type 3 DejaVu Sans fonts without Unicode mapping.** They render correctly, but some publishers reject Type 3 fonts. Regeneration should set Matplotlib's `pdf.fonttype` to 42.

3. **The evaluation bars hide the experimental unit and pairing.** Six same-color bars with long rotated labels conceal the five checkpoint replicates and the checkpoint-paired projection-off/on design. The error bars are not labeled as across-seed sample standard deviations.

4. **Rate uncertainty extends outside the feasible interval.** Mean plus or minus standard deviation produces negative tails for several bounded rates. This is mathematically possible for a descriptive error bar but visually suggests impossible rates. Paired seed points and connecting lines are more faithful for five replicates.

5. **Timeout is absent from the outcome figures.** Because collision-to-timeout conversion is a central result, success and collision alone are incomplete as the primary visual summary.

6. **Several metric meanings and units are not self-explanatory.** Clearance figures must state the distance unit and explain that negative clearance indicates overlap. Correction and slack figures need their action-space or constraint-scale interpretation.

7. **The training-return panel can mislead across methods.** High-penalty PPO was trained under a different collision penalty, so absolute return levels are not directly comparable. Any retained panel requires an explicit caveat or within-method framing.

8. **Cumulative collision count is exposure-dependent.** It varies with episode length and completed-episode count per transition. Rolling collision rate is the more interpretable main diagnostic.

### Duplicate generated evidence

The four generated training CSV files are byte-for-byte identical between the fixed-geometry and transfer table directories:

- `training_scalar_events.csv`
- `training_episode_diagnostics.csv`
- `training_rollout_diagnostics.csv`
- `training_curve_points.csv`

All ten `training_*.pdf` files in the transfer figure directory are pixel-identical to their fixed-geometry counterparts. They arise from the same 15 training runs and are not transfer-specific evidence. Committing both copies would add approximately 30 MB of redundant result content and imply a suite distinction that does not exist.

Training diagnostics should therefore be generated once in a canonical location or retained only with the primary fixed-geometry result build. The transfer build should contain only transfer evaluation and transfer trajectory outputs, with its audit explicitly recording that shared training diagnostics were omitted by design.

### Paper, supplement, and omission decisions

The current PDFs are a complete diagnostic output set. They should be curated as follows after regeneration and redesign:

**Main paper candidates**

- A paired seed-level outcome figure covering success, collision, and timeout for both evaluation suites.
- A combined training learning figure using rolling success and rolling collision; return may be included only with the reward-scale caveat.
- A projection-dependence figure combining intervention burden during training and protected evaluation.
- Redesigned prespecified trajectory small multiples for fixed geometry and transfer.

**Supplementary diagnostics**

- Return, clearance, action-bound clipping rate and norm, mean correction, and mean slack.
- Cumulative collisions only if explicitly framed as exposure rather than rate.
- Maximum correction and slack values are better reported in a table unless rare extrema are directly discussed.

**Do not publish as currently rendered**

- Both single-axis trajectory overlays.
- Standalone maximum-correction and maximum-slack charts unless required by a specific claim.
- Duplicate transfer training tables and figures.

### Required remediation before result commit

The numerical tables and raw evidence must remain unchanged. The plotting and result-layout code should receive a narrow, presentation-only revision that:

1. writes Salvador Tenorio into PDF metadata and uses TrueType-compatible PDF fonts;
2. exposes the five checkpoint replicates and off/on pairing in evaluation plots;
3. includes timeout in the outcome visualization;
4. redesigns the frozen trajectory examples as annotated method panels without changing their selections;
5. emits training diagnostics only once; and
6. records these presentation choices in the generated figure audits.

The result directories should then be regenerated from the already frozen evidence and visually re-audited. No retraining, reevaluation, protocol change, or selection change is warranted.

## Presentation-only remediation patch

`Predictive_Action_Projection_Figure_Remediation.patch` was prepared against frozen source commit `ba64926aed98b08b7b285266cf85989d466f9f1c`. It changes only:

- `analysis/plot_projection_results.py`
- `tests/test_result_aggregation.py`

It does not change aggregation, evaluation, environments, PPO training, checkpoints, protocols, raw evidence, numerical tables, or selected trajectory keys.

The patch implements the following presentation and repository-hygiene corrections:

1. PDF metadata includes Author `Salvador Tenorio`, a descriptive title, and the study subject.
2. Matplotlib PDF and PostScript font types are set to 42, producing embedded CID TrueType fonts with Unicode mapping rather than Type 3 fonts.
3. Evaluation figures show all five checkpoint values, connect projection off/on values within the same training seed, and mark the across-checkpoint mean separately.
4. Timeout rate is derived as `1 - success - collision` and emitted as `evaluation_timeout_rate.pdf`.
5. Bounded rate plots use the feasible display interval without mean-plus/minus-standard-deviation tails outside `[0,1]`.
6. The prespecified trajectory examples are drawn as three method panels with shared axes, start and goal markers, distinct terminal-outcome markers, off/on outcome and length text, protected intervention rate, coincident-path disclosure, a human-readable layout title, and an external legend.
7. Trajectory outcomes are accepted only when success, collision, or truncated timeout forms one valid exhaustive terminal state. Ambiguous or technical termination raises an error rather than being silently labeled timeout.
8. `--skip-training-diagnostics` supports an evaluation-only secondary build. Its audit records `artifact_scope = evaluation_only` and the intentional omission, and the command refuses stale shared training CSVs rather than leaving hidden duplicates.
9. Tests cover PDF metadata/font structure, evaluation-only audit scope, valid outcome classification, and rejection of unclassified terminal states.

### Patch validation

- Unified patch preflight and round-trip application: pass.
- Python compilation of both changed files: pass.
- Synthetic complete-protocol figure build with training diagnostics intentionally disabled: pass.
- Synthetic output: 13 PDFs and 15 total generated artifacts, with no `training_*.pdf` files.
- PDF metadata: Author, Title, and Subject present.
- PDF font inspection: embedded CID TrueType with Unicode mapping; no Type 3 font.
- Actual fixed-geometry checkpoint table: all 12 evaluation figures generated successfully, including timeout.
- Actual transfer checkpoint table: all 12 evaluation figures generated successfully, including timeout.
- Visual review of the actual paired success, collision, timeout, and intervention plots: pass for legibility and preservation of checkpoint pairing.
- Exhaustive outcome validation over all frozen episode rows: 3,000 primary and 720 transfer rows classified without ambiguity.
- Independent blocker-only source and patch review: pass; no remaining blocker found.

The patch SHA-256 is:

```text
f7b5ba074c2844468a764c31d411afb300050b28a62b486aef108773815051ae
```

Actual trajectory rendering still requires the local NPZ archives, which were deliberately excluded from the uploaded result bundle. Therefore the patched trajectory figures must be regenerated locally and uploaded for final visual QA before any result commit.

## Complete-file handoff

At the user's request, the reviewed patch was also materialized as a complete two-file replacement package:

```text
Predictive_Action_Projection_Figure_Remediation_Full_Files.zip
    analysis/plot_projection_results.py
    tests/test_result_aggregation.py
```

The archive contains no other members and preserves the repository-relative paths. Both extracted files are byte-identical to the versions produced by the reviewed patch and compile successfully. A separate independent verification found no extra or unsafe paths and no blocker.

SHA-256 identities:

```text
816349aab57da01e97990ab512ce76ce2e8804766ee3b3339c1c450655c77caf  analysis/plot_projection_results.py
3b360171351b686f3fa0b7de03691e7948e893bd865b485c3a96c8283100604b  tests/test_result_aggregation.py
aa583ff244f101c2f7ddea51da8c9636526bc63be2a461e1e36c8d8896f37c06  Predictive_Action_Projection_Figure_Remediation_Full_Files.zip
```

The complete-file package is now the preferred local handoff. It changes the same two source-controlled files as the patch and does not contain or modify any evidence or generated result artifact.

### Local installation check

The two complete files were installed on the Windows analysis branch. The command

```bat
git diff --check -- analysis\plot_projection_results.py tests\test_result_aggregation.py
```

completed with only Git's expected LF-to-CRLF working-copy notices. These notices reflect the repository's Windows line-ending conversion configuration; they are not whitespace errors, content changes, or test failures. No Git configuration change or manual line-ending rewrite is warranted.

### Targeted test result

The complete plotting and aggregation test module was run after local installation:

```bat
python -m pytest tests\test_result_aggregation.py -q
```

Result: `19 passed`. The revised plotting behavior, metadata checks, evaluation-only audit behavior, and trajectory outcome validation therefore pass in the actual project environment. Full-suite regression testing remains required before result regeneration.

### Full-suite native abort and test isolation

The first complete-suite run completed 47 cases and then terminated the Python process during `test_result_figure_pdf_metadata_and_font_type`, inside Matplotlib's PDF draw path. This is not evidence of a plotting-result defect:

- the same metadata/font test passed as part of the isolated `19 passed` module run;
- `save_figure` already closes the figure after saving;
- the failing test is the suite's first actual PDF render after earlier modules have loaded and exercised PyTorch and other native numerical libraries; and
- the real figure-building commands run in fresh Python processes and do not import PyTorch.

The process-dependent pattern matches the current upstream-documented Windows OpenMP-runtime conflict in which pip PyTorch and conda numerical/plotting packages can load competing `libiomp5md.dll` copies before Matplotlib renders: <https://github.com/pytorch/pytorch/issues/191367>. The exact abort message did not expose the OpenMP diagnostic line, so this is recorded as the leading native-runtime diagnosis rather than a numerical-study finding.

The robust remediation is test-only. The metadata/font smoke test now launches a clean Python subprocess, generates the PDF through the real `save_figure` function, and inspects the resulting bytes in the parent process. This mirrors the standalone plotting workflow, preserves every metadata/font assertion, and converts any child native failure into an ordinary pytest failure with captured output. `analysis/plot_projection_results.py` is unchanged. No unsafe `KMP_DUPLICATE_LIB_OK` workaround or environment mutation is used.

Updated handoff identities:

```text
816349aab57da01e97990ab512ce76ce2e8804766ee3b3339c1c450655c77caf  analysis/plot_projection_results.py
f6a2502d2c0ea352a2a74fdc47d303afe96f878daddb9d9194af595a1ef5d1cc  tests/test_result_aggregation.py
309fe817c7bc3009d3c423b7e4687e6ca278a153d69d897eea0d31ca4d8b7c07  Predictive_Action_Projection_Figure_Remediation_Full_Files.zip
```

### Isolated-test validation

After installing the revised test file, the targeted plotting and aggregation module was rerun:

```bat
python -m pytest tests\test_result_aggregation.py -q
```

Result: `19 passed`. The subprocess-isolated PDF metadata/font test therefore passes in the actual Windows project environment.

### Complete-suite validation

The complete repository suite was then rerun:

```bat
python -m pytest -q
```

Result: `65 passed`. The plotting remediation and test isolation now pass both targeted and full-suite regression testing. Result regeneration is cleared to proceed, beginning with the primary fixed-training-geometry figure build.

### Primary figure-build preflight refusal

The first primary regeneration command stopped before writing any artifact with:

```text
FileExistsError: Result figure directory already exists and is not empty: results\figures\fixed_training_geometry
```

This is the revised builder's intended clean-output safeguard, not a plotting or evidence failure. The directory contains only the superseded generated primary figures; the source tables, frozen CSV/NPZ evidence, checkpoints, and code are outside this exact target. Remove only `results\figures\fixed_training_geometry`, allow the builder to recreate it, and do not touch `results\tables\fixed_training_geometry`.

The superseded primary figure directory was then removed as instructed. No evidence, source table, checkpoint, calibration artifact, or source file was removed. The clean primary figure target is ready for regeneration.

### Primary figure regeneration

The clean fixed-training-geometry figure build completed successfully. It generated:

- 12 evaluation PDFs, now including `evaluation_timeout_rate.pdf`;
- 10 training-diagnostic PDFs;
- the redesigned prespecified trajectory PDF;
- five supporting CSV files; and
- `figure_build_audit.json`.

The command reported no skipped or failed artifact. A compact post-build audit remains required to verify the audit scope, expected PDF count, metadata, embedded font structure, and absence of Type 3 fonts before proceeding to transfer outputs.

### Primary post-build audit

The regenerated primary output passed its compact structural audit:

```text
PASS: primary audit, 23 PDFs, 28 generated artifacts, metadata/fonts valid, no skips
```

This verifies `status = PASS`, `artifact_scope = evaluation_and_training`, inclusion of training diagnostics, 23 PDFs, 28 generated artifacts recorded before the audit itself, the timeout-rate figure, Salvador Tenorio author metadata, title and subject metadata, embedded Type 0/TrueType-compatible fonts, no Type 3 fonts, and no skipped artifact. Primary regeneration is structurally complete; visual QA remains a later explicit step.

### Transfer clean-output preparation

The superseded `results\figures\core_layout_transfer` directory was removed. This affected only generated transfer figures. Transfer tables and frozen transfer evidence remained untouched.

Before the new evaluation-only transfer build, four old shared training-diagnostic CSVs must also be removed from `results\tables\core_layout_transfer`. They are duplicated training outputs already retained authoritatively under `results\tables\fixed_training_geometry`; their presence is intentionally rejected by the revised evaluation-only builder.

The four duplicated transfer training-diagnostic CSVs were removed successfully. The authoritative training diagnostics remain under `results\tables\fixed_training_geometry`. The transfer figure directory is clean, its numerical aggregation tables remain intact, and the transfer suite is ready for an evaluation-only figure build.

At this point, no research evidence is being changed. The current task is only to regenerate publication-ready transfer presentation artifacts without duplicating shared training outputs.

### Transfer figure regeneration

The evaluation-only core-layout-transfer build completed successfully. It generated 12 evaluation PDFs, including the new timeout-rate figure, the redesigned prespecified trajectory PDF, the trajectory-selection CSV, and `figure_build_audit.json`. It emitted no transfer training PDF or training CSV, as intended.

A compact structural audit remains to verify the evaluation-only scope, intentional training-diagnostic omission, artifact counts, metadata, embedded fonts, absence of Type 3 fonts, absence of duplicated training outputs, and absence of skipped artifacts.

### Transfer post-build audit

The regenerated transfer output passed its compact structural audit:

```text
PASS: transfer audit, 13 PDFs, 14 generated artifacts, evaluation-only, metadata/fonts valid, no duplicates or skips
```

This verifies `status = PASS`, `artifact_scope = evaluation_only`, intentional omission of shared training diagnostics, 13 PDFs, 14 generated artifacts recorded before the audit itself, the timeout-rate figure, Salvador Tenorio author metadata, title and subject metadata, embedded Type 0/TrueType-compatible fonts, no Type 3 fonts, no duplicated transfer training artifacts, and no skipped artifact.

Both regenerated suites are now structurally valid. The next phase is direct visual QA of all 36 PDFs, with the tables included so visual claims can be checked against their numerical sources. No further result regeneration is indicated unless that review finds a concrete presentation defect.

## Regenerated-figure visual and numerical QA

The packaged regenerated result set was rendered and reviewed against its source CSV tables. Numerical integrity passed:

- all checkpoint markers and across-checkpoint means match `checkpoint_summary.csv`;
- paired lines join projection-off/on values for the same training seed;
- timeout rates match the mutually exclusive truncated-episode rates;
- all training curves reproduce their five-seed source diagnostics;
- both six-row trajectory selections join one-to-one to exact episode evidence; and
- trajectory outcomes, lengths, intervention annotations, geometry, and terminal markers are correct.

The trajectory redesign passes visual review in both suites. PDF rendering, legends, titles, data marks, and ordinary paired evaluation figures are otherwise legible and uncorrupted.

The figure set is not yet ready to commit because direct visual review found three narrow presentation blockers affecting 13 of 36 PDFs:

1. **Ten projection-only evaluation figures clip the rightmost method label.** In each suite, intervention, correction, maximum correction, slack, and maximum slack truncate the final glyph of `PPO trained with projection` at the right page edge. The plot needs a small horizontal margin or equivalent tick-label accommodation.
2. **Two bounded training-rate uncertainty bands enter impossible regions.** `training_rolling_collision_rate.pdf` reaches approximately `-0.062`; `training_rolling_success_rate.pdf` reaches approximately `-0.161` and `1.044`. The displayed mean-plus/minus-one-sample-SD bands must be clipped to `[0,1]` while leaving the mean curves unchanged.
3. **`training_return.pdf` still lacks the reward-scale caveat.** High-penalty PPO uses collision penalty 50 while the other methods use 10, so absolute return levels are not directly comparable. The figure must state this explicitly.

These are presentation-only defects. They do not alter any table, checkpoint, frozen CSV/NPZ artifact, protocol, trajectory selection, or scientific result. A narrow second plotting remediation and regression tests are warranted before one final regeneration and visual check.

### Narrow visual-QA remediation handoff

A second complete-file replacement package was prepared for the same two tracked files:

```text
analysis/plot_projection_results.py
tests/test_result_aggregation.py
```

The plotting revision:

- reserves additional right-page margin only for projection-only checkpoint figures;
- clips the displayed mean-plus/minus-sample-SD envelopes of all four bounded training-rate figures to `[0,1]` and uses feasible rate axes;
- derives the training-return warning from each method's checkpoint collision-penalty metadata rather than hardcoding campaign values; and
- leaves mean curves, checkpoint markers, aggregation, trajectory selection, and evidence inputs unchanged.

Regression coverage checks the rightmost label extent, all four bounded-rate filenames, the metadata-derived return warning, and the equal-penalty no-warning case. Both changed files compile. Real-data smoke rendering produced 29 affected or neighboring PDFs; all ten projection-only PDFs keep their text within the 475.2-point page, leaving approximately 14.8 points at the right edge. All four real-data rate bands remain within `[0,1]`, and the return warning correctly states collision penalty 50 for PPO high penalty versus 10 for the other methods. Independent source review found no blocker.

The archive contains exactly the two repository-relative files, extracts without error, and is byte-identical to the reviewed sources:

```text
5d0abf9e3207d4f01dbc7947094f65ef49cc109d7461491d814b6c492550f8bc  analysis/plot_projection_results.py
194c3a553ec614379168989a66d57be8ae7e001d049fff29898fff28ed41cb2e  tests/test_result_aggregation.py
047a481248b46c721a2afb2ada082765316efb9429a528243fd5026957b9d522  Predictive_Action_Projection_Visual_QA_Remediation_Full_Files.zip
```

Local targeted and full-suite tests remain required after installation before any figure directory is regenerated.

### Full-suite abort caused by the new in-process label test

After the narrow visual-QA files were installed, the complete suite again terminated the Python process inside Matplotlib. The stack identifies the newly added `test_projection_only_figure_reserves_right_label_margin`, specifically its direct `figure.canvas.draw()` call. This reproduces the already diagnosed Windows process-state problem: after earlier tests load native numerical libraries, an in-process Matplotlib render can abort rather than raise a Python exception.

This failure does not implicate the plotting source, frozen evidence, generated numbers, or the label-margin correction. The test itself violated the isolation rule already adopted for PDF rendering. The correction is test-only: the label-layout render and bounding-box assertion now execute in a clean Python subprocess with `Agg` and an isolated Matplotlib configuration directory. A child native failure becomes an ordinary captured pytest failure rather than terminating the full suite.

The revised test file compiles, contains no parent-process canvas draw, and preserves the original assertions: projection-only figures must request right margin `0.94`, and the complete rightmost tick-label extent must remain inside the figure boundary. The one-file archive contains only `tests/test_result_aggregation.py` and extracts byte-identically:

```text
13202dac910c7e797b89fcef30a86db4e6a8b27584e569bfe4f281cc5b92e88d  tests/test_result_aggregation.py
1bb1d5fe9c7b80cce0e9323dadb67f121a1de0ef20771eb9532193cf5530f2c4  Predictive_Action_Projection_Label_Test_Subprocess_Fix.zip
```

No figure directory should be removed or regenerated until the corrected targeted and complete suites pass locally.

### Corrected local test validation

After installing the isolated label-layout test, both required local gates passed:

```text
25 passed  — tests/test_result_aggregation.py
71 passed  — complete repository suite
```

The subprocess correction therefore preserves all six new visual-remediation test cases while preventing the known Windows native-library conflict from terminating the parent pytest process. The plotting revision is cleared for final figure regeneration. Frozen evaluation evidence and aggregation tables remain unchanged.

### Final primary figure regeneration

After the corrected targeted and complete test suites passed, only the superseded primary figure directory was removed and rebuilt from the unchanged primary tables, frozen evaluation artifacts, trajectory archives, and training logs.

The final primary build completed successfully and produced the expected inventory:

- 12 evaluation PDFs, including `evaluation_timeout_rate.pdf`
- 10 training-diagnostic PDFs
- `representative_trajectories.pdf`
- Five supporting CSV files covering training events/diagnostics, curve points, and trajectory selection
- `figure_build_audit.json`

This is 23 PDFs plus five supporting CSV files and the audit record. The rebuilt figures contain the narrow presentation corrections for right-edge labels, bounded rate bands, and the metadata-derived training-return comparability caveat. No training, evaluation, aggregation, protocol, checkpoint, or numerical evidence was changed. A compact structural and PDF-metadata audit is the next gate before rebuilding the transfer figures.

The compact final-primary audit subsequently passed:

```text
PASS: primary audit, 23 PDFs, 28 generated artifacts, metadata/fonts valid, no skips
```

This confirms the final primary output is structurally complete: the audit reports `PASS`, the artifact scope includes evaluation and training diagnostics, all 23 PDFs are present, all 28 pre-audit generated artifacts are recorded, `evaluation_timeout_rate.pdf` is present, document metadata and embedded fonts are valid, Type 3 fonts are absent, and no artifact was skipped. The next controlled action is to regenerate the evaluation-only transfer figures with the same final plotting source.

### Final transfer figure regeneration

Only the superseded `results\figures\core_layout_transfer` directory was removed. The evaluation-only transfer figures were then rebuilt from the unchanged transfer tables, frozen transfer evaluation artifacts, and trajectory archives using the final plotting source and `--skip-training-diagnostics`.

The build completed successfully and produced the expected evaluation-only inventory:

- 12 evaluation PDFs, including `evaluation_timeout_rate.pdf`
- `representative_trajectories.pdf`
- `representative_trajectory_selection.csv`
- `figure_build_audit.json`

No duplicated training diagnostics were generated in the transfer result set. No training, evaluation, aggregation, protocol, checkpoint, or numerical evidence was changed.

The compact final-transfer audit subsequently passed:

```text
PASS: transfer audit, 13 PDFs, 14 generated artifacts, evaluation-only, metadata/fonts valid, no duplicates or skips
```

This confirms the final transfer output is structurally complete: the audit reports `PASS`, the artifact scope is evaluation-only, shared training diagnostics are intentionally omitted, all 13 PDFs are present, all 14 pre-audit generated artifacts are recorded, `evaluation_timeout_rate.pdf` is present, document metadata and embedded fonts are valid, Type 3 fonts are absent, no duplicated transfer training artifacts remain, and no artifact was skipped.

Both final result suites have now passed source tests, build checks, artifact-scope checks, and PDF structural checks. The next controlled action is to package the complete regenerated figures and tables for the final direct visual and numerical review. No commit should be made until that review passes.

### Final regenerated-result visual and numerical QA

The final review archive passed integrity testing and contained the exact intended review inventory: 23 primary PDFs, 13 evaluation-only transfer PDFs, both figure-build audits, both result-table suites, and the shared primary training diagnostics. All 36 one-page PDFs were rendered at readable resolution and inspected directly.

The three previously identified presentation defects are resolved:

- all ten projection-only evaluation figures contain the complete two-line `PPO trained with projection` label, with 14.76 points of right-page clearance;
- `training_rolling_collision_rate.pdf` and `training_rolling_success_rate.pdf` use feasible `[0,1]` axes and clip only the displayed uncertainty bands at those bounds while leaving the means unchanged; and
- `training_return.pdf` clearly states that collision penalty 50 for PPO high penalty is not directly comparable with penalty 10 for the other methods, matching the checkpoint metadata.

No clipping, overlap, missing label, malformed glyph, legend collision, trajectory annotation defect, or other visual blocker was found. Every rendered page retains visible whitespace on all four sides. PDF author/title/subject metadata, embedded Type 0/TrueType-compatible fonts, absence of Type 3 fonts, and complete extractable text all pass.

The numerical review independently reconciled:

- all 3,000 fixed-geometry and 720 transfer episode rows;
- all 60 checkpoint summaries;
- all 12 method summaries;
- all 30 checkpoint-paired projection-difference rows and six across-seed paired summaries;
- all 12 prespecified trajectory selections and their displayed outcomes, lengths, intervention rates, checkpoint hashes, and episode identities; and
- 36,536 training scalar events, 5,131 completed training episodes, 750 rollout rows, and 14,811 plotted training-curve points.

All outcome rows are mutually exclusive and exhaustive, all table aggregation and sample-SD calculations reproduce exactly to floating-point tolerance, and no projection solver failure is present. The final figure and table set is cleared for commit review. The next gate is a complete Git inventory; no file should be staged or committed until that inventory is inspected.

### Pre-commit Git inventory

The complete untracked-file inventory contains exactly the intended final scope:

- two modified tracked files: `analysis/plot_projection_results.py` and `tests/test_result_aggregation.py`;
- two new repository records: the analysis record and analysis command record;
- 38 generated figure artifacts across the two suites, comprising 36 PDFs and two figure-build audits; and
- 22 generated table artifacts across the two suites, including the complete episode tables, checkpoint/method/paired summaries, trajectory selections, result-build audits, generated LaTeX tables, and the single authoritative primary training-diagnostic set.

No review ZIP, checkpoint, raw evaluation archive, duplicated transfer training diagnostic, or unrelated file appears in the proposed commit scope. `git diff --check` reports only the expected Windows `core.autocrlf` warning that the two modified Python files will be converted from LF to CRLF when Git next rewrites the working copy; it reports no whitespace error. The tracked diff contains 904 insertions and 53 deletions across the two previously reviewed replacement files. Generated untracked artifacts are correctly absent from `git diff --stat` until staged.

The inventory is approved for exact-path staging after the two repository records are synchronized to this latest version.

### Staged whitespace gate

The approved 64-file scope was staged successfully: two modified Python files, two repository records, 38 figure artifacts, and 22 table artifacts. The cached statistic is 64,094 insertions and 53 deletions.

The first cached whitespace check identified nine trailing-space findings, confined to the compact metadata headers of the two Markdown records. Those spaces were intentional Markdown hard-line breaks, not content corruption, but they violate the repository whitespace gate. No Python, CSV, JSON, LaTeX, or PDF artifact failed the check.

The record headers were corrected by replacing trailing-space hard breaks with explicit Markdown backslash breaks. At the same time, the documented repository paths were corrected to match the actual case-preserving filenames. The corrected full record files contain no trailing whitespace and must replace and be restaged over the initially staged copies before the cached check is rerun.

### Final staged scope and public-release privacy gate

After the corrected records were installed and restaged, `git diff --cached --check` returned no output. The staged inventory contains exactly the approved 64 files: two modified Python files, two new repository records, 38 figure artifacts, and 22 table artifacts. The cached statistic is 64,128 insertions and 53 deletions, and no unstaged or unrelated file appears.

A public-release scan found no credential, access token, API key, client secret, private key, password, or email address in the project records. It found only two historical commands containing a literal local Windows user-profile path. Those path assignments were generalized to resolve the Downloads directory from `$env:USERPROFILE`; this changes documentation portability and privacy only, not any executed experiment, source behavior, test, table, figure, or scientific result.

The privacy-clean records must be installed and restaged, followed by a silent cached whitespace check, a passing staged scan for machine-specific Windows user paths, confirmation of the exact staged status, and confirmation that the cached statistic still reports 64 files. Commit authorization remains pending those final checks.

### Final result commit and remote branch verification

The final staged privacy and scope gate passed with exactly 64 files. The approved result set was committed as:

```text
b005123cb2c6c754a991d1e7fdc709437b90e915
Add audited predictive action projection results
```

The working tree was clean after the commit, and `git diff-tree` confirmed that the commit contains exactly 64 files. The branch `final_evaluation_runs` was pushed to `origin` and configured to track `origin/final_evaluation_runs`.

Direct remote verification confirmed that the repository remains private, its default branch is `main`, the pushed commit is the tip of `final_evaluation_runs`, and the branch is exactly one commit ahead of and zero commits behind `main`. No GitHub Actions workflow or commit-status check is configured. No pull request has been opened and no merge or visibility change has occurred.

### Full-history public-release audit

A local audit traversed the complete reachable Git patch history and object inventory. It found no high-confidence private-key or service-token signature, no machine-specific Windows user-profile path, no suspicious credential-file name, and no blob larger than 20 MB. The audit passed:

```text
PASS: full-history public-release audit
```

The GitHub-connected current-tree search independently found no private-key marker, API-key assignment, client-secret assignment, access-token assignment, password assignment, GitHub token signature, OpenAI-style key signature, literal Windows user-profile path, or embedded project-contact email. The committed result branch is therefore cleared from the security and privacy perspective for eventual public exposure.

### Public landing-page and attribution review

The security audit found no blocker, but the repository landing page still described a pre-results workflow and did not guide reviewers to the completed result set. The repository also lacked a license even though `algorithms/ppo/ppo_continuous_action.py` is explicitly adapted from CleanRL's MIT-licensed continuous-action PPO implementation.

A documentation-only release patch has therefore been prepared with:

- a full `README.md` replacement that states the completed evaluation scope, reports the paired projection deltas with an explicit preliminary-analysis boundary, links directly to the committed tables, figures, and audit records, and preserves the non-overclaiming scope statement;
- a source-code-scoped `LICENSE` containing the MIT terms and both Salvador Tenorio and CleanRL developer notices while excluding generated figures, result tables, datasets, and documentation from that source-code grant; and
- `THIRD_PARTY_NOTICES.md`, which records the adapted CleanRL file, upstream links, and retained MIT notice.

Every percentage and paired-difference value in the proposed README was independently reconciled against the committed fixed-geometry and transfer CSV summaries. The patch changes documentation and attribution only; it does not alter source behavior, tests, protocols, evidence, figures, tables, or scientific interpretation inputs.

## Scientific interpretation phase

### Phase I, Step 1: frozen interpretation workspace

**Completed:** 2026-08-27\
**Repository mode:** read-only\
**Scientific judgment:** PASS. The audited evidence identity is identified and
stable enough to begin reconstruction of the experimental design. This step
establishes evidence identity only; it does not promote any preliminary result
interpretation into a final study claim.

#### Commit identities

| Role | Full commit | Verification and meaning |
|---|---|---|
| Current public `main` revision | `5a5cc2041ad5b6194a86aff9e61460873ce185c9` | Resolved directly from `refs/heads/main` on 2026-08-27. The repository currently has one remote branch, `main`; this revision controls its default public view. |
| Audited-results audit anchor | `b005123cb2c6c754a991d1e7fdc709437b90e915` | Commit titled `Add audited predictive action projection results`. It is a reachable ancestor of current `main` and identifies the provenance of the single audited evidence set. The relevant evidence paths are byte-identical at this anchor and the recorded current revision. |
| Frozen protocol source commit | `ba64926aed98b08b7b285266cf85989d466f9f1c` | Tagged `predictive-action-projection-protocol-v1`; it is an ancestor of the audit anchor. It identifies the frozen protocol source, not a separate dataset or active branch. |
| Public-release documentation commit | `97f375a5ecc22db16d1cfc641dbff0e7f7ac9eed` | Adds release documentation, licensing, attribution, and final public-release record entries. It does not change the evidence paths. |

The evidence commit contains 64 changed files. Sixty are committed result-table
and result-figure artifacts; the remainder are the reviewed analysis/plotting
and record changes described by the release record. A path-restricted Git diff
from the evidence commit to current `main` is empty for both protocols,
`analysis/aggregate_projection_results.py`,
`analysis/plot_projection_results.py`, both complete result-table trees, and
both complete result-figure trees. The only later repository differences are
`README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and the two repository record
files. Commit `5a5cc204...` changes only `README.md` relative to the public-release
documentation commit.

The following Git object identities are identical at the evidence commit,
public-release documentation commit, and current `main`:

| Frozen object | Git object ID |
|---|---|
| `results/tables/fixed_training_geometry/` | `5a974a12ff5e869b1a77b6ad25e53f7f25965040` |
| `results/tables/core_layout_transfer/` | `b176b54f9f1afa1c110eda0d09fd2fce649cea9b` |
| Fixed-geometry `figure_build_audit.json` | `29b3d5f4946a2b83b723403960647b12a0f166ec` |
| Transfer `figure_build_audit.json` | `3f5731849cf583ace7c1c0907393738ed8ee39cf` |
| `analysis/plot_projection_results.py` | `a599eaf2c400c1cce51cf8ccc3ca1d4b2346bd3e` |
| Fixed-geometry protocol | `bdcf11c601b119f4c0a077ea9164d768d26fba43` |
| Transfer protocol | `128de5af6352d6480d3074675ef8067bccf5c8df` |

#### Evidence-count derivation

The fixed-geometry protocol contains three methods, five expected training
seeds, two required projection modes for every method, one audited layout, and
100 repeats per layout:

```text
3 methods x 5 checkpoints x 2 modes x 1 layout x 100 repeats
= 3 x 5 x 2 x 1 x 100
= 3,000 episode rows.
```

Sources: `experiments/fixed_training_geometry_analysis_protocol.json`, fields
`methods`, `expected_train_seeds`, each method's
`required_projection_modes`, and `expected_repeats_per_layout`; and
`results/tables/fixed_training_geometry/result_build_audit.json`, fields
`layout_count = 1`, `episode_row_count = 3000`, and `status = PASS`.

The transfer protocol contains the same three methods, five expected training
seeds, and two projection modes, with 24 audited layouts and one repeat per
layout:

```text
3 methods x 5 checkpoints x 2 modes x 24 layouts x 1 repeat
= 3 x 5 x 2 x 24 x 1
= 720 episode rows.
```

Sources: `experiments/projection_analysis_protocol.json`, the same named
fields; and
`results/tables/core_layout_transfer/result_build_audit.json`, fields
`layout_count = 24`, `episode_row_count = 720`, and `status = PASS`.

The complete frozen evidence count is therefore:

```text
3,000 fixed-geometry rows + 720 transfer rows = 3,720 episode rows.
```

Direct line counts independently give 3,001 and 721 lines in the two committed
episode CSVs, respectively. Subtracting one header line from each gives 3,000
and 720 data rows. The two suites remain separate estimands and must not be
pooled into a headline performance rate merely because their inventory counts
sum to 3,720.

#### Audit status recovered from the frozen sources

| Gate | Verified frozen result |
|---|---|
| Fixed result-table build | `PASS`; 3,000 episode rows, 30 checkpoint rows, six method-mode rows, 15 paired-checkpoint rows, three paired-method rows, zero recorded solver failures. |
| Transfer result-table build | `PASS`; 720 episode rows, 30 checkpoint rows, six method-mode rows, 15 paired-checkpoint rows, three paired-method rows, zero recorded solver failures. |
| Fixed figure build | `PASS`; evaluation-and-training scope, 23 PDFs, 28 generated pre-audit artifacts, zero skipped artifacts. |
| Transfer figure build | `PASS`; evaluation-only scope, 13 PDFs, 14 generated pre-audit artifacts, zero skipped artifacts; shared training diagnostics intentionally omitted. |
| Source regression gates | 25 targeted tests and 71 complete-repository tests passed before the final figure builds. |
| Direct visual/numerical gate | All 36 PDFs were rendered and inspected; the 3,720 episode rows and derived table layers reconciled. |
| Public-release history gate | The full-history public-release audit is recorded as `PASS` in commit `97f375a...`. |

The committed episode tables also recover the evaluation exposure for the
zero-failure statement. Projection was enabled for 1,500 fixed-geometry
episodes totaling 236,084 executed steps and 360 transfer episodes totaling
62,900 executed steps:

```text
enabled episodes = 1,500 + 360 = 1,860
enabled steps    = 236,084 + 62,900 = 298,984
failure count    = 0 + 0 = 0
```

These are enabled-step exposure counts from the fields `projection_mode`,
`episode_length`, and `projection_solver_failure_count` in the two committed
`evaluation_episode_results.csv` files. They are not labeled solver-call
counts here; that label requires the separate implementation check that one
solve occurs per enabled step. Training-time exposure remains separate.

#### Frozen source map

| Evidence role | Frozen source location |
|---|---|
| Intended design and estimands | `experiments/fixed_training_geometry_analysis_protocol.json`; `experiments/projection_analysis_protocol.json`; referenced layout suites. |
| Numerical episode evidence | `results/tables/fixed_training_geometry/evaluation_episode_results.csv`; `results/tables/core_layout_transfer/evaluation_episode_results.csv`. |
| Checkpoint, method, and paired summaries | The remaining CSV and generated-LaTeX files in both result-table directories. |
| Validation, pairing, and aggregation semantics | `analysis/aggregate_projection_results.py`, especially `REQUIRED_COLUMNS`, `validate_episodes`, `checkpoint_summary`, `method_summary`, `paired_deltas`, and `paired_summary`. |
| Plot-field mapping and presentation derivations | `analysis/plot_projection_results.py`, especially `EVALUATION_PLOTS`, `plot_evaluation_checkpoints`, the derived timeout definition, and the projection-only plotting boundary. |
| Build completeness and scope | Both `result_build_audit.json` files and both `figure_build_audit.json` files. |
| Frozen provenance and executed release procedures | The repository copies of this Analysis Record and the Analysis Command Record at the relevant commit. |
| Orientation only | Current-public-`main` `README.md`; it is not a numerical or interpretive authority. |
| Interpretation method | The continuation prompt, regeneration blueprint, and printable scientific-interpretation guide from the verified handoff package. |

#### Metric-dictionary location

The explanatory working dictionary is Section 4 of the printable guide and
Section 9 of the Markdown blueprint. The authoritative field and reduction
definitions to be used when the formal metric dictionary is built in Phase I,
Step 5 are located in:

1. the exact headers of both committed episode and summary CSV layers;
2. `analysis/aggregate_projection_results.py`, including the required columns,
   validation equations, episode-to-checkpoint reductions, checkpoint-to-method
   reductions, and paired enabled-minus-disabled construction; and
3. `analysis/plot_projection_results.py`, including plotted field names,
   projection-only structural handling, and
   `timeout_rate = 1 - success_rate - collision_rate`.

No plausible field definition will be inferred from a figure title. Step 5
will reconcile the working dictionary against these frozen sources before any
ambiguous metric is interpreted.

#### Living-record and sidecar procedure

- The committed repository copies remain frozen and unchanged during
  interpretation unless Salvador Tenorio later authorizes a repository update.
- Verified milestones replace the existing Library identities of the Analysis
  Record, Analysis Command Record, and continuation prompt; they do not create
  duplicate copies.
- Exact new commands and calculation procedures are recorded in the command
  record. Verified reasoning, decisions, evidence boundaries, and caveats are
  recorded here.
- `Predictive_Action_Projection_Interpretation_Workbook.md` stores the active
  source map, decisions, teach-back notes, and analysis queue.
- `Predictive_Action_Projection_Claim_Evidence_Matrix.csv` stores only
  traceable candidate claims and preserves rejected interpretations.
- `Predictive_Action_Projection_Report_Ready_Sentences.md` stores scoped
  sentences that have already passed their applicable gate; result conclusions
  will not be added before the interpretation gate.
- The continuation prompt is updated whenever the current numbered step
  changes materially.

#### Missing sources, discrepancies, and operational boundary

No numerical or protocol conflict was found among the authoritative Step 1
sources. Raw checkpoints, raw evaluation shards, NPZ trajectory archives, and
TensorBoard logs are intentionally outside ordinary public source control; this
is a scope limitation of the public curated layer, not an unresolved Step 1
discrepancy, because the committed audits and records identify their role and
the present step does not require regeneration.

The printable guide's Step 1 checklist suggests creating a dedicated
analysis/report branch. The controlling continuation prompt makes the
interpretation phase read-only by default and does not authorize repository
mutation. The higher-authority operational rule therefore controls: no branch,
checkout, edit, stage, commit, merge, or push was performed. The interpretation
workspace is frozen by reference to the audit anchor and the versioned Library
working records. A repository report branch can be created
later only after explicit authorization.

#### Step gate

Phase I, Step 1 is complete. The next analytical action is Phase I, Step 2:
reconstruct methods, checkpoints, projection modes, episodes, layouts, seeds,
and totals from the frozen sources. No scientific outcome claim was promoted
during Step 1.

### Phase I, Step 2: reconstructed frozen experimental design

**Completed:** 2026-08-27\
**Repository mode:** read-only\
**Scientific judgment:** PASS, qualified by one explained and bounded
provenance-hash discrepancy.
The frozen design, the 15 executed checkpoint identities, both execution-time
projection modes, both evaluation-suite grids, and all row totals reconcile.
This step establishes the experimental structure only. It does not establish
an outcome effect, inferential independence, cross-method seed blocking, or a
causal learning mechanism.

#### Purpose and scientific importance

Step 2 reconstructs what was varied, what was held fixed, and how every
evaluation row enters the design. This prevents three common errors:

1. confusing the training intervention with the evaluation-time projector;
2. counting repeated episodes or layouts as additional trained policies; and
3. pooling the stochastic familiar-geometry suite with the deterministic
   transfer suite merely because both contain episode rows.

The governing design sources are:

- `docs/records/predictive_action_projection_experimental_protocol.md`;
- `experiments/fixed_training_geometry_analysis_protocol.json`;
- `experiments/projection_analysis_protocol.json`;
- `evaluation/layouts/fixed_training_geometry.json`;
- `evaluation/layouts/core_navigation_layouts.json`;
- both committed `evaluation_episode_results.csv` files;
- both committed `checkpoint_summary.csv` files; and
- both `result_build_audit.json` files.

All paths and calculations below refer to the audited evidence set identified
by audit-anchor commit
`b005123cb2c6c754a991d1e7fdc709437b90e915`; the two protocol JSON blobs are
unchanged from protocol commit
`ba64926aed98b08b7b285266cf85989d466f9f1c` through the recorded public
`main` tip.

#### Explained and bounded transfer-protocol identity discrepancy

Step 2 found and resolved a narrow discrepancy that the Step 1 commit-tree
comparison did not expose:

| Source | Transfer-protocol canonical SHA-256 |
|---|---|
| Human-readable frozen protocol record | `dfc0e1e3de29c0f63eb6152a3f063ad1d17c4461430855f33222f93e827c6e90` |
| Actual committed `experiments/projection_analysis_protocol.json` | `f0f853fb53b910cdd9227e1562fc227201b08a6926d8895d430e507944b659d1` |
| Transfer `result_build_audit.json`, field `protocol_sha256` | `f0f853fb53b910cdd9227e1562fc227201b08a6926d8895d430e507944b659d1` |

The canonicalization function is
`evaluation/layout_suite.py::canonical_json_sha256`: UTF-8 JSON with sorted
keys, separators `(',', ':')`, `ensure_ascii=False`, and `allow_nan=False`.
Applying it to the committed transfer protocol gives `f0f853...`.

The declared `dfc0e1...` value was reproduced exactly by adding this one field
to the committed JSON object before canonicalization:

```text
validated_implementation_base_commit = d0548b3e6729571113675b6f4ccad6401f27f167
```

That provenance field is present in the fixed-geometry protocol and in the
freeze-status clause of the human-readable protocol record, but is absent from
the committed transfer JSON. Repository-wide inspection found no runtime or
analysis use of that key. No method, penalty, projection mode, training seed,
training budget, layout, repeat count, evaluation seed, policy mode, device,
episode limit, or projection parameter differs. The executed result build
records the actual committed identity `f0f853...`.

**Analytical treatment:** use `f0f853...` as the executed transfer-protocol
identity; retain `dfc0e1...` in the discrepancy register as the hash of an
otherwise identical object containing one additional, operationally unused
provenance field. This construction explains the two hash values exactly but
does not prove the historical cause of the difference. The discrepancy is a
documentation and machine-readable-provenance defect, not evidence of a
different experiment. The repository was not edited. If a documentation
correction is later authorized, it should explain the discrepancy rather than
silently altering the committed JSON.

This finding corrects the broad Step 1 statement that no protocol discrepancy
had been found. It does not invalidate the Step 1 commit boundary or episode
inventory.

#### Training-condition map

The `methods` array in both protocol JSON files is identical:

| Method key | Display name | Training collision penalty | Projection during training | Training seeds | Required evaluation modes |
|---|---|---:|---:|---|---|
| `ppo_baseline` | PPO baseline | 10.0 | disabled | 1, 2, 3, 4, 5 | disabled, enabled |
| `ppo_high_penalty` | PPO high penalty | 50.0 | disabled | 1, 2, 3, 4, 5 | disabled, enabled |
| `ppo_train_projection` | PPO trained with projection | 10.0 | enabled | 1, 2, 3, 4, 5 | disabled, enabled |

The first axis is the **training condition**. The second is the
**execution-time projection mode applied to a frozen checkpoint**. Therefore,
`ppo_train_projection` with evaluation projection disabled is a policy trained
with projection but executed nominally; `ppo_baseline` with evaluation
projection enabled is a baseline-trained policy executed as a policy-plus-
projector composite controller.

The common final training budget is `expected_training_timesteps = 51200` per
run. The protocol record also fixes `num_envs = 4`, `num_steps = 256`,
`num_minibatches = 8`, and `update_epochs = 4`. The derived training geometry
is:

```text
transitions per rollout
  = 4 environments x 256 steps/environment
  = 1,024 transitions/rollout

rollouts per run
  = 51,200 transitions/run / 1,024 transitions/rollout
  = 50 rollouts/run

minibatch optimizer steps per run
  = 50 rollouts/run x 8 minibatches/rollout x 4 epochs/minibatch
  = 1,600 optimizer steps/run

runs per method
  = 5 training seeds

transitions per method
  = 5 runs/method x 51,200 transitions/run
  = 256,000 transitions/method

all final training transitions
  = 3 methods x 5 runs/method x 51,200 transitions/run
  = 768,000 transitions

all PPO rollout/update iterations
  = 15 runs x 50 rollouts/run
  = 750 iterations

all minibatch optimizer steps
  = 15 runs x 1,600 optimizer steps/run
  = 24,000 optimizer steps
```

These totals describe computational exposure. They are not independent-policy
sample sizes and do not make reward levels comparable across the penalty-10
and penalty-50 training conditions.

#### Frozen checkpoint identity map

The `method`, `train_seed`, `checkpoint`, and `checkpoint_sha256` columns in
both `checkpoint_summary.csv` files produce the same 15-row mapping. Each
identity is also invariant across disabled and enabled episode rows within
both suites.

| Method | Training seed | Frozen checkpoint path | SHA-256 |
|---|---:|---|---|
| `ppo_baseline` | 1 | `runs\checkpoints\final\ppo_baseline_51200_seed1.pt` | `ac9747daee76239c878cc1339dfbe72a8ed020eaae9d3f5013a7ce78b8a07836` |
| `ppo_baseline` | 2 | `runs\checkpoints\final\ppo_baseline_51200_seed2.pt` | `b739ee92df7f8c5ecbcf3078764cb694532485a0a8395cabcc5b5e283ae77f58` |
| `ppo_baseline` | 3 | `runs\checkpoints\final\ppo_baseline_51200_seed3.pt` | `8a5e7c0edb9e277062afa2433ca3ef1568bb909b75bf3bdb82e42469e8e81171` |
| `ppo_baseline` | 4 | `runs\checkpoints\final\ppo_baseline_51200_seed4.pt` | `d0cdbe0eae3327affadf64e343d3519ee3e91fd82a805e833c9068c467b1af7c` |
| `ppo_baseline` | 5 | `runs\checkpoints\final\ppo_baseline_51200_seed5.pt` | `c7238f982b6e88a89665a3aced8c737fb05966ff76f926fe7592d6e0924664e1` |
| `ppo_high_penalty` | 1 | `runs\checkpoints\final\ppo_high_penalty_51200_seed1.pt` | `f946099fe7e2006e7a6b1a59938504d3e9d7e397fcb8eed0c86e6acc71637c88` |
| `ppo_high_penalty` | 2 | `runs\checkpoints\final\ppo_high_penalty_51200_seed2.pt` | `a3f683d8d1172d90a5f45a00f31759384b305350c48b5c24f09595e607c90545` |
| `ppo_high_penalty` | 3 | `runs\checkpoints\final\ppo_high_penalty_51200_seed3.pt` | `48e6b29f11086f82bb7f139dd80c052e298b8823ceb40af4967097c4f8743b97` |
| `ppo_high_penalty` | 4 | `runs\checkpoints\final\ppo_high_penalty_51200_seed4.pt` | `c57feec3843b8e3e585dc713ff618665322279b1975ee8277d305a134b7bbb80` |
| `ppo_high_penalty` | 5 | `runs\checkpoints\final\ppo_high_penalty_51200_seed5.pt` | `45d8ccefb072cbd8badc1551770d69a5c5a31c61b936add71224ea5172ff83d4` |
| `ppo_train_projection` | 1 | `runs\checkpoints\final\ppo_train_projection_51200_seed1.pt` | `3c5949efe582518ed18dd33bd0066b919e2240bf8d1a1675995de14bf291227d` |
| `ppo_train_projection` | 2 | `runs\checkpoints\final\ppo_train_projection_51200_seed2.pt` | `94a7e01271b6c4fdc2c40ab4df16731f351c85303fbfdd4a66df84cf7df61528` |
| `ppo_train_projection` | 3 | `runs\checkpoints\final\ppo_train_projection_51200_seed3.pt` | `8dd06c30b16fafd91d3e679f5262588e83f69345442e3f605e40213d266208ef` |
| `ppo_train_projection` | 4 | `runs\checkpoints\final\ppo_train_projection_51200_seed4.pt` | `fe5cf72639d7b42227099dba859c4bc05d33760b1ab46464c8237636967b6f16` |
| `ppo_train_projection` | 5 | `runs\checkpoints\final\ppo_train_projection_51200_seed5.pt` | `152c446a47682b6accaa1d5d54a6c547bcf0772aac856079d10c709b5dea791d` |

There are exactly three methods times five training seeds, or 15 frozen final
checkpoints. A common numeric seed label across two methods does not by itself
prove that those methods were intentionally blocked for cross-method
inference. That question remains reserved for Step 4.

#### Evaluation-suite reconstruction

| Property | Fixed training geometry | Core-layout transfer |
|---|---|---|
| Protocol role | `primary_stochastic_fixed_training_geometry` | `secondary_deterministic_layout_transfer` |
| Layout suite | `fixed_training_geometry_v1` | `core_navigation_layouts_v1` |
| Layout count | 1 | 24 prespecified layouts |
| Policy action | stochastic Gaussian sample | deterministic actor mean |
| Repeats per layout/checkpoint/mode | 100 | 1 |
| Evaluation seeds | 10000 through 10099 | 1000 through 1023 |
| Projection modes | disabled and enabled | disabled and enabled |
| Evaluation collision penalty | 10.0 | 10.0 |
| Device | CPU | CPU |
| Maximum episode length | 200 steps | 200 steps |
| Method-seed-mode cells | 30 | 30 |
| Rows per method-seed-mode cell | 100 | 24 |
| Rows per checkpoint across both modes | 200 | 48 |
| Rows per method | 1,000 | 240 |
| Projection-disabled rows | 1,500 | 360 |
| Projection-enabled rows | 1,500 | 360 |
| Suite total | 3,000 | 720 |

The exact primary mapping in every method-seed-mode cell is:

```text
layout_id       = fixed_training_geometry
layout_repeat i = 0, ..., 99
evaluation_seed = 10000 + i
episode         = i
seed            = evaluation_seed
```

The exact transfer mapping in every method-seed-mode cell follows the frozen
layout-array order:

| Episode index | Evaluation seed | Layout ID |
|---:|---:|---|
| 0 | 1000 | `control_open_route` |
| 1 | 1001 | `control_upper_clearance` |
| 2 | 1002 | `control_lower_clearance` |
| 3 | 1003 | `control_symmetric_clearance` |
| 4 | 1004 | `single_near_early_upper` |
| 5 | 1005 | `single_near_early_lower` |
| 6 | 1006 | `single_near_late_upper` |
| 7 | 1007 | `single_near_late_lower` |
| 8 | 1008 | `single_blocked_central_upper` |
| 9 | 1009 | `single_blocked_central_lower` |
| 10 | 1010 | `double_near_staggered_upper_first` |
| 11 | 1011 | `double_near_staggered_lower_first` |
| 12 | 1012 | `double_near_same_side_upper` |
| 13 | 1013 | `double_near_same_side_lower` |
| 14 | 1014 | `double_blocked_staggered_upper_first` |
| 15 | 1015 | `double_blocked_staggered_lower_first` |
| 16 | 1016 | `double_blocked_same_side_upper` |
| 17 | 1017 | `double_blocked_same_side_lower` |
| 18 | 1018 | `triple_mild_slalom_upper_first` |
| 19 | 1019 | `triple_mild_slalom_lower_first` |
| 20 | 1020 | `triple_narrowing_slalom_upper_first` |
| 21 | 1021 | `triple_narrowing_slalom_lower_first` |
| 22 | 1022 | `triple_shifted_slalom_upper_first` |
| 23 | 1023 | `triple_shifted_slalom_lower_first` |

For all transfer rows, `layout_repeat = 0`, `episode` is the zero-based layout
index, and `seed = evaluation_seed`. These mappings were verified in all 30
method-seed-mode cells, not inferred from only the first checkpoint.

#### Complete episode derivation and independent cross-checks

Primary suite:

```text
method-seed-mode cells
  = 3 methods x 5 checkpoints/method x 2 modes/checkpoint
  = 30 cells

rows
  = 30 cells x 1 layout/cell x 100 repeats/layout
  = 3,000 episode rows
```

Transfer suite:

```text
method-seed-mode cells
  = 3 x 5 x 2
  = 30 cells

rows
  = 30 cells x 24 layouts/cell x 1 repeat/layout
  = 720 episode rows
```

Alternative checkpoint-level reconstruction:

```text
fixed rows per checkpoint    = 2 modes x 100 episodes = 200
transfer rows per checkpoint = 2 modes x 24 layouts   = 48
combined rows per checkpoint = 200 + 48               = 248
all rows                     = 15 checkpoints x 248   = 3,720
```

Alternative method-level reconstruction:

```text
fixed rows per method    = 5 checkpoints x 2 modes x 100 = 1,000
transfer rows per method = 5 checkpoints x 2 modes x 24  =   240
combined rows per method = 1,000 + 240                   = 1,240
all rows                 = 3 methods x 1,240              = 3,720
```

The prescribed products, direct episode-table row counts, complete cell
counts, checkpoint summaries, and result-audit `episode_row_count` values all
agree without interpolation or rounding.

#### Scientific classification

- **Fact:** Three training conditions produced five final checkpoints each;
  the same 15 SHA-identified checkpoints occur in both suites.
- **Fact:** Every checkpoint was evaluated with execution-time projection both
  disabled and enabled under a common evaluation collision penalty of 10.0.
- **Calculation:** The training campaign consumed 768,000 environment
  transitions across the 15 final runs.
- **Calculation:** The evaluation inventory contains 3,000 fixed-geometry rows
  and 720 transfer rows, totaling 3,720.
- **Interpretation:** The design crosses a training-condition axis with an
  execution-time intervention axis, allowing nominal-policy and composite-
  controller questions to be separated later.
- **Caveat:** One hundred stochastic episodes refine the within-checkpoint
  estimate on one familiar geometry; they do not create 100 trained policies.
- **Caveat:** Twenty-four prespecified layouts broaden deterministic geometric
  coverage; they are not 24 independent training replicates or a random sample
  of all navigation geometries.
- **Caveat:** Numeric training-seed labels 1 through 5 recur across methods,
  but Step 2 does not establish intentional cross-method blocking.
- **Caveat:** The two suites differ in geometry, action-selection mode, and
  repetition structure, so the 3,720-row inventory is not a pooled performance
  estimand.
- **Hypothesis:** None. Step 2 does not require or support a mechanism
  hypothesis.

#### Step gate

Phase I, Step 2 is complete because:

- [x] all three training conditions and their only designed differences were
  recovered;
- [x] all 15 final checkpoint paths and SHA-256 identities were reconciled
  across both suites;
- [x] disabled and enabled projection coverage is complete for every
  checkpoint;
- [x] the 100 fixed-geometry seed/repeat mappings were verified in every cell;
- [x] the 24 transfer layout/seed mappings were verified in every cell;
- [x] prescribed, observed, summarized, and audited totals agree;
- [x] the transfer-protocol hash discrepancy was reproduced, classified, and
  bounded without changing the repository; and
- [x] no outcome claim or cross-method pairing assumption was introduced.

The next action is Phase I, Step 3: explain and then, after Salvador's explicit
authorization, lock the research questions before outcome interpretation
begins.

#### Independent continuity audit and terminology correction

An independent read-only audit on 2026-08-28 reconfirmed the Step 2 design
products and corrected three terminology issues:

- the remote exposes one branch, `main`; `b005123...` is an audit-anchor commit
  reachable from the recorded current revision, not a second active branch or
  second dataset;
- the training loop executes 750 PPO rollout/update iterations and 24,000
  minibatch-level `optimizer.step()` calls across the 15 runs, so the latter is
  reported as **minibatch optimizer steps**, not the ambiguous phrase
  **optimizer updates**; and
- adding the implementation-base provenance field reproduces the declared
  transfer-protocol hash exactly, but this proves the mathematical relationship
  between the objects rather than the historical cause of the discrepancy.

The Step 2 design reconstruction remains PASS. Protocol-identity consistency
remains qualified by the bounded transfer-protocol hash discrepancy. No
outcome value was inspected or promoted during this continuity audit.

### Phase I, Step 3: locked research questions and claim boundaries

**Completed:** 2026-08-28\
**Repository mode:** read-only\
**Scientific judgment:** PASS after narrowing the collision-penalty question
to the comparisons actually predeclared before the result commit. The gate was
adjudicated exclusively from pre-result design and comparison sources; no
outcome value was used to add, remove, or reword a question.

#### Pre-result source verification

The human-readable experimental protocol record at commit
`ba64926aed98b08b7b285266cf85989d466f9f1c` predeclares the complete 3-by-2
deployment matrix, outcome roles, six direct comparisons, transition-table
analysis, checkpoint-level replicate rule, and separate primary/secondary suite
roles. Git ancestry and path-restricted comparisons establish that:

1. the protocol commit precedes audited-results commit `b005123...`;
2. the human-readable protocol record did not change from the protocol commit
   through the audited-results commit; and
3. that record and both machine-readable protocol JSON files remain unchanged
   through recorded current `main` revision `5a5cc204...`.

The six frozen comparisons map to the locked questions as follows:

| Frozen comparison | Locked use |
|---|---|
| Baseline off versus baseline on | RQ1 execution-time projection. |
| Baseline off versus high-penalty off | RQ3 nominal penalty-training contrast. |
| Baseline off versus projection-trained off | RQ2 nominal training-condition contrast. |
| Projection-trained off versus projection-trained on | RQ1 execution-time projection and RQ5 filter-use support. |
| Baseline on versus projection-trained on | RQ2 composite-controller contrast and RQ5 operational diagnostics. |
| High-penalty off versus high-penalty on | RQ1 execution-time projection; supporting reward/filter-complementarity context for RQ3 and RQ5. |

#### Locked research questions

**RQ1 — primary fixed-geometry execution question**

> For each training condition and evaluation suite, what paired
> checkpoint-level changes in terminal outcomes and supporting behavioral
> metrics are observed when execution-time projection is enabled rather than
> disabled for the same checkpoint?

For outcome (Y), method (m), checkpoint (s), and suite (q), the direct
contrast is

\[
\Delta_{m,s}^{(q)}(Y)
=
\bar Y_{m,s,\mathrm{on}}^{(q)}
-
\bar Y_{m,s,\mathrm{off}}^{(q)}.
\]

This estimates the execution-time contribution of the projector under the
tested protocol. It does not establish that the nominal policy learned safety,
that projection guarantees safety, or that the result generalizes beyond the
tested controller and environments.

**RQ2 — primary fixed-geometry projection-training question**

> Under collision penalty 10, how do projection-trained checkpoints and
> baseline checkpoints differ when both are evaluated under the same execution
> mode, considered separately for projection-off and projection-on and
> separately by evaluation suite?

The direct contrasts are

\[
C_{P,e}^{(q)}(Y)
=
\mu_{P,e}^{(q)}(Y)-\mu_{B,e}^{(q)}(Y),
\qquad e\in\{\mathrm{off},\mathrm{on}\}.
\]

Projection-off compares nominal execution behavior. Projection-on compares
policy-plus-projector composite controllers. Separately trained methods are not
paired merely because they reuse numeric seed labels. The comparison does not
identify a causal learning mechanism or prove policy--projector co-adaptation.

**RQ3 — primary fixed-geometry reward-shaping question**

> Under the frozen PPO configuration and training budget, how do nominal
> checkpoints trained with collision penalty 50 differ from baseline nominal
> checkpoints trained with collision penalty 10 when both are evaluated
> without projection?

The direct contrast is

\[
C_{H,\mathrm{off}}^{(q)}(Y)
=
\mu_{H,\mathrm{off}}^{(q)}(Y)
-
\mu_{B,\mathrm{off}}^{(q)}(Y).
\]

The high-penalty off/on comparison is already part of RQ1 and may provide
supporting reward/filter-complementarity context. High-penalty on versus
baseline on is not a predeclared direct comparison and is exploratory if later
calculated. The study does not predeclare an equivalence margin and cannot
establish that penalties and projection are interchangeable.

**RQ4 — secondary prespecified transfer question**

> Across the 24 prespecified deterministic transfer layouts, are the directions
> and tradeoffs of the predeclared within-suite contrasts consistent with or
> different from those observed in the stochastic fixed-geometry suite, with
> the suites analyzed separately?

This permits a statement that the same direction was observed in both suites.
It does not permit pooling the suites, attributing cross-suite differences only
to geometry, or generalizing to arbitrary environments. Geometry,
action-selection mode, and repetition structure all differ between suites.

**RQ5 — supporting filter-use question**

> What filter-use profile---intervention frequency, correction magnitude,
> slack, clipping, clearance, solver status, and joint terminal
> outcomes---characterizes projection-enabled execution by training condition
> and evaluation suite?

RQ5 uses prespecified diagnostic metrics and descriptive matched terminal-
outcome transitions. It does not introduce a new independent intervention
contrast. It cannot establish latency, computational cost, real-time
suitability, intervention-causes-success, formal constraint satisfaction, or a
causal co-adaptation mechanism.

#### Outcome hierarchy and exploratory boundary

- Primary safety outcome: collision rate.
- Principal task outcome: success rate.
- Competing terminal outcome: timeout rate.
- Supporting outcomes: common-reward return, episode length, and minimum signed
  clearance.
- Filter-use diagnostics: clipping, intervention, correction, slack, and
  solver status.
- Planned descriptive analysis: matched collision/success/timeout transitions.
- Exploratory only: unlisted pairwise comparisons, formal
  difference-in-differences interactions, pooled or formal cross-suite effect
  differences, post hoc layout groups, intervention-outcome associations, and
  substitution/equivalence assessments.

No post hoc composite score is admitted. Collision, success, and timeout must
be interpreted jointly.

#### Step gate

Phase I, Step 3 passes because every locked question maps to the frozen design,
predeclared comparisons, prespecified metric families, or secondary-suite role;
RQ3 has been narrowed to the declared nominal penalty contrast; RQ5 is
supporting rather than an additional primary intervention question; and all
causal, equivalence, formal-safety, arbitrary-generalization, runtime-cost, and
pseudoreplication boundaries are explicit. Formal weighting, nesting, pairing,
and uncertainty rules remain reserved for Step 4, and exact metric definitions
remain reserved for Step 5.

The next analytical action is Phase I, Step 4: explain and then, after
Salvador's explicit authorization, establish the statistical units, nesting,
pairing keys, and permitted uncertainty calculations.

### Phase I, Step 4: verified units, pairing, aggregation, and uncertainty

Step 4 was executed read-only against audited result commit
`b005123cb2c6c754a991d1e7fdc709437b90e915`. The canonical protocols,
aggregation implementation, and result tables are unchanged at recorded public
`main` revision `5a5cc2041ad5b6194a86aff9e61460873ce185c9`.

#### Raw-table identities

The fixed table contains exactly 3,000 rows:

\[
3\text{ methods}\times5\text{ checkpoints}\times2\text{ modes}
\times100\text{ stochastic episodes}=3{,}000.
\]

Every one of the 30 method--checkpoint--mode cells contains 100 rows, uses the
single `fixed_training_geometry` layout, and covers evaluation seeds 10000
through 10099. The transfer table contains exactly 720 rows:

\[
3\times5\times2\times24\text{ layouts}=720.
\]

Every transfer cell contains the same 24 layout identifiers exactly once and
uses evaluation seeds 1000 through 1023. Both tables contain 15 distinct
checkpoint SHA-256 identities, one for each method--training-seed run. There
are no duplicate episode keys.

The validated raw key in both suites is:

```text
method, train_seed, checkpoint_sha256, projection_mode,
layout_id, layout_repeat, evaluation_seed
```

#### Pairing and nesting

For each method and checkpoint, disabled and enabled rows have identical
`layout_id`, `layout_repeat`, and `evaluation_seed` sets, and use the same
checkpoint SHA-256. All 1,500 fixed-suite and 360 transfer-suite off/on merges
are one-to-one. These matched evaluation cells support within-checkpoint
effects and descriptive outcome transitions.

Checkpoints are nested within training method. Projection mode is repeated
within checkpoint. The fixed episodes are repeated stochastic evaluations;
the 24 transfer layouts are fixed task conditions crossed with every
checkpoint and mode. Neither source of repeated evaluation changes the number
of independently trained policies.

Cross-method seeds are not protocol-declared blocks. The aggregation code does
not pair them. RQ2 and RQ3 therefore compare independent samples of five
checkpoints per method rather than five index-matched seed differences.

#### Verified aggregation implementation

`analysis/aggregate_projection_results.py` implements the declared hierarchy.
For ordinary checkpoint metrics:

\[
\bar Y^{F}_{m,s,e}
=\frac1{100}\sum_{i=1}^{100}Y^{F}_{m,s,e,i},
\qquad
\bar Y^{T}_{m,s,e}
=\frac1{24}\sum_{\ell=1}^{24}Y^{T}_{m,s,e,\ell}.
\]

Method means and sample standard deviations are then computed across the five
checkpoint summaries:

\[
\bar Y^{(q)}_{m,e}=\frac1{5}\sum_{s=1}^{5}\bar Y^{(q)}_{m,s,e},
\]

\[
s^{(q)}_{m,e}
=\sqrt{\frac1{4}\sum_{s=1}^{5}
\left(\bar Y^{(q)}_{m,s,e}-\bar Y^{(q)}_{m,e}\right)^2}.
\]

For each checkpoint, `paired_deltas()` merges disabled and enabled evaluation
cells one-to-one and averages enabled-minus-disabled cell differences. The
paired method summary then averages the five checkpoint effects and uses
sample SD with `ddof=1`. Independent reconstruction matched every audited
mean and SD to floating-point error below \(1.5\times10^{-14}\).

Because both suites are balanced, pooling raw rows happens to reproduce the
same point mean. It does not reproduce the correct independent replication.
For example, baseline fixed-geometry disabled success is 0.152. The checkpoint
SD is approximately 0.2121, giving checkpoint-based standard error

\[
0.2121/\sqrt5\approx0.0948.
\]

A naive binomial calculation using 500 episodes gives

\[
\sqrt{0.152(1-0.152)/500}\approx0.0161,
\]

almost six times smaller. The naive calculation falsely treats repeated
evaluation of five policies as 500 independent trained policies.

#### Locked uncertainty rules

All five checkpoint values, their mean, sample SD, range, and effect magnitude
will be primary. Within-method off/on effects additionally admit sign
consistency and leave-one-checkpoint-out stability. Optional (t_4) intervals
or sign-flip calculations are supplementary and must state their assumptions
and low resolution; no formal significance test was predeclared.

Cross-method contrasts are unpaired (5+5). A supplementary interval, if
used, requires an independent-sample standard error and Welch degrees of
freedom. Cross-method leave-one-out removes each of ten checkpoints singly.
Five index-matched signs, paired (t_4) intervals, or same-numbered seed
deletions are prohibited.

#### Outcome and metric cautions exposed by Step 4

All 3,720 rows are mutually exclusive and exhaustive success, collision, or
timeout observations. Thus

\[
T=1-S-C,
\qquad
\Delta T=-(\Delta S+\Delta C).
\]

`paired_layout_count` is a generic output label. Its fixed-suite value 100
means paired episode/repeat cells, not 100 layouts or independent replicates.
The obstacle-free transfer layout `control_open_route` has structurally
undefined clearance, so transfer clearance averages use 23 defined layouts.
Projection-disabled filter fields are not applicable rather than observed
zero burden. Episode/layout-averaged rates must not be silently replaced with
pooled step rates, which weight long episodes and represent another estimand.
Finally, the paired maximum-clipping delta averages matched cell-level maximum
differences; it is not the difference between checkpoint-wide maxima.

#### Training-budget hypothesis classification

The conjecture that ten times more training would substantially improve
success is scientifically plausible but not established by the frozen curves.
Four projection-trained checkpoints still improve late, but protected
fixed-geometry performance is already near ceiling. Baseline improvement is
dominated by seed 3, several baseline checkpoints lack a positive late trend,
and every high-penalty checkpoint has zero late success. The transfer layouts
were never used during training, so more fixed-geometry optimization supplies
no new geometric experience and may deepen specialization or filter use.

The learning-rate schedule is also budget-dependent. With base rate
\(3\times10^{-4}\), iteration 50 receives \(6\times10^{-6}\) in the frozen
50-iteration campaign but \(2.706\times10^{-4}\) in a 500-iteration campaign.
A 10x declared budget would change optimization from the beginning, not merely
continue the frozen runs. Retain this only as a private post-study hypothesis;
additional geometries or curriculum training would be a more directly targeted
transfer intervention, but the outcome is unknown and excluded from the report.

#### Step gate

Phase I, Step 4 passes. Row identity, checkpoint nesting, permitted pairing,
aggregation order, equal checkpoint weighting, sample-SD denominator,
cross-method independence, suite separation, outcome accounting, and
uncertainty boundaries are now verified. No repository file was modified.

The next analytical action is Phase I, Step 5: construct the metric and
evidence dictionary from exact source fields and reduction semantics.

### Phase I, Step 5: verified metric and evidence dictionary

Step 5 was executed read-only against audited result commit
`b005123cb2c6c754a991d1e7fdc709437b90e915`. All metric-generating,
aggregation, plotting, contract, and result-table paths inspected in this step
are byte-unchanged at recorded public revision
`5a5cc2041ad5b6194a86aff9e61460873ce185c9`.

The lowest-level committed numerical evidence consists of the two canonical
episode-row tables, with 3,000 fixed rows and 720 transfer rows. They share the
same 53-column schema. All fields are complete except 30 structural clearance
`NaN`s: the obstacle-free `control_open_route` layout for 15 checkpoints and
two projection modes. The original evaluator shards and full per-step
trajectory archives are intentionally absent from public source control.

#### Outcome and reward definitions

Success is final goal distance at most 0.25 environment-coordinate units.
Collision is signed agent--obstacle boundary clearance at most zero. The
environment terminates on success or collision and truncates at 200 steps only
when neither occurs. All 3,720 rows independently satisfy the exact identity

\[
S+C+T=1,
\qquad T=1-S-C.
\]

The common final-evaluation per-step reward is

\[
r_t=(d_{t-1}-d_t)-0.01(v_t^2+\omega_t^2)-0.01
    +10S_t-10C_t-d_tT_t.
\]

`episode_return` is its undiscounted sum. It is a constructed reward score,
not physical energy, distance, or a value estimate. All final evaluation rows
use collision penalty 10, including the policy trained with penalty 50, so
evaluation return is comparable across methods within each suite. Training
returns remain incomparable across the penalty-10 and penalty-50 reward scales.

Episode length is a transition count with ambiguous direction: a short success
and a short collision have opposite meanings. Final distance is a verified raw
field but is not carried into the checkpoint, method, paired, LaTeX, or plot
layers. Any method-level use would require a new audited analysis. Minimum
clearance is the trajectory minimum signed boundary margin, including the
initial state. Transfer checkpoint clearance means use 23 obstacle-containing
layouts; obstacle-free clearance is undefined rather than zero.

No SI distance or time system is declared. Distances must be reported as
environment-coordinate units and duration as transitions or simulated steps.

#### Action clipping and projector order

The audited execution order is:

```text
raw normalized action
-> clipping to [-1,1]^2
-> physical [v, omega] mapping
-> optional CBF-QP projection
-> environment execution
```

Clipping rate is the fraction of episode transitions on which either raw
normalized component exceeds its bound. Mean clipping norm is the mean
normalized-coordinate excess norm over all episode transitions, including
zeros; the episode maximum is its largest value. Speed and turn clipping use
the same episode-length denominator and may overlap on one transition.
Clipping is applicable in both execution modes and is distinct from physical
projector correction.

For projection-enabled transitions,

\[
k_t=\lVert u_t^{exec}-u_t^{raw}\rVert_2,
\qquad I_t=\mathbf1\{k_t>10^{-6}\}.
\]

The episode intervention rate is `sum(I_t)/L`; mean correction is
`sum(k_t)/L`; maximum correction is `max(k_t)`. Correction is a numerical norm
over physical `[v, omega]` coordinates with different component meanings and
scales. It is not distance, energy, or commensurate with normalized clipping
norm.

Slack metrics derive from nonnegative CBF relaxation variables. Mean summed
slack averages the per-step sum across active constraints over all episode
steps; maximum slack is the largest individual constraint slack across the
episode. Slack is not clearance, penetration, or collision probability, and
depends on barrier scaling, parameters, geometry, and active-constraint count.

Projection-disabled raw zeros are schema placeholders. Canonical checkpoint
summaries convert intervention, correction, and slack to `NaN`/not applicable.
Enabled zero correction/slack is a real observed zero. No-obstacle projection
returns genuine zero with `no_active_constraints`. A failed solve increments
the failure count and makes slack unknown/`NaN`; unknown slack is never zero.

#### Reduction and weighting

Ordinary checkpoint metrics are equal means over 100 fixed episodes or 24
transfer layouts. Transfer clearance is the exception with 23 defined layouts.
Rates are equal means of episode/layout rates, not pooled transition fractions:

\[
\frac1N\sum_i\frac{c_i}{L_i}
\ne
\frac{\sum_i c_i}{\sum_iL_i}
\quad\text{in general}.
\]

For projection-trained transfer evaluation, the canonical mean layout-level
intervention rate is 0.567, while the pooled transition fraction is 0.690.
Both are valid for different estimands. Canonical report wording must say mean
episode-level rate for fixed geometry and mean layout-level rate for transfer.

Episode maximum clipping, correction, and slack values are first maximized
within checkpoint. The method table reports the mean and sample SD of the five
checkpoint maxima, not the study-wide maximum. The paired maximum-clipping
quantity instead averages matched episode/layout maximum differences and is
not the difference of two checkpoint maxima. Projector-only burdens have no
off/on delta because projection-off burden is not applicable.

#### Solver exposure

| Context | Enabled transitions | Active-constraint QP attempts | Failures |
|---|---:|---:|---:|
| Projection-enabled training | 256,000 | 256,000 | 0 |
| Fixed evaluation | 236,084 | 236,084 | 0 |
| Transfer evaluation | 62,900 | 60,185 | 0 |
| Combined accepted evidence | 554,984 | 552,269 | 0 |

The 2,715 remaining transfer transitions are from the obstacle-free layout and
bypass OSQP. `optimal_inaccurate` is accepted as success, but the committed
episode layer does not retain the solver-status distribution. Zero failures
therefore establishes empirical numerical reliability only for the frozen
accepted simulations; it does not establish universal reliability, runtime,
real-time suitability, invariance, or formal safety.

#### Training diagnostics

Training diagnostics are supporting learning histories. The structured
training evidence contains 5,131 completed episodes, 750 rollout rows of 1,024
transitions, and 14,811 aggregated curve points. Only completed episodes enter
episode and rolling curves; episodes still active at the training-budget
boundary are absent. Rolling outcomes use the latest 20 completed episodes,
with early denominators 1 through 19. Direct curves linearly interpolate each
seed onto a common overlapping step grid; derived outcome curves carry forward
the most recent completed-episode state. Shading is mean plus/minus one sample
SD across five seeds, not a confidence interval. Interpolated points are not
new observations.

Training final clearance is a final-state margin and is not evaluation minimum
trajectory clearance. Training policies change during data collection; final
evaluation freezes the checkpoint. Training observations never replace final
evaluation evidence.

#### Verification boundary and validation gaps

Independent checks passed for outcome identities, count/rate formulas,
component clipping relations, intervention formulas, mean/max ordering,
success/final-distance and collision/clearance consistency, structural
missingness, disabled placeholders, and failure totals. All committed table and
plot mappings reconcile.

Because full per-step archives are not public, every episode return and every
mean/max norm cannot be recomputed from public per-step values. Their generating
formulas and committed episode aggregates are verified; this is the exact
public reproducibility boundary.

Two nonblocking builder gaps were found. `projection_intervention_count` is not
a declared required column and the builder does not check its rate against
episode length. The builder also does not assert disabled raw placeholders are
zero before converting burden summaries to `N/A`. The frozen rows independently
pass both checks, so neither is a frozen-result defect.

The compact generated LaTeX tables must not be reused unchanged: their outcome
table omits timeout, and fixed three-decimal formatting can display small
nonzero values as `0.000`. Final report tables will be rebuilt from CSV sources
with all terminal outcomes and metric-appropriate precision.

The larger-budget/two-additional-geometry/five-obstacle idea is retained only
as a private post-study design note and is excluded from the report. The frozen
capacity is three obstacles; five-obstacle capacity changes observation
dimension and requires new training.

#### Step gate

Phase I, Step 5 passes. Definitions, units, denominators, reductions,
applicability, structural missingness, directionality, verification limits, and
caption/table rules are locked. Phase I is complete. No repository file was
modified.

The next analytical action is Phase II, Step 6: fixed-geometry absolute
performance before paired effects.

### Phase II, Step 6: fixed-geometry absolute performance

Step 6 was executed read-only on 29 August 2026 against the 3,000 committed
rows in `results/tables/fixed_training_geometry/evaluation_episode_results.csv`.
Each method--seed--mode cell contained exactly 100 stochastic episodes. Timeout
was reconstructed as `1 - success - collision` and matched `truncated` in every
row.

For each checkpoint and mode, the analysis calculated the arithmetic mean of
the 100 episode observations. It then gave the five checkpoint summaries equal
weight and calculated sample SD with `ddof=1`. The reconstruction matched all
30 committed checkpoint rows and all six committed method-summary rows. The
largest differences were \(1.78\times10^{-15}\) for checkpoint return,
\(9.71\times10^{-17}\) for checkpoint clearance, and
\(2.84\times10^{-14}\) for a method-level value.

#### Absolute outcomes

| Training condition | Projection | Success mean +/- SD | Collision mean +/- SD | Timeout mean +/- SD |
|---|---|---:|---:|---:|
| PPO baseline | Disabled | 15.2% +/- 21.2 pp | 14.0% +/- 7.2 pp | 70.8% +/- 23.7 pp |
| PPO baseline | Enabled | 17.8% +/- 25.0 pp | 0.2% +/- 0.4 pp | 82.0% +/- 24.9 pp |
| PPO high penalty | Disabled | 0.0% +/- 0.0 pp | 13.4% +/- 6.3 pp | 86.6% +/- 6.3 pp |
| PPO high penalty | Enabled | 0.2% +/- 0.4 pp | 0.6% +/- 0.9 pp | 99.2% +/- 0.8 pp |
| PPO trained with projection | Disabled | 17.4% +/- 8.0 pp | 81.8% +/- 7.6 pp | 0.8% +/- 0.8 pp |
| PPO trained with projection | Enabled | 93.6% +/- 8.8 pp | 1.4% +/- 2.1 pp | 5.0% +/- 6.8 pp |

The enabled projection-trained success calculation used checkpoint values
`0.78, 0.96, 0.97, 0.99, 0.98`. Their sum is `4.68`, giving mean `0.936`.
The squared deviations sum to `0.030920`; division by four gives sample
variance `0.007730`, whose square root is `0.087920419`.

The corresponding descriptive episode counts were baseline disabled
`76/70/354`, baseline enabled `89/1/410`, high-penalty disabled `0/67/433`,
high-penalty enabled `1/3/496`, projection-trained disabled `87/409/4`, and
projection-trained enabled `468/7/25`, in success/collision/timeout order.
These counts provide coverage and arithmetic reconciliation, not independent
training replication.

#### Supporting absolute metrics

| Training condition | Projection | Return | Clearance | Length |
|---|---|---:|---:|---:|
| PPO baseline | Disabled | -5.0858 +/- 3.9965 | 0.3753 +/- 0.1310 | 176.37 +/- 25.54 |
| PPO baseline | Enabled | -3.9123 +/- 4.5648 | 0.4144 +/- 0.1264 | 188.03 +/- 21.53 |
| PPO high penalty | Disabled | -9.3636 +/- 0.4477 | 0.5432 +/- 0.1162 | 190.15 +/- 7.52 |
| PPO high penalty | Enabled | -8.6595 +/- 0.6946 | 0.5822 +/- 0.1020 | 199.80 +/- 0.23 |
| PPO trained with projection | Disabled | -5.3570 +/- 1.6443 | -0.0155 +/- 0.0090 | 41.69 +/- 6.36 |
| PPO trained with projection | Enabled | 10.0042 +/- 1.5995 | 0.0674 +/- 0.0144 | 84.33 +/- 8.82 |

The high-penalty controller demonstrates that high mean clearance and low
collision are not synonymous with competence: its enabled configuration timed
out in 99.2% of episodes. The projection-trained disabled controller shows that
short duration is not automatically beneficial: its episodes were short
because 81.8% ended in collision.

#### Checkpoint structure and interpretation

Baseline success ranged from 0--51% disabled and 0--60% enabled. Seed 3 was
visibly influential; removing it only as a descriptive sensitivity check
reduced disabled mean success from 15.2% to 6.25%. It remains in every primary
summary and is not classified as a formal outlier.

Every high-penalty disabled checkpoint had 0% success. Enabled high-penalty
timeout ranged from 98--100%. Projection-trained disabled collision ranged from
76--92%. Projection-trained enabled success ranged from 78--99%; seed 1 was
weaker than seeds 2--5 but the dominant outcome remained success.

The defensible absolute conclusion is that only the projection-trained,
projection-enabled composite controller was success-dominated under the frozen
fixed-geometry stochastic protocol. Baseline and high-penalty controllers were
timeout-dominated, while the projection-trained nominal component was
collision-dominated.

This step describes controller endpoints. It does not yet calculate the paired
within-checkpoint projector effect, prove that projection caused the complete
cross-method result, establish policy--projector co-adaptation, or generalize
beyond the single stochastic fixed geometry.

#### Step gate

Phase II, Step 6 passes. All absolute endpoints, five-checkpoint variability,
outcome accounting, and supporting-metric cautions are verified. Phase II,
Step 7 is recorded below.

### Phase II, Step 7: fixed-geometry paired projection effects

The fixed-geometry episode rows were paired within checkpoint by layout,
repeat, and evaluation seed. Projection-enabled minus projection-disabled
outcome rates were first computed separately for each of the five independently
trained checkpoints, then summarized with the equal-checkpoint mean and sample
SD.

| Training condition | Success delta | Collision delta | Timeout delta |
|---|---:|---:|---:|
| PPO baseline | +2.6 +/- 3.8 pp | -13.8 +/- 7.3 pp | +11.2 +/- 7.5 pp |
| PPO high penalty | +0.2 +/- 0.4 pp | -12.8 +/- 5.8 pp | +12.6 +/- 6.1 pp |
| PPO trained with projection | +76.2 +/- 4.5 pp | -80.4 +/- 6.1 pp | +4.2 +/- 6.1 pp |

For projection-trained PPO, success increased by 72 to 84 pp and collision
decreased by 75 to 87 pp in all five checkpoints. Its leave-one-checkpoint-out
success mean ranged from +74.25 to +77.25 pp. Baseline and high-penalty
collision reductions were instead balanced primarily by additional timeout.

The outcome identity was checked for every checkpoint:

\[
\Delta S+\Delta C+\Delta T=0.
\]

This permits net outcome accounting but does not identify the individual
episode transitions. The paired fixed-geometry evidence supports substantial
empirical filter dependence for projection-trained PPO, but not deliberate
reliance, a causal co-adaptation mechanism, or formal safety.

#### Step gate

Phase II, Step 7 passes. All 15 same-checkpoint effects, sample SDs, ranges,
signs, and leave-one-checkpoint-out summaries were reconciled. Phase II,
Step 8 is recorded below.

### Phase II, Step 8: transfer-suite performance

Step 8 was executed read-only against the 720 committed rows in
`results/tables/core_layout_transfer/evaluation_episode_results.csv`. The
transfer suite contains three methods, five checkpoints per method, two
projection modes, and 24 prespecified layouts. Each checkpoint-mode-layout
cell contains one deterministic actor-mean episode.

The checkpoint remains the independent training replicate, with \(n=5\) per
method. The 24 layouts are fixed task conditions crossed with all checkpoints.
A per-layout rate is therefore the fraction of the five observed checkpoints
with an outcome in that named layout, not a stochastic episode probability or
a new independent-policy sample.

For terminal indicator \(Y_{m,s,q,\ell}\), checkpoint aggregation and the
paired effect were reconstructed as

\[
\bar Y_{m,s,q}=\frac{1}{24}\sum_{\ell=1}^{24}Y_{m,s,q,\ell},
\qquad
\Delta_{m,s}=\bar Y_{m,s,\mathrm{enabled}}
-\bar Y_{m,s,\mathrm{disabled}}.
\]

The five checkpoint effects were summarized by

\[
\bar\Delta_m=\frac{1}{5}\sum_{s=1}^{5}\Delta_{m,s},
\qquad
s_{\Delta,m}=\sqrt{\frac{1}{4}\sum_{s=1}^{5}
(\Delta_{m,s}-\bar\Delta_m)^2}.
\]

The direct reconstruction was reconciled against:

```text
results/tables/core_layout_transfer/checkpoint_summary.csv
results/tables/core_layout_transfer/method_summary.csv
results/tables/core_layout_transfer/paired_projection_deltas.csv
results/tables/core_layout_transfer/paired_projection_summary.csv
results/tables/core_layout_transfer/result_build_audit.json
experiments/projection_analysis_protocol.json
evaluation/layouts/core_navigation_layouts.json
```

#### Absolute transfer outcomes

| Training condition | Projection | Success mean +/- SD | Collision mean +/- SD | Timeout mean +/- SD |
|---|---|---:|---:|---:|
| PPO baseline | Disabled | 7.50% +/- 16.77 pp | 5.00% +/- 9.03 pp | 87.50% +/- 25.69 pp |
| PPO baseline | Enabled | 7.50% +/- 16.77 pp | 3.33% +/- 7.45 pp | 89.17% +/- 24.22 pp |
| PPO high penalty | Disabled | 0.00% +/- 0.00 pp | 1.67% +/- 3.73 pp | 98.33% +/- 3.73 pp |
| PPO high penalty | Enabled | 0.00% +/- 0.00 pp | 0.00% +/- 0.00 pp | 100.00% +/- 0.00 pp |
| PPO trained with projection | Disabled | 27.50% +/- 3.73 pp | 63.33% +/- 3.49 pp | 9.17% +/- 5.43 pp |
| PPO trained with projection | Enabled | 33.33% +/- 4.17 pp | 12.50% +/- 6.59 pp | 54.17% +/- 7.80 pp |

Baseline success was contributed entirely by seed 3, which succeeded in 9 of
24 layouts in both modes; the other four checkpoints had zero success. No
high-penalty checkpoint succeeded. Projection-trained enabled execution had
the highest success and much lower collision than its disabled counterpart,
but timeout was its dominant enabled terminal outcome.

#### Paired transfer effects

| Training condition | Success delta | Collision delta | Timeout delta |
|---|---:|---:|---:|
| PPO baseline | 0.00 +/- 0.00 pp | -1.67 +/- 2.28 pp | +1.67 +/- 2.28 pp |
| PPO high penalty | 0.00 +/- 0.00 pp | -1.67 +/- 3.73 pp | +1.67 +/- 3.73 pp |
| PPO trained with projection | +5.83 +/- 2.28 pp | -50.83 +/- 9.50 pp | +45.00 +/- 9.95 pp |

Projection-trained checkpoint effects in seed order were:

```text
success:   +4.17,  +8.33,  +4.17,  +4.17,  +8.33 pp
collision: -41.67, -41.67, -50.00, -62.50, -58.33 pp
timeout:   +37.50, +33.33, +45.83, +58.33, +50.00 pp
```

Baseline checkpoint S/C/T effects in seed order were
`0.00/-4.17/+4.17`, `0.00/0.00/0.00`, `0.00/-4.17/+4.17`,
`0.00/0.00/0.00`, and `0.00/0.00/0.00` pp. High-penalty checkpoint effects
were `0.00/-8.33/+8.33` for seed 1 and `0.00/0.00/0.00` for seeds 2 through
5.

All five checkpoints retained positive success, negative collision, and
positive timeout effects. Leave-one-checkpoint-out mean ranges were +5.21 to
+6.25 pp for success, -53.13 to -47.92 pp for collision, and +41.67 to
+47.92 pp for timeout. For baseline, collision decreased in two checkpoints
and was unchanged in three. For high-penalty PPO, it decreased in one and was
unchanged in four. Success was unchanged in all five checkpoints for both
nominally trained methods.

Leave-one-checkpoint-out terminal-outcome mean ranges were:

| Training condition | Success range | Collision range | Timeout range |
|---|---:|---:|---:|
| PPO baseline | 0.00 to 0.00 pp | -2.08 to -1.04 pp | +1.04 to +2.08 pp |
| PPO high penalty | 0.00 to 0.00 pp | -2.08 to 0.00 pp | 0.00 to +2.08 pp |
| PPO trained with projection | +5.21 to +6.25 pp | -53.13 to -47.92 pp | +41.67 to +47.92 pp |

Projection-trained supporting effects were positive in all five checkpoints:

```text
evaluation return:          +3.4877 +/- 0.6047
episode length:            +90.4333 +/- 14.0422 transitions
minimum obstacle clearance: +0.0991 +/- 0.0180 coordinate units
```

The clearance summary excludes the obstacle-free `control_open_route` layout,
where clearance is structurally undefined. Increased episode length is
interpreted jointly with outcomes because many collision-avoiding runs
continued until timeout.

#### Layout breadth and exploratory observations

The breadth of nonzero layout-mean effects was:

| Training condition | Success positive / zero | Collision negative / zero | Timeout positive / zero |
|---|---:|---:|---:|
| PPO baseline | 0 / 24 | 2 / 22 | 2 / 22 |
| PPO high penalty | 0 / 24 | 2 / 22 | 2 / 22 |
| PPO trained with projection | 6 / 18 | 20 / 4 | 15 / 9 |

No layout had a negative mean success effect or a positive mean collision
effect for any method. These counts describe breadth over fixed, potentially
related task conditions and are not a 24-replicate inferential sample.

Under enabled projection-trained execution, 12 layouts had zero success across
all five checkpoints and four layouts had success across all five. The four
all-success layouts were `control_symmetric_clearance`,
`control_upper_clearance`, `double_near_staggered_upper_first`, and
`single_near_late_upper`. All six triple-obstacle layouts had zero success.

Two observations remain exploratory and are not final claims. On obstacle-free
`control_open_route`, two projection-trained checkpoints succeeded and three
timed out in both modes, with projection inactive. Across 11 name-matched
upper/lower layout pairs, enabled projection-trained success averaged 45.45%
for upper variants and 14.55% for lower variants, a descriptive difference of
30.91 pp. These observations do not identify policy specialization,
directional bias, or their mechanism.

#### Separate-suite interpretation

For projection-trained PPO, the fixed-suite paired changes were
+76.2/-80.4/+4.2 pp for success/collision/timeout. The corresponding transfer
changes were +5.83/-50.83/+45.00 pp. Collision reduction persisted under both
protocols, but its net accounting differed: fixed-geometry reduction was
balanced mainly by success, while transfer reduction was balanced mainly by
timeout.

This does not establish that geometry alone caused the difference. The fixed
suite used stochastic action sampling and 100 repeats in one geometry. The
transfer suite used the deterministic actor mean once in each of 24 layouts.
They remain separate estimands.

The transfer result supports broad empirical collision reduction and continued
filter dependence for projection-trained PPO. It also shows that collision
avoidance did not become comparably strong task completion across the tested
transfer suite. It does not establish formal safety, arbitrary geometric
generalization, or a causal policy-projector learning mechanism. Marginal
outcome effects also do not identify paired terminal transitions.

#### Step gate

Phase II, Step 8 passes. Absolute and paired transfer outcomes, all five
checkpoint effects, sample SD, sign consistency, leave-one-checkpoint-out
stability, and layout breadth are verified without modifying frozen evidence.
The next analytical action is Phase II, Step 9: terminal-outcome transition
analysis within each matched checkpoint and evaluation key.

### Phase II, Step 9: paired terminal-outcome correspondences

Step 9 reconstructed which terminal outcome occurred under projection-enabled
execution for every matching projection-disabled evaluation key. Pairing used
method, training seed, checkpoint SHA-256, layout ID, layout repeat, and
evaluation seed. Projection mode was the only excluded key field. The frozen
episode rows were read without rerunning a policy or changing a result table.

For disabled outcome (i), enabled outcome (j), method (m), and checkpoint
(s), the count was

\[
N_{ij}^{(m,s)}=\sum_k
\mathbf{1}\{Y_{m,s,k,\mathrm{off}}=i,
Y_{m,s,k,\mathrm{on}}=j\},
\qquad i,j\in\{S,C,T\}.
\]

Every checkpoint matrix retains all nine cells. Rows are disabled outcomes,
columns are enabled outcomes, and both use the order success, collision,
timeout. Row margins reproduce disabled terminal counts, column margins
reproduce enabled terminal counts, and their differences reproduce the
enabled-minus-disabled paired outcome changes. Method summaries use the five
trained checkpoints as independent units. Pooled counts describe matched
evaluation coverage only.

#### Fixed-geometry checkpoint profiles

Each compact matrix is written as
`[SS, SC, ST; CS, CC, CT; TS, TC, TT]`. Each checkpoint contains 100 matched
episode pairs.

| Training condition | Seed | Complete transition-count profile |
|---|---:|---|
| PPO baseline | 1 | `[8, 0, 0; 1, 1, 8; 0, 0, 82]` |
| PPO baseline | 2 | `[0, 0, 0; 0, 0, 5; 0, 0, 95]` |
| PPO baseline | 3 | `[49, 0, 2; 8, 0, 9; 3, 0, 29]` |
| PPO baseline | 4 | `[16, 0, 1; 3, 0, 11; 1, 0, 68]` |
| PPO baseline | 5 | `[0, 0, 0; 0, 0, 24; 0, 0, 76]` |
| PPO high penalty | 1 | `[0, 0, 0; 0, 0, 7; 1, 0, 92]` |
| PPO high penalty | 2 | `[0, 0, 0; 0, 0, 12; 0, 0, 88]` |
| PPO high penalty | 3 | `[0, 0, 0; 0, 2, 14; 0, 0, 84]` |
| PPO high penalty | 4 | `[0, 0, 0; 0, 1, 22; 0, 0, 77]` |
| PPO high penalty | 5 | `[0, 0, 0; 0, 0, 9; 0, 0, 91]` |
| PPO trained with projection | 1 | `[5, 0, 1; 73, 5, 14; 0, 0, 2]` |
| PPO trained with projection | 2 | `[12, 0, 0; 84, 1, 3; 0, 0, 0]` |
| PPO trained with projection | 3 | `[21, 0, 1; 75, 0, 2; 1, 0, 0]` |
| PPO trained with projection | 4 | `[24, 0, 0; 75, 0, 1; 0, 0, 0]` |
| PPO trained with projection | 5 | `[22, 1, 0; 75, 0, 1; 1, 0, 0]` |

The pooled fixed-geometry matrices, each over 500 matched pairs, were:

| Training condition | Pooled 3 by 3 matrix |
|---|---|
| PPO baseline | `[[73, 0, 3], [12, 1, 57], [4, 0, 350]]` |
| PPO high penalty | `[[0, 0, 0], [0, 3, 64], [1, 0, 432]]` |
| PPO trained with projection | `[[84, 1, 2], [382, 6, 21], [2, 0, 2]]` |

The projection-trained result resolves the Step 7 marginal accounting. Of 409
disabled collisions, 382 corresponded to enabled success, 21 to enabled
timeout, and six to enabled collision. The collision-to-success proportion was
therefore 382/409, or 93.4%. Its count was large in every checkpoint: 73, 84,
75, 75, and 75. Most of the fixed-geometry collision reduction was thus paired
with successful completion under the composite controller.

The matrix also prevents an absolute preservation claim. Among 87 disabled
successes for projection-trained PPO, 84 remained successes, one corresponded
to enabled collision, and two corresponded to enabled timeout. Two of four
disabled timeouts corresponded to enabled success and two remained timeouts.

The nominally trained methods showed a different correspondence. Baseline
disabled collisions mapped to success in 12 of 70 cases, timeout in 57, and
collision in one. High-penalty disabled collisions mapped to success in zero
of 67 cases, timeout in 64, and collision in three. Their fixed-suite collision
reductions were therefore associated mainly with noncompletion rather than
success.

#### Transfer checkpoint profiles

Each transfer checkpoint contains 24 matched layout pairs.

| Training condition | Seed | Complete transition-count profile |
|---|---:|---|
| PPO baseline | 1 | `[0, 0, 0; 0, 0, 1; 0, 0, 23]` |
| PPO baseline | 2 | `[0, 0, 0; 0, 0, 0; 0, 0, 24]` |
| PPO baseline | 3 | `[9, 0, 0; 0, 4, 1; 0, 0, 10]` |
| PPO baseline | 4 | `[0, 0, 0; 0, 0, 0; 0, 0, 24]` |
| PPO baseline | 5 | `[0, 0, 0; 0, 0, 0; 0, 0, 24]` |
| PPO high penalty | 1 | `[0, 0, 0; 0, 0, 2; 0, 0, 22]` |
| PPO high penalty | 2 | `[0, 0, 0; 0, 0, 0; 0, 0, 24]` |
| PPO high penalty | 3 | `[0, 0, 0; 0, 0, 0; 0, 0, 24]` |
| PPO high penalty | 4 | `[0, 0, 0; 0, 0, 0; 0, 0, 24]` |
| PPO high penalty | 5 | `[0, 0, 0; 0, 0, 0; 0, 0, 24]` |
| PPO trained with projection | 1 | `[6, 0, 0; 1, 4, 9; 0, 0, 4]` |
| PPO trained with projection | 2 | `[6, 0, 0; 2, 5, 8; 0, 0, 3]` |
| PPO trained with projection | 3 | `[8, 0, 0; 1, 3, 11; 0, 0, 1]` |
| PPO trained with projection | 4 | `[6, 0, 0; 1, 1, 14; 0, 0, 2]` |
| PPO trained with projection | 5 | `[7, 0, 0; 2, 2, 12; 0, 0, 1]` |

The pooled transfer matrices, each over 120 matched pairs, were:

| Training condition | Pooled 3 by 3 matrix |
|---|---|
| PPO baseline | `[[9, 0, 0], [0, 4, 2], [0, 0, 105]]` |
| PPO high penalty | `[[0, 0, 0], [0, 0, 2], [0, 0, 118]]` |
| PPO trained with projection | `[[33, 0, 0], [7, 15, 54], [0, 0, 11]]` |

All transfer off-diagonal correspondences originated from disabled collisions.
Every disabled success remained a success, and every disabled timeout remained
a timeout. For projection-trained PPO, 7 of 76 disabled collisions corresponded
to enabled success, 54 corresponded to enabled timeout, and 15 remained
collisions. These conditional proportions were 9.2%, 71.1%, and 19.7%,
respectively. Collision-to-success occurred in only one or two layouts per
checkpoint, whereas collision-to-timeout occurred in 8 to 14.

Baseline disabled collisions produced zero successes: four of six remained
collisions and two corresponded to timeout. Both high-penalty disabled
collisions corresponded to timeout. The transition analysis therefore
strengthens the Step 8 statement that collision reduction transferred more
broadly than successful task completion under the tested protocol.

#### Reconciliation and interpretation boundary

The fixed output contains 1,500 matched pairs and the transfer output contains
360. In both suites, all integer row and column margins were exact, all outcome
count deltas summed to zero, and 270 comparisons against the checkpoint,
method, paired-delta, and paired-summary hierarchy passed. The largest
floating-point discrepancy was (1.11\times10^{-16}), below the declared
(10^{-12}) tolerance. All 21 focused interpretation tests passed.

The row-to-column arrow is a descriptive correspondence between matched
disabled and enabled controller executions. It is not a temporal transition
within one trajectory because the executions may diverge after their actions
differ. The matrices do not establish deliberate actor reliance, a causal
policy-projector co-adaptation mechanism, formal safety, performance under a
different projector, or arbitrary geometric generalization. Cross-suite
differences also cannot be assigned to geometry alone because action-selection
mode and repetition structure differ.

#### Step gate

Phase II, Step 9 passes. Every checkpoint profile and every matrix cell was
retained, reconciled, and interpreted with the trained checkpoint as the
independent unit. The next scientific step is Phase II, Step 10: attribute the
verified evidence carefully to nominal-policy behavior and policy-plus-
projector composite-controller behavior. Step 10 requires a separate purpose,
source, procedure, assumptions, and authorization before execution.

## Analysis still to add

- Nominal-policy versus composite-controller attribution in Phase II, Step 10.
- Final separation of supported conclusions, limitations, and follow-up hypotheses.
- Scientific interpretation, empirical technical-report drafting, and final
  report traceability audit.


## Completed-repository handoff, 7 September 2026

The final source, evidence-inclusion, documentation, and verification decisions
are recorded in [repository release audit](repository_release_audit.md). The
completed empirical findings and inferential boundaries are consolidated in the
[final study record](final_study_record.md). Read-only interpretation outputs and
the exact reconstruction argument arrays are retained under
`results/interpretation/`. These reconstructions use the same frozen input data;
no final training, policy evaluation, calibration, or benchmark was repeated.

For the completed release, run the full tests and
`python -m evaluation.verify_repository_release --base-ref main`. After staging,
add `--require-tracked` to verify the exact file inventory and staged content.
The historical pre-experiment runner remains available for future development
and is not the merge gate for this completed study.
