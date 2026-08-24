# Predictive Action Projection with PPO: Analysis Record

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

3. **A larger budget may help some baseline runs, but budget alone is not established as the remedy for failed seeds.** Seed 3 was still improving at the cutoff, yet seeds 1, 2, and 5 show no positive late success trend. The evidence supports a controlled budget-only follow-up, not a claim that longer training will remove seed variability.

4. **High-penalty PPO exhibits no late goal learning in any checkpoint.** Success is exactly zero in both late windows for all five seeds. Return changes are heterogeneous, including a large seed-4 decline, and cannot be interpreted as navigation progress. Together with the aggregate collision-to-timeout shift, this is more consistent with conservative reward optimization than with demonstrated budget insufficiency.

5. **Projection-trained PPO is consistently competent and still improving in four checkpoints.** Seeds 1, 2, 3, and 5 gain both return and success; seed 4 is effectively at a ceiling near 97% success. The method is not universally plateaued, but the frozen budget was already sufficient for a strong composite controller.

6. **Additional training targets differ by method.** For projection-trained PPO, more transitions may improve already-strong protected performance but do not directly address nominal-actor dependence or transfer. For baseline PPO, a budget-only experiment remains diagnostically meaningful. For high-penalty PPO, the frozen evidence gives no sign that budget alone will create goal completion.

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

## Analysis still to add

- Final separation of supported conclusions, limitations, and follow-up hypotheses.
- Documentation-only release commit, pull-request merge, visibility verification, and paper-results drafting.
