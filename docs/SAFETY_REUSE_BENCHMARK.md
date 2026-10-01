# Three-objective certified-reuse benchmark

Run from the SubRep directory, using the existing `subrep-safety` conda environment.
This runner is independent of the older development and full-evaluation scripts.
It does not retrain or overwrite checkpoints or previous reports.

## Smoke test

```bash
conda run --no-capture-output -n subrep-safety \
  python -m demo.run_safety_reuse_benchmark \
  --output data/safety_reuse_smoke \
  --development-contexts 2 --contexts 2 --batches 1 \
  --max-steps 5 --bootstrap-samples 50 --seed 900000
```

Smoke results test execution only. Five steps cannot establish navigation quality.
Output directories must be new; use a different name for a rerun.

## Full experiment

```bash
conda run --no-capture-output -n subrep-safety \
  python -m demo.run_safety_reuse_benchmark \
  --output data/safety_reuse_fixed_e02 \
  --training-seeds 42 43 44 \
  --development-contexts 30 --contexts 30 --batches 5 \
  --max-steps 200 --epsilon 0.2 --bootstrap-samples 2000 --seed 60000
```

Default checkpoints are `models/safety_ppo_fixed_seed{seed}_updates50.pt` and
`models/safety_ppo_lagrangian_fixed_seed{seed}_updates50.pt`. Override
`--ppo-template` and `--lagrangian-template` to use other files. The runner
validates checkpoint seed, policy type, environment, and observation/action
compatibility, rejects identical parameters across training seeds, and records
checkpoint hashes and training settings. It does not infer training correctness
from a filename or enforce a particular policy performance.

The full experiment has three separate training-seed pools, each with eight
candidate policies (six simple behaviors, PPO, PPO-Lagrangian) plus idle. It
collects up to 1,782,000 environment steps: three training seeds × nine policies ×
(30 development + five batches × 60 evidence/evaluation contexts) × 200 steps.
Every method shares these paired policy outcomes; episodes are not rerun for each
method or priority scenario. Equivalent selections use exactly the same outcome.

## Protocol

1. Pool development measurements from all three training-seed pools to fit common
   positive scales (mean absolute improvement magnitudes over idle). Constant-zero
   objectives use scale one. Write the frozen scales before admission/evaluation.
2. For each training seed and batch, collect new admission evidence. Certify
   policy mean improvements using CDS, then PDS with the configured epsilon.
3. Construct actual `Certificate` objects and insert executable policies through
   `SkillLibrary.add_skill()`, including mathematical reverification. Save the
   library and certificates as JSON. Hyperon/MeTTa is not required for this runner.
4. Query `SkillLibrary.query_admissible()` for every priority. Rank the resulting
   entries using the production library selection helper. Save all method
   decisions before collecting any held-out evaluation outcomes.
5. Execute every fixed candidate policy on matching held-out environment seeds.
   Selected methods receive the saved outcome of their preselected policy. This
   is a paired fixed-policy reuse experiment, not an adaptive rollout controller.
6. Summarize balanced, safety-, task-, effort-focused, gradual, and abrupt
   priorities, plus a separate deliberately empty-library control.

Seven methods: SubRep best certified, always PPO, always PPO-Lagrangian,
unrestricted best, random candidate, random certified, and idle. Random draws
are reproducible. Unrestricted best uses the same evidence and score as SubRep,
but skips certification; both exclude idle from their candidate pools. Certified
methods execute idle if no certified skill is available. Thus the ablation
measures certification together with its specified fallback behavior.

The evidence and evaluation seed ranges are disjoint from each other and from
development. They are shared across training seeds to retain paired environment
comparisons. The default full range starts at 60001, separate from the earlier
40001–40030 policy-development run. Reserve these final seeds: do not select
checkpoints, epsilon, scales, or training settings using the final results.
Epsilon 0.2 is a declared normalized score budget, not a failure probability or
a tuned guarantee. Set another value before evaluation if required by the study.

## Outputs and interpretation

- `manifest.json`: configuration, exact split seeds, checkpoint/source hashes,
  dependency versions and code revision/dirty status.
- `frozen_scales.json`: common objective divisors from development only.
- `seed_*/development.json`: raw development measurements.
- `seed_*/batch_*/evidence.json`: admission episodes.
- `seed_*/batch_*/certificates.json`, `library.json`: actual admitted records.
- `seed_*/batch_*/selection_plan.json`: gate audit, selected IDs, weights,
  evidence scores, certificate thresholds, eligibility, and abstention.
- `seed_*/batch_*/evaluation.json`: every policy's held-out measurements.
- `seed_*/batch_*/results.json`: per-method realized outcomes.
- `admissions.json`: CDS/PDS counts, admission/rejection counts and rates.
- `summary.json`, `report.md`: main results and uncertainty estimates.

Three motive coordinates are negative hazard cost, task reward, and negative
mean squared normalized action. Values are discounted cumulative episode totals;
undiscounted hazard costs are also reported. The scalar payoff is task reward,
so task contributes twice in `delta_r + w dot delta_n`. State this convention
when interpreting relative priority strengths. Goal success is a native
`goal_met` event during the rollout, not positive reward or termination.

Report fallback separately from actual certified reuse. `certified_only` metrics
are null when no certified execution occurred. Negative transfer means score
below zero; common budget violation means below `-epsilon`. Certificate violation
uses the selected skill's own threshold (zero for CDS, `-epsilon` for PDS), and is
reported only for the certified-selection methods' non-fallback executions.
A negative PDS score within its budget is not a certificate violation.

Confidence intervals resample training seeds and environment batches/contexts
while keeping paired method differences and shared environment samples aligned.
The paired difference is method minus SubRep: negative favors SubRep; an interval
spanning zero is inconclusive. Three training seeds give limited precision.
Intervals are conditional on the frozen training configuration and development
scales. Priority scenarios reuse episodes and are not independent experiments.

Certificates apply to empirical policy means, not each state or unseen episode.
FULL_SIMPLEX certification remains eligible for all tested weights. These results
do not establish state-conditioned retrieval, a learned MDN, within-episode skill
switching, MetaMo integration, or absolute safety guarantees. A run is valid even
if every candidate is rejected or SubRep does not win. The empty-library control
must not be counted as successful skill reuse.

## Tests

```bash
.venv/bin/python -m pytest tests/test_safety_reuse_benchmark.py \
  tests/test_certification_gates.py tests/test_skill_library.py -q
```
