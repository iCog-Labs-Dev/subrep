# Three-objective held-out benchmark

Run from the SubRep root. The runner collects development data, freezes scales,
then builds a separate empirical admission pool per batch using fresh evidence.
It saves all method selections before collecting disjoint evaluation episodes.
No held-out outcomes are used for ranking or admission. Every policy runs on the
same held-out context seeds, enabling paired method comparisons. This evaluates
fixed policies selected from batch-level evidence, not context-conditioned skill
retrieval or a live MetaMo/Omega controller.

```bash
.venv/bin/python -m pytest tests/test_safety_full_benchmark.py tests/test_safety_three_pipeline.py tests/test_safety_objectives.py tests/test_safety_gymnasium_adapter.py tests/test_safety_gymnasium_pipeline.py tests/test_safety_gymnasium_certification_ablation.py tests/test_safety_gymnasium_pareto.py tests/test_safety_gymnasium_reuse_curve.py tests/test_executor.py -q

conda run --no-capture-output -n subrep-safety python -m demo.run_safety_full_benchmark --output data/safety_full_smoke --contexts 2 --batches 1 --max-steps 5

conda run --no-capture-output -n subrep-safety python -m demo.run_safety_full_benchmark --output data/safety_full_evaluation --contexts 20 --batches 5 --max-steps 200 --seed 2000 --epsilon 0
```

Output directories must not already exist. For reruns, choose another name.
All stages use the same candidate policies and episode horizon. Optional
`--ppo-checkpoint PATH` and `--ppo-lagrangian-checkpoint PATH` add fixed pretrained
policies in every split, including development. Without these, only built-in
policies are used; no training occurs. Checkpoint hashes are recorded. Do not
select checkpoints using evaluation results.

Five methods: best admissible, random candidate, random admissible, zero-action
baseline, and best unrestricted. Random choices are seeded. Non-baseline policies
constitute the candidate pool. Certified methods abstain and execute the baseline
when the admission pool is empty. Scenarios cover balanced priorities, each
objective focus, gradual and abrupt safety-to-task shifts, and a deliberately
empty library control. Shifts happen between reset episodes; they do not simulate
within-episode adaptation. FULL_SIMPLEX admission remains eligible under all of
these weights. No learned MDN region is claimed.

Inspect:

- `manifest.json`: settings, versions, checkpoint hashes, commit and dirty status.
- `frozen_scales.json`: scales fitted only on development deltas.
- `admission_summary.json`: per-batch admission/rejection counts and rates.
- `batch_*/selection_plan.json`: per-policy gate audit and frozen decisions.
- `batch_*/results.json`: eligibility, abstention, evidence selection validity,
  actual held-out returns for all three objectives, and threshold violations.
- `summary.json`: per-seed means, between-seed standard deviation, paired score
  differences versus SubRep, and raw objective means.
- `report.md`: readable score and failure summary.

The score is normalized task payoff plus weighted normalized motives; task
performance contributes twice by the existing convention. Epsilon is in these
normalized units. Certification is mathematical checking of empirical mean
improvements over zero action. It does not guarantee improvement on new episodes:
held-out violations must be reported. A violation is not automatically a gate
implementation bug. Sample-size uncertainty and policy-training seed variation
are not resolved by these environment-seed batches.

The default full run collects 220 contexts across development/evidence/evaluation
and seven built-in policies: up to 308,000 environment steps. A smoke run tests
plumbing only; its short horizon is not useful benchmark evidence. Do not require
a particular method to win or a nonempty admitted set for a run to count as
successfully completed.

## PPO with PDS epsilon 0.2

The `ppo-pds` CLI preset adds the existing
`models/safety_ppo_point_goal_seed42_updates50.pt` policy to the seven built-in
policies in every split and defaults epsilon to 0.2. Explicit `--epsilon` and
`--ppo-checkpoint` values override the preset. The simple preset retains epsilon
zero. The CLI now defaults to seed 10000 to avoid the initial seed-2000 run.

```bash
conda run --no-capture-output -n subrep-safety python -m demo.run_safety_full_benchmark --preset ppo-pds --output data/safety_ppo_pds_e02 --contexts 20 --batches 5 --max-steps 200 --seed 10000
```

Scales are refitted using fresh development data including PPO, then frozen.
Do not compare normalized scores directly against the old simple-policy run,
which used different scales. Compare methods within this new run. Epsilon 0.2
allows a worst-case empirical score down to -0.2 in normalized units; it is not
a 20% probability of failure. CDS still requires a nonnegative worst-case score.
PPO inclusion and relaxed PDS do not guarantee admission or task success.
The checkpoint is fixed across batches; these are environment-seed comparisons,
not independent PPO training runs. The short smoke test confirms compatibility,
not that the checkpoint is a strong navigator.

## Compare policies before another benchmark

```bash
conda run --no-capture-output -n subrep-safety python -m demo.evaluate_safety_policies --output data/safety_policy_development --contexts 30 --max-steps 200 --seed 30000
```

This evaluates every `models/safety_ppo*.pt` checkpoint, plus built-in policies,
on matching new development contexts. Repeat `--checkpoint PATH` to evaluate a
subset. Existing output directories are rejected. It records checkpoint hashes,
raw rollouts, a JSON summary with variation and paired improvements over idle,
and a Markdown report. Costs and effort are positive quantities here; lower is
better. Measurements are discounted episode totals, not physical energy or
per-step means. There is no composite score or gate filtering. Checkpoint
selection based on this report is development work: do not reuse these seeds
for final evaluation. Incompatible checkpoints fail explicitly.
