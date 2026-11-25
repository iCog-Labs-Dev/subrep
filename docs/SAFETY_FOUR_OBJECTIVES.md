# Four-objective Safety-Gymnasium benchmark

## Objectives and scope

Use `SafetyPointGoal1-v0` with a fixed episode horizon (200 steps in these
examples). The four motive coordinates are, in order:

1. **Safety:** negative environment cost.
2. **Task:** environment reward.
3. **Control efficiency:** negative mean squared normalized applied action.
4. **Action smoothness:** negative mean squared half-difference between
   consecutive normalized applied actions; the first step contributes zero.

Effort is an action-magnitude proxy, not measured energy. Smoothness measures
action changes, not physical jerk. The collector discounts each motive with
its configured gamma. Policy evaluation also reports undiscounted episode cost,
effort, and action-change totals: do not interchange these quantities.

The architecture supports a variable objective dimension in the data/MDN path.
This does not establish that a trained policy succeeds under all four objectives.
First establish useful task/safety behavior, then investigate action penalties.
Keep the original runs as controls and measure all four objectives throughout.

## Environment

Run from the repository root in the separate Safety-Gymnasium environment:

```bash
conda activate subrep-safety
python -m pip install -r requirements-safety.txt
```

The main `.venv` can run tests that use fake simulators. Simulator training uses
the dependencies in `requirements-safety.txt`. Neither environment is committed.

## Controlled actor comparison

Train both actors from scratch with the same settings. `linear` removes the
output tanh on the Gaussian mean; actions sent to the environment are still
clipped. Raw sampled actions are retained for PPO log-probabilities. `tanh`
remains the default, including for older checkpoints without an actor-type field.
A checkpoint cannot be warm-started into a different actor type.

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
for ACTOR in tanh linear; do
  python -m pilot.train_safety_gymnasium_ppo \
    --actor-mean-type "$ACTOR" \
    --output "models/safety_actor_${ACTOR}_${RUN_ID}.pt" \
    --use-lagrangian --seed 42 --total-updates 200 \
    --learning-rate 0.0003 --cost-limit 1 --lambda-lr 0.005 \
    --initial-lagrange-multiplier 0.01 \
    --effort-penalty 0 --smoothness-penalty 0 \
    --max-episode-steps 200 --eval-every 25 --eval-episodes 100 \
    --eval-seed 420000 --device cpu || break
done
```

For the unconstrained PPO baseline, omit `--use-lagrangian` and explicitly set
`--cost-penalty 0`; use a different output path. Action penalties are optional
training settings, not extra hard constraints. Add them only as a separately
labelled experiment. A 200-update run is a development comparison, not a
convergence or safety guarantee. Repeat promising settings across training seeds.

The selected policy is written to `--output`. The sibling `<stem>_run/` directory
contains metrics, summary, and evaluation checkpoints including update 0.
Lagrangian selection prioritizes development budget feasibility; if no snapshot
is feasible, it chooses lower cost. A low-cost policy can still fail to reach
goals. Always inspect both metrics. Saved policies are inference checkpoints,
not complete optimizer/cost-critic states for training resumption.

## Diagnostics

- `--init-checkpoint`: copy compatible actor and reward critic; optimizers and
  the cost critic start fresh.
- `--cost-warmup-updates`: train the cost critic on extra rollouts with actor and
  multiplier fixed, before the requested number of policy updates.
- `--actor-learning-rate`: separate actor rate in Lagrangian mode.
- `--cost-only-diagnostic`: optimize only cost, freeze the reward critic and
  multiplier, and compare sampled/deterministic behavior. It is not a final
  goal-reaching training objective.
- `demo.audit_safety_actor`: inspect mean sensitivity, clipping, and raw-action
  probability consistency on an explicitly selected checkpoint.
- `demo.screen_safety_contexts`: screen layouts, then verify selected layouts
  using new action-noise seeds. Reports empirical suitability of a sampled
  policy on exact layouts, not a statistical certificate or deterministic-policy
  guarantee.

Passing tests or detecting saturation does not prove why a particular run failed.
Warm-up, learning rates, and actor type remain experimental controls.

## Collect local data

Use explicit paths to your chosen PPO and Lagrangian policies:

```bash
python -m data_collector.collect_safety_scaling \
  --output data/safety_four_development \
  --ppo-checkpoint models/my_ppo.pt \
  --lagrangian-checkpoint models/my_lagrangian.pt \
  --contexts 300 --seed 410000 --training-seed 42 --max-steps 200
```

This writes raw evidence, provenance, `objectives4/` candidate NPZs,
`scales4.json`, and `benchmark_contract4.json`. Matched `objectives3/` projections
are also exported for optional dimension comparisons. Development scales use
mean absolute baseline deltas; a zero-magnitude coordinate uses divisor 1.
Final evaluation must not refit scales on its outcomes.

## Exact-context benchmark

```bash
python -m demo.run_contextual_safety_benchmark \
  --output data/safety_four_evaluation \
  --objectives 4 --contexts 300 --development-contexts 10 \
  --seed 520000 --training-seeds 42 --max-steps 200 --epsilon 0.2 \
  --ppo-template models/my_ppo.pt \
  --lagrangian-template models/my_lagrangian.pt \
  --scales-file data/safety_four_development/scales4.json
```

Choose unused evaluation seeds after development. This command runs full-simplex
gates without an MDN. For MDN-region evaluation, train a compatible four-objective
model and supply `--mdn-checkpoint` plus the associated `--mdn-contract`; do not
reuse a three-objective support head. The contract records the environment,
native observation schema, and exact objective order.

The contextual runner reports admission and re-execution from exact starts.
It does not establish transfer across layouts. The older held-out batch runners
and three-objective plots have separate scopes; do not label their output as a
four-objective result without verifying the analysis supports four dimensions.

## Validation

```bash
python -m pytest -q \
  tests/test_safety_scaling.py tests/test_contextual_safety_benchmark.py \
  tests/test_safety_ppo_training.py tests/test_safety_warm_start.py \
  tests/test_safety_actor_mean.py tests/test_safety_cost_learning_audit.py \
  tests/test_screen_safety_contexts.py
```

Tests cover implementation contracts. Claims of safe and successful reuse need
fresh execution evidence, budget checks, and uncertainty reporting. Keep outputs
local and report the policy library's limitations, including rejected candidates
and fallbacks, rather than tuning thresholds to force acceptance.

## Development readiness before MDN training

`python -m demo.evaluate_safety_readiness` compares an explicit list of checkpoints
on identical reset contexts, with zero action as the reference. Repeat
`--checkpoint /path/to/policy.pt` for each trained policy. Supply `--output` with
an unused directory; `--seed 610000 --development-contexts 30 --contexts 100
--max-steps 200 --epsilon 0.2 --cost-limit 1` is one example configuration.
Verify that these seeds are unused in your previous runs before treating them as
fresh. The first 30 seeds fit scales; the following 100 compare policies.

The report records undiscounted goal success, cost, return, effort and action
change, and the existing full-simplex CDS/PDS decision for every policy/context.
Certification uses discounted scaled improvements against zero action. Acceptance
is empirical and specific to the measured start, not a confidence guarantee or
transfer result. MDN-region certification is unavailable until a compatible MDN
exists. Raw records and frozen scales are saved for inspection.

The admission denominator includes only requested policies, excluding idle and
heuristics. Context coverage separately measures whether any requested policy
passes. The joint outcome column additionally requires reaching a goal and meeting
the episode cost threshold; this is different from an average-cost constraint.
Do not tune epsilon, scales or context selection simply to achieve 60% admission.

## Collect and train the four-objective MDN

`demo.run_four_objective_mdn` accepts repeated `--checkpoint` arguments, retains
all supplied policies plus zero action, and runs collection, MDN training and
margin support training. It does not retrain PPO or change certification gates.
Use `--stage collect` or `--stage train` to separate collection and training;
`all` is the default. Training refuses to overwrite an existing `mdn/` or
`support/` directory. This is not an optimizer-resume command.

```bash
python -m demo.run_four_objective_mdn \
  --checkpoint /path/to/ppo.pt \
  --checkpoint /path/to/lagrangian.pt \
  --output "data/four_objective_mdn_$(date +%Y%m%d_%H%M%S)" \
  --context-seed 620000 --scale-contexts 30 --contexts 300 \
  --max-steps 200 --epsilon 0.2 --regularization 0.1 \
  --support-epochs 500 --seed 42
```

Use unused context seeds. Scales are fitted only on the first collection; the
following disjoint contexts supply candidate data. NPZ values remain raw while
scale metadata makes the MDN adapter apply the same divisors used by the support
trainer, including the task payoff divisor. All four objectives are inferred by
the MDN trainer. The support stage freezes the representation and trains only
the support head with the existing optional margin loss.

Outputs include `manifest.json`, raw evidence, `scales.json`, `candidate_sets/`,
`mdn/auxiliary.pth`, `support/support.pth`, and `benchmark_contract.json`. These
are ignored local artifacts. Trainers use internal context-group splits; these
are development diagnostics, not independent end-to-end test evidence, because
the representation is already fitted on this collection. Final regional
certification must use unused contexts and the saved scales/contract. Full-simplex
passes take precedence; additional MDN-region passes are possible, not guaranteed.

## Compare the trained support head on unused contexts

```bash
python -m demo.compare_four_objective_support \
  --run-dir data/YOUR_FOUR_OBJECTIVE_MDN_RUN \
  --output "data/four_objective_support_comparison_$(date +%Y%m%d_%H%M%S)" \
  --contexts 300 --seed 630000 --fixed-caps 0.8 0.8 0.8 0.8 --cost-limit 1
```

The runner loads all checkpoint policies from the training manifest, verifies
policy and scale hashes, reads epsilon from the support checkpoint, and rejects
context seeds overlapping the scale/training collection. Choose seeds unused in
other experiments too; only the named training run can be checked automatically.
No checkpoints are retrained and no scales are refitted.

The three methods are full simplex, full simplex plus a predeclared fixed region,
and full simplex plus the learned region. Global admissions take precedence in
all methods. Extra regional admissions are counted separately. Fixed-region
results are audit records, not labelled MDN predictions. Each context stores
`full_simplex_library.json` and `full_plus_mdn_library.json`; learned regional
passes have `MDN_WX` certificates with context, alpha, support geometry and margin.
No learned regional passes is a valid possible result.

Selection is performed from evidence before same-start re-execution. Five fixed
priority profiles use each method's eligible skills, with zero action as fallback.
Regional skills are excluded outside their region. The report counts admissions,
CDS/PDS gates, context coverage, uniform-simplex region volume, fallback, goal
success, cost, effort, smoothness and joint admitted/goal/episode-budget outcomes.
These exact-start empirical certificates are not statistical safety guarantees.
Deterministic re-execution is not an independent replicate. This is a support
comparison, not the complete baseline/uncertainty study for publication.

## Optional priority-aware support improvement

The original margin loss can admit skills only inside regions that no requested
priority visits. `generator.train_mdn_margin_support --loss-mode priority-reuse`
adds an opt-in alternative. It uses a declared JSON array of simplex priorities
(`configs/safety_four_objective_priorities.json` contains the existing balanced
and four 80%-emphasis profiles). It never reads comparison/evaluation outcomes.

For each training context and priority, it finds candidates that could pass at
that priority and score better than the current full-simplex selection. The loss
combines priority containment with the smallest exact-region margin shortfall
among those candidates, plus the existing full-simplex anchor. Global certificates
remain available. Validation chooses the largest empirical scalarized reuse gain,
then the lower loss. Only support-head parameters change; epsilon, gate formulas,
policy checkpoints and objective scales remain fixed. This is an implementation
choice, not a claim of implementing a paper-prescribed loss.

The logs record potential improvement, priority coverage, improved-query rate,
and reuse gain. Zero achieved gain is possible even if some pointwise opportunity
exists: one context-dependent capped region may not represent all useful regions.
Positive scalarized gain does not imply lower absolute cost or more goals; those
must be evaluated separately. Grouped head validation is still development-only
because the representation has seen this collection.

Start from the previous support checkpoint and save into a new output directory.
Use the original raw candidate sets, frozen scale JSON, and its task divisor as
`--payoff-scale`. Example additional training options:

```bash
--loss-mode priority-reuse \
--priority-file configs/safety_four_objective_priorities.json \
--priority-weight 1 --eligibility feasible --epsilon 0.2 --regularization 0.1
```

Then run `demo.compare_four_objective_support` with the original `--run-dir` and
`--support-checkpoint PATH_TO_NEW_SUPPORT/support.pth`. Use new evaluation seeds
(e.g. 650000 if unused), retain the same fixed-priority profiles, and compare
actual selections/outcomes with full-simplex and fixed-region baselines. Earlier
comparison contexts are now development evidence and should not serve as the
final evaluation of this changed method. Do not widen a certified region after
certification or bypass a failed gate merely to increase reuse.

## Four report figures from saved comparison evidence

Generate four PNG figures (plus vector PDF copies) without simulation or retraining:

```bash
python -m utils.plot_four_objective_comparison \
  --benchmark-dir data/four_objective_priority_comparison_20261125_115918 \
  --output-dir "data/four_objective_figures_$(date +%Y%m%d_%H%M%S)" \
  --bootstrap-samples 2000 --seed 42
```

The input directory must contain `support_comparison.json` and `manifest.json`
from the four-objective priority comparison. Use an environment with NumPy and
Matplotlib. The output directory must be new. The generated `README.md` contains
captions and suggested report placements; `figure_statistics.json` records
plotted values, source hashes, and uncertainty settings.

1. Certification and actual reuse: after Pilot Results.
2. Task performance and safety–control trade-offs: under Safety-Cost Result.
3. Outcomes under changing priorities: after Baseline Comparison.
4. Learned-region coverage and regional selection: under Interpretation.

Outcome intervals resample whole contexts with all methods and priorities kept
paired. They describe context variation for the fixed models, not variation
across training seeds. Idle fallback is included in outcome averages. The cost
budget is a separate diagnostic, not an absolute safety guarantee supplied by
regional certification. Preference-region volume does not measure environmental
safety coverage. These comparisons evaluate region strategies within SubRep;
they do not establish superiority over standalone PPO or PPO-Lagrangian.

## Eight-candidate protocol

`demo.run_four_objective_mdn --include-fixed-controls` requires exactly four
checkpoint policies and adds `axis0_positive`, `axis0_negative`, `axis1_positive`,
and `axis1_negative` with constant actions `[0.35,0]`, `[-0.35,0]`, `[0,0.35]`,
and `[0,-0.35]`. Zero action remains the separate baseline. The manifest records
candidate order and action values; comparison reconstructs and validates that
pool. Existing checkpoint-only manifests remain supported.

Use a new collection directory, separate scale/training contexts, and optionally
`--support-loss-mode priority-reuse` to use the predefined priority-aware loss.
The default remains the original margin loss. Changing the pool requires new
candidate data and MDN/support training, not new PPO training.

Evaluate with `demo.compare_four_objective_support --contexts 300` and an unused
seed range. This gives 2,400 non-baseline pairs per method, 2,700 outcomes per
collection including idle, and 1,500 priority queries per method. Evidence and
re-execution each collect outcomes, so 2,700 is not the total simulator rollout
count. The evaluation manifest records consecutive groups of up to 100 contexts;
these are reporting groups, not independent policy-training seeds.

This matches the earlier two-objective evaluation size, but its candidate mix
(two trained plus six simple policies) differs from this four-trained/four-fixed
pool. Objective scaling, epsilon, and training differences also prevent attributing
cross-report differences solely to objective count.

## Offline standalone baselines and certification ablation

After comparison, use saved evidence and execution outcomes:

```bash
python -m utils.analyze_four_objective_baselines \
  --benchmark-dir data/four_objective_eight_policy_20261125_160526_comparison \
  --output-dir "data/four_objective_baselines_$(date +%Y%m%d_%H%M%S)" \
  --ppo-policy safety_ppo_four_objective_seed42_v1 \
  --lagrangian-policy safety_ppo_lagrangian_four_objective_seed42_v1 \
  --bootstrap-samples 2000 --seed 42
```

Policy arguments must be exact saved candidate names. Choose them before viewing
baseline results; do not silently substitute the best-performing Lagrangian
variant. The command reconstructs every saved selection from evidence and fails
if it differs. Outcomes are measured from execution only. It reports all three
region strategies against always-PPO, always-PPO-Lagrangian, idle, uniform random
candidate/certified selection, and unrestricted selection. Random baselines are
exact expectations across saved outcomes, not newly simulated random policies.

The output includes raw four-objective outcomes, goal/budget outcomes, filtering
costs, descriptive four-objective Pareto membership, priority reuse success and
paired context-bootstrap differences. Reuse success requires a nonfallback
certified selection scoring at least as well as idle. This is distinct from goal
success. Removing certification removes both admission and region restrictions;
the report separates gate-rejected from outside-region selections. It is not an
isolated CDS-only versus PDS-only ablation. All priority comparisons reuse the
same frozen library per context, between reset episodes; no within-episode
adaptation or independent safety guarantee is established. Full results, rows,
source hashes and uncertainty settings are in `analysis.json`; tables are in
`report.md`. No simulator or policy retraining is required.

After the offline analyser has written `end_to_end_analysis/analysis.json`, generate
Pareto projections and an exploratory frozen-library preference sweep:

```bash
python -m utils.plot_four_objective_extensions \
  --benchmark-dir data/four_objective_eight_policy_20261125_160526_comparison \
  --output-dir "data/four_objective_extra_figures_$(date +%Y%m%d_%H%M%S)" \
  --points-per-leg 20
```

This reads the baseline analysis at the specified location and verifies its source
hashes. Pareto status uses all four raw mean objectives. The 61-point sweep moves
from safety to task to effort to smoothness, retaining context-specific
certificates and support caps without retraining. It is exploratory offline
analysis of existing outcomes, not additional independent testing or adaptation
within a trajectory. PNG/PDF outputs and placement captions accompany plot data.
