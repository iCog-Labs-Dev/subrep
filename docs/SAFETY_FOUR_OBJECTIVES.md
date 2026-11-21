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
