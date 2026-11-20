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
