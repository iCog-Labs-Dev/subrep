# Three-objective benchmark workflows

Run commands from the repository root. Results and trained checkpoints are local
artifacts and are not committed. Existing local outputs remain available after
the repository cleanup. Keep the original checkpoints, scales, run manifests and
software environment to reproduce a historical run; retraining does not guarantee
identical numbers. Use fresh output directories for new experiments.

## Portable support comparison

Activate your simulator environment, then supply your own inputs:

```bash
export DATA_DIR=/path/to/training/candidate_sets
export SCALES_FILE=/path/to/scales.json
export BASE_CHECKPOINT=/path/to/auxiliary.pth
export PPO_CHECKPOINT=/path/to/ppo.pt
export LAGRANGIAN_CHECKPOINT=/path/to/lagrangian.pt
bash demo/run_safety_margin_comparison.sh
```

If training and simulation use different environments, set `TRAINER_PYTHON`
and `SAFETY_PYTHON` to their Python executables. The script derives the task
scale from the supplied frozen scales instead of using a previous run’s value.
The examples below include historical filenames; replace these with your own
inputs. Certificate acceptance is distinct from task success or a safety guarantee.

### Exact-context certification and reuse pilot

Run from this directory in the `subrep-safety` environment:

```bash
conda run --no-capture-output -n subrep-safety python -m demo.run_contextual_safety_benchmark \
  --output data/contextual_safety_smoke_v1 \
  --training-seeds 42 --development-contexts 10 --contexts 4 --max-steps 200
```

Use a new output directory for every run. This runs existing policies; it does
not train them. `report.md` lists individual admission paths and re-execution
outcomes. `admissions.json` includes margins and unavailable MDN checks.
Each seed directory contains raw development/evidence records with initial
observations; each context directory contains its library, pre-execution
selection, and subsequent execution outcomes.

Certificates apply only to the same environment configuration, reset seed and
exact initial observation. Full-simplex certification does not waive this
context check. This pilot does not establish transfer to unseen situations.
Policies are re-executed from the same start; deterministic policies can produce
identical outcomes. Random policies use a separate repetition seed. Task payoff
still also appears in the motive vector, matching the previous benchmark.

The ordered admission path is full-simplex CDS, full-simplex PDS, MDN-region CDS,
then MDN-region PDS, stopping at the first pass. Without a checkpoint, MDN checks
are explicitly unavailable. To enable them, append `--mdn-checkpoint PATH
--mdn-contract CONTRACT.json`. The model must use three objectives and raw native
flat observations, with a compatible SASP head. The training-associated contract
must contain:

```json
{"environment":"SafetyPointGoal1-v0","observation_schema":"native_flat_v1","objectives":["safety","task","control_efficiency"]}
```

This contract must describe the actual training inputs; it is not a conversion
for an incompatible model. Learned regions are stored with certificates and
queries outside those regions cannot reuse those entries. The contextual runner
applies these checks explicitly; the older batch-average runner is unchanged.

### Three-objective Safety-Gymnasium data with the existing MDN trainer

The MDN architecture and its context-only inference / context-plus-skill-ID
auxiliary paths are unchanged. The candidate-set record builder infers M from
the outcomes. Use `--baseline-candidate-id zero_action` to use the measured
baseline from each starting context, rather than running a LunarLander baseline.
The baseline remains in the candidate set, as an explicit zero-improvement option.

```bash
python -m data_collector.collect_safety_gymnasium_rollouts \
  --objectives 3 --contexts 300 --seed 120000 --max-steps 200 \
  --save-dir data/safety_mdn3_train_v1 --prefix train \
  --ppo-checkpoint models/safety_ppo_fixed_seed42_updates50.pt \
  --ppo-lagrangian-checkpoint models/safety_ppo_lagrangian_fixed_seed42_updates50.pt

python -m generator.train_mdn_candidate_sets \
  --data-dir data/safety_mdn3_train_v1 --baseline-candidate-id zero_action \
  --device cpu --seed 42 \
  --policy-checkpoint models/safety_mdn3_policy_v1.pth \
  --auxiliary-checkpoint models/safety_mdn3_auxiliary_v1.pth
```

Use fresh directories/checkpoint paths and keep separate context seeds reserved
for final evaluation. These commands retain the existing training objectives,
normalization, skill-ID mapping and policy/auxiliary phases. They do not introduce
manual priority labels. The existing candidate-set trainer does not explicitly
supervise support-region targets; SASP feasibility alone does not validate a
learned region. Review support learning and held-out region coverage separately
before claiming MDN-based certification quality. Its internal validation is not
a substitute for a separate untouched benchmark test set.

### Export contextual selections for support fitting

#### Paper alignment and support-training provenance

Section 3.3 of `SubRep-Minecraft-AIRIS_v2.pdf` describes context-conditioned
support outputs, learning from logged proposals, gate outcomes and returns, and
the update `W_next = conv(W_current union W_new)`. It does not specify maximizing
the area of a region for any passing candidate. The experimental
`generator.build_safety_support_targets` grid teacher has been removed for that
reason. Its generated targets, support checkpoints and benchmark results are
historical heuristic experiments, not a reproduction of that learning rule.
Do not use those targets for new paper-aligned support experiments. Changing
epsilon does not change this distinction.

The retained empirical workflow uses recorded decision weights, through
`generator.train_mdn_support --runtime-log-dir`, or the audited selection export
below. `WeightSetStore` accumulates observed vertices and computes their support
values; support of their convex hull equals support of these vertices. The
support trainer regresses these empirical values while retaining the existing
MDN architecture. This regression is an implementation choice, not an explicit
loss prescribed by the paper. The distribution producing new weights must be
recorded and justified; do not invent weights to increase certification counts.

The present standard-basis support head represents a capped simplex, which can
outer-approximate the empirical convex hull. Always recheck certificates on the
actual predicted region. Neither accumulating weights nor fitting their supports
guarantees that previous certificates remain valid as a region expands.
The paper's displayed gate/Q loss does not specify how support outputs receive
training gradients or how `W_new` is generated. Exact replication of that part
requires resolving those details rather than silently adding a new objective.

The existing MDN architecture is unchanged. Export a completed **non-MDN**
contextual run to the existing support trainer's WeightSetStore format:

```bash
python -m generator.export_contextual_support \
  --benchmark data/contextual_safety_smoke_v2 \
  --output data/contextual_support_export_v1
```

The exporter rechecks selected full-simplex certificates against the saved
same-context candidate/idle measurements and scales. It excludes fallback
selections, preserves evidence files, and writes source hashes and selection
provenance to `manifest.json`. It does not run environments or train anything.
Targets remain coordinate maxima of observed certified-selection priorities,
not proven safe-region boundaries. Identical priority sets yield constant
targets: fitting these does not demonstrate context-dependent region learning.

At least three admitted contexts are required for the trainer's disjoint split;
the four-context smoke run only has two and cannot support that split. Collect
a separate support-development run (reserve other seeds for final benchmarking):

```bash
python -m demo.run_contextual_safety_benchmark \
  --output data/contextual_support_development_v1 \
  --training-seeds 42 --seed 130000 \
  --development-contexts 30 --contexts 100 --max-steps 200
python -m generator.export_contextual_support \
  --benchmark data/contextual_support_development_v1 \
  --output data/contextual_support_export_v2
```

Check `ready_for_split` and `constant_targets` in the export summary. With a
previously trained **three-objective Safety MDN** using the same native
observation schema (see training commands above), run:

```bash
python -m generator.train_mdn_support \
  --base-checkpoint models/safety_mdn3_auxiliary_v1.pth \
  --weight-store data/contextual_support_export_v2/weights.json \
  --output-checkpoint models/safety_mdn3_support_v1.pth \
  --output-dir data/safety_mdn3_support_evaluation_v1 \
  --device cpu --seed 42
```

This command fits and evaluates support targets only. It does not establish
certification correctness or performance improvement in fresh executions.
Do not use a two-objective or LunarLander checkpoint. Base-model training data
should also exclude the support validation/test and final benchmark contexts.

Check the bridge without running benchmark experiments:

```bash
python -m pytest -q tests/test_export_contextual_support.py \
  tests/test_mdn_support_trainer.py tests/test_train_mdn_support.py
```

### Monitored policy retraining

`pilot.train_safety_gymnasium_ppo` accepts `--eval-every` (default 25) and
`--eval-seed` (default 180000). These seeds are development data used for
checkpoint selection, not a final evaluation set. Use a fresh `--output` path.
The output checkpoint is the best development checkpoint; its sibling
`<checkpoint_stem>_run/` contains `metrics.jsonl`, `summary.json`, and periodic
`update_*.pt` snapshots. Snapshots are policy checkpoints, not optimizer-resume
checkpoints. PPO is selected by goal-reaching rate, then return. Lagrangian
selection prefers checkpoints satisfying mean episode cost <= `--cost-limit`,
then goal rate and return. If none satisfies the budget, lower cost is preferred
and `summary.json` reports `budget_met: false`. Budget satisfaction on development
episodes does not guarantee safety on new contexts.

Metrics include deterministic development goal success, raw episode return and
cost, multiplier, sampled-action clipping, and mean-action saturation. No training
architecture or action-distribution change is made. The Lagrangian multiplier
updates only when completed episode cost evidence is available.

```bash
python -m pytest -q tests/test_safety_ppo_training.py
python -m pilot.train_safety_gymnasium_ppo \
  --output models/safety_ppo_monitored_seed42_v1.pt \
  --seed 42 --total-updates 500 --cost-penalty 0 \
  --max-episode-steps 200 --eval-every 25 --eval-episodes 20 \
  --eval-seed 180000 --device cpu
python -m pilot.train_safety_gymnasium_ppo \
  --output models/safety_ppo_lagrangian_monitored_seed42_v1.pt \
  --seed 42 --total-updates 500 --use-lagrangian --cost-limit 1 \
  --lambda-lr 0.005 --max-episode-steps 200 \
  --eval-every 25 --eval-episodes 20 --eval-seed 180000 --device cpu
```

The smaller multiplier step is a development experiment, not a demonstrated fix.
The 500-update budget is 512,000 environment transitions, excluding evaluation.
Keep old policies/data. Review these runs before replacing policies or recollecting
MDN data, and reserve new contexts for final evaluation.

### Lagrangian PPO trainer v2

New `--use-lagrangian` runs use independent actor, reward-critic, and
cost-critic networks. Reward and cost GAE are computed separately, then
`A_reward - lambda * A_cost` is combined and normalized for the actor.
Critic gradients do not update the actor backbone or share its gradient
clipping budget. Episode costs and the cost limit remain in raw units.
Plain PPO and the MDN architecture are unchanged. Older policy checkpoints
remain loadable; new policy checkpoints record the separate critic layout.
The cost critic is training-only; saved checkpoints are for inference, not
optimizer/training resumption.

Use a new output filename for retraining. Inspect the run directory's
`metrics.jsonl` (including policy loss, reward/cost value losses and approximate
KL) and `summary.json`. Passing tests verifies the implementation paths, not
that a run will learn a successful policy within the safety budget.

Lagrangian trainer v3 (`finite_episode_cost_v3`) aligns cost value targets with
raw episode cost: cost gamma is 1, cost continuation is zero at termination,
truncation or the configured episode cutoff, and the cost critic receives the
remaining fraction of the episode. Unfinished rollout buffers still bootstrap
cost values. Reward discounting/bootstrapping, the actor observation interface,
and MDN are unchanged. This fixes the cost-target definition; it does not
promise convergence or budget compliance. Retrain under a new checkpoint name.
