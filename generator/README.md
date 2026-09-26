# Generator and MDN Training

This directory contains two learning components:

- `SkillGenerator`: a 2-head MLP that predicts rollout payoff and 2D motive returns.
- `MotiveDecompositionNetwork` (MDN): a shared network that predicts motive weights,
  support geometry, admission gates, and auxiliary motive returns.

The shipping configuration targets MO-LunarLander with two objectives
`[Safety, Fuel]`, but the MDN support head and the whole certification chain
(validation, worst-case evaluation, certificate schema, MeTTa storage) are
correct for any objective count `M >= 2`.

## Skill Generator

The `SkillGenerator` is a supervised model that maps an 8-value starting
observation to a predicted payoff and two motive returns (`Safety`, `Fuel`).
It can rank or filter candidate contexts without running each candidate in the
simulator. Its predictions are not certification: final admission still
requires measured execution through CDS/PDS (`certification/`).

| File | Purpose |
|---|---|
| `skill_generator.py` | Two-head model for payoff and motive-return predictions |
| `losses.py` | Weighted payoff and motive MSE loss |
| `dataset_split.py` | Creates, saves, loads, and applies the train/validation/test assignment |
| `train_generator.py` | Trains from one rollout directory; selects the checkpoint using validation loss |
| `evaluate_generator_mse.py` | Compares model and training-mean baseline MSE on the held-out test split |
| `evaluate_generator_report.py` | Summarizes CDS/PDS outcomes and candidate performance on held-out candidate-set contexts |
| `compare_dataset_sizes.py` | Trains/evaluates independent dataset-size experiments and saves their splits and results |

### Training data

The standard training set is single-policy rollout data, normally collected
from the deterministic PPO pilot into `data/raw`. Each top-level `.npz` file is
one record with an observation, payoff, and motive returns. Training loads all
matching files in the selected directory.

`data/raw_mixed` is another supported source: the mixed collector runs its
candidate policies from shared starting contexts and saves one record per
policy outcome. The current `SkillGenerator` does not take a policy identifier
as input, so training on this directory models the pooled outcome distribution;
it does not predict a particular policy's outcome. Use single-policy data when
the prediction target is specifically the deterministic PPO pilot, or mixed
data when the target is the pooled candidate-outcome distribution.

### Training workflows

Collect and train the single-policy PPO rollout model:

```bash
python -m data_collector.collect --episodes 2000 --save-dir data/raw --seed 42
python -m generator.train_generator --data-dir data/raw --output models/generator.pt
```

The trainer uses 75% of records for training, 12.5% for validation, and 12.5%
for testing. It selects the best checkpoint using validation loss and does not
use the test split. It writes `models/generator.pt`, a mid-training checkpoint
beside it, `data/generator_split_manifest.json`,
`plots/generator_training_log.csv`, and `plots/generator_training.png`.

The mixed-policy collection and training workflow is also supported:

```bash
python -m data_collector.collect_mixed_generator_data \
  --episodes 1000 \
  --save-dir data/raw_mixed \
  --seed 42

python -m generator.train_generator \
  --data-dir data/raw_mixed \
  --output models/generator.pt
```

The mixed collector records outcomes for multiple policies from shared
starting contexts. Since `SkillGenerator` has no policy-ID input, a model
trained on `data/raw_mixed` predicts the pooled outcome distribution rather
than one particular policy's outcome. Both workflows use the same trainer
and model output path.

Each training run writes `data/generator_split_manifest.json`. Use the
manifest produced by the matching training run when evaluating that model and
data directory; a subsequent run replaces this default manifest. Pass
`--split-manifest` to retain separate manifests for separate datasets.

## Evaluation

#### Held-out MSE

Evaluate against the mean-outcome baseline on the held-out test split. The
baseline is computed from training records only, and the evaluator accepts
one data directory per invocation:

```bash
python -m generator.evaluate_generator_mse \
  --model-path models/generator.pt \
  --data-dir data/raw \
  --split-manifest data/generator_split_manifest.json
```

For a model trained on mixed data, use `--data-dir data/raw_mixed` and the
manifest produced by that training run. The default report path is
`demo/artifacts/generator_mse_report.json`; `--output-dir` changes it.

#### Certification report

Collect held-out candidate-set contexts with seed bases 1000, 2000, and 3000:

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 1000 --prefix seed1000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 2000 --prefix seed2000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 3000 --prefix seed3000
```

The collector sets each context seed to the base seed plus its 1-based context
index. The batches therefore cover 1001-2000, 2001-3000, and 3001-4000 without
overlapping. Use this directory for both the generator report and the MDN
evaluation below.

```bash
python -m generator.evaluate_generator_report \
  --model-path models/generator.pt \
  --eval-dir data/mdn_candidate_sets_eval \
  --output demo/artifacts/generator_evaluation_report.json
```

The report compares recorded candidate outcomes with an idle-policy baseline
(20 episodes by default) through CDS/PDS. It summarizes per-skill admission
rates, rejection reasons, and payoff/motive improvements. For
`ppo_deterministic`, it also reports predicted-versus-actual payoff correlation
and averages. The JSON is written to the path passed with `--output`.

#### Dataset-size comparison

Run the default comparison with:

```bash
python -m generator.compare_dataset_sizes --data-dir data/raw
```

The input directory needs at least 7,000 top-level `.npz` rollout files to run
all default sizes. The script compares 1,000, 3,000, and 7,000 records, each
with a reproducible 75% / 12.5% / 12.5% train/validation/test split. Their
split counts are 750/125/125, 2,250/375/375, and 5,250/875/875. A size larger
than the available pool is skipped. Use `--sizes` to select sizes and
`--data-dir` to choose the rollout pool.

Selected files are copied to
`data/dataset_size_comparison_rollouts/size_N/{train,val,test}`. By default,
plots and results go to `plots/dataset_size_comparison`:
`combined_training_curves_epochs150.png`,
`test_mse_vs_dataset_size_epochs150.png`, and
`dataset_size_comparison_results_epochs150.json`. Use `--rollout-output-dir`
and `--output-dir` to change those locations. The selected datasets are nested
by size but each size is split independently, so a file can be in the test
split for one size and the training split for another.

### MDN Model Contract

`mdn.py` exposes two paths:

- `forward_inference(context)` returns Dirichlet concentration parameters
  (`alpha`) and support values.
- `forward_auxiliary(context, skill_id)` returns an admission-gate logit and
  predicted motive returns (`q_hat`) for the given skill.

The shipped LunarLander configuration has two objectives, `[Safety, Fuel]`.
The support parameterization and certification geometry support any objective
count `M >= 2`.

### Support geometry: SASP

The support head uses Softmax-Anchored Slack Parameterization (SASP). It emits
`2M` values: `M` base-allocation logits and `M` slack-gate logits. The decoder
computes:

```text
p = softmax(base_logits)
g = slack_floor + (1 - slack_floor) * sigmoid(gate_logits)
s = p + (1 - p) * g
```

The resulting support values satisfy `0 <= s_i <= 1` and `sum(s) >= 1` for
any finite network output. These constraints make
`W_x = {w in simplex : w_i <= s_i}` non-empty by construction, rather than
relying on a loss penalty. The construction is symmetric across objectives.
`slack_floor` defaults to `0.02` and must be in `[0, 1)`; it prevents the
support region from collapsing to the single point `s = p`.

### Checkpoint compatibility

SASP widened the support head from `M` outputs to `2M`. Checkpoints from before
this change are not compatible and must be retrained; their weights are not
reinterpreted. `utils.mdn_checkpoint_loader.load_mdn_checkpoint` raises
`IncompatibleCheckpointError`. `utils.mdn_stub.load_mdn_or_stub` reports the
migration requirement and falls back to `StubMDN`.

### MDN training data and phases

Candidate-set files contain one starting context and the recorded outcomes of
the default seven candidate policies on that same context. Three 1,000-context
collections produce 3,000 contexts and 21,000 candidate outcomes:

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 10000 --prefix seed10000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 11000 --prefix seed11000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 12000 --prefix seed12000
```

Each collection uses context seeds `base seed + 1` through `base seed +
1000`; these training ranges are disjoint from each other and from the held-out
evaluation seeds below.

Train from those files with the policy and auxiliary phases:

```bash
python -m generator.train_mdn_candidate_sets \
  --data-dir data/mdn_candidate_sets \
  --pattern "*.npz" \
  --seed 42 \
  --device cpu \
  --policy-checkpoint models/mdn_policy_best.pth \
  --auxiliary-checkpoint models/mdn_auxiliary_best.pth \
  --q-loss mse
```

The policy phase learns the alpha distribution for candidate selection. The
auxiliary phase learns gate acceptance and motive-return predictions. Q-target
normalization is enabled by default and is saved with the checkpoint. The
best auxiliary validation state is restored before the shared model is saved.
Optional auxiliary settings include `--q-loss huber`,
`--calibrate-auxiliary-q`, `--use-ips`, and `--use-doubly-robust`; IPS and
doubly robust estimation cannot be enabled together.

Candidate-set training does not fit the support head. Support values are
trained separately by `MDNSupportTrainer` against support-function targets
stored in `WeightSetStore`; `utils.mdn_support_pipeline.observe_and_train_support`
records a certified weight and runs a support-training step. The trainer's
`last_feasibility_violation_rate` is a diagnostic and should remain exactly
`0.0` under SASP. Targets may include the exact value `1.0`, while SASP outputs
remain strictly below `1.0` for finite logits, so support MSE can plateau above
zero; this is expected and does not invalidate feasibility.

### Held-out candidate-set evaluation

Collect held-out contexts once, using seed bases 1000, 2000, and 3000. The
collector adds each 1-based context index to its base seed, so the resulting
context-seed ranges (1001-2000, 2001-3000, and 3001-4000) do not overlap.
Use this evaluation directory for both the generator certification report and
the MDN evaluation; do not include it in the MDN training directory.

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 1000 --prefix seed1000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 2000 --prefix seed2000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 3000 --prefix seed3000
```

Evaluate the MDN on those held-out candidate sets:

```bash
python -m generator.evaluate_mdn_candidate_sets \
  --checkpoint models/mdn_policy_best.pth \
  --data-dir data/mdn_candidate_sets_eval \
  --pattern "*.npz" \
  --seed 1000 \
  --device cpu
```

The evaluator reports candidate-selection lift against random certified
candidates and deterministic PPO, regret and balanced top-1 metrics, gate
precision/recall/F1, motive-return MSE/MAE, and bootstrap confidence intervals.

## Tests

Run the focused SkillGenerator evaluation regressions:

```bash
python -m pytest tests/test_generator_evaluations.py -v
```

They cover comparison splits and defaults, held-out MSE and its training-only
baseline, candidate-set loading, and certification-report aggregation.

The support-geometry and MDN test suites cover model behavior, selection,
training, checkpoint compatibility, and held-out evaluation:

```bash
python -m pytest tests/test_generator.py tests/test_generator_training.py -v
python -m pytest tests/test_mdn.py tests/test_mdn_skill_selection.py -v
python -m pytest tests/test_skill_library.py tests/test_mdn_support_trainer.py tests/test_mdn_stub.py -v
python -m pytest tests/test_mdn_support_geometry.py tests/test_mdn_support_pipeline.py -v
python -m pytest tests/test_train_mdn_candidate_sets.py tests/test_evaluate_mdn_candidate_sets.py -v
python -m pytest tests/test_trained_mdn_end_to_end.py tests/test_trained_mdn_zero_shot.py -v
```