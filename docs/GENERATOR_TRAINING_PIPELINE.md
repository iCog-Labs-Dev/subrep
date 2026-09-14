# SkillGenerator Training/Evaluation Pipeline

This document describes the end-to-end, rerunnable pipeline for training
and evaluating the neural `SkillGenerator` (`generator/skill_generator.py`).

## 1. Collect training data 

```bash
python -m data_collector.collect --episodes 2000 --save-dir data/raw --seed 42
```

## 2. Collect held-out evaluation data

Used for the certification-focused report (Step 5) and for the
"generalization to unseen seeds" metric. Seeds 100-102 are never used in
Step 1, so a context evaluated here was never seen during training.

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 100 --prefix seed100
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 101 --prefix seed101
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 102 --prefix seed102
```

## 3. Train the generator (train/val split, model selection, early stopping)

```bash
python -m generator.train_generator \
  --data-dir data/raw \
  --output models/generator.pt \
  --train-frac 0.75 --val-frac 0.125 --test-frac 0.125 \
  --epochs 100 --patience 10 --seed 42
```

What this does, concretely:
- Splits every collected `.npz` file into train/val/test by filename (see
  `generator/dataset_split.py`), and **saves the assignment** to
  `data/generator_split_manifest.json`.
- Trains only on the train split.
- Tracks validation loss every epoch; saves the checkpoint from whichever
  epoch had the best (lowest) validation loss -- not necessarily the last.
- Stops early if validation loss hasn't improved for `--patience` epochs.
- Writes `plots/generator_training_log.csv` (per-epoch train/val loss) and
  `plots/generator_training.png` (both curves, with the selected best epoch
  marked).
- Never touches the test split.

## 4. Evaluate raw prediction error on the held-out TEST split

```bash
python -m generator.evaluate_generator_mse \
  --model-path models/generator.pt \
  --data-dirs data/raw \
  --split-manifest data/generator_split_manifest.json
```

Loads the manifest Step 3 saved and evaluates MSE **only** on the records
labeled `"test"` in that manifest -- data the model never trained on.

## 5. Evaluate certification-focused usefulness on unseen seeds

```bash
python -m generator.evaluate_generator_report \
  --model-path models/generator.pt \
  --eval-dir data/mdn_candidate_sets_eval \
  --output demo/artifacts/generator_evaluation_report.json
```

 For every held-out context (from Step 2's seeds 100-102), it:
- Runs every real candidate skill's recorded outcome through the
  unmodified `CDSGate`/`PDSGate` (via `ImprovementCalculator` against the  `IdlePolicy` baseline) to get admit/reject + reason.
- Reports, per skill (`ppo_deterministic`, `ppo_stochastic`, `random`,
  `noop`, `left_engine`, `main_engine`, `right_engine`):
  - candidate skill success rate
  - certification admission rate after CDS/PDS checks
  - rejection rate and reason for rejection
  - average payoff improvement over the idle baseline
  - average motive-feature (Safety/Fuel) improvement over the idle baseline
- Compares `ppo_deterministic` (the skill the generator is meant to
  pre-filter contexts for, per `generator/README.md`) against the simple
  non-neural baselines (`random`, fixed-action policies).
- Reports the correlation between the generator's predicted payoff and the
  real recorded `ppo_deterministic` payoff for the same held-out contexts
  -- directly answering "is the model actually learning useful candidate
  skill-generation behavior."

Output is written to `demo/artifacts/generator_evaluation_report.json`.

## Rerunning end to end

```bash
python -m data_collector.collect --episodes 2000 --save-dir data/raw --seed 42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 100 --prefix seed100
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 101 --prefix seed101
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 102 --prefix seed102
python -m generator.train_generator --data-dir data/raw --output models/generator.pt
python -m generator.evaluate_generator_mse --model-path models/generator.pt --data-dirs data/raw
python -m generator.evaluate_generator_report --model-path models/generator.pt --eval-dir data/mdn_candidate_sets_eval
```

## Files important for this pipeline

| File | Role |
|---|---|
| `generator/dataset_split.py` | Computes and persists the train/val/test file assignment |
| `generator/train_generator.py` | Trains on train split, selects best checkpoint via val split |
| `generator/evaluate_generator_mse.py` | Raw MSE on the test split only |
| `generator/evaluate_generator_report.py` | Certification/baseline metrics on held-out seeds |
