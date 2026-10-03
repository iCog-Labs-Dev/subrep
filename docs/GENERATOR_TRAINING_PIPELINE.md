# SkillGenerator Training/Evaluation Pipeline

Documented, rerunnable pipeline for training and evaluating the neural
`SkillGenerator` (`generator/skill_generator.py`). No changes are made to
`certification/*` or `baseline/*` -- this pipeline only feeds real,
already-computed outcomes into those existing modules and reports what
comes back.

The generator trains on `data/raw` (single-policy, `ppo_deterministic`
rollouts) only. `data/raw_mixed` reuses one starting context across
multiple policies and is out of scope for this model -- see
`generator/README.md`.

## 1. Collect training data

```bash
python -m data_collector.collect --episodes 2000 --save-dir data/raw --seed 42
```

## 2. Collect held-out evaluation data (non-overlapping seeds)

Each `--seed` value below is spaced 1000 apart, which is larger than
`--contexts`, so the three runs' `context_seed` ranges cannot overlap:
```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 1000 --prefix seed1000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 2000 --prefix seed2000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 3000 --prefix seed3000
```

## 3. Train

```bash
python -m generator.train_generator --data-dir data/raw --output models/generator.pt
```

Splits collected files into train (75%), validation (12.5%), and test
(12.5%) sets by filename, saves the assignment to
`data/generator_split_manifest.json`, trains on the train split only,
tracks validation loss for model selection and early stopping, and writes
mid-training checkpoints (`models/generator_checkpoint.pt`) as new best
epochs are found. Device (GPU/CPU) is selected automatically.

## 4. Evaluate prediction error on the held-out test split

```bash
python -m generator.evaluate_generator_mse --model-path models/generator.pt --data-dir data/raw
```

Reads back the manifest from Step 3 and evaluates only the records labeled
`"test"`. Reports payoff MSE and each motive feature's MSE (Safety, Fuel)
separately, and compares the trained model against a mean-outcome baseline
(the training set's mean payoff/motives, predicted for every input,
ignoring the state) -- a model that does not beat this baseline has not
demonstrably learned a useful, state-dependent pattern.

## 5. Certification-focused report on held-out seeds

```bash
python -m generator.evaluate_generator_report --model-path models/generator.pt --eval-dir data/mdn_candidate_sets_eval
```

For every held-out context (Step 2), certifies every real candidate
outcome under the unmodified `CDSGate` and `PDSGate`, reporting each
separately: `cds_admission_rate` (unconditionally beneficial, zero
tolerance) and `pds_admission_rate` (usable under permitted trade-off --
the operative admission decision, since PDS with `epsilon >= 0` is a
relaxation of CDS: `cds_admit` always implies `pds_admit`). Rejection
reasons are tracked against PDS only, since a PDS rejection is what
actually excludes a candidate. Also reports the generator's predicted
payoff correlated against the real `ppo_deterministic` outcome (the only
skill the generator's training data covers), plus average predicted vs.
actual payoff and motives, and a comparison of `ppo_deterministic` against
the five non-neural candidate policies.

## 6. Does more training data help?

Requires at least 7,000 episodes already collected into `data/raw` if the default `--sizes` is used .
```bash
python -m data_collector.collect --episodes 7000 --save-dir data/raw --seed 42
python -m generator.compare_dataset_sizes --data-dir data/raw --sizes 1000 3000 7000 --epochs 150 --patience 10
```

Trains one model per size, all sharing the same seed/epoch-ceiling/patience. these models are trained independently on the given dataset according to the ration 0.75 for training, 0.125 for validation and test each. See `generator/README.md` for the full output list.

## Rerunning end to end

```bash
python -m data_collector.collect --episodes 2000 --save-dir data/raw --seed 42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 1000 --prefix seed1000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 2000 --prefix seed2000
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 3000 --prefix seed3000
python -m generator.train_generator --data-dir data/raw --output models/generator.pt
python -m generator.evaluate_generator_mse --model-path models/generator.pt --data-dir data/raw
python -m generator.evaluate_generator_report --model-path models/generator.pt --eval-dir data/mdn_candidate_sets_eval
```

## Files in this pipeline

| File | Role |
|---|---|
| `generator/dataset_split.py` | Computes and persists the train/val/test file assignment |
| `generator/train_generator.py` | Trains on train split, selects best checkpoint via val split |
| `generator/evaluate_generator_mse.py` | Test-split MSE per feature, vs. mean-outcome baseline |
| `generator/evaluate_generator_report.py` | CDS/PDS admission, rejection reasons, baseline comparison, on held-out seeds |
| `generator/compare_dataset_sizes.py` | Test MSE across multiple training-data sizes, fixed val/test |
| `generator/skill_generator.py` | Model definition (unchanged) |
| `certification/*`, `baseline/*` | Reused as-is; not modified |