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

The skill generator is a supervised rollout-outcome model: given a starting
state, it predicts the payoff and motive returns a policy would achieve from
that state, without running the simulator. It is used as a cheap pre-filter
over candidate starting contexts — final skill admission always requires a
measured real-world execution through CDS/PDS (see `certification/`).

| File | Purpose |
|---|---|
| `skill_generator.py` | 2-head MLP: scalar payoff + motive vector |
| `losses.py` | Weighted MSE loss for payoff and motives |
| `dataset_split.py` | Computes and persists the train/val/test file assignment |
| `train_generator.py` | Trains from `--data-dir` on the train/val split only; saves the split manifest and the best-validation checkpoint |
| `evaluate_generator_mse.py` | Reports payoff/motive MSE on the held-out test split only |
| `evaluate_generator_report.py` | Certification-focused evaluation: success/admission rate, rejection reasons, payoff/motive improvement over baseline, comparison against simple candidate policies, generalization to held-out seeds |

### Training data

The generator trains on `data/raw` — single-policy rollout records (the
trained PPO pilot, deterministic) keyed by starting state. `data/raw_mixed`
(below) provides multi-policy rollouts from shared starting contexts, for
use with a policy-conditioned model.

### Training, evaluation, and reporting

```bash
# 1. Collect training rollouts
python -m data_collector.collect --episodes 2000 --save-dir data/raw --seed 42

# 2. Collect held-out evaluation contexts
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 100 --prefix seed100
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 101 --prefix seed101
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 102 --prefix seed102

# 3. Train
python -m generator.train_generator --data-dir data/raw --output models/generator.pt

# 4. Evaluate on the held-out test split
python -m generator.evaluate_generator_mse --model-path models/generator.pt --data-dirs data/raw

# 5. Certification-focused report on held-out seeds
python -m generator.evaluate_generator_report --model-path models/generator.pt --eval-dir data/mdn_candidate_sets_eval
```

`train_generator.py` splits collected files into train (75%), validation
(12.5%), and test (12.5%) sets, and saves the assignment to
`data/generator_split_manifest.json` for reuse by the evaluation scripts.
Training uses the train split; validation loss drives model selection and
early stopping; checkpoints are written to disk as new best-validation
epochs are found; device (GPU/CPU) is selected automatically.
`evaluate_generator_mse.py` reports MSE on the test split.
`evaluate_generator_report.py` reports certification admission rate,
rejection reasons, payoff/motive improvement over baseline, and comparison
against the non-neural candidate policies, on held-out seeds. Full command
reference: [`docs/GENERATOR_TRAINING_PIPELINE.md`](../docs/GENERATOR_TRAINING_PIPELINE.md).

### Quickstart (mixed candidate training)

**Note:** `data/raw_mixed` contains rollouts from the full candidate pool
(PPO variants, fixed engines, and random policies), sharing starting
contexts across policies. Intended for a policy-conditioned model variant.

To reproduce or update the trained model (`models/generator.pt`):

1. **Collect Mixed Data:**
```bash
   python -m data_collector.collect_mixed_generator_data --episodes 1000 --save-dir data/raw_mixed --seed 42
```
2. **Train Generator:**
```bash
   python -m generator.train_generator --data-dir data/raw_mixed --output models/generator.pt
```
3. **Evaluate Predictions:**
```bash
   python -m generator.evaluate_generator_mse --model-path models/generator.pt --data-dirs data/raw data/raw_mixed
```