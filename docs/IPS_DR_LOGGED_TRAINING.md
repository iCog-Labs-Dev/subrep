# IPS/DR Logged-Data Training

This explains how to collect the probability-aware runtime logs needed for
IPS and Doubly Robust MDN training, and how to train the logged, IPS, and DR
checkpoints.

## Why This Dataset Is Needed

IPS/DR training cannot use regular candidate-set files directly because it needs
the probability that the logging policy used to select the executed candidate.

Required field:

```text
behavior_probability = P(selected candidate | context, candidate set)
```

This is candidate-selection probability, not PPO action probability. Missing or
fake probabilities should not be used.

## Collect Training Logs

Collect the main probability-aware training set:

```bash
python -m data_collector.collect_probability_aware_runtime_logs \
  --decisions 21000 \
  --save-dir data/mdn_probability_aware_logs_train \
  --seed 42 \
  --prefix train \
  --behavior-epsilon 0.2 \
  --behavior-temperature 1.0 \
  --behavior-weights 0.5 0.5
```


## Validate Logs

Validate the collected training files before training:

```bash
python -c "from utils.probability_aware_logs import load_probability_aware_log, probability_aware_log_files; files=probability_aware_log_files('data/mdn_probability_aware_logs_train', pattern='train_*.npz'); [load_probability_aware_log(path) for path in files]; print(f'validated {len(files)} probability-aware logs')"
```

Expected target for the main run:

```text
validated 21000 probability-aware logs
```

## Collect Validation Logs

Use a different seed for validation data:

```bash
python -m data_collector.collect_probability_aware_runtime_logs \
  --decisions 3000 \
  --save-dir data/mdn_probability_aware_logs_val \
  --seed 50042 \
  --prefix val \
  --behavior-epsilon 0.2 \
  --behavior-temperature 1.0 \
  --behavior-weights 0.5 0.5
```

Validate it:

```bash
python -c "from utils.probability_aware_logs import load_probability_aware_log, probability_aware_log_files; files=probability_aware_log_files('data/mdn_probability_aware_logs_val', pattern='val_*.npz'); [load_probability_aware_log(path) for path in files]; print(f'validated {len(files)} probability-aware logs')"
```

## Train Checkpoints

Train the unweighted logged-data baseline:

```bash
python -m generator.train_mdn_probability_aware_logs \
  --data-dir data/mdn_probability_aware_logs_train \
  --pattern "train_*.npz" \
  --seed 42 \
  --device cpu \
  --q-loss mse
```

Train the IPS checkpoint:

```bash
python -m generator.train_mdn_probability_aware_logs \
  --data-dir data/mdn_probability_aware_logs_train \
  --pattern "train_*.npz" \
  --seed 42 \
  --device cpu \
  --q-loss mse \
  --use-ips
```

Train the DR checkpoint using the frozen unweighted auxiliary checkpoint as the
Q baseline:

```bash
python -m generator.train_mdn_probability_aware_logs \
  --data-dir data/mdn_probability_aware_logs_train \
  --pattern "train_*.npz" \
  --seed 42 \
  --device cpu \
  --q-loss mse \
  --use-doubly-robust \
  --dr-baseline-checkpoint models/mdn_auxiliary_unweighted.pth
```

Default checkpoint names are estimator-specific:

```text
models/mdn_policy_unweighted.pth
models/mdn_auxiliary_unweighted.pth
models/mdn_policy_ips.pth
models/mdn_auxiliary_ips.pth
models/mdn_policy_dr.pth
models/mdn_auxiliary_dr.pth
```
