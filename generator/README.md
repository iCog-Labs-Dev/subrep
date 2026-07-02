# Generator and MDN Training

This directory contains two learning components:

- `SkillGenerator`: a 2-head MLP that predicts rollout payoff and 2D motive returns.
- `MotiveDecompositionNetwork` (MDN): a shared network that predicts motive weights,
  2D support geometry, admission gates, and auxiliary motive returns.

The current implementation targets MO-LunarLander with two objectives:
`[Safety, Fuel]`.

## Skill Generator

The skill generator is a supervised rollout-outcome model.

| File | Purpose |
|---|---|
| `skill_generator.py` | 2-head MLP: scalar payoff + motive vector |
| `losses.py` | Weighted MSE loss for payoff and motives |
| `train_generator.py` | Trains from `data/raw/*.npz` and writes `models/generator.pt` |

Train or refresh the generator checkpoint:

```bash
python -m data_collector.collect
python -m generator.train_generator
```

The generator predicts collected rollout totals. It is not a bootstrapped TD
learner in the current implementation.

## MDN Model Contract

`mdn.py` exposes:

- `forward_inference(context) -> (alpha, support_values)`
- `forward_auxiliary(context, skill_id) -> (gate_logit, q_hat)`

For `num_objectives == 2`, support values are decoded as feasible interval
geometry by construction:

- `0 <= s0 <= 1`
- `0 <= s1 <= 1`
- `s0 + s1 >= 1`

For non-2D objective counts, the model preserves the older non-negative
Softplus support path. General higher-dimensional `W_x` projection is future
work.

## Candidate-Set Data Collection

Candidate-set files are the preferred MDN training input. Each file stores one
shared context and multiple candidate policy outcomes from that same reset seed.

Recommended training collection:

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 42 --prefix seed42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 43 --prefix seed43
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 44 --prefix seed44
```

This gives 3,000 contexts and 21,000 candidate outcomes with the default seven
candidate policies.

Recommended held-out collection:

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 100 --prefix seed100
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 101 --prefix seed101
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 102 --prefix seed102
```

## Train the MDN

Final recommended configuration:

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

Training phases:

- policy phase: learns alpha/selection behavior from candidate outcomes,
- auxiliary phase: learns gate acceptance and motive-return prediction,
- Q-target normalization: enabled by default and stored in checkpoints,
- best auxiliary checkpoint restore: final policy and auxiliary checkpoints share
  the best validation state.

Optional experimental flags:

- `--q-loss huber`: supported, but did not improve held-out Q error in final validation.
- `--calibrate-auxiliary-q`: supported, but kept disabled because it worsened held-out Q error.
- `--use-ips` / `--use-doubly-robust`: available for future off-policy logged-data settings; not used for the final candidate-set checkpoint.

## Evaluate the MDN

```bash
python -m generator.evaluate_mdn_candidate_sets \
  --checkpoint models/mdn_policy_best.pth \
  --data-dir data/mdn_candidate_sets_eval \
  --pattern "*.npz" \
  --seed 100 \
  --device cpu
```

The evaluator reports:

- lift vs deterministic PPO,
- lift vs random certified candidate,
- balanced top-1 accuracy,
- balanced regret,
- gate precision/recall/F1,
- Q/motive MSE and MAE,
- per-objective Q MSE and MAE,
- bootstrap confidence intervals.

Reference held-out validation after the support-geometry fix:

| Metric | Mean |
|---|---:|
| Lift vs always-PPO | +9.54 |
| Lift vs random certified | +49.34 |
| Balanced top-1 accuracy | 0.989 |
| Gate F1 | 0.900 |
| Q/motive MSE | 601.65 |
| Q/motive MAE | 13.37 |

### 5. Validate 2-Objective Support Geometry

After training, the MDN should still produce valid 2-objective support values:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
import torch
from generator.evaluate_mdn_candidate_sets import load_mdn_checkpoint

model = load_mdn_checkpoint("models/mdn_policy_best.pth", map_location="cpu")
files = sorted(Path("data/mdn_candidate_sets_eval").glob("*.npz"))[:500]
contexts = np.stack([np.load(path)["context"] for path in files], axis=0)

with torch.no_grad():
    alpha, support = model.forward_inference(torch.tensor(contexts, dtype=torch.float32))

print("contexts_checked:", len(files))
print("alpha_min:", float(alpha.min()))
print("support_min:", float(support.min()))
print("support_max:", float(support.max()))
print("support_sum_min:", float(support.sum(dim=-1).min()))

assert torch.all(alpha > 0)
assert torch.all(support >= 0)
assert torch.all(support <= 1)
assert torch.all(support.sum(dim=-1) >= 1.0)
print("MDN support geometry check passed")
PY
```

## Tests

```bash
python -m pytest tests/test_generator.py tests/test_generator_training.py -v
python -m pytest tests/test_mdn.py tests/test_mdn_skill_selection.py -v
python -m pytest tests/test_train_mdn_candidate_sets.py tests/test_evaluate_mdn_candidate_sets.py -v
python -m pytest tests/test_trained_mdn_end_to_end.py tests/test_trained_mdn_zero_shot.py -v
```
