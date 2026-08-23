# Generator and MDN Training

This directory contains two learning components:

- `SkillGenerator`: a 2-head MLP that predicts rollout payoff and 2D motive returns.
- `MotiveDecompositionNetwork` (MDN): a shared network that predicts motive weights,
  support geometry (any number of objectives), admission gates, and auxiliary motive returns.

The current implementation targets MO-LunarLander with two objectives:
`[Safety, Fuel]`; the MDN architecture itself supports any number >= 2.

## Skill Generator

The skill generator is a supervised rollout-outcome model.

| File | Purpose |
|---|---|
| `skill_generator.py` | 2-head MLP: scalar payoff + motive vector |
| `losses.py` | Weighted MSE loss for payoff and motives |
| `train_generator.py` | Trains from `--data-dir` (default `data/raw`) and writes `models/generator.pt` |

### Quickstart (mixed candidate training)

**Note:** The Generator is trained on a "mixed candidate set" (PPO variants, fixed engines, and random policies) to match the actual candidates encountered by the SubRep admission pipeline. The generator remains purely a **prediction pre-filter**. Final skill admission always requires a measured real-world execution.

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
The generator predicts collected rollout totals. It is not a bootstrapped TD
learner in the current implementation.

## MDN Model Contract

`mdn.py` exposes:

- `forward_inference(context) -> (alpha, support_values)`
- `forward_auxiliary(context, skill_id) -> (gate_logit, q_hat)`

Support values are produced by `constrained_support_activation`, guaranteeing
the following for any `num_objectives >= 2`, by mathematical construction:

- `0 <= s_i <= 1` for every objective `i`
- `sum_i(s_i) >= 1`

Internally: `p = softmax(z / tau)` (a learned temperature `tau`, keeping the
usable range from collapsing at high confidence), `t = t_max ** sigmoid(z_scale)`
(log-space interpolation between 1 and the largest safe scale-up for `p`),
`s = clamp(t * p, 0, 1)`. `p_max` is computed via a log-sum-exp smoothing of
`max(p)` rather than a hard max, so the whole construction is differentiable
everywhere.

`support_head` outputs `num_objectives + 1` values (the extra output is
`z_scale`). This is a breaking change to checkpoint compatibility: checkpoints
saved before this change cannot be loaded under the current architecture and
must be retrained (`load_mdn_checkpoint` raises a clear error explaining this
if attempted).

`W_x` worst-case computation, certificate schema validation, and MeTTa
storage now support any `num_objectives >= 2` as well — none of these are
restricted to 2 objectives anymore.

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
