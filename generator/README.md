
# Skill Generator 

**Purpose:** Generates skill summaries (payoff + motive features) from environment states using a 2-head MLP.  


## Goal
Learn to predict skill outcomes from state inputs to enable certification without full execution every time.

## Quickstart

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

## MDN Candidate-Set Training and Evaluation

The MDN candidate-set workflow trains the shared MDN policy and auxiliary heads
from same-context candidate outcomes. Generated candidate-set data and MDN
checkpoints are local artifacts by default:

- Training data: `data/mdn_candidate_sets/`
- Held-out evaluation data: `data/mdn_candidate_sets_eval/`
- Policy checkpoint: `models/mdn_policy_best.pth`
- Auxiliary checkpoint: `models/mdn_auxiliary_best.pth`

The checkpoint paths above are the standard locations used by the MDN
candidate-set evaluator and runtime integration. If the checkpoint files are not
committed or uploaded separately, each developer should regenerate them locally
with the commands below.

### 1. Collect Training Candidate Sets

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 42 --prefix seed42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 43 --prefix seed43
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 44 --prefix seed44
```

This produces 3,000 contexts. With the default 7 candidate policies per context,
that is 21,000 candidate outcomes.

### 2. Train the MDN

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

Expected completion output includes:

```text
MDN Candidate-Set Training Complete
policy checkpoint:  models/mdn_policy_best.pth
aux checkpoint:     models/mdn_auxiliary_best.pth
```

### 3. Collect Held-Out Evaluation Candidate Sets

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 100 --prefix seed100

python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 101 --prefix seed101

python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 102 --prefix seed102
```

### 4. Evaluate the Trained MDN

```bash
python -m generator.evaluate_mdn_candidate_sets \
  --checkpoint models/mdn_policy_best.pth \
  --data-dir data/mdn_candidate_sets_eval \
  --pattern "*.npz" \
  --seed 100 \
  --device cpu
```

The evaluator reports lift versus PPO and random certified baselines, balanced
top-1 accuracy, regret, gate precision/recall/F1, bootstrap confidence
intervals, and per-objective Q error.

Reference local result after the 2-objective support-geometry fix:

| Metric | Mean |
| --- | ---: |
| Lift vs always-PPO | +9.54 |
| Lift vs random certified | +49.34 |
| Balanced top-1 accuracy | 0.989 |
| Balanced regret | 5.746 |
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

This check confirms:

- alpha remains positive for `alpha_to_mean_weights()`
- support values stay in `[0, 1]`
- each support vector sums to at least 1, so the 2-objective `W_x` region is non-empty


## Key Files
| File | Purpose |
|------|---------|
| `skill_generator.py` | PyTorch definition for 2-head MLP |
| `losses.py` | Composite loss |
| `train_loop.py` | Training logic using TD errors|
| `mdn.py` | Motive Decomposition Network definition |
| `train_mdn_candidate_sets.py` | Train MDN policy and auxiliary heads from candidate-set data |
| `evaluate_mdn_candidate_sets.py` | Evaluate a trained MDN checkpoint on held-out candidate sets |

## Current Skeleton
- Shared trunk: `Linear(8, 64) -> ReLU -> Linear(64, 64) -> ReLU`
- Payoff head: `Linear(64, 1)`
- Motive head: `Linear(64, 2)`
- Weight initialization: Xavier-uniform for all linear weights, zero bias
- Save/load: `state_dict`-based `save(path)` and `load(path, map_location="cpu")` (`map_location` is optional and defaults to `"cpu"`)

## Input / Output Contract
- Single input observation shape: `(8,)`
- Batched input observation shape: `(N, 8)`
- Single output shapes:
  - `payoff`: `(1,)`
  - `motives`: `(2,)`
- Batched output shapes:
  - `payoff`: `(N, 1)`
  - `motives`: `(N, 2)`

## Validation
Run `python -m pytest tests/test_generator.py -v` to verify:
- Output shapes match specification above
- Gradients flow correctly for both heads
- Model saves/loads without error
- Loss decreases over training steps
