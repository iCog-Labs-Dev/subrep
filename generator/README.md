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

### Support Geometry: SASP (Softmax-Anchored Slack Parameterization)

Support values are decoded so that the admissible region
`W_x = { w in simplex : w_i <= s_i }` is **feasible by construction at every
objective count M**, not just at M = 2:

- `0 <= s_i <= 1` for every objective (boundedness)
- `sum(s) >= 1` (non-emptiness — otherwise no weight vector can respect every
  per-objective cap while summing to 1, and `W_x` describes nothing)

The support head emits `2M` logits, split into two groups:

```python
p = softmax(raw[..., :M])                              # sums to 1, p_i in (0, 1)
g = slack_floor + (1 - slack_floor) * sigmoid(raw[..., M:])  # g_i in (g_min, 1)
s = p + (1 - p) * g
```

Each `s_i` interpolates between its base allocation `p_i` and the ceiling 1.
Boundedness holds because `s_i` is a convex combination of `p_i` and 1;
non-emptiness because `sum(s) = sum(p) + sum((1 - p_i) * g_i) >= sum(p) = 1`.
Both are algebraic, so they hold for any network weights, at every training
step and at inference — no penalty term or loss tuning is involved.

The construction is **permutation-equivariant**: softmax is applied jointly and
symmetrically across objectives and the gates are elementwise, so no objective
is structurally privileged by its index. This is what a sequentially chained
construction cannot offer, and SubRep's objectives have no natural ordering.

`slack_floor` (default `0.02`, must lie in `[0, 1)`) keeps `W_x` from
collapsing to the single point `s = p`. A collapsed region would certify skills
against essentially one weighting — mathematically valid but a fragile
certificate.

> **Replaces the previous behavior.** Earlier versions decoded a feasible
> interval only for `num_objectives == 2` and fell back to a raw Softplus path
> (range `(0, inf)`) for every other M. Softplus enforces positivity but neither
> constraint above, so support values could exceed 1 or sum below 1. The
> downstream effect was silent: the skill library excluded every MDN_WX-certified
> skill at that context and fell back to full-simplex skills behind a log line.

### Checkpoint compatibility (breaking change)

The support head widened from `M` to `2M` outputs, so **pre-SASP checkpoints
cannot be loaded** — their weights are not a subset of the new head's meaning.
Both loaders detect this and raise `IncompatibleCheckpointError` with a
migration message:

- `utils.mdn_stub.load_mdn_or_stub` — logs the message and falls back to
  `StubMDN`; weights are never reinterpreted.
- `utils.mdn_checkpoint_loader.load_mdn_checkpoint` — propagates, since callers
  of this function have no stub contract.

A legacy checkpoint must be retrained (see *Train the MDN* below).

### Where support values are actually trained

`train_mdn_candidate_sets` optimizes a **policy** loss
(`compute_mdn_policy_loss(log_prob, advantage)`) that is a function of the
Dirichlet `alpha` only, so **the support head receives no gradient from it**.
Re-running candidate-set training after a SASP migration produces a
shape-correct checkpoint with a freshly initialized support head — which is
still feasible by construction, but not *fit*.

Support values are trained separately by `MDNSupportTrainer`
(`generator/mdn_support_trainer.py`), which regresses them against
support-function targets from a `WeightSetStore`, driven via
`utils.mdn_support_pipeline.observe_and_train_support`.

Two things to know about that trainer:

- It exposes `last_feasibility_violation_rate`, a **diagnostic only** — never
  added to the loss. It must read exactly `0.0`; a nonzero value indicates a
  code regression, not a hyperparameter to tune.
- Targets are left at their measured values, including the exact `1.0` that the
  full simplex produces. SASP yields `s_i < 1` strictly, so the support-head MSE
  **plateaus slightly above zero** rather than converging to ~0, and slack-gate
  logits grow large. This is expected, not a bug: gradient clipping is already
  applied, and the feasibility guarantee is unaffected.

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

### 5. Validate Support Geometry

After training, the MDN must still produce feasible support values. Under SASP
this is guaranteed algebraically, so this check is a regression tripwire rather
than a quality measurement — it should be impossible to fail:

```bash
python - <<'PY'
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
print("objectives:", support.shape[-1])
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

Note this reads `data/mdn_candidate_sets_eval`, which is **not** committed —
collect it first with the held-out command under *Candidate-Set Data
Collection* above. A legacy checkpoint will raise
`IncompatibleCheckpointError` here rather than producing wrong geometry.

## Tests

```bash
python -m pytest tests/test_generator.py tests/test_generator_training.py -v
python -m pytest tests/test_mdn.py tests/test_mdn_skill_selection.py -v
# SASP guarantees + downstream generalization
python -m pytest tests/test_skill_library.py tests/test_mdn_support_trainer.py tests/test_mdn_stub.py -v
python -m pytest tests/test_train_mdn_candidate_sets.py tests/test_evaluate_mdn_candidate_sets.py -v
python -m pytest tests/test_trained_mdn_end_to_end.py tests/test_trained_mdn_zero_shot.py -v
```
