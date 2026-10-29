# Multi-policy SkillGenerator: commands and outputs

## Scope

`PolicyConditionedSkillGenerator(obs, policy_id)` predicts rollout payoff and two motives (Safety and Fuel) for the nine existing candidate policies. The legacy `SkillGenerator(obs)` and existing single-policy/MDN workflows remain available and unchanged. Predictions are used only to select candidates; actual CDS/PDS certification always uses measured payoff and motives. The corrected fuel mapping and existing certification implementations are not modified.

All commands below run **from the repository root**. Each command is one line and works in Windows PowerShell. Install `requirements.txt` and provide the trained pilot checkpoint `models/pilot_ppo.pt` before collecting or running live policies. Use a fresh `data/raw_mixed` directory; do not mix records from older collections with different outcome definitions.

## 1. Collect nine-policy data

```bash
python -m data_collector.collect_mixed_generator_data --episodes 1000 --save-dir data/raw_mixed --seed 42
```

Executes all nine registered policies from each of 200 shared starting contexts and saves one measured rollout per policy. **1000 contexts produce 9000 `.npz` records**, provided collection completes successfully. `--episodes` counts starting contexts, not total policy executions. `--seed` controls reproducible starting contexts; `--save-dir` controls the output folder. `--pilot-checkpoint` can override the default `models/pilot_ppo.pt`. Output: `data/raw_mixed/*.npz`. Complete context/policy coverage is required for the comparison and offline filter.

## 2. Train both comparison models

```bash
python -m generator.multi_policy_pipeline train --data-dir data/raw_mixed --epochs 50
```

Creates/reuses the context-disjoint train/validation/test split and trains **two** models on the same training records: the policy-conditioned model (`obs + policy_id`) and the observation-only baseline (`obs` without policy ID). Validation loss selects the best weights; held-out test outcomes are not used for training. Training progress prints every five epochs, at the first and last epoch, and when early stopping occurs.

Outputs:

- `models/multi_policy_generator.pt` — conditioned model checkpoint (`--model-path`).
- `models/observation_only_baseline.pt` — observation-only checkpoint (`--baseline-path`); this is **not** the idle-policy baseline.
- `data/mixed_split_manifest.json` — reusable context-level train/validation/test assignment (`--split-manifest`).
 The direct entry point is equivalent to `python -m generator.train_generator --multi-policy --data-dir data/raw_mixed --epochs 50`.

## 3. Evaluate prediction accuracy

```bash
python -m generator.multi_policy_pipeline evaluate --data-dir data/raw_mixed
```

Loads both saved checkpoints and the same split manifest. Evaluates only held-out test contexts. Compares **three** methods: `multi_policy`, `observation_only`, and `per_policy_training_mean`. The last predicts the mean measured payoff/Safety/Fuel **separately for each policy**, computed from training records only. Overall and per-policy MAE and MSE are reported for payoff, Safety, and Fuel. Lower is better.

Output: `reports/multi_policy/prediction_metrics.json` (set folder with `--output-dir`). The printed JSON contains overall metrics; open the saved report for the per-policy breakdown. Equivalent wrapper:

```bash
python -m generator.evaluate_generator_mse --multi-policy --data-dir data/raw_mixed
```

## 4. Compare training dataset sizes

```bash
python -m generator.multi_policy_pipeline sizes --data-dir data/raw_mixed --sizes 100 300 500 --epochs 50
```

Trains a fresh policy-conditioned model for each requested number of **complete training contexts**, not individual rollout files. Each context contains nine policy records. The validation/test contexts remain fixed across sizes. Each size is evaluated against its own per-policy training means; this is an exploratory sample-efficiency comparison, not permission to tune using the test set. The requested sizes must not exceed the available training contexts.

Output: `reports/multi_policy/dataset_size_metrics.json` (training size, record count, validation statistics, and held-out metrics). Equivalent wrapper:

```bash
python -m generator.compare_dataset_sizes --multi-policy --data-dir data/raw_mixed --sizes 100 300 500 --epochs 50
```

## 5. Generate the held-out multi-policy evaluation report

```bash
python -m generator.evaluate_generator_report --multi-policy --data-dir data/raw_mixed
```

Runs the integrated multi-policy report path on the held-out mixed dataset. It is distinct from the legacy `evaluate_generator_report` command, which evaluates the original single-policy generator and candidate-set certification workflow.

Output: `reports/multi_policy/generator_evaluation_report.json`.

## 6. Evaluate candidate filtering offline

```bash
python -m generator.evaluate_candidate_filter --data-dir data/raw_mixed --k 3 --baseline-episodes 50
```

Uses complete **held-out** contexts with previously measured outcomes for all nine policies. Compares neural top-K, random top-K, fixed-order top-K (same execution budget), and all candidates (full budget). `--k 3` selects three of nine candidates per context. 

Outputs: `reports/multi_policy/filter_metrics.json` (detailed metrics) and `reports/multi_policy/results_summary.md` (comparison table). The reported execution savings are **counterfactual**: collecting the offline benchmark still executed all nine candidates. Certifiable means passing **both** unchanged CDS and PDS checks on measured outcomes. Missed-certifiable rate counts certifiable candidates excluded by the filter. CDS/PDS admission rates are **not** task-success rates.

## 7. Run the live candidate filter

```bash
python -u -m data_collector.run_filtered_candidates --model-path models/multi_policy_generator.pt --pilot-checkpoint models/pilot_ppo.pt --contexts 20 --top-k 3 --baseline-episodes 50
```

Runs a freshly measured idle baseline, predicts the nine policies at each starting context, executes only the top three candidates, and applies existing CDS/PDS checks to each measured rollout. With ten contexts and top-K of three, this executes **30 selected candidates** instead of 90, plus 50 separate idle-baseline episodes. `--contexts` sets the number of starting contexts; `--top-k` sets candidates executed per context; 

Output: `reports/multi_policy/live_filter.json` (override with `--output`). This is an opt-in experimental runtime.

## Direct pipeline command reference

`generator.multi_policy_pipeline` accepts exactly one positional command: `train`, `evaluate`, or `sizes`. Put the command **before** options:

| Command | Required preparation | Main output |
|---|---|---|
| `train` | Complete `data/raw_mixed` records | Two `.pt` checkpoints and context split manifest |
| `evaluate` | Completed `train` and saved checkpoints | `prediction_metrics.json` |
| `sizes` | Complete records and saved split | `dataset_size_metrics.json` |

## Tests

```bash
python -m pytest -q tests/test_multi_policy_generator.py tests/test_policy_registry.py tests/test_mixed_generator_data.py tests/test_candidate_filter.py tests/test_candidate_filter_evaluation.py tests/test_multi_policy_report.py
```
