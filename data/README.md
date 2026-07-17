# Data Artifacts

The repo uses two generated data formats:

- raw rollout records for the `SkillGenerator`,
- same-context candidate-set records for MDN training and evaluation.

Generated datasets are ignored by git and should be regenerated locally unless
attached as an external artifact.

## Raw Rollout Records

Default path:

```text
data/raw/*.npz
```

Collect raw rollout records:

```bash
python -m data_collector.collect
```

Per-file schema:

| Key | Shape | Description |
|---|---:|---|
| `obs` | `(8,)` | Initial MO-LunarLander observation |
| `payoff` | scalar | Discounted scalar payoff |
| `motives` | `(2,)` | Discounted `[Safety, Fuel]` motive returns |
| `skill_id` | scalar/string | Policy or skill identifier |
| `terminated` | scalar/bool | Whether the episode ended naturally |
| `behavior_probability` | scalar/float, optional | Behavior-policy probability when available |

These records train `models/generator.pt` through supervised payoff/motive
regression.

## Candidate-Set Records

Default paths:

```text
data/mdn_candidate_sets/*.npz
data/mdn_candidate_sets_eval/*.npz
```

Candidate-set records are the preferred MDN input because each file contains
multiple candidate outcomes from the same starting context.

Collect training candidate sets:

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 42 --prefix seed42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 43 --prefix seed43
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 44 --prefix seed44
```

Collect held-out candidate sets:

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 100 --prefix seed100
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 101 --prefix seed101
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets_eval --seed 102 --prefix seed102
```

Per-file schema:

| Key | Shape | Description |
|---|---:|---|
| `context` | `(8,)` | Shared initial observation |
| `context_seed` | scalar/int | Reset seed used for the shared context |
| `candidate_skill_ids` | `(K,)` | Candidate policy identifiers |
| `candidate_payoffs` | `(K,)` | Discounted scalar payoff per candidate |
| `candidate_motives` | `(K, 2)` | Discounted `[Safety, Fuel]` returns per candidate |
| `terminated_flags` | `(K,)` | Candidate terminal flags |
| `behavior_probabilities` | `(K,)` | Candidate behavior probabilities or `NaN` |
| `step_counts` | `(K,)` | Executed step count per candidate |
| `stop_reasons` | `(K,)` | Stop reason per candidate rollout |

## Generated Pipeline Outputs

The demo pipeline writes:

```text
data/certificates.metta
data/library.json
demo/artifacts/admission_report.json
demo/artifacts/admission_report.md
```

These files are run outputs. Regenerate them with:

```bash
python -m demo.run_full_pipeline
```

If `models/mdn_policy_best.pth` is present, the admission report records
`mdn_source: trained_checkpoint`. Otherwise it records `mdn_source: stub`.

## Optional Safety-Gymnasium Rollout Records

Default path:

```text
data/safety_gymnasium_rollouts/*.npz
```

Collect SafeRL candidate rollouts:

```bash
python -m data_collector.collect_safety_gymnasium_rollouts \
  --env-id SafetyPointGoal1-v0 \
  --contexts 25 \
  --max-steps 200 \
  --save-dir data/safety_gymnasium_rollouts \
  --seed 42
```

Per-file schema:

| Key | Shape | Description |
|---|---:|---|
| `env_id` | scalar/string | Safety-Gymnasium environment ID |
| `context` | `(obs_dim,)` | Shared initial observation |
| `context_seed` | scalar/int | Reset seed used for the shared context |
| `candidate_skill_ids` | `(K,)` | Candidate policy identifiers |
| `candidate_payoffs` | `(K,)` | Discounted task payoff per candidate |
| `candidate_motives` | `(K, 2)` | Discounted `[Safety, Task]` returns, where `Safety = -cost` |
| `candidate_safety_costs` | `(K,)` | Positive discounted safety cost |
| `candidate_task_returns` | `(K,)` | Discounted task reward |
| `step_counts` | `(K,)` | Executed step count per candidate |
| `stop_reasons` | `(K,)` | Stop reason per candidate rollout |

Certify collected SafeRL rollouts:

```bash
python -m demo.run_safety_gymnasium_pipeline \
  --rollout-dir data/safety_gymnasium_rollouts \
  --pds-epsilon 1.0
```

This uses `zero_action` as the same-context baseline, computes `delta_r` and
`delta_n` for the remaining candidates, runs CDS/PDS, stores only admitted
certificates, and writes:

```text
data/safety_gymnasium_certificates.metta
data/safety_gymnasium_library.json
demo/artifacts/safety_gymnasium_admission_report.json
demo/artifacts/safety_gymnasium_admission_report.md
```

## Tests

```bash
python -m pytest tests/test_data_collector.py tests/test_mdn_data_adapter.py tests/test_safety_gymnasium_pipeline.py -v
```
