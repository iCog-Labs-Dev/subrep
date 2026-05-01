# SubRep Rollout Dataset

## File Structure
data/
  raw/
    random_ep001.npz
    random_ep002.npz
    ...

## Schema (per .npz file)
| Key         | Shape | Type    | Description                          |
|-------------|-------|---------|--------------------------------------|
| obs         | (8,)  | float32 | Initial state observation            |
| payoff      | ()    | float32 | Discounted cumulative payoff          |
| motives     | (2,)  | float32 | [Safety_delta, Fuel_delta]           |
| skill_id    | ()    | str     | Policy identifier                    |
| terminated  | ()    | bool    | True if episode ended naturally      |

## Usage Example
```python
import numpy as np
data = np.load('data/raw/random_ep001.npz', allow_pickle=True)
obs      = data['obs']       # shape (8,)
payoff   = float(data['payoff'])
motives  = data['motives']   # shape (2,)
skill_id = str(data['skill_id'])
terminated = bool(data['terminated'])
```

## Notes
- ALL episodes are collected (certified and uncertified) for unbiased training
- Use seed parameter in DataCollector for reproducibility

---

## Testing the Pipeline
You can verify the data collection logic by running the automated test suite:

```bash
python -m pytest tests/test_data_collector.py -v
```

### What the tests verify:
- **`test_collector_runs_without_error`**: Confirms the environment, executor, and collector work together without crashing.
- **`test_saved_files_have_correct_keys`**: Ensures saved `.npz` files contain all required keys (`obs`, `payoff`, `motives`, etc.) with correct dimensions.
- **`test_summary_statistics_are_correct`**: Verifies that the summary printed to the console accurately reflects the collected data.
- **`test_seed_produces_consistent_results`**: Guarantees bit-perfect reproducibility across `random`, `numpy`, and `torch` libraries.
- **`test_custom_prefix_prevents_overwriting`**: Confirms that different skill prefixes create unique filenames, preventing accidental data loss when switching data sources.
