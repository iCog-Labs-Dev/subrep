# Utilities (`utils/`)

**Purpose:** Shared helper functions for TD error computation, logging, and configuration.  

## Key Files
| File | Purpose 
|------|---------|
| `td_utils.py` | Computes Temporal Difference errors for payoff & motive heads|
| `logger.py` | Standardizes logging format (console + file) | 
| `config.py` | Loads `.env` variables (seed, paths, hyperparams) |
| `rewards.py` | Helper functions for MO-LunarLander reward processing | 


## Validation
Run `python tests/test_utils.py` to verify:
- TD error computation matches manual calculation
- Logger writes to both console and file
- Config loads all variables from `.env`
- Reward helpers correctly parse MO-LunarLander vectors

