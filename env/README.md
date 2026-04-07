# Environment Wrapper 

**Purpose:** Wraps MO-LunarLander (MO-Gymnasium) to standardize vector reward output for SubRep certification.  

## Goal
Provide a stable interface that returns **observation vectors** and **multi-objective reward vectors** (Safety, Fuel) for every step.


## Key Files
- `lunar_lander_wrapper.py`: Wraps `mo-gymnasium` to enforce reward shape.
- `config.py`: Environment constants (max steps, seed, etc.). * Note: Load system env variables from utils/config.py, not here.
## Validation
Run `python tests/test_env.py` to verify:
- Observation shape is `(8,)`.
- Reward shape is `(2,)`.
- Episode terminates correctly on crash/landing.


