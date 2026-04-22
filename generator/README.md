
# Skill Generator 

**Purpose:** Generates skill summaries (payoff + motive features) from environment states using a 2-head MLP.  


## Goal
Learn to predict skill outcomes from state inputs to enable certification without full execution every time.


## Key Files
| File | Purpose |
|------|---------|
| `skill_generator.py` | PyTorch definition for 2-head MLP |
| `losses.py` | Composite loss |
| `train_loop.py` | Training logic using TD errors|

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