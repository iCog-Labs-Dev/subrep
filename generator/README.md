
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

## Validation
Run `python tests/test_generator.py` to verify:
- Output shapes match specification above
- Gradients flow correctly for both heads
- Model saves/loads without error
- Loss decreases over training steps



