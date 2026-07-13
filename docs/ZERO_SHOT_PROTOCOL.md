# Zero-Shot Reuse Protocol

Zero-shot reuse is the core SubRep promise: **a certified skill can be safely deployed under new motive weights without retraining.** This protocol defines how that claim is validated mathematically and, secondarily, empirically.

## 2. Two Reuse Modes

| Mode | Constant | When It Applies |
|---|---|---|
| **Full-Simplex** | `FULL_SIMPLEX` | Skill was certified over the entire simplex — safe under *any* valid weight. |
| **MDN/Contextual** | `MDN_WX` | Skill was certified within a context-conditioned weight set W_x learned by the MDN. |

## Full-Simplex Reuse

`FULL_SIMPLEX` certificates are globally reusable.

Runtime requirements:

- provide a valid simplex weight `w`,
- verify `w >= 0` and `sum(w) = 1`,
- admit the skill without MDN support geometry.

For CDS certificates this is safe because certification already proved:

```text
delta_r + min(delta_n) >= 0
```

For PDS certificates the runtime check is:

```text
delta_r + w^T delta_n >= -epsilon
```

## MDN_WX Reuse

`MDN_WX` certificates use context-conditioned support geometry. Runtime reuse
requires:

- current simplex weight from MDN alpha,
- current support directions,
- current support values.

The runtime path is:

```text
MDN.forward_inference(context)
  -> alpha, support_values
alpha_to_mean_weights(alpha)
  -> current_weight
SkillLibrary.query_admissible(current_weight, support_directions, support_values)
```

All final admissibility decisions still flow through `SkillLibrary`. The MDN
only supplies weights and support geometry.

## 2-Objective Support Geometry

The current implementation is 2-objective MO-LunarLander. Support
directions are the standard basis:

```text
u0 = [1, 0]
u1 = [0, 1]
```

Support values are thresholds, not weights. A valid example is:

```text
support_values = [0.8, 0.4]
```

This means:

```text
w0 <= 0.8
w1 <= 0.4
w0 + w1 = 1
```

So the feasible interval is:

```text
w0 in [0.6, 0.8]
w1 = 1 - w0
```

The worst-case motive cost is:

```text
h_Wx(-delta_n) = max_{w in W_x} w dot (-delta_n)
```

For CDS:

```text
delta_r >= h_Wx(-delta_n)
```

For PDS:

```text
delta_r >= h_Wx(-delta_n) - epsilon
```

## Safety Behavior

Runtime selection is conservative:

- `FULL_SIMPLEX` skills remain available without support geometry.
- `MDN_WX` skills require current support geometry.
- Missing support geometry blocks `MDN_WX` reuse.
- Infeasible support values exclude `MDN_WX` skills for that step.
- Reuse checks are centralized in `SkillLibrary.query_admissible()`.

## Mid-Episode Motive Shifts

The zero-shot claim also covers the case where motive priorities change during
an episode. The runtime behavior is to re-query the frozen certified library at
the shift step:

```text
weights_before_shift -> query_admissible(...)
weights_after_shift  -> query_admissible(...)
```

No skill is re-certified or retrained during the episode. A globally certified
`FULL_SIMPLEX` skill can remain reusable after the shift, while a contextual
`MDN_WX` skill can be rejected if the current support geometry no longer makes
it admissible.

`utils.mid_episode_reuse_demo.demonstrate_mid_episode_motive_shift()` provides a
deterministic trace of this behavior for tests and reporting.

## Trained MDN Checkpoints

`MDNRuntimeSelector.from_checkpoint()` can load a trained checkpoint from:

```text
models/mdn_policy_best.pth
```

The loader infers architecture dimensions from the checkpoint state dict, so
candidate-set models with large skill embedding tables can be loaded without
hardcoded constructor arguments.

If the checkpoint is missing in the demo pipeline, `StubMDN` is used only as a
test/smoke fallback.

## Future Work

For 3+ objectives, `W_x` becomes a higher-dimensional polytope. The current 2D
interval arithmetic should be replaced with a proper support-function or linear
programming solver before claiming general higher-dimensional contextual reuse.

## Tests

```bash
python -m pytest tests/test_zero_shot_reuse.py tests/test_trained_mdn_zero_shot.py -v
python -m pytest tests/test_mdn_skill_selection.py tests/test_mdn_runtime_selector.py -v
python -m pytest tests/test_mid_episode_reuse_demo.py -v
```
