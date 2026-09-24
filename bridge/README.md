# `bridge/` — MetaMo ↔ SubRep coupling layer

All coupling logic lives here. MetaMo stays a pure motivational engine and is
consumed read-only; nothing in this package is pushed upstream.

## What this does

Each step, MetaMo emits three quantities that SubRep previously took from
static config:

| Quantity | Where it goes |
|---|---|
| `weights` (on the objective simplex) | `score_skill_entry(entry, weight)` — `library/skill_selector.py:21` |
| `pds_epsilon` | stamped onto `candidate.epsilon` → `PDSGate(epsilon=…)` — see [Two budgets, two routes](#two-budgets-two-routes) |
| `cvar_tail_level` | `certify_candidate_skills(cvar_confidence=…)` → `CVaRGate(confidence=…)` — `certification/cvar_test.py:19` |

The executed outcome feeds back through `stimulus.py` into MetaMo's appraisal
comonad Ψ, closing the loop.

## Two budgets, two routes

The two risk budgets reach the gates by **different mechanisms**. This looks
like accidental inconsistency and is not — do not "simplify" it away.

`RuntimeCertificationPipeline.certify_candidate_skills` accepts
`cvar_confidence` as a call parameter but has **no `epsilon` parameter**
(`utils/mdn_runtime_pipeline.py:205-213`). It resolves the PDS budget per
candidate instead, via `_candidate_epsilon` → `_effective_epsilon`, which reads
`candidate.epsilon` and falls back to the static `config.pds_epsilon` when that
is `None` (`utils/mdn_runtime_pipeline.py:301-309`).

So:

| Budget | Route |
|---|---|
| `cvar_tail_level` | call parameter — passed straight through to the gate |
| `pds_epsilon` | **stamped onto each candidate record** before certification |

`MetaMoController.certify` does the stamping in `_with_epsilon`
(`bridge/controller.py`), using `dataclasses.replace` on the frozen
`CandidateSkillRecord`. The helper is duck-typed: callers may pass lightweight
stand-ins that are not `CandidateSkillRecord`, and those pass through
untouched.

**Why this matters.** Without the stamp, the governor's ε is computed every
step, stored on `StepRecord`, printed by the demo — and silently ignored by the
gate, which keeps using the static config value. The failure is invisible:
every other signal still moves, so nothing looks wrong. `test_bridge_e2e.py`'s
S1 is the regression guard.

## Layout

| File | Imports MetaMo? | Purpose |
|---|---|---|
| `protocol.py` | no | `SkillOutcome`, `GovernorSignal`, `MotivationalGovernor` |
| `budget.py` | no | ε/α formulas — **the sign correction lives here** |
| `weights.py` | no | `w_meta`: 8 goals → 6 objectives |
| `stimulus.py` | no | outcome → appraisal inputs |
| `controller.py` | no | per-step orchestration, torch seeding |
| `_loader.py` | path only | locates the MetaMo checkout |
| `governor.py` | **yes — only here** | `MetaMoGovernor`, plus `FakeGovernor` |

Everything except `governor.py` is testable with no MetaMo present. The import
in `governor.py` is lazy, so even that module imports cleanly without it.

**Why the isolation:** MetaMo's own `usecase/` code imports `metamo.core` /
`metamo.state` (`usecase/agents/metamo_agent.py:10`,
`usecase/simulation/runner.py:27`) while the repository root actually exposes
`core/`, `category/`, `dynamics/` — there is no `metamo/` package. Upstream's
import surface has demonstrably churned, so a future rename should be a
one-file fix.

## Getting MetaMo on the path

Not pip-installable (no `pyproject.toml` / `setup.py`), and its modules use
root-relative absolute imports (`core/state.py:6` does
`from core.config import …`), so its repo root must be importable.

`_loader.py` resolves, in order:

1. `$SUBREP_METAMO_PATH`
2. `<subrep>/external/metamo` — the pinned submodule
3. `<workspace>/MetaMo-Python` — sibling checkout

To pin it as a submodule:

```bash
git submodule add https://github.com/iCog-Labs-Dev/MetaMo-Python external/metamo
git -C external/metamo checkout ceb108eba92ff2f2c7e0ce9bf2d073e78044669b
```

No new dependencies: the modules actually imported —
`{core, category, dynamics, openpsi, magus}` — form a closed subgraph needing
only numpy. `pygame` is imported solely by `usecase/simulation/`, which is
never touched.

---

## ⚠️ The α sign correction

**This is the single most important thing in this package.**

The reference specification
([SubRep-Minecraft-AIRIS_v2](https://drive.google.com/file/d/1Rpvusi_nIEIheElX7kUKQY88Dw0fazgS/view))
gives:

```
ε = ε₀ − a₁·securing + a₃·approach
α = α₀ + b₁·securing + b₂·threshold − b₃·approach
```

SubRep's CVaR gate is a **lower-tail mean at quantile `confidence`**
(`certification/cvar_test.py:54-58`):

```python
var_threshold = np.quantile(values, self.confidence)
tail_values   = values[values <= var_threshold]
return float(np.mean(tail_values))
```

so **α ↑ → shallower tail → CVaR ↑ → easier to admit → *less* conservative.**

Under the paper's formulas as written, rising `securing` therefore **tightens
PDS while loosening CVaR**. The two gates move against each other on the same
modulator, which cannot be intended.

### Resolution

1. **`confidence = α_t` numerically — no remapping.** The conventions already
   agree: the specification states α = 0.1 and ε₀ = 0.10, and
   SubRep defaults to `cvar_confidence = 0.1` / `pds_epsilon = 0.1`
   (`utils/mdn_runtime_pipeline.py:97-99`). A `1 − α_t` remap would send
   0.1 → 0.9 — the mean of the worst 90%, essentially the plain expectation —
   silently disabling the gate.
2. **Flip the b-signs** so α tightens with securing.

```
ε_t = clip(ε₀ − a₁·(securing−0.5) + a₃·(approach−0.5),  0.0,   ε_max)
α_t = clip(α₀ − b₁·(securing−0.5) − b₂·(threshold−0.5)
                + b₃·(approach−0.5),                     α_min, α_max)
```

`test_bridge_budget.py::test_securing_tightens_both_gates` fails loudly if the
published signs are ever reintroduced.

### Why deviations from 0.5

MetaMo squashes modulators through `1/(1+exp(−4(M−0.5)))` every step
(`openpsi/appraisal.py:97`) and initialises them to 0.5
(`core/engine.py:81`), so **M ∈ (0,1) with neutral 0.5**. Using raw values
would mean ε₀/α₀ did not hold at the neutral state.

### Coefficients

`ε₀ = α₀ = 0.1`, `a₁ = 0.4`, `a₃ = 0.2`, `b₁ = 0.4`, `b₂ = 0.2`, `b₃ = 0.2`.

`a₁ = 0.4` is **pinned by the specification's own trace** and reproduces its
reported values exactly at neutral approach:

| Modulator | Formula | Specified |
|---|---|---|
| securing 0.55 | `0.1 − 0.4(0.05) = 0.08` | ε = 0.08 |
| securing 0.45 | `0.1 − 0.4(−0.05) = 0.12` | ε = 0.12 |

The sigmoid compresses toward 0.5, so realistic securing spans roughly
[0.2, 0.85] — usable range ≈ ±0.35, not ±0.5.

### `α_min` is numerical, not stylistic

The CVaR tail holds ≈ `α · n_samples` draws. At n=1000, α=0.01 leaves 10 —
a noisy estimate and a flickering gate. Floor:
`max(0.02, min_tail_samples / n_samples)`, default 50 samples.

---

## The two α's must never touch

|  | `cvar_tail_level` (MetaMo) | `mdn_alpha` (MDN) |
|---|---|---|
| Type | scalar `float` | `np.ndarray`, length m |
| Range | (0, 1] | strictly positive |
| Meaning | CVaR tail mass | Dirichlet concentration |
| Enters via | `CVaRGate(confidence=…)` | `.admit(…, mdn_alpha=…)` |
| Source | modulators | `generator/mdn.py` |

They share the letter α and nothing else. This package never names a variable
`alpha`; `GovernorSignal.validate()` rejects a vector where the scalar belongs.

---

## Determinism

`CVaRGate.get_cvar` draws from `Dirichlet(...).sample()` on the **global torch
RNG, unseeded** (`certification/cvar_test.py:51`) — so certification is not
reproducible by default. `MetaMoController` seeds before each certification
pass and records the seed on every `StepRecord`.

---

## Calibration status — read before trusting the numbers

The reference specification reports three weight vectors:

```
w̄0 (dusk, patrol risk)  = [0.35, 0.15, 0.20, 0.20, 0.05, 0.05]
w̄1 (patrol appears)     = [0.38, 0.17, 0.18, 0.17, 0.05, 0.05]
w̄3 (villagers, trading) = [0.25, 0.25, 0.20, 0.20, 0.05, 0.05]
```

**These cannot be reproduced exactly**, because the paper never states the goal
vector `G` at those moments — only qualitative modulator movement. Any matrix
fitted to hit them numerically would be inventing `G`.

So `DEFAULT_GOAL_AFFINITY` and `DEFAULT_MODULATOR_GAIN` are **semantically
motivated, not fitted**, and the tests assert what the paper actually
determines: ordering, direction of change, and simplex invariants. Treat the
coefficients as a defensible starting point to tune against a real
environment, not as reproductions of published values.

### Appraisal scaling matters

`stimulus.py` squashes through `tanh`. Leaving `payoff_scale` / `motive_scale`
at 1.0 while the environment reports deltas of order 10 saturates risk on the
first step and pins the modulators at their bounds, flattening the coupling
into a constant. Set them from the environment's actual magnitudes —
`demo/run_metamo_pipeline.py` derives them from the candidate spread.

---

## Known limitation: OR semantics + an untrained MDN

The demo runs `gate_type="PDS"`, `use_cvar=True`, `require_cds_or_cvar=True`,
which returns `result or cvar_result` (`utils/mdn_runtime_pipeline.py:402-404`).
There is no AND mode.

With an **untrained** MDN the Dirichlet concentration is arbitrary and the CVaR
gate admits nearly everything, so it can overrule PDS rejections and ε stops
being observable in the admitted count. The demo prints a PDS-only column
alongside for exactly this reason. Train the MDN, or switch to `PDS` without
`use_cvar`, before drawing conclusions about gate behaviour.

Every assertion in `test_bridge_e2e.py` that concerns ε or abstention runs with
`use_cvar=False` for this reason.

---

## The borderline candidate

`env/minecraft_stub.py` defines a `RiskyForage` action that exists purely so ε
has something to act on.

The PDS gate admits when `Δr + min(Δn) ≥ −ε`, and MetaMo drives ε over roughly
`[0, 0.1]`. So ε can only change an admission for a skill whose margin lands
inside `(−0.1, 0)`. The original five actions have margins from −2.0 to +11.4 —
all far outside that band, which means **ε provably could not flip any decision**
and no test could demonstrate the budget coupling at all.

`RiskyForage` is constructed with a **noiseless** margin of ≈ −0.05: ε above
~0.05 admits it, ε below rejects it. Its reward row is derived analytically
rather than tuned by hand — the full derivation is in the
`BORDERLINE CANDIDATE` comment block in `env/minecraft_stub.py`, and the
margin is pinned by
`test_minecraft_stub.py::test_riskyforage_margin_sits_inside_the_epsilon_band`.

### It only works with noise off

The margin is ≈ −0.05, while the stub's default `noise_scale=0.02`, accumulated
over a discounted episode across six objectives, perturbs it by a comparable or
larger amount. With noise on, the margin wanders well outside the band — in the
demo it lands near −0.104, so PDS rejects `RiskyForage` at *every* ε in
`[0, 0.1]`.

So any test that depends on ε flipping this candidate must build the env with
`noise_scale=0.0`; `test_bridge_e2e.py` does. `demo/run_metamo_pipeline.py`
uses the default noise, so its PDS-only column stays flat and the demo does
**not** illustrate the ε coupling — S1 is where that behaviour is demonstrated.

If the episode length, γ, or the idle reward row ever change, that test fails
and the row must be **re-derived from the comment**, not nudged until the test
passes.
