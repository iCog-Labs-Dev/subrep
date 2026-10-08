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

### Cached certificates follow the current budgets

The pipeline caches certificates per `(context, skill)`. Once budgets vary per
step, a cached approval can go stale: a skill certified at ε = 0.10 would stay
admitted, and be selected, after ε fell to 0.00, although a fresh check
rejects it.

So on a cache hit the pipeline compares the stored certificate's budgets (its
`epsilon`, and its `cvar_confidence` when a CVaR gate is in use) with the
current ones. If they match, the stored verdict is reused, exactly as before.
If they differ, the stored delta is **rechecked under the current budgets**, and
the returned record carries the current verdict, margin and ε. The stored
certificate is never modified; it stays as the historical record
(`get_certification_result`).

The recheck uses the W_x region the certificate was issued under, not the
current one. Certifying a skill records the selection weight into W_x, so the
current region was partly shaped by that very certification. Only the risk
budget should differ from the original check.

Regression guards: `test_bridge_e2e.py::test_s1c_cached_approval_is_not_selected_after_epsilon_tightens`
(through the controller, down to selection) and the budget-recheck tests in
`test_mdn_runtime_pipeline.py`.

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
`env/minecraft_rollout.py::appraisal_scales` derives them from the candidate
spread.

Too **small** a scale saturates just the same. The task reward is sparse (only
trading earns it), so the mean |Δr| is ~0.002 and one trade would read as five
units of payoff. `appraisal_scales` therefore floors both scales at
`APPRAISAL_SCALE_FLOOR`, the specified trade value (0.01): one specified trade
reads as one unit. The governor takes its scales once, at construction, so they
come from the start-state candidates.

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

## How Δr and Δn are estimated

All estimation on the MetaMo path lives in `env/minecraft_rollout.py`, used by
the demo, the end-to-end tests and (later) the ablation harness. It follows the
reference specification's definitions for an option of duration τ from state x:

```
r̂(x, o) = Σ_{t<τ} γ^t · r(x_t, a_t)     task reward: info["task_reward"]
n̂(x, o) = Σ_{t<τ} γ^t · φ(x_t)          state features: env.phi()
Δ(x, o) = (r̂, n̂)(x, o) − (r̂, n̂)(x, idle)
```

| Rule | Why |
|---|---|
| Task reward separate from the objectives | The specification's r and φ are different functions. `Δr = ΣΔn` made the score `Σ(1 + wᵢ)Δnᵢ`, where w barely moves the result |
| φ summed as **levels**, including x₀ | "How safe the agent was", not "how much safety changed" |
| One horizon, `DEFAULT_HORIZON = 3`, for evaluation, execution and feedback | Measuring an option over 24 steps and running it for 1 made the estimates meaningless |
| Re-evaluated at every decision, from the current state | Threat changes during an episode; a Δ frozen at step 0 cannot follow it |
| Rollouts on `copy.deepcopy(env)` | Exact for the numpy stub; never touches the live episode. `evaluate_option` is the one place to change for a real environment |

It does **not** use `baseline/idle_policy.py`. That code sums the reward vector
as the task reward and accumulates per-step changes — correct for the
2-objective LunarLander environment it was written for, where the objectives
*are* reward components, and left untouched. The replaced lines are kept as
`# OLD:` comments in `run_option` with that explanation.

### Stub calibration

The PDS gate admits when `Δr + min(Δn) ≥ −ε`, and MetaMo drives ε over
`[0, 0.1]`, so ε can only decide options whose margin lies in `(−0.1, 0)`.
Under the rules above the stub's original reward tables put every margin
between −0.3 and −1.0: PDS rejected everything at every ε.

`_REWARD_SCALE = 0.08` in `env/minecraft_stub.py` (one uniform factor on the
whole objective vector, noise included) was **measured** across a full threat
cycle to put every option inside the band, spread from ≈ −0.015 to ≈ −0.083.
Each ε therefore admits a different subset; IronGolemSpawn's worst change
(−0.083) matches the specification's O5 (−0.08); and DiscountChain's margin
falls as threat peaks, so a tightening ε stops the agent trading under attack.
`test_minecraft_stub.py` pins all of this. If a test there fails, re-measure —
don't nudge.

This replaces the earlier hand-built `RiskyForage` action, which existed only
because nothing else could reach the band.

### What the demo shows — and what it can't yet

With the estimation fixed, the demo's selection changes over a run. It also
shows why the next step is needed: under threat, Securing still rails to 1.0
within two decisions, pinning ε at 0 — and with every margin negative, PDS then
admits nothing, so the agent abstains until threat eases. Bounding MetaMo's
step size is separate, planned work. With ε held at its 0.1 baseline instead,
selection tracks threat: trade when calm, defend as threat peaks, trade again
as it falls.
