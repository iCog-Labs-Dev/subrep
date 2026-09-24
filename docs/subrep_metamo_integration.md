# SubRep ↔ MetaMo Integration

**Purpose:** Let MetaMo's motivational state drive SubRep's skill admission and
selection at runtime, without modifying MetaMo.

## Goal

SubRep certifies skills through CDS/PDS/CVaR gates and selects among them using
a weight vector on the objective simplex. Three quantities govern that process:

| Quantity | Controls |
|---|---|
| `w` — selection weights | which admitted skill is chosen |
| `ε` — PDS budget | how much motive damage a skill may trade for payoff |
| `α` — CVaR tail level | how deep into the bad-outcome tail the risk test looks |

Before this integration all three were **static configuration**. Now MetaMo
emits them fresh every step from its motivational state `X = (G, M)`, and the
executed outcome feeds back into MetaMo's appraisal layer. The agent's risk
posture tightens under threat and relaxes when conditions are calm.

Relevant code:

- `bridge/` — all coupling logic
- `external/metamo/` — MetaMo, a pinned git submodule (never edited)
- `env/minecraft_stub.py` — 6-objective stand-in environment
- `demo/run_metamo_pipeline.py` — runnable end-to-end demo
- `tests/test_bridge_*.py`, `tests/test_minecraft_stub.py`

---

## 1. What the MetaMo analysis found

MetaMo was read before any code was written, to determine whether it could be
consumed as-is. Paths below are relative to `external/metamo/`.

**`MetaMoPseudoBimonad.step()` is pure** — `category/bimonad.py:153-171`. It
takes a state and returns a new one; it never mutates `self` or its argument.
This is the single most important finding: state can be held and threaded
entirely from outside.

**Appraisal and decision are abstract base classes, injected via the
constructor** — `category/functors.py:7,32`, injected at
`category/bimonad.py:36`. Behaviour can be extended by supplying different
implementations, without editing MetaMo.

**Modulators are bounded and centered** — `openpsi/appraisal.py:97` squashes
every modulator through `1/(1+exp(−4(M−0.5)))` each step, and
`core/engine.py:81` initialises them to `0.5`. So `M ∈ (0,1)` with a neutral
point at `0.5`. This is why the budget formulas use *deviations from 0.5*
rather than raw values (§7).

**Goal and modulator orderings are fixed constants** — `core/config.py:1-7`:

```
Goals (8):      Individuation, Transcendence, Help, Curiosity,
                Novelty, Self, Ethical, Social
Modulators (6): Valence, Arousal, Approach, Resolution, Threshold, Securing
```

**The import subgraph we need is closed.** The modules actually imported —
`{core, category, dynamics, openpsi, magus}` — depend only on each other and
numpy. `pygame` is imported solely by `usecase/simulation/`, and `llm/` is
reached only through `core/engine.py`. Neither is touched, so MetaMo adds **no
new dependencies**.

**`TranslationFunctor` cannot be reused for the projection.** It looks like a
candidate for mapping goals into another space, but it rejects non-square
matrices (`category/functors.py:76-79`) because it exists for same-space peer
simulation, and it returns a `MotivationalState` whose `__post_init__` pins `G`
to 8 entries. An 8→6 projection would require changing MetaMo. See §5.

**Upstream's import surface has churned.** `usecase/agents/metamo_agent.py:10`
and `usecase/simulation/runner.py:27` import from a `metamo` package that does
not exist at the repository root — the root exposes `core/`, `category/`,
`dynamics/` directly. This is why all MetaMo imports are confined to one file
(§4).

---

## 2. Why MetaMo needs zero changes

Each finding above removes a reason we might otherwise have had to patch it:

| If MetaMo had… | We would need to… | But it doesn't, because |
|---|---|---|
| a mutating `step()` | fork it to thread state safely | `step()` is pure |
| hardcoded appraisal/decision | fork it to change behaviour | both are constructor-injected ABCs |
| unbounded modulators | fork it to clamp them | they are sigmoid-squashed into (0,1) |
| heavyweight dependencies | vendor or strip them | the needed subgraph is numpy-only |

The remaining coupling — projecting goals onto objectives, computing risk
budgets, translating outcomes back into stimuli — is **SubRep's concern, not
MetaMo's**. MetaMo is a motivational engine; how a downstream planner consumes
its state is not its business. So all of it lives in `bridge/`.

**Enforcement:** `external/metamo` is a pinned submodule.
`git -C external/metamo status --porcelain` must always be empty. If MetaMo
genuinely needs a change, it goes through a separate upstream PR — additive,
behaviour-preserving, and small enough to land trivially.

---

## 3. Why a git submodule

MetaMo has no `pyproject.toml` and no `setup.py`, so `pip install` is not an
option. That leaves vendoring (copying the files in) or a submodule.

Vendoring was rejected: the copy silently diverges from upstream over time,
with no record of which version is actually running, and it carries the
original project's licence and attribution obligations into this repository.

A submodule instead:

- pins **one exact commit**, so behaviour is reproducible and upstream changes
  cannot reach us until the pin is deliberately moved
- keeps MetaMo's own history and licence intact
- makes "which MetaMo are we running?" a one-line answer

```bash
git -C external/metamo rev-parse HEAD
```

### Setting it up

```bash
git submodule add https://github.com/iCog-Labs-Dev/MetaMo-Python external/metamo
git -C external/metamo checkout ceb108eba92ff2f2c7e0ce9bf2d073e78044669b
```

On a fresh clone, use `--recurse-submodules`, or run `git submodule update
--init` afterwards. CI fetches it via `submodules: recursive` in
`.github/workflows/ci.yml`; without that the MetaMo-dependent tests would
silently **skip** rather than fail, and the seam would go untested.

### Getting it on `sys.path`

MetaMo's modules use root-relative absolute imports — `core/state.py:6` does
`from core.config import …` — so its repository root must be importable.
`bridge/_loader.py` is the only place in SubRep that touches `sys.path` for
this, and it resolves in order:

1. `$SUBREP_METAMO_PATH` — explicit override
2. `<subrep>/external/metamo` — the pinned submodule
3. `<workspace>/MetaMo-Python` — a sibling checkout, for local development

The third entry is a convenience but also a trap: if the submodule fails to
populate, the loader silently falls back to a sibling copy and everything
passes locally while CI skips. Always confirm which one is live:

```bash
python -c "from bridge._loader import find_metamo_root; print(find_metamo_root())"
```

It must print a path ending in `external/metamo`.

---

## 4. The SubRep side: `bridge/`

All coupling logic is in one package. **Only `bridge/governor.py` imports
MetaMo, and it does so lazily** — so every other module is plain
Python/numpy and fully testable with no MetaMo checkout present, and even
`governor.py` imports cleanly without one.

That isolation is deliberate: given upstream's import surface has already
shifted once (§1), a future rename should be a one-file fix rather than a
repository-wide search-and-replace.

| File | Imports MetaMo? | Role |
|---|---|---|
| `protocol.py` | no | `SkillOutcome`, `GovernorSignal`, and the `MotivationalGovernor` Protocol. Defines the seam in SubRep's own vocabulary. Dimension-generic — no objective count is hardcoded. |
| `budget.py` | no | The ε and α formulas. **The sign correction lives here** (§7). |
| `weights.py` | no | `w_meta` — projects an 8-goal, 6-modulator state onto the objective simplex (§5). |
| `stimulus.py` | no | Translates an executed outcome into MetaMo appraisal inputs, closing the loop. |
| `governor.py` | **yes, lazily** | `MetaMoGovernor`, the adapter that holds the `MotivationalState` and drives `step()`. Also `FakeGovernor` for MetaMo-free tests. |
| `controller.py` | no | Per-step orchestration: read signal, seed torch, certify, select, feed back. |
| `_loader.py` | path only | Locates the MetaMo checkout (§3). |

`bridge/README.md` is the canonical record of design decisions for this
package; update it when a decision changes.

---

## 5. `w_meta` — projecting 8 goals onto m objectives

MetaMo reasons over 8 goals; SubRep's Minecraft objective vector is 6-dimensional:

```
φ(x) = [Safety, Reputation, DeadlineSlack, InventoryValue,
        Sustainability, Infrastructure]
```

`weights.py` computes:

```python
scores = goal_affinity @ G + modulator_gain @ (M_clamped − 0.5)
w      = softmax(scores, temperature)
```

- **`goal_affinity`** `(m, 8)`, **non-negative** — "which motives make this
  objective matter". Goals are standing dispositions; they only add relevance.
- **`modulator_gain`** `(m, 6)`, **signed** — "how does current affective state
  tilt the balance". Modulators are transient context and must be able to
  suppress as well as promote: Securing→Safety is `+0.70`, Securing→Reputation
  is `−0.15` (fear pulls attention away from trading).

**Why softmax rather than clip-then-floor-then-normalize.** SubRep's selection
path does not validate weights at all — `score_skill_entry`
(`library/skill_selector.py:21-25`) accepts any array, with no shape, sign, or
sum check. A degenerate `[0,0,0,0,0,1]` would be silently accepted. Enforcement
therefore has to be structural in the producer. Softmax of finite input is
always strictly positive in every component, so no floor step is needed and no
floor can be violated. `GovernorSignal.validate()` re-checks non-negativity and
sum-to-1 as defence in depth.

**Calibration status.** The reference material reports three weight vectors at
different points in a scenario but never states the goal vector `G` at those
moments — only qualitative modulator movement. A matrix fitted to reproduce
those numbers would be inventing data. The default matrices are therefore
**semantically motivated, not fitted**, and the tests assert direction and
ordering (securing up → Safety weight up; social goals up → Reputation up)
rather than exact values. Treat the coefficients as a defensible starting point
to tune against a real environment.

---

## 6. One step, end to end

```
 1. governor.signal()          → w, ε, α from the current (G, M)
 2. controller seeds torch     → CVaR sampling becomes reproducible
 3. ε stamped onto candidates  → see §8
 4. pipeline.certify(...)      → CDS/PDS/CVaR gates run under ε and α
 5. controller.select(...)     → best admitted skill under w
 6. skill executes             → environment returns an outcome
 7. stimulus.py                → outcome becomes (novelty, conduciveness,
                                  risk, effort)
 8. governor.step(outcome)     → MetaMo's Ψ appraises, D decides,
                                  F = D∘Ψ produces the next state
 → back to 1
```

Step 8 is what makes it a loop rather than a one-way feed. `MetaMoGovernor`
holds the `MotivationalState` and replaces it with the value `step()` returns —
never mutating the previous one, which is what `step()`'s purity permits.

**Certification is cached.** `RuntimeCertificationPipeline` keys results by
`(context_key, skill_id)` (`utils/mdn_runtime_pipeline.py:219-227`). With a
constant context, a second certification returns the cached verdict and budget
changes have no effect. Callers must vary the context each step — the demo
passes the live environment observation.

---

## 7. ⚠️ The α sign correction

**The most consequential detail in this integration.** The reference formula is:

```
α = α₀ + b₁·securing + b₂·threshold − b₃·approach
```

Applied literally against SubRep's CVaR gate, this is **directionally wrong**.

`CVaRGate` averages the **lower tail at quantile `confidence`**
(`certification/cvar_test.py:54-58`):

```python
var_threshold = np.quantile(values, self.confidence)
tail_values   = values[values <= var_threshold]
return float(np.mean(tail_values))
```

So a *higher* α means a *shallower* tail, a *higher* CVaR, and an *easier* gate
— **less** conservative. Under the published formula, rising `securing` would
tighten the PDS budget while simultaneously loosening the CVaR gate. The two
admission gates would move in opposite directions on the same signal.

### The fix

1. **`confidence = α` as a direct numeric pass-through.** The conventions
   already agree — the reference states α = 0.1, and both `CVaRGate` and
   `RuntimePipelineConfig` default to `0.1`.
2. **Flip the b-coefficient signs** so α decreases as securing and threshold rise:

```
dev(x) = clip(x, 0, 1) − 0.5

ε = clip(ε₀ − a₁·dev(securing) + a₃·dev(approach),                    0, ε_max)
α = clip(α₀ − b₁·dev(securing) − b₂·dev(threshold) + b₃·dev(approach), α_min, α_max)
```

### Do **not** use `confidence = 1 − α`

It fixes the direction but destroys the baseline: α₀ = 0.1 would become
confidence 0.9, averaging the bottom 90% of the distribution — essentially the
plain mean — leaving the gate effectively inert at exactly the reference point.

### Why deviations from 0.5

Modulators are squashed around a 0.5 fixed point (§1), so raw values would mean
ε₀ and α₀ do not hold at the neutral state. The deviation form also reproduces
the reference trace exactly with `a₁ = 0.4`: securing 0.55 → ε = 0.08;
securing 0.45 → ε = 0.12.

### `α_min` is numerical, not stylistic

The CVaR tail holds ≈ `α · n_samples` draws. At `n_samples=1000`, α=0.01 leaves
10 — a noisy estimate and a flickering gate. The floor is
`max(0.02, min_tail_samples / n_samples)`, i.e. 0.05 at default settings.

### Regression guards

`test_bridge_budget.py::test_securing_tightens_both_gates` (unit) and
`test_bridge_e2e.py::test_s2_epsilon_and_alpha_never_diverge_under_sustained_threat`
(full loop) both fail if the published signs are reintroduced.

### The two α's must never touch

|  | `cvar_tail_level` | `mdn_concentration` |
|---|---|---|
| Type | scalar `float` | `np.ndarray`, length m |
| Range | (0, 1] | strictly positive |
| Meaning | CVaR tail mass | Dirichlet concentration |
| Source | MetaMo modulators | `generator/mdn.py` |
| Enters via | `CVaRGate(confidence=…)` | `.admit(…, mdn_alpha=…)` |

They share a letter and nothing else. This package never names a variable
`alpha`, and `GovernorSignal.validate()` rejects a vector where the scalar
belongs.

---

## 8. Two budgets, two routes

ε and α reach the gates by **different mechanisms**. This is not accidental
inconsistency — do not collapse it.

`certify_candidate_skills` accepts `cvar_confidence` as a call parameter but has
**no `epsilon` parameter** (`utils/mdn_runtime_pipeline.py:205-213`). It
resolves the PDS budget per candidate via `_candidate_epsilon` →
`_effective_epsilon`, which reads `candidate.epsilon` and falls back to the
static `config.pds_epsilon` when that is `None`
(`utils/mdn_runtime_pipeline.py:301-309`).

So `MetaMoController.certify` stamps `signal.pds_epsilon` onto each candidate
record before certification, via `_with_epsilon` (`dataclasses.replace` on the
frozen `CandidateSkillRecord`). The helper is duck-typed, so lightweight
non-dataclass stand-ins pass through untouched.

**Why it matters.** Without the stamp, ε is computed every step, stored on
`StepRecord`, and printed by the demo — while the gate keeps using the static
config value. Nothing looks broken: the number moves, the logs look right, and
only the gate's behaviour is wrong.

---

## 9. Determinism

`CVaRGate.get_cvar` draws from `torch.distributions.Dirichlet(...).sample()` on
the **global, unseeded** torch RNG (`certification/cvar_test.py:51`), so
certification decisions are not reproducible by default. `MetaMoController`
seeds torch before each certification pass and records the seed on every
`StepRecord`, so a run can be replayed exactly.

---

## 10. The stub environment

`env/minecraft_stub.py` is a deterministic, seeded 6-objective environment with
a threat level that rises and falls across an episode. **It is not Minecraft**
and makes no claim to simulate it — it exists so the m=6 loop can be exercised
end to end while a real environment does not exist in this repository.

### The borderline candidate

The PDS gate admits when `Δr + min(Δn) ≥ −ε`, and MetaMo drives ε over roughly
`[0, 0.1]`. So ε can only change an admission for a skill whose margin lands
inside `(−0.1, 0)`. The original five actions have margins from −2.0 to +11.4 —
all far outside that band, meaning ε **provably could not flip any decision**.

`RiskyForage` was added with a **noiseless** margin of ≈ −0.05 so that ε has
something to act on: above ~0.05 it is admitted, below it is rejected. Its
reward row is derived analytically rather than hand-tuned; the derivation is in
the `BORDERLINE CANDIDATE` comment in `env/minecraft_stub.py`, and the margin is
pinned by `test_minecraft_stub.py`.

If the episode length, γ, or the idle reward row change, that pin fails and the
row must be **re-derived from the comment**, not nudged until it passes.

### The margin only holds with noise off

The margin is ≈ −0.05. The stub's default `noise_scale=0.02`, accumulated over
a discounted 24-step episode across six objectives, perturbs it by a comparable
or larger amount — enough to push it outside the `(−0.1, 0)` band entirely.

Consequences:

- Any test that depends on ε flipping this candidate **must** construct the env
  with `noise_scale=0.0`. `test_bridge_e2e.py` does.
- `demo/run_metamo_pipeline.py` runs with the default noise, so `RiskyForage`
  typically sits just outside the band there (around −0.10) and PDS rejects it
  at every ε. The demo's PDS-only column therefore stays flat, and **the demo
  does not illustrate the ε coupling** — that is what S1 exists for.

The demo still shows the other two couplings clearly: ε and α tightening
together as threat rises (§7), and the Safety weight climbing (§5).

---

## 11. Testing

| File | Governor | Pipeline | Covers |
|---|---|---|---|
| `test_bridge_budget.py` | — | — | ε/α formulas, the sign correction, α clamping |
| `test_bridge_weights.py` | — | — | `w_meta` invariants, direction, ordering |
| `test_bridge_controller.py` | fake | fake | orchestration, forwarding, stimulus scaling |
| `test_bridge_governor.py` | **real** | — | the MetaMo adapter, `step()` purity |
| `test_minecraft_stub.py` | — | — | env contract, determinism, the borderline margin |
| `test_bridge_e2e.py` | **real** | **real** | the full loop |

The first five each substitute a fake for at least one half of the system, so
coupling bugs between the halves are invisible to them. `test_bridge_e2e.py`
exists for exactly that gap: it asserts only what requires the real stack
assembled together — budgets reaching certification, weights reaching
selection, ε/α coherence, safety-weight adaptation, abstention, and seed
determinism.

```bash
pytest tests/test_bridge_budget.py tests/test_bridge_weights.py -v   # no MetaMo needed
pytest tests/test_minecraft_stub.py tests/test_bridge_controller.py -v
pytest tests/test_bridge_governor.py -v      # skips without a MetaMo checkout
pytest tests/test_bridge_e2e.py -v
pytest tests/ -q                             # full regression

python demo/run_metamo_pipeline.py --steps 14
```

MetaMo-dependent tests **skip** rather than fail when no checkout is found. A
result of "13 skipped" means the submodule is not resolving — check §3.

### Observed behavior

Recorded from a full run of the suite and the demo, seed 42.

**Suite:** `pytest tests/ -q` → **633 passed, 2 xfailed, 0 failed.**
`test_bridge_e2e.py` alone: 10 passed.

**Budgets tighten together, never diverge.** Over a threatening run the governor's
modulators saturate within four steps, and both budgets fall with them:

| step | securing | ε | α |
|---:|---:|---:|---:|
| 0 | 0.500 | 0.1000 | 0.1000 |
| 1 | 0.672 | 0.0489 | 0.0500 |
| 2 | 0.843 | 0.0000 | 0.0500 |
| 3+ | ≥ 0.987 | 0.0000 | 0.0500 |

ε reaches its floor of 0.0; α reaches its sampling floor of 0.05 (`min_tail_samples
/ n_samples` at 1000 samples). At no step do they move in opposite directions. Under
the reference formula as published, α would have *risen* here while ε fell.

**Safety priority adapts.** The Safety weight climbs from 0.323 at step 0 to a peak
of 0.570 at step 3, then settles near 0.50 as the goal vector drifts under
MetaMo's decision monad. Reputation, the initially competing objective, falls
from 0.26 to 0.12 over the same run.

**Budgets reach certification.** With CVaR disabled so the PDS verdict is
observable, the borderline candidate is admitted at ε = 0.10 and rejected once
ε falls below its margin of ≈ −0.05 — first with hand-picked ε values (S1), then
using the ε MetaMo itself emits across the run (S1b), where the admitted count
drops as the budget tightens and never rises while it is falling.

**Priorities reach selection.** Weights at step 9 differ from step 0 for every
candidate's score, and the score gap between the safest and least-safe candidate
widens as Safety weight rises (S6b). With sufficiently separated weights the
selected skill changes outright (S6). With this candidate set the argmax does not
flip under MetaMo's own drift — `SwingGateBarricade` has both the highest Δr and
strong Safety, so it wins throughout.

**Abstention.** Given only inadmissible candidates the controller selects nothing,
and the governor's modulators still change on the fed-back outcome — abstaining
does not stall the loop.

**Determinism.** Two runs at the same seed produce identical ε, α, weights, and
selections at every step.

**What the demo does not show.** Run with its default noise, the demo's PDS-only
column stays at 4 and the combined admitted count at 6/6 throughout. The former is
because reward noise pushes the borderline margin to ≈ −0.104, outside the band ε
spans; the latter because the untrained MDN's CVaR gate admits everything under OR
semantics. Neither is a defect in the coupling — see §10 and §12.

> Results from the stub environment are **controlled test evidence** that the
> coupling behaves as designed. They are not real-environment validation — no
> real environment and no live agent are involved.

---

## 12. Known limitations

1. **OR gate semantics.** The demo runs `use_cvar=True` with
   `require_cds_or_cvar=True`, which returns `result or cvar_result` — there is
   no AND mode. With an **untrained** MDN the Dirichlet concentration is
   arbitrary and the CVaR gate admits nearly everything, overruling PDS
   rejections. The demo prints a PDS-only column alongside for this reason, and
   every e2e assertion about ε or abstention runs with `use_cvar=False`.
2. **`w_meta` coefficients are semantically motivated, not fitted** (§5).
3. **Affinity row sums act as an implicit prior.** Row totals differ (Reputation
   1.50 vs DeadlineSlack 0.90), so an objective's baseline weight depends partly
   on its row total regardless of `G`. Row-normalising would separate "which
   goals map here" from "how much this objective matters a priori".
4. **Softmax temperature is coupled to matrix scale.** Rescaling the affinity
   matrix changes the distribution's sharpness; the two must be retuned
   together. The same fix as (3) decouples them.
5. **Appraisal scaling must match the environment.** `stimulus.py` squashes
   through `tanh`; leaving `payoff_scale`/`motive_scale` at 1.0 while the
   environment reports deltas of order 10 saturates risk on the first step and
   flattens the coupling into a constant. The demo derives both from the
   candidate spread.
6. **The stub is not Minecraft** (§10).

---

## 13. Setup for a new contributor

```bash
git clone --recurse-submodules <repo-url>
cd subrep

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Confirm the submodule, not a sibling checkout, is live
python -c "from bridge._loader import find_metamo_root; print(find_metamo_root())"

pytest tests/ -q
python demo/run_metamo_pipeline.py --steps 14
```

Common pitfalls, in the order people hit them:

- **`find_metamo_root()` prints a path ending in `MetaMo-Python`** — the
  submodule did not populate and the loader fell back to a sibling checkout.
  Run `git submodule update --init`.
- **`test_bridge_governor.py` reports skips instead of passes** — same cause.
- **Editing anything under `external/metamo/`** — never do this.
  `git -C external/metamo status --porcelain` must stay empty.
- **Reintroducing the published α signs** — see §7 before touching
  `bridge/budget.py`.
