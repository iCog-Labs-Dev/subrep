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
git -C external/metamo checkout 4171b1d459aba897dc17489e2e8a6a31cc9ac4f2
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

### LunarLander: two-objective mapping

LunarLander uses a dedicated two-objective MetaMo mapping in
`env/lunar_lander_metamo.py`, producing normalized weights for **Safety** and
**Fuel**. The mapping represents caution through the Threshold and Securing
modulators, which increase the Safety weight, and boldness through Approach,
which increases the Fuel weight. Its goal-affinity and modulator-gain
coefficients are semantically designed and evaluated with a synthetic grid
search; they are **not fitted to actual LunarLander episode outcomes** and
should be revisited against real episode outcomes.

**Known limitations of the LunarLander loop**
(`env/lunar_lander_rollout.py`, `tests/test_lunar_lander_integration.py`):

- **The landing result counts in both Δr and Δn_Safety, by decision.** See
  "Payoff and Safety" below for why, and what counting it once would change.
- **Without the ε scale, ε cannot decide an admission.** ε is at most 0.5,
  while Δn values are in the hundreds, so crash-only candidates are rejected
  and landings admitted whatever ε is. With `--epsilon-scale on`, ε can decide
  the trade-off candidates (see "ε in LunarLander units" below). With the
  default motive risk, the live ε stays small after landings (0–0.109 in the
  measured run, so the tolerance stays under 33). With `--risk task`, it
  reaches 0.37 (see "Feeding MetaMo the task danger").
- **Evidence and execution share a seed.** Each decision step certifies the
  candidates and flies the chosen one from the same start. This shows the loop
  is wired correctly, not that a certificate holds on unseen starts.
- **CVaR is off.** The MDN is untrained, and an untrained MDN makes the CVaR
  gate admit nearly everything, so the loop certifies with PDS only.

### Payoff and Safety

**The convention.** The payoff r̂ is the landing result alone (+100 landed,
−100 crashed; `env/lunar_lander_rollout.py:174`). Safety is the landing result
plus the dense shaping (`env/lunar_lander_wrapper.py:73`). So a landing or crash
counts in both Δr and Δn_Safety. This is kept on purpose, and
`tests/test_lunar_lander_rollout.py::test_safety_includes_the_landing_result`
pins it.

**Why it is harmless.** The two conventions were compared on the same rollouts.
With γ = 1, the count-once Safety delta is exactly ΔSafety − Δr, so no second set
of episodes is needed. Measured with MetaMo `4171b1d`:

| | Landing counted twice (kept) | Counted once |
|---|---|---|
| PDS admissions, ε in [0, 0.5], 800 candidate–seed pairs (seeds 1–40) | — | identical |
| Selected skill, weights (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), 40 seeds | — | identical |
| Objective that binds min Δn | Fuel 592, Safety 208 | identical |
| Motive scale from `appraisal_scales`, seeds 1–20 | 250.9 | 226.0 |
| Demo, `--candidates ppo`, seeds 1–16: steps with ε = 0 | 1/16 | 3/16 |
| Demo, `--candidates extended`, seeds 1–16: steps with ε = 0 | 2/16 | 6/16 |
| Demo, `--candidates ci` | — | identical (nothing is admitted) |

Fuel is the objective that binds in most pairs, and the double count shifts
every landing's Safety by the same amount, so no admission or selection moves.

**The alternative: counting the landing once,** in the payoff only and in this
rollout only.

*Benefits:*
- **A cleaner meaning for Safety.** It would measure only how the flight went
  (the shaping: distance, speed, tilt and leg contact), and the payoff alone
  would carry the landing result.
- **No shared term.** Δr and Δn_Safety would be independent, so the PDS sum
  Δr + min Δn would never count the landing twice when Safety is the objective
  that binds.
- **Safer for rules that depend on the size of Safety,** for example
  per-objective scaling: the landing would not inflate it.

*Downsides:*
- **No measured decision gain:** admissions and selections are identical (table
  above). The only change is the smaller motive scale, which moves any rule
  built on it.
- **More caution with the current risk signal.** A median landing's risk rises
  from 0.393 to 0.435, and its novelty falls from 0.760 to 0.461 (novelty lowers
  Securing, `openpsi/appraisal.py:79-84`). The demo has more steps with ε = 0
  (table above), and after 30 identical landings fed to the real governor,
  Securing ends at 0.943 instead of 0.896.
- **Two meanings of Safety in one codebase:** shaping only in this rollout, and
  landing plus shaping everywhere else.
- **Every documented LunarLander number changes,** and a flag is needed to
  reproduce the current ones.
- **More code and tests to maintain:** the flag has to pass through
  `run_episode`, `evaluate_candidates`, `execute_candidate` and the demo, and be
  recorded in its JSON.

Under sustained crashes, Securing reaches 0.998–0.999 under both conventions, so
the choice does not change how MetaMo behaves under threat.

**Why the wrapper is not changed instead.** `SkillExecutor` builds its payoff
as the sum of the objective vector (`env/skill_executor.py:44`), and goal
decomposition follows it (`decomposition/effects.py:109-125`). Removing the
landing from Safety in the wrapper would remove it from their payoff altogether.
With the wrapper patched that way, four tests fail: the PPO pilot's payoff gain
over idling becomes −9.5 and it no longer passes any gate. The committed effect
tables in `decomposition/data/` would also no longer match the objectives they
were measured with.

**When to revisit.** Recheck this choice if a rule makes decisions depend on the
size of Safety, for example per-objective scaling, or if the risk signal fed
back to MetaMo changes.

### Running the LunarLander demo

`demo/run_lunar_lander_metamo.py` runs the full loop on MO-LunarLander, one
episode per decision step (step *k* uses seed `seed + k`):

```bash
python demo/run_lunar_lander_metamo.py --steps 16 --candidates ppo
python demo/run_lunar_lander_metamo.py --steps 10 --candidates ci      # no PPO checkpoint needed
```

| Option | Meaning |
|---|---|
| `--steps`, `--seed` | number of decision steps, and the first seed |
| `--candidates ci\|ppo\|extended\|tradeoff` | the candidate set; defaults to `ppo` when `models/pilot_ppo.pt` exists, otherwise `ci`. `tradeoff` is `extended` plus six PPO candidates that burn extra fuel (`env/lunar_lander_candidates.py`) |
| `--risk motive\|task` | the risk fed back to MetaMo. `task`: the flown episode's own danger (crash 1.0, landing 0.1, otherwise 0.5), and a `risk` column. `motive`: the worst objective change against the baseline. Default `motive` |
| `--epsilon-scale off\|on` | `on`: PDS tolerates ε·U with U = 300.9, and candidates that crash are rejected first; adds the `tol` and `fail` columns. Default `off` |
| `--no-selection noop\|stop` | when nothing is admitted: fly the noop baseline, or fly nothing. Both feed MetaMo a zero-Δ outcome with `admitted=False` |
| `--cvar`, `--cvar-samples` | also apply the CVaR gate (OR semantics); a `pds` column then shows the PDS-only count |
| `--verbose` | print every candidate's Δr, Δn, PDS margin and verdict |
| `--json-out PATH` | save the configuration, every step and the summary as JSON |

Example rows (`--candidates ci`, seeds 1–2):

```
step seed  secur thresh   appr     eps  alpha Safety   Fuel   adm  selected               result        d_r   min d_n
   0    1  0.500  0.500  0.500  0.1000  0.100  0.632  0.368  0/4   -                      crashed      +0.0      +0.0
   1    2  0.693  0.666  0.489  0.0207  0.050  0.718  0.282  0/4   -                      crashed      +0.0      +0.0
```

- `secur`, `thresh`, `appr`: Securing, Threshold and Approach at decision time.
- `eps`, `alpha`, `Safety`, `Fuel`: the signal SubRep certified and selected
  under. Without `--cvar`, no gate reads `alpha`.
- `adm`: admitted/scored candidates, as the pipeline decided them.
- `selected`: the chosen skill; `-` when nothing was admitted and the noop
  baseline flew; `STOP` when nothing flew.
- `result`, `d_r`, `min d_n`: how the flown episode ended, and the realised
  outcome fed back to MetaMo.

The summary reports measured diagnostics: the ε and α ranges, the first and
last weights, admissions, selections and flown results, the first step where
Securing reaches 0.95, and the number of steps where ε anywhere in [0, 0.5]
would change a PDS admission.

With `--epsilon-scale on`, the PDS-only count (`pds` column) and the
ε-sensitivity check apply the same rule as the gate: task failures are
rejected, and the tolerance is ε·U.

**Comparing two runs.** Save each run with `--json-out`, using the same options
and seeds, for example before and after updating the MetaMo submodule. Then
compare the `summary` blocks, or the per-step `signal` and `modulators`. The
demo is deterministic, so any difference comes from the change, not from
sampling.

### Before and after the MetaMo saturation fix

MetaMo `ceb108e` raised Securing and Threshold by 0.1 every time it projected a
state into the safe region, including states that were already safe. Caution
therefore climbed on every step, whatever the outcome. MetaMo `4171b1d` leaves
safe states unchanged, and when it does project, it shrinks the boost as the
modulator approaches 1 (`dynamics/stability.py`, `project_to_safe_region`).

The same demo run on both commits (`--steps 16 --candidates ppo`, seeds 1–16):

| | MetaMo `ceb108e` | MetaMo `4171b1d` |
|---|---|---|
| Securing at decision time | 0.500–1.000 | 0.500–0.955 |
| First step with Securing ≥ 0.95 | 3 | 8 (once, then back down) |
| Securing pinned at 1.0 | step 3 to the end | never |
| ε range | 0–0.100 | 0–0.132 |
| Steps with ε = 0 | 13/16 | 1/16 |
| Admitted / landed | 30/96, 16/16 | 30/96, 16/16 |
| Selections | ppo_deterministic 10, ppo_stochastic 6 | identical, step for step |

With the fix, caution follows the outcomes. Securing rose after the landings
with the largest objective losses (min Δn of −144, −174 and −121 at steps 5, 7
and 12) and eased after most of the others, so ε stays an active signal
through the run. What SubRep admits, selects and flies is unchanged, because ε does not
decide any admission at this scale (see the known limitations above).

On the `ci` candidates, where every episode crashes, Securing now settles at
about 0.925 instead of 1.0. ε still reaches 0 from step 2: above 0.75 at neutral
Approach, `a1_securing = 0.4` takes ε to 0 on its own. That is the bridge's
budget gain, not MetaMo.

To reproduce the `ceb108e` column without moving the submodule, export that
commit and point the loader at it:

```bash
git -C external/metamo archive ceb108e | (mkdir -p /tmp/metamo_ceb108e && tar -x -C /tmp/metamo_ceb108e)
SUBREP_METAMO_PATH=/tmp/metamo_ceb108e \
  python demo/run_lunar_lander_metamo.py --steps 16 --candidates ppo
```

### ε in LunarLander units

With `--epsilon-scale on`, the PDS gate tolerates ε·U, U = 300.9
(`LUNAR_LANDER_EPSILON_UNIT`), and candidates whose own payoff is below 0, the
crashes, are rejected first. The design decision and the alternatives are in
"ε in environment units" in `bridge/README.md`.

The same run with the scale off and on (`--steps 16 --candidates tradeoff`,
seeds 1–16, MetaMo `4171b1d`):

| | Scale off | Scale on |
|---|---|---|
| ε range | 0–0.109 | 0–0.109 |
| PDS tolerance | equal to ε | 0–32.7 |
| Rejected as task failures, before PDS | — | 99/224 |
| Admitted | 75/224 | 76/224 |
| Steps where some ε in [0, 0.5] would change an admission | 0/16 | 8/16 |
| Selections | ppo_deterministic 8, ppo_stochastic 6, ppo_waste_main_12 1, ppo_waste_side_15 1 | identical, step for step |
| Flown results | 16/16 landed | 16/16 landed |

ε could now decide half the steps, but the live ε is small, so it decides one
in this run: at step 15, `ppo_waste_side_15` lands with margin −7 and is
admitted under a tolerance of 11.8. The scale is off by default, and with it
off the demo's output and JSON carry no scale fields.

```bash
python demo/run_lunar_lander_metamo.py --steps 16 --candidates tradeoff --epsilon-scale on --verbose
```

### Feeding MetaMo the task danger

By default, the risk MetaMo receives is the worst objective change against the
baseline (`bridge/stimulus.py`). The noop baseline crashes, so a clean landing
still spends more fuel than doing nothing, and reads as moderately risky (about
0.39 for a median landing). Caution then stays high after landings, and ε stays
near 0.

**What MetaMo receives under the default risk** (seeds 1–40, 840 episodes, the
landing counted in both payoff and Safety; `experiments/caution_recovery/`):

| Outcome | Episodes | Risk: median (range) | 30 identical outcomes: Securing / ε |
|---|---|---|---|
| landing, task set | 75 | 0.407 (0.308–0.935) | 0.907 / 0.028 |
| landing, all sets | 279 | 0.494 (0.308–0.972) | — |
| landing, fuel-wasting trade-off set | 132 | 0.604 (0.393–0.972) | — |
| crash, task set | 241 | 0.862 (0.152–1.000) | 0.998 / 0 |
| crash, all sets | 417 | 0.407 (0.004–1.000) | — |
| timeout, all sets | 104 | 0.852 (0.703–0.998) | 0.998 / 0 |
| abstention, `noop` and `stop` | 40 / 1 | 0.250 (the rejection penalty) | 0.924 and 0.917 / 0 |

- **The default risk does not tell a crash from a landing.** A crash drawn at
  random reads as riskier than a landing drawn at random 71% of the time in
  the task set, and 45% of the time across all candidate sets, which is worse
  than chance.
- **Some crashes read as safer than a clean landing.** When the pilot's engines
  cut out early, the episode crashes much like noop does, so its worst
  objective change against noop is small (median ΔSafety −0.6 and ΔFuel −17 for
  `ppo_cut_50`), and so is its risk: a median of 0.115 for `ppo_cut_50` and 0.260
  for `ppo_cut_100`.
- **Fuel-wasting landings read as dangerous** (median 0.604). Repeated, 100 of
  the 132 drive ε to 0, although every one of them lands.
- **Landings alone hold Securing near 0.91 and ε near 0.03,** so caution never
  relaxes after a run of good outcomes.

The full tables, with the spread across episodes, the per-policy breakdown and
the alternative Safety convention, are in
`experiments/caution_recovery/results.md`. Regenerate them with
`python experiments/caution_recovery/measure.py`, on Linux: flight results
differ slightly by platform.

With `--risk task`, each outcome carries the flown episode's own danger
(`episode_outcome(..., task_risk=True)`, `env/lunar_lander_rollout.py`):

| Flown result | Risk |
|---|---|
| crashed | 1.0 |
| landed | 0.1 |
| neither (time limit, or a step cap) | 0.5 |

The rejection penalty is still added for a non-admitted outcome. In `stop`
mode nothing flies, so an abstention's risk is that penalty alone (0.25). The
landing level and why it is 0.1 are in "Measured risk" in `bridge/README.md`.

The same runs with each risk (seeds 1–16, MetaMo `4171b1d`):

| Run | Risk | ε range | Steps with ε = 0 | Securing range | Final Safety weight | Live ε changed an admission | Flown results |
|---|---|---|---|---|---|---|---|
| `--candidates ppo` | motive | 0–0.132 | 1/16 | 0.50–0.955 | 0.642 | — | 16/16 landed |
| `--candidates ppo` | **task** | **0.100–0.371** | **0/16** | 0.069–0.50 | 0.255 | — | 16/16 landed |
| `--candidates tradeoff --epsilon-scale on` | motive | 0–0.109 | 3/16 | 0.50–0.980 | 0.659 | 1/16 | 16/16 landed |
| `--candidates tradeoff --epsilon-scale on` | **task** | **0.100–0.372** | **0/16** | 0.067–0.50 | 0.261 | **3/16** | 16/16 landed |
| `--candidates ci --steps 10` | task | 0–0.100 | 9/10 | 0.50–0.998 | 0.907 | — | 10/10 crashed |

- **Caution now relaxes after landings and returns with crashes.** On `ppo`,
  Securing falls from 0.50 to about 0.07 and ε rises to 0.37, below ε_max 0.5.
  On `ci`, where every episode crashes, Securing climbs to 0.998 and ε goes to
  0, as before.
- **With the ε scale, the tolerance reaches 30–112,** and the live ε changes
  an admission on 3 of 16 steps (1 with the motive risk). Crashes are still
  rejected first, 99 of 224.
- **Selections change on a few steps** (2 of 16 on `ppo`, 3 of 16 on
  `tradeoff`), with every flight still landing. The weights move with the
  modulators: as caution relaxes, the Safety weight falls from 0.632 to about
  0.26.
- **Approach rises to about 0.99 while landings continue.** It is part of the
  relaxed state, and falls again with crashes (0.018 on `ci`).
- **α rises to 0.43 as caution relaxes.** With CVaR on, that would loosen the
  CVaR gate along with PDS. The demo certifies with PDS only.

`--risk motive`, the default, gives output and JSON byte-identical to the
earlier behaviour.

```bash
python demo/run_lunar_lander_metamo.py --steps 16 --candidates ppo --risk task
python demo/run_lunar_lander_metamo.py --steps 16 --candidates tradeoff --epsilon-scale on --risk task --verbose
```

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
                                  risk, effort); risk is the outcome's
                                  measured task danger when it carries one,
                                  else the worst objective change
 8. governor.step(outcome)     → MetaMo's Ψ appraises, D decides,
                                  F = D∘Ψ produces the next state
 → back to 1
```

Step 8 is what makes it a loop rather than a one-way feed. `MetaMoGovernor`
holds the `MotivationalState` and replaces it with the value `step()` returns —
never mutating the previous one, which is what `step()`'s purity permits.

**Certification is cached.** `RuntimeCertificationPipeline` keys results by
`(context_key, skill_id)`. The cached certificate is kept as the historical
record, but eligibility follows the **current** risk budgets: if ε (or the CVaR
tail level, when a CVaR gate is in use) differs from the values the certificate
was issued under, the stored delta is rechecked under the current budgets. A
skill certified at ε = 0.10 is therefore no longer admitted, or selected, at
ε = 0.00 in the same context. See §8.

The cache still returns the **stored** delta for a repeated context. Callers that
need fresh deltas pass a context that changes each step — the demo passes the
live environment observation.

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

**Cached certificates are rechecked when a budget changes.** Delivering the
budget is only half of it: the pipeline caches certificates per
`(context, skill)`, and a cached approval would otherwise outlive the budget it
was issued under. On a cache hit, the pipeline compares the stored certificate's
ε and CVaR tail level with the current ones (the tail level only when a CVaR
gate is in use). If they differ, it re-runs the gates on the stored delta under
the current budgets and returns the current verdict, margin and ε, leaving the
stored certificate untouched as the historical record. The recheck uses the W_x
region the certificate was originally gated against, because certifying a skill
records its selection weight into W_x, so only the budget differs from the
original check.

Regression guard: `test_s1c_cached_approval_is_not_selected_after_epsilon_tightens`
runs one pipeline, one context and one skill through the controller at
ε = 0.10 and then 0.00. The skill is selected at the first and not at the second,
and the historical certificate survives.

**An optional scale puts ε into the environment's units.** ε is dimensionless,
but the PDS gate compares it with Δr + min Δn, which is in reward units. Given
an `EpsilonScale(unit, payoff_floor)` (`bridge/epsilon_scale.py`), the
controller stamps the tolerance ε·U instead of ε, and first rejects any
candidate whose own payoff (baseline payoff + Δr) is below the floor. Those
candidates never reach the pipeline, so they are never cached and never grow
W_x, and their records carry `metadata["rejection_reason"] == "task_failure"`.
`StepRecord` gains `pds_tolerance` and `task_failures`. With the scale, the
stored ε is the tolerance, so the recheck above compares tolerances. Without
it, nothing changes. See "ε in environment units" in `bridge/README.md`.

---

## 9. Determinism

`CVaRGate.get_cvar` draws from `torch.distributions.Dirichlet(...).sample()` on
the **global, unseeded** torch RNG (`certification/cvar_test.py:51`), so
certification decisions are not reproducible by default. `MetaMoController`
seeds torch before each certification pass and records the seed on every
`StepRecord`, so a run can be replayed exactly.

The untrained MDN's **initial weights** are drawn from the same global RNG, but
when the model is built — before any certification. The demo and the e2e tests
therefore seed torch before constructing it; otherwise two same-seed runs would
certify with two different models, and diverge whenever the CVaR gate decides
an admission.

---

## 10. The stub environment

`env/minecraft_stub.py` is a deterministic, seeded 6-objective environment with
a threat level that rises and falls across an episode. **It is not Minecraft**
and makes no claim to simulate it — it exists so the m=6 loop can be exercised
end to end while a real environment does not exist in this repository.

It exposes what the rollout needs: `info["task_reward"]` (the task reward, sparse
trade value — only `DiscountChain` earns it), `phi()` (the current objective
levels), and read-only `observation()`, `threat` and `episode_over`.

### How Δr and Δn are estimated

All estimation on the MetaMo path lives in `env/minecraft_rollout.py`, following
the reference specification's definitions for an option of duration τ from
state x:

```
r̂(x, o) = Σ_{t<τ} γ^t · r(x_t, a_t)     task reward, info["task_reward"]
n̂(x, o) = Σ_{t<τ} γ^t · φ(x_t)          state levels, env.phi(), x₀ included
Δ(x, o) = (r̂, n̂)(x, o) − (r̂, n̂)(x, idle)
```

- **One horizon**, `DEFAULT_HORIZON = 3`, for candidate evaluation, execution
  and the feedback MetaMo appraises. An episode is 8 decisions.
- **Re-evaluated at every decision**, from the current state, on
  `copy.deepcopy(env)` so the live episode is never touched.
- **A fresh certification pipeline per episode.** The pipeline caches
  certificates by rounded context. On a hit it rechecks eligibility under the
  current risk budgets (§8), but returns the *stored* Δ. Every episode starts
  from the same observation, so reusing one pipeline would open each new episode
  on the previous episode's Δ.

It deliberately does **not** use `baseline/idle_policy.py`, which sums the reward
vector as the task reward and accumulates per-step changes. That is correct for
the 2-objective LunarLander environment it was written for — its objectives are
the environment's own reward components — and it is left untouched. The
replaced lines are kept as `# OLD:` comments in `run_option`.

### Calibration

The PDS gate admits when `Δr + min(Δn) ≥ −ε`, and MetaMo drives ε over
`[0, 0.1]`, so ε can only decide options whose margin lies in `(−0.1, 0)`.
Under the estimation above, the stub's original reward tables put every margin
between −0.3 and −1.0, so PDS rejected everything at every ε.

`_REWARD_SCALE = 0.08` (one uniform factor on the whole objective vector, noise
included) was **measured** across a full threat cycle to put every option in the
band, spread from ≈ −0.015 to ≈ −0.083. Each ε admits a different subset;
IronGolemSpawn's worst change (−0.083) matches the specification's O5 (−0.08);
and DiscountChain's margin falls as threat peaks, so a tightening ε stops the
agent trading under attack. `test_minecraft_stub.py` pins all of it — if one of
those tests fails, re-measure rather than nudge.

This replaces the earlier hand-built `RiskyForage` action, which existed only
because no natural option could reach the band.

The appraisal scales are floored at the specified trade value (0.01): the task
reward is sparse, so the mean |Δr| is ~0.002, and unfloored a single trade would
read as five units of payoff in MetaMo's appraisal.

---

## 11. Testing

| File | Governor | Pipeline | Covers |
|---|---|---|---|
| `test_bridge_budget.py` | — | — | ε/α formulas, the sign correction, α clamping |
| `test_bridge_weights.py` | — | — | `w_meta` invariants, direction, ordering |
| `test_bridge_controller.py` | fake | fake | orchestration, forwarding, stimulus scaling |
| `test_bridge_governor.py` | **real** | — | the MetaMo adapter, `step()` purity |
| `test_minecraft_stub.py` | — | — | env contract, determinism, task reward, φ, calibration |
| `test_minecraft_rollout.py` | — | — | Δ estimation: horizon, φ levels, task reward, per-state rollouts on copies |
| `test_spec_score_anchor.py` | — | — | the specification's scores (0.1310, 0.0620, 0.126) via SubRep's own scoring |
| `test_run_metamo_pipeline.py` | real (smoke) | real (smoke) | the demo: per-episode pipeline, a run across an episode reset |
| `test_bridge_e2e.py` | **real** | **real** | the full loop |

The bridge-level files above each substitute a fake for at least one half of the system, so
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

**Suite:** `pytest tests/ -q` → **1154 passed, 2 xfailed, 0 failed**
(MetaMo `4171b1d`). `test_bridge_e2e.py` alone: 11 passed.

**Budgets tighten together, never diverge.** Over a threatening run (the e2e
`run_loop`: synthetic threatening outcomes, CVaR off) the governor's modulators
climb to their ceiling within two steps, and both budgets fall with them:

| step | securing | ε | α | Safety w | admitted |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.500 | 0.1000 | 0.1000 | 0.323 | 5/5 |
| 1 | 0.744 | 0.0000 | 0.0500 | 0.597 | 0/5 |
| 2+ | 0.996 → 0.999 | 0.0000 | 0.0500 | 0.816 → 0.853 | 0/5 |

Every outcome in this run is a rejected option with a collapsing Safety
motive, so a caution level near its ceiling is the expected response.

ε reaches its floor of 0.0; α reaches its sampling floor of 0.05 (`min_tail_samples
/ n_samples` at 1000 samples). At no step do they move in opposite directions. Under
the reference formula as published, α would have *risen* here while ε fell.

**Safety priority adapts.** The Safety weight climbs from 0.323 to 0.853 over ten
steps; Reputation, the initially competing objective, falls from 0.133 to 0.009.

**Budgets reach certification.** With CVaR disabled so the PDS verdict is
observable, every calibrated option is admitted at ε = 0.10 and rejected at
ε = 0 — first with ε values derived from a natural candidate's margin
(`test_s1_epsilon_change_flips_a_borderline_admission`), then using the ε MetaMo
itself emits across the run (`test_s1b_governor_epsilon_change_alters_admissions`),
where the admitted count drops from 5/5 to 0/5 as the budget tightens and never
rises while it is falling.

**Priorities reach selection.** Weights at step 9 differ from step 0 for every
candidate's score, and the score gap between the safest and least-safe candidate
widens as Safety weight rises (`test_s6b_governor_weights_change_skill_scores`).
With sufficiently separated weights the selected skill changes outright
(`test_s6_different_weights_select_different_skills`).

**Selection follows threat.** In the demo the selected option changes over a run.
With ε held at its 0.1 baseline, selection under the specification's w̄0 weights
tracks threat: DiscountChain when calm, SwingGateBarricade as threat peaks,
DiscountChain again as it falls.

**Abstention.** Given only inadmissible candidates the controller selects nothing,
and the governor's modulators still change on the fed-back outcome — abstaining
does not stall the loop.

**Determinism.** Two runs at the same seed produce identical ε, α, weights, and
selections at every step, and two demo runs produce byte-identical output.

**What the demo shows that needs fixing next.** At the first threat Securing
jumps from 0.500 to about 0.94, then stays between 0.93 and 0.98 for the rest of
the run; it no longer pins at 1.0. ε is still 0 from step 1, but the cause is
now the budget gain rather than MetaMo: with `a1_securing = 0.4`
(`bridge/budget.py`), ε reaches 0 once Securing passes 0.75 at neutral Approach.
With every margin negative, PDS then admits nothing, and the agent abstains
until threat eases and the CVaR gate admits DiscountChain. Retuning the budget
gains is not addressed here.

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
   flattens the coupling into a constant. `appraisal_scales` derives both from
   the candidate spread and floors them at the specified trade value (§10).
6. **The stub is not Minecraft** (§10). Rollouts on a deep copy also carry its
   RNG, so candidate Δ values are *exact forecasts* of what execution will
   produce; a real environment would give noisy estimates.
7. **ε reaches 0 under sustained threat.** MetaMo no longer pins Securing at
   1.0 (its safe-region projection leaves already-safe states unchanged), but
   under threat Securing still settles above 0.75. There, with neutral
   Approach, `a1_securing = 0.4` takes ε to 0, and the b-coefficients hold α
   at its floor (§11, Observed behavior). Under sustained threat this is the
   intended direction. What kept ε near 0 after *good* outcomes on
   LunarLander was the risk signal, not the gains: with `--risk task`, ε
   recovers to 0.37 after landings (see "Feeding MetaMo the task danger").
   Retuning the budget gains is not addressed here.

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
