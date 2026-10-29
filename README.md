# SubRep: Subgoal Refinement and Representation Learning

## Executive Summary
This project develops a standalone **SubRep** implementation that transforms skill discovery into a **certificate-driven, auditable process**. SubRep certifies skills via two mathematical tests (**CDS/PDS**) that guarantee composition safety across motive shifts, preventing negative transfer before skills enter the library.

This project validates the core mechanism in **MO-LunarLander**, storing certified skills as native **MeTTa Atoms** for future Hyperon integration.

## Objectives & Key Results (OKRs)


| Objective | Goal | Implemented Capabilities |
| :--- | :--- | :--- |
| **1. Neural Skill Generator + MDN** | Generate skill summaries and learn motive geometry from experience | 2-head MLP for payoff/motives; MDN input/output contract with **feasibility guaranteed by construction at any objective count** (SASP); candidate-set MDN training and evaluation; auxiliary gate/Q heads |
| **2. Core Certification** | Implement CDS/PDS admission tests | CDS test; PDS-epsilon test; MO-LunarLander integration |
| **3. MeTTa Certificate Storage** | Store certificates as native atoms | Certificate schema; Hyperon-backed MeTTa bridge; store/retrieve/query operations |
| **4. Validation** | Demonstrate the certificate-driven mechanism works | Certified skills pass; unsafe skills are rejected; admission reports document pass/fail behavior |

## Quick Start

### 1. Prerequisites
- Python 3.8+
- Git

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/iCog-Labs-Dev/subrep.git
cd subrep


#Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

python -m pip install -r requirements.txt
```

### 3. Validation

```bash
# Run all tests
python -m pytest -v

# Run certification tests only
python -m pytest tests/test_certification_gates.py -v

# Validate the MDN support-value feasibility guarantee at M = 2, 3, 5, 10, 50
python -m pytest tests/test_mdn.py -v

# Validate the exact greedy W_x solver and the M >= 2 certification chain
python -m pytest tests/test_skill_library.py -v

# Run the full demo pipeline
python -m demo.run_full_pipeline

# Validate mid-episode motive-shift reuse behavior
python -m pytest tests/test_mid_episode_reuse_demo.py -v
```

> MeTTa-backed tests require the `hyperon` package, which is Linux/macOS only. On Windows they are
> skipped automatically via `pytest.importorskip`; run the suite under WSL to exercise them.

## Running the Demo Pipeline

```bash
python -m demo.run_full_pipeline
```

The demo pipeline:

- computes an idle baseline,
- executes a mixed candidate pool: deterministic PPO, stochastic PPO, perturbed/noisy PPO, fixed-action policies, and random policy,
- uses the SkillGenerator, when available, only as a pre-filter for promising base-PPO starting contexts,
- computes `delta_r` and `delta_n`,
- certifies skills with CDS/PDS,
- stores admitted certificates in MeTTa and `SkillLibrary`,
- writes admission reports to `demo/artifacts/`,
- runs MDN-based skill selection from the certified library.
- demonstrates zero-shot reuse after a mid-episode motive-priority shift:
  a previously certified global skill remains reusable without retraining,
  while a contextual MDN_WX skill can be correctly rejected under shifted
  runtime support geometry.

To open the Streamlit demo app:

```bash
streamlit run demo/streamlit_subrep_demo.py
```

The Streamlit app is the demo interface. It can run the real
pipeline from the sidebar, then presents the full SubRep story in one place:
skill execution, improvement calculation, CDS/PDS admission, certificate
storage, trained-MDN selection, zero-shot reuse, and the final audit tables.

## PPO Pilot Reproducibility

```bash
# Regenerate the committed PPO pilot checkpoint
python -m pilot.train_pilot --seed 7 --output models/pilot_ppo.pt

# Validate the checkpoint without retraining
python -m pytest tests/test_pilot_performance.py -v
```

## Optional SafeRL Benchmark Pilot

For the held-out three-objective benchmark using the retrained PPO policies,
actual SkillLibrary admission/querying, fixed-policy baselines, and paired
confidence intervals, see [Safety reuse benchmark](docs/SAFETY_REUSE_BENCHMARK.md).

Safety-Gymnasium uses older pinned Gymnasium/Pygame versions, so keep it in a
separate Python 3.10 environment instead of the main SubRep `.venv`:

```bash
conda create -n subrep-safety python=3.10 -y
conda activate subrep-safety
python -m pip install -r requirements-safety.txt
```

Smoke-test the benchmark install:

```bash
python - <<'PY'
import safety_gymnasium

env = safety_gymnasium.make("SafetyPointGoal1-v0")
obs, info = env.reset(seed=42)
obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
env.close()

print("Safety-Gymnasium works")
print("obs shape:", obs.shape)
print("reward:", reward)
print("cost:", cost)
PY
```

Train the lightweight PPO baseline and collect first-pass SafeRL candidate
rollouts. `--no-capture-output` makes conda stream progress logs live:

```bash
conda run --no-capture-output -n subrep-safety python -m pilot.train_safety_gymnasium_ppo \
  --env-id SafetyPointGoal1-v0 \
  --total-updates 5 \
  --rollout-steps 256 \
  --update-epochs 2 \
  --minibatch-size 128 \
  --max-episode-steps 200 \
  --eval-episodes 5 \
  --output models/safety_ppo_point_goal.pt

conda run --no-capture-output -n subrep-safety python -m data_collector.collect_safety_gymnasium_rollouts \
  --env-id SafetyPointGoal1-v0 \
  --contexts 25 \
  --max-steps 200 \
  --save-dir data/safety_gymnasium_rollouts \
  --seed 42 \
  --ppo-checkpoint models/safety_ppo_point_goal.pt
```

Then switch back to the main SubRep environment before certification. The
Safety-Gymnasium conda environment is only for collecting benchmark rollouts;
the certification/report path uses the normal SubRep dependencies, including
Hyperon/MeTTa:

```bash
conda deactivate
source .venv/bin/activate
```

Certify the collected SafeRL rollouts and generate the admission/reuse report:

```bash
python -m demo.run_safety_gymnasium_pipeline \
  --rollout-dir data/safety_gymnasium_rollouts \
  --pds-epsilon 1.0
```

The SafeRL wrapper maps benchmark outputs into SubRep format:

```text
reward -> task payoff / task motive
cost   -> safety motive, with larger values meaning safer behavior
```

The SafeRL certification pilot uses `zero_action` as the same-context baseline.
Every other candidate, including the trained PPO candidate when present, is
compared against that baseline, certified with CDS/PDS, and only admitted
certificates enter `CertificateStore` and `SkillLibrary`. The generated report
also includes a zero-shot reuse query under task-focused and safety-focused
weights without retraining.

SafeRL report outputs:

- `demo/artifacts/safety_gymnasium_admission_report.json`
- `demo/artifacts/safety_gymnasium_admission_report.md`
- `data/safety_gymnasium_certificates.metta`
- `data/safety_gymnasium_library.json`

The written pilot summary is in
[`docs/SAFERL_BENCHMARK_PILOT.md`](docs/SAFERL_BENCHMARK_PILOT.md).

## Admission Report Output

After running the demo pipeline, admission statistics are generated at:

- `demo/artifacts/admission_report.json`
- `demo/artifacts/admission_report.md`

The report includes:

- total attempted, admitted, and rejected skills,
- admission and rejection rates,
- CDS and PDS pass counts,
- failure reasons for rejected skills,
- example admitted/rejected records,
- MDN source and support-geometry metadata,
- `infeasible_support_events`: a permanent feasibility counter, expected to read `0`. SASP makes an
  empty `W_x` algebraically impossible, so any nonzero value signals a code regression rather than a
  tuning problem.

### Full Certification Audit

Beyond the summary counts, the report carries **one audit entry per evaluated candidate**, so every
admission and rejection can be inspected individually rather than through a handful of examples:

- `audit_entries`: one record per candidate, each containing the skill ID, candidate policy, gate
  type, admitted/rejected, `delta_r`, the complete `delta_n` vector, weight region type, support
  values and support feasibility for `MDN_WX`, epsilon, admission margin, rejection category and
  reason, baseline policy, environment, seed, and episode length,
- `rejection_categories`: rejections grouped by cause — `GATE_FAILED`,
  `DUPLICATE_SKILL_ID`, `LIBRARY_REVERIFICATION_FAILED`, `INFEASIBLE_SUPPORT` — rather than by
  message text, so one cause counts once,
- `by_policy` and `by_weight_region`: attempted/admitted/rejected/CDS/PDS totals per candidate
  policy and per weight region.

Each entry records the **exact inequality** the gates compared, with both sides exposed:

```text
CDS:  delta_r + worst_case >= 0
PDS:  delta_r + worst_case >= -epsilon

worst_case = min(delta_n)                    # FULL_SIMPLEX
           = min_w (w . delta_n) over W_x    # MDN_WX, via the exact greedy solver
```

Both gate evaluations are stored for every candidate, admitted or rejected, so the report shows
which gates were tried and how each fared — not only the gate that happened to admit. Vectors are
rendered at their full length, so records stay readable at any objective count.

The Markdown report adds matching sections: `Per-Candidate Audit` (with an `MDN_WX Support Geometry`
sub-table when contextual decisions are present), `Summary by Candidate Policy`,
`Summary by Weight Region`, and `Rejection Categories`.

A representative mixed-candidate run produces both accepted and
rejected skills:

| Metric | Value |
| :--- | ---: |
| Total attempted | 10 |
| Admitted | 3 |
| Rejected | 7 |
| CDS admissions | 2 |
| PDS admissions | 1 |

`ppo_deterministic` and `noop` clear CDS. `ppo_then_side_tradeoff` is the
bounded-trade-off case: worst case `-2.73`, so CDS rejects while PDS admits inside
the `5.0` budget. Everything else is rejected before entering `CertificateStore`
or `SkillLibrary`. Episode 10 re-runs `ppo_deterministic` from a different start
state and is rejected — admission is a property of the rollout, not the policy.

> **These counts changed in `fix/lunar-lander-fuel-reward-sign`.** The previous
> split (7 admitted / 3 rejected) came from a sign-inverted fuel objective that
> credited wasted fuel as a benefit; a random policy cleared CDS under it. See
> *Objective Mapping* below and `docs/INTEGRATION_REPORT.md` §2.2.

## MDN Checkpoint Behavior

The pipeline looks for the trained MDN checkpoint at:

```text
models/mdn_policy_best.pth
```

If a **SASP-compatible** checkpoint is present, the pipeline uses the trained MDN and records:

```text
mdn_source: trained_checkpoint
```

The pipeline falls back to `StubMDN` in two cases, so tests and smoke runs always work. The stub
returns fixed alpha/support values and should not be confused with the trained MDN.

1. **The checkpoint is missing.**
2. **The checkpoint predates SASP.** The support head widened from `M` to `2M` outputs, so older
   weights cannot be reinterpreted — the first `M` logits are now softmax base-allocation logits and
   the last `M` are slack gates, which is not what the old weights meant. Both loaders detect this
   and raise `IncompatibleCheckpointError` rather than silently producing wrong support geometry.

> **The checkpoint currently committed to this repository is pre-SASP.** Running the demo today
> therefore prints `MIGRATION REQUIRED` and reports `mdn_source: stub`. **This is correct behavior,
> not a failure.** Retrain to restore `trained_checkpoint` — see *MDN Training and Evaluation* below.

Note that `train_mdn_candidate_sets` optimizes a policy loss that is a function of the Dirichlet
`alpha` alone, so it does **not** train the support head. Support values are trained separately by
`MDNSupportTrainer` (`generator/mdn_support_trainer.py`) against `WeightSetStore` targets. A retrain
produces a shape-correct checkpoint with a freshly initialized support head — still feasible by
construction, but not fitted.

## MDN Training and Evaluation

### Collect Training Candidate Sets

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 42 --prefix seed42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 10042 --prefix seed10042
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 20042 --prefix seed20042
```

This produces 3,000 contexts and 21,000 candidate outcomes with the default
candidate set: deterministic PPO, stochastic PPO, fixed noop/engine policies,
and random policy.

### Train the MDN

```bash
python -m generator.train_mdn_candidate_sets \
  --data-dir data/mdn_candidate_sets \
  --pattern "*.npz" \
  --seed 42 \
  --device cpu \
  --policy-checkpoint models/mdn_policy_best.pth \
  --auxiliary-checkpoint models/mdn_auxiliary_best.pth \
  --q-loss mse
```

Outputs:

- `models/mdn_policy_best.pth`: trained MDN policy/runtime checkpoint
- `models/mdn_auxiliary_best.pth`: trained auxiliary checkpoint

Final MDN training uses candidate-set supervised training with normalized Q targets and MSE Q loss. IPS/DR support exists in the auxiliary trainer for future off-policy logged-data settings, but the final candidate-set checkpoint is not DR-trained.

### Evaluate the MDN

```bash
python -m generator.evaluate_mdn_candidate_sets \
  --checkpoint models/mdn_policy_best.pth \
  --data-dir data/mdn_candidate_sets_eval \
  --pattern "*.npz" \
  --seed 100 \
  --device cpu
```

The evaluator reports lift versus PPO/random baselines, balanced top-1 accuracy, regret, gate F1, Q/motive error, per-objective Q diagnostics, and bootstrap confidence intervals.

## Multi-Objective Benchmark

SubRep includes a lightweight synthetic benchmark for validating the
certification chain beyond the original two-objective LunarLander setup. It
checks CDS/PDS admission, `SkillLibrary.query_admissible()`, `MDN_WX` support
regions, reuse under motive shifts, negative-transfer cases, query timing, and
simple baselines for `M = 3, 4, 5+`.

```bash
python -m demo.run_multi_objective_benchmark \
  --objectives 3 4 5 \
  --candidates 48 \
  --seeds 11 23 37 \
  --output demo/artifacts/multi_objective_benchmark.json \
  --markdown-output demo/artifacts/multi_objective_benchmark.md
```

Outputs:

- `demo/artifacts/multi_objective_benchmark.json`
- `demo/artifacts/multi_objective_benchmark.md`

## Project Structure

| Folder | Description |
| :--- | :--- |
| `env/` | MO-LunarLander wrapper and skill execution loop |
| `baseline/` | Idle baseline and improvement computation |
| `generator/` | Skill generator, MDN model, trainers, and evaluators |
| `pilot/` | PPO pilot policy, training entry point, and checkpoint utilities |
| `certification/` | CDS/PDS gates, certificate schema, and MeTTa storage |
| `library/` | Runtime skill library and selection strategies |
| `utils/` | Shared MDN, geometry, data, checkpoint, and report helpers |
| `data_collector/` | Raw rollout and candidate-set data collectors |
| `demo/` | End-to-end pipeline and generated admission reports |
| `tests/` | Unit, integration, runtime, and end-to-end tests |

## Technical Specifications

### Environment

- **Platform:** `mo-gymnasium` (`MO-LunarLander-v3`)
- **Observation Space:** `(8,)`
- **Reward Space:** `(2,)` mapped to `[Safety, Fuel]`

#### Objective Mapping

`MO-LunarLander-v3` emits four raw objectives, which `SubRepEnv._map_rewards`
folds into SubRep's two:

```text
raw[0] terminal result (+100 landed / -100 crashed)  ->  Safety = raw[0] + raw[1]
raw[1] dense potential-based shaping                 ->
raw[2] main engine cost  (-m_power, <= 0)            ->  Fuel   = raw[2] + raw[3]
raw[3] side engine cost  (-s_power, <= 0)            ->
```

Every motive is **larger-is-better**, since CDS admits on
`delta_r + min_i(delta_n_i) >= 0`.

The engine entries arrive as non-positive costs, so their sum is already oriented
correctly — `0.0` when no engine fires, negative as fuel burns — and is passed
through unchanged. Negating it would make burning fuel outrank conserving it;
`tests/test_env.py::test_fuel_objective_prefers_less_fuel` pins the ordering.

The SafeRL adapter *does* negate, because Safety-Gymnasium's `cost` is
positive-is-worse. Same transform, opposite input convention — check the sign of
the source before mapping a new environment.

### Neural Generator

- **Architecture:** 2-head MLP
- **Input:** state vector `(8,)`
- **Outputs:** scalar payoff `(1,)`, motive vector `(2,)`
- **Training:** supervised MSE on collected rollout payoff/motive totals

### MDN

- **Input:** context vector `(8,)`
- **Outputs:** Dirichlet alpha, support values, auxiliary gate logit, auxiliary Q prediction
- **Support head width:** `2M` logits — first `M` are base-allocation, last `M` are slack gates
- **Support Contract (any M):** `0 <= s_i <= 1` and `sum(s) >= 1`

Support values define the admissible weighting region `W_x = { w in simplex : w_i <= s_i }` that the
CDS and PDS gates evaluate against. Both constraints are required for that region to be non-empty:
if `sum(s) < 1`, no weight vector can respect every per-objective cap while summing to 1.

They are decoded by **SASP** (Softmax-Anchored Slack Parameterization), which makes both constraints
hold **by construction at any objective count**:

```python
p = softmax(raw[..., :M])                                     # sums to 1
g = slack_floor + (1 - slack_floor) * sigmoid(raw[..., M:])   # in (g_min, 1)
s = p + (1 - p) * g
```

`s_i` is a convex combination of `p_i` and 1, so `s_i` is in `[0, 1]`; and
`sum(s) = sum(p) + sum((1 - p_i) * g_i) >= 1`. Both are algebraic, so they hold for any network
weights at every training step and at inference — no penalty term or coefficient tuning is involved.
The construction is permutation-equivariant, so no objective is privileged by its index, and
`slack_floor` prevents `W_x` collapsing to a single point.

### Certification

- **CDS:** Cone-Dominant Subtask, universal-benefit admission
- **PDS-epsilon:** Pareto-Dominant Subtask, bounded trade-off admission
- **Supported regions:** `FULL_SIMPLEX` and `MDN_WX` at any `M >= 2`
- **Worst-case evaluation:** exact `O(M log M)` greedy support function, no vertex enumeration

The gates need `h_Wx(c) = max { w . c : w in simplex, w_i <= s_i }`. This is a linear program with a
closed-form greedy solution — sort coordinates by coefficient descending and fill each to its cap
until total mass 1 is placed — exact because the feasible set is the base polytope of a polymatroid.
Vertex enumeration was retired because its cost grows combinatorially with M.

### MeTTa Integration

- **Package:** `hyperon`
- **Active implementation:** `certification/metta_bridge.py` and `certification/metta_storage.py`
- **Persistence:** `data/certificates.metta`

## Documentation

- `generator/README.md`: skill-generator and MDN training/evaluation
- `data/README.md`: rollout and candidate-set data schemas
- `docs/CERTIFICATE_STORAGE.md`: certificate schema and MeTTa atom format
- `docs/ZERO_SHOT_PROTOCOL.md`: full-simplex and MDN_WX reuse protocol
- `docs/INTEGRATION_REPORT.md`: integration and validation report
- `docs/METTA_INTEGRATION.md`: MeTTa and Hyperon integration notes
- `docs/IPS_DR_LOGGED_TRAINING.md`: probability-aware logged-data workflow for IPS/DR checkpoints
- [MeTTa Python Integration Guide](https://metta-lang.dev/docs/learn/tutorials/python_use/metta_python_basics.html)

## Future Work

- Retrain the MDN under SASP and validate held-out metrics; the committed checkpoint is pre-SASP.
- Exercise the MDN on a testbed with more than two objectives. The support head, certification chain,
  certificate schema and MeTTa storage are all correct at any `M >= 2`, but MO-LunarLander only
  provides `[Safety, Fuel]`, so M > 2 is currently covered by tests rather than by a live environment.
- Add MetaMo integration for dynamic weight management and risk budgets.
- Explore cross-paradigm skill sources through logic macros and evolutionary programs.
- Expand benchmark comparisons against MORL baselines.

### Exact-context certification and reuse pilot

Run from this directory in the `subrep-safety` environment:

```bash
conda run --no-capture-output -n subrep-safety python -m demo.run_contextual_safety_benchmark \
  --output data/contextual_safety_smoke_v1 \
  --training-seeds 42 --development-contexts 10 --contexts 4 --max-steps 200
```

Use a new output directory for every run. This runs existing policies; it does
not train them. `report.md` lists individual admission paths and re-execution
outcomes. `admissions.json` includes margins and unavailable MDN checks.
Each seed directory contains raw development/evidence records with initial
observations; each context directory contains its library, pre-execution
selection, and subsequent execution outcomes.

Certificates apply only to the same environment configuration, reset seed and
exact initial observation. Full-simplex certification does not waive this
context check. This pilot does not establish transfer to unseen situations.
Policies are re-executed from the same start; deterministic policies can produce
identical outcomes. Random policies use a separate repetition seed. Task payoff
still also appears in the motive vector, matching the previous benchmark.

The ordered admission path is full-simplex CDS, full-simplex PDS, MDN-region CDS,
then MDN-region PDS, stopping at the first pass. Without a checkpoint, MDN checks
are explicitly unavailable. To enable them, append `--mdn-checkpoint PATH
--mdn-contract CONTRACT.json`. The model must use three objectives and raw native
flat observations, with a compatible SASP head. The training-associated contract
must contain:

```json
{"environment":"SafetyPointGoal1-v0","observation_schema":"native_flat_v1","objectives":["safety","task","control_efficiency"]}
```

This contract must describe the actual training inputs; it is not a conversion
for an incompatible model. Learned regions are stored with certificates and
queries outside those regions cannot reuse those entries. The contextual runner
applies these checks explicitly; the older batch-average runner is unchanged.

### Three-objective Safety-Gymnasium data with the existing MDN trainer

The MDN architecture and its context-only inference / context-plus-skill-ID
auxiliary paths are unchanged. The candidate-set record builder infers M from
the outcomes. Use `--baseline-candidate-id zero_action` to use the measured
baseline from each starting context, rather than running a LunarLander baseline.
The baseline remains in the candidate set, as an explicit zero-improvement option.

```bash
python -m data_collector.collect_safety_gymnasium_rollouts \
  --objectives 3 --contexts 300 --seed 120000 --max-steps 200 \
  --save-dir data/safety_mdn3_train_v1 --prefix train \
  --ppo-checkpoint models/safety_ppo_fixed_seed42_updates50.pt \
  --ppo-lagrangian-checkpoint models/safety_ppo_lagrangian_fixed_seed42_updates50.pt

python -m generator.train_mdn_candidate_sets \
  --data-dir data/safety_mdn3_train_v1 --baseline-candidate-id zero_action \
  --device cpu --seed 42 \
  --policy-checkpoint models/safety_mdn3_policy_v1.pth \
  --auxiliary-checkpoint models/safety_mdn3_auxiliary_v1.pth
```

Use fresh directories/checkpoint paths and keep separate context seeds reserved
for final evaluation. These commands retain the existing training objectives,
normalization, skill-ID mapping and policy/auxiliary phases. They do not introduce
manual priority labels. The existing candidate-set trainer does not explicitly
supervise support-region targets; SASP feasibility alone does not validate a
learned region. Review support learning and held-out region coverage separately
before claiming MDN-based certification quality. Its internal validation is not
a substitute for a separate untouched benchmark test set.

### Export contextual selections for support fitting

The existing MDN architecture is unchanged. Export a completed **non-MDN**
contextual run to the existing support trainer's WeightSetStore format:

```bash
python -m generator.export_contextual_support \
  --benchmark data/contextual_safety_smoke_v2 \
  --output data/contextual_support_export_v1
```

The exporter rechecks selected full-simplex certificates against the saved
same-context candidate/idle measurements and scales. It excludes fallback
selections, preserves evidence files, and writes source hashes and selection
provenance to `manifest.json`. It does not run environments or train anything.
Targets remain coordinate maxima of observed certified-selection priorities,
not proven safe-region boundaries. Identical priority sets yield constant
targets: fitting these does not demonstrate context-dependent region learning.

At least three admitted contexts are required for the trainer's disjoint split;
the four-context smoke run only has two and cannot support that split. Collect
a separate support-development run (reserve other seeds for final benchmarking):

```bash
python -m demo.run_contextual_safety_benchmark \
  --output data/contextual_support_development_v1 \
  --training-seeds 42 --seed 130000 \
  --development-contexts 30 --contexts 100 --max-steps 200
python -m generator.export_contextual_support \
  --benchmark data/contextual_support_development_v1 \
  --output data/contextual_support_export_v2
```

Check `ready_for_split` and `constant_targets` in the export summary. With a
previously trained **three-objective Safety MDN** using the same native
observation schema (see training commands above), run:

```bash
python -m generator.train_mdn_support \
  --base-checkpoint models/safety_mdn3_auxiliary_v1.pth \
  --weight-store data/contextual_support_export_v2/weights.json \
  --output-checkpoint models/safety_mdn3_support_v1.pth \
  --output-dir data/safety_mdn3_support_evaluation_v1 \
  --device cpu --seed 42
```

This command fits and evaluates support targets only. It does not establish
certification correctness or performance improvement in fresh executions.
Do not use a two-objective or LunarLander checkpoint. Base-model training data
should also exclude the support validation/test and final benchmark contexts.

Check the bridge without running benchmark experiments:

```bash
python -m pytest -q tests/test_export_contextual_support.py \
  tests/test_mdn_support_trainer.py tests/test_train_mdn_support.py
```

### Monitored policy retraining

`pilot.train_safety_gymnasium_ppo` accepts `--eval-every` (default 25) and
`--eval-seed` (default 180000). These seeds are development data used for
checkpoint selection, not a final evaluation set. Use a fresh `--output` path.
The output checkpoint is the best development checkpoint; its sibling
`<checkpoint_stem>_run/` contains `metrics.jsonl`, `summary.json`, and periodic
`update_*.pt` snapshots. Snapshots are policy checkpoints, not optimizer-resume
checkpoints. PPO is selected by goal-reaching rate, then return. Lagrangian
selection prefers checkpoints satisfying mean episode cost <= `--cost-limit`,
then goal rate and return. If none satisfies the budget, lower cost is preferred
and `summary.json` reports `budget_met: false`. Budget satisfaction on development
episodes does not guarantee safety on new contexts.

Metrics include deterministic development goal success, raw episode return and
cost, multiplier, sampled-action clipping, and mean-action saturation. No training
architecture or action-distribution change is made. The Lagrangian multiplier
updates only when completed episode cost evidence is available.

```bash
python -m pytest -q tests/test_safety_ppo_training.py
python -m pilot.train_safety_gymnasium_ppo \
  --output models/safety_ppo_monitored_seed42_v1.pt \
  --seed 42 --total-updates 500 --cost-penalty 0 \
  --max-episode-steps 200 --eval-every 25 --eval-episodes 20 \
  --eval-seed 180000 --device cpu
python -m pilot.train_safety_gymnasium_ppo \
  --output models/safety_ppo_lagrangian_monitored_seed42_v1.pt \
  --seed 42 --total-updates 500 --use-lagrangian --cost-limit 1 \
  --lambda-lr 0.005 --max-episode-steps 200 \
  --eval-every 25 --eval-episodes 20 --eval-seed 180000 --device cpu
```

The smaller multiplier step is a development experiment, not a demonstrated fix.
The 500-update budget is 512,000 environment transitions, excluding evaluation.
Keep old policies/data. Review these runs before replacing policies or recollecting
MDN data, and reserve new contexts for final evaluation.

### Lagrangian PPO trainer v2

New `--use-lagrangian` runs use independent actor, reward-critic, and
cost-critic networks. Reward and cost GAE are computed separately, then
`A_reward - lambda * A_cost` is combined and normalized for the actor.
Critic gradients do not update the actor backbone or share its gradient
clipping budget. Episode costs and the cost limit remain in raw units.
Plain PPO and the MDN architecture are unchanged. Older policy checkpoints
remain loadable; new policy checkpoints record the separate critic layout.
The cost critic is training-only; saved checkpoints are for inference, not
optimizer/training resumption.

Use a new output filename for retraining. Inspect the run directory's
`metrics.jsonl` (including policy loss, reward/cost value losses and approximate
KL) and `summary.json`. Passing tests verifies the implementation paths, not
that a run will learn a successful policy within the safety budget.

Lagrangian trainer v3 (`finite_episode_cost_v3`) aligns cost value targets with
raw episode cost: cost gamma is 1, cost continuation is zero at termination,
truncation or the configured episode cutoff, and the cost critic receives the
remaining fraction of the episode. Unfinished rollout buffers still bootstrap
cost values. Reward discounting/bootstrapping, the actor observation interface,
and MDN are unchanged. This fixes the cost-target definition; it does not
promise convergence or budget compliance. Retrain under a new checkpoint name.
