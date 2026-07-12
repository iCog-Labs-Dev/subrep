# SubRep: Subgoal Refinement and Representation Learning

## Executive Summary
This project develops a standalone **SubRep** implementation that transforms skill discovery into a **certificate-driven, auditable process**. SubRep certifies skills via two mathematical tests (**CDS/PDS**) that guarantee composition safety across motive shifts, preventing negative transfer before skills enter the library.

This project validates the core mechanism in **MO-LunarLander**, storing certified skills as native **MeTTa Atoms** for future Hyperon integration.

## Objectives & Key Results (OKRs)


| Objective | Goal | Implemented Capabilities |
| :--- | :--- | :--- |
| **1. Neural Skill Generator + MDN** | Generate skill summaries and learn motive geometry from experience | 2-head MLP for payoff/motives; MDN input/output contract; candidate-set MDN training and evaluation; auxiliary gate/Q heads |
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

# Run the full demo pipeline
python -m demo.run_full_pipeline
```

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
- MDN source and support-geometry metadata.

A representative mixed-candidate run produces both accepted and
rejected skills:

| Metric | Value |
| :--- | ---: |
| Total attempted | 10 |
| Admitted | 7 |
| Rejected | 3 |
| CDS admissions | 6 |
| PDS admissions | 1 |

The perturbed PPO candidate demonstrates a bounded trade-off case where CDS
fails but PDS admits within the demo epsilon budget (`5.0` on the discounted
rollout-return scale). Fixed-action candidates still make the report a realistic
safety check: rejected candidates are discarded before entering both
`CertificateStore` and `SkillLibrary`.

## MDN Checkpoint Behavior

The pipeline looks for the trained MDN checkpoint at:

```text
models/mdn_policy_best.pth
```

If that file is present, the pipeline uses the trained MDN and records:

```text
mdn_source: trained_checkpoint
```

If the checkpoint is missing, the pipeline falls back to `StubMDN` so tests and smoke runs still work. The stub returns fixed alpha/support values and should not be confused with the trained MDN.

## MDN Training and Evaluation

### Collect Training Candidate Sets

```bash
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 42 --prefix seed42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 43 --prefix seed43
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 44 --prefix seed44
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

### Neural Generator

- **Architecture:** 2-head MLP
- **Input:** state vector `(8,)`
- **Outputs:** scalar payoff `(1,)`, motive vector `(2,)`
- **Training:** supervised MSE on collected rollout payoff/motive totals

### MDN

- **Input:** context vector `(8,)`
- **Outputs:** Dirichlet alpha, 2D support values, auxiliary gate logit, auxiliary Q prediction
- **2D Support Contract:** support values satisfy `0 <= s0,s1 <= 1` and `s0 + s1 >= 1`

### Certification

- **CDS:** Cone-Dominant Subtask, universal-benefit admission
- **PDS-epsilon:** Pareto-Dominant Subtask, bounded trade-off admission
- **Supported regions:** `FULL_SIMPLEX` and 2D `MDN_WX`

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

- Extend candidate-set MDN evaluation beyond the current 2-objective MO-LunarLander testbed.
- Add MetaMo integration for dynamic weight management and risk budgets.
- Explore cross-paradigm skill sources through logic macros and evolutionary programs.
- Expand benchmark comparisons against MORL baselines.
