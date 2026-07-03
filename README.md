# SubRep: Subgoal Refinement and Representation Learning

## Executive Summary
This project develops a standalone **SubRep** implementation that transforms skill discovery into a **certificate-driven, auditable process**. SubRep certifies skills via two mathematical tests (**CDS/PDS**) that guarantee composition safety across motive shifts, preventing negative transfer before skills enter the library.

This project validates the core mechanism in **MO-LunarLander**, storing certified skills as native **MeTTa Atoms** for future Hyperon integration.

## Objectives & Key Results (OKRs)
Aligned with Approved Quarter Plan:

| Objective | Goal | Key Results |
| :--- | :--- | :--- |
| **1. Neural Skill Generator** | Generate skill summaries from experience | • 2-head MLP (Payoff + Motives)<br>• MDN Interface Defined<br>• TD Error Computation |
| **2. Core Certification** | Implement CDS/PDS admission tests | • CDS Test (Universal Benefit)<br>• PDS-ε Test (Acceptable Trade-off)<br>• MO-LunarLander Integration |
| **3. MeTTa Storage** | Store certificates as native Atoms | • Certificate Schema Defined<br>• PyMeTTa Bridge (`hyperon`)<br>• Zero-Shot Reuse Demo |
| **4. Minimal Validation** | Demonstrate core mechanism works | • Certified Skills Pass Tests<br>• Uncertified Skills Rejected<br>• Admission Rates Documented |

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

#On Linux / macOS:
source .venv/bin/activate


#On Windows:
.venv\Scripts\activate


# Install dependencies
pip install -r requirements.txt

```

### 3. Validation
```bash

#Run all tests:
python -m pytest -v

#Run a specific test file:
python -m pytest tests/test_certification_gates.py -v

# Run Full Pipeline (Phase 3+)
python -m demo.run_full_pipeline
```

### 4. Running the Demo Pipeline

> [!NOTE]
> `models/generator.pt` is gitignored. You must train the generator
> before running the demo.

**Step 1 — Collect environment data:**
```bash
python -m data_collector.collect
```

**Step 2 — Train the Skill Generator:**
```bash
python -m generator.train_generator
```

**Step 3 — Run the end-to-end demo:**
```bash
python -m demo.run_full_pipeline
```

**Step 4 — Open the demo app:**
```bash
streamlit run demo/streamlit_subrep_demo.py
```

The Streamlit app is the demo interface. It can run the real
pipeline from the sidebar, then presents the full SubRep story in one place:
skill execution, improvement calculation, CDS/PDS admission, certificate
storage, trained-MDN selection, zero-shot reuse, and the final audit tables.

### 5. PPO Pilot Reproducibility
```bash
# Regenerate the committed PPO pilot checkpoint:
python -m pilot.train_pilot --seed 7 --output models/pilot_ppo.pt

# Validate the checkpoint without retraining:
python -m pytest tests/test_pilot_performance.py -v
```

### 6. Admission Report Output
After running the demo pipeline, admission statistics are automatically generated:

```bash
# Run the pipeline (report is generated automatically)
python -m demo.run_full_pipeline

# View the JSON report
cat demo/artifacts/admission_report.json

# View the Markdown report
cat demo/artifacts/admission_report.md
```

**Report location**: `demo/artifacts/admission_report.json` and `demo/artifacts/admission_report.md`

**Report contents**:
- Total attempted, admitted, and rejected skills
- Admission/rejection rates
- CDS and PDS pass counts
- Failure reasons for rejected skills
- Example admitted and rejected skills with full metrics

### 7. MDN Stub Configuration
The pipeline uses a **deterministic MDN stub** by default for testing and demonstration. This allows the MDN selection pipeline to run without a trained checkpoint.

**Default behavior**:
- Pipeline looks for `models/mdn_policy_best.pth`
- If not found → falls back to `StubMDN` with fixed outputs
- Stub returns `alpha=[2.0, 2.0]` and `support_values=[1.0, 1.0]`

**To use a trained MDN checkpoint**:
1. Train your MDN model (see `generator/README.md` section "MDN Candidate-Set Training and Evaluation")
2. Save the checkpoint to `models/mdn_policy_best.pth`
3. Run the pipeline — it will automatically use the trained model

**Code location**: `utils/mdn_stub.py` contains the `load_mdn_or_stub()` helper that handles this swap transparently.

**No code changes required** — the pipeline works identically with stub or trained MDN.

**MDN Metadata**: The admission report (`demo/artifacts/admission_report.json`) includes metadata about which MDN was used:
- `mdn_source`: "trained_checkpoint" or "stub"
- `checkpoint_path`: Path to checkpoint file
- `alpha_values`: MDN alpha output (mixture weights)
- `derived_weights`: Mean weights derived from alpha
- `support_values`: MDN support output (support geometry)
- `support_geometry_feasible`: Whether support values satisfy constraints

### 8. MDN Training and Integration

We have successfully trained the MDN and integrated it into the pipeline. Here's the complete workflow:

#### **Step 1: Collect Training Data**
```bash
# Collect 3,000 contexts (21,000 candidate outcomes)
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 42 --prefix seed42
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 43 --prefix seed43
python -m data_collector.collect_candidate_sets --contexts 1000 --save-dir data/mdn_candidate_sets --seed 44 --prefix seed44
```

#### **Step 2: Train the MDN**
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

**Output:**
- `models/mdn_policy_best.pth` — Trained MDN policy checkpoint
- `models/mdn_auxiliary_best.pth` — Auxiliary heads checkpoint

#### **Step 3: Run Pipeline with Trained MDN**
```bash
python -m demo.run_full_pipeline
```

The pipeline will automatically:
- Load the trained MDN from `models/mdn_policy_best.pth`
- Use it for Phase 5 (MDN-Based Skill Selection Demo)
- Include MDN metadata in the admission report

#### **Results: Stub vs Trained MDN**

| Aspect | Stub MDN | Trained MDN |
|--------|----------|-------------|
| **Alpha values** | Fixed `[2.00, 2.00]` | Context-aware (varies by observation) |
| **Scores** | Fixed `306.39` | Varies (e.g., `308.83`, `314.21`, `307.28`) |
| **Selection behavior** | Same skill for all observations | Different skills based on context |
| **Use case** | Testing/CI/CD | Production |

**Example output with trained MDN:**
```
[MDN Loader] Successfully loaded checkpoint from: models/mdn_policy_best.pth
[MDN Loader] Inferred dimensions: input=8, objectives=2, skills=100000
[Report] MDN source: trained_checkpoint

  Obs 1: Selected skill 'skill_005' (score=308.8314, alpha=[0.63, 0.56])
  Obs 2: Selected skill 'skill_005' (score=314.2143, alpha=[0.71, 0.50])
  Obs 3: Selected skill 'skill_005' (score=307.2794, alpha=[0.73, 0.71])
```

**Note:** The alpha values are now context-aware and vary by observation, demonstrating that the trained MDN is being used for selection.
## Project Structure
| Folder | Description| 
| :--- | :---|
| `env/` | MO-LunarLander wrapper & vector reward handling| 
| `generator/` | 2-head MLP skill generator (PyTorch)| 
| `pilot/` | PPO pilot policy, training entry point, and checkpoint utilities|
| `certification/` | CDS/PDS admission gate logic|
| `metta/` | PyMeTTa bridge & certificate schema| 
| `utils/` | TD error computation, logging, helpers| 
| `tests/` | Validation scripts for each component| 



## Technical Specifications

### Environment
- **Platform:** `mo-gymnasium` (MO-LunarLander-v3)
- **Observation Space:** `(8,)` – State vector (position, velocity, fuel, etc.)
- **Reward Space:** `(2,)` – `[Safety_Reward, Fuel_Reward]`

### Neural Generator
- **Architecture:** 2-head MLP (Payoff + Motives)
- **Input:** State vector `(8,)`
- **Output:** 
  - `payoff`: Scalar `(1,)`
  - `motives`: Vector `(2,)`

### Certification
- **CDS:** Cone-Dominant Subtask (Universal Benefit)
- **PDS-ε:** Pareto-Dominant Subtask (Acceptable Trade-off)
- **Cones:** Full-simplex (Phase 3) -> MDN-learned (Phase 4+)

### MeTTa Integration
- **Package:** `hyperon` (Python bindings)
- **Operations:** `add_atom`, `match`, `space`

## Documentation
- [Quarter Plan](https://docs.google.com/document/d/111xeC5gMT-JcX04iyH3KH-oE2RZIHx3kvvbZmzUaxeE/edit?usp=sharing)
- [SubRep Paper](https://chat.singularitynet.io/chat/pl/hhhg89sykbn7zpuhgfr973jear)
- [Hyperon Whitepaper](https://drive.google.com/file/d/1f2xDbHGoqaBJpNfWdpoi3QOHnAWOFTSD/view)
- [Metta Integration Guide](https://metta-lang.dev/docs/learn/tutorials/python_use/metta_python_basics.html)

## Roadmap (Q2+)
- **MDN Training:** Full Motive Decomposition Network implementation.
- **MetaMo Integration:** Dynamic weight management & risk budgets.
- **Cross-Paradigm Skills:** Logic macros & evolutionary programs.
- **Benchmarking:** Hypervolume efficiency vs. standard MORL baselines.

