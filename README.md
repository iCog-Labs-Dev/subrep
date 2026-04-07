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
git clone https://github.com/your-username/subrep-q1.git
cd subrep-q1

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 3. Validation
```bash
# Verify Environment Setup
python tests/test_env.py

# Verify Generator Output
python tests/test_generator.py

# Run Full Pipeline (Phase 5+)
python main.py
```

## Project Structure
| Folder | Description| 
| :--- | :---|
| `env/` | MO-LunarLander wrapper & vector reward handling| 
| `generator/` | 2-head MLP skill generator (PyTorch)| 
| `certification/` | CDS/PDS admission gate logic|
| `metta/` | PyMeTTa bridge & certificate schema| 
| `utils/` | TD error computation, logging, helpers| 
| `tests/` | Validation scripts for each component| 
| `docs/` | Quarter Plan, Papers, Meeting Notes|


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

## Collaboration
- **Issues:** Track tasks in GitHub Issues (Labels: `env`, `generator`, `certification`, `metta`).
- **PRs:** All changes require review before merging to `main`.
- **Documentation:** Update relevant `README.md` files when changing interfaces.

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


