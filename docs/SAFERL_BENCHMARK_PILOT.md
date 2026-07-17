# SafeRL Benchmark Pilot

This document summarizes the first Safety-Gymnasium pilot for SubRep. The goal
is to test the same certificate-driven mechanism outside MO-LunarLander:
collect candidate rollouts, compute payoff and safety-motive improvements,
certify with CDS/PDS, store admitted certificates, and compare reuse against
simple baselines.

## Environment

The first benchmark target is:

```text
SafetyPointGoal1-v0
```

This environment is a good first SafeRL target because it exposes both task
reward and safety cost while remaining lighter than robot agents such as Car,
Ant, or Doggo.

The Safety-Gymnasium wrapper maps benchmark outputs into SubRep's 2-objective
format:

```text
Safety = -cost
Task   = reward
```

This keeps the SubRep convention that larger motive values are better.

## Reproduction Commands

Collect rollouts in the Python 3.10 Safety-Gymnasium environment:

```bash
conda activate subrep-safety

python -m data_collector.collect_safety_gymnasium_rollouts \
  --env-id SafetyPointGoal1-v0 \
  --contexts 25 \
  --max-steps 200 \
  --save-dir data/safety_gymnasium_rollouts \
  --seed 42
```

Certify and evaluate the collected rollouts in the main SubRep environment:

```bash
conda deactivate
source .venv/bin/activate

.venv/bin/python -m demo.run_safety_gymnasium_pipeline \
  --rollout-dir data/safety_gymnasium_rollouts \
  --pds-epsilon 1.0
```

The generated reports are:

```text
demo/artifacts/safety_gymnasium_admission_report.json
demo/artifacts/safety_gymnasium_admission_report.md
```

These are generated artifacts and are ignored by git.

## Current Pilot Result

The first 25-context local run produced:

| Metric | Value |
|---|---:|
| Contexts processed | 25 |
| Candidate outcomes loaded | 175 |
| Candidate outcomes certified | 150 |
| Admitted | 104 |
| Rejected | 46 |
| Admission rate | 69.3% |
| CDS admissions | 58 |
| PDS admissions | 46 |
| Certificate store count | 104 |
| SkillLibrary size | 104 |

Safety-cost behavior:

| Metric | Value |
|---|---:|
| Mean baseline safety cost | 0.0000 |
| Mean admitted safety cost | 0.0000 |
| Mean rejected safety cost | 2.9481 |
| Candidates with higher cost than baseline | 20 |
| Admitted higher-cost candidates | 0 |
| Rejected higher-cost candidates | 20 |

## Baseline Comparison

Scores use the same SubRep reuse scalarization:

```text
score = delta_r + weight dot delta_n
```

The current candidate set includes simple continuous-control policies, so PPO is
reported as unavailable rather than estimated indirectly.

| Query | SubRep certified | Zero action | Random candidate | Random certified | PPO | Lift vs random |
|---|---:|---:|---:|---:|---:|---:|
| Task-focused | 0.2114 | 0.0000 | -0.7887 | -0.0028 | n/a | 1.0001 |
| Safety-focused | 0.1224 | 0.0000 | -1.2179 | -0.0016 | n/a | 1.3404 |

Interpretation:

- SubRep admits both CDS and bounded PDS skills while rejecting unsafe candidates.
- Rejected candidates have much higher safety cost on average.
- The certified SubRep choice outperforms zero-action and random baselines in
  this first pilot.
- A trained PPO Safety-Gymnasium policy is still needed for a true PPO baseline.

## Demo Story

The SafeRL demo story is:

1. Collect candidate rollouts in Safety-Gymnasium.
2. Use `zero_action` as the same-context baseline.
3. Compute payoff improvement and safety/task motive improvement.
4. Certify candidates with CDS/PDS.
5. Store only admitted certificates in MeTTa-backed `CertificateStore` and
   `SkillLibrary`.
6. Generate admission, safety-cost, and baseline comparison metrics.
7. Query the frozen certified library under changed motive weights without
   retraining.

## Next Step

The next benchmark step is to add a trained PPO candidate for
`SafetyPointGoal1-v0`, collect the same candidate-set format with that policy
included, and rerun the same report. No certification logic needs to change.
