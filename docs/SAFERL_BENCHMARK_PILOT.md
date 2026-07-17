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

Train a lightweight PPO baseline and collect rollouts in the Python 3.10
Safety-Gymnasium environment. Use `--no-capture-output` so conda streams
progress logs instead of printing everything only after the command exits:

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

The current 25-context local run with PPO included produced:

| Metric | Value |
|---|---:|
| Contexts processed | 25 |
| Candidate outcomes loaded | 200 |
| Candidate outcomes certified | 175 |
| Admitted | 125 |
| Rejected | 50 |
| Admission rate | 71.4% |
| CDS admissions | 70 |
| PDS admissions | 55 |
| Certificate store count | 125 |
| SkillLibrary size | 125 |

Safety-cost behavior:

| Metric | Value |
|---|---:|
| Mean baseline safety cost | 0.0000 |
| Mean admitted safety cost | 0.0000 |
| Mean rejected safety cost | 3.4159 |
| Candidates with higher cost than baseline | 23 |
| Admitted higher-cost candidates | 0 |
| Rejected higher-cost candidates | 23 |

## Baseline Comparison

Scores use the same SubRep reuse scalarization:

```text
score = delta_r + weight dot delta_n
```

| Query | SubRep certified | Zero action | Random candidate | Random certified | PPO | Lift vs random |
|---|---:|---:|---:|---:|---:|---:|
| Task-focused | 0.2660 | 0.0000 | -0.6941 | 0.0101 | -0.1263 | 0.9600 |
| Safety-focused | 0.1540 | 0.0000 | -1.2237 | 0.0059 | -1.2582 | 1.3777 |

Interpretation:

- SubRep admits both CDS and bounded PDS skills while rejecting unsafe candidates.
- Rejected candidates have much higher safety cost on average.
- The certified SubRep choice outperforms zero-action, random candidate, random
  certified, and the lightweight PPO baseline in this first pilot.
- The PPO checkpoint used here is intentionally lightweight so the benchmark can
  be reproduced quickly. A longer PPO run across more seeds would make the
  baseline stronger.

## Demo Story

The SafeRL demo story is:

1. Train a lightweight PPO baseline for Safety-Gymnasium.
2. Collect candidate rollouts in Safety-Gymnasium, including PPO when the
   checkpoint is available.
3. Use `zero_action` as the same-context baseline.
4. Compute payoff improvement and safety/task motive improvement.
5. Certify candidates with CDS/PDS.
6. Store only admitted certificates in MeTTa-backed `CertificateStore` and
   `SkillLibrary`.
7. Generate admission, safety-cost, and baseline comparison metrics.
8. Query the frozen certified library under changed motive weights without
   retraining.

## Next Step

The next benchmark step is to scale this beyond the first pilot by running more
seeds/contexts and training a stronger PPO baseline with more updates. No
certification logic needs to change.
