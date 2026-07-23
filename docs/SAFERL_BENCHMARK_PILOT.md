# SubRep SafeRL Benchmark: Safety-Gymnasium PointGoal

This document summarizes the scaled SafeRL benchmark run for SubRep on
Safety-Gymnasium `SafetyPointGoal1-v0`. The goal was to test whether the same
certificate-driven SubRep mechanism used in MO-LunarLander can operate on an
external SafeRL environment with explicit task reward and safety cost signals.

## Summary

SubRep has now been evaluated on Safety-Gymnasium `SafetyPointGoal1-v0` across
three seeds with 100 contexts per seed. The environment provides task reward and
safety cost directly. SubRep maps these into its two-objective format as:

```text
Task   = reward
Safety = -cost
```

Candidate skills are executed from the same context, compared against a
same-context `zero_action` baseline, certified with CDS/PDS gates, and admitted
only if the certificate passes runtime validation. The resulting certified skill
set is compared against zero-action, random candidate, random-certified, and PPO
baselines.

## What Was Implemented

- Safety-Gymnasium adapter for SubRep's 2-objective interface.
- Candidate rollout collector for Safety-Gymnasium.
- PPO baseline candidate for `SafetyPointGoal1-v0`.
- Multi-seed rollout collection with 100 contexts per seed.
- CDS/PDS certification pipeline for SafeRL rollout outcomes.
- MeTTa-backed `CertificateStore` and runtime `SkillLibrary` integration.
- Admission report and benchmark comparison generation.
- Progress logging for PPO training and rollout collection.
- Runtime certificate-store indexing to avoid repeated Hyperon atom scans during
  larger SafeRL runs.

## Benchmark Setup

| Item | Value |
|---|---|
| Environment | `SafetyPointGoal1-v0` |
| Seeds | 42, 43, 44 |
| Contexts per seed | 100 |
| Total contexts | 300 |
| Max steps per rollout | 200 |
| Baseline candidate | `zero_action` |
| Candidate set | fixed controls, random policies, PPO |
| Certification | CDS and PDS |
| PDS epsilon | 1.0 |

Safety-Gymnasium defines the safety signal through environment costs. In
`SafetyPointGoal1-v0`, the agent is rewarded for reaching the goal and receives
cost when it violates environment constraints such as unsafe obstacle/hazard
regions. SubRep treats lower cost as better safety by using `Safety = -cost`.

## Reproduction Commands

Train a PPO baseline and collect rollouts in the Python 3.10 Safety-Gymnasium
environment. `--no-capture-output` makes conda stream progress logs live:

```bash
conda run --no-capture-output -n subrep-safety python -m pilot.train_safety_gymnasium_ppo \
  --env-id SafetyPointGoal1-v0 \
  --total-updates 50 \
  --rollout-steps 1024 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --max-episode-steps 200 \
  --eval-episodes 20 \
  --seed 42 \
  --output models/safety_ppo_point_goal_seed42_updates50.pt

conda run --no-capture-output -n subrep-safety python -m data_collector.collect_safety_gymnasium_rollouts \
  --env-id SafetyPointGoal1-v0 \
  --contexts 100 \
  --max-steps 200 \
  --save-dir data/safety_gymnasium_rollouts_seed42_ctx100 \
  --prefix safety_seed42_ctx100 \
  --seed 42 \
  --ppo-checkpoint models/safety_ppo_point_goal_seed42_updates50.pt

.venv/bin/python -m demo.run_safety_gymnasium_pipeline \
  --rollout-dir data/safety_gymnasium_rollouts_seed42_ctx100 \
  --pds-epsilon 1.0 \
  --report-json demo/artifacts/safety_gymnasium_seed42_ctx100_report.json \
  --report-md demo/artifacts/safety_gymnasium_seed42_ctx100_report.md
```

Repeat the same commands for seeds `43` and `44`, changing the seed, checkpoint
name, rollout directory, prefix, and report paths.

Generated rollout data, checkpoints, and reports are ignored by git and should
be regenerated locally or attached externally when needed.

## Aggregate Results

Across seeds 42, 43, and 44:

| Metric | Value |
|---|---:|
| Contexts processed | 300 |
| Candidate outcomes loaded | 2,400 |
| Candidate outcomes certified | 2,100 |
| Admitted | 1,465 |
| Rejected | 635 |
| Admission rate | 69.76% |
| CDS admissions | 708 |
| PDS admissions | 757 |

Per-seed admission results:

| Seed | Contexts | Certified | Admitted | Rejected | Admission Rate | CDS | PDS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 100 | 700 | 481 | 219 | 68.71% | 242 | 239 |
| 43 | 100 | 700 | 488 | 212 | 69.71% | 233 | 255 |
| 44 | 100 | 700 | 496 | 204 | 70.86% | 233 | 263 |

## Safety-Cost Results

| Metric | Aggregate |
|---|---:|
| Mean admitted safety cost | 0.0027 +/- 0.0022 |
| Mean rejected safety cost | 4.2082 +/- 0.6740 |
| Candidates with higher cost than baseline | 296 |
| Admitted higher-cost candidates | 5 |
| Rejected higher-cost candidates | 291 |

This is the main safety result. Rejected candidates had much higher safety cost
than admitted candidates. Of the 296 candidates with higher cost than the
same-context baseline, 291 were rejected. This shows that CDS/PDS certification
is filtering unsafe behavior rather than blindly admitting high-reward but
higher-cost candidates.

## Baseline Comparison

Scores use the SubRep reuse scalarization:

```text
score = delta_r + weight dot delta_n
```

The table reports mean +/- standard deviation across seeds.

| Query | SubRep Certified | Zero Action | Random Candidate | Random Certified | PPO | Lift vs Random | Lift vs PPO |
|---|---:|---:|---:|---:|---:|---:|---:|
| Task-focused | 0.2828 +/- 0.0472 | 0.0000 | -0.7874 +/- 0.0252 | -0.0153 +/- 0.0101 | -0.5184 +/- 0.1972 | 1.0702 +/- 0.0374 | 0.8012 +/- 0.1824 |
| Safety-focused | 0.1601 +/- 0.0252 | 0.0000 | -1.5331 +/- 0.2134 | -0.0110 +/- 0.0052 | -3.1174 +/- 1.6866 | 1.6932 +/- 0.2289 | 3.2775 +/- 1.6995 |

Win-rate summary:

| Query | Win Rate vs Zero Action | Win Rate vs Random Candidate |
|---|---:|---:|
| Task-focused | 94.3% +/- 0.6% | 100.0% +/- 0.0% |
| Safety-focused | 94.0% +/- 1.0% | 100.0% +/- 0.0% |

## Pareto Frontier Plot

A Pareto plot was added to visualize the task/safety tradeoff directly:

```text
X-axis: Safety cost / constraint violation, lower is better
Y-axis: Task return, higher is better
```

Generate the plot from the three 100-context rollout directories:

```bash
.venv/bin/python -m demo.plot_safety_gymnasium_pareto \
  --rollout-dir data/safety_gymnasium_rollouts_seed42_ctx100 \
  --rollout-dir data/safety_gymnasium_rollouts_seed43_ctx100 \
  --rollout-dir data/safety_gymnasium_rollouts_seed44_ctx100 \
  --pds-epsilon 1.0 \
  --output demo/artifacts/safety_gymnasium_pareto_frontier.png \
  --summary-json demo/artifacts/safety_gymnasium_pareto_frontier.json
```

Current Pareto summary from the 300-context rollout set:

| Method | Contexts | Mean Safety Cost | Mean Task Return | Zero-Cost Rate |
|---|---:|---:|---:|---:|
| SubRep certified | 300 | 0.0133 +/- 0.1137 | 0.1495 +/- 0.2100 | 98.3% |
| PPO | 300 | 3.3456 +/- 7.2963 | -0.0968 +/- 0.4016 | 73.7% |
| PPO-Lagrangian | not in current rollout set | n/a | n/a | n/a |

The current plot shows SubRep on the favorable part of the task/safety tradeoff:
it achieves higher mean task return than PPO while keeping average safety cost
near zero. PPO-Lagrangian support has been added, but the existing 300-context
artifact set was collected before that candidate was included. To produce the
full three-way Pareto plot, train a PPO-Lagrangian checkpoint and recollect
rollouts with both PPO candidates:

```bash
conda run --no-capture-output -n subrep-safety python -m pilot.train_safety_gymnasium_ppo \
  --env-id SafetyPointGoal1-v0 \
  --use-lagrangian \
  --cost-limit 1.0 \
  --lambda-lr 0.05 \
  --total-updates 50 \
  --rollout-steps 1024 \
  --update-epochs 4 \
  --minibatch-size 256 \
  --max-episode-steps 200 \
  --eval-episodes 20 \
  --seed 42 \
  --output models/safety_ppo_lagrangian_point_goal_seed42_updates50.pt

conda run --no-capture-output -n subrep-safety python -m data_collector.collect_safety_gymnasium_rollouts \
  --env-id SafetyPointGoal1-v0 \
  --contexts 100 \
  --max-steps 200 \
  --save-dir data/safety_gymnasium_rollouts_seed42_ctx100_lagrangian \
  --prefix safety_seed42_ctx100_lagrangian \
  --seed 42 \
  --ppo-checkpoint models/safety_ppo_point_goal_seed42_updates50.pt \
  --ppo-lagrangian-checkpoint models/safety_ppo_lagrangian_point_goal_seed42_updates50.pt
```

Repeat for seeds `43` and `44`, then rerun
`demo.plot_safety_gymnasium_pareto` on the new rollout directories.

## Interpretation

The scaled benchmark supports the same core conclusion as the smaller pilot:
SubRep can operate on an external SafeRL environment and use explicit
certificate checks to control skill admission.

Key observations:

- SubRep admitted both strict CDS skills and bounded-tradeoff PDS skills.
- Unsafe or low-quality candidates were rejected before entering the library.
- Rejected candidates had much higher safety cost than admitted candidates.
- Certified SubRep selection outperformed random candidate, random-certified,
  zero-action, and PPO baselines under both task-focused and safety-focused
  scoring.
- The Pareto plot shows SubRep achieving higher mean task return than PPO with
  much lower safety cost in the current rollout set.
- Certificate counts and SkillLibrary counts remained consistent, preserving the
  store/library safety invariant.

## Limitations

This is stronger than the first 25-context pilot, but it is still not a full
SafeRL benchmark suite.

Current limitations:

- Only one Safety-Gymnasium environment was evaluated.
- PPO is included, but stronger PPO training or external SafeRL baselines could
  make the comparison more rigorous.
- PPO-Lagrangian support is implemented, but the current 300-context artifact
  set must be regenerated with `--ppo-lagrangian-checkpoint` before reporting
  full three-way Pareto numbers.
- Results are reported over three seeds; more seeds would improve confidence.
- The current SubRep geometry remains the 2-objective prototype.

## Next Steps

The next benchmark step is to repeat this process on another Safety-Gymnasium
environment or a broader SafeRL benchmark set. Useful extensions include:

- running more than three seeds,
- training stronger PPO or constrained-RL baselines,
- adding confidence intervals in the report generator,
- testing a harder Safety-Gymnasium task,
- regenerating the 100-context seeds with PPO-Lagrangian included,
- comparing against standard SafeRL algorithms beyond project-native PPO.

The SubRep certification logic does not need to change for these extensions.
The main remaining work is benchmark scale and stronger baseline coverage.
