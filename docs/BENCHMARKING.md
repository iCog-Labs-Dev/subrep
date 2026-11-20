# SubRep benchmarking

This branch contains reproducible benchmarking code, tests, and instructions.
It extends `main` without external controller integrations. Generated evidence,
model weights, plots, and reports belong in ignored local directories. No
performance result is asserted by the presence of a runner or passing unit tests.

## Choose the appropriate experiment

| Experiment | Entry point | Scope |
|---|---|---|
| Synthetic objective-count comparisons | `demo.run_multi_objective_benchmark` | Gate and selection mechanics; no simulator outcomes |
| Three-objective held-out batches | `demo.run_safety_full_benchmark` | Separate evidence and execution layouts |
| Three-objective reuse comparisons | `demo.run_safety_reuse_benchmark` | Fixed policy library, priorities between episodes |
| Exact-context reuse, three or four objectives | `demo.run_contextual_safety_benchmark` | Certificates tied to reset seed and observation |
| Four-objective policy/data workflow | [Four-objective guide](SAFETY_FOUR_OBJECTIVES.md) | Training, collection, and exact-context evaluation |

See [metric definitions](BENCHMARK_METRICS.md), [three-objective measurements](SAFETY_THREE_OBJECTIVES.md),
[held-out batch evaluation](SAFETY_FULL_BENCHMARK.md), and
[reuse comparisons](SAFETY_REUSE_BENCHMARK.md) for their respective protocols.
The three-objective analysis and plotting utilities are not automatically
four-objective analyses. Check their supported dimensions before using them.

## Exact-context and MDN support experiments

The contextual runner tries full-simplex CDS, full-simplex PDS, then CDS/PDS on
the predicted support region if an MDN is supplied. Without an MDN, support-based
checks are reported as unavailable. Use `--mdn-checkpoint` and `--mdn-contract`
only with a checkpoint trained for the same observation and objective schema.
A three-objective checkpoint is not interchangeable with a four-objective one.

The candidate-set adapter supports measured same-context baselines using
`--baseline-candidate-id zero_action`. Keep task payoff scaling consistent with
the task motive scale. The optional margin support trainer is documented in the
main README; it is not the default candidate-set training objective. Predicted
regions still require exact gate checks. Acceptance is not an absolute safety
budget guarantee.

For a three-objective support comparison, supply your local input paths:

```bash
conda activate subrep-safety
export DATA_DIR=data/my_support_training
export SCALES_FILE=data/my_development/scales.json
export BASE_CHECKPOINT=models/my_mdn_auxiliary.pth
export PPO_CHECKPOINT=models/my_ppo.pt
export LAGRANGIAN_CHECKPOINT=models/my_lagrangian.pt
export CONTEXT_SEED=510000
bash demo/run_safety_margin_comparison.sh
```

`TRAINER_PYTHON` and `SAFETY_PYTHON` can select separate installed Python
interpreters. Both default to the active `python`. The script derives payoff
scaling from the supplied scales file; it contains no machine-specific paths
or experiment-specific fitted constants. It is a three-objective runner.

`generator.export_contextual_support` exports certified selection priorities
from non-MDN contextual runs for the existing support-target trainer. These are
observed priority targets, not independently proven safety-region boundaries.
Reserve disjoint context groups for training, validation, and final evaluation.

## Evidence and reporting

- Freeze policies, scales, thresholds, and evaluation seeds before final testing.
- Keep training/development contexts separate from final evaluation contexts.
- Report actual goal success, raw safety cost, effort, and smoothness alongside
  admission, eligibility, selected skill, and fallback frequency.
- Distinguish exact-start re-execution from transfer to unseen layouts. Repeating
  a deterministic policy from the same start is not independent evidence.
- Compare methods on matched contexts. Report uncertainty and multiple training
  seeds before making general performance claims.
- State that the current score adds task payoff to weighted motives: task also
  appears in the motive vector. Epsilon uses the resulting normalized units.
- Do not interpret a weighted trade-off certificate as proof that raw cost is
  below an absolute budget. Check mandatory requirements separately.

For existing three-objective evidence, `utils.analyze_contextual_baselines`,
`utils.diagnose_contextual_goals`, `utils.plot_contextual_pareto`, and
`utils.analyze_priority_reuse` provide retrospective reports. Priority reuse is
an offline query over frozen evidence, not switching policies during a flight.

## Repository hygiene

Keep generated files under `data/`, `models/`, `plots/`, `runs/`, `outputs/`, or
`demo/artifacts/`. These directories are ignored except existing documentation
and the upstream pilot fixture. Do not force-add generated results. Test fixtures
should be small, synthetic, and located under `tests/`.

Before committing, inspect `git diff --cached --stat` and
`git diff --cached --name-only`. Stage source, tests, and documentation explicitly.
The presence of local integration checkouts or virtual environments does not
make them part of the benchmarking change.

## Continuing experiments from an existing worktree

A clean source worktree does not need a copy of old result directories. Use
absolute checkpoint and scale paths from your original workspace when invoking
collectors or analysis tools. Write new outputs under the clean worktree's
ignored `data/` or `models/` directories. Checkpoints record their architecture;
do not rename or convert them to change actor or objective dimensions.

Keep the original worktree until its local data is backed up. Pushing this branch
does not publish those files or remove them from the original branch's history.
Create the pull request against `main`; do not merge the old benchmarking branch
into this clean branch, because that would reintroduce the unwanted history.
