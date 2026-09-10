# Three measured Safety-Gymnasium objectives

Enable with `SafeRLGymnasiumEnv(include_control_efficiency=True)`.
The default remains the original two-objective mode. The step reward and
`info['raw_objective_values']` preserve raw values in this order:

1. Safety = negative environment cost.
2. Task = environment reward.
3. ControlEfficiency = negative mean squared normalized applied action.

Each action coordinate is divided by its positive bound for positive actions,
or the magnitude of its negative bound for negative actions. Thus zero action
has zero effort, half-strength action on every coordinate has effort 0.25,
and full-strength action has effort 1. Bounds must be finite and straddle zero.
Invalid/out-of-range actions are rejected before stepping; they are not silently
clipped. Control effort is a dimensionless action-magnitude proxy, not electrical
energy or a guarantee of smooth movement. `info` retains task reward, safety cost,
control effort, objective names and measurement version.

## Frozen scaling

Raw step measurements are never normalized in the wrapper. Accumulate them with
consistent horizons/discounting, compute improvements against a matching baseline,
then fit `FrozenObjectiveScales.fit_development(delta_returns, source=...)` using
ONLY development data shaped (N,3). The estimator uses the mean absolute delta
return per objective. A dimension with magnitude <=1e-12 uses divisor 1.
Save the result with `.save(path)` and reload the same file for evaluation. No
online fitting occurs during `.score()`. The caller must enforce the dataset
split and record the development seeds/source; a source label alone does not
prevent leakage.

The explicit score is:

    delta_payoff / task_scale + sum_i weights[i] * delta_motives[i] / scale[i]

The current collector defines separate payoff as task reward. Task therefore
contributes both as payoff and as a weighted motive (effective coefficient
1 + task_weight when delta_payoff equals the task delta). This convention is
preserved, not silently replaced by pure weighted objectives.

Use the same normalized payoff/motive values for certification and scoring;
epsilon must be in those normalized score units. Do not attach normalized scores
to certificates computed under a different scaling convention.

## Collection and reporting

The collector, executor, certificate pipeline, audit reports, and comparison
loaders preserve either two or three objectives. NPZ schema version 2 records
objective names, definitions, scale divisors, scaling source, raw-value status,
and discount factor. The loader applies the frozen scales to motives and payoff
before certification and scoring; raw returns remain available. Legacy two-value
files remain supported, but cannot be requested as three-value data. Missing
control effort is never padded or inferred.

Collect fresh three-objective smoke data in your Safety-Gymnasium environment:

```bash
python -m data_collector.collect_safety_gymnasium_rollouts --objectives 3 --contexts 2 --max-steps 20 --save-dir data/safety_3d_smoke
python -m demo.run_safety_gymnasium_pipeline --objectives 3 --rollout-dir data/safety_3d_smoke --cert-file data/safety_3d_smoke_certificates.metta --library-file data/safety_3d_smoke_library.json --report-json demo/artifacts/safety_3d_smoke.json --report-md demo/artifacts/safety_3d_smoke.md
```

For calibrated evaluation, add `--scales-file path/to/frozen_scales.json` to the
collector. Without it, divisors are explicitly 1 and labelled uncalibrated;
no fitted development scales are claimed. Use a separate output folder per
measurement/scaling configuration. Omit `--objectives 3` to collect in the
original two-objective mode.

The ablation Python utility accepts three-value `task_weight` and `safety_weight`;
the Pareto utility accepts a three-value `selection_weight`. Its plot is still
a task/safety projection; JSON points retain all raw objective returns. The reuse
curve accepts `control_weight`, reserving that fraction while shifting the
remaining weight between safety and task. These are retrospective comparisons
of collected evidence, not independent evaluation executions.

## Tests

```bash
.venv/bin/python -m pytest tests/test_safety_three_pipeline.py tests/test_safety_objectives.py tests/test_safety_gymnasium_adapter.py -q
```

These tests use a controlled fake simulator to verify measurements and scaling.
They do not constitute a real Safety-Gymnasium benchmark run.
