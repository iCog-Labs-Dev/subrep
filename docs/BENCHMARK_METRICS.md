# Synthetic benchmark metrics (schema version 2)

Run from the SubRep directory:

```bash
.venv/bin/python -m demo.run_multi_objective_benchmark
.venv/bin/python -m pytest tests/test_multi_objective_benchmark.py tests/test_skill_library.py tests/test_admission_report.py tests/test_support_geometry.py -q
```

JSON and Markdown are written to `demo/artifacts/multi_objective_benchmark.*`.
These generated files may be ignored by Git; rerun the command to reproduce them.

- `admission_rate` / `rejection_rate`: admitted/rejected candidates divided by attempted candidates.
- `reuse_eligibility_rate`: total admissible entries across queries divided by stored skills times query count. Zero for an empty library.
- `selection_validity_rate`: fraction of non-abstaining selections whose score is at least that skill's threshold (`-epsilon`, zero for CDS), allowing 1e-9 numerical tolerance. Null when there are no selections.
- `selected_threshold_violation_count`: selected skills below their permitted threshold. A negative score within PDS epsilon is not a violation.
- `abstention_rate`: fraction of queries with no selected skill. No baseline action is executed by this benchmark.
- `execution_performance`: null, because synthetic payoff/motive evidence is generated directly; the policy placeholders are not executed.
- `query_time_ms`: total library filtering time across the three queries, excluding ranking/report construction. This is not a repeated latency benchmark.

The misleading version-1 keys `reuse_success_rate` and `negative_transfer_rate` are removed rather than silently assigned different meanings. Consumers must use the new keys. The old negative-transfer statistic examined all stored skills even outside their applicability region; it did not measure failures of selected actions.

Primary focused weights are now 0.70, within the runtime cap of 0.75. Regression tests separately exercise weights outside the checked region: contextual skills are excluded, while dimension-compatible global skills remain available. Objective-count mismatches are excluded for both types.

`admission_audit` reuses AdmissionReport and includes every attempted candidate, full motive vectors, policy/source, rejection details and gate comparisons. CDS is evaluated over the full simplex; fallback PDS is evaluated over hand-constructed contextual caps. Each inequality's expression identifies its region. These caps depend on generated candidate effects and are test fixtures, not learned MDN evidence or independent performance validation. Rejected entries retain the attempted contextual region and PDS margin.

This work does not connect the live MDN runtime to reporting, change the gate formulas, add independent rollouts, or benchmark MetaMo/Omega. Those remain separate stages.

## Multi-objective comparisons (schema version 3)

Default runs cover M=3,4,5,6,8,10 with seeds 11,23,37,51,67 and 48 candidates
per seed. Each seed builds its own library; no candidates cross seed boundaries.
`seed_results` retains each run's audits, queries and comparisons. Top-level
admission counts are summed, rates averaged across seeds, and `comparisons`
contains per-scenario means and population standard deviations across seeds.
The legacy `reuse` detail is an explicitly labelled first-seed example.
`query_time_ms` is summed over seeds and the three original queries, not all
comparison scenarios. It remains diagnostic, not a scalability benchmark.

Five methods share the same candidate pool and weight vector: best admissible,
uniform random candidate, uniform random admissible, idle, and unrestricted best.
Random methods report exact expected score and vector effects over their pool;
there is no sampled winning skill ID. Ties for deterministic selection use skill ID.
If the eligible pool is empty, the method abstains and receives baseline deltas
of zero for comparison. This is a scoring convention, not an executed fallback.

Scenarios include balanced weights, focus on every objective, a five-step gradual
shift, a three-step abrupt switch and return, and an explicit rejected-only pool.
These are priority trajectories over fixed synthetic evidence, not simulated
state transitions. For every step, JSON records weights and each method's scalar
payoff improvement, full objective-effect vector, combined score and abstention.
Markdown shows aggregate comparisons and the existing admission audits. Objective
labels are objective_0, objective_1, etc.; the values have no implied physical units.

Hand-designed support caps and perfect knowledge of generated candidate deltas
make this a mechanics/comparison experiment. It does not establish trained-MDN
quality, empirical policy performance, or superiority on independent rollouts.
The rejected-only case is a separate constructed scenario and does not inflate
the primary candidate-admission counts.
