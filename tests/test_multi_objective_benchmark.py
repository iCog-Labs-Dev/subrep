from __future__ import annotations

import json

import numpy as np

from certification.certificate_schema import Certificate
from library.skill_library import SkillLibrary
from library.skill_metadata import MDN_WX
from utils.multi_objective_benchmark import (
    render_multi_objective_benchmark_markdown,
    run_multi_objective_benchmark,
)
from utils.support_geometry import make_basis_query_directions


def test_multi_objective_benchmark_runs_for_three_and_four_objectives(tmp_path):
    output = tmp_path / "multi_objective_benchmark.json"

    summary = run_multi_objective_benchmark(
        objective_counts=(3, 4),
        candidates_per_objective_count=16,
        seeds=(5, 7),
        output_json=output,
    )

    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["objective_counts"] == [3, 4]
    assert len(summary["results"]) == 2
    for result in summary["results"]:
        assert result["candidate_skills_evaluated"] == 32
        assert result["admitted"] > 0
        assert result["rejected"] > 0
        assert result["cds_admissions"] + result["pds_admissions"] == result["admitted"]
        assert 0.0 <= result["reuse_eligibility_rate"] <= 1.0
        assert 0.0 <= result["abstention_rate"] <= 1.0
        assert result["query_time_ms"] >= 0.0
        assert result["infeasible_support_events"] == 0


def test_multi_objective_benchmark_writes_markdown_report(tmp_path):
    output_json = tmp_path / "multi_objective_benchmark.json"
    output_md = tmp_path / "multi_objective_benchmark.md"

    summary = run_multi_objective_benchmark(
        objective_counts=(3,),
        candidates_per_objective_count=12,
        seeds=(5,),
        output_json=output_json,
        output_markdown=output_md,
    )

    markdown = output_md.read_text(encoding="utf-8")
    assert "# SubRep Multi-Objective Benchmark" in markdown
    assert "| M | Candidates | Admitted | Rejected | CDS | PDS |" in markdown
    assert "M=3" in markdown
    assert render_multi_objective_benchmark_markdown(summary) == markdown


def test_skill_library_accepts_and_queries_mdn_wx_for_five_objectives():
    num_objectives = 5
    support_values = (0.45, 0.45, 0.45, 0.45, 0.45)
    directions = tuple(tuple(float(v) for v in row) for row in make_basis_query_directions(num_objectives))
    cert = Certificate(
        skill_id="m5_navigation_tradeoff",
        gate_type="PDS",
        delta_r=0.2,
        delta_n=(-0.1, 0.3, 0.2, 0.4, 0.1),
        admission_margin=0.15,
        epsilon=0.05,
        timestamp="2026-09-01T00:00:00+00:00",
        seed=42,
        gamma=0.99,
        baseline_id="synthetic_idle_v1",
        environment="Synthetic-MO-5D-v0",
        episode_length=50,
        version="0.1.0",
        weight_region_type=MDN_WX,
        certification_context=(0.0, 1.0),
        mdn_alpha=(1.0, 1.0, 1.0, 1.0, 1.0),
        wx_support_directions=directions,
        wx_support_values=support_values,
    )
    library = SkillLibrary()

    assert library.add_skill(
        cert.skill_id,
        cert,
        lambda obs: 0,
        weight_region_type=MDN_WX,
        certification_context=cert.certification_context,
        mdn_alpha=cert.mdn_alpha,
        wx_support_directions=cert.wx_support_directions,
        wx_support_values=cert.wx_support_values,
    )

    admissible = library.query_admissible(
        current_weight=np.full(num_objectives, 1.0 / num_objectives),
        support_directions=np.eye(num_objectives),
        support_values=np.asarray(support_values),
    )
    assert [entry.skill_id for entry in admissible] == ["m5_navigation_tradeoff"]


def test_corrected_metrics_and_complete_audit():
    result = run_multi_objective_benchmark(objective_counts=(3,), seeds=(5,), candidates_per_objective_count=16)['results'][0]
    assert len(result['admission_audit']['audit_entries']) == 16
    assert result['admission_audit']['admitted'] == result['admitted']
    assert result['execution_performance'] is None
    assert result['selection_validity_rate'] == 1.0
    assert result['selected_threshold_violation_count'] == 0
    assert 'reuse_success_rate' not in result
    for query in result['reuse'].values():
        assert np.all(np.asarray(query['weights']) <= query['support_values'])
    for entry in result['admission_audit']['audit_entries']:
        assert len(entry['gate_evaluations']) == 2
        if entry['gate_type'] != 'CDS':
            pds = entry['gate_evaluations'][1]
            assert np.isclose(entry['margin'], pds['lhs'] - pds['rhs'])


def test_empty_library_abstains_without_claiming_execution_success():
    result = run_multi_objective_benchmark(objective_counts=(3,), seeds=(5,), candidates_per_objective_count=0)['results'][0]
    assert result['abstention_rate'] == 1.0
    assert result['selection_validity_rate'] is None
    assert result['reuse_eligibility_rate'] == 0.0
    assert result['execution_performance'] is None
    assert all(q['selection_valid'] is None for q in result['reuse'].values())


def test_out_of_region_excludes_contextual_but_preserves_global():
    from dataclasses import replace
    cert = Certificate(
        skill_id='global', gate_type='CDS', delta_r=1., delta_n=(0.1, 0.1, 0.1),
        admission_margin=1.1, epsilon=0., timestamp='2026-09-10T00:00:00',
        seed=1, gamma=0.99, baseline_id='idle', environment='synthetic',
        episode_length=1, version='1')
    lib = SkillLibrary()
    assert lib.add_skill('global', cert, None)
    directions = tuple(map(tuple, np.eye(3)))
    wx = replace(cert, skill_id='contextual', weight_region_type='MDN_WX',
                 certification_context=(0.,), mdn_alpha=(1.,1.,1.),
                 wx_support_directions=directions, wx_support_values=(0.75,)*3)
    assert lib.add_skill('contextual', wx, None, weight_region_type='MDN_WX',
                         certification_context=wx.certification_context, mdn_alpha=wx.mdn_alpha,
                         wx_support_directions=directions, wx_support_values=wx.wx_support_values)
    assert len(lib.query_admissible([0.7,0.15,0.15], np.eye(3), np.full(3,0.75))) == 2
    assert [e.skill_id for e in lib.query_admissible([0.9,0.05,0.05], np.eye(3), np.full(3,0.75))] == ['global']
    assert lib.query_admissible([0.5,0.5]) == []


def test_independent_seeds_and_paired_method_expectations():
    result = run_multi_objective_benchmark(objective_counts=(3,), seeds=(5,7), candidates_per_objective_count=12)['results'][0]
    assert len(result['seed_results']) == 2
    for run in result['seed_results']:
        assert run['candidate_skills_evaluated'] == 12
        for entry in run['admission_audit']['audit_entries']:
            assert entry['seed'] == run['seed']
        rows = run['comparisons']
        assert {'balanced','gradual_shift','abrupt_shift','empty_admissible'} <= {r['scenario'] for r in rows}
        for row in rows:
            assert np.isclose(sum(row['weights']),1)
            assert max(row['weights']) <= 0.75
            methods = row['methods']
            assert len(methods) == 5
            for value in methods.values():
                assert np.isclose(value['score'], value['delta_r'] + np.dot(row['weights'], value['delta_n']))
            if row['admissible_count']:
                assert methods['subrep']['score'] >= methods['random_admissible']['score'] - 1e-9
                assert methods['unrestricted_best']['score'] >= methods['subrep']['score'] - 1e-9
            if row['scenario'] == 'empty_admissible':
                assert methods['subrep']['abstained']
                assert methods['random_admissible']['baseline_fallback']
                assert methods['subrep']['score'] == 0
                assert methods['unrestricted_best']['score'] < 0
        separate = run_multi_objective_benchmark(objective_counts=(3,), seeds=(run['seed'],), candidates_per_objective_count=12)['results'][0]
        assert separate['seed_results'][0]['comparisons'] == rows
    for idx,row in enumerate(result['comparisons']):
        for method,value in row['methods'].items():
            assert np.isclose(value['mean_score'], np.mean([r['comparisons'][idx]['methods'][method]['score'] for r in result['seed_results']]))


def test_default_dimensions_and_five_seeds():
    summary = run_multi_objective_benchmark(candidates_per_objective_count=4)
    assert summary['objective_counts'] == [3,4,5,6,8,10]
    assert len(summary['seeds']) == 5
    for result in summary['results']:
        assert len(result['seed_results']) == 5
        assert result['candidate_skills_evaluated'] == 20
