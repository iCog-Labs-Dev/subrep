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
        assert 0.0 <= result["reuse_success_rate"] <= 1.0
        assert 0.0 <= result["negative_transfer_rate"] <= 1.0
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
