from __future__ import annotations

import json

from integration.omegaclaw.compare_runs import main as compare_main
from integration.omegaclaw.demo import main as demo_main
from integration.omegaclaw.demo import summarize
from integration.omegaclaw.scenarios import predefined_scenarios


def test_synthetic_demo_writes_comparable_report(tmp_path):
    report_path = tmp_path / "baseline.json"
    exit_code = demo_main(
        [
            "--backend",
            "synthetic",
            "--run-label",
            "subrep-baseline",
            "--audit-path",
            str(tmp_path / "audit.jsonl"),
            "--report-path",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["run_label"] == "subrep-baseline"
    assert report["backend"] == "synthetic"
    assert report["summary"]["scenario_count"] == len(predefined_scenarios())
    assert report["summary"]["valid_outcome_rate"] == 1.0
    assert report["summary"]["first_pass_valid_rate"] == 1.0
    assert report["summary"]["format"]["first_attempt_format_valid_rate"] == 1.0
    assert report["summary"]["factual"]["first_attempt_accuracy_rate"] == 1.0
    assert report["summary"]["factual"]["finding_counts"] == {}


def test_demo_can_run_one_named_scenario(tmp_path):
    report_path = tmp_path / "excluded.json"
    exit_code = demo_main(
        [
            "--backend",
            "synthetic",
            "--scenario",
            "excluded_skill",
            "--audit-path",
            str(tmp_path / "audit.jsonl"),
            "--report-path",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["summary"]["scenario_count"] == 1
    assert report["results"][0]["scenario"] == "excluded_skill"


def test_comparison_command_renders_all_run_labels(tmp_path):
    report_paths = []
    for label, backend in (
        ("subrep-baseline", "synthetic"),
        ("minimax-m3", "live"),
        ("gpt-oss-120b", "live"),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(
            json.dumps(
                {
                    "run_label": label,
                    "backend": backend,
                    "provider": "ASICloud" if backend == "live" else None,
                    "model": label if backend == "live" else None,
                    "evidence_label": "SYNTHETIC",
                    "results": [],
                    "summary": {
                        "valid_outcome_rate": 1.0,
                        "first_pass_valid_rate": 1.0,
                        "retry_rate": 0.0,
                        "median_latency_seconds": 0.1,
                        "p95_latency_seconds": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )
        report_paths.append(path)

    output = tmp_path / "comparison.md"
    exit_code = compare_main(
        [
            "--report",
            str(report_paths[0]),
            "--report",
            str(report_paths[1]),
            "--report",
            str(report_paths[2]),
            "--output",
            str(output),
        ]
    )

    rendered = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "subrep-baseline" in rendered
    assert "minimax-m3" in rendered
    assert "gpt-oss-120b" in rendered


def _attempt(status, findings, checked=True):
    return {
        "attempt": 1,
        "status": status,
        "error": None,
        "findings": [
            {"code": code, "category": category, "message": code} for code, category in findings
        ],
        "explanation_checked": checked,
        "raw_response": json.dumps({"explanation": "text"}),
    }


def _result(scenario, *attempts, status="accepted"):
    return {
        "scenario": scenario,
        "trial": 1,
        "latency_seconds": 1.0,
        "outcome": {"request_id": f"r-{scenario}", "status": status, "attempts": list(attempts)},
    }


def test_summary_reports_format_and_factual_accuracy_apart():
    results = [
        _result("a", _attempt("valid", [])),
        _result(
            "b",
            _attempt("invalid_response", [("WRONG_EXCLUSION_STAGE", "factual")]),
            _attempt("valid", []),
        ),
        _result(
            "c",
            _attempt("invalid_response", [("MALFORMED_RESPONSE", "format")], checked=False),
            status="invalid_response",
        ),
    ]

    summary = summarize(results)

    assert summary["request_count"] == 3
    assert summary["valid_outcome_rate"] == 2 / 3
    assert summary["format"]["first_attempt_format_valid_rate"] == 2 / 3
    assert summary["format"]["finding_counts"] == {"MALFORMED_RESPONSE": 1}
    assert summary["factual"]["checked_attempt_count"] == 3
    assert summary["factual"]["first_attempt_accuracy_rate"] == 1 / 2
    assert summary["factual"]["accuracy_rate"] == 2 / 3
    assert summary["factual"]["finding_counts"] == {"WRONG_EXCLUSION_STAGE": 1}


def test_summary_of_only_backend_errors_has_no_rates():
    summary = summarize([_result("a", {"attempt": 1, "status": "backend_error"}, status="backend_error")])

    assert summary["format"]["first_attempt_format_valid_rate"] is None
    assert summary["factual"]["accuracy_rate"] is None


def test_demo_repeats_each_scenario(tmp_path):
    report_path = tmp_path / "repeat.json"
    exit_code = demo_main(
        [
            "--backend",
            "synthetic",
            "--scenario",
            "mixed_exclusions",
            "--scenario",
            "abstention_mixed_reasons",
            "--repeat",
            "3",
            "--audit-path",
            str(tmp_path / "audit.jsonl"),
            "--report-path",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["summary"]["scenario_count"] == 2
    assert report["summary"]["request_count"] == 6
    assert [item["trial"] for item in report["results"]] == [1, 1, 2, 2, 3, 3]


def test_comparison_pools_trials_and_writes_a_review_sheet(tmp_path):
    paths = []
    for trial in (1, 2):
        path = tmp_path / f"trial{trial}.json"
        assert demo_main(
            [
                "--backend",
                "synthetic",
                "--scenario",
                "excluded_skill",
                "--run-label",
                "pooled",
                "--omega-image",
                "omega@sha256:test",
                "--audit-path",
                str(tmp_path / "audit.jsonl"),
                "--report-path",
                str(path),
            ]
        ) == 0
        paths.append(path)
    output = tmp_path / "comparison.md"
    review = tmp_path / "review.csv"

    exit_code = compare_main(
        [
            "--report",
            str(paths[0]),
            "--report",
            str(paths[1]),
            "--output",
            str(output),
            "--review-csv",
            str(review),
        ]
    )

    rendered = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert rendered.count("| pooled |") == 2
    assert "## Factual accuracy (automated checks)" in rendered
    assert "2 report(s); Omega image omega@sha256:test" in rendered
    rows = review.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("run_label,scenario,trial,request_id")
    assert len(rows) == 3
    assert "skill_efficiency was excluded" in rows[1]
