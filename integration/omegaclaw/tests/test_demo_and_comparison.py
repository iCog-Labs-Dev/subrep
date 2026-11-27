from __future__ import annotations

import csv
import json

import pytest

from integration.omegaclaw.compare_runs import load_reviews
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
    assert report["summary"]["factual"]["first_attempt_all_claims_verified_rate"] == 1.0
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
    # Live reports without an Omega image are shown as missing, not silently blank.
    assert "| minimax-m3 | live | ASICloud | minimax-m3 | **missing** |" in rendered
    assert "| subrep-baseline | synthetic | N/A | N/A | N/A |" in rendered


def _attempt(status, findings, checked=True):
    return {
        "attempt": 1,
        "status": status,
        "error": None,
        "findings": [
            {"code": code, "category": category, "message": code} for code, category in findings
        ],
        "claims_checked": checked,
        "explanation": "text" if status == "valid" else None,
        "raw_response": json.dumps({"claims": []}),
    }


def _result(scenario, *attempts, status="accepted"):
    return {
        "scenario": scenario,
        "trial": 1,
        "latency_seconds": 1.0,
        "outcome": {"request_id": f"r-{scenario}", "status": status, "attempts": list(attempts)},
    }


def test_summary_reports_format_and_factual_checks_apart():
    results = [
        _result("a", _attempt("valid", [])),
        _result(
            "b",
            _attempt("invalid_response", [("CLAIM_VALUE_MISMATCH", "factual")]),
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
    assert summary["factual"]["first_attempt_all_claims_verified_rate"] == 1 / 2
    assert summary["factual"]["all_claims_verified_rate"] == 2 / 3
    assert summary["factual"]["finding_counts"] == {"CLAIM_VALUE_MISMATCH": 1}


def test_summary_of_only_backend_errors_has_no_rates():
    summary = summarize([_result("a", {"attempt": 1, "status": "backend_error"}, status="backend_error")])

    assert summary["format"]["first_attempt_format_valid_rate"] is None
    assert summary["factual"]["all_claims_verified_rate"] is None


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
    assert "| pooled | synthetic | N/A | N/A | omega@sha256:test |" in rendered
    assert "## Claim verification" in rendered
    assert "All claims verified (first attempt)" in rendered
    # No measure is labelled as accuracy; a person judges that in the review.
    assert "accuracy" not in rendered.lower()
    assert rendered.count("| pooled | excluded_skill |") == 2
    assert "Not reviewed yet." in rendered
    rows = review.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("run_label,scenario,trial,request_id")
    assert len(rows) == 3
    assert "skill_efficiency was removed before ranking at the OPERATOR stage" in rows[1]


def test_report_marks_a_commit_with_uncommitted_changes(monkeypatch):
    from integration.omegaclaw import demo

    answers = {("rev-parse", "HEAD"): "abc123", ("status", "--porcelain", "--untracked-files=no"): ""}
    monkeypatch.setattr(demo, "_git", lambda *arguments: answers[arguments])
    assert demo._subrep_commit() == "abc123"

    answers[("status", "--porcelain", "--untracked-files=no")] = " M integration/omegaclaw/facts.py"
    assert demo._subrep_commit() == "abc123 (with uncommitted changes)"


def test_first_trial_numbers_a_run_started_once_per_trial(tmp_path):
    report_path = tmp_path / "trial3.json"
    assert demo_main(
        [
            "--backend",
            "synthetic",
            "--scenario",
            "excluded_skill",
            "--first-trial",
            "3",
            "--repeat",
            "2",
            "--audit-path",
            str(tmp_path / "audit.jsonl"),
            "--report-path",
            str(report_path),
        ]
    ) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [item["trial"] for item in report["results"]] == [3, 4]


def test_live_run_must_record_provider_model_and_image(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        demo_main(
            [
                "--backend",
                "live",
                "--token",
                "test-token",
                "--provider",
                "ASICloud",
                "--audit-path",
                str(tmp_path / "audit.jsonl"),
            ]
        )

    assert raised.value.code == 2
    assert "--model, --omega-image" in capsys.readouterr().err


def _synthetic_run(tmp_path, label="reviewed"):
    report = tmp_path / f"{label}.json"
    assert demo_main(
        [
            "--backend",
            "synthetic",
            "--scenario",
            "excluded_skill",
            "--scenario",
            "no_admissible_options",
            "--run-label",
            label,
            "--audit-path",
            str(tmp_path / "audit.jsonl"),
            "--report-path",
            str(report),
        ]
    ) == 0
    return report


def test_reviewed_sheet_is_merged_into_the_report(tmp_path):
    report = _synthetic_run(tmp_path)
    sheet = tmp_path / "review.csv"
    assert compare_main(
        ["--report", str(report), "--output", str(tmp_path / "first.md"), "--review-csv", str(sheet)]
    ) == 0
    with sheet.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["reviewer_factually_correct"] = "no"
    rows[0]["reviewer_notes"] = "does not say the operator excluded skill_efficiency"
    rows[1]["reviewer_factually_correct"] = "yes"
    reviewed = tmp_path / "reviewed.csv"
    with reviewed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "reviewed.md"
    assert compare_main(
        ["--report", str(report), "--output", str(output), "--reviewed-csv", str(reviewed)]
    ) == 0

    rendered = output.read_text(encoding="utf-8")
    assert "- Responses with an explanation: 2; reviewed: 2." in rendered
    assert "- Judged factually correct: 1; incorrect: 1." in rendered
    assert "- Incorrect, yet every claim verified: 1." in rendered
    assert "incorrect: does not say the operator excluded skill_efficiency" in rendered
    assert "Every claim in it verified." in rendered


def test_reviewed_sheet_rejects_an_unclear_verdict(tmp_path):
    sheet = tmp_path / "reviewed.csv"
    sheet.write_text(
        "request_id,attempt,reviewer_factually_correct,reviewer_notes\nr-1,1,maybe,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be yes or no"):
        load_reviews(sheet)
