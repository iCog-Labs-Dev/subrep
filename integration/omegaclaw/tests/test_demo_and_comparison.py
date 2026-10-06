from __future__ import annotations

import json

from integration.omegaclaw.compare_runs import main as compare_main
from integration.omegaclaw.demo import main as demo_main


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
    assert report["summary"]["scenario_count"] == 4
    assert report["summary"]["valid_outcome_rate"] == 1.0
    assert report["summary"]["first_pass_valid_rate"] == 1.0


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
