"""Compare OmegaClaw demo runs: response format and factual accuracy, reported apart."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .contracts import FACTUAL_FINDING, FORMAT_FINDING
from .demo import summarize
from .validation import parse_structured_response

REVIEW_COLUMNS = (
    "run_label",
    "scenario",
    "trial",
    "request_id",
    "attempt",
    "status",
    "explanation_checked",
    "format_findings",
    "factual_findings",
    "explanation",
    "reviewer_factually_correct",
    "reviewer_notes",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OmegaClaw run reports. Reports that share a run label, backend, "
            "provider, and model are pooled, so one report per trial is fine."
        )
    )
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Input JSON report. Repeat once per report.",
    )
    parser.add_argument("--output", required=True, help="Markdown output path.")
    parser.add_argument(
        "--review-csv",
        default=None,
        help="Optional CSV with one row per attempt, for a person to review each explanation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runs = pool_reports([_load_report(Path(path)) for path in args.report])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(runs), encoding="utf-8")
    print(f"Wrote comparison report to {output}")
    if args.review_csv:
        review = Path(args.review_csv)
        review.parent.mkdir(parents=True, exist_ok=True)
        _write_review_sheet(runs, review)
        print(f"Wrote review sheet to {review}")
    return 0


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_label", "backend", "evidence_label", "results", "summary"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    if payload["evidence_label"] != "SYNTHETIC":
        raise ValueError(f"{path} must be labeled SYNTHETIC")
    return payload


def pool_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge reports of the same run, in first-seen order, and summarise the pooled results."""

    runs: dict[tuple, dict[str, Any]] = {}
    for report in reports:
        key = (
            report["run_label"],
            report["backend"],
            report.get("provider"),
            report.get("model"),
        )
        run = runs.setdefault(
            key,
            {
                "run_label": report["run_label"],
                "backend": report["backend"],
                "provider": report.get("provider"),
                "model": report.get("model"),
                "omega_images": [],
                "subrep_commits": [],
                "report_count": 0,
                "results": [],
                "summary": report["summary"],
            },
        )
        run["report_count"] += 1
        run["results"].extend(report["results"])
        for field, values in (("omega_image", "omega_images"), ("subrep_commit", "subrep_commits")):
            value = report.get(field)
            if value and value not in run[values]:
                run[values].append(value)
    for run in runs.values():
        if run["results"]:
            run["summary"] = summarize(run["results"])
    return list(runs.values())


def _render_markdown(runs: list[dict[str, Any]]) -> str:
    format_rows = []
    factual_rows = []
    for run in runs:
        summary = run["summary"]
        response_format = summary.get("format") or {}
        factual = summary.get("factual") or {}
        name = f"{run['run_label']} | {run['backend']} | {run['provider'] or 'N/A'} | {run['model'] or 'N/A'}"
        format_rows.append(
            f"| {name} | {summary.get('request_count', 'N/A')} | "
            f"{_percent(summary.get('valid_outcome_rate'))} | "
            f"{_percent(response_format.get('first_attempt_format_valid_rate'))} | "
            f"{_percent(response_format.get('format_valid_attempt_rate'))} | "
            f"{_percent(summary.get('retry_rate'))} | "
            f"{response_format.get('backend_error_count', 'N/A')} | "
            f"{_seconds(summary.get('median_latency_seconds'))} | "
            f"{_seconds(summary.get('p95_latency_seconds'))} | "
            f"{_counts(response_format.get('finding_counts'))} |"
        )
        factual_rows.append(
            f"| {name} | {factual.get('checked_attempt_count', 'N/A')} | "
            f"{_percent(factual.get('first_attempt_accuracy_rate'))} | "
            f"{_percent(factual.get('accuracy_rate'))} | "
            f"{_counts(factual.get('finding_counts'))} |"
        )
    provenance = [
        f"- {run['run_label']}: {run['report_count']} report(s); Omega image "
        f"{', '.join(run['omega_images']) or 'not recorded'}; SubRep commit "
        f"{', '.join(run['subrep_commits']) or 'not recorded'}."
        for run in runs
    ]
    return "\n".join(
        [
            "# OmegaClaw Run Comparison",
            "",
            "All scenarios use synthetic evidence. No selected skill was executed.",
            "SubRep keeps selection authority, so the runs are compared on how well",
            "they explain SubRep's decision, not on what they would have chosen.",
            "",
            "## Response format",
            "",
            "Did the response follow the contract: valid JSON, schema, decision echo",
            "fields, and citations that exist in the request?",
            "",
            "| Run | Backend | Provider | Model | Requests | Valid outcomes | "
            "First-attempt format valid | Format valid (all attempts) | Retry rate | "
            "Backend errors | Median latency (s) | P95 latency (s) | Format findings |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *format_rows,
            "",
            "## Factual accuracy (automated checks)",
            "",
            "Did the explanation match the evidence: decision, roles, exclusion stages,",
            "scores, no execution claims, and synthetic evidence disclosed? Only attempts",
            "with an explanation to check are counted. The checks catch known error",
            "patterns; a person must still review the explanations (see the review sheet).",
            "",
            "| Run | Backend | Provider | Model | Checked attempts | First-attempt accuracy | "
            "Accuracy (all attempts) | Factual findings |",
            "|---|---|---|---|---:|---:|---:|---|",
            *factual_rows,
            "",
            "## Provenance",
            "",
            *provenance,
            "",
            "## Interpretation",
            "",
            f"- `{FORMAT_FINDING}` findings concern the response shape and citations;",
            f"  `{FACTUAL_FINDING}` findings concern whether the explanation is true.",
            "- An invalid or backend-error outcome never replaces SubRep's decision.",
            "- The synthetic baseline does not call a live model and is not a provider",
            "  latency comparison target.",
            "",
        ]
    )


def _write_review_sheet(runs: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for run in runs:
            for result in run["results"]:
                outcome = result["outcome"]
                for attempt in outcome["attempts"]:
                    findings = attempt.get("findings") or ()
                    writer.writerow(
                        {
                            "run_label": run["run_label"],
                            "scenario": result["scenario"],
                            "trial": result.get("trial", 1),
                            "request_id": outcome["request_id"],
                            "attempt": attempt["attempt"],
                            "status": attempt["status"],
                            "explanation_checked": bool(attempt.get("explanation_checked")),
                            "format_findings": _joined(findings, FORMAT_FINDING),
                            "factual_findings": _joined(findings, FACTUAL_FINDING),
                            "explanation": _explanation(attempt.get("raw_response")),
                            "reviewer_factually_correct": "",
                            "reviewer_notes": "",
                        }
                    )


def _joined(findings, category: str) -> str:
    return "; ".join(item["code"] for item in findings if item["category"] == category)


def _explanation(raw_response: str | None) -> str:
    if not raw_response:
        return ""
    try:
        explanation = parse_structured_response(raw_response).get("explanation")
    except ValueError:
        return raw_response
    return explanation if isinstance(explanation, str) else raw_response


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.1%}"


def _seconds(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{code} x{count}" for code, count in counts.items())


if __name__ == "__main__":
    raise SystemExit(main())
