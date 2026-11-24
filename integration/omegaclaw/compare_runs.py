"""Compare OmegaClaw demo runs: response format and factual checks, reported apart."""

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
CORRECT_VERDICTS = frozenset({"yes", "y", "true", "correct"})
INCORRECT_VERDICTS = frozenset({"no", "n", "false", "incorrect"})
NO_ERRORS_DETECTED = "No errors detected by the checks"
EXPLANATION_CHARS = 500


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
    parser.add_argument(
        "--reviewed-csv",
        default=None,
        help=(
            "A review sheet filled in by a person (reviewer_factually_correct = yes or no), "
            "merged into the report."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runs = pool_reports([_load_report(Path(path)) for path in args.report])
    reviews = load_reviews(Path(args.reviewed_csv)) if args.reviewed_csv else {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(runs, reviews), encoding="utf-8")
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


def load_reviews(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Read a person's verdicts, keyed by request ID and attempt number."""

    reviews: dict[tuple[str, int], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            verdict = (row.get("reviewer_factually_correct") or "").strip().lower()
            if not verdict:
                continue
            if verdict in CORRECT_VERDICTS:
                normalised = "correct"
            elif verdict in INCORRECT_VERDICTS:
                normalised = "incorrect"
            else:
                raise ValueError(
                    f"{path}:{line_number}: reviewer_factually_correct must be yes or no, "
                    f"got {verdict!r}"
                )
            reviews[(row["request_id"], int(row["attempt"]))] = {
                "verdict": normalised,
                "notes": (row.get("reviewer_notes") or "").strip(),
            }
    return reviews


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


def _attempt_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for result in run["results"]:
            outcome = result["outcome"]
            for attempt in outcome["attempts"]:
                findings = attempt.get("findings") or ()
                rows.append(
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
                    }
                )
    return rows


def _render_markdown(
    runs: list[dict[str, Any]],
    reviews: dict[tuple[str, int], dict[str, str]] | None = None,
) -> str:
    reviews = reviews or {}
    rows = _attempt_rows(runs)
    return "\n".join(
        [
            "# OmegaClaw Run Comparison",
            "",
            "All scenarios use synthetic evidence. No selected skill was executed.",
            "SubRep keeps selection authority, so the runs are compared on how well",
            "they explain SubRep's decision, not on what they would have chosen.",
            "",
            *_runs_section(runs),
            *_format_section(runs),
            *_factual_section(runs),
            *_review_section(rows, reviews),
            *_responses_section(rows, reviews),
            *_remaining_errors_section(runs, rows, reviews),
            "## Interpretation",
            "",
            f"- `{FORMAT_FINDING}` findings concern the response shape and citations;",
            f"  `{FACTUAL_FINDING}` findings concern what the explanation states.",
            f"- \"{NO_ERRORS_DETECTED}\" means no pattern check found an error. The checks",
            "  cannot prove that an explanation is accurate; the review by a person can",
            "  find errors they miss.",
            "- An invalid or backend-error outcome never replaces SubRep's decision.",
            "- The synthetic baseline does not call a live model and is not a provider",
            "  latency comparison target.",
            "",
        ]
    )


def _runs_section(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Runs",
        "",
        "| Run | Backend | Provider | Model | Omega image | SubRep commit | Reports | Requests |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for run in runs:
        live = run["backend"] != "synthetic"
        lines.append(
            f"| {run['run_label']} | {run['backend']} | {_metadata(run['provider'], live)} | "
            f"{_metadata(run['model'], live)} | "
            f"{_metadata(', '.join(run['omega_images']), live)} | "
            f"{', '.join(run['subrep_commits']) or 'not recorded'} | {run['report_count']} | "
            f"{run['summary'].get('request_count', 'N/A')} |"
        )
    return [*lines, ""]


def _format_section(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Response format",
        "",
        "Did the response follow the contract: valid JSON, schema, decision echo",
        "fields, and citations that exist in the request?",
        "",
        "| Run | Requests | Valid outcomes | First-attempt format valid | "
        "Format valid (all attempts) | Retry rate | Backend errors | Median latency (s) | "
        "P95 latency (s) | Format findings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        summary = run["summary"]
        response_format = summary.get("format") or {}
        lines.append(
            f"| {run['run_label']} | {summary.get('request_count', 'N/A')} | "
            f"{_percent(summary.get('valid_outcome_rate'))} | "
            f"{_percent(response_format.get('first_attempt_format_valid_rate'))} | "
            f"{_percent(response_format.get('format_valid_attempt_rate'))} | "
            f"{_percent(summary.get('retry_rate'))} | "
            f"{response_format.get('backend_error_count', 'N/A')} | "
            f"{_seconds(summary.get('median_latency_seconds'))} | "
            f"{_seconds(summary.get('p95_latency_seconds'))} | "
            f"{_counts(response_format.get('finding_counts'))} |"
        )
    return [*lines, ""]


def _factual_section(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Factual checks (automated)",
        "",
        "Pattern checks of what the explanation states: decision, roles, exclusion",
        "stages, scores and their signs, decision fields pasted into the text, skills",
        "outside the request, execution claims, and synthetic disclosure. Only attempts",
        f"with an explanation to check are counted. \"{NO_ERRORS_DETECTED}\" is not",
        "proof of accuracy.",
        "",
        f"| Run | Checked attempts | {NO_ERRORS_DETECTED} (first attempt) | "
        f"{NO_ERRORS_DETECTED} (all attempts) | Factual findings |",
        "|---|---:|---:|---:|---|",
    ]
    for run in runs:
        factual = run["summary"].get("factual") or {}
        lines.append(
            f"| {run['run_label']} | {factual.get('checked_attempt_count', 'N/A')} | "
            f"{_percent(factual.get('first_attempt_no_errors_detected_rate'))} | "
            f"{_percent(factual.get('no_errors_detected_rate'))} | "
            f"{_counts(factual.get('finding_counts'))} |"
        )
    return [*lines, ""]


def _review_section(
    rows: list[dict[str, Any]],
    reviews: dict[tuple[str, int], dict[str, str]],
) -> list[str]:
    reviewable = [row for row in rows if row["explanation"]]
    reviewed = [(row, reviews[_key(row)]) for row in reviewable if _key(row) in reviews]
    lines = ["## Review by a person", ""]
    if not reviewed:
        return [
            *lines,
            "Not reviewed yet. Fill reviewer_factually_correct (yes or no) in the review",
            "sheet and pass it with --reviewed-csv.",
            "",
        ]
    incorrect = [(row, review) for row, review in reviewed if review["verdict"] == "incorrect"]
    missed = [row for row, _ in incorrect if not row["factual_findings"]]
    false_alarms = [
        row for row, review in reviewed if review["verdict"] == "correct" and row["factual_findings"]
    ]
    return [
        *lines,
        f"- Responses with an explanation: {len(reviewable)}; reviewed: {len(reviewed)}.",
        f"- Judged factually correct: {len(reviewed) - len(incorrect)}; incorrect: {len(incorrect)}.",
        f"- Incorrect, yet no errors detected by the checks: {len(missed)}.",
        f"- Flagged by the checks, yet judged correct: {len(false_alarms)}.",
        "",
    ]


def _responses_section(
    rows: list[dict[str, Any]],
    reviews: dict[tuple[str, int], dict[str, str]],
) -> list[str]:
    lines = [
        "## Responses",
        "",
        "| Run | Scenario | Trial | Attempt | Status | Findings | Reviewer | Explanation |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        review = reviews.get(_key(row))
        verdict = "not reviewed" if review is None else review["verdict"]
        if review and review["notes"]:
            verdict += f": {review['notes']}"
        findings = "; ".join(part for part in (row["format_findings"], row["factual_findings"]) if part)
        lines.append(
            f"| {row['run_label']} | {row['scenario']} | {row['trial']} | {row['attempt']} | "
            f"{row['status']} | {_cell(findings or 'none')} | {_cell(verdict)} | "
            f"{_cell(_shorten(row['explanation']) or '(no response)')} |"
        )
    return [*lines, ""]


def _remaining_errors_section(
    runs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    reviews: dict[tuple[str, int], dict[str, str]],
) -> list[str]:
    lines = ["## Remaining errors", ""]
    for run in runs:
        outcomes = [result["outcome"] for result in run["results"]]
        invalid = sum(outcome["status"] == "invalid_response" for outcome in outcomes)
        backend = sum(outcome["status"] == "backend_error" for outcome in outcomes)
        summary = run["summary"]
        lines.append(
            f"- {run['run_label']}: {invalid} request(s) ended invalid after the retry and "
            f"{backend} with a backend error. Format findings: "
            f"{_counts((summary.get('format') or {}).get('finding_counts'))}. Factual "
            f"findings: {_counts((summary.get('factual') or {}).get('finding_counts'))}."
        )
    for row in rows:
        review = reviews.get(_key(row))
        if review and review["verdict"] == "incorrect":
            lines.append(
                f"- Reviewer: {row['run_label']} {row['scenario']} trial {row['trial']} "
                f"attempt {row['attempt']} is incorrect"
                + (f": {review['notes']}" if review["notes"] else ".")
                + ("" if row["factual_findings"] else " The checks detected no error here.")
            )
    return [*lines, ""]


def _write_review_sheet(runs: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in _attempt_rows(runs):
            writer.writerow({**row, "reviewer_factually_correct": "", "reviewer_notes": ""})


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (row["request_id"], int(row["attempt"]))


def _metadata(value: str | None, live: bool) -> str:
    if value:
        return value
    return "**missing**" if live else "N/A"


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


def _shorten(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= EXPLANATION_CHARS else text[: EXPLANATION_CHARS - 3] + "..."


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


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
