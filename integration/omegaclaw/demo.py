"""Run synthetic or live SubRep-to-Omega explanation scenarios."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

from .adapter import OmegaRecommendationAdapter
from .audit import JsonlAuditStore
from .contracts import FACTUAL_FINDING, FORMAT_FINDING
from .scenarios import predefined_scenarios
from .service import SubRepOmegaRecommendationService
from .synthetic_backend import SyntheticOmegaBackend
from .websocket_gateway import OmegaWebSocketGateway

VALID_STATUSES = frozenset({"accepted", "abstained"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run explanation-only SubRep to Omega scenarios."
    )
    parser.add_argument("--backend", choices=("synthetic", "live"), default="synthetic")
    parser.add_argument(
        "--audit-path",
        default="data/omegaclaw/demo_recommendations.jsonl",
        help="Append-only JSONL audit destination.",
    )
    parser.add_argument("--report-path", default=None, help="Optional JSON summary destination.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default="/agent")
    parser.add_argument("--token", default=os.environ.get("SUBREP_OMEGA_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--run-label", default=None, help="Label stored in the JSON report.")
    parser.add_argument("--provider", default=None, help="Live provider name for reporting.")
    parser.add_argument("--model", default=None, help="Live model ID for reporting.")
    parser.add_argument(
        "--omega-image",
        default=None,
        help="Omega image name or digest, recorded in the report for reproduction.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.name for scenario in predefined_scenarios()),
        help="Run only this scenario. Repeat to select multiple scenarios.",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        help="Run every selected scenario this many times, one full pass at a time.",
    )
    parser.add_argument(
        "--first-trial",
        type=_positive_int,
        default=1,
        help=(
            "Number of the first trial in this run, for a script that restarts Omega "
            "between trials and calls the demo once per trial."
        ),
    )
    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.backend == "live":
        missing = [
            flag
            for flag, value in (
                ("--provider", args.provider),
                ("--model", args.model),
                ("--omega-image", args.omega_image),
            )
            if not (value and value.strip())
        ]
        if missing:
            parser.error(
                f"live runs must record {', '.join(missing)} so the report shows what was "
                "tested; for Omega outside Docker, pass its version or commit as --omega-image"
            )
    audit_store = JsonlAuditStore(args.audit_path)
    settings = {
        "timeout": args.timeout,
        "report_path": args.report_path,
        "provider": args.provider,
        "model": args.model,
        "omega_image": args.omega_image,
        "scenario_names": args.scenario,
        "repeat": args.repeat,
        "first_trial": args.first_trial,
    }

    if args.backend == "synthetic":
        return _run_scenarios(
            SyntheticOmegaBackend(),
            audit_store,
            backend_name="synthetic",
            run_label=args.run_label or "subrep_deterministic_baseline",
            **settings,
        )

    if not args.token:
        raise SystemExit(
            "Live mode requires --token or the SUBREP_OMEGA_TOKEN environment variable."
        )
    gateway = OmegaWebSocketGateway(
        host=args.host,
        port=args.port,
        token=args.token,
        path=args.path,
    )
    with gateway:
        print(f"Waiting for Omega at {gateway.url}")
        return _run_scenarios(
            gateway,
            audit_store,
            backend_name="live",
            run_label=args.run_label or "omega_live",
            **settings,
        )


def _run_scenarios(
    backend,
    audit_store,
    *,
    timeout: float,
    report_path: str | None,
    backend_name: str,
    run_label: str,
    provider: str | None,
    model: str | None,
    omega_image: str | None,
    scenario_names: list[str] | None,
    repeat: int,
    first_trial: int = 1,
) -> int:
    adapter = OmegaRecommendationAdapter(
        backend=backend,
        audit_store=audit_store,
        timeout_seconds=timeout,
    )
    service = SubRepOmegaRecommendationService(adapter)
    results = []
    selected_names = set(scenario_names or ())
    scenarios = tuple(
        scenario
        for scenario in predefined_scenarios()
        if not selected_names or scenario.name in selected_names
    )
    for trial in range(first_trial, first_trial + repeat):
        for scenario in scenarios:
            started_at = perf_counter()
            outcome = service.recommend_from_library(
                **scenario.service_arguments(),
                evidence_label="SYNTHETIC",
            )
            latency_seconds = perf_counter() - started_at
            result = {
                "scenario": scenario.name,
                "trial": trial,
                "description": scenario.description,
                "evidence_label": "SYNTHETIC",
                "latency_seconds": latency_seconds,
                "outcome": outcome.to_dict(),
            }
            results.append(result)
            print(json.dumps(result, indent=2, sort_keys=True))

    report = {
        "report_type": "omegaclaw_explanation_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "backend": backend_name,
        "provider": provider,
        "model": model,
        "omega_image": omega_image,
        "subrep_commit": _subrep_commit(),
        "evidence_label": "SYNTHETIC",
        "results": results,
        "summary": summarize(results),
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0 if all(item["outcome"]["status"] in VALID_STATUSES for item in results) else 1


def summarize(results: list[dict]) -> dict:
    """Summarise outcomes, response format, and the factual checks as separate measures."""

    latencies = sorted(float(item["latency_seconds"]) for item in results)
    outcomes = [item["outcome"] for item in results]
    valid_count = sum(outcome["status"] in VALID_STATUSES for outcome in outcomes)
    first_pass_valid = sum(
        bool(outcome["attempts"]) and outcome["attempts"][0]["status"] == "valid"
        for outcome in outcomes
    )
    retry_count = sum(len(outcome["attempts"]) > 1 for outcome in outcomes)
    p95_index = max(0, ceil(0.95 * len(latencies)) - 1)
    attempts = [attempt for outcome in outcomes for attempt in outcome["attempts"]]
    first_attempts = [outcome["attempts"][0] for outcome in outcomes if outcome["attempts"]]
    return {
        "scenario_count": len({item["scenario"] for item in results}),
        "request_count": len(results),
        "valid_outcome_count": valid_count,
        "valid_outcome_rate": _rate(valid_count, len(results)),
        "first_pass_valid_count": first_pass_valid,
        "first_pass_valid_rate": _rate(first_pass_valid, len(results)),
        "retry_count": retry_count,
        "retry_rate": _rate(retry_count, len(results)),
        "median_latency_seconds": median(latencies) if latencies else None,
        "p95_latency_seconds": latencies[p95_index] if latencies else None,
        "format": _format_summary(attempts, first_attempts),
        "factual": _factual_summary(attempts, first_attempts),
    }


def _format_summary(attempts: list[dict], first_attempts: list[dict]) -> dict:
    """Did Omega answer with a response that follows the contract?"""

    answered = [attempt for attempt in attempts if attempt["status"] != "backend_error"]
    first_answered = [attempt for attempt in first_attempts if attempt["status"] != "backend_error"]
    valid = [attempt for attempt in answered if not _codes(attempt, FORMAT_FINDING)]
    first_valid = [attempt for attempt in first_answered if not _codes(attempt, FORMAT_FINDING)]
    return {
        "backend_error_count": len(attempts) - len(answered),
        "answered_attempt_count": len(answered),
        "format_valid_attempt_count": len(valid),
        "format_valid_attempt_rate": _rate(len(valid), len(answered)),
        "first_attempt_format_valid_rate": _rate(len(first_valid), len(first_answered)),
        "finding_counts": _finding_counts(answered, FORMAT_FINDING),
    }


def _factual_summary(attempts: list[dict], first_attempts: list[dict]) -> dict:
    """How often every claim and the decision echo matched the request.

    Each claim is compared with the value at its evidence path, so a factual finding is
    a verified error, not a guess from wording.
    """

    checked = [attempt for attempt in attempts if attempt.get("claims_checked")]
    first_checked = [attempt for attempt in first_attempts if attempt.get("claims_checked")]
    clean = [attempt for attempt in checked if not _codes(attempt, FACTUAL_FINDING)]
    first_clean = [attempt for attempt in first_checked if not _codes(attempt, FACTUAL_FINDING)]
    return {
        "checked_attempt_count": len(checked),
        "all_claims_verified_count": len(clean),
        "all_claims_verified_rate": _rate(len(clean), len(checked)),
        "first_attempt_all_claims_verified_rate": _rate(len(first_clean), len(first_checked)),
        "finding_counts": _finding_counts(checked, FACTUAL_FINDING),
    }


def _codes(attempt: dict, category: str) -> list[str]:
    return [
        finding["code"]
        for finding in attempt.get("findings") or ()
        if finding["category"] == category
    ]


def _finding_counts(attempts: list[dict], category: str) -> dict[str, int]:
    counts = Counter(code for attempt in attempts for code in _codes(attempt, category))
    return dict(sorted(counts.items()))


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _subrep_commit() -> str | None:
    """The commit the run used, marked when tracked files differ from it."""

    try:
        commit = _git("rev-parse", "HEAD")
        changed = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    if not commit:
        return None
    return f"{commit} (with uncommitted changes)" if changed else commit


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        capture_output=True,
        check=True,
        cwd=Path(__file__).resolve().parent,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
