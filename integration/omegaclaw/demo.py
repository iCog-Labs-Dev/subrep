"""Run synthetic or live SubRep-to-Omega explanation scenarios."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

from .adapter import OmegaRecommendationAdapter
from .audit import JsonlAuditStore
from .scenarios import predefined_scenarios
from .service import SubRepOmegaRecommendationService
from .synthetic_backend import SyntheticOmegaBackend
from .websocket_gateway import OmegaWebSocketGateway


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
        "--scenario",
        action="append",
        choices=(
            "clear_preferred_skill",
            "changed_task_priorities",
            "excluded_skill",
            "no_admissible_options",
        ),
        help="Run only this scenario. Repeat to select multiple scenarios.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit_store = JsonlAuditStore(args.audit_path)

    if args.backend == "synthetic":
        backend = SyntheticOmegaBackend()
        return _run_scenarios(
            backend,
            audit_store,
            args.timeout,
            args.report_path,
            backend_name="synthetic",
            run_label=args.run_label or "subrep_deterministic_baseline",
            provider=args.provider,
            model=args.model,
            scenario_names=args.scenario,
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
            args.timeout,
            args.report_path,
            backend_name="live",
            run_label=args.run_label or "omega_live",
            provider=args.provider,
            model=args.model,
            scenario_names=args.scenario,
        )


def _run_scenarios(
    backend,
    audit_store,
    timeout: float,
    report_path: str | None,
    *,
    backend_name: str,
    run_label: str,
    provider: str | None,
    model: str | None,
    scenario_names: list[str] | None,
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
    for scenario in scenarios:
        started_at = perf_counter()
        outcome = service.recommend_from_library(
            task_context=scenario.task_context,
            objective_weights=scenario.objective_weights,
            objective_order=scenario.objective_order,
            risk_budget=scenario.risk_budget,
            skill_library=scenario.skill_library,
            exclusions=scenario.exclusions,
            evidence_label="SYNTHETIC",
        )
        latency_seconds = perf_counter() - started_at
        result = {
            "scenario": scenario.name,
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
        "evidence_label": "SYNTHETIC",
        "results": results,
        "summary": _summarize(results),
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    valid_statuses = {"accepted", "abstained"}
    return 0 if all(item["outcome"]["status"] in valid_statuses for item in results) else 1


def _summarize(results: list[dict]) -> dict:
    valid_statuses = {"accepted", "abstained"}
    latencies = sorted(float(item["latency_seconds"]) for item in results)
    valid_count = sum(item["outcome"]["status"] in valid_statuses for item in results)
    first_pass_valid = sum(
        bool(item["outcome"]["attempts"])
        and item["outcome"]["attempts"][0]["status"] == "valid"
        for item in results
    )
    retry_count = sum(len(item["outcome"]["attempts"]) > 1 for item in results)
    p95_index = max(0, ceil(0.95 * len(latencies)) - 1)
    return {
        "scenario_count": len(results),
        "valid_outcome_count": valid_count,
        "valid_outcome_rate": valid_count / len(results),
        "first_pass_valid_count": first_pass_valid,
        "first_pass_valid_rate": first_pass_valid / len(results),
        "retry_count": retry_count,
        "retry_rate": retry_count / len(results),
        "median_latency_seconds": median(latencies),
        "p95_latency_seconds": latencies[p95_index],
    }


if __name__ == "__main__":
    raise SystemExit(main())
