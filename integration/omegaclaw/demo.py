"""Run synthetic or live SubRep-to-Omega recommendation scenarios."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapter import OmegaRecommendationAdapter
from .audit import JsonlAuditStore
from .scenarios import predefined_scenarios
from .service import SubRepOmegaRecommendationService
from .synthetic_backend import SyntheticOmegaBackend
from .websocket_gateway import OmegaWebSocketGateway


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run recommendation-only SubRep to Omega scenarios."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit_store = JsonlAuditStore(args.audit_path)

    if args.backend == "synthetic":
        backend = SyntheticOmegaBackend()
        return _run_scenarios(backend, audit_store, args.timeout, args.report_path)

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
        return _run_scenarios(gateway, audit_store, args.timeout, args.report_path)


def _run_scenarios(backend, audit_store, timeout: float, report_path: str | None) -> int:
    adapter = OmegaRecommendationAdapter(
        backend=backend,
        audit_store=audit_store,
        timeout_seconds=timeout,
    )
    service = SubRepOmegaRecommendationService(adapter)
    results = []
    for scenario in predefined_scenarios():
        outcome = service.recommend_from_library(
            task_context=scenario.task_context,
            objective_weights=scenario.objective_weights,
            risk_budget=scenario.risk_budget,
            skill_library=scenario.skill_library,
            exclusions=scenario.exclusions,
            evidence_label="SYNTHETIC",
        )
        result = {
            "scenario": scenario.name,
            "description": scenario.description,
            "evidence_label": "SYNTHETIC",
            "outcome": outcome.to_dict(),
        }
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))

    report = {
        "report_type": "synthetic_scenario_review",
        "evidence_label": "SYNTHETIC",
        "results": results,
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    valid_statuses = {"accepted", "abstained"}
    return 0 if all(item["outcome"]["status"] in valid_statuses for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
