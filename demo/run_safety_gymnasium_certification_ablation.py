"""Run the Safety-Gymnasium certification ablation study."""

from __future__ import annotations

import argparse

from utils.safety_gymnasium_certification_ablation import (
    build_safety_gymnasium_certification_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Safety-Gymnasium SubRep selection with and without CDS/PDS."
    )
    parser.add_argument(
        "--rollout-dir",
        action="append",
        required=True,
        help="Rollout directory. Repeat for multiple seeds.",
    )
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--baseline-candidate", type=str, default="zero_action")
    parser.add_argument("--pds-epsilon", type=float, default=1.0)
    parser.add_argument(
        "--summary-json",
        type=str,
        default="demo/artifacts/safety_gymnasium_certification_ablation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_safety_gymnasium_certification_ablation(
        rollout_dirs=args.rollout_dir,
        summary_json_path=args.summary_json,
        pattern=args.pattern,
        baseline_candidate_id=args.baseline_candidate,
        pds_epsilon=args.pds_epsilon,
    )
    print("Safety-Gymnasium Certification Ablation Complete")
    print("===============================================")
    print(f"summary: {args.summary_json}")
    print(f"contexts: {summary['total_contexts']}")
    for query, stats in summary["queries"].items():
        print(
            f"{query}: with_cert_cost={stats['with_certification_mean_safety_cost']:.4f} "
            f"without_cert_cost={stats['without_certification_mean_safety_cost']:.4f} "
            f"blocked_rate={100.0 * stats['blocked_without_certification_selection_rate']:.1f}%"
        )


if __name__ == "__main__":
    main()
