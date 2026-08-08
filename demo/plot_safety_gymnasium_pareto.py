"""Generate a Pareto plot for Safety-Gymnasium SubRep benchmark rollouts."""

from __future__ import annotations

import argparse

from utils.safety_gymnasium_pareto import build_safety_gymnasium_pareto_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Safety-Gymnasium Pareto points for SubRep/PPO baselines."
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
        "--selection-weight",
        nargs=2,
        type=float,
        default=(0.10, 0.90),
        metavar=("SAFETY_WEIGHT", "TASK_WEIGHT"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo/artifacts/safety_gymnasium_pareto_frontier.png",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="demo/artifacts/safety_gymnasium_pareto_frontier.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_safety_gymnasium_pareto_report(
        rollout_dirs=args.rollout_dir,
        output_path=args.output,
        summary_json_path=args.summary_json,
        pattern=args.pattern,
        baseline_candidate_id=args.baseline_candidate,
        pds_epsilon=args.pds_epsilon,
        selection_weight=args.selection_weight,
    )
    print("Safety-Gymnasium Pareto Report Complete")
    print("======================================")
    print(f"plot: {args.output}")
    print(f"summary: {args.summary_json}")
    for method, method_summary in summary["methods"].items():
        print(
            f"{method}: available={method_summary['available']} "
            f"count={method_summary['count']} "
            f"mean_cost={method_summary['mean_safety_cost']:.4f} "
            f"mean_return={method_summary['mean_task_return']:.4f}"
        )


if __name__ == "__main__":
    main()
