"""Generate a zero-shot reuse curve for Safety-Gymnasium SubRep rollouts."""

from __future__ import annotations

import argparse

from utils.safety_gymnasium_reuse_curve import build_safety_gymnasium_reuse_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SubRep zero-shot reuse success under shifted motive weights."
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
        "--shift-points",
        nargs="*",
        type=float,
        default=None,
        help="Optional safety weights to evaluate, e.g. 0.0 0.25 0.5 0.75 1.0.",
    )
    parser.add_argument(
        "--baseline-retraining-steps",
        type=int,
        default=51_200,
        help="Reference environment-step cost for retraining a baseline per shifted weight.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo/artifacts/safety_gymnasium_zero_shot_reuse_curve.png",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="demo/artifacts/safety_gymnasium_zero_shot_reuse_curve.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_safety_gymnasium_reuse_curve(
        rollout_dirs=args.rollout_dir,
        output_path=args.output,
        summary_json_path=args.summary_json,
        pattern=args.pattern,
        baseline_candidate_id=args.baseline_candidate,
        pds_epsilon=args.pds_epsilon,
        shift_points=args.shift_points,
        baseline_retraining_steps=args.baseline_retraining_steps,
    )
    print("Safety-Gymnasium Zero-Shot Reuse Curve Complete")
    print("===============================================")
    print(f"plot: {args.output}")
    print(f"summary: {args.summary_json}")
    print(f"contexts: {summary['total_contexts']}")
    print(f"mean_success_rate: {100.0 * summary['mean_success_rate']:.1f}%")
    print(f"min_success_rate: {100.0 * summary['min_success_rate']:.1f}%")
    print(f"max_success_rate: {100.0 * summary['max_success_rate']:.1f}%")
    print(f"baseline_retraining_steps: {summary['baseline_retraining_steps']}")


if __name__ == "__main__":
    main()
