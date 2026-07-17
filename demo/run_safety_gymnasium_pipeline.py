"""Run the SafeRL/Safety-Gymnasium SubRep certification pilot.

Usage:
    python -m demo.run_safety_gymnasium_pipeline
"""

from __future__ import annotations

import argparse

from utils.safety_gymnasium_pipeline import run_safety_gymnasium_certification_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Certify Safety-Gymnasium rollout artifacts with SubRep CDS/PDS gates."
    )
    parser.add_argument("--rollout-dir", type=str, default="data/safety_gymnasium_rollouts")
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--baseline-candidate", type=str, default="zero_action")
    parser.add_argument("--pds-epsilon", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cert-file", type=str, default="data/safety_gymnasium_certificates.metta")
    parser.add_argument("--library-file", type=str, default="data/safety_gymnasium_library.json")
    parser.add_argument(
        "--report-json",
        type=str,
        default="demo/artifacts/safety_gymnasium_admission_report.json",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default="demo/artifacts/safety_gymnasium_admission_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_safety_gymnasium_certification_pipeline(
        rollout_dir=args.rollout_dir,
        pattern=args.pattern,
        baseline_candidate_id=args.baseline_candidate,
        pds_epsilon=args.pds_epsilon,
        gamma=args.gamma,
        cert_file=args.cert_file,
        library_file=args.library_file,
        report_json_path=args.report_json,
        report_md_path=args.report_md,
    )

    stats = result.stats
    print("SubRep Safety-Gymnasium Certification Complete")
    print("============================================")
    print(f"contexts processed: {stats['contexts_processed']}")
    print(f"candidate outcomes certified: {stats['candidate_outcomes_certified']}")
    print(f"admitted: {stats['admitted']}")
    print(f"rejected: {stats['rejected']}")
    print(f"CDS admissions: {stats['cds_pass_count']}")
    print(f"PDS admissions: {stats['pds_pass_count']}")
    print(f"certificate store count: {stats['cert_store_count']}")
    print(f"skill library size: {stats['library_size']}")
    print(f"zero-shot task selection: {stats['zero_shot_reuse']['task_focused_selected_skill']}")
    print(f"zero-shot safety selection: {stats['zero_shot_reuse']['safety_focused_selected_skill']}")
    print(f"JSON report: {args.report_json}")
    print(f"Markdown report: {args.report_md}")


if __name__ == "__main__":
    main()
