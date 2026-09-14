"""Run SubRep's synthetic multi-objective benchmark."""

from __future__ import annotations

import argparse

from utils.multi_objective_benchmark import run_multi_objective_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SubRep's synthetic M > 2 benchmark.")
    parser.add_argument("--objectives", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--output", default="demo/artifacts/multi_objective_benchmark.json")
    parser.add_argument("--markdown-output", default="demo/artifacts/multi_objective_benchmark.md")
    args = parser.parse_args()

    summary = run_multi_objective_benchmark(
        objective_counts=tuple(args.objectives),
        candidates_per_objective_count=args.candidates,
        seeds=tuple(args.seeds),
        epsilon=args.epsilon,
        output_json=args.output,
        output_markdown=args.markdown_output,
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown_output}")
    for result in summary["results"]:
        print(
            "M={num_objectives}: candidates={candidate_skills_evaluated}, "
            "admitted={admitted}, rejected={rejected}, "
            "reuse_success={reuse_success_rate:.3f}, negative_transfer={negative_transfer_rate:.3f}, "
            "query_ms={query_time_ms:.3f}".format(**result)
        )


if __name__ == "__main__":
    main()
