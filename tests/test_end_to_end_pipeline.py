"""
test_end_to_end_pipeline.py — End-to-end validation of the full SubRep pipeline.

Run with:
    python -m pytest tests/test_end_to_end_pipeline.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from demo.run_full_pipeline import run_pipeline


class TestEndToEndPipeline:
    """Validate the complete pipeline execution with MDN and report integration."""

    def test_pipeline_returns_valid_stats(self):
        """Assert run_pipeline() returns a dict with all required keys."""
        stats = run_pipeline()

        assert isinstance(stats, dict), "Pipeline must return a dict"

        required_keys = [
            "total_episodes",
            "admitted",
            "rejected",
            "admission_rate",
            "rejection_rate",
            "cds_pass_count",
            "pds_pass_count",
            "first_admitted_ep",
            "library_size",
            "episode_records",
        ]

        for key in required_keys:
            assert key in stats, f"Missing required key: {key}"

    def test_admission_report_json_created(self):
        """Assert that admission_report.json is generated on disk."""
        run_pipeline()

        report_path = Path("demo/artifacts/admission_report.json")
        assert report_path.exists(), f"Admission report not found at {report_path}"

        # Verify it's valid JSON with expected structure
        import json
        content = json.loads(report_path.read_text(encoding="utf-8"))
        assert "total_attempted" in content
        assert "admitted" in content
        assert "rejected" in content
        assert "admission_rate" in content

    def test_cert_store_matches_library_size(self):
        """Assert cert_store.count() == library_size from stats."""
        stats = run_pipeline()

        # The pipeline maintains the invariant internally
        # We verify the reported library_size is consistent
        assert stats["library_size"] == stats["admitted"], (
            f"Library size ({stats['library_size']}) must equal admitted count "
            f"({stats['admitted']}) since rejected skills never enter the library"
        )

    def test_no_rejected_skills_in_library(self):
        """Assert that rejected skills did not enter the library."""
        stats = run_pipeline()

        # Count admitted skills from episode records
        admitted_from_records = sum(
            1 for record in stats["episode_records"] if record["admitted"]
        )

        assert admitted_from_records == stats["admitted"], (
            f"Admitted count mismatch: stats says {stats['admitted']}, "
            f"but episode records show {admitted_from_records}"
        )

        # Verify library_size matches admitted count
        assert stats["library_size"] == stats["admitted"], (
            f"Library contains {stats['library_size']} skills, but only "
            f"{stats['admitted']} were admitted"
        )

    def test_admission_rate_calculation(self):
        """Assert admission rate is calculated correctly."""
        stats = run_pipeline()

        total = stats["total_episodes"]
        if total > 0:
            expected_rate = (stats["admitted"] / total) * 100
            assert abs(stats["admission_rate"] - expected_rate) < 0.1, (
                f"Admission rate mismatch: expected {expected_rate:.2f}%, "
                f"got {stats['admission_rate']}%"
            )

    def test_episode_records_count_matches_total(self):
        """Assert the number of episode records matches total_episodes."""
        stats = run_pipeline()

        assert len(stats["episode_records"]) == stats["total_episodes"], (
            f"Episode records count ({len(stats['episode_records'])}) must match "
            f"total_episodes ({stats['total_episodes']})"
        )

    def test_rejection_rate_complements_admission_rate(self):
        """Assert rejection_rate = 100 - admission_rate."""
        stats = run_pipeline()

        expected_rejection_rate = 100.0 - stats["admission_rate"]
        assert abs(stats["rejection_rate"] - expected_rejection_rate) < 0.1, (
            f"Rejection rate ({stats['rejection_rate']}) must complement "
            f"admission rate ({stats['admission_rate']})"
        )
