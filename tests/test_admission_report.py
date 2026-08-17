"""
test_admission_report.py — Unit tests for the AdmissionReport utility.

Run with:
    python -m pytest tests/test_admission_report.py -v
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from utils.admission_report import AdmissionRecord, AdmissionReport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _admitted_record(
    skill_id: str = "skill_001",
    gate_type: str = "CDS",
    epsilon: float = 0.0,
) -> AdmissionRecord:
    return AdmissionRecord(
        skill_id=skill_id,
        admitted=True,
        gate_type=gate_type,
        delta_r=5.0,
        delta_n=(2.0, 3.0),
        margin=7.0,
        failure_reason=None,
        epsilon=epsilon,
    )


def _rejected_record(skill_id: str = "skill_bad", reason: str = "delta_r + min(delta_n) < 0") -> AdmissionRecord:
    return AdmissionRecord(
        skill_id=skill_id,
        admitted=False,
        gate_type=None,
        delta_r=-10.0,
        delta_n=(-5.0, -3.0),
        margin=-15.0,
        failure_reason=reason,
    )


# ── Tests: compile() ──────────────────────────────────────────────────────────

class TestAdmissionReportCompile:
    def test_empty_report_has_zero_totals(self):
        report = AdmissionReport()
        stats = report.compile()
        assert stats["total_attempted"] == 0
        assert stats["admitted"] == 0
        assert stats["rejected"] == 0
        assert stats["admission_rate"] == 0.0

    def test_counts_admitted_correctly(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("s1"))
        report.add_record(_admitted_record("s2"))
        report.add_record(_rejected_record("s3"))
        stats = report.compile()
        assert stats["total_attempted"] == 3
        assert stats["admitted"] == 2
        assert stats["rejected"] == 1

    def test_admission_rate_calculation(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("s1"))
        report.add_record(_rejected_record("s2"))
        stats = report.compile()
        assert stats["admission_rate"] == pytest.approx(50.0)

    def test_cds_pass_count(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("s1", gate_type="CDS"))
        report.add_record(_admitted_record("s2", gate_type="CDS"))
        report.add_record(_admitted_record("s3", gate_type="PDS"))
        stats = report.compile()
        assert stats["cds_pass_count"] == 2
        assert stats["pds_pass_count"] == 1

    def test_pds_pass_count(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("s1", gate_type="PDS"))
        stats = report.compile()
        assert stats["pds_pass_count"] == 1
        assert stats["cds_pass_count"] == 0

    def test_failure_reasons_are_counted(self):
        reason = "delta_r + min(delta_n) < 0"
        report = AdmissionReport()
        report.add_record(_rejected_record("s1", reason=reason))
        report.add_record(_rejected_record("s2", reason=reason))
        stats = report.compile()
        assert stats["failure_reasons"][reason] == 2

    def test_multiple_distinct_failure_reasons(self):
        report = AdmissionReport()
        report.add_record(_rejected_record("s1", reason="reason_A"))
        report.add_record(_rejected_record("s2", reason="reason_B"))
        stats = report.compile()
        assert set(stats["failure_reasons"].keys()) == {"reason_A", "reason_B"}

    def test_example_admitted_skill_is_first_admitted(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("first_admitted"))
        report.add_record(_admitted_record("second_admitted"))
        stats = report.compile()
        assert stats["example_admitted_skill"]["skill_id"] == "first_admitted"

    def test_example_pds_skill_is_first_pds_admission(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("cds_skill", gate_type="CDS"))
        report.add_record(_admitted_record("pds_skill", gate_type="PDS", epsilon=5.0))
        stats = report.compile()
        assert stats["example_pds_skill"]["skill_id"] == "pds_skill"
        assert stats["example_pds_skill"]["epsilon"] == 5.0

    def test_example_rejected_skill_is_first_rejected(self):
        report = AdmissionReport()
        report.add_record(_rejected_record("first_rejected"))
        report.add_record(_rejected_record("second_rejected"))
        stats = report.compile()
        assert stats["example_rejected_skill"]["skill_id"] == "first_rejected"

    def test_no_example_admitted_when_all_rejected(self):
        report = AdmissionReport()
        report.add_record(_rejected_record())
        stats = report.compile()
        assert stats["example_admitted_skill"] is None

    def test_no_example_rejected_when_all_admitted(self):
        report = AdmissionReport()
        report.add_record(_admitted_record())
        stats = report.compile()
        assert stats["example_rejected_skill"] is None


# ── Tests: add_from_dict() ────────────────────────────────────────────────────

class TestAdmissionReportAddFromDict:
    def test_add_from_dict_admitted(self):
        report = AdmissionReport()
        ep = {
            "skill_id": "s1",
            "admitted": True,
            "gate_type": "CDS",
            "delta_r": 5.0,
            "delta_n": (2.0, 3.0),
            "margin": 7.0,
            "failure_reason": None,
        }
        report.add_from_dict(ep)
        assert report.compile()["admitted"] == 1

    def test_add_from_dict_rejected(self):
        report = AdmissionReport()
        ep = {
            "skill_id": "s2",
            "candidate_policy": "noop",
            "admitted": False,
            "gate_type": None,
            "delta_r": -1.0,
            "delta_n": (-2.0, -3.0),
            "margin": -3.0,
            "failure_reason": "some reason",
        }
        report.add_from_dict(ep)
        stats = report.compile()
        assert stats["rejected"] == 1
        assert stats["example_rejected_skill"]["candidate_policy"] == "noop"


# ── Tests: save_json() ────────────────────────────────────────────────────────

class TestAdmissionReportSaveJson:
    def test_saves_valid_json(self):
        report = AdmissionReport()
        report.add_record(_admitted_record())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        assert "total_attempted" in data
        assert "admission_rate" in data
        assert "failure_reasons" in data

    def test_creates_parent_directories(self):
        report = AdmissionReport()
        report.add_record(_admitted_record())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "report.json"
            report.save_json(path)
            assert path.exists()

    def test_json_counts_match_compile(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("s1"))
        report.add_record(_admitted_record("s2"))
        report.add_record(_rejected_record("s3"))
        stats = report.compile()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        assert data["admitted"] == stats["admitted"]
        assert data["rejected"] == stats["rejected"]
        assert data["total_attempted"] == stats["total_attempted"]


# ── Tests: save_markdown() ────────────────────────────────────────────────────

class TestAdmissionReportSaveMarkdown:
    def test_creates_markdown_file(self):
        report = AdmissionReport()
        report.add_record(_admitted_record())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            assert path.exists()

    def test_markdown_contains_summary_header(self):
        report = AdmissionReport()
        report.add_record(_admitted_record())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")
        assert "## Summary Statistics" in content

    def test_markdown_contains_failure_reasons(self):
        report = AdmissionReport()
        report.add_record(_rejected_record(reason="test failure reason"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")
        assert "test failure reason" in content

    def test_markdown_mentions_example_admitted_skill(self):
        report = AdmissionReport()
        report.add_record(_admitted_record(skill_id="my_special_skill"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")
        assert "my_special_skill" in content

    def test_markdown_mentions_candidate_policy_when_available(self):
        report = AdmissionReport()
        report.add_record(
            AdmissionRecord(
                skill_id="skill_001_ppo",
                admitted=True,
                gate_type="CDS",
                delta_r=5.0,
                delta_n=(2.0, 3.0),
                margin=7.0,
                failure_reason=None,
                candidate_policy="ppo_deterministic",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")
        assert "ppo_deterministic" in content

    def test_markdown_mentions_pds_tradeoff_example(self):
        report = AdmissionReport()
        report.add_record(
            AdmissionRecord(
                skill_id="skill_002_tradeoff",
                admitted=True,
                gate_type="PDS",
                delta_r=6.9,
                delta_n=(-10.3, 17.2),
                margin=1.6,
                failure_reason=None,
                candidate_policy="ppo_then_side_tradeoff",
                epsilon=5.0,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")
        assert "Example PDS Trade-Off Skill" in content
        assert "ppo_then_side_tradeoff" in content
        assert "CDS failed" in content

    def test_markdown_no_skills_admitted_message(self):
        report = AdmissionReport()
        report.add_record(_rejected_record())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")
        assert "No skills were admitted" in content


class TestMDNFeasibilityTelemetry:
    """The infeasible-support counter must be recorded and surfaced.

    The original support-geometry bug excluded every MDN_WX skill from runtime
    selection behind a single log line. Counting and displaying the events is
    what converts that silent failure class into a loud one.
    """

    def _report_with_mdn_metadata(self, **overrides) -> AdmissionReport:
        report = AdmissionReport()
        report.add_record(_admitted_record())
        kwargs = {
            "source": "stub",
            "checkpoint_path": "models/mdn_policy_best.pth",
            "alpha_values": [2.0, 2.0],
            "derived_weights": [0.5, 0.5],
            "support_values": [0.8, 0.4],
            "support_geometry_feasible": True,
        }
        kwargs.update(overrides)
        report.set_mdn_metadata(**kwargs)
        return report

    def test_defaults_to_zero_when_caller_omits_it(self):
        """Existing callers pass every arg by keyword and know nothing of it."""
        stats = self._report_with_mdn_metadata().compile()

        assert stats["infeasible_support_events"] == 0

    def test_records_supplied_count(self):
        stats = self._report_with_mdn_metadata(
            infeasible_support_events=3
        ).compile()

        assert stats["infeasible_support_events"] == 3

    def test_json_exposes_counter_at_top_level(self):
        """compile() flattens MDN metadata, so the key is not nested."""
        report = self._report_with_mdn_metadata(infeasible_support_events=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["infeasible_support_events"] == 0
        assert "mdn_metadata" not in payload

    def test_markdown_reports_ok_when_zero(self):
        report = self._report_with_mdn_metadata(infeasible_support_events=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        assert "Infeasible Support Events" in content
        assert "0 (OK)" in content

    def test_markdown_flags_regression_when_nonzero(self):
        report = self._report_with_mdn_metadata(infeasible_support_events=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        assert "7 (REGRESSION - investigate)" in content

    def test_markdown_omits_line_for_legacy_metadata_without_counter(self):
        """A report compiled before this field existed must still render."""
        report = AdmissionReport()
        report.add_record(_admitted_record())
        report._mdn_metadata = {
            "mdn_source": "stub",
            "checkpoint_path": "x",
            "alpha_values": [2.0, 2.0],
            "derived_weights": [0.5, 0.5],
            "support_values": [0.8, 0.4],
            "support_geometry_feasible": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        assert "MDN Selection Metadata" in content
        assert "Infeasible Support Events" not in content
