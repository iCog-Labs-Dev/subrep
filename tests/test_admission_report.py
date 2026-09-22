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

from utils.admission_report import (
    AdmissionRecord,
    AdmissionReport,
    RejectionCategory,
    evaluate_gates,
)


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


def _admitted_record_with(skill_id: str = "skill_x", **overrides) -> AdmissionRecord:
    """An admitted record with arbitrary audit fields overridden."""
    fields = {
        "skill_id": skill_id,
        "admitted": True,
        "gate_type": "CDS",
        "delta_r": 5.0,
        "delta_n": (2.0, 3.0),
        "margin": 7.0,
        "failure_reason": None,
        "epsilon": 0.0,
    }
    fields.update(overrides)
    return AdmissionRecord(**fields)


def _rejected_record_with(category: str, skill_id: str = "skill_r", **overrides) -> AdmissionRecord:
    """A rejected record carrying a specific rejection category."""
    fields = {
        "skill_id": skill_id,
        "admitted": False,
        "gate_type": None,
        "delta_r": -1.0,
        "delta_n": (-1.0, -1.0),
        "margin": -2.0,
        "failure_reason": "rejected",
        "rejection_category": category,
    }
    fields.update(overrides)
    return AdmissionRecord(**fields)


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


# ── Tests: full certification audit ───────────────────────────────────────────

class TestGateEvaluation:
    """The exact CDS/PDS inequality values must be recorded, not just described."""

    def test_full_simplex_inequality_values(self):
        cds, pds = evaluate_gates(delta_r=0.5, delta_n=[0.3, 0.2], epsilon=0.1)

        assert cds.gate == "CDS" and pds.gate == "PDS"
        # Both gates compare the same left-hand side; only the threshold differs.
        assert cds.lhs == pytest.approx(0.7)   # delta_r + min(delta_n)
        assert pds.lhs == pytest.approx(0.7)
        assert cds.rhs == pytest.approx(0.0)
        assert pds.rhs == pytest.approx(-0.1)  # -epsilon
        assert cds.satisfied is True and pds.satisfied is True

    def test_records_a_failing_inequality(self):
        cds, pds = evaluate_gates(delta_r=-1.0, delta_n=[0.3, 0.2], epsilon=0.1)

        assert cds.lhs == pytest.approx(-0.8)
        assert cds.satisfied is False
        assert pds.satisfied is False

    def test_pds_admits_inside_epsilon_where_cds_fails(self):
        """The bounded trade-off case must show two different verdicts."""
        cds, pds = evaluate_gates(delta_r=0.0, delta_n=[0.5, -0.05], epsilon=0.1)

        assert cds.satisfied is False
        assert pds.satisfied is True

    def test_mdn_wx_uses_the_exact_greedy_support_function(self):
        """The W_x branch must defer to the shared solver, not re-derive it."""
        from utils.support_geometry import worst_case_over_support_region

        delta_n = [-0.2, 0.1]
        support_values = [0.8, 0.4]
        cds, _ = evaluate_gates(
            delta_r=0.5, delta_n=delta_n, epsilon=0.1, support_values=support_values
        )

        expected = 0.5 + worst_case_over_support_region(delta_n, support_values)
        assert cds.lhs == pytest.approx(expected, abs=1e-12)

    def test_mdn_wx_differs_from_full_simplex(self):
        """A restricted region must give a different worst case than the simplex."""
        delta_n = [-0.2, 0.1]
        simplex, _ = evaluate_gates(delta_r=0.5, delta_n=delta_n, epsilon=0.0)
        restricted, _ = evaluate_gates(
            delta_r=0.5, delta_n=delta_n, epsilon=0.0, support_values=[0.8, 0.4]
        )

        assert simplex.lhs != pytest.approx(restricted.lhs)

    def test_evaluate_gates_rejects_empty_delta_n(self):
        with pytest.raises(ValueError, match="delta_n"):
            evaluate_gates(delta_r=0.5, delta_n=[], epsilon=0.0)

    def test_evaluate_gates_supports_more_than_two_objectives(self):
        cds, _ = evaluate_gates(delta_r=1.0, delta_n=[0.4, 0.2, 0.9, 0.7, 0.1], epsilon=0.0)

        assert cds.lhs == pytest.approx(1.1)  # 1.0 + min(...) = 1.0 + 0.1


class TestPerCandidateAuditEntries:
    """Every evaluated candidate must appear, not only three examples."""

    def test_every_record_appears_in_audit_entries(self):
        report = AdmissionReport()
        for index in range(10):
            report.add_record(_admitted_record(f"skill_{index}"))
        report.add_record(_rejected_record("skill_bad"))

        stats = report.compile()

        # The defect this task fixes: compile() used to emit 3 records total.
        assert len(stats["audit_entries"]) == 11
        assert stats["audit_entries"][0]["skill_id"] == "skill_0"
        assert len(stats["audit_entries"]) == stats["total_attempted"]

    def test_audit_entries_carry_the_full_field_set(self):
        report = AdmissionReport()
        report.add_record(
            _admitted_record_with(
                skill_id="full",
                candidate_policy="ppo_deterministic",
                gate_evaluations=evaluate_gates(5.0, [2.0, 3.0], 0.0),
                baseline_id="idle_policy_v1",
                environment="MO-LunarLander-v3",
                seed=42,
                episode_length=167,
            )
        )

        entry = report.compile()["audit_entries"][0]

        assert entry["baseline_id"] == "idle_policy_v1"
        assert entry["environment"] == "MO-LunarLander-v3"
        assert entry["seed"] == 42
        assert entry["episode_length"] == 167
        assert entry["weight_region_type"] == "FULL_SIMPLEX"
        assert len(entry["gate_evaluations"]) == 2

    def test_rejected_record_still_records_both_gates(self):
        """Which gates were evaluated must survive rejection.

        `gate_type` intentionally stays None when rejected -- existing callers
        depend on that -- so `gate_evaluations` is what preserves the evidence.
        """
        report = AdmissionReport()
        report.add_record(
            _rejected_record_with(
                RejectionCategory.GATE_FAILED,
                skill_id="rejected",
                delta_r=-1.0,
                delta_n=(0.3, 0.2),
                margin=-0.8,
                gate_evaluations=evaluate_gates(-1.0, [0.3, 0.2], 0.1),
            )
        )

        entry = report.compile()["audit_entries"][0]

        assert entry["gate_type"] is None          # backward compatible
        gates = {e["gate"] for e in entry["gate_evaluations"]}
        assert gates == {"CDS", "PDS"}             # but the evidence is kept
        assert all(e["satisfied"] is False for e in entry["gate_evaluations"])

    def test_audit_entries_are_json_serializable(self):
        report = AdmissionReport()
        report.add_record(
            _admitted_record_with(gate_evaluations=evaluate_gates(5.0, [2.0, 3.0], 0.0))
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        # Nested dataclasses must flatten to plain dicts.
        assert payload["audit_entries"][0]["gate_evaluations"][0]["gate"] == "CDS"

    def test_markdown_shows_the_failed_condition_with_both_sides(self):
        report = AdmissionReport()
        report.add_record(
            _rejected_record_with(
                RejectionCategory.GATE_FAILED,
                delta_r=-1.0,
                delta_n=(0.3, 0.2),
                gate_evaluations=evaluate_gates(-1.0, [0.3, 0.2], 0.1),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        # The numbers the gate compared, not just a description.
        assert "-0.8000" in content
        assert "delta_r + min(delta_n) >= 0" in content


class TestMDNWXAuditFields:
    """MDN_WX decisions must carry their support-region details."""

    def _wx_record(self) -> AdmissionRecord:
        return _admitted_record_with(
            skill_id="wx_skill",
            gate_type="PDS",
            delta_r=0.5,
            delta_n=(-0.2, 0.1),
            margin=0.4,
            candidate_policy="mdn_contextual",
            epsilon=0.1,
            weight_region_type="MDN_WX",
            support_values=(0.8, 0.4),
            support_feasible=True,
            gate_evaluations=evaluate_gates(
                0.5, [-0.2, 0.1], 0.1, support_values=[0.8, 0.4]
            ),
        )

    def test_support_values_round_trip_through_json(self):
        report = AdmissionReport()
        report.add_record(self._wx_record())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        entry = payload["audit_entries"][0]
        assert entry["weight_region_type"] == "MDN_WX"
        assert entry["support_values"] == [0.8, 0.4]
        assert entry["support_feasible"] is True

    def test_markdown_renders_a_support_geometry_section(self):
        report = AdmissionReport()
        report.add_record(self._wx_record())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        assert "MDN_WX Support Geometry" in content
        assert "wx_skill" in content

    def test_full_simplex_record_omits_the_support_section(self):
        report = AdmissionReport()
        report.add_record(_admitted_record())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        assert "MDN_WX Support Geometry" not in content

    def test_support_fields_default_to_none_for_existing_callers(self):
        """A caller written against the old schema must still work."""
        report = AdmissionReport()
        report.add_record(_admitted_record())

        entry = report.compile()["audit_entries"][0]
        assert entry["weight_region_type"] == "FULL_SIMPLEX"
        assert entry["support_values"] is None
        assert entry["support_feasible"] is None


class TestRejectionCategories:
    """Rejections must group by cause, not by message text."""

    def test_same_category_with_different_messages_groups_together(self):
        report = AdmissionReport()
        for index, message in enumerate(["score=-0.4000", "score=-9.1234"]):
            report.add_record(
                _rejected_record_with(
                    RejectionCategory.GATE_FAILED,
                    skill_id=f"r{index}",
                    failure_reason=message,
                )
            )

        stats = report.compile()

        # Two distinct messages, but a single cause.
        assert stats["rejection_categories"] == {"GATE_FAILED": 2}
        assert len(stats["failure_reasons"]) == 2  # legacy grouping unchanged

    def test_distinct_categories_are_counted_separately(self):
        report = AdmissionReport()
        report.add_record(_rejected_record_with(RejectionCategory.GATE_FAILED, "a"))
        report.add_record(_rejected_record_with(RejectionCategory.DUPLICATE_SKILL_ID, "b"))
        report.add_record(
            _rejected_record_with(RejectionCategory.LIBRARY_REVERIFICATION_FAILED, "c")
        )

        assert report.compile()["rejection_categories"] == {
            "GATE_FAILED": 1,
            "DUPLICATE_SKILL_ID": 1,
            "LIBRARY_REVERIFICATION_FAILED": 1,
        }

    def test_uncategorized_rejections_are_not_dropped(self):
        report = AdmissionReport()
        report.add_record(_rejected_record("legacy"))

        assert report.compile()["rejection_categories"] == {"UNCATEGORIZED": 1}


class TestSummaryTables:
    """Summary tables by policy and by weight region."""

    def test_by_policy_totals_reconcile(self):
        report = AdmissionReport()
        report.add_record(_admitted_record_with("a", candidate_policy="ppo"))
        report.add_record(_admitted_record_with("b", candidate_policy="ppo", gate_type="PDS"))
        report.add_record(
            _rejected_record_with(RejectionCategory.GATE_FAILED, "c", candidate_policy="random")
        )

        stats = report.compile()
        by_policy = stats["by_policy"]

        assert by_policy["ppo"] == {
            "attempted": 2, "admitted": 2, "rejected": 0, "cds": 1, "pds": 1
        }
        assert by_policy["random"]["rejected"] == 1
        assert sum(v["attempted"] for v in by_policy.values()) == stats["total_attempted"]

    def test_by_weight_region_totals_reconcile(self):
        report = AdmissionReport()
        report.add_record(_admitted_record())
        report.add_record(_admitted_record_with("wx", weight_region_type="MDN_WX"))

        stats = report.compile()
        by_region = stats["by_weight_region"]

        assert by_region["FULL_SIMPLEX"]["attempted"] == 1
        assert by_region["MDN_WX"]["attempted"] == 1
        assert sum(v["attempted"] for v in by_region.values()) == stats["total_attempted"]

    def test_markdown_contains_all_new_sections(self):
        report = AdmissionReport()
        report.add_record(_admitted_record_with("a", candidate_policy="ppo"))
        report.add_record(_rejected_record_with(RejectionCategory.GATE_FAILED, "r"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        for heading in (
            "## Per-Candidate Audit",
            "## Summary by Candidate Policy",
            "## Summary by Weight Region",
            "## Rejection Categories",
        ):
            assert heading in content, heading


class TestMultiObjectiveRecords:
    """Records must stay readable and inspectable beyond two objectives."""

    @pytest.mark.parametrize("num_objectives", [2, 3, 5, 10])
    def test_full_delta_n_survives_compile_json_and_markdown(self, num_objectives):
        delta_n = tuple(float(i) + 0.5 for i in range(num_objectives))
        report = AdmissionReport()
        report.add_record(
            _admitted_record_with(
                skill_id="multi",
                delta_r=10.0,
                delta_n=delta_n,
                margin=10.5,
                candidate_policy="ppo",
                gate_evaluations=evaluate_gates(10.0, list(delta_n), 0.0),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "report.json"
            md_path = Path(tmpdir) / "report.md"
            report.save_json(json_path)
            report.save_markdown(md_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            content = md_path.read_text(encoding="utf-8")

        # No truncation to the first two components.
        assert tuple(payload["audit_entries"][0]["delta_n"]) == delta_n
        assert len(payload["audit_entries"][0]["delta_n"]) == num_objectives
        # Every component is rendered in the Markdown table.
        for value in delta_n:
            assert f"{value:.4f}" in content

    def test_mdn_wx_support_values_survive_at_five_objectives(self):
        support_values = (0.5, 0.5, 0.5, 0.5, 0.5)
        delta_n = (0.3, 0.4, 0.5, 0.2, 0.6)
        report = AdmissionReport()
        report.add_record(
            _admitted_record_with(
                skill_id="wx_m5",
                delta_r=0.9,
                delta_n=delta_n,
                margin=1.1,
                weight_region_type="MDN_WX",
                support_values=support_values,
                support_feasible=True,
                gate_evaluations=evaluate_gates(
                    0.9, list(delta_n), 0.0, support_values=list(support_values)
                ),
            )
        )

        entry = report.compile()["audit_entries"][0]
        assert tuple(entry["support_values"]) == support_values
        assert len(entry["delta_n"]) == 5


class TestAuditIsAdditive:
    """The new audit must not displace anything the report already produced."""

    def test_existing_json_keys_all_survive(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("s1"))
        report.add_record(_admitted_record("s2", gate_type="PDS", epsilon=5.0))
        report.add_record(_rejected_record("s3"))

        stats = report.compile()

        for key in (
            "total_attempted", "admitted", "rejected", "admission_rate",
            "cds_pass_count", "pds_pass_count", "failure_reasons",
            "example_admitted_skill", "example_pds_skill", "example_rejected_skill",
        ):
            assert key in stats, f"{key} was dropped -- the additive rule is broken"

    def test_existing_markdown_sections_all_survive(self):
        report = AdmissionReport()
        report.add_record(_admitted_record("s1", gate_type="PDS", epsilon=5.0))
        report.add_record(_rejected_record("s2"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)
            content = path.read_text(encoding="utf-8")

        for heading in (
            "## Summary Statistics",
            "## Rejection Failure Reasons",
            "## Example Admitted Skill",
            "## Example PDS Trade-Off Skill",
            "## Example Rejected Skill",
        ):
            assert heading in content, heading

    def test_nine_field_constructor_still_works(self):
        """The pre-Task-10 constructor signature must remain valid."""
        record = AdmissionRecord(
            skill_id="legacy",
            admitted=True,
            gate_type="CDS",
            delta_r=1.0,
            delta_n=(0.5, 0.5),
            margin=1.5,
            failure_reason=None,
            candidate_policy="ppo",
            epsilon=0.0,
        )

        assert record.weight_region_type == "FULL_SIMPLEX"
        assert record.gate_evaluations == ()
        assert record.rejection_category is None

    def test_add_from_dict_accepts_the_original_key_set(self):
        """A producer supplying only the original keys must keep working.

        This is what allows other AdmissionReport producers to remain untouched.
        """
        report = AdmissionReport()
        report.add_from_dict(
            {
                "skill_id": "legacy_dict",
                "admitted": True,
                "gate_type": "CDS",
                "delta_r": 1.0,
                "delta_n": (0.5, 0.5),
                "margin": 1.5,
                "failure_reason": None,
                "candidate_policy": "ppo",
                "epsilon": 0.0,
            }
        )

        entry = report.compile()["audit_entries"][0]
        assert entry["skill_id"] == "legacy_dict"
        assert entry["weight_region_type"] == "FULL_SIMPLEX"
        assert entry["support_values"] is None
