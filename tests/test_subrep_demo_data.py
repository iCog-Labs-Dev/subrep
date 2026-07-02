from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from certification.certificate_schema import Certificate
from library.skill_library import SkillLibrary
from utils.subrep_demo_data import (
    build_mdn_selection_trace,
    build_skill_rows,
    count_metta_certificates,
    load_demo_artifacts,
    support_geometry_feasible,
)


def _certificate(skill_id: str = "skill_a") -> Certificate:
    return Certificate(
        skill_id=skill_id,
        gate_type="CDS",
        delta_r=1.0,
        delta_n=(0.5, 0.2),
        admission_margin=1.2,
        epsilon=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="idle_policy_v1",
        environment="MO-LunarLander-v3",
        episode_length=100,
        version="0.1.0",
    )


def test_load_demo_artifacts_counts_and_flattens_rows(tmp_path: Path):
    report_path = tmp_path / "admission_report.json"
    library_path = tmp_path / "library.json"
    certificate_path = tmp_path / "certificates.metta"

    report_path.write_text(
        json.dumps({"total_attempted": 1, "admitted": 1, "rejected": 0}),
        encoding="utf-8",
    )
    library = SkillLibrary(save_path=str(library_path))
    cert = _certificate("skill_a")
    assert library.add_skill("skill_a", cert, lambda obs: 0)
    library.save(str(library_path))
    certificate_path.write_text("(Certificate (skill_id skill_a))\n", encoding="utf-8")

    artifacts = load_demo_artifacts(
        report_path=report_path,
        library_path=library_path,
        certificate_path=certificate_path,
    )

    assert artifacts.report["admitted"] == 1
    assert artifacts.certificate_count == 1
    assert artifacts.store_synced is True
    assert artifacts.skill_rows[0]["skill_id"] == "skill_a"
    assert artifacts.skill_rows[0]["delta_n_0"] == 0.5


def test_build_mdn_selection_trace_uses_stub_when_checkpoint_missing(tmp_path: Path):
    library_path = tmp_path / "library.json"
    checkpoint_path = tmp_path / "missing_mdn.pth"

    library = SkillLibrary(save_path=str(library_path))
    cert = _certificate("skill_a")
    assert library.add_skill("skill_a", cert, lambda obs: 0)
    library.save(str(library_path))

    trace = build_mdn_selection_trace(
        library_path=library_path,
        checkpoint_path=checkpoint_path,
        observations=(np.zeros(8, dtype=np.float32),),
    )

    assert trace["mdn_source"] == "stub"
    assert len(trace["decisions"]) == 1
    assert trace["decisions"][0]["status"] == "selected"
    assert trace["decisions"][0]["selected_skill_id"] == "skill_a"
    assert trace["decisions"][0]["no_retraining"] is True


def test_support_geometry_feasible_for_two_objective_interval():
    assert support_geometry_feasible([0.8, 0.4]) is True
    assert support_geometry_feasible([1.2, 0.4]) is False
    assert support_geometry_feasible([0.4, 0.4]) is False


def test_count_metta_certificates_missing_file_is_zero(tmp_path: Path):
    assert count_metta_certificates(tmp_path / "missing.metta") == 0


def test_build_skill_rows_handles_empty_library():
    assert build_skill_rows({}) == []
