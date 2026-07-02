"""Data helpers for the evaluator-facing SubRep demo.

The Streamlit app should tell the story from real pipeline artifacts, not from
hard-coded demo numbers. This module keeps that artifact loading and MDN
selection trace logic separate from the UI.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from certification.cds_test import CDSGate
from certification.metta_storage import CertificateStore
from certification.pds_test import PDSGate
from generator.mdn_runtime_selector import MDNRuntimeSelector
from library.skill_library import SkillLibrary
from utils.mdn_stub import StubMDN, load_mdn_or_stub


REPORT_PATH = Path("demo/artifacts/admission_report.json")
LIBRARY_PATH = Path("data/library.json")
CERTIFICATE_PATH = Path("data/certificates.metta")
MDN_CHECKPOINT_PATH = Path("models/mdn_policy_best.pth")

DEFAULT_OBSERVATIONS = (
    np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    np.array([-0.1, 0.5, -0.2, 0.3, 0.1, -0.3, 0.4, 0.2], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
)


@dataclass(frozen=True)
class DemoArtifacts:
    report: dict[str, Any]
    library: dict[str, Any]
    certificate_count: int
    skill_rows: list[dict[str, Any]]
    store_synced: bool


def load_demo_artifacts(
    *,
    report_path: Path | str = REPORT_PATH,
    library_path: Path | str = LIBRARY_PATH,
    certificate_path: Path | str = CERTIFICATE_PATH,
) -> DemoArtifacts:
    """Load the core artifacts produced by ``demo.run_full_pipeline``."""
    report = load_json(report_path)
    library = load_json(library_path)
    certificate_count = count_metta_certificates(certificate_path)
    skill_rows = build_skill_rows(library)
    library_count = int(library.get("skill_count", len(skill_rows)) or 0)
    return DemoArtifacts(
        report=report,
        library=library,
        certificate_count=certificate_count,
        skill_rows=skill_rows,
        store_synced=certificate_count == library_count,
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Return parsed JSON or an empty dict when the artifact is missing."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def count_metta_certificates(path: Path | str) -> int:
    """Count serialized certificate atoms in the MeTTa certificate file."""
    file_path = Path(path)
    if not file_path.exists():
        return 0
    return sum(1 for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_skill_rows(library: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten serialized SkillLibrary data into table-friendly rows."""
    skills = library.get("skills", {}) or {}
    rows: list[dict[str, Any]] = []
    for skill_id, entry in sorted(skills.items()):
        cert = entry.get("certificate", {}) or {}
        rows.append(
            {
                "skill_id": skill_id,
                "gate_type": entry.get("gate_type", cert.get("gate_type")),
                "region": entry.get("weight_region_type", cert.get("weight_region_type", "FULL_SIMPLEX")),
                "delta_r": cert.get("delta_r"),
                "delta_n_0": _delta_n_at(cert, 0),
                "delta_n_1": _delta_n_at(cert, 1),
                "margin": cert.get("admission_margin"),
                "epsilon": cert.get("epsilon"),
            }
        )
    return rows


def build_mdn_selection_trace(
    *,
    library_path: Path | str = LIBRARY_PATH,
    checkpoint_path: Path | str = MDN_CHECKPOINT_PATH,
    observations: tuple[np.ndarray, ...] = DEFAULT_OBSERVATIONS,
) -> dict[str, Any]:
    """Run trained/stub MDN selection against the serialized skill library."""
    library_path = Path(library_path)
    checkpoint_path = Path(checkpoint_path)
    if not library_path.exists():
        return {
            "mdn_source": "missing_library",
            "checkpoint_path": str(checkpoint_path),
            "decisions": [],
            "error": f"Missing skill library at {library_path}",
        }

    library = SkillLibrary()
    library.load(str(library_path))
    if library.count() == 0:
        return {
            "mdn_source": "empty_library",
            "checkpoint_path": str(checkpoint_path),
            "decisions": [],
            "error": "Skill library is empty",
        }

    with contextlib.redirect_stdout(io.StringIO()):
        model = load_mdn_or_stub(
            checkpoint_path=checkpoint_path,
            input_dim=8,
            num_objectives=2,
        )
    mdn_source = "stub" if isinstance(model, StubMDN) else "trained_checkpoint"
    selector = MDNRuntimeSelector(model)

    decisions: list[dict[str, Any]] = []
    for index, observation in enumerate(observations, start=1):
        record: dict[str, Any] = {
            "observation": f"Context {index}",
            "context": observation.tolist(),
            "no_retraining": True,
        }
        try:
            result = selector.select_from_library(observation, library)
            record.update(
                {
                    "status": "selected",
                    "selected_skill_id": result.selected_skill_id,
                    "score": float(result.selected_score),
                    "alpha_0": _vector_at(result.alpha, 0),
                    "alpha_1": _vector_at(result.alpha, 1),
                    "weight_0": _vector_at(result.weights_used, 0),
                    "weight_1": _vector_at(result.weights_used, 1),
                    "support_0": _vector_at(result.support_values, 0),
                    "support_1": _vector_at(result.support_values, 1),
                    "admissible_count": len(result.candidate_skills),
                }
            )
        except Exception as exc:  # Keep the demo page usable if selection fails.
            record.update(
                {
                    "status": "failed",
                    "selected_skill_id": None,
                    "score": None,
                    "error": str(exc),
                }
            )
        decisions.append(record)

    return {
        "mdn_source": mdn_source,
        "checkpoint_path": str(checkpoint_path),
        "decisions": decisions,
    }


def support_geometry_feasible(values: Any) -> bool:
    """Return whether 2-objective support values define a non-empty region."""
    if values is None:
        return False
    support = np.asarray(values, dtype=np.float64).reshape(-1)
    if support.shape != (2,):
        return False
    return bool(np.all(support >= 0.0) and np.all(support <= 1.0) and float(np.sum(support)) >= 1.0)


def build_failed_skill_rejection_probe() -> dict[str, Any]:
    """Return a small live proof that an unsafe skill is blocked before storage."""
    delta_r = -0.2
    delta_n = np.array([-0.5, 0.1], dtype=np.float64)
    epsilon = 0.1

    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=epsilon)
    cds_pass = cds_gate.admit(delta_r, delta_n)
    pds_pass = pds_gate.admit(delta_r, delta_n)
    cds_margin = cds_gate.get_admission_margin(delta_r, delta_n)
    pds_margin = pds_gate.get_admission_margin(delta_r, delta_n)

    cert_store = CertificateStore()
    library = SkillLibrary(cert_store=cert_store)
    blocked = not cds_pass and not pds_pass and cert_store.count() == 0 and library.count() == 0

    return {
        "skill_id": "unsafe_probe_skill",
        "delta_r": delta_r,
        "delta_n": delta_n.tolist(),
        "epsilon": epsilon,
        "cds_pass": cds_pass,
        "pds_pass": pds_pass,
        "cds_margin": cds_margin,
        "pds_margin": pds_margin,
        "blocked": blocked,
        "cert_store_count": cert_store.count(),
        "library_count": library.count(),
        "reason": "delta_r + min(delta_n) = -0.70, which is below both the CDS threshold 0.00 and the PDS threshold -0.10.",
    }


def _delta_n_at(cert: dict[str, Any], index: int) -> float | None:
    values = cert.get("delta_n")
    if not isinstance(values, list) or len(values) <= index:
        return None
    return float(values[index])


def _vector_at(values: Any, index: int) -> float | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape[0] <= index:
        return None
    return float(array[index])
