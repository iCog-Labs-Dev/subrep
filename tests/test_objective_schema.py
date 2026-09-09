"""
Tests — Named N-Dimensional Objectives.

Verifies:
1.  Named 2D schema creation (LunarLander)
2.  Named 6D Minecraft schema creation
3.  Vector/schema length validation
4.  Invalid/duplicate objective names
5.  Six-dimensional certificates
6.  Six-dimensional candidate records
7.  Six-dimensional decision records
8.  Six-dimensional simplex weights
9.  Cross-domain rejection (different domain_id)
10. Same-dimension/different-objective-schema rejection
11. Legacy artifact loading (backward compatibility)
12. Serialization/deserialization preserving schema metadata
13. Existing LunarLander behavior remains functional
14. Removed inappropriate fixed-dimension assumptions

Reference: SubRep Minecraft Environment Migration Roadmap.
"""

from __future__ import annotations

import json
from datetime import datetime
from math import isfinite

import numpy as np
import pytest

from schemas.objective_schema import (
    ObjectiveSchema,
    schemas_compatible,
    validate_vector_against_schema,
)
from schemas.minecraft_objective_schema import (
    MINECRAFT_OBJECTIVE_SCHEMA,
    LUNARLANDER_OBJECTIVE_SCHEMA,
    MINECRAFT_MOTIVE_NAMES,
)
from certification.certificate_schema import Certificate
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_record_builder import PreparedCandidateOutcome
from utils.cone_utils import validate_simplex_weights


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().isoformat()


def _make_cert(
    skill_id: str = "skill_001",
    gate_type: str = "CDS",
    delta_n: tuple = (0.3, 0.2),
    epsilon: float = 0.0,
    domain_id: str | None = None,
    motive_schema_version: str | None = None,
    motive_names: tuple[str, ...] | None = None,
) -> Certificate:
    """Build a Certificate with configurable delta_n and schema identity."""
    return Certificate(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=0.5,
        delta_n=delta_n,
        admission_margin=max(0.0, 0.5 + min(delta_n)),
        epsilon=epsilon,
        timestamp=_ts(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline_noop",
        environment="test_env",
        episode_length=100,
        version="0.1.0",
        domain_id=domain_id,
        motive_schema_version=motive_schema_version,
        motive_names=motive_names,
    )


def _make_candidate_record(
    delta_n: tuple,
    skill_id: str = "skill_001",
) -> CandidateSkillRecord:
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=0.5,
        delta_n=delta_n,
        is_certified=True,
        gate_type="CDS",
    )


def _make_decision_record(
    n_dim: int,
    actual_motives: tuple | None = None,
) -> MDNDecisionRecord:
    """Build an MDNDecisionRecord with an n_dim-dimensional alpha/weights."""
    alpha = tuple(1.0 for _ in range(n_dim))
    weights = tuple(1.0 / n_dim for _ in range(n_dim))
    support_values = tuple(1.0 for _ in range(n_dim))
    candidate = _make_candidate_record(delta_n=tuple(0.1 for _ in range(n_dim)))
    return MDNDecisionRecord(
        context=(0.5, 0.3),
        alpha=alpha,
        support_values=support_values,
        weights_used=weights,
        candidate_skills=(candidate,),
        selected_skill_id="skill_001",
        actual_motives=actual_motives,
    )


# ===========================================================================
# 1. Named 2D schema creation (LunarLander)
# ===========================================================================

class TestNamedTwoDimensionalSchema:

    def test_lunarlander_schema_creation(self):
        schema = ObjectiveSchema(
            domain_id="lunarlander",
            motive_schema_version="1.0",
            motive_names=("Safety", "Fuel"),
        )
        assert schema.domain_id == "lunarlander"
        assert schema.motive_schema_version == "1.0"
        assert schema.motive_names == ("Safety", "Fuel")
        assert schema.n_objectives == 2

    def test_canonical_lunarlander_constant(self):
        assert LUNARLANDER_OBJECTIVE_SCHEMA.domain_id == "lunarlander"
        assert LUNARLANDER_OBJECTIVE_SCHEMA.n_objectives == 2
        assert "Safety" in LUNARLANDER_OBJECTIVE_SCHEMA.motive_names
        assert "Fuel" in LUNARLANDER_OBJECTIVE_SCHEMA.motive_names

    def test_lunarlander_schema_is_hashable(self):
        schema = LUNARLANDER_OBJECTIVE_SCHEMA
        # Frozen dataclass must be usable as a dict key.
        d = {schema: "ok"}
        assert d[schema] == "ok"


# ===========================================================================
# 2. Named 6D Minecraft schema creation
# ===========================================================================

class TestNamedSixDimensionalSchema:

    def test_minecraft_schema_creation(self):
        schema = ObjectiveSchema(
            domain_id="minecraft_village_defense_trade",
            motive_schema_version="1.0",
            motive_names=(
                "Safety", "Reputation", "DeadlineSlack",
                "InventoryValue", "Sustainability", "Infrastructure",
            ),
        )
        assert schema.n_objectives == 6
        assert schema.motive_names[0] == "Safety"
        assert schema.motive_names[5] == "Infrastructure"

    def test_canonical_minecraft_schema_constant(self):
        assert MINECRAFT_OBJECTIVE_SCHEMA.domain_id == "minecraft_village_defense_trade"
        assert MINECRAFT_OBJECTIVE_SCHEMA.n_objectives == 6
        assert MINECRAFT_OBJECTIVE_SCHEMA.motive_names == (
            "Safety", "Reputation", "DeadlineSlack",
            "InventoryValue", "Sustainability", "Infrastructure",
        )

    def test_minecraft_motive_names_canonical_order(self):
        """Canonical motive order must match roadmap Section 8 exactly."""
        assert MINECRAFT_MOTIVE_NAMES == (
            "Safety",
            "Reputation",
            "DeadlineSlack",
            "InventoryValue",
            "Sustainability",
            "Infrastructure",
        )

    def test_minecraft_schema_is_hashable(self):
        d = {MINECRAFT_OBJECTIVE_SCHEMA: "mc"}
        assert d[MINECRAFT_OBJECTIVE_SCHEMA] == "mc"


# ===========================================================================
# 3. Vector / schema length validation
# ===========================================================================

class TestVectorSchemaLengthValidation:

    def test_validate_vector_correct_length_passes(self):
        MINECRAFT_OBJECTIVE_SCHEMA.validate_vector([0.1] * 6)  # no exception

    def test_validate_vector_wrong_length_raises(self):
        with pytest.raises(ValueError, match="expects 6"):
            MINECRAFT_OBJECTIVE_SCHEMA.validate_vector([0.1, 0.2])

    def test_validate_vector_2d_schema(self):
        LUNARLANDER_OBJECTIVE_SCHEMA.validate_vector([0.3, 0.7])  # ok
        with pytest.raises(ValueError):
            LUNARLANDER_OBJECTIVE_SCHEMA.validate_vector([0.3, 0.3, 0.4])

    def test_validate_vector_against_schema_helper_none_skips(self):
        # None schema means unknown/legacy — never raises.
        validate_vector_against_schema([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], None)

    def test_validate_vector_against_schema_helper_enforces_length(self):
        with pytest.raises(ValueError):
            validate_vector_against_schema(
                [0.1, 0.2],
                MINECRAFT_OBJECTIVE_SCHEMA,
            )


# ===========================================================================
# 4. Invalid / duplicate objective names
# ===========================================================================

class TestInvalidObjectiveNames:

    def test_empty_motive_names_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ObjectiveSchema(
                domain_id="test",
                motive_schema_version="1.0",
                motive_names=(),
            )

    def test_duplicate_motive_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            ObjectiveSchema(
                domain_id="test",
                motive_schema_version="1.0",
                motive_names=("Safety", "Safety"),
            )

    def test_empty_string_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ObjectiveSchema(
                domain_id="test",
                motive_schema_version="1.0",
                motive_names=("Safety", ""),
            )

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ObjectiveSchema(
                domain_id="test",
                motive_schema_version="1.0",
                motive_names=("Safety", "   "),
            )

    def test_empty_domain_id_raises(self):
        with pytest.raises(ValueError, match="domain_id"):
            ObjectiveSchema(
                domain_id="",
                motive_schema_version="1.0",
                motive_names=("Safety",),
            )

    def test_empty_schema_version_raises(self):
        with pytest.raises(ValueError, match="motive_schema_version"):
            ObjectiveSchema(
                domain_id="test",
                motive_schema_version="",
                motive_names=("Safety",),
            )


# ===========================================================================
# 5. Six-dimensional certificates
# ===========================================================================

class TestSixDimensionalCertificates:

    def test_6d_cds_certificate_creates(self):
        delta_n_6d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        cert = _make_cert(delta_n=delta_n_6d)
        assert len(cert.delta_n) == 6
        assert all(isfinite(v) for v in cert.delta_n)

    def test_6d_pds_certificate_creates(self):
        delta_n_6d = (-0.05, 0.2, 0.3, 0.4, 0.5, 0.6)
        cert = _make_cert(
            gate_type="PDS",
            delta_n=delta_n_6d,
            epsilon=0.1,
        )
        assert len(cert.delta_n) == 6
        assert cert.epsilon == 0.1

    def test_6d_certificate_with_schema_identity(self):
        delta_n_6d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        cert = _make_cert(
            delta_n=delta_n_6d,
            domain_id="minecraft_village_defense_trade",
            motive_schema_version="1.0",
            motive_names=MINECRAFT_MOTIVE_NAMES,
        )
        assert cert.domain_id == "minecraft_village_defense_trade"
        assert cert.motive_schema_version == "1.0"
        assert cert.motive_names == MINECRAFT_MOTIVE_NAMES

    def test_6d_certificate_rejects_empty_delta_n(self):
        with pytest.raises((ValueError, TypeError)):
            _make_cert(delta_n=())

    def test_6d_certificate_rejects_non_finite(self):
        with pytest.raises(ValueError):
            _make_cert(delta_n=(0.1, float("inf"), 0.3, 0.4, 0.5, 0.6))

    def test_6d_certificate_motive_names_length_must_match_delta_n(self):
        """motive_names length must match delta_n length when provided."""
        with pytest.raises(ValueError, match="motive_names length"):
            _make_cert(
                delta_n=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                motive_names=("Safety", "Fuel"),  # length mismatch
            )

    def test_6d_certificate_duplicate_motive_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _make_cert(
                delta_n=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                motive_names=("A", "B", "C", "A", "E", "F"),
            )


# ===========================================================================
# 6. Six-dimensional candidate records
# ===========================================================================

class TestSixDimensionalCandidateRecords:

    def test_6d_candidate_record_creates(self):
        delta_n_6d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        rec = _make_candidate_record(delta_n=delta_n_6d)
        assert len(rec.delta_n) == 6

    def test_6d_candidate_rejects_empty_delta_n(self):
        with pytest.raises(ValueError, match="non-empty"):
            _make_candidate_record(delta_n=())

    def test_6d_candidate_rejects_non_finite(self):
        with pytest.raises(ValueError):
            _make_candidate_record(delta_n=(0.1, float("nan"), 0.3, 0.4, 0.5, 0.6))

    def test_6d_prepared_candidate_outcome_creates(self):
        outcome = PreparedCandidateOutcome(
            context=(0.1, 0.2, 0.3),
            skill_id="torch_corridor",
            payoff=0.8,
            motives=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4),
        )
        assert len(outcome.motives) == 6

    def test_prepared_candidate_rejects_empty_motives(self):
        with pytest.raises(ValueError):
            PreparedCandidateOutcome(
                context=(0.1,),
                skill_id="s",
                payoff=0.0,
                motives=(),
            )


# ===========================================================================
# 7. Six-dimensional decision records
# ===========================================================================

class TestSixDimensionalDecisionRecords:

    def test_6d_decision_record_creates(self):
        rec = _make_decision_record(n_dim=6)
        assert len(rec.alpha) == 6
        assert len(rec.weights_used) == 6

    def test_6d_decision_record_with_actual_motives(self):
        actual = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
        rec = _make_decision_record(n_dim=6, actual_motives=actual)
        assert rec.actual_motives == actual
        assert len(rec.actual_motives) == 6

    def test_6d_decision_record_actual_motives_wrong_dim_raises(self):
        """actual_motives length must match alpha length."""
        with pytest.raises(ValueError, match="actual_motives length"):
            _make_decision_record(n_dim=6, actual_motives=(0.1, 0.2))


# ===========================================================================
# 8. Six-dimensional simplex weights
# ===========================================================================

class TestSixDimensionalSimplexWeights:

    def test_uniform_6d_simplex_weights_valid(self):
        w = np.full(6, 1.0 / 6)
        assert validate_simplex_weights(w)

    def test_corner_6d_simplex_weight_valid(self):
        w = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert validate_simplex_weights(w)

    def test_6d_weights_not_summing_to_one_invalid(self):
        w = np.full(6, 0.1)  # sum = 0.6
        assert not validate_simplex_weights(w)

    def test_6d_certificate_store_query_accepts_6d_weights(self):
        pytest.importorskip("hyperon")
        from certification.metta_storage import CertificateStore

        store = CertificateStore()
        cert = _make_cert(delta_n=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        store.add(cert)

        # 6D query — should not raise the old "must be length-2" error.
        w6 = [1.0 / 6] * 6
        results = store.query_by_weights(w6)
        assert len(results) >= 1

    def test_certificate_store_query_rejects_non_simplex_weights(self):
        pytest.importorskip("hyperon")
        from certification.metta_storage import CertificateStore

        store = CertificateStore()
        with pytest.raises(ValueError):
            store.query_by_weights([0.5, 0.3, 0.1])  # does not sum to 1


# ===========================================================================
# 9. Cross-domain rejection
# ===========================================================================

class TestCrossDomainRejection:

    def test_schemas_compatible_same_schema(self):
        assert schemas_compatible(LUNARLANDER_OBJECTIVE_SCHEMA, LUNARLANDER_OBJECTIVE_SCHEMA)
        assert schemas_compatible(MINECRAFT_OBJECTIVE_SCHEMA, MINECRAFT_OBJECTIVE_SCHEMA)

    def test_schemas_incompatible_different_domain(self):
        assert not schemas_compatible(LUNARLANDER_OBJECTIVE_SCHEMA, MINECRAFT_OBJECTIVE_SCHEMA)

    def test_schemas_incompatible_different_names_same_dim(self):
        """Same dimension but different motive names → incompatible."""
        lunar = ObjectiveSchema("d", "1.0", ("Safety", "Fuel"))
        other = ObjectiveSchema("d", "1.0", ("Safety", "Reputation"))
        assert not schemas_compatible(lunar, other)

    def test_schemas_incompatible_different_version(self):
        a = ObjectiveSchema("mc", "1.0", ("Safety",))
        b = ObjectiveSchema("mc", "2.0", ("Safety",))
        assert not schemas_compatible(a, b)

    def test_library_query_by_schema_filters_incompatible(self):
        """query_by_schema must exclude skills from the wrong domain."""
        from library.skill_library import SkillLibrary
        # Import directly from module, not via __init__ which pulls torch.

        lib = SkillLibrary()

        # Add a 2D LunarLander skill
        lunar_cert = _make_cert(
            skill_id="lunar_skill",
            delta_n=(0.3, 0.2),
            domain_id="lunarlander",
            motive_schema_version="1.0",
            motive_names=("Safety", "Fuel"),
        )
        lib.add_skill("lunar_skill", lunar_cert, policy=lambda obs: 0)

        # Add a 6D Minecraft skill
        mc_cert = _make_cert(
            skill_id="mc_skill",
            delta_n=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            domain_id="minecraft_village_defense_trade",
            motive_schema_version="1.0",
            motive_names=MINECRAFT_MOTIVE_NAMES,
        )
        lib.add_skill("mc_skill", mc_cert, policy=lambda obs: 0)

        mc_skills = lib.query_by_schema(MINECRAFT_OBJECTIVE_SCHEMA)
        mc_ids = {s.skill_id for s in mc_skills}
        assert "mc_skill" in mc_ids
        assert "lunar_skill" not in mc_ids

        lunar_skills = lib.query_by_schema(LUNARLANDER_OBJECTIVE_SCHEMA)
        lunar_ids = {s.skill_id for s in lunar_skills}
        assert "lunar_skill" in lunar_ids
        assert "mc_skill" not in lunar_ids


# ===========================================================================
# 10. Same-dimension / different-objective-schema rejection
# ===========================================================================

class TestSameDimDifferentSchemaRejection:

    def test_same_dim_different_names_not_compatible(self):
        """2D [Safety, Fuel] vs 2D [Safety, Reputation] must be incompatible."""
        sch_a = ObjectiveSchema("d", "1.0", ("Safety", "Fuel"))
        sch_b = ObjectiveSchema("d", "1.0", ("Safety", "Reputation"))
        assert not schemas_compatible(sch_a, sch_b)

    def test_6d_different_names_same_dim_not_compatible(self):
        sch_a = ObjectiveSchema("mc", "1.0", ("A", "B", "C", "D", "E", "F"))
        sch_b = ObjectiveSchema("mc", "1.0", ("A", "B", "C", "D", "E", "X"))
        assert not schemas_compatible(sch_a, sch_b)

    def test_library_filters_same_dim_different_schema(self):
        """Library must exclude same-dim skills from a different objective schema."""
        from library.skill_library import SkillLibrary
        # Import directly from module, not via __init__ which pulls torch.

        lib = SkillLibrary()

        cert_fuel = _make_cert(
            skill_id="fuel_skill",
            delta_n=(0.3, 0.2),
            domain_id="lunarlander",
            motive_schema_version="1.0",
            motive_names=("Safety", "Fuel"),
        )
        lib.add_skill("fuel_skill", cert_fuel, policy=lambda obs: 0)

        cert_rep = _make_cert(
            skill_id="rep_skill",
            delta_n=(0.3, 0.2),
            domain_id="lunarlander",
            motive_schema_version="1.0",
            motive_names=("Safety", "Reputation"),
        )
        lib.add_skill("rep_skill", cert_rep, policy=lambda obs: 0)

        fuel_schema = ObjectiveSchema("lunarlander", "1.0", ("Safety", "Fuel"))
        result = lib.query_by_schema(fuel_schema)
        result_ids = {s.skill_id for s in result}
        assert "fuel_skill" in result_ids
        assert "rep_skill" not in result_ids


# ===========================================================================
# 11. Legacy 2D artifact loading (backward compatibility)
# ===========================================================================

class TestLegacyArtifactBackwardCompatibility:

    def test_legacy_2d_cert_from_dict_no_schema_fields(self):
        """Certificates without schema identity fields must load successfully."""
        legacy_dict = {
            "skill_id": "legacy_01",
            "gate_type": "CDS",
            "delta_r": 0.5,
            "delta_n": [0.3, 0.2],  # 2D legacy
            "admission_margin": 0.7,
            "epsilon": 0.0,
            "timestamp": datetime.now().isoformat(),
            "seed": 42,
            "gamma": 0.99,
            "baseline_id": "idle",
            "environment": "MO-LunarLander-v2",
            "episode_length": 200,
            "version": "0.1.0",
            # No domain_id, motive_schema_version, motive_names
        }
        cert = Certificate.from_dict(legacy_dict)
        assert cert.skill_id == "legacy_01"
        assert len(cert.delta_n) == 2
        assert cert.domain_id is None
        assert cert.motive_schema_version is None
        assert cert.motive_names is None

    def test_legacy_2d_cert_is_accepted_by_schema_compatible_check(self):
        """None schema (legacy) is always compatible."""
        cert = Certificate.from_dict({
            "skill_id": "leg2",
            "gate_type": "CDS",
            "delta_r": 0.2,
            "delta_n": [0.1, 0.1],
            "admission_margin": 0.3,
            "epsilon": 0.0,
            "timestamp": datetime.now().isoformat(),
            "seed": 1,
            "gamma": 0.9,
            "baseline_id": "idle",
            "environment": "MO-LunarLander-v2",
            "episode_length": 100,
            "version": "0.1.0",
        })
        # Legacy cert has no schema fields → should appear in any query_by_schema.
        assert schemas_compatible(None, LUNARLANDER_OBJECTIVE_SCHEMA)
        assert schemas_compatible(None, MINECRAFT_OBJECTIVE_SCHEMA)

    def test_legacy_cert_in_library_appears_in_all_schema_queries(self):
        """Legacy cert (no schema) must be included in any query_by_schema."""
        from library.skill_library import SkillLibrary
        # Import directly from module, not via __init__ which pulls torch.

        lib = SkillLibrary()
        cert = _make_cert(skill_id="legacy_no_schema", delta_n=(0.3, 0.2))
        # No domain_id / schema fields set
        lib.add_skill("legacy_no_schema", cert, policy=lambda obs: 0)

        mc_result = lib.query_by_schema(MINECRAFT_OBJECTIVE_SCHEMA)
        ll_result = lib.query_by_schema(LUNARLANDER_OBJECTIVE_SCHEMA)

        # Legacy cert appears in both — schema unknown → accepted.
        mc_ids = {s.skill_id for s in mc_result}
        ll_ids = {s.skill_id for s in ll_result}
        assert "legacy_no_schema" in mc_ids
        assert "legacy_no_schema" in ll_ids


# ===========================================================================
# 12. Serialization / deserialization preserving schema metadata
# ===========================================================================

class TestSerializationWithSchemaMetadata:

    def test_6d_cert_to_dict_includes_schema_fields(self):
        delta_n_6d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        cert = _make_cert(
            delta_n=delta_n_6d,
            domain_id="minecraft_village_defense_trade",
            motive_schema_version="1.0",
            motive_names=MINECRAFT_MOTIVE_NAMES,
        )
        d = cert.to_dict()
        assert d["domain_id"] == "minecraft_village_defense_trade"
        assert d["motive_schema_version"] == "1.0"
        assert d["motive_names"] == list(MINECRAFT_MOTIVE_NAMES)
        assert len(d["delta_n"]) == 6

    def test_6d_cert_round_trips_through_dict(self):
        delta_n_6d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        cert = _make_cert(
            delta_n=delta_n_6d,
            domain_id="minecraft_village_defense_trade",
            motive_schema_version="1.0",
            motive_names=MINECRAFT_MOTIVE_NAMES,
        )
        reloaded = Certificate.from_dict(cert.to_dict())
        assert reloaded.skill_id == cert.skill_id
        assert reloaded.delta_n == cert.delta_n
        assert reloaded.domain_id == cert.domain_id
        assert reloaded.motive_schema_version == cert.motive_schema_version
        assert reloaded.motive_names == cert.motive_names

    def test_6d_cert_round_trips_through_json(self):
        delta_n_6d = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        cert = _make_cert(
            delta_n=delta_n_6d,
            domain_id="minecraft_village_defense_trade",
            motive_schema_version="1.0",
            motive_names=MINECRAFT_MOTIVE_NAMES,
        )
        json_str = json.dumps(cert.to_dict())
        loaded_dict = json.loads(json_str)
        reloaded = Certificate.from_dict(loaded_dict)
        assert reloaded.delta_n == cert.delta_n
        assert reloaded.motive_names == cert.motive_names

    def test_objective_schema_to_dict_round_trip(self):
        schema = MINECRAFT_OBJECTIVE_SCHEMA
        d = schema.to_dict()
        reloaded = ObjectiveSchema.from_dict(d)
        assert reloaded == schema

    def test_objective_schema_round_trip_through_json(self):
        schema = LUNARLANDER_OBJECTIVE_SCHEMA
        json_str = json.dumps(schema.to_dict())
        reloaded = ObjectiveSchema.from_dict(json.loads(json_str))
        assert reloaded == schema


# ===========================================================================
# 13. Existing LunarLander behavior remains functional
# ===========================================================================

class TestExistingLunarLanderBehavior:

    def test_2d_certificate_creates_without_schema_fields(self):
        """Original 2D certificates must still work with no schema fields."""
        cert = _make_cert(delta_n=(0.3, 0.2))
        assert len(cert.delta_n) == 2
        assert cert.domain_id is None

    def test_2d_cds_gate_still_passes(self):
        from certification.cds_test import CDSGate

        gate = CDSGate()
        assert gate.admit(0.5, np.array([0.3, 0.2]))

    def test_2d_pds_gate_still_passes(self):
        from certification.pds_test import PDSGate

        gate = PDSGate(epsilon=0.1)
        assert gate.admit(0.5, np.array([0.8, -0.6]))

    def test_2d_candidate_record_still_creates(self):
        rec = _make_candidate_record(delta_n=(0.3, 0.2))
        assert len(rec.delta_n) == 2

    def test_2d_decision_record_still_creates(self):
        rec = _make_decision_record(n_dim=2)
        assert len(rec.alpha) == 2

    def test_2d_improvement_calculator_still_works(self):
        from baseline.improvement_calculator import ImprovementCalculator

        calc = ImprovementCalculator({"baseline_payoff": 0.0, "baseline_motives": [0.5, 0.5]})
        delta_r, delta_n = calc.compute_improvements(0.8, [0.7, 0.6])
        assert np.isclose(delta_r, 0.8)
        assert delta_n.shape == (2,)


# ===========================================================================
# 14. Removed fixed-dimension assumptions
# ===========================================================================

class TestRemovedTwoDimensionalAssumptions:

    def test_certificate_no_longer_rejects_6d_delta_n(self):
        """Certificate must accept 6D delta_n without raising."""
        cert = _make_cert(delta_n=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        assert len(cert.delta_n) == 6

    def test_certificate_no_longer_rejects_1d_delta_n(self):
        """Any non-empty N is valid."""
        cert = _make_cert(delta_n=(0.5,))
        assert len(cert.delta_n) == 1

    def test_candidate_record_no_longer_rejects_6d_delta_n(self):
        rec = _make_candidate_record(delta_n=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        assert len(rec.delta_n) == 6

    def test_prepared_outcome_no_longer_rejects_6d_motives(self):
        outcome = PreparedCandidateOutcome(
            context=(0.5,),
            skill_id="mc_01",
            payoff=0.8,
            motives=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4),
        )
        assert len(outcome.motives) == 6

    def test_certificate_store_no_longer_rejects_6d_query_weights(self):
        """CertificateStore.query_by_weights must accept 6D simplex weights."""
        pytest.importorskip("hyperon")
        from certification.metta_storage import CertificateStore

        store = CertificateStore()
        cert = _make_cert(delta_n=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        store.add(cert)

        results = store.query_by_weights([1.0 / 6] * 6)
        assert isinstance(results, list)
