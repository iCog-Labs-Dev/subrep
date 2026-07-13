from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from certification.certificate_schema import Certificate
from library.skill_library import SkillLibrary
from library.skill_metadata import FULL_SIMPLEX, MDN_WX
from utils.mid_episode_reuse_demo import demonstrate_mid_episode_motive_shift


STANDARD_DIRECTIONS = ((1.0, 0.0), (0.0, 1.0))


def _certificate(
    *,
    skill_id: str,
    delta_r: float,
    delta_n: tuple[float, float],
    weight_region_type: str = FULL_SIMPLEX,
    wx_support_values: tuple[float, float] | None = None,
) -> Certificate:
    mdn_fields = {}
    if weight_region_type == MDN_WX:
        mdn_fields = {
            "certification_context": (0.0,) * 14,
            "mdn_alpha": (1.0, 1.0),
            "wx_support_directions": STANDARD_DIRECTIONS,
            "wx_support_values": wx_support_values,
        }

    return Certificate(
        skill_id=skill_id,
        gate_type="CDS",
        delta_r=delta_r,
        delta_n=delta_n,
        admission_margin=max(0.0, delta_r + min(delta_n)),
        epsilon=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="idle_v1",
        environment="MO-LunarLander-v3",
        episode_length=200,
        version="0.1.0",
        weight_region_type=weight_region_type,
        **mdn_fields,
    )


def _library_with_reuse_and_rejection() -> SkillLibrary:
    library = SkillLibrary()

    global_cert = _certificate(
        skill_id="global-certified-reuse",
        delta_r=0.30,
        delta_n=(0.10, 0.20),
    )
    assert library.add_skill(
        "global-certified-reuse",
        global_cert,
        policy=lambda _obs: 0,
        weight_region_type=FULL_SIMPLEX,
    )

    contextual_cert = _certificate(
        skill_id="contextual-certified-before-shift",
        delta_r=0.09,
        delta_n=(-0.20, 0.10),
        weight_region_type=MDN_WX,
        wx_support_values=(0.60, 0.60),
    )
    assert library.add_skill(
        "contextual-certified-before-shift",
        contextual_cert,
        policy=lambda _obs: 1,
        weight_region_type=MDN_WX,
        certification_context=contextual_cert.certification_context,
        mdn_alpha=contextual_cert.mdn_alpha,
        wx_support_directions=contextual_cert.wx_support_directions,
        wx_support_values=contextual_cert.wx_support_values,
    )

    return library


def test_mid_episode_shift_reuses_certified_skill_and_rejects_contextual_skill():
    library = _library_with_reuse_and_rejection()

    trace = demonstrate_mid_episode_motive_shift(
        library,
        initial_weight=(0.50, 0.50),
        shifted_weight=(0.90, 0.10),
        shift_step=100,
        initial_support_directions=STANDARD_DIRECTIONS,
        initial_support_values=(0.60, 0.60),
        shifted_support_directions=STANDARD_DIRECTIONS,
        shifted_support_values=(1.00, 1.00),
    )

    assert trace.shift_step == 100
    assert trace.retraining_performed is False
    assert "global-certified-reuse" in trace.pre_shift_admissible
    assert "contextual-certified-before-shift" in trace.pre_shift_admissible
    assert trace.selected_after_shift == "global-certified-reuse"
    assert "global-certified-reuse" in trace.post_shift_admissible
    assert "contextual-certified-before-shift" not in trace.post_shift_admissible
    assert trace.rejected_after_shift == ("contextual-certified-before-shift",)


def test_mid_episode_shift_requires_positive_shift_step():
    with pytest.raises(ValueError, match="shift_step"):
        demonstrate_mid_episode_motive_shift(
            _library_with_reuse_and_rejection(),
            initial_weight=(0.50, 0.50),
            shifted_weight=(0.90, 0.10),
            shift_step=0,
            initial_support_directions=STANDARD_DIRECTIONS,
            initial_support_values=(0.60, 0.60),
            shifted_support_directions=STANDARD_DIRECTIONS,
            shifted_support_values=(1.00, 1.00),
        )


def test_mid_episode_shift_uses_library_weight_validation():
    with pytest.raises(ValueError, match="simplex"):
        demonstrate_mid_episode_motive_shift(
            _library_with_reuse_and_rejection(),
            initial_weight=(0.50, 0.50),
            shifted_weight=(0.40, 0.40),
            shift_step=50,
            initial_support_directions=STANDARD_DIRECTIONS,
            initial_support_values=(0.60, 0.60),
            shifted_support_directions=STANDARD_DIRECTIONS,
            shifted_support_values=(1.00, 1.00),
        )
