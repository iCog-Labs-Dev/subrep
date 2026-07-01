from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from certification.certificate_schema import Certificate
from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_runtime_selector import MDNRuntimeSelector
from library.skill_library import SkillLibrary
from library.skill_metadata import FULL_SIMPLEX, MDN_WX
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.mdn_selection import alpha_to_mean_weights
from utils.support_geometry import make_basis_query_directions


def _save_deterministic_mdn_checkpoint(
    path: Path,
    *,
    num_skills: int = 2048,
) -> None:
    model = MotiveDecompositionNetwork(
        input_dim=8,
        num_objectives=2,
        hidden_dim=32,
        num_hidden_layers=3,
        num_skills=num_skills,
        skill_embedding_dim=11,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    torch.save({"model_state_dict": model.state_dict()}, path)


def _make_certificate(
    *,
    skill_id: str,
    delta_r: float,
    delta_n: tuple[float, float],
    weight_region_type: str,
    admission_margin: float,
    certification_context: tuple[float, ...] | None = None,
    mdn_alpha: tuple[float, ...] | None = None,
    wx_support_directions: tuple[tuple[float, ...], ...] | None = None,
    wx_support_values: tuple[float, ...] | None = None,
) -> Certificate:
    return Certificate(
        skill_id=skill_id,
        gate_type="CDS",
        delta_r=delta_r,
        delta_n=delta_n,
        admission_margin=admission_margin,
        epsilon=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="idle_v1",
        environment="MO-LunarLander-v3",
        episode_length=200,
        version="0.1.0",
        weight_region_type=weight_region_type,
        certification_context=certification_context,
        mdn_alpha=mdn_alpha,
        wx_support_directions=wx_support_directions,
        wx_support_values=wx_support_values,
    )


def test_checkpoint_loader_infers_candidate_set_model_shape(tmp_path):
    checkpoint_path = tmp_path / "mdn_policy_best.pth"
    _save_deterministic_mdn_checkpoint(checkpoint_path, num_skills=4096)

    model = load_mdn_checkpoint(checkpoint_path)

    assert model.input_dim == 8
    assert model.num_objectives == 2
    assert model.hidden_dim == 32
    assert model.num_hidden_layers == 3
    assert model.num_skills == 4096
    assert model.skill_embedding_dim == 11

    with pytest.raises(ValueError, match="num_skills"):
        MDNRuntimeSelector.from_checkpoint(
            str(checkpoint_path),
            input_dim=8,
            num_objectives=2,
            num_skills=128,
        )


def test_loaded_mdn_checkpoint_supplies_runtime_geometry_to_library(tmp_path):
    checkpoint_path = tmp_path / "mdn_policy_best.pth"
    _save_deterministic_mdn_checkpoint(checkpoint_path, num_skills=2048)

    loaded_model = load_mdn_checkpoint(checkpoint_path)
    observation = np.zeros(loaded_model.input_dim, dtype=np.float32)

    with torch.no_grad():
        alpha_tensor, support_tensor = loaded_model.forward_inference(
            torch.tensor(observation, dtype=torch.float32)
        )

    alpha = alpha_tensor.detach().cpu().numpy()
    support_values = support_tensor.detach().cpu().numpy()
    weights = alpha_to_mean_weights(alpha)
    support_directions = make_basis_query_directions(len(support_values))

    assert np.all(alpha > 0.0)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(support_values >= 0.0)
    assert np.all(support_values <= 1.0)
    assert float(np.sum(support_values)) >= 1.0

    library = SkillLibrary()
    global_cert = _make_certificate(
        skill_id="global-safe",
        delta_r=0.05,
        delta_n=(0.0, 0.0),
        weight_region_type=FULL_SIMPLEX,
        admission_margin=0.05,
    )
    contextual_cert = _make_certificate(
        skill_id="trained-mdn-contextual",
        delta_r=0.15,
        delta_n=(-0.2, 0.1),
        weight_region_type=MDN_WX,
        admission_margin=0.025,
        certification_context=tuple(float(v) for v in observation),
        mdn_alpha=tuple(float(v) for v in alpha),
        wx_support_directions=tuple(
            tuple(float(v) for v in row) for row in support_directions
        ),
        wx_support_values=tuple(float(v) for v in support_values),
    )

    assert library.add_skill(
        "global-safe",
        global_cert,
        lambda obs: 0,
        weight_region_type=FULL_SIMPLEX,
    )
    assert library.add_skill(
        "trained-mdn-contextual",
        contextual_cert,
        lambda obs: 0,
        weight_region_type=MDN_WX,
        certification_context=contextual_cert.certification_context,
        mdn_alpha=contextual_cert.mdn_alpha,
        wx_support_directions=contextual_cert.wx_support_directions,
        wx_support_values=contextual_cert.wx_support_values,
    )

    selector = MDNRuntimeSelector.from_checkpoint(
        str(checkpoint_path),
        input_dim=8,
        num_objectives=2,
        num_skills=2048,
    )
    with patch.object(
        library,
        "query_admissible",
        wraps=library.query_admissible,
    ) as query_admissible:
        result = selector.select_from_library(observation, library)

    query_admissible.assert_called_once()
    assert result.selected_skill_id == "trained-mdn-contextual"
    assert np.allclose(result.alpha, alpha)
    assert np.allclose(result.support_values, support_values)
    assert np.allclose(result.weights_used, weights)
