from __future__ import annotations

import numpy as np

from utils.return_targets import ips_weighted_return


def test_ips_weighted_return_supports_importance_weight_clipping():
    motives = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    behavior_probability = np.array([[0.1, 0.1]], dtype=np.float32)
    target_probability = np.array([[10.0, 10.0]], dtype=np.float32)

    unclipped = ips_weighted_return(
        motives,
        behavior_probability=behavior_probability,
        target_probability=target_probability,
        clip_value=None,
    )
    clipped = ips_weighted_return(
        motives,
        behavior_probability=behavior_probability,
        target_probability=target_probability,
        clip_value=10.0,
    )

    assert np.all(unclipped > clipped)
    assert clipped.shape == (1, 2)
