import json

import pytest

from generator.ranking_config import load_motive_weights


def test_load_motive_weights_rejects_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ranking configuration not found"):
        load_motive_weights(tmp_path / "missing.json")


def test_load_motive_weights_accepts_matching_split(tmp_path):
    config_path = tmp_path / "weights.json"
    config_path.write_text(
        json.dumps({
            "motive_weights": {"safety": 0.7, "fuel": 0.3},
            "split_signature": "split-a",
        }),
        encoding="utf-8",
    )

    assert load_motive_weights(
        config_path,
        expected_split_signature="split-a",
    ) == (0.7, 0.3)


def test_load_motive_weights_rejects_different_split(tmp_path):
    config_path = tmp_path / "weights.json"
    config_path.write_text(
        json.dumps({
            "motive_weights": {"safety": 0.7, "fuel": 0.3},
            "split_signature": "split-a",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different split"):
        load_motive_weights(
            config_path,
            expected_split_signature="split-b",
        )
