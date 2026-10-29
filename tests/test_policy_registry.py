"""Policy ID invariants across collection, training and inference."""
import pytest
from generator.policy_registry import POLICY_NAMES, POLICY_TO_ID, policy_index


def test_registry_is_unique_and_contiguous():
    assert len(POLICY_NAMES) == 9
    assert len(set(POLICY_NAMES)) == len(POLICY_NAMES)
    assert [policy_index(name) for name in POLICY_NAMES] == list(range(len(POLICY_NAMES)))
    assert set(POLICY_TO_ID) == set(POLICY_NAMES)


def test_registry_rejects_unknown_policy():
    with pytest.raises(ValueError, match='Unknown candidate policy'):
        policy_index('invented_policy')
