"""Stable identities for the existing nine candidate policies (no behavior changes)."""
POLICY_NAMES = (
    'ppo_deterministic', 'ppo_stochastic', 'noop', 'left_engine',
    'main_engine', 'right_engine', 'random',
    'ppo_then_side_tradeoff', 'ppo_noisy_actions',
)
POLICY_TO_ID = {name: index for index, name in enumerate(POLICY_NAMES)}


def policy_index(name: str) -> int:
    if name not in POLICY_TO_ID:
        raise ValueError(f'Unknown candidate policy {name!r}; registry: {POLICY_NAMES}')
    return POLICY_TO_ID[name]
