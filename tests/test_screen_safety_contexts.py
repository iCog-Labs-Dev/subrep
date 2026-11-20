from demo.screen_safety_contexts import screen_then_verify


def test_verification_only_uses_selected_contexts_and_keeps_failure():
    calls = []
    def run(context, index, phase, count):
        calls.append((context, phase, count))
        cost = 0 if context == 1 and phase == 'screen' else 3
        return [dict(cost=cost, goals=1, effort=1, action_change=0)] * count
    result = screen_then_verify([1, 2], run, 5, 10, 1, .9)
    assert calls == [(1, 'screen', 5), (2, 'screen', 5), (1, 'verify', 10)]
    assert result[0]['screen']['empirical_pass']
    assert not result[0]['verification']['empirical_pass']
    assert result[1]['verification'] is None
