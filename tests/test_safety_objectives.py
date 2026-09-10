import numpy as np
import pytest
from utils.safety_objectives import control_effort, FrozenObjectiveScales
from env.safety_gymnasium_wrapper import SafeRLGymnasiumEnv
from tests.test_safety_gymnasium_adapter import _make_fake_env


def test_three_objectives_are_measured_from_actual_step_inputs_and_outputs():
    env = SafeRLGymnasiumEnv(make_env=_make_fake_env, include_control_efficiency=True)
    _, values, _, _, info = env.step(np.array([0.8, 0.0], dtype=np.float32))
    assert env.reward_space.shape == (3,)
    np.testing.assert_allclose(values, [-0.25, 0.92, -0.32])
    np.testing.assert_allclose(info['raw_objective_values'], values)
    assert info['control_effort'] == pytest.approx(0.32)
    env.close()


def test_zero_and_asymmetric_bound_normalization():
    assert control_effort([0,0], [-2,-1], [1,3]) == 0
    assert control_effort([-2,3], [-2,-1], [1,3]) == 1
    assert control_effort([-1,1.5], [-2,-1], [1,3]) == 0.25


@pytest.mark.parametrize('action', [[2,0], [float('nan'),0], [0]])
def test_invalid_actions_rejected(action):
    with pytest.raises(ValueError):
        control_effort(action, [-1,-1], [1,1])


def test_frozen_development_scaling_roundtrip_and_double_task_contribution(tmp_path):
    scales = FrozenObjectiveScales.fit_development([[-2,4,-0.5],[2,-4,0.5]], source='development-seeds-only')
    assert scales.divisors == (2,4,0.5)
    path = tmp_path/'scales.json'
    scales.save(path)
    frozen = FrozenObjectiveScales.load(path)
    assert frozen == scales
    assert frozen.score(4, [-2,4,-0.5], [0.2,0.6,0.2]) == pytest.approx(1.2)
    assert frozen.score(4, [-2,4,-0.5], [0,1,0]) == 2
    assert frozen.divisors == (2,4,0.5)


def test_zero_development_dimension_and_invalid_scaling():
    assert FrozenObjectiveScales.fit_development([[0,0,0]],source='dev').divisors == (1,1,1)
    with pytest.raises(ValueError):
        FrozenObjectiveScales((1,0,1), 'dev')
    with pytest.raises(ValueError):
        FrozenObjectiveScales.fit_development([[1,2]],source='dev')
