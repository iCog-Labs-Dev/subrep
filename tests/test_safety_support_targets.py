import numpy as np

from certification.pds_test import PDSGate
from generator.build_safety_support_targets import choose_region, region_grid, vertices


def test_teacher_rejects_no_task_improvement():
    caps, areas = region_grid(10)
    assert choose_region(np.array([[0., 0., 0.], [-2., -1., -3.]]), caps, areas, .2) is None


def test_teacher_certifies_entire_reconstructed_region():
    caps, areas = region_grid(10)
    dn = np.array([-2., 1., -3.])
    _, _, selected = choose_region(dn[None], caps, areas, .2)
    points = vertices(selected)
    target = points.max(axis=0)
    assert np.allclose(points.sum(axis=1), 1)
    assert np.allclose(target, selected)
    assert PDSGate(.2).admit(dn[1], dn, support_values=target)
    assert not PDSGate(.2).admit(dn[1], dn)


def test_full_simplex_area_and_single_witness():
    caps, areas = region_grid(10)
    rank, index, selected = choose_region(np.array([[-2., 1., -3.], [1., 1., 1.]]), caps, areas, .2)
    assert index == 1
    assert np.isclose(rank[0], 1)
    assert np.allclose(selected, 1)
