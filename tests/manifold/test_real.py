import pytest
import numpy as np
import riemopt.manifold as rm


def test_can_instantiate():
    r = rm.Real(4)


def test_negative_value_error():
    with pytest.raises(ValueError):
        _ = rm.Real(-1)


def test_point_projects_to_same():
    r = rm.Real(4)

    p = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(
        np.array([1, 2, 3, 4]),
        r.point_project(p)
    )


def test_tangent_projects_to_same():
    r = rm.Real(3)

    p = np.array([3, 2, 1])
    np.testing.assert_array_equal(
        np.array([3, 2, 1]),
        r.tangent_project(np.array([0, 0, 1]), p)
    )