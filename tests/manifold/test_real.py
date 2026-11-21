import numpy as np
import riemopt.manifold as rm


def test_can_instantiate():
    r = rm.Real(4)


def test_projects_to_same():
    r = rm.Real(4)

    p = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(
        np.array([1, 2, 3, 4]),
        r.point_project(p)
    )
