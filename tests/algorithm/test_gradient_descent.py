import numpy as np
import riemopt.manifold as rm
from riemopt.algorithm.gradient_descent import GradientDescent
from .utils import rayleigh_quotient


def test_diagonal_matrix():
    a = np.diag([2.0, 3.0, -1.0])
    f = rayleigh_quotient(a)

    opt = GradientDescent(
        manifold=rm.Sphere(2),
        step_size=1e-3,
        max_steps=5000,
        gradient_threshold=1e-3,
    )

    r = opt.minimize(f, np.array([1.0, 1.0, 1.0]) / np.sqrt(3))

    assert r.converged is True

    assert r.value_opt is not None
    assert np.isclose(-3.0, r.value_opt)

    assert r.point_opt is not None
    np.testing.assert_almost_equal(np.array([0.0, 1.0, 0.0]), np.abs(r.point_opt), decimal=3)
