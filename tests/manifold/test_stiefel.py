import pytest
import numpy as np
from hypothesis import given, assume, strategies as st
import riemopt.manifold as rm
from ..conftest import draw_manifold, draw_arrays


def test_negative_dimension_error():
    with pytest.raises(ValueError):
        _ = rm.Stiefel(-1, 1)


def test_negative_amount_error():
    with pytest.raises(ValueError):
        _ = rm.Stiefel(4, -2)


def test_zero_dimension_error():
    with pytest.raises(ValueError):
        _ = rm.Stiefel(0, 1)


def test_zero_amount_error():
    with pytest.raises(ValueError):
        _ = rm.Stiefel(1, 0)


def test_more_vectors_than_dimensions():
    with pytest.raises(ValueError):
        _ = rm.Stiefel(2, 6)


@given(data=st.data())
def test_point_project_is_stiefel(data):
    s = draw_manifold(rm.Stiefel, data)
    _, k = s.shape

    (p, ) = draw_arrays(data, s, 1)
    assume(np.all(np.linalg.norm(p, axis=1) > 0))

    projected = s.point_project(p)
    np.testing.assert_array_almost_equal(projected.T @ projected, np.eye(k, k))


@given(data=st.data())
def test_tangent_project_on_tangent_space(data):
    s = draw_manifold(rm.Stiefel, data)
    _, k = s.shape

    p, vector = draw_arrays(data, s, 2)
    assume(np.all(np.linalg.norm(p, axis=1) > 0))
    point = s.point_project(p)

    tangent = s.tangent_project(point, vector)
    np.testing.assert_array_almost_equal(
        point.T @ tangent + tangent.T @ point,
        np.zeros((k, k)),
    )


@given(data=st.data())
def test_retract_on_stiefel(data):
    s = draw_manifold(rm.Stiefel, data)
    _, k = s.shape

    p, v = draw_arrays(data, s, 2)
    assume(np.all(np.linalg.norm(p, axis=1) > 0))

    point = s.point_project(p)
    tangent = s.tangent_project(point, v)

    next = s.retract(point, tangent)
    np.testing.assert_array_almost_equal(next.T @ next, np.eye(k, k))
