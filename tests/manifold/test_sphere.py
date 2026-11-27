import pytest
import numpy as np
from hypothesis import given, assume, strategies as st
import riemopt.manifold as rm
from ..conftest import draw_manifold, draw_arrays


def test_negative_dimension_error():
    with pytest.raises(ValueError):
        _ = rm.Sphere(-1)


def test_project_origin_invalid():
    s = rm.Sphere(2)

    with pytest.raises(ZeroDivisionError):
        _ = s.point_project(np.array([0.0, 0.0, 0.0]))


@given(data=st.data())
def test_point_project_has_unit_norm(data):
    s = draw_manifold(rm.Sphere, data)
    (p, ) = draw_arrays(data, s, 1)
    assume(np.linalg.norm(p) > 0)

    projected = s.point_project(p)
    assert np.isclose(1.0, np.linalg.norm(projected))


@given(data=st.data())
def test_tangent_project_on_tangent_space(data):
    s = draw_manifold(rm.Sphere, data)
    p, vector = draw_arrays(data, s, 2)

    assume(np.linalg.norm(p) > 0)
    point = s.point_project(p)

    tangent = s.tangent_project(point, vector)
    assert np.isclose(0.0, np.dot(point, tangent))


@given(data=st.data())
def test_retract_on_sphere(data):
    s = draw_manifold(rm.Sphere, data)
    p, v = draw_arrays(data, s, 2)
    assume(np.linalg.norm(p) > 0)

    point = s.point_project(p)
    tangent = s.tangent_project(point, v)

    next = s.retract(point, tangent)
    assert np.isclose(1.0, np.linalg.norm(next))
