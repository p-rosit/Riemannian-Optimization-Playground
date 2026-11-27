import pytest
import numpy as np
from hypothesis import given, strategies as st
import riemopt.manifold as rm
from ..conftest import draw_manifold, draw_arrays


def test_can_instantiate():
    _ = rm.Real(4)


def test_negative_dimension_error():
    with pytest.raises(ValueError):
        _ = rm.Real(-1)


@given(data=st.data())
def test_point_project_is_identity(data):
    r = draw_manifold(rm.Real, data)
    p = draw_arrays(data, r, 1)
    np.testing.assert_array_equal(r.point_project(p), p)


@given(data=st.data())
def test_tangent_project_is_identity(data):
    r = draw_manifold(rm.Real, data)
    p, v = draw_arrays(data, r, 2)
    np.testing.assert_array_equal(r.tangent_project(p, v), v)


@given(data=st.data())
def test_retract_is_addition(data):
    r = draw_manifold(rm.Real, data)
    p, v = draw_arrays(data, r, 2)
    np.testing.assert_array_equal(r.retract(p, v), p + v)
