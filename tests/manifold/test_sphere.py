import pytest
import numpy as np
from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays
import riemopt.manifold as rm


def test_can_instantiate():
    _ = rm.Sphere(2)


def test_negative_dimension_error():
    with pytest.raises(ValueError):
        _ = rm.Sphere(-1)


def test_project_origin_invalid():
    s = rm.Sphere(2)

    with pytest.raises(ZeroDivisionError):
        _ = s.point_project(np.array([0.0, 0.0, 0.0]))


dims = st.integers(min_value=1, max_value=20)
floats = st.floats(-1e6, 1e6)


@st.composite
def manifold_and_vectors(draw, k: int):
    n = draw(dims)
    vecs = [draw(arrays(np.float64, n + 1, elements=floats)) for _ in range(k)]
    return rm.Sphere(n), *vecs


@given(manifold_and_vectors(1))
def test_point_project_has_unit_norm(args):
    s, p = args
    assume(np.linalg.norm(p) > 0)

    projected = s.point_project(p)
    assert np.isclose(1.0, np.linalg.norm(projected))


@given(manifold_and_vectors(1))
def test_point_project_idempotent(args):
    s, p = args
    assume(np.linalg.norm(p) > 0)

    projected_1 = s.point_project(p)
    projected_2 = s.point_project(projected_1)
    np.testing.assert_array_almost_equal(projected_1, projected_2)


@given(manifold_and_vectors(2))
def test_tangent_project_on_tangent_space(args):
    s, p, vector = args
    
    assume(np.linalg.norm(p) > 0)
    point = s.point_project(p)

    tangent = s.tangent_project(point, vector)
    assert np.isclose(0.0, np.dot(point, tangent))


@given(manifold_and_vectors(2))
def test_retract_on_sphere(args):
    s, p, v = args
    assume(np.linalg.norm(p) > 0)

    point = s.point_project(p)
    tangent = s.tangent_project(point, v)

    next = s.retract(point, tangent)
    assert np.isclose(1.0, np.linalg.norm(next))


@given(dims, st.integers(min_value=0))
def test_random_point_and_tangent_have_right_shape(n, seed):
    r = rm.Sphere(n)
    p = r.point_random(seed)
    v = r.tangent_random(p, seed)
    assert (n + 1, ) == p.shape
    assert (n + 1, ) == v.shape
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(v))


@given(dims, st.integers(min_value=0))
def test_random_point_reproducible(n, seed):
    r = rm.Sphere(n)

    p1 = r.point_random(seed)
    p2 = r.point_random(seed)
    np.testing.assert_equal(p1, p2)


@given(manifold_and_vectors(1), st.integers(min_value=0))
def test_random_tangent_reproducible(args, seed):
    r, p = args

    t1 = r.tangent_random(p, seed)
    t2 = r.tangent_random(p, seed)
    np.testing.assert_equal(t1, t2)
