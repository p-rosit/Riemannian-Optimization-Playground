import pytest
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import riemopt.manifold as rm


def test_can_instantiate():
    _ = rm.Real(4)


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


def test_generate_random_point():
    r = rm.Real(2)
    assert (2, ) == r.point_random(rng=1).shape


def test_generate_random_tangent():
    r = rm.Real(3)
    assert (3, ) == r.tangent_random(np.array([1, 2, 3]), rng=2).shape


dims = st.integers(min_value=1, max_value=20)
floats = st.floats(-1e6, 1e6)


@st.composite
def manifold_and_vectors(draw, k: int):
    n = draw(dims)
    vecs = [draw(arrays(np.float64, n, elements=floats)) for _ in range(k)]
    return rm.Real(n), *vecs


@given(manifold_and_vectors(1))
def test_point_project_is_identity(args):
    r, p = args
    np.testing.assert_array_equal(r.point_project(p.copy()), p)


@given(manifold_and_vectors(2))
def test_tangent_project_is_identity(args):
    r, p, v = args
    np.testing.assert_array_equal(r.tangent_project(p.copy(), v.copy()), v)


@given(manifold_and_vectors(2))
def test_retract_is_addition(args):
    r, p, v = args
    np.testing.assert_array_equal(r.retract(p.copy(), v.copy()), p + v)


@given(dims, st.integers(min_value=0))
def test_random_point_and_tangent_have_right_shape(n, seed):
    r = rm.Real(n)
    p = r.point_random(seed)
    v = r.tangent_random(p, seed)
    assert (n, ) == p.shape
    assert (n, ) == v.shape
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(v))