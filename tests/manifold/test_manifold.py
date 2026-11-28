import numpy as np
from hypothesis import given, assume, strategies as st
from riemopt.manifold import Manifold, NotProjectableError
from ..conftest import draw_manifold, draw_arrays, try_point_project


@given(data=st.data(), seed=st.integers(min_value=0))
def test_generate_random_point(manifold: type[Manifold], data, seed: int):
    m = draw_manifold(manifold, data)
    assert m.shape == m.point_random(rng=seed).shape


@given(data=st.data(), seed_point=st.integers(min_value=0), seed=st.integers(min_value=0))
def test_generate_random_tangent(manifold: type[Manifold], data, seed_point: int, seed: int):
    m = draw_manifold(manifold, data)
    p = m.point_random(rng=seed_point)
    assert m.shape == m.tangent_random(p, rng=seed).shape


@given(data=st.data(), seed=st.integers(min_value=0))
def test_point_random_reproducible(manifold: type[Manifold], data, seed: int):
    m = draw_manifold(manifold, data)

    p1 = m.point_random(seed)
    p2 = m.point_random(seed)
    np.testing.assert_equal(p1, p2)


@given(data=st.data(), seed_point=st.integers(min_value=0), seed=st.integers(min_value=0))
def test_tangent_random_reproducible(manifold: type[Manifold], data, seed_point: int, seed: int):
    m = draw_manifold(manifold, data)

    p = m.point_random(seed_point)
    t1 = m.tangent_random(p, seed)
    t2 = m.tangent_random(p, seed)
    np.testing.assert_equal(t1, t2)


@given(data=st.data())
def test_retract_input_constant(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p, v = draw_arrays(data, m, 2)
    p = try_point_project(m, p)
    v = m.tangent_project(p, v)
    p_copy = p.copy()
    v_copy = v.copy()

    _ = m.retract(p, v)
    np.testing.assert_equal(p_copy, p)
    np.testing.assert_equal(v_copy, v)


@given(data=st.data())
def test_retraction_identity(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)
    (p, ) = draw_arrays(data, m, 1)
    p = try_point_project(m, p)

    r = m.retract(p, np.zeros(shape=m.shape))
    np.testing.assert_almost_equal(p, r)


@given(data=st.data())
def test_retraction_gradient(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p, v = draw_arrays(data, m, 2)
    p = try_point_project(m, p)
    v = m.tangent_project(p, v)

    v = v / np.linalg.norm(v)

    q = m.retract(p, 1e-6 * v)
    w = (q - p) / np.linalg.norm(q - p)
    np.testing.assert_almost_equal(v, w)


@given(data=st.data())
def test_point_project_input_constant(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    (p, ) = draw_arrays(data, m, 1)
    p_copy = p.copy()

    try:
        _ = m.point_project(p)
    except NotProjectableError:
        assume(False)
    np.testing.assert_equal(p_copy, p)


@given(data=st.data())
def test_point_project_twice(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    (p, ) = draw_arrays(data, m, 1)

    try:
        r1 = m.point_project(p)
    except NotProjectableError:
        assume(False)
    r2 = m.point_project(r1)
    np.testing.assert_almost_equal(r1, r2)


@given(data=st.data())
def test_tangent_project_input_constant(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p, v = draw_arrays(data, m, 2)
    p = try_point_project(m, p)
    p_copy = p.copy()
    v_copy = v.copy()

    _ = m.tangent_project(p, v)
    np.testing.assert_equal(p_copy, p)
    np.testing.assert_equal(v_copy, v)


@given(data=st.data())
def test_tangent_project_twice(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p, v = draw_arrays(data, m, 2)
    p = try_point_project(m, p)

    r1 = m.tangent_project(p, v)
    r2 = m.tangent_project(p, r1)
    np.testing.assert_almost_equal(r1, r2)
