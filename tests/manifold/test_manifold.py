import numpy as np
from hypothesis import given, strategies as st
from riemopt.manifold import Manifold
from ..conftest import draw_manifold, draw_arrays


@given(data=st.data(), seed=st.integers(min_value=0))
def test_generate_random_point(manifold: type[Manifold], data, seed: int):
    m = draw_manifold(data, manifold)
    assert m.shape == m.point_random(rng=seed).shape


@given(data=st.data(), seed_point=st.integers(min_value=0), seed=st.intergers(min_value=0))
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
    p_copy = p.copy()
    v_copy = v.copy()

    _ = m.retract(p, v)
    np.testing.assert_equal(p_copy, p)
    np.testing.assert_equal(v_copy, v)


@given(data=st.data())
def test_point_project_input_constant(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p = draw_arrays(data, m, 1)
    p_copy = p.copy()

    # What about sphere which requires p != 0?
    _ = m.point_project(p)
    np.testing.assert_equal(p_copy, p)


@given(data=st.data())
def test_point_project_twice(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p = draw_arrays(data, m, 1)

    # What about sphere which requires p != 0?
    r1 = m.point_project(p)
    r2 = m.point_project(r1)
    np.testing.assert_equal(r1, r2)


@given(data=st.data())
def test_tangent_project_input_constant(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p, v = draw_arrays(data, m, 2)
    p = m.point_project(p)  # TODO: what about invalid input?
    p_copy = p.copy()
    v_copy = v.copy()

    _ = m.tangent_project(p, v)
    np.testing.assert_equal(p_copy, p)
    np.testing.assert_equal(v_copy, v)


@given(data=st.data())
def test_tangent_project_twice(manifold: type[Manifold], data):
    m = draw_manifold(manifold, data)

    p, v = draw_arrays(data, m, 2)
    p = m.point_project(p)  # TODO: what about invalid input?

    r1 = m.tangent_project(p, v)
    r2 = m.tangent_project(p, r1)
    np.testing.assert_equal(r1, r2)
